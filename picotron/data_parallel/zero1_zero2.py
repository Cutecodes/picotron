import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List

import picotron.process_group_manager as pgm
class ZeroOptimizer:
    def __init__(
        self, 
        params: List[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        partition_grads=True
    ):
        self.partition_grads = partition_grads
        self.dp_rank = pgm.process_group_manager.dp_rank
        self.dp_world_size = pgm.process_group_manager.dp_world_size

        self.all_params = list(params)
        
        # 参数分片, 按元素数分片，适合zero1和zero2

        param_numels = [p.numel() for p in self.all_params]
        total_numel = sum(param_numels)
        # 计算每个rank应该分配的理想元素数量
        target_numel_per_rank = total_numel // self.dp_world_size

        # 基于元素数量进行分片
        self.param_indices_per_rank = [[] for _ in range(self.dp_world_size)]
        current_rank = 0
        current_numel = 0
        
        for idx, numel in enumerate(param_numels):
            # 如果当前rank已分配的元素数量超过目标，且不是最后一个rank，则移动到下一个rank
            if current_numel + numel > target_numel_per_rank and current_rank < self.dp_world_size - 1:
                current_rank += 1
                current_numel = 0
            
            self.param_indices_per_rank[current_rank].append(idx)
            current_numel += numel
        
        # 当前rank负责的参数索引
        self.my_param_indices = self.param_indices_per_rank[self.dp_rank]
        start_idx = self.my_param_indices[0]
        end_idx = self.my_param_indices[-1] + 1
        
        self.local_params = self.all_params[start_idx:end_idx]
        # 只为本地分片创建优化器（节省优化器状态显存）
        # 注意：如果local_params为空，创建一个dummy优化器
        if len(self.local_params) > 0:
            self.optimizer = torch.optim.Adam(
                self.local_params,
                lr=lr,
                betas=betas,
                eps=eps
            )
        else:
            # 为空参数列表创建dummy优化器
            dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self.optimizer = torch.optim.Adam([dummy_param], lr=lr)
            self.local_params = []  # 保持为空列表

        # 记录参数归属
        self.param_to_rank = {}
        for dp_rank in range(self.dp_world_size):
            for param_indices in self.param_indices_per_rank[dp_rank]:
                self.param_to_rank[param_indices] = dp_rank
        
        if self.partition_grads:
            self.register_backward_hook()
    
    def register_backward_hook(self):
        self.grad_accs = []
        for idx, param in enumerate(self.all_params):
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(idx, param))
                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(self, idx, param: torch.nn.Parameter):
        
        def param_hook(*unused):
            if param.requires_grad:
                assert param.grad is not None
                owner_rank = self.param_to_rank[idx]
                
                if self.dp_world_size > 1:
                    dist.reduce(
                        param.grad,
                        dst=owner_rank,
                        op=dist.ReduceOp.SUM,
                        group=pgm.process_group_manager.dp_group
                    )
                    
                    # 非owner释放梯度（节省显存）
                    if self.dp_rank != owner_rank:
                        param.grad = None
                        
                    else:
                        #print(owner_rank)
                        param.grad.div_(self.dp_world_size)
                                    
        return param_hook

    def zero_grad(self):
        for param in self.all_params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """
        优化步骤:
        1. Reduce-Scatter: 聚合梯度到对应的owner rank
        2. 本地更新: 每个GPU更新自己负责的参数
        3. All-Gather: 广播更新后的参数
        """
        
        
        # Step 1: Reduce梯度到owner rank (模拟reduce-scatter)
        if not self.partition_grads:
            for idx, param in enumerate(self.all_params):
                if param.grad is not None:
                    owner_rank = self.param_to_rank[idx]

                    if self.dp_world_size > 1:
                        dist.reduce(
                            param.grad.data,
                            dst=owner_rank,
                            op=dist.ReduceOp.SUM,
                            group=pgm.process_group_manager.dp_group
                        )
                        param.grad.data /= self.dp_world_size

        # Step 2: 本地更新（只更新本rank的参数）
        self.optimizer.step()

        # Step 3: All-Gather参数（所有rank都参与广播）
        if self.dp_world_size > 1:
            for idx, param in enumerate(self.all_params):
                owner_rank = self.param_to_rank[idx]
                dist.broadcast(param.data, src=owner_rank, group=pgm.process_group_manager.dp_group)

        dist.barrier()