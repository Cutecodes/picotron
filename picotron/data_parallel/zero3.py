import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List

import picotron.process_group_manager as pgm
class Zero3Optimizer:
    def __init__(
        self, 
        model,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):

        self.model = model
        self.dp_rank = pgm.process_group_manager.dp_rank
        self.dp_world_size = pgm.process_group_manager.dp_world_size

        # 只为本rank拥有的参数创建优化器
        local_params = [
            p for p in model.params
            if p._zero3_owner_rank == self.dp_rank
        ]

        # 处理空参数列表的情况
        if len(local_params) > 0:
            self.optimizer = torch.optim.Adam(local_params, lr=lr)
        else:
            dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self.optimizer = torch.optim.Adam([dummy_param], lr=lr)
        

    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        """
        优化步骤:
        1. Reduce-Scatter: 梯度聚合并分片
        2. 本地更新: 每个GPU更新自己的参数分片
        3. 参数保持分片状态（不需要All-Gather）
        """

        # Step 1: Reduce梯度到owner
        for param in self.model.params:
            if param.grad is not None:
                owner_rank = param._zero3_owner_rank

                if self.dp_world_size > 1:
                    dist.reduce(
                        param.grad.data,
                        dst=owner_rank,
                        op=dist.ReduceOp.SUM,
                        group=pgm.process_group_manager.dp_group
                    )

                    # 非owner释放梯度
                    if self.dp_rank != owner_rank:
                        param.grad = None

        # Step 2: 本地更新
        self.optimizer.step()

        dist.barrier()