import torch
import torch.distributed as dist
import contextlib
from torch import nn
from torch.autograd import Variable
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor, Shard, Replicate

from picotron.data_parallel.bucket import BucketManager
import picotron.process_group_manager as pgm
from contextlib import contextmanager

class DataParallelNaive(nn.Module):
    """
    Naive Data Parallelism. Not used in practice. But it is a good starting point to understand how data parallelism works.
    It implements a simple all-reduce operation to synchronize gradients across multiple processes.
    And `no_sync` context manager to disable gradient synchronization.
    """
    def __init__(self, module):
        """
        Initializes the DataParallel wrapper for a given module.

        Args:
            module (nn.Module): The model to be wrapped for data parallelism.
            process_group (torch.distributed.ProcessGroup): The process group used for gradient synchronization. 
                                                            It could be a data parallel or context parallel group.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        """
        Registers a backward hook for all parameters of the model that require gradients.    
        """
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_post_accumulate_grad_hook(hook)
                
    def _allreduce_grads(self, grad):
        """
        Performs an all-reduce operation to synchronize gradients across multiple processes.    
        """
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            grad /= pgm.process_group_manager.cp_dp_world_size
        return grad 
    
    @contextlib.contextmanager
    def no_sync(self):
        """
        A context manager to temporarily disable gradient synchronization. 
        This is useful for performing multiple backward passes during gradient accumulation without synchronizing 
        gradients in between.
        """
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

class DataParallelZero3(nn.Module):
    """
    基于DTensor的ZeRO-3风格参数分片实现
    核心改进：用DTensor自动管理参数分片
    """

    def __init__(self, module: nn.Module, shard_dim: int = 0):
        super().__init__()

        self.dp_rank = pgm.process_group_manager.dp_rank
        self.dp_world_size = pgm.process_group_manager.dp_world_size

        self.module = module
        self.params = list(module.parameters())
        # 参数分片, 按元素数分片，适合zero1和zero2

        param_numels = [p.numel() for p in self.params]
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
        

        # 记录参数归属
        self.param_to_rank = {}
        for dp_rank in range(self.dp_world_size):
            for param_indices in self.param_indices_per_rank[dp_rank]:
                self.param_to_rank[param_indices] = dp_rank

        self._shard_parameters()

    def _shard_parameters(self):
        """
        参数分片保存
        """
        for idx, param in enumerate(self.params):
            owner_rank = self.param_to_rank[idx]

            # 保存完整参数形状
            param._zero3_full_shape = param.data.shape
            param._zero3_owner_rank = owner_rank

            if self.dp_rank == owner_rank:
                # Owner保留完整参数
                param._zero3_full_param = param.data.clone()
            else:
                # 非owner释放参数显存
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
                param._zero3_full_param = None
    @contextmanager
    def _gather_parameters(self):
        """临时收集所有参数"""
        try:
            # All-Gather收集参数
            for param in self.params:
                owner_rank = param._zero3_owner_rank

                # 恢复完整参数空间
                if param.data.numel() == 0:
                    param.data = torch.empty(
                        param._zero3_full_shape,
                        dtype=param.dtype,
                        device=param.device
                    )

                # 广播参数
                if self.dp_world_size > 1:
                    dist.broadcast(param.data, src=owner_rank, group=pgm.process_group_manager.dp_group)

            yield

        finally:
            # 释放非本地参数
            for param in self.params:
                if self.dp_rank != param._zero3_owner_rank:
                    param.data = torch.empty(0, dtype=param.dtype, device=param.device)

    def forward(self, *args, **kwargs):
        """前向传播：自动聚合参数→计算→恢复分片"""
        with self._gather_parameters():
            return self.module(*args, **kwargs)
    

class DataParallelBucket(nn.Module):
    """
    Data Parallelism with gradient grouped into buckets to reduce the communication overhead.
    """
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        """
        Initialize the DataParallelBucket module.
        
        Args:
            module (nn.Module): The model to be parallelized.
            process_group: The process group for gradient synchronization, which can be either 
                           a data parallel group or a context parallel group.
            bucket_cap_mb (int, optional): The maximum size of each gradient synchronization bucket in megabytes. 
                                           Defaults to 25 MB.
            grad_type (torch.dtype, optional): The data type of gradients, defaulting to float32.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4 # float32 gradient: 4 bytes
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size # number of gradients in one bucket
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.cp_dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False # whether the callback for wait gradient synchronization is set
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.
        
        The gradient accumulation functions are stored to prevent them from going out of scope.
        
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies 
        the synchronized gradients back to the parameters' grad attribute.
        
        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    def reset(self):
        """
        Reset the bucket manager and zero out gradients in the model
        """
        self.bucket_manager.reset() 