import torch.distributed as dist
import torch
import picotron.process_group_manager as pgm
import torch.nn.functional as F


class ScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    Scatter in forward pass, all-gather in backward pass.
    This is the `g` function
    """
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        
        chunks = torch.chunk(x, pgm.process_group_manager.tp_world_size, dim=1)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output

        grad_output = grad_output.contiguous()
        tensor_list = [torch.empty_like(grad_output) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = grad_output
        dist.all_gather(tensor_list, grad_output, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=1).contiguous()
        return output

class GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x

        x = x.contiguous()
        tensor_list = [torch.empty_like(x) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = x
        dist.all_gather(tensor_list, x, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=1).contiguous()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size
        chunks = torch.chunk(grad_output, pgm.process_group_manager.tp_world_size, dim=1)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()
