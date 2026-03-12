import math
from typing import Optional
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import picotron.process_group_manager as pgm
from picotron.tensor_parallel.tp_communications import ReduceFromModelParallelRegion, GatherFromModelParallelRegion, linear_with_all_reduce, linear_with_async_all_reduce
from picotron.tensor_parallel.sp_communications import ScatterToSequenceParallelRegion, GatherFromSequenceParallelRegion

def apply_sequence_parallel(model):

    def _replace_module(_module, _layer_norm_name, args={}):
        layernorm_layer = getattr(_module, _layer_norm_name)
        
        new_layernorm_layer = SequenceParallelLlamaRMSNorm(
            hidden_size = layernorm_layer.weight.size(0)
        )
           
        setattr(_module, _layer_norm_name, new_layernorm_layer)

    module_layernorm_name_mapping_list = [
        ("attention", "input_layernorm"),
        ("attention", "post_attention_layernorm"),
    ]

    for layer in model.decoder_layers:
        for module_name, layernorm_name in module_layernorm_name_mapping_list:
            _replace_module(layer, layernorm_name)
            
    _replace_module(model, "final_norm")
    
    return model


class SequenceParallelLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.variance_epsilon = eps

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        hidden_states = ScatterToSequenceParallelRegion.apply(hidden_states)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
            # 一次性计算所有需要的统计量
        b, sp_seq_len, h = hidden_states.shape

        # 准备聚合数据 - 将所有统计量打包在一起
        stats = torch.zeros(b, h * 2 + 1, device=hidden_states.device)
        stats[:, :h] = hidden_states.sum(dim=-2)  # sum
        stats[:, h:2*h] = (hidden_states ** 2).sum(dim=-2)  # square sum
        stats[0, 2*h] = sp_seq_len * b  # count

        # 一次AllReduce聚合所有统计量
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)

        # 解包统计量
        global_sum = stats[:, :h]  # (b, h)
        global_square_sum = stats[:, h:2*h]  # (b, h)
        global_count = stats[0, 2*h]  # scalar

        # 计算全局统计量
        global_mean = global_sum / global_count  # (b, h)
        global_var = (global_square_sum / global_count) - (global_mean ** 2)

        # 重塑用于广播
        global_mean = global_mean.unsqueeze(1)  # (b, 1, h)
        global_var = global_var.unsqueeze(1)    # (b, 1, h)

        # 归一化
        hidden_states = (hidden_states - global_mean) / torch.sqrt(global_var + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        return GatherFromSequenceParallelRegion.apply(hidden_states.to(input_dtype))
    
    
    