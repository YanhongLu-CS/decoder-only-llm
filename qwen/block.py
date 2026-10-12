"""
Input:
hidden_states: [B, T, hidden_size]
position_ids:  [B, T]

Output:
[B, T, hidden_size]
"""
import torch
from torch import nn
from .config import QwenConfig
from .mlp import MLP
from .norm import RMSNorm
from .qwen_attention import QwenAttention

class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.self_attn = QwenAttention(config=config)
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(config=config)
        self.post_attention_layernorm = RMSNorm(config=config)
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        after_prenorm = self.input_layernorm(hidden_states)
        after_self_attn = self.self_attn(after_prenorm, position_ids) + hidden_states
        after_postnorm = self.post_attention_layernorm(after_self_attn)    
        after_mlp = self.mlp(after_postnorm) + after_self_attn

        return after_mlp