from .config import QwenConfig
import torch
from torch import nn

"""
Input:
query:        [B, Hq, T, D]
key:          [B, Hkv, T, D]
position_ids: [B, T]

Output:
rotated_query: [B, Hq, T, D]
rotated_key:   [B, Hkv, T, D]
"""

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # 把Tensor沿着最后一个维度切成两半
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

# 写成class防止inverse_frequency重复计算
class RotaryEmbedding(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()

        head_dim = config.hidden_size // config.num_attention_heads

        # 旋转频率[D // 2]
        exponent = torch.arange(0, head_dim, 2) / head_dim
        inverse_frequency = 1.0 / (config.rope_theta ** exponent)
        self.register_buffer("inverse_frequency", inverse_frequency, persistent = False)

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids.to(
            dtype = torch.float32,
            device = self.inverse_frequency.device
        )
        # [B, T, 1] * [1, 1, D // 2]
        frequencies = positions.unsqueeze(-1) * self.inverse_frequency.view(1, 1, -1)
        angles = torch.cat((frequencies, frequencies), dim = -1)

        cos = angles.cos().to(dtype = dtype)
        sin = angles.sin().to(dtype = dtype)

        # 变成[B, 1, T, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return cos, sin

def RoPE(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotated_query = query * cos + rotate_half(query) * sin
    rotated_key = key * cos + rotate_half(key) * sin
    return rotated_query, rotated_key