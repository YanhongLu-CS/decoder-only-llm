import torch
from torch import nn

from .attention import scaled_dot_product_causal_attention
from .rope import RoPE, RotaryEmbedding
from .config import QwenConfig

'''
Input:
hidden_states: [B, T, hidden_size]
position_ids:  [B, T]

Output:
attention_output: [B, T, hidden_size]
'''

class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.num_attention_heads * self.head_dim,
            bias = True
        )
        self.k_proj = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.num_key_value_heads * self.head_dim,
            bias = True
        )
        self.v_proj = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.num_key_value_heads * self.head_dim,
            bias = True
        )
        self.o_proj = nn.Linear(
            in_features = self.num_heads * self.head_dim,
            out_features = config.hidden_size,
            bias = False
        )
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor, # [B, T, hidden_size]
        position_ids: torch.Tensor, # [B, T]
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states) # [B, T, 2048]
        key = self.k_proj(hidden_states) # [B, T, 256]
        value = self.v_proj(hidden_states) # [B, T, 256]

        # 拆分成不同group
        # 目标形状是：
        # query: [B, num_heads, T, head_dim = 128]
        # key:   [B, num_key_value_heads, T, head_dim]
        # value: [B, num_key_value_heads, T, head_dim] 
        query = query.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        # RoPE
        cos, sin = self.rotary_emb(
            position_ids,
            dtype = query.dtype,
        )
        query, key = RoPE(
            query,
            key,
            cos,
            sin,
        )

        # GQA：扩展 K/V heads
        key = key.repeat_interleave(
            self.num_key_value_groups,
            dim = 1,
        )
        value = value.repeat_interleave(
            self.num_key_value_groups,
            dim = 1,
        )

        # Attention
        # 形状[batch_size, num_heads, sequence_length, head_dim]
        attention_output = scaled_dot_product_causal_attention(
            query,
            key,
            value
        )
        # 拼接
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim
        )
        #线性映射
        attention_output = self.o_proj(attention_output)

        return attention_output
