import torch
from torch import nn
from .config import QwenConfig

#需要silu的函数
from torch.nn import functional as F

class MLP(nn.Module):
    """
    Map token IDs to hidden vectors.
    
    Input shape:
        (batch_size, sequence_length)
    
    Output shape:
        (batch_size, sequence_length, hidden_size)
    """
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.intermediate_size,
            bias = False,
        )
        self.up_proj = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.intermediate_size,
            bias = False,
        )
        self.down_proj = nn.Linear(
            in_features = config.intermediate_size,
            out_features = config.hidden_size,
            bias = False,
        )
        # for training
        nn.init.normal_(
            self.up_proj.weight,
            mean = 0.0,
            std = config.initializer_range,
        )
        nn.init.normal_(
            self.down_proj.weight,
            mean = 0.0,
            std = config.initializer_range,
        )
        nn.init.normal_(
            self.gate_proj.weight,
            mean = 0.0,
            std = config.initializer_range,
        )
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)

        hidden_states = gate * up
        return self.down_proj(hidden_states)
