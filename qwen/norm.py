import torch
from torch import nn
from .config import QwenConfig

class RMSNorm(nn.Module):
    '''
    一般继承nn.Module之后要定义
    __init__()
    forward()
    '''
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Input shape:
            (batch_size, sequence_length, hidden_size)
        
        Output shape:
            (batch_size, sequence_length, hidden_size)
        '''
        # 一般是BF16
        input_dtype = hidden_states.dtype
        # 转换成 FP32，减少低精度平方和求均值带来的数值误差
        hidden_states = hidden_states.to(torch.float32)

        square_mean = hidden_states.pow(2).mean(
            dim = -1,
            keepdim = True
        )
        # rsqrt(x)表示 1 / 根号下x
        hidden_states = hidden_states * torch.rsqrt(square_mean + self.eps)

        hidden_states = hidden_states.to(input_dtype)

        return self.weight * hidden_states
