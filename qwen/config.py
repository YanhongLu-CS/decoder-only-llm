'''
dataclass的基本语法:
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    city: str = "Beijing"
'''
from dataclasses import dataclass

@dataclass
class QwenConfig:
    # Model dimensions
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36

    # Attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    attention_dropout: float = 0.0

    # Position encoding
    max_position_embeddings: int = 32768
    # RoPE旋转频率
    rope_theta: float = 1000000.0

    # Normalization and initialization
    rms_norm_eps: float = 1e-6
    # 模型参数随机初始化时正态分布的标准差
    initializer_range: float = 0.02

    # Model behaviour
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    use_cache: bool = True

    # Special tokens
    bos_token_id: int = 151643
    eos_token_id: int = 151645