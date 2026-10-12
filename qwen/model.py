import torch
from torch import nn
from .block import QwenDecoderLayer
from .norm import RMSNorm
from .embedding import TokenEmbedding
from .config import QwenConfig

class QwenModel(nn.Module):
    '''
    Input:
    input_token_ids: [batch_size, sequence_length]
    position_ids: [batch_size, sequence_length]

    Output:
    [batch_size, sequence_length, hidden_size]
    '''
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.embed_tokens = TokenEmbedding(config=config)
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(QwenDecoderLayer(config = config))
        self.norm = RMSNorm(config=config)

    def forward(
        self,
        input_token_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_token_ids)

        # 逐层调用layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class QwenCausalLM(nn.Module):
    '''
    Input:
    input_token_ids: [batch_size, sequence_length]
    position_ids: [batch_size, sequence_length]
    
    Output:
    [batch_size, sequence_length, vocab_size]
    '''
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.model = QwenModel(config = config)
        self.lm_head = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.vocab_size,
            bias = False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_token_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        after_model = self.model(input_token_ids, position_ids)
        after_lm_head = self.lm_head(after_model)
        return after_lm_head