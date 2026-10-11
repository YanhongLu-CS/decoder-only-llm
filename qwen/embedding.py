from torch import nn

from .config import QwenConfig


class TokenEmbedding(nn.Embedding):
    """
    Map token IDs to hidden vectors.

    Input shape:
        (batch_size, sequence_length)

    Output shape:
        (batch_size, sequence_length, hidden_size)
    """

    def __init__(self, config: QwenConfig) -> None:
        super().__init__(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # nn.init.normal_()原地操作weight
        nn.init.normal_(
            self.weight,
            mean = 0.0,
            std = config.initializer_range,
        )

'''
Embedding的组成
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

        self.reset_parameters()

    def forward(self, input):
        return F.embedding(input, self.weight)
'''