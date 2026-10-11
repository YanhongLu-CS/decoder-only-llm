import torch
import math
'''
Input:
query: [batch_size, num_heads, sequence_length, head_dim]
key:   [batch_size, num_heads, sequence_length, head_dim]
value: [batch_size, num_heads, sequence_length, head_dim]

e.g.
Q &: [B,16,L,128]
K &: [B,2,L,128]
V &: [B,2,L,128]

Output:
[batch_size, num_heads, sequence_length, head_dim]
'''

def scaled_dot_product_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    '''
    scaled: 除以根号下d_k
    dot_product: Q和K的转制点积
    causal: 因果掩码
    attention: 对value加权求和
    '''

    attention_scores = (
        query @ key.transpose(-2, -1)
    ) / math.sqrt(query.size(-1))

    # 加Mask
    casual_mask = torch.ones(
        query.size(-2),
        query.size(-2),
        # 要声明dtype和device，否则会mismatch
        dtype = torch.bool,
        device = query.device,
    )
    casual_mask = torch.triu(
        casual_mask,
        #diagonal 表示从哪条对角线开始保留：
        #diagonal=0：包含主对角线的右上区域
        #diagonal=1：从主对角线上方第一条线开始
        #diagonal=-1：从主对角线下方第一条线开始
        diagonal = 1,
    )
    attention_scores = attention_scores.masked_fill(
        casual_mask,
        float("-inf"),
    )

    attention_weights = torch.softmax(
        attention_scores,
        dim = -1,
        # 先吧attention_scores转成float32
        dtype = torch.float32,
    ).to(query.dtype)

    return attention_weights @ value
    
        