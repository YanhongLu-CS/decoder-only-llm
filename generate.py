import torch

from qwen.model import QwenCausalLM


@torch.inference_mode()
def greedy_generate(
    model: QwenCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    使用 greedy decoding 生成 token。

    Args:
        model:
            Qwen causal language model.
        input_ids:
            输入 token IDs，形状为 [batch_size, sequence_length]。
        max_new_tokens:
            最多生成多少个新 token。
        eos_token_id:
            可选的结束 token ID。

    Returns:
        生成后的完整 token IDs，形状为
        [batch_size, sequence_length + generated_length]。
    """
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]"
        )

    if input_ids.size(1) == 0:
        raise ValueError("input_ids must contain at least one token")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated_ids = input_ids
    batch_size = generated_ids.size(0)

    # 标记 batch 中哪些序列已经生成 EOS。
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=generated_ids.device,
    )

    for _ in range(max_new_tokens):
        sequence_length = generated_ids.size(1)

        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=generated_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        logits = model(
            generated_ids,
            position_ids,
        )

        # 最后一个位置的 logits 用于预测下一个 token。
        next_token_logits = logits[:, -1, :]

        # Greedy decoding：选择 logits 最大的 token。
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        if eos_token_id is not None:
            # 已结束的序列后续始终补 EOS，使 batch 保持相同长度。
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )

        generated_ids = torch.cat(
            (generated_ids, next_token),
            dim=1,
        )

        if eos_token_id is not None:
            finished = finished | (
                next_token.squeeze(-1) == eos_token_id
            )

            if finished.all():
                break

    return generated_ids