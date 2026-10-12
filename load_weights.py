"""Load an official Qwen2.5 safetensors checkpoint into our model.

This module deliberately accepts a local checkpoint directory. Downloading the
multi-GB checkpoint is kept as a separate, explicit step.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from qwen.model import QwenCausalLM


TIED_WEIGHT_KEYS = {
    "model.embed_tokens.weight",
    "lm_head.weight",
}


def _checkpoint_weight_map(checkpoint_dir: Path) -> dict[str, Path]:
    """Return a mapping from every checkpoint key to its safetensors shard."""
    index_path = checkpoint_dir / "model.safetensors.index.json"

    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid checkpoint index: {index_path}")

        result = {
            key: checkpoint_dir / shard_name
            for key, shard_name in weight_map.items()
        }
    else:
        shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No safetensors checkpoint found in {checkpoint_dir}"
            )

        result: dict[str, Path] = {}
        for shard_path in shard_paths:
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key in result:
                        raise ValueError(
                            f"Duplicate checkpoint key {key!r} in "
                            f"{result[key].name} and {shard_path.name}"
                        )
                    result[key] = shard_path

    missing_shards = sorted({path for path in result.values() if not path.is_file()})
    if missing_shards:
        names = ", ".join(path.name for path in missing_shards)
        raise FileNotFoundError(f"Checkpoint shards are missing: {names}")

    return result


def _validate_checkpoint(
    model: QwenCausalLM,
    weight_map: dict[str, Path],
) -> None:
    """Validate parameter names and shapes before changing the model."""
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(weight_map)

    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys

    weights_are_tied = (
        model.lm_head.weight is model.model.embed_tokens.weight
    )
    if weights_are_tied and checkpoint_keys & TIED_WEIGHT_KEYS:
        missing_keys -= TIED_WEIGHT_KEYS

    if missing_keys or unexpected_keys:
        messages = []
        if missing_keys:
            messages.append(f"missing keys: {sorted(missing_keys)}")
        if unexpected_keys:
            messages.append(f"unexpected keys: {sorted(unexpected_keys)}")
        raise RuntimeError("Checkpoint key mismatch; " + "; ".join(messages))

    keys_by_shard: dict[Path, list[str]] = {}
    for key, shard_path in weight_map.items():
        keys_by_shard.setdefault(shard_path, []).append(key)

    shape_mismatches = []
    for shard_path, keys in keys_by_shard.items():
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in keys:
                checkpoint_shape = tuple(shard.get_slice(key).get_shape())
                model_shape = tuple(model_state[key].shape)
                if checkpoint_shape != model_shape:
                    shape_mismatches.append(
                        f"{key}: checkpoint {checkpoint_shape}, model {model_shape}"
                    )

    if shape_mismatches:
        details = "\n".join(shape_mismatches)
        raise RuntimeError(f"Checkpoint shape mismatch:\n{details}")


def load_qwen_weights(
    model: QwenCausalLM,
    checkpoint_dir: str | Path,
) -> QwenCausalLM:
    """Load local safetensors weights into ``model`` and return that model.

    The model controls the destination device and dtype. Each checkpoint tensor
    is loaded on CPU, converted to the matching model tensor, copied, and then
    released before the next tensor is loaded.
    """
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError(
            "This streaming loader requires a materialized model, not a meta model."
        )

    weight_map = _checkpoint_weight_map(checkpoint_dir)
    _validate_checkpoint(model, weight_map)

    model_state = model.state_dict()
    keys_by_shard: dict[Path, list[str]] = {}
    for key, shard_path in weight_map.items():
        keys_by_shard.setdefault(shard_path, []).append(key)

    with torch.no_grad():
        for shard_path, keys in keys_by_shard.items():
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                for key in keys:
                    checkpoint_tensor = shard.get_tensor(key)
                    target_tensor = model_state[key]
                    target_tensor.copy_(
                        checkpoint_tensor.to(
                            device=target_tensor.device,
                            dtype=target_tensor.dtype,
                        )
                    )

    # Loading libraries can replace Parameter objects. Reassert the official
    # Qwen2.5-3B tied-weight relationship after loading.
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        model.lm_head.weight = model.model.embed_tokens.weight

    return model
