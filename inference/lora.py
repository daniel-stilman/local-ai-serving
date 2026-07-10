"""Minimal LoRA weight application for the direct Anima and SDXL engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from runtime import InferenceError, SafeTensorFile


@dataclass(frozen=True)
class LoraTarget:
    parameter: torch.nn.Parameter
    selection: tuple[slice, ...] | None = None


def kohya_targets(module: torch.nn.Module, namespace: str) -> dict[str, LoraTarget]:
    targets: dict[str, LoraTarget] = {}
    for name, parameter in module.named_parameters():
        if not name.endswith(".weight"):
            continue
        stem = name[:-7].replace(".", "_")
        targets[namespace + stem] = LoraTarget(parameter)
    return targets


def peft_targets(module: torch.nn.Module, namespace: str) -> dict[str, LoraTarget]:
    targets: dict[str, LoraTarget] = {}
    for name, parameter in module.named_parameters():
        if name.endswith(".weight"):
            targets[namespace + name[:-7]] = LoraTarget(parameter)
    return targets


def openclip_kohya_targets(module: torch.nn.Module, namespace: str = "lora_te2_") -> dict[str, LoraTarget]:
    """Map split Diffusers CLIP projections onto OpenCLIP's combined QKV weights."""

    parameters = dict(module.named_parameters())
    targets: dict[str, LoraTarget] = {}
    layer = 0
    while f"transformer.resblocks.{layer}.attn.in_proj_weight" in parameters:
        prefix = f"{namespace}text_model_encoder_layers_{layer}_"
        targets[prefix + "mlp_fc1"] = LoraTarget(parameters[f"transformer.resblocks.{layer}.mlp.c_fc.weight"])
        targets[prefix + "mlp_fc2"] = LoraTarget(parameters[f"transformer.resblocks.{layer}.mlp.c_proj.weight"])
        targets[prefix + "self_attn_out_proj"] = LoraTarget(
            parameters[f"transformer.resblocks.{layer}.attn.out_proj.weight"]
        )
        qkv = parameters[f"transformer.resblocks.{layer}.attn.in_proj_weight"]
        dimension = qkv.shape[1]
        for index, projection in enumerate(("q", "k", "v")):
            targets[prefix + f"self_attn_{projection}_proj"] = LoraTarget(
                qkv,
                (slice(index * dimension, (index + 1) * dimension), slice(None)),
            )
        layer += 1
    return targets


def count_lora_matches(checkpoint: SafeTensorFile, targets: dict[str, LoraTarget]) -> int:
    return sum(
        1
        for base in targets
        if (
            base + ".lora_down.weight" in checkpoint
            and base + ".lora_up.weight" in checkpoint
        ) or (
            base + ".lora_A.weight" in checkpoint
            and base + ".lora_B.weight" in checkpoint
        )
    )


def validate_lora_targets(
    checkpoint: SafeTensorFile,
    targets: dict[str, LoraTarget],
) -> tuple[int, list[str]]:
    matched = 0
    errors: list[str] = []
    allowed_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    for base, target in targets.items():
        pairs = (
            (base + ".lora_down.weight", base + ".lora_up.weight"),
            (base + ".lora_A.weight", base + ".lora_B.weight"),
        )
        for down_name, up_name in pairs:
            if down_name not in checkpoint or up_name not in checkpoint:
                continue
            matched += 1
            if checkpoint.dtype(down_name) not in allowed_dtypes or checkpoint.dtype(up_name) not in allowed_dtypes:
                errors.append(f"{base}: unsupported dtype")
                continue
            down_shape = checkpoint.shape(down_name)
            up_shape = checkpoint.shape(up_name)
            delta_shape = _composed_shape(down_shape, up_shape)
            expected_shape = tuple(
                target.parameter[target.selection].shape if target.selection is not None else target.parameter.shape
            )
            if delta_shape != expected_shape:
                errors.append(f"{base}: {delta_shape} != {expected_shape}")
            if base + ".dora_scale" in checkpoint:
                errors.append(f"{base}: DoRA is not supported")
            break
    return matched, errors


def _composed_shape(down: tuple[int, ...], up: tuple[int, ...]) -> tuple[int, ...] | None:
    if len(down) != len(up) or len(down) not in (2, 4) or up[1] != down[0]:
        return None
    if len(down) == 2:
        return up[0], down[1]
    if up[2:] == (1, 1):
        return up[0], down[1], down[2], down[3]
    if down[2:] == (1, 1):
        return up[0], down[1], up[2], up[3]
    return None


def apply_loras(
    targets: dict[str, LoraTarget],
    loras: Iterable[tuple[Path, float]],
) -> int:
    applied = 0
    for path, strength in loras:
        with SafeTensorFile(path) as checkpoint:
            applied += _apply_checkpoint(targets, checkpoint, float(strength))
    return applied


def _apply_checkpoint(
    targets: dict[str, LoraTarget],
    checkpoint: SafeTensorFile,
    strength: float,
) -> int:
    applied = 0
    for base, target in targets.items():
        down_name = base + ".lora_down.weight"
        up_name = base + ".lora_up.weight"
        if down_name in checkpoint and up_name in checkpoint:
            if base + ".dora_scale" in checkpoint:
                raise InferenceError(f"{checkpoint.path.name} uses DoRA weights, which are not supported yet.")
            down = checkpoint.tensor(down_name)
            up = checkpoint.tensor(up_name)
            rank = down.shape[0]
            alpha_name = base + ".alpha"
            if alpha_name in checkpoint:
                alpha_tensor = checkpoint.tensor(alpha_name)
                alpha = float(alpha_tensor.float().item())
                del alpha_tensor
            else:
                alpha = float(rank)
            _apply_pair(target, down, up, strength * alpha / max(rank, 1), checkpoint.path.name)
            del down, up
            applied += 1
            continue

        down_name = base + ".lora_A.weight"
        up_name = base + ".lora_B.weight"
        if down_name in checkpoint and up_name in checkpoint:
            down = checkpoint.tensor(down_name)
            up = checkpoint.tensor(up_name)
            _apply_pair(target, down, up, strength, checkpoint.path.name)
            del down, up
            applied += 1
    return applied


def _apply_pair(
    target: LoraTarget,
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float,
    filename: str,
) -> None:
    parameter = target.parameter
    view = parameter[target.selection] if target.selection is not None else parameter
    device = parameter.device
    dtype = parameter.dtype
    down_device = down.to(device=device, dtype=dtype)
    up_device = up.to(device=device, dtype=dtype)

    if down_device.ndim == 2 and up_device.ndim == 2:
        delta = up_device @ down_device
    elif down_device.ndim == 4 and up_device.ndim == 4 and up_device.shape[2:] == (1, 1):
        delta = torch.einsum("or,rihw->oihw", up_device[:, :, 0, 0], down_device)
    elif down_device.ndim == 4 and up_device.ndim == 4 and down_device.shape[2:] == (1, 1):
        delta = torch.einsum("orhw,ri->oihw", up_device, down_device[:, :, 0, 0])
    else:
        raise InferenceError(f"{filename} contains an unsupported LoRA layer shape.")

    if tuple(delta.shape) != tuple(view.shape):
        raise InferenceError(
            f"{filename} has a LoRA layer with shape {tuple(delta.shape)}; expected {tuple(view.shape)}."
        )
    with torch.no_grad():
        view.add_(delta, alpha=float(scale))
    del down_device, up_device, delta
