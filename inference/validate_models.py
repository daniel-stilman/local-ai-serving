"""Audit local Anima and SDXL checkpoints against the direct engine.

This reads safetensors headers only. It validates every parameter name and
shape used by the bespoke models without allocating checkpoint weights or
writing any files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import torch

from anima import AnimaDenoiser, detect_anima_prefix
from lora import (
    kohya_targets,
    openclip_kohya_targets,
    peft_targets,
    validate_lora_targets,
)
from runtime import InferenceError, SafeTensorFile
from sdxl import ClipL, OpenClipBigG, SDXLImageDecoder, SDXLUNet


def _expected(module: torch.nn.Module, prefix: str) -> list[tuple[str, tuple[int, ...]]]:
    return [(prefix + name, tuple(parameter.shape)) for name, parameter in module.named_parameters()]


def _check(checkpoint: SafeTensorFile, expected: list[tuple[str, tuple[int, ...]]]) -> tuple[list[str], list[str]]:
    missing = [name for name, _shape in expected if name not in checkpoint]
    wrong = [
        f"{name}: {checkpoint.shape(name)} != {shape}"
        for name, shape in expected
        if name in checkpoint and checkpoint.shape(name) != shape
    ]
    return missing, wrong


def _validate_anima(path: Path, expected_module: torch.nn.Module) -> tuple[bool, str] | None:
    with SafeTensorFile(path) as checkpoint:
        try:
            prefix = detect_anima_prefix(checkpoint)
        except InferenceError:
            return None
        expected = _expected(expected_module, prefix)
        missing, wrong = _check(checkpoint, expected)
    if missing or wrong:
        return False, f"missing={len(missing)}, wrong_shape={len(wrong)}"
    return True, f"layout={prefix}, parameters={len(expected)}"


def _is_sdxl(checkpoint: SafeTensorFile) -> bool:
    signature = {
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": (49408, 768),
        "conditioner.embedders.1.model.token_embedding.weight": (49408, 1280),
        "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "model.diffusion_model.out.2.weight": (4, 320, 3, 3),
        "first_stage_model.decoder.conv_out.weight": (3, 128, 3, 3),
    }
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    return all(
        name in checkpoint and checkpoint.shape(name) == shape and checkpoint.dtype(name) in dtypes
        for name, shape in signature.items()
    )


def _validate_sdxl(
    path: Path,
    expected_groups: list[tuple[torch.nn.Module, str]],
) -> tuple[bool, str] | None:
    with SafeTensorFile(path) as checkpoint:
        if not _is_sdxl(checkpoint):
            return None
        missing: list[str] = []
        wrong: list[str] = []
        parameter_count = 0
        for module, prefix in expected_groups:
            expected = _expected(module, prefix)
            parameter_count += len(expected)
            group_missing, group_wrong = _check(checkpoint, expected)
            missing.extend(group_missing)
            wrong.extend(group_wrong)
    if missing or wrong:
        return False, f"missing={len(missing)}, wrong_shape={len(wrong)}"
    return True, f"parameters={parameter_count}"


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("*.safetensors"), key=lambda item: str(item).casefold())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local checkpoints against the direct image engine.")
    parser.add_argument("--models-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.models_root.resolve()
    show_identifiers = os.environ.get("MODEL_AUDIT_SHOW_IDENTIFIERS", "").strip().lower() in {
        "1", "true", "on", "yes"
    }

    anima_module = AnimaDenoiser(device="meta", dtype=torch.bfloat16)
    sdxl_groups = [
        (ClipL(device="meta", dtype=torch.float16), "conditioner.embedders.0.transformer."),
        (OpenClipBigG(device="meta", dtype=torch.float16), "conditioner.embedders.1.model."),
        (SDXLUNet(device="meta", dtype=torch.float16), "model.diffusion_model."),
        (SDXLImageDecoder(device="meta", dtype=torch.float32), "first_stage_model."),
    ]

    checked = 0
    failures = 0
    for family, directory, validator in (
        ("Anima", root / "diffusion_models", lambda path: _validate_anima(path, anima_module)),
        ("SDXL", root / "checkpoints", lambda path: _validate_sdxl(path, sdxl_groups)),
    ):
        for path in _files(directory):
            try:
                result = validator(path)
            except (InferenceError, OSError, ValueError) as error:
                result = (False, str(error) if show_identifiers else "read/validation error")
            if result is None:
                continue
            checked += 1
            compatible, detail = result
            failures += not compatible
            status = "PASS" if compatible else "FAIL"
            label = str(path.relative_to(directory)) if show_identifiers else f"checkpoint #{checked}"
            print(f"{status} {family}: {label} ({detail})")

    anima_lora_targets = {
        **kohya_targets(anima_module, "lora_unet_"),
        **peft_targets(anima_module, "diffusion_model."),
    }
    sdxl_lora_targets = [
        kohya_targets(sdxl_groups[0][0], "lora_te1_"),
        openclip_kohya_targets(sdxl_groups[1][0]),
        kohya_targets(sdxl_groups[2][0], "lora_unet_"),
    ]
    checked_loras = 0
    lora_directory = root / "loras"
    for path in _files(lora_directory):
        try:
            with SafeTensorFile(path) as checkpoint:
                if any(key.endswith(".dora_scale") for key in checkpoint.keys()):
                    continue
                anima_matches, anima_errors = validate_lora_targets(checkpoint, anima_lora_targets)
                sdxl_results = [validate_lora_targets(checkpoint, targets) for targets in sdxl_lora_targets]
                sdxl_matches = sum(result[0] for result in sdxl_results)
                sdxl_errors = [error for result in sdxl_results for error in result[1]]
        except (InferenceError, OSError, ValueError) as error:
            label = str(path.relative_to(lora_directory)) if show_identifiers else "candidate"
            detail = str(error) if show_identifiers else "read/validation error"
            print(f"FAIL LoRA: {label} ({detail})")
            failures += 1
            continue
        family = "Anima" if anima_matches else "SDXL" if sdxl_matches else ""
        matches = anima_matches or sdxl_matches
        errors = anima_errors if anima_matches else sdxl_errors
        if not family:
            continue
        checked_loras += 1
        compatible = not errors
        failures += not compatible
        status = "PASS" if compatible else "FAIL"
        detail = f"layers={matches}" if compatible else f"layers={matches}, errors={len(errors)}"
        label = str(path.relative_to(lora_directory)) if show_identifiers else f"adapter #{checked_loras}"
        print(f"{status} {family} LoRA: {label} ({detail})")

    if not checked:
        print("No compatible Anima or SDXL checkpoints were found in the configured model root.")
        return 1
    print(
        f"Checked {checked} compatible checkpoints and {checked_loras} compatible LoRAs; "
        f"{failures} failed the structural audit."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
