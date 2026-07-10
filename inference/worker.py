"""Private image worker with one-shot and short-lived warm modes.

The one-shot protocol accepts one JSON document. Warm mode accepts newline-
delimited jobs, retains model weights briefly, and clears prompt token caches
after every response. The parent stops it before text inference needs the GPU.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from anima import AnimaSession, generate_anima
from runtime import BytePairEncoder, InferenceError, tokenizer_assets_ready
from sdxl import SDXLSession, generate_sdxl


ENGINE_VERSION = 2
_image_session: AnimaSession | SDXLSession | None = None


def probe() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "ok": bool(cuda and tokenizer_assets_ready()),
        "engine": "Direct PyTorch/CUDA",
        "engineVersion": ENGINE_VERSION,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "cudaAvailable": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else "",
        "tokenizerAssets": tokenizer_assets_ready(),
    }


def _path(value: object, label: str) -> Path:
    path = Path(str(value or "")).resolve()
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        raise InferenceError(f"{label} is missing or is not a safetensors file.")
    return path


def _integer(value: object, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool):
        raise InferenceError(f"{label} is invalid.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise InferenceError(f"{label} is invalid.") from error
    if isinstance(value, float) and not value.is_integer():
        raise InferenceError(f"{label} must be a whole number.")
    if isinstance(value, str) and value.strip() != str(parsed):
        raise InferenceError(f"{label} is invalid.")
    if parsed < minimum or parsed > maximum:
        raise InferenceError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _number(value: object, minimum: float, maximum: float, label: str) -> float:
    if isinstance(value, bool):
        raise InferenceError(f"{label} is invalid.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise InferenceError(f"{label} is invalid.") from error
    if not minimum <= parsed <= maximum:
        raise InferenceError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _loras(value: object) -> list[tuple[Path, float]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 4:
        raise InferenceError("Choose no more than four LoRAs.")
    result: list[tuple[Path, float]] = []
    for item in value:
        if not isinstance(item, dict):
            raise InferenceError("A selected LoRA is invalid.")
        path = _path(item.get("path"), "A selected LoRA")
        strength = _number(item.get("strength"), -2.0, 2.0, "LoRA strength")
        if strength:
            result.append((path, strength))
    return result


def generate(payload: dict, use_session: bool = False) -> dict:
    kind = str(payload.get("kind", ""))
    if kind not in ("anima", "sdxl"):
        raise InferenceError("Choose Anima or SDXL image generation.")
    prompt_value = payload.get("prompt", "")
    negative_prompt_value = payload.get("negativePrompt", "")
    if not isinstance(prompt_value, str) or not isinstance(negative_prompt_value, str):
        raise InferenceError("Image prompts must be text.")
    prompt = prompt_value.strip()
    negative_prompt = negative_prompt_value.strip()
    if not prompt:
        raise InferenceError("Describe the image you want to generate.")
    if len(prompt) > 5000 or len(negative_prompt) > 3000:
        raise InferenceError("The image prompt is too long.")
    width = _integer(payload.get("width"), 256, 1536, "Width")
    height = _integer(payload.get("height"), 256, 1536, "Height")
    if width % 16 or height % 16:
        raise InferenceError("Image dimensions must be divisible by 16.")
    if kind == "sdxl" and (width % 32 or height % 32):
        raise InferenceError("SDXL image dimensions must be divisible by 32.")
    steps = _integer(payload.get("steps"), 1, 80, "Steps")
    cfg = _number(payload.get("cfg"), 0.0, 20.0, "CFG")
    seed = _integer(payload.get("seed"), 0, 2**63 - 1, "Seed")
    model_path = _path(payload.get("modelPath"), "The selected model")
    loras = _loras(payload.get("loras"))
    if not tokenizer_assets_ready():
        raise InferenceError("Tokenizer data is missing. Run npm run setup:image once on the computer.")

    if kind == "anima":
        text_encoder_path = _path(payload.get("textEncoderPath"), "The Anima text encoder")
        vae_path = _path(payload.get("vaePath"), "The Anima VAE")
        if use_session:
            session = _get_image_session(kind, model_path, loras, text_encoder_path, vae_path)
            png = session.generate(prompt, negative_prompt, width, height, steps, cfg, seed)
        else:
            png = generate_anima(
                model_path=model_path,
                text_encoder_path=text_encoder_path,
                vae_path=vae_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
                loras=loras,
            )
    elif use_session:
        session = _get_image_session(kind, model_path, loras)
        png = session.generate(prompt, negative_prompt, width, height, steps, cfg, seed)
    else:
        png = generate_sdxl(
            model_path=model_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            loras=loras,
        )
    return {
        "ok": True,
        "imageBase64": base64.b64encode(png).decode("ascii"),
        "mimeType": "image/png",
    }


def _get_image_session(
    kind: str,
    model_path: Path,
    loras: list[tuple[Path, float]],
    text_encoder_path: Path | None = None,
    vae_path: Path | None = None,
) -> AnimaSession | SDXLSession:
    global _image_session
    lora_key = tuple((str(path), float(strength)) for path, strength in loras)
    desired_key = (
        ("anima", str(model_path), str(text_encoder_path), str(vae_path), lora_key)
        if kind == "anima"
        else ("sdxl", str(model_path), lora_key)
    )
    if _image_session is not None and _image_session.key == desired_key:
        return _image_session
    _close_image_session()
    if kind == "anima":
        if text_encoder_path is None or vae_path is None:
            raise InferenceError("Anima model dependencies are missing.")
        _image_session = AnimaSession(model_path, text_encoder_path, vae_path, loras)
    else:
        _image_session = SDXLSession(model_path, loras)
    return _image_session


def _close_image_session() -> None:
    global _image_session
    session = _image_session
    _image_session = None
    if session is not None:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args, _unknown = parser.parse_known_args()
    if args.probe:
        print(json.dumps(probe(), separators=(",", ":")), flush=True)
        return 0
    if args.serve:
        return serve()
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise InferenceError("The image request is invalid.")
        response = generate(payload)
    except InferenceError as error:
        response = {"ok": False, "error": str(error)}
    except torch.cuda.OutOfMemoryError:
        response = {
            "ok": False,
            "error": "The GPU ran out of memory. Close another GPU model, choose a smaller image, or try again.",
        }
    except Exception as error:  # Keep prompt contents and tracebacks out of the API response.
        response = {"ok": False, "error": f"Direct inference failed ({type(error).__name__})."}
    print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0 if response.get("ok") else 1


def serve() -> int:
    """Process newline-delimited jobs until the parent closes stdin."""

    for line in sys.stdin:
        payload = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise InferenceError("The image request is invalid.")
            response = generate(payload, use_session=True)
        except InferenceError as error:
            response = {"ok": False, "error": str(error)}
        except torch.cuda.OutOfMemoryError:
            _close_image_session()
            response = {
                "ok": False,
                "error": "The GPU ran out of memory. Close another GPU model, choose a smaller image, or try again.",
            }
        except Exception as error:
            _close_image_session()
            response = {"ok": False, "error": f"Direct inference failed ({type(error).__name__})."}
        print(json.dumps(response, separators=(",", ":")), flush=True)
        # The BPE speed cache otherwise retains pieces of private prompts for the
        # life of a warm worker. Model weights stay warm; user data does not.
        BytePairEncoder.encode_piece.cache_clear()
        del payload, response
    _close_image_session()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
