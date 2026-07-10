"""Small, framework-independent runtime helpers for local image inference.

Only PyTorch is used as the numerical/CUDA layer. Checkpoints are read directly
from the safetensors file format, tokenization is implemented here, and images
are encoded as PNG bytes without writing a temporary file.
"""

from __future__ import annotations

import binascii
import ctypes
import functools
import gzip
import hashlib
import html
import json
import math
import mmap
import os
import struct
import sys
import time
import unicodedata
import warnings
import zlib
from pathlib import Path
from typing import Callable, Iterable

import torch


ASSET_DIR = Path(__file__).resolve().parent / "assets"
TOKENIZER_ASSET_HASHES = {
    "clip_bpe.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    "qwen_vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "qwen_merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "t5_tokenizer.json": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
}


class InferenceError(RuntimeError):
    """A user-actionable inference error safe to return over the local API."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}.")
        result[key] = value
    return result


class StageProfiler:
    """Opt-in synchronized timings for local optimization runs."""

    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.enabled = os.environ.get("IMAGE_PROFILE") == "1"
        self.started = time.perf_counter()
        self.previous = self.started
        self.stages: dict[str, float] = {}
        self.stage_peak_vram: dict[str, float] = {}
        self.stage_end_vram: dict[str, float] = {}
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.stages[name] = round(now - self.previous, 4)
        if torch.cuda.is_available():
            scale = 1024 * 1024
            self.stage_peak_vram[name] = round(torch.cuda.max_memory_allocated() / scale, 1)
            self.stage_end_vram[name] = round(torch.cuda.memory_allocated() / scale, 1)
            torch.cuda.reset_peak_memory_stats()
        self.previous = now

    def finish(self) -> None:
        if not self.enabled:
            return
        payload = {
            "pipeline": self.pipeline,
            "stagesSeconds": self.stages,
            "stagePeakVramMiB": self.stage_peak_vram,
            "stageEndVramMiB": self.stage_end_vram,
            "totalSeconds": round(time.perf_counter() - self.started, 4),
            "peakVramMiB": max(self.stage_peak_vram.values(), default=0.0),
        }
        print("IMAGE_PROFILE " + json.dumps(payload, separators=(",", ":")), file=sys.stderr, flush=True)


class SafeTensorFile:
    """Minimal, read-only safetensors reader backed by an OS memory map."""

    _DTYPES = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "U16": getattr(torch, "uint16", None),
        "I32": torch.int32,
        "U32": getattr(torch, "uint32", None),
        "I64": torch.int64,
        "U64": getattr(torch, "uint64", None),
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "C64": torch.complex64,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
        "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
        "F8_E4M3FNUZ": getattr(torch, "float8_e4m3fnuz", None),
        "F8_E5M2FNUZ": getattr(torch, "float8_e5m2fnuz", None),
    }
    _DTYPE_BITS = {
        "F4": 4,
        "F6_E2M3": 6,
        "F6_E3M2": 6,
        "BOOL": 8,
        "U8": 8,
        "I8": 8,
        "F8_E4M3": 8,
        "F8_E5M2": 8,
        "F8_E8M0": 8,
        "F8_E4M3FNUZ": 8,
        "F8_E5M2FNUZ": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "C64": 64,
        "F64": 64,
        "I64": 64,
        "U64": 64,
    }

    def __init__(self, filename: str | os.PathLike[str]):
        self.path = Path(filename)
        self._file = self.path.open("rb")
        header_size_bytes = self._file.read(8)
        if len(header_size_bytes) != 8:
            self.close()
            raise InferenceError(f"{self.path.name} is not a valid safetensors file.")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size <= 1 or header_size > 100_000_000:
            self.close()
            raise InferenceError(f"{self.path.name} has an invalid safetensors header.")
        header_bytes = self._file.read(header_size)
        if header_bytes[:1] != b"{" or header_bytes.rstrip(b" ")[-1:] != b"}":
            self.close()
            raise InferenceError(f"{self.path.name} has an invalid safetensors header encoding.")
        try:
            self.header = json.loads(header_bytes, object_pairs_hook=_unique_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.close()
            raise InferenceError(f"{self.path.name} has a damaged safetensors header.") from error
        if not isinstance(self.header, dict):
            self.close()
            raise InferenceError(f"{self.path.name} has an invalid safetensors header.")
        self.metadata = self.header.get("__metadata__", {})
        if not isinstance(self.metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            self.close()
            raise InferenceError(f"{self.path.name} has invalid safetensors metadata.")
        self.data_offset = 8 + header_size
        self._file.seek(0, os.SEEK_END)
        data_size = self._file.tell() - self.data_offset
        try:
            self._validate_entries(data_size)
        except InferenceError:
            self.close()
            raise
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __contains__(self, name: str) -> bool:
        return name in self.header and name != "__metadata__"

    def keys(self) -> Iterable[str]:
        return (name for name in self.header if name != "__metadata__")

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._entry(name)["shape"])

    def dtype(self, name: str) -> torch.dtype:
        dtype_name = self._entry(name)["dtype"]
        dtype = self._DTYPES.get(dtype_name)
        if dtype is None:
            raise InferenceError(f"Tensor {name} uses unsupported dtype {dtype_name}.")
        return dtype

    def tensor(self, name: str) -> torch.Tensor:
        entry = self._entry(name)
        dtype = self.dtype(name)
        shape = tuple(entry["shape"])
        count = math.prod(shape)
        start, end = entry["data_offsets"]
        if start < 0 or end < start or self.data_offset + end > len(self._map):
            raise InferenceError(f"Tensor {name} has invalid data offsets.")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                return torch.frombuffer(
                    self._map,
                    dtype=dtype,
                    count=count,
                    offset=self.data_offset + start,
                ).reshape(shape)
        except (RuntimeError, ValueError) as error:
            raise InferenceError(f"Tensor {name} could not be read from {self.path.name}.") from error

    def _entry(self, name: str) -> dict:
        entry = self.header.get(name)
        if not isinstance(entry, dict):
            raise InferenceError(f"Checkpoint is missing required tensor {name}.")
        return entry

    def _validate_entries(self, data_size: int) -> None:
        if data_size < 0:
            raise InferenceError(f"{self.path.name} is truncated after its safetensors header.")
        spans: list[tuple[int, int, str]] = []
        for name, entry in self.header.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict):
                raise InferenceError(f"Tensor {name} has an invalid safetensors entry.")
            dtype_name = entry.get("dtype")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if not isinstance(dtype_name, str) or dtype_name not in self._DTYPE_BITS:
                raise InferenceError(f"Tensor {name} uses unsupported dtype {dtype_name}.")
            if not isinstance(shape, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            ):
                raise InferenceError(f"Tensor {name} has an invalid shape.")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
            ):
                raise InferenceError(f"Tensor {name} has invalid data offsets.")
            start, end = offsets
            expected_bits = math.prod(shape) * self._DTYPE_BITS[dtype_name]
            if expected_bits % 8:
                raise InferenceError(f"Tensor {name} is not aligned to a whole byte.")
            expected_size = expected_bits // 8
            if start < 0 or end < start or end > data_size or end - start != expected_size:
                raise InferenceError(f"Tensor {name} has invalid data offsets.")
            spans.append((start, end, name))
        spans.sort()
        expected_start = 0
        for start, end, name in spans:
            if start != expected_start:
                raise InferenceError(f"Tensor {name} leaves a gap or overlaps another tensor.")
            expected_start = end
        if expected_start != data_size:
            raise InferenceError(f"{self.path.name} contains unindexed trailing data.")

    def close(self) -> None:
        mapped = getattr(self, "_map", None)
        if mapped is not None:
            try:
                mapped.close()
            except BufferError:
                # A short-lived torch view can keep the mapping alive. The process
                # exits after the request, so retaining a read-only map is harmless.
                pass
            self._map = None
        file_handle = getattr(self, "_file", None)
        if file_handle is not None:
            file_handle.close()
            self._file = None

    def __enter__(self) -> "SafeTensorFile":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def materialize_module(
    module: torch.nn.Module,
    checkpoint: SafeTensorFile,
    prefix: str,
    device: torch.device,
    rename: Callable[[str], str] | None = None,
    memory_format: torch.memory_format | None = None,
) -> torch.nn.Module:
    """Allocate a meta module and copy matching checkpoint parameters into it."""

    if memory_format is not None:
        # Establish convolution-friendly strides while tensors are still meta so
        # changing layouts never requires a second, peak-memory-heavy GPU copy.
        module.to(memory_format=memory_format)
    module.to_empty(device=device)
    missing: list[str] = []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            checkpoint_name = prefix + (rename(name) if rename else name)
            if checkpoint_name not in checkpoint:
                missing.append(checkpoint_name)
                continue
            source = checkpoint.tensor(checkpoint_name)
            if tuple(source.shape) != tuple(parameter.shape):
                raise InferenceError(
                    f"Checkpoint tensor {checkpoint_name} has shape {tuple(source.shape)}; "
                    f"expected {tuple(parameter.shape)}."
                )
            # copy_ performs the host-to-device transfer and dtype conversion in
            # one operation.  Building a temporary CUDA tensor here used to add
            # an avoidable allocation and a second device copy for every weight.
            parameter.copy_(source)
            del source
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise InferenceError(f"Checkpoint is missing required weights: {preview}{suffix}")
    module.eval()
    return module


def require_cuda() -> tuple[torch.device, torch.dtype]:
    if not torch.cuda.is_available():
        raise InferenceError("Direct image generation needs an NVIDIA CUDA GPU.")
    device = torch.device("cuda")
    major, _minor = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    return device, dtype


def release_cuda(*objects: object) -> None:
    for item in objects:
        del item
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)


def timestep_embedding(values: torch.Tensor, dimension: int, max_period: int = 10_000) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=values.device)
        / half
    )
    angles = values.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(x.dtype) * weight


def encode_png(image: torch.Tensor) -> bytes:
    """Encode [H,W,3] uint8-compatible data as an RGB PNG in memory."""

    if image.ndim != 3 or image.shape[-1] != 3:
        raise InferenceError("The decoder returned an invalid image shape.")
    pixels = image.detach().clamp(0, 255).to(device="cpu", dtype=torch.uint8).contiguous()
    height, width, _channels = pixels.shape
    # bytes(UntypedStorage) iterates element-by-element through Python on current
    # PyTorch builds.  A single native copy from the contiguous CPU buffer is
    # several seconds faster for a megapixel image and has identical bytes.
    packed = ctypes.string_at(pixels.data_ptr(), pixels.numel())
    stride = width * 3
    scanlines = bytearray((stride + 1) * height)
    for row in range(height):
        target = row * (stride + 1)
        scanlines[target] = 0
        source = row * stride
        scanlines[target + 1 : target + stride + 1] = packed[source : source + stride]

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", binascii.crc32(payload) & 0xFFFFFFFF)

    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6)),
            chunk(b"IEND", b""),
        )
    )


def image_tensor_to_png(image: torch.Tensor) -> bytes:
    """Convert a decoder result in [-1,1], BCHW/BCTHW form to PNG bytes."""

    if image.ndim == 5:
        image = image[:, :, 0]
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] < 3:
        raise InferenceError("The decoder returned an invalid image tensor.")
    rgb = ((image[0, :3].float() + 1.0) * 127.5).round().permute(1, 2, 0)
    return encode_png(rgb)


def _bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, (chr(value) for value in characters)))


BYTE_ENCODER = _bytes_to_unicode()

CLIP_SPECIAL_TOKEN_IDS = {
    "<|startoftext|>": 49406,
    "<|endoftext|>": 49407,
}

# Qwen3's vocabulary JSON contains the BPE vocabulary only. These added
# tokens are defined by the matching tokenizer configuration and occupy the
# IDs immediately after that vocabulary.
QWEN_SPECIAL_TOKEN_IDS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
    "<|quad_start|>": 151650,
    "<|quad_end|>": 151651,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<tool_call>": 151657,
    "</tool_call>": 151658,
    "<|fim_prefix|>": 151659,
    "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661,
    "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663,
    "<|file_sep|>": 151664,
    "<tool_response>": 151665,
    "</tool_response>": 151666,
    "<think>": 151667,
    "</think>": 151668,
}


class BytePairEncoder:
    def __init__(self, vocabulary: dict[str, int], merges: Iterable[str], end_suffix: str = ""):
        self.vocabulary = vocabulary
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, line in enumerate(merges):
            pair = line.strip().split()
            if len(pair) == 2:
                self.ranks[(pair[0], pair[1])] = rank
        self.end_suffix = end_suffix

    @functools.lru_cache(maxsize=8192)
    def encode_piece(self, piece: str) -> tuple[int, ...]:
        encoded = "".join(BYTE_ENCODER[byte] for byte in piece.encode("utf-8"))
        if not encoded:
            return ()
        word = list(encoded)
        if self.end_suffix:
            word[-1] += self.end_suffix
        while len(word) > 1:
            candidates = ((self.ranks.get((word[index], word[index + 1]), 1 << 60), index) for index in range(len(word) - 1))
            best_rank, best_index = min(candidates)
            if best_rank == 1 << 60:
                break
            word[best_index : best_index + 2] = [word[best_index] + word[best_index + 1]]
        try:
            return tuple(self.vocabulary[token] for token in word)
        except KeyError as error:
            raise InferenceError("Tokenizer vocabulary is incomplete.") from error


def _unicode_kind(character: str) -> str:
    category = unicodedata.category(character)
    if category.startswith("L"):
        return "letter"
    if category.startswith("N"):
        return "number"
    return "other"


def _special_at(text: str, index: int, tokens: dict[str, int]) -> str | None:
    if index >= len(text) or text[index] != "<":
        return None
    return next((token for token in tokens if text.startswith(token, index)), None)


def _scan_clip(text: str) -> list[str]:
    pieces: list[str] = []
    index = 0
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        special = _special_at(text, index, CLIP_SPECIAL_TOKEN_IDS)
        if special:
            pieces.append(special)
            index += len(special)
            continue
        contraction = next((part for part in contractions if text.startswith(part, index)), None)
        if contraction:
            pieces.append(contraction)
            index += len(contraction)
            continue
        category = _unicode_kind(char)
        end = index + 1
        # CLIP's canonical pattern emits each Unicode number separately while
        # grouping runs of letters and punctuation.
        if category != "number":
            while end < len(text) and not text[end].isspace():
                if _special_at(text, end, CLIP_SPECIAL_TOKEN_IDS):
                    break
                next_category = _unicode_kind(text[end])
                if next_category != category or any(text.startswith(part, end) for part in contractions):
                    break
                end += 1
        pieces.append(text[index:end])
        index = end
    return pieces


class ClipTokenizer:
    def __init__(self, asset_path: Path = ASSET_DIR / "clip_bpe.txt.gz"):
        _require_asset(asset_path)
        with gzip.open(asset_path, "rt", encoding="utf-8") as source:
            merge_lines = source.read().splitlines()[1 : 49152 - 256 - 2 + 1]
        byte_values = list(BYTE_ENCODER.values())
        vocabulary_tokens = byte_values + [value + "</w>" for value in byte_values]
        vocabulary_tokens += ["".join(line.split()) for line in merge_lines if len(line.split()) == 2]
        vocabulary_tokens += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = BytePairEncoder(dict(zip(vocabulary_tokens, range(len(vocabulary_tokens)))), merge_lines, "</w>")

    def encode(self, text: str, length: int = 77) -> list[int]:
        cleaned = unicodedata.normalize(
            "NFC",
            html.unescape(html.unescape(text)),
        )
        cleaned = " ".join(cleaned.strip().lower().split())
        tokens = [49406]
        for piece in _scan_clip(cleaned):
            special_id = CLIP_SPECIAL_TOKEN_IDS.get(piece)
            if special_id is not None:
                tokens.append(special_id)
            else:
                tokens.extend(self.encoder.encode_piece(piece))
        tokens = tokens[: length - 1] + [49407]
        tokens.extend([49407] * (length - len(tokens)))
        return tokens


def _scan_qwen(text: str) -> list[str]:
    """Match Qwen2/Qwen3's dependency-free PRETOKENIZE_REGEX equivalent."""

    pieces: list[str] = []
    index = 0
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while index < len(text):
        special = _special_at(text, index, QWEN_SPECIAL_TOKEN_IDS)
        if special:
            pieces.append(special)
            index += len(special)
            continue

        contraction = next(
            (
                text[index : index + len(part)]
                for part in contractions
                if text[index : index + len(part)].lower() == part
            ),
            None,
        )
        if contraction:
            pieces.append(contraction)
            index += len(contraction)
            continue

        char = text[index]

        # [^\r\n\p{L}\p{N}]?\p{L}+
        if _unicode_kind(char) == "letter":
            end = index + 1
            while end < len(text) and _unicode_kind(text[end]) == "letter":
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        if (
            char not in "\r\n"
            and _unicode_kind(char) == "other"
            and index + 1 < len(text)
            and _unicode_kind(text[index + 1]) == "letter"
            and not _special_at(text, index, QWEN_SPECIAL_TOKEN_IDS)
        ):
            end = index + 2
            while end < len(text) and _unicode_kind(text[end]) == "letter":
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        # \p{N}
        if _unicode_kind(char) == "number":
            pieces.append(char)
            index += 1
            continue

        # ?[^\s\p{L}\p{N}]+[\r\n]* (the optional prefix is one ASCII
        # space). Added tokens are isolated before BPE pre-tokenization.
        punctuation_start = index
        punctuation_index = index
        if char == " ":
            if index + 1 >= len(text) or _special_at(text, index + 1, QWEN_SPECIAL_TOKEN_IDS):
                punctuation_index = index
            elif not text[index + 1].isspace() and _unicode_kind(text[index + 1]) == "other":
                punctuation_index = index + 1
        if punctuation_index > index or (not char.isspace() and _unicode_kind(char) == "other"):
            end = punctuation_index
            while (
                end < len(text)
                and not text[end].isspace()
                and _unicode_kind(text[end]) == "other"
                and not _special_at(text, end, QWEN_SPECIAL_TOKEN_IDS)
            ):
                end += 1
            if end > punctuation_index:
                while end < len(text) and text[end] in "\r\n":
                    end += 1
                pieces.append(text[punctuation_start:end])
                index = end
                continue

        if char.isspace():
            end = index
            while end < len(text) and text[end].isspace():
                end += 1
            whitespace = text[index:end]
            last_newline = max(whitespace.rfind("\r"), whitespace.rfind("\n"))
            if last_newline >= 0:
                # \s*[\r\n]+ consumes through the final newline but leaves
                # any following horizontal whitespace for the next match.
                length = last_newline + 1
            elif end == len(text):
                length = len(whitespace)
            else:
                # \s+(?!\S) retains the last whitespace so it can become the
                # optional prefix of the following letter/punctuation piece.
                length = max(1, len(whitespace) - 1)
            pieces.append(whitespace[:length])
            index += length
            continue

        # The regex alternatives cover every Unicode codepoint; retain a safe
        # fallback so malformed surrogate-containing strings still progress.
        pieces.append(char)
        index += 1
    return pieces


class QwenTokenizer:
    def __init__(
        self,
        vocabulary_path: Path = ASSET_DIR / "qwen_vocab.json",
        merges_path: Path = ASSET_DIR / "qwen_merges.txt",
    ):
        _require_asset(vocabulary_path)
        _require_asset(merges_path)
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        merges = merges_path.read_text(encoding="utf-8").splitlines()
        self.encoder = BytePairEncoder(vocabulary, merges[1:] if merges and merges[0].startswith("#") else merges)

    def encode(self, text: str, maximum: int = 512) -> list[int]:
        tokens: list[int] = []
        for piece in _scan_qwen(text):
            special_id = QWEN_SPECIAL_TOKEN_IDS.get(piece)
            if special_id is not None:
                tokens.append(special_id)
            else:
                tokens.extend(self.encoder.encode_piece(piece))
            if len(tokens) >= maximum:
                break
        return tokens[:maximum]


class T5Tokenizer:
    """Small unigram tokenizer for the T5 token IDs consumed by Anima."""

    def __init__(self, asset_path: Path = ASSET_DIR / "t5_tokenizer.json"):
        _require_asset(asset_path)
        payload = json.loads(asset_path.read_text(encoding="utf-8"))
        vocabulary = payload.get("model", {}).get("vocab", [])
        if len(vocabulary) < 32_000:
            raise InferenceError("The T5 tokenizer table is invalid.")
        self.trie: dict = {}
        for token_id, item in enumerate(vocabulary):
            piece, score = item
            if token_id < 3 or not piece or piece.startswith("<extra_id_"):
                continue
            node = self.trie
            for character in piece:
                node = node.setdefault(character, {})
            node[None] = (token_id, float(score))

    def encode(self, text: str, maximum: int = 512) -> list[int]:
        normalized = unicodedata.normalize("NFKC", text).strip()
        if not normalized:
            return [1]
        normalized = "▁" + "▁".join(normalized.split())
        length = len(normalized)
        scores = [-math.inf] * (length + 1)
        previous: list[tuple[int, int] | None] = [None] * (length + 1)
        scores[0] = 0.0
        for start in range(length):
            if not math.isfinite(scores[start]):
                continue
            node = self.trie
            end = start
            found = False
            while end < length and normalized[end] in node:
                node = node[normalized[end]]
                end += 1
                terminal = node.get(None)
                if terminal:
                    found = True
                    token_id, token_score = terminal
                    candidate = scores[start] + token_score
                    if candidate > scores[end]:
                        scores[end] = candidate
                        previous[end] = (start, token_id)
            if not found and scores[start] - 100.0 > scores[start + 1]:
                scores[start + 1] = scores[start] - 100.0
                previous[start + 1] = (start, 2)
        tokens: list[int] = []
        cursor = length
        while cursor > 0 and previous[cursor] is not None:
            cursor, token_id = previous[cursor]
            tokens.append(token_id)
        tokens.reverse()
        tokens = [token for index, token in enumerate(tokens) if token != 2 or index == 0 or tokens[index - 1] != 2]
        tokens = tokens[: max(0, maximum - 1)]
        tokens.append(1)
        return tokens


def _require_asset(path: Path) -> None:
    if not path.is_file():
        raise InferenceError(
            f"Missing tokenizer data {path.name}. Run npm run setup:image once on the computer."
        )


@functools.lru_cache(maxsize=1)
def tokenizer_assets_ready() -> bool:
    return all(
        _asset_matches(ASSET_DIR / filename, expected_hash)
        for filename, expected_hash in TOKENIZER_ASSET_HASHES.items()
    )


def _asset_matches(path: Path, expected_hash: str) -> bool:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_hash
    except OSError:
        return False
