"""Direct Anima inference implemented with PyTorch primitives only."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import apply_loras, kohya_targets, peft_targets
from runtime import (
    InferenceError,
    QwenTokenizer,
    SafeTensorFile,
    StageProfiler,
    T5Tokenizer,
    image_tensor_to_png,
    make_generator,
    materialize_module,
    require_cuda,
    rms_norm,
)


ANIMA_MODEL_SIGNATURE = {
    "llm_adapter.embed.weight": (32128, 1024),
    "x_embedder.proj.1.weight": (2048, 68),
    "blocks.0.self_attn.q_proj.weight": (2048, 2048),
    "final_layer.linear.weight": (64, 2048),
}
ANIMA_MODEL_PREFIXES = ("net.", "model.diffusion_model.")
ANIMA_WEIGHT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def detect_anima_prefix(checkpoint: SafeTensorFile) -> str:
    """Return the supported weight prefix after checking an Anima signature."""

    for prefix in ANIMA_MODEL_PREFIXES:
        if all(
            prefix + name in checkpoint
            and checkpoint.shape(prefix + name) == shape
            and checkpoint.dtype(prefix + name) in ANIMA_WEIGHT_DTYPES
            for name, shape in ANIMA_MODEL_SIGNATURE.items()
        ):
            return prefix
    raise InferenceError(
        "The selected checkpoint is not a compatible Anima diffusion model. "
        "Choose a full Anima model rather than a LoRA or another model family."
    )


def _linear(in_features: int, out_features: int, bias: bool, device, dtype) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension, device=device, dtype=dtype))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.epsilon)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rope_cos_sin(length: int, dimension: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = 1.0 / (
        theta ** (torch.arange(0, dimension, 2, device=device, dtype=torch.float32) / dimension)
    )
    positions = torch.arange(length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos()[None, None], angles.sin()[None, None]


class QwenAttention(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.q_proj = _linear(1024, 2048, False, device, dtype)
        self.k_proj = _linear(1024, 1024, False, device, dtype)
        self.v_proj = _linear(1024, 1024, False, device, dtype)
        self.o_proj = _linear(2048, 1024, False, device, dtype)
        self.q_norm = RMSNorm(128, device=device, dtype=dtype)
        self.k_norm = RMSNorm(128, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, length, _dimension = x.shape
        query = self.q_norm(self.q_proj(x).view(batch, length, 16, 128)).transpose(1, 2)
        key = self.k_norm(self.k_proj(x).view(batch, length, 8, 128)).transpose(1, 2)
        value = self.v_proj(x).view(batch, length, 8, 128).transpose(1, 2)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=True,
        )
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, 2048))


class QwenMLP(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.gate_proj = _linear(1024, 3072, False, device, dtype)
        self.up_proj = _linear(1024, 3072, False, device, dtype)
        self.down_proj = _linear(3072, 1024, False, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenLayer(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.self_attn = QwenAttention(device=device, dtype=dtype)
        self.mlp = QwenMLP(device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(1024, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(1024, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenModel(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(151936, 1024, device=device, dtype=dtype)
        self.layers = nn.ModuleList(QwenLayer(device=device, dtype=dtype) for _ in range(28))
        self.norm = RMSNorm(1024, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(token_ids)
        cos, sin = _rope_cos_sin(x.shape[1], 128, 1_000_000.0, x.device)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class QwenEncoder(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.model = QwenModel(device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids)


class AdapterAttention(nn.Module):
    def __init__(self, query_dimension: int, context_dimension: int, heads: int, device=None, dtype=None):
        super().__init__()
        head_dimension = query_dimension // heads
        self.heads = heads
        self.head_dimension = head_dimension
        self.q_proj = _linear(query_dimension, query_dimension, False, device, dtype)
        self.q_norm = RMSNorm(head_dimension, device=device, dtype=dtype)
        self.k_proj = _linear(context_dimension, query_dimension, False, device, dtype)
        self.k_norm = RMSNorm(head_dimension, device=device, dtype=dtype)
        self.v_proj = _linear(context_dimension, query_dimension, False, device, dtype)
        self.o_proj = _linear(query_dimension, query_dimension, False, device, dtype)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        query_position: tuple[torch.Tensor, torch.Tensor],
        key_position: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        context = x if context is None else context
        batch, query_length, dimension = x.shape
        key_length = context.shape[1]
        query = self.q_norm(self.q_proj(x).view(batch, query_length, self.heads, self.head_dimension)).transpose(1, 2)
        key = self.k_norm(self.k_proj(context).view(batch, key_length, self.heads, self.head_dimension)).transpose(1, 2)
        value = self.v_proj(context).view(batch, key_length, self.heads, self.head_dimension).transpose(1, 2)
        query = query * query_position[0] + _rotate_half(query) * query_position[1]
        key = key * key_position[0] + _rotate_half(key) * key_position[1]
        output = F.scaled_dot_product_attention(query, key, value)
        return self.o_proj(output.transpose(1, 2).reshape(batch, query_length, dimension))


class AdapterBlock(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.norm_self_attn = RMSNorm(1024, device=device, dtype=dtype)
        self.self_attn = AdapterAttention(1024, 1024, 16, device=device, dtype=dtype)
        self.norm_cross_attn = RMSNorm(1024, device=device, dtype=dtype)
        self.cross_attn = AdapterAttention(1024, 1024, 16, device=device, dtype=dtype)
        self.norm_mlp = RMSNorm(1024, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            _linear(1024, 4096, True, device, dtype),
            nn.GELU(),
            _linear(4096, 1024, True, device, dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        target_position: tuple[torch.Tensor, torch.Tensor],
        source_position: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normalized = self.norm_self_attn(x)
        x = x + self.self_attn(normalized, None, target_position, target_position)
        x = x + self.cross_attn(self.norm_cross_attn(x), context, target_position, source_position)
        return x + self.mlp(self.norm_mlp(x))


class LLMAdapter(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.embed = nn.Embedding(32128, 1024, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(AdapterBlock(device=device, dtype=dtype) for _ in range(6))
        self.out_proj = _linear(1024, 1024, True, device, dtype)
        self.norm = RMSNorm(1024, device=device, dtype=dtype)

    def forward(self, source: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(target_ids)
        target_position = _rope_cos_sin(x.shape[1], 64, 10_000.0, x.device)
        source_position = _rope_cos_sin(source.shape[1], 64, 10_000.0, x.device)
        target_position = tuple(item.to(x.dtype) for item in target_position)
        source_position = tuple(item.to(x.dtype) for item in source_position)
        for block in self.blocks:
            x = block(x, source, target_position, source_position)
        return self.norm(self.out_proj(x))


def _apply_cosmos_rope(tensor: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    original_shape = tensor.shape
    paired = tensor.reshape(*original_shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    output = frequencies[..., 0] * paired[..., 0] + frequencies[..., 1] * paired[..., 1]
    return output.movedim(-1, -2).reshape(original_shape).to(tensor.dtype)


def _cosmos_rope(height: int, width: int, device: torch.device) -> torch.Tensor:
    head_dimension = 128
    height_dimension = head_dimension // 6 * 2
    width_dimension = height_dimension
    time_dimension = head_dimension - height_dimension - width_dimension

    def axis(length: int, dimension: int, theta: float) -> torch.Tensor:
        powers = torch.arange(0, dimension, 2, device=device, dtype=torch.float32) / dimension
        angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), 1.0 / (theta ** powers))
        return torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), dim=-1).reshape(length, dimension // 2, 2, 2)

    # Anima uses 4x spatial extrapolation from the Cosmos training grid.
    height_factor = 4.0 ** (height_dimension / (height_dimension - 2))
    width_factor = 4.0 ** (width_dimension / (width_dimension - 2))
    temporal = axis(1, time_dimension, 10_000.0)
    vertical = axis(height, height_dimension, 10_000.0 * height_factor)
    horizontal = axis(width, width_dimension, 10_000.0 * width_factor)
    temporal = temporal[:, None, None].expand(1, height, width, -1, -1, -1)
    vertical = vertical[None, :, None].expand(1, height, width, -1, -1, -1)
    horizontal = horizontal[None, None, :].expand(1, height, width, -1, -1, -1)
    rope = torch.cat((temporal, vertical, horizontal), dim=3).reshape(height * width, head_dimension // 2, 2, 2)
    return rope.unsqueeze(0).unsqueeze(2)


class CosmosAttention(nn.Module):
    def __init__(self, query_dimension: int, context_dimension: int | None, device=None, dtype=None):
        super().__init__()
        self.is_self_attention = context_dimension is None
        context_dimension = query_dimension if context_dimension is None else context_dimension
        self.heads = 16
        self.head_dimension = query_dimension // self.heads
        self.q_proj = _linear(query_dimension, query_dimension, False, device, dtype)
        self.q_norm = RMSNorm(self.head_dimension, device=device, dtype=dtype)
        self.k_proj = _linear(context_dimension, query_dimension, False, device, dtype)
        self.k_norm = RMSNorm(self.head_dimension, device=device, dtype=dtype)
        self.v_proj = _linear(context_dimension, query_dimension, False, device, dtype)
        self.output_proj = _linear(query_dimension, query_dimension, False, device, dtype)

    def project_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, key_length, _dimension = context.shape
        key = self.k_norm(self.k_proj(context).view(batch, key_length, self.heads, self.head_dimension))
        value = self.v_proj(context).view(batch, key_length, self.heads, self.head_dimension)
        return key, value

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        rope: torch.Tensor | None,
        projected_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        batch, query_length, dimension = x.shape
        query = self.q_norm(self.q_proj(x).view(batch, query_length, self.heads, self.head_dimension))
        if projected_context is None:
            key, value = self.project_context(context)
        else:
            key, value = projected_context
        if self.is_self_attention and rope is not None:
            query = _apply_cosmos_rope(query, rope)
            key = _apply_cosmos_rope(key, rope)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        return self.output_proj(attended.transpose(1, 2).reshape(batch, query_length, dimension))


class CosmosFeedForward(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.layer1 = _linear(2048, 8192, False, device, dtype)
        self.layer2 = _linear(8192, 2048, False, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


class CosmosBlock(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.self_attn = CosmosAttention(2048, None, device=device, dtype=dtype)
        self.cross_attn = CosmosAttention(2048, 1024, device=device, dtype=dtype)
        self.mlp = CosmosFeedForward(device=device, dtype=dtype)
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(),
            _linear(2048, 256, False, device, dtype),
            _linear(256, 6144, False, device, dtype),
        )
        self.adaln_modulation_cross_attn = nn.Sequential(
            nn.SiLU(),
            _linear(2048, 256, False, device, dtype),
            _linear(256, 6144, False, device, dtype),
        )
        self.adaln_modulation_mlp = nn.Sequential(
            nn.SiLU(),
            _linear(2048, 256, False, device, dtype),
            _linear(256, 6144, False, device, dtype),
        )

    @staticmethod
    def _condition(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],)) * (1.0 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        adaln: torch.Tensor,
        context: torch.Tensor,
        rope: torch.Tensor,
        context_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self_parts = (self.adaln_modulation_self_attn(time_embedding) + adaln).chunk(3, dim=-1)
        cross_parts = (self.adaln_modulation_cross_attn(time_embedding) + adaln).chunk(3, dim=-1)
        mlp_parts = (self.adaln_modulation_mlp(time_embedding) + adaln).chunk(3, dim=-1)
        self_shift, self_scale, self_gate = (part[:, :, None, None] for part in self_parts)
        cross_shift, cross_scale, cross_gate = (part[:, :, None, None] for part in cross_parts)
        mlp_shift, mlp_scale, mlp_gate = (part[:, :, None, None] for part in mlp_parts)
        batch, frames, height, width, dimension = x.shape

        normalized = self._condition(x, self_shift, self_scale)
        result = self.self_attn(normalized.reshape(batch, frames * height * width, dimension).to(time_embedding.dtype), None, rope)
        x = x + self_gate.to(x.dtype) * result.reshape_as(x).to(x.dtype)

        normalized = self._condition(x, cross_shift, cross_scale)
        result = self.cross_attn(
            normalized.reshape(batch, frames * height * width, dimension).to(time_embedding.dtype),
            context,
            None,
            context_kv,
        )
        x = x + cross_gate.to(x.dtype) * result.reshape_as(x).to(x.dtype)

        normalized = self._condition(x, mlp_shift, mlp_scale)
        return x + mlp_gate.to(x.dtype) * self.mlp(normalized.to(time_embedding.dtype)).to(x.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.proj = nn.Sequential(nn.Identity(), _linear(68, 2048, False, device, dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        x = x.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).reshape(batch, frames, height // 2, width // 2, channels * 4)
        # The checkpoint's last four input features multiply the synthetic zero
        # channel used by Cosmos video models. Images always have one frame and
        # that channel is identically zero, so omit both its allocation and GEMM.
        linear = self.proj[1]
        return F.linear(x, linear.weight[:, : channels * 4], linear.bias)


class CosmosTimesteps(nn.Module):
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        values = timesteps.flatten().float()
        half = 1024
        exponent = -math.log(10_000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / half
        angles = values[:, None] * torch.exp(exponent)[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return embedding.reshape(timesteps.shape[0], timesteps.shape[1], 2048)


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.linear_1 = _linear(2048, 2048, False, device, dtype)
        self.linear_2 = _linear(2048, 6144, False, device, dtype)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adaln = self.linear_2(F.silu(self.linear_1(sample)))
        return sample, adaln


class CosmosFinalLayer(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            _linear(2048, 256, False, device, dtype),
            _linear(256, 4096, False, device, dtype),
        )
        self.linear = _linear(2048, 64, False, device, dtype)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, adaln: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.adaln_modulation(time_embedding) + adaln[:, :, :4096]).chunk(2, dim=-1)
        x = F.layer_norm(x, (x.shape[-1],)) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.linear(x)


class AnimaDenoiser(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.x_embedder = PatchEmbed(device=device, dtype=dtype)
        self.t_embedder = nn.Sequential(CosmosTimesteps(), CosmosTimestepEmbedding(device=device, dtype=dtype))
        self.t_embedding_norm = RMSNorm(2048, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(CosmosBlock(device=device, dtype=dtype) for _ in range(28))
        self.final_layer = CosmosFinalLayer(device=device, dtype=dtype)
        self.llm_adapter = LLMAdapter(device=device, dtype=dtype)

    def prepare_context(self, source: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        output = self.llm_adapter(source, target_ids)
        if output.shape[1] < 512:
            output = F.pad(output, (0, 0, 0, 512 - output.shape[1]))
        return output[:, :512]

    def prepare_denoising(
        self,
        context: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        rope = _cosmos_rope(latent_height // 2, latent_width // 2, context.device)
        context_kv = [block.cross_attn.project_context(context) for block in self.blocks]
        return rope, context_kv

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rope: torch.Tensor | None = None,
        context_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        batch, _channels, frames, height, width = x.shape
        embedded = self.x_embedder(x)
        time_values = timestep.reshape(batch, 1)
        time_raw = self.t_embedder[0](time_values).to(embedded.dtype)
        time_embedding, adaln = self.t_embedder[1](time_raw)
        time_embedding = self.t_embedding_norm(time_embedding)
        if rope is None:
            rope = _cosmos_rope(embedded.shape[2], embedded.shape[3], x.device)
        if embedded.dtype == torch.float16:
            embedded = embedded.float()
        for index, block in enumerate(self.blocks):
            cached = context_kv[index] if context_kv is not None else None
            embedded = block(embedded, time_embedding, adaln, context, rope, cached)
        output = self.final_layer(embedded.to(context.dtype), time_embedding, adaln)
        output = output.reshape(batch, frames, output.shape[2], output.shape[3], 1, 2, 2, 16)
        return output.permute(0, 7, 1, 2, 5, 3, 6, 4).reshape(batch, 16, frames, height, width)


class CausalConv3d(nn.Conv3d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias=True, device=None, dtype=None):
        kernel = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        stride_value = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        padding_value = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        self.time_padding = 2 * padding_value[0]
        super().__init__(
            in_channels,
            out_channels,
            kernel,
            stride=stride_value,
            padding=(0, padding_value[1], padding_value[2]),
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_padding:
            x = F.pad(x, (0, 0, 0, 0, self.time_padding, 0))
        return super().forward(x)


class WanRMSNorm(nn.Module):
    def __init__(self, dimension: int, images: bool, device=None, dtype=None):
        super().__init__()
        shape = (dimension, 1, 1) if images else (dimension, 1, 1, 1)
        self.gamma = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
        self.scale = dimension**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x)


class WanResidualBlock(nn.Module):
    def __init__(self, in_dimension: int, out_dimension: int, device=None, dtype=None):
        super().__init__()
        self.residual = nn.Sequential(
            WanRMSNorm(in_dimension, images=False, device=device, dtype=dtype),
            nn.SiLU(),
            CausalConv3d(in_dimension, out_dimension, 3, padding=1, device=device, dtype=dtype),
            WanRMSNorm(out_dimension, images=False, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(0.0),
            CausalConv3d(out_dimension, out_dimension, 3, padding=1, device=device, dtype=dtype),
        )
        self.shortcut = (
            CausalConv3d(in_dimension, out_dimension, 1, device=device, dtype=dtype)
            if in_dimension != out_dimension
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class WanAttentionBlock(nn.Module):
    def __init__(self, dimension: int, device=None, dtype=None):
        super().__init__()
        self.norm = WanRMSNorm(dimension, images=True, device=device, dtype=dtype)
        self.to_qkv = nn.Conv2d(dimension, dimension * 3, 1, device=device, dtype=dtype)
        self.proj = nn.Conv2d(dimension, dimension, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch, channels, frames, height, width = x.shape
        flattened = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        query, key, value = self.to_qkv(self.norm(flattened)).chunk(3, dim=1)
        query = query.flatten(2).transpose(1, 2).unsqueeze(1)
        key = key.flatten(2).transpose(1, 2).unsqueeze(1)
        value = value.flatten(2).transpose(1, 2).unsqueeze(1)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.squeeze(1).transpose(1, 2).reshape(batch * frames, channels, height, width)
        attended = self.proj(attended).reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return identity + attended


class WanResample(nn.Module):
    def __init__(self, dimension: int, mode: str, device=None, dtype=None):
        super().__init__()
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dimension, dimension // 2, 3, padding=1, device=device, dtype=dtype),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dimension, dimension // 2, 3, padding=1, device=device, dtype=dtype),
            )
            self.time_conv = CausalConv3d(dimension, dimension * 2, (3, 1, 1), padding=(1, 0, 0), device=device, dtype=dtype)
        else:
            raise ValueError(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A one-frame image intentionally skips temporal expansion.
        batch, channels, frames, height, width = x.shape
        flattened = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        flattened = self.resample(flattened)
        out_channels, out_height, out_width = flattened.shape[1:]
        return flattened.reshape(batch, frames, out_channels, out_height, out_width).permute(0, 2, 1, 3, 4)


class WanDecoder3d(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        dimensions = [384, 384, 384, 192, 96]
        self.conv1 = CausalConv3d(16, 384, 3, padding=1, device=device, dtype=dtype)
        self.middle = nn.Sequential(
            WanResidualBlock(384, 384, device=device, dtype=dtype),
            WanAttentionBlock(384, device=device, dtype=dtype),
            WanResidualBlock(384, 384, device=device, dtype=dtype),
        )
        upsamples: list[nn.Module] = []
        temporal_modes = [True, True, False]
        for index, (input_dimension, output_dimension) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            if index in (1, 2, 3):
                input_dimension //= 2
            for _block in range(3):
                upsamples.append(WanResidualBlock(input_dimension, output_dimension, device=device, dtype=dtype))
                input_dimension = output_dimension
            if index != 3:
                mode = "upsample3d" if temporal_modes[index] else "upsample2d"
                upsamples.append(WanResample(output_dimension, mode, device=device, dtype=dtype))
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            WanRMSNorm(96, images=False, device=device, dtype=dtype),
            nn.SiLU(),
            CausalConv3d(96, 3, 3, padding=1, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        return self.head(x)


class WanImageDecoder(nn.Module):
    LATENT_MEAN = (-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921)
    LATENT_STD = (2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160)

    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.conv2 = CausalConv3d(16, 16, 1, device=device, dtype=dtype)
        self.decoder = WanDecoder3d(device=device, dtype=dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.LATENT_MEAN, device=latent.device, dtype=latent.dtype).reshape(1, 16, 1, 1, 1)
        std = torch.tensor(self.LATENT_STD, device=latent.device, dtype=latent.dtype).reshape(1, 16, 1, 1, 1)
        return self.decoder(self.conv2(latent * std + mean))


def _encode_texts(
    text_encoder_path: Path,
    tokenizer: QwenTokenizer,
    texts: tuple[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    with SafeTensorFile(text_encoder_path) as checkpoint:
        model = QwenEncoder(device="meta", dtype=dtype)
        materialize_module(model, checkpoint, "", device)
    outputs: list[torch.Tensor] = []
    for text in texts:
        if not text.strip():
            outputs.append(torch.empty((1, 0, 1024), device="cpu", dtype=dtype))
            continue
        ids = torch.tensor([tokenizer.encode(text)], device=device, dtype=torch.long)
        outputs.append(model(ids).detach().to("cpu"))
    del model
    torch.cuda.empty_cache()
    return outputs[0], outputs[1]


def _prepare_anima_context(
    denoiser: AnimaDenoiser,
    positive_source: torch.Tensor,
    negative_source: torch.Tensor,
    t5_tokenizer: T5Tokenizer,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    contexts: list[torch.Tensor] = []
    for source, text in ((negative_source, negative_prompt), (positive_source, prompt)):
        if not text.strip():
            contexts.append(torch.zeros((source.shape[0], 512, 1024), device=device, dtype=dtype))
            continue
        source = source.to(device=device, dtype=dtype)
        target = torch.tensor([t5_tokenizer.encode(text)], device=device, dtype=torch.long)
        contexts.append(denoiser.prepare_context(source, target))
        del source, target
    return torch.cat(contexts, dim=0)


def _sample_anima(
    denoiser: AnimaDenoiser,
    context: torch.Tensor,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_height, latent_width = height // 8, width // 8
    rope, context_kv = denoiser.prepare_denoising(context, latent_height, latent_width)
    latent = torch.randn(
        (1, 16, 1, latent_height, latent_width),
        generator=make_generator(seed, device),
        device=device,
        dtype=torch.float32,
    )
    base_schedule = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = 3.0 * base_schedule / (1.0 + 2.0 * base_schedule)
    for index in range(steps):
        sigma = sigmas[index]
        batch_latent = latent.to(dtype).expand(2, -1, -1, -1, -1)
        timestep = sigma.expand(2).to(dtype)
        prediction = denoiser(batch_latent, timestep, context, rope, context_kv)
        unconditional, conditional = prediction.float().chunk(2)
        velocity = unconditional + float(cfg) * (conditional - unconditional)
        latent = latent + velocity * (sigmas[index + 1] - sigma)
        del batch_latent, prediction, unconditional, conditional, velocity
    del rope, context_kv, base_schedule, sigmas
    return latent


class AnimaSession:
    """Warm Anima weights for serialized jobs in the short-lived worker."""

    def __init__(
        self,
        model_path: Path,
        text_encoder_path: Path,
        vae_path: Path,
        loras: list[tuple[Path, float]],
    ):
        self.key = (
            "anima",
            str(model_path),
            str(text_encoder_path),
            str(vae_path),
            tuple((str(path), float(strength)) for path, strength in loras),
        )
        self.device, preferred_dtype = require_cuda()
        self.dtype = torch.bfloat16 if preferred_dtype == torch.bfloat16 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        self.qwen_tokenizer = QwenTokenizer()
        self.t5_tokenizer = T5Tokenizer()

        with SafeTensorFile(text_encoder_path) as checkpoint:
            self.text_encoder = QwenEncoder(device="meta", dtype=self.dtype)
            materialize_module(self.text_encoder, checkpoint, "", self.device)
        with SafeTensorFile(model_path) as checkpoint:
            model_prefix = detect_anima_prefix(checkpoint)
            self.denoiser = AnimaDenoiser(device="meta", dtype=self.dtype)
            materialize_module(self.denoiser, checkpoint, model_prefix, self.device)
        if loras:
            targets = {
                **kohya_targets(self.denoiser, "lora_unet_"),
                **peft_targets(self.denoiser, "diffusion_model."),
            }
            for lora in loras:
                if not apply_loras(targets, [lora]):
                    raise InferenceError(f"{lora[0].name} is not compatible with Anima.")
            del targets
        with SafeTensorFile(vae_path) as checkpoint:
            if "decoder.middle.0.residual.0.gamma" not in checkpoint:
                raise InferenceError("The selected Anima VAE is not compatible.")
            self.decoder = WanImageDecoder(device="meta", dtype=self.dtype)
            materialize_module(self.decoder, checkpoint, "", self.device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
    ) -> bytes:
        profiler = StageProfiler("anima-warm")
        sources: list[torch.Tensor] = []
        for text in (prompt, negative_prompt):
            if not text.strip():
                sources.append(torch.empty((1, 0, 1024), device=self.device, dtype=self.dtype))
                continue
            ids = torch.tensor([self.qwen_tokenizer.encode(text)], device=self.device, dtype=torch.long)
            sources.append(self.text_encoder(ids).detach())
            del ids
        profiler.mark("text_encoder")
        positive_source, negative_source = sources
        context = _prepare_anima_context(
            self.denoiser,
            positive_source,
            negative_source,
            self.t5_tokenizer,
            prompt,
            negative_prompt,
            self.device,
            self.dtype,
        )
        del sources, positive_source, negative_source
        profiler.mark("conditioning")
        latent = _sample_anima(
            self.denoiser,
            context,
            width,
            height,
            steps,
            cfg,
            seed,
            self.device,
            self.dtype,
        )
        profiler.mark("sampling")
        del context
        image = self.decoder(latent.to(self.dtype))
        profiler.mark("decode")
        png = image_tensor_to_png(image)
        profiler.mark("png")
        del latent, image
        profiler.finish()
        return png

    def close(self) -> None:
        self.text_encoder = None
        self.denoiser = None
        self.decoder = None
        torch.cuda.empty_cache()


@torch.inference_mode()
def generate_anima(
    model_path: Path,
    text_encoder_path: Path,
    vae_path: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    loras: list[tuple[Path, float]],
) -> bytes:
    profiler = StageProfiler("anima")
    device, preferred_dtype = require_cuda()
    dtype = torch.bfloat16 if preferred_dtype == torch.bfloat16 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    qwen_tokenizer = QwenTokenizer()
    t5_tokenizer = T5Tokenizer()
    positive_source, negative_source = _encode_texts(
        text_encoder_path,
        qwen_tokenizer,
        (prompt, negative_prompt),
        device,
        dtype,
    )
    profiler.mark("text_encoder")

    with SafeTensorFile(model_path) as checkpoint:
        model_prefix = detect_anima_prefix(checkpoint)
        denoiser = AnimaDenoiser(device="meta", dtype=dtype)
        materialize_module(denoiser, checkpoint, model_prefix, device)
    if loras:
        lora_targets = {
            **kohya_targets(denoiser, "lora_unet_"),
            **peft_targets(denoiser, "diffusion_model."),
        }
        for lora in loras:
            if not apply_loras(lora_targets, [lora]):
                raise InferenceError(f"{lora[0].name} is not compatible with Anima.")
    profiler.mark("denoiser_load")

    context = _prepare_anima_context(
        denoiser,
        positive_source,
        negative_source,
        t5_tokenizer,
        prompt,
        negative_prompt,
        device,
        dtype,
    )
    del positive_source, negative_source
    profiler.mark("conditioning")
    latent = _sample_anima(denoiser, context, width, height, steps, cfg, seed, device, dtype)
    profiler.mark("sampling")

    del denoiser, context
    torch.cuda.empty_cache()

    with SafeTensorFile(vae_path) as checkpoint:
        if "decoder.middle.0.residual.0.gamma" not in checkpoint:
            raise InferenceError("The selected Anima VAE is not compatible.")
        decoder = WanImageDecoder(device="meta", dtype=dtype)
        materialize_module(decoder, checkpoint, "", device)
    profiler.mark("decoder_load")
    image = decoder(latent.to(dtype))
    profiler.mark("decode")
    png = image_tensor_to_png(image)
    profiler.mark("png")
    del decoder, latent, image
    torch.cuda.empty_cache()
    profiler.finish()
    return png
