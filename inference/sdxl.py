"""Direct SDXL inference using only PyTorch and local checkpoint data."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import apply_loras, kohya_targets, openclip_kohya_targets
from runtime import (
    ClipTokenizer,
    InferenceError,
    SafeTensorFile,
    StageProfiler,
    image_tensor_to_png,
    make_generator,
    materialize_module,
    require_cuda,
    timestep_embedding,
)


def _linear(in_features: int, out_features: int, bias: bool, device, dtype) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)


class ClipSelfAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, device=None, dtype=None):
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.k_proj = _linear(dimension, dimension, True, device, dtype)
        self.v_proj = _linear(dimension, dimension, True, device, dtype)
        self.q_proj = _linear(dimension, dimension, True, device, dtype)
        self.out_proj = _linear(dimension, dimension, True, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = x.shape
        head_dimension = dimension // self.heads
        query = self.q_proj(x).view(batch, length, self.heads, head_dimension).transpose(1, 2)
        key = self.k_proj(x).view(batch, length, self.heads, head_dimension).transpose(1, 2)
        value = self.v_proj(x).view(batch, length, self.heads, head_dimension).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, length, dimension))


class ClipMLP(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.fc1 = _linear(768, 3072, True, device, dtype)
        self.fc2 = _linear(3072, 768, True, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        return self.fc2(hidden * torch.sigmoid(1.702 * hidden))


class ClipEncoderLayer(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.self_attn = ClipSelfAttention(768, 12, device=device, dtype=dtype)
        self.layer_norm1 = nn.LayerNorm(768, device=device, dtype=dtype)
        self.mlp = ClipMLP(device=device, dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(768, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class ClipTextEmbeddings(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.token_embedding = nn.Embedding(49408, 768, device=device, dtype=dtype)
        self.position_embedding = nn.Embedding(77, 768, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)[None]


class ClipEncoder(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        # SDXL consumes the hidden state immediately before CLIP-L's last layer.
        self.layers = nn.ModuleList(ClipEncoderLayer(device=device, dtype=dtype) for _ in range(11))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ClipTextModel(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.embeddings = ClipTextEmbeddings(device=device, dtype=dtype)
        self.encoder = ClipEncoder(device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embeddings(token_ids))


class ClipL(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.text_model = ClipTextModel(device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model(token_ids)


class OpenAttention(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty((3840, 1280), device=device, dtype=dtype))
        self.in_proj_bias = nn.Parameter(torch.empty(3840, device=device, dtype=dtype))
        self.out_proj = _linear(1280, 1280, True, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = x.shape
        query, key, value = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        query = query.view(batch, length, 20, 64).transpose(1, 2)
        key = key.view(batch, length, 20, 64).transpose(1, 2)
        value = value.view(batch, length, 20, 64).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, length, dimension))


class OpenMLP(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.c_fc = _linear(1280, 5120, True, device, dtype)
        self.c_proj = _linear(5120, 1280, True, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class OpenResidualBlock(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(1280, device=device, dtype=dtype)
        self.attn = OpenAttention(device=device, dtype=dtype)
        self.ln_2 = nn.LayerNorm(1280, device=device, dtype=dtype)
        self.mlp = OpenMLP(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class OpenTransformer(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.resblocks = nn.ModuleList(OpenResidualBlock(device=device, dtype=dtype) for _ in range(32))


class OpenClipBigG(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.token_embedding = nn.Embedding(49408, 1280, device=device, dtype=dtype)
        self.positional_embedding = nn.Parameter(torch.empty((77, 1280), device=device, dtype=dtype))
        self.transformer = OpenTransformer(device=device, dtype=dtype)
        self.ln_final = nn.LayerNorm(1280, device=device, dtype=dtype)
        self.text_projection = nn.Parameter(torch.empty((1280, 1280), device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(token_ids) + self.positional_embedding[None, : token_ids.shape[1]]
        hidden = None
        for index, block in enumerate(self.transformer.resblocks):
            x = block(x)
            if index == 30:
                hidden = x
        normalized = self.ln_final(x)
        end_positions = token_ids.argmax(dim=-1)
        pooled = normalized[torch.arange(token_ids.shape[0], device=x.device), end_positions] @ self.text_projection
        if hidden is None:
            raise InferenceError("OpenCLIP did not return its penultimate hidden state.")
        return hidden, pooled


class CrossAttention(nn.Module):
    def __init__(self, query_dimension: int, context_dimension: int | None = None, device=None, dtype=None):
        super().__init__()
        context_dimension = query_dimension if context_dimension is None else context_dimension
        self.heads = query_dimension // 64
        self.to_q = _linear(query_dimension, query_dimension, False, device, dtype)
        self.to_k = _linear(context_dimension, query_dimension, False, device, dtype)
        self.to_v = _linear(context_dimension, query_dimension, False, device, dtype)
        self.to_out = nn.Sequential(_linear(query_dimension, query_dimension, True, device, dtype), nn.Dropout(0.0))

    def project_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, key_length, _dimension = context.shape
        key = self.to_k(context).view(batch, key_length, self.heads, 64).transpose(1, 2)
        value = self.to_v(context).view(batch, key_length, self.heads, 64).transpose(1, 2)
        return key, value

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        projected_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        batch, query_length, dimension = x.shape
        query = self.to_q(x).view(batch, query_length, self.heads, 64).transpose(1, 2)
        if projected_context is None:
            key, value = self.project_context(context)
        else:
            key, value = projected_context
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.to_out(attended.transpose(1, 2).reshape(batch, query_length, dimension))


class GEGLU(nn.Module):
    def __init__(self, dimension: int, device=None, dtype=None):
        super().__init__()
        self.proj = _linear(dimension, dimension * 8, True, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dimension: int, device=None, dtype=None):
        super().__init__()
        self.net = nn.Sequential(
            GEGLU(dimension, device=device, dtype=dtype),
            nn.Dropout(0.0),
            _linear(dimension * 4, dimension, True, device, dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dimension: int, device=None, dtype=None):
        super().__init__()
        self.attn1 = CrossAttention(dimension, device=device, dtype=dtype)
        self.ff = FeedForward(dimension, device=device, dtype=dtype)
        self.attn2 = CrossAttention(dimension, 2048, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(dimension, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(dimension, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(dimension, device=device, dtype=dtype)
        self._context_kv: tuple[torch.Tensor, torch.Tensor] | None = None

    def prepare_context(self, context: torch.Tensor) -> None:
        self._context_kv = self.attn2.project_context(context)

    def clear_context(self) -> None:
        self._context_kv = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context, self._context_kv)
        return x + self.ff(self.norm3(x))


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, depth: int, device=None, dtype=None):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, device=device, dtype=dtype)
        self.proj_in = _linear(channels, channels, True, device, dtype)
        self.transformer_blocks = nn.ModuleList(
            BasicTransformerBlock(channels, device=device, dtype=dtype) for _ in range(depth)
        )
        self.proj_out = _linear(channels, channels, True, device, dtype)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, channels, height, width = x.shape
        x = self.norm(x).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context)
        x = self.proj_out(x).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return residual + x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device=None, dtype=None):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels, eps=1e-5, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device, dtype=dtype),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), _linear(1280, out_channels, True, device, dtype))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels, eps=1e-5, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, device=device, dtype=dtype),
        )
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, 1, device=device, dtype=dtype)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(x)
        embedded = self.emb_layers(embedding).to(hidden.dtype)
        hidden = hidden + embedded[:, :, None, None]
        return self.skip_connection(x) + self.out_layers(hidden)


class Downsample(nn.Module):
    def __init__(self, channels: int, device=None, dtype=None):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, device=None, dtype=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, embedding)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class SDXLUNet(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.time_embed = nn.Sequential(
            _linear(320, 1280, True, device, dtype),
            nn.SiLU(),
            _linear(1280, 1280, True, device, dtype),
        )
        self.label_emb = nn.Sequential(
            nn.Sequential(
                _linear(2816, 1280, True, device, dtype),
                nn.SiLU(),
                _linear(1280, 1280, True, device, dtype),
            )
        )
        self.input_blocks = nn.ModuleList(
            (
                TimestepEmbedSequential(nn.Conv2d(4, 320, 3, padding=1, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(320, 320, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(320, 320, device=device, dtype=dtype)),
                TimestepEmbedSequential(Downsample(320, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(320, 640, device=device, dtype=dtype), SpatialTransformer(640, 2, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(640, 640, device=device, dtype=dtype), SpatialTransformer(640, 2, device=device, dtype=dtype)),
                TimestepEmbedSequential(Downsample(640, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(640, 1280, device=device, dtype=dtype), SpatialTransformer(1280, 10, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(1280, 1280, device=device, dtype=dtype), SpatialTransformer(1280, 10, device=device, dtype=dtype)),
            )
        )
        self.middle_block = TimestepEmbedSequential(
            ResBlock(1280, 1280, device=device, dtype=dtype),
            SpatialTransformer(1280, 10, device=device, dtype=dtype),
            ResBlock(1280, 1280, device=device, dtype=dtype),
        )
        self.output_blocks = nn.ModuleList(
            (
                TimestepEmbedSequential(ResBlock(2560, 1280, device=device, dtype=dtype), SpatialTransformer(1280, 10, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(2560, 1280, device=device, dtype=dtype), SpatialTransformer(1280, 10, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(1920, 1280, device=device, dtype=dtype), SpatialTransformer(1280, 10, device=device, dtype=dtype), Upsample(1280, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(1920, 640, device=device, dtype=dtype), SpatialTransformer(640, 2, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(1280, 640, device=device, dtype=dtype), SpatialTransformer(640, 2, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(960, 640, device=device, dtype=dtype), SpatialTransformer(640, 2, device=device, dtype=dtype), Upsample(640, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(960, 320, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(640, 320, device=device, dtype=dtype)),
                TimestepEmbedSequential(ResBlock(640, 320, device=device, dtype=dtype)),
            )
        )
        self.out = nn.Sequential(
            nn.GroupNorm(32, 320, eps=1e-5, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Conv2d(320, 4, 3, padding=1, device=device, dtype=dtype),
        )

    def prepare_context(self, context: torch.Tensor) -> None:
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                module.prepare_context(context)

    def clear_context(self) -> None:
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                module.clear_context()

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding = self.time_embed(timestep_embedding(timestep, 320).to(x.dtype))
        embedding = embedding + self.label_emb(labels)
        hidden_states: list[torch.Tensor] = []
        hidden = x
        for block in self.input_blocks:
            hidden = block(hidden, embedding, context)
            hidden_states.append(hidden)
        hidden = self.middle_block(hidden, embedding, context)
        for block in self.output_blocks:
            hidden = torch.cat((hidden, hidden_states.pop()), dim=1)
            hidden = block(hidden, embedding, context)
        return self.out(hidden)


class VaeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device=None, dtype=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, device=device, dtype=dtype)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device, dtype=dtype)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, device=device, dtype=dtype)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, device=device, dtype=dtype)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, device=device, dtype=dtype)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return self.nin_shortcut(x) + hidden


class VaeAttention(nn.Module):
    def __init__(self, channels: int, device=None, dtype=None):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, device=device, dtype=dtype)
        self.q = nn.Conv2d(channels, channels, 1, device=device, dtype=dtype)
        self.k = nn.Conv2d(channels, channels, 1, device=device, dtype=dtype)
        self.v = nn.Conv2d(channels, channels, 1, device=device, dtype=dtype)
        self.proj_out = nn.Conv2d(channels, channels, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        batch, channels, height, width = normalized.shape
        query = self.q(normalized).flatten(2).transpose(1, 2).unsqueeze(1)
        key = self.k(normalized).flatten(2).transpose(1, 2).unsqueeze(1)
        value = self.v(normalized).flatten(2).transpose(1, 2).unsqueeze(1)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.squeeze(1).transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.proj_out(attended)


class VaeUpsample(nn.Module):
    def __init__(self, channels: int, device=None, dtype=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class VaeUpLevel(nn.Module):
    def __init__(self, blocks: list[nn.Module], upsample: nn.Module | None):
        super().__init__()
        self.block = nn.ModuleList(blocks)
        if upsample is not None:
            self.upsample = upsample


class VaeMiddle(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.block_1 = VaeResBlock(512, 512, device=device, dtype=dtype)
        self.attn_1 = VaeAttention(512, device=device, dtype=dtype)
        self.block_2 = VaeResBlock(512, 512, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_2(self.attn_1(self.block_1(x)))


class VaeDecoder(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1, device=device, dtype=dtype)
        self.mid = VaeMiddle(device=device, dtype=dtype)
        multipliers = (1, 2, 4, 4)
        current = 512
        levels: list[VaeUpLevel | None] = [None, None, None, None]
        for level in reversed(range(4)):
            output = 128 * multipliers[level]
            blocks: list[nn.Module] = []
            for _index in range(3):
                blocks.append(VaeResBlock(current, output, device=device, dtype=dtype))
                current = output
            upsample = VaeUpsample(current, device=device, dtype=dtype) if level != 0 else None
            levels[level] = VaeUpLevel(blocks, upsample)
        self.up = nn.ModuleList(levels)
        self.norm_out = nn.GroupNorm(32, 128, eps=1e-6, device=device, dtype=dtype)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1, device=device, dtype=dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.mid(self.conv_in(latent))
        for level in reversed(range(4)):
            for block in self.up[level].block:
                hidden = block(hidden)
            if level != 0:
                hidden = self.up[level].upsample(hidden)
        return self.conv_out(F.silu(self.norm_out(hidden)))


class SDXLImageDecoder(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, 1, device=device, dtype=dtype)
        self.decoder = VaeDecoder(device=device, dtype=dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latent / 0.13025))


def _encode_prompts(
    checkpoint: SafeTensorFile,
    token_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    loras: list[tuple[Path, float]],
    profiler: StageProfiler | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    applied_counts = [0 for _lora in loras]
    clip_l = ClipL(device="meta", dtype=dtype)
    materialize_module(clip_l, checkpoint, "conditioner.embedders.0.transformer.", device)
    clip_l_targets = kohya_targets(clip_l, "lora_te1_")
    for index, lora in enumerate(loras):
        applied_counts[index] += apply_loras(clip_l_targets, [lora])
    if profiler is not None:
        profiler.mark("clip_l_load")
    hidden_l = clip_l(token_ids).detach().to("cpu")
    if profiler is not None:
        profiler.mark("clip_l_encode")
    del clip_l_targets, clip_l

    clip_g = OpenClipBigG(device="meta", dtype=dtype)
    materialize_module(clip_g, checkpoint, "conditioner.embedders.1.model.", device)
    clip_g_targets = openclip_kohya_targets(clip_g)
    for index, lora in enumerate(loras):
        applied_counts[index] += apply_loras(clip_g_targets, [lora])
    if profiler is not None:
        profiler.mark("clip_g_load")
    hidden_g, pooled = clip_g(token_ids)
    hidden_g = hidden_g.detach().to("cpu")
    pooled = pooled.detach().to("cpu")
    if profiler is not None:
        profiler.mark("clip_g_encode")
    del clip_g_targets, clip_g
    return torch.cat((hidden_l, hidden_g), dim=-1), pooled, applied_counts


def _training_sigmas(device: torch.device) -> torch.Tensor:
    betas = torch.linspace(math.sqrt(0.00085), math.sqrt(0.012), 1000, device=device).square()
    alphas = torch.cumprod(1.0 - betas, dim=0)
    return torch.sqrt((1.0 - alphas) / alphas)


def _karras_sigmas(count: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    rho = 7.0
    ramp = torch.linspace(0.0, 1.0, count, device=device)
    maximum = sigma_max ** (1.0 / rho)
    minimum = sigma_min ** (1.0 / rho)
    values = (maximum + ramp * (minimum - maximum)) ** rho
    return torch.cat((values, values.new_zeros(1)))


def _sigma_to_timestep(sigma: torch.Tensor, training_sigmas: torch.Tensor) -> torch.Tensor:
    log_schedule = training_sigmas.log()
    log_sigma = sigma.log().clamp(min=log_schedule[0], max=log_schedule[-1])
    upper = torch.searchsorted(log_schedule, log_sigma).clamp(1, len(training_sigmas) - 1)
    lower = upper - 1
    lower_log = log_schedule[lower]
    upper_log = log_schedule[upper]
    fraction = (log_sigma - lower_log) / (upper_log - lower_log)
    return lower.to(log_sigma.dtype) + fraction


def _size_condition(pooled: torch.Tensor, width: int, height: int) -> torch.Tensor:
    values = torch.tensor(
        [height, width, 0, 0, height, width],
        device=pooled.device,
        dtype=torch.float32,
    )
    embedded = timestep_embedding(values, 256).reshape(1, -1).to(pooled.dtype)
    return torch.cat((pooled, embedded.expand(pooled.shape[0], -1)), dim=-1)


def _ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    variance = sigma_to.square() * (sigma_from.square() - sigma_to.square()) / sigma_from.square()
    sigma_up = torch.minimum(sigma_to, variance.clamp_min(0).sqrt())
    sigma_down = (sigma_to.square() - sigma_up.square()).clamp_min(0).sqrt()
    return sigma_down, sigma_up


def _sample_sdxl(
    denoiser: SDXLUNet,
    context: torch.Tensor,
    labels: torch.Tensor,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    training_sigmas = _training_sigmas(device)
    sigmas = _karras_sigmas(steps, float(training_sigmas[0]), float(training_sigmas[-1]), device)
    generator = make_generator(seed, device)
    latent = torch.randn(
        (1, 4, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    ).to(memory_format=torch.channels_last) * torch.sqrt(1.0 + sigmas[0].square()).to(dtype)

    def random_noise() -> torch.Tensor:
        return torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)

    def predict_denoised(current: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        model_input = current / torch.sqrt(1.0 + sigma.square()).to(dtype)
        timestep = _sigma_to_timestep(sigma, training_sigmas).expand(2)
        prediction = denoiser(model_input.expand(2, -1, -1, -1), timestep, context, labels)
        unconditional, conditional = prediction.chunk(2)
        epsilon = unconditional + float(cfg) * (conditional - unconditional)
        denoised = current - epsilon * sigma.to(dtype)
        del model_input, timestep, prediction, unconditional, conditional, epsilon
        return denoised

    previous_denoised = None
    previous_time = None
    for index in range(steps):
        sigma = sigmas[index]
        next_sigma = sigmas[index + 1]
        denoised = predict_denoised(latent, sigma)

        if next_sigma == 0:
            latent = denoised
        elif sampler == "euler_karras":
            derivative = (latent - denoised) / sigma.to(dtype)
            latent = latent + derivative * (next_sigma - sigma).to(dtype)
            del derivative
        elif sampler == "euler_ancestral_karras":
            sigma_down, sigma_up = _ancestral_step(sigma, next_sigma)
            derivative = (latent - denoised) / sigma.to(dtype)
            latent = latent + derivative * (sigma_down - sigma).to(dtype)
            latent = latent + random_noise() * sigma_up.to(dtype)
            del sigma_down, sigma_up, derivative
        elif sampler == "dpmpp_sde_karras":
            current_time = -sigma.log()
            next_time = -next_sigma.log()
            interval = next_time - current_time
            midpoint_time = current_time + 0.5 * interval
            midpoint_sigma = (-midpoint_time).exp()

            midpoint_down, midpoint_up = _ancestral_step(sigma, midpoint_sigma)
            midpoint_latent = (midpoint_down / sigma).to(dtype) * latent
            midpoint_latent = midpoint_latent - torch.expm1(-0.5 * interval).to(dtype) * denoised

            first_noise = random_noise()
            remaining_noise = random_noise()
            first_span = (sigma - midpoint_sigma).abs()
            remaining_span = (midpoint_sigma - next_sigma).abs()
            full_noise = (
                first_noise * first_span.sqrt().to(dtype)
                + remaining_noise * remaining_span.sqrt().to(dtype)
            ) / (first_span + remaining_span).sqrt().to(dtype)
            midpoint_latent = midpoint_latent + first_noise * midpoint_up.to(dtype)
            midpoint_denoised = predict_denoised(midpoint_latent, midpoint_sigma)

            next_down, next_up = _ancestral_step(sigma, next_sigma)
            latent = (next_down / sigma).to(dtype) * latent
            latent = latent - torch.expm1(-interval).to(dtype) * midpoint_denoised
            latent = latent + full_noise * next_up.to(dtype)
            del (
                current_time,
                next_time,
                interval,
                midpoint_time,
                midpoint_sigma,
                midpoint_down,
                midpoint_up,
                midpoint_latent,
                first_noise,
                remaining_noise,
                first_span,
                remaining_span,
                full_noise,
                midpoint_denoised,
                next_down,
                next_up,
            )
        elif sampler == "dpmpp_2m_karras":
            current_time = -sigma.log()
            next_time = -next_sigma.log()
            interval = next_time - current_time
            if previous_denoised is None or previous_time is None:
                corrected = denoised
            else:
                previous_interval = current_time - previous_time
                ratio = previous_interval / interval
                corrected = (1.0 + 1.0 / (2.0 * ratio)) * denoised - previous_denoised / (2.0 * ratio)
            latent = (next_sigma / sigma).to(dtype) * latent - torch.expm1(-interval).to(dtype) * corrected
            previous_time = current_time
            previous_denoised = denoised
        else:
            raise InferenceError("The selected SDXL sampler is not available.")
    del previous_denoised, previous_time, training_sigmas, sigmas, generator
    return latent


class SDXLSession:
    """Warm SDXL UNet/VAE weights while prompt encoders remain transient."""

    def __init__(self, model_path: Path, loras: list[tuple[Path, float]]):
        self.key = (
            "sdxl",
            str(model_path),
            tuple((str(path), float(strength)) for path, strength in loras),
        )
        self.device, preferred_dtype = require_cuda()
        self.dtype = torch.float16
        self.decoder_dtype = torch.bfloat16 if preferred_dtype == torch.bfloat16 else torch.float32
        self.loras = list(loras)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")

        self.checkpoint = SafeTensorFile(model_path)
        if (
            "model.diffusion_model.input_blocks.0.0.weight" not in self.checkpoint
            or "conditioner.embedders.1.model.token_embedding.weight" not in self.checkpoint
        ):
            self.checkpoint.close()
            raise InferenceError("The selected checkpoint is not a standard SDXL model.")
        self.denoiser = SDXLUNet(device="meta", dtype=self.dtype)
        materialize_module(
            self.denoiser,
            self.checkpoint,
            "model.diffusion_model.",
            self.device,
            memory_format=torch.channels_last,
        )
        targets = kohya_targets(self.denoiser, "lora_unet_")
        self.unet_lora_counts = [apply_loras(targets, [lora]) for lora in self.loras]
        del targets
        self.decoder = SDXLImageDecoder(device="meta", dtype=self.decoder_dtype)
        materialize_module(
            self.decoder,
            self.checkpoint,
            "first_stage_model.",
            self.device,
            memory_format=torch.channels_last,
        )
        self.tokenizer = ClipTokenizer()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler: str,
        seed: int,
    ) -> bytes:
        profiler = StageProfiler("sdxl-warm")
        token_ids = torch.tensor(
            [self.tokenizer.encode(negative_prompt), self.tokenizer.encode(prompt)],
            device=self.device,
            dtype=torch.long,
        )
        context_cpu, pooled_cpu, text_lora_counts = _encode_prompts(
            self.checkpoint,
            token_ids,
            self.device,
            self.dtype,
            self.loras,
            profiler,
        )
        del token_ids
        for index, count in enumerate(text_lora_counts):
            if count + self.unet_lora_counts[index] == 0:
                raise InferenceError(f"{self.loras[index][0].name} is not compatible with SDXL.")
        context = context_cpu.to(device=self.device, dtype=self.dtype)
        pooled = pooled_cpu.to(device=self.device, dtype=self.dtype)
        labels = _size_condition(pooled, width, height)
        del context_cpu, pooled_cpu, pooled
        try:
            self.denoiser.prepare_context(context)
            profiler.mark("conditioning")
            latent = _sample_sdxl(
                self.denoiser,
                context,
                labels,
                width,
                height,
                steps,
                cfg,
                sampler,
                seed,
                self.device,
                self.dtype,
            )
            profiler.mark("sampling")
        finally:
            # Cross-attention K/V is derived from the private prompt. Keep the
            # model weights warm, but never retain request conditioning.
            self.denoiser.clear_context()
        del context, labels
        image = self.decoder(latent.to(dtype=self.decoder_dtype, memory_format=torch.channels_last))
        profiler.mark("decode")
        png = image_tensor_to_png(image)
        profiler.mark("png")
        del latent, image
        profiler.finish()
        return png

    def close(self) -> None:
        self.denoiser = None
        self.decoder = None
        checkpoint = self.checkpoint
        self.checkpoint = None
        if checkpoint is not None:
            checkpoint.close()
        torch.cuda.empty_cache()


@torch.inference_mode()
def generate_sdxl(
    model_path: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    seed: int,
    loras: list[tuple[Path, float]],
) -> bytes:
    profiler = StageProfiler("sdxl")
    device, preferred_dtype = require_cuda()
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    tokenizer = ClipTokenizer()
    token_ids = torch.tensor(
        [tokenizer.encode(negative_prompt), tokenizer.encode(prompt)],
        device=device,
        dtype=torch.long,
    )
    checkpoint = SafeTensorFile(model_path)
    if "model.diffusion_model.input_blocks.0.0.weight" not in checkpoint or "conditioner.embedders.1.model.token_embedding.weight" not in checkpoint:
        checkpoint.close()
        raise InferenceError("The selected checkpoint is not a standard SDXL model.")
    context_cpu, pooled_cpu, lora_counts = _encode_prompts(
        checkpoint, token_ids, device, dtype, loras, profiler,
    )
    del token_ids

    denoiser = SDXLUNet(device="meta", dtype=dtype)
    materialize_module(
        denoiser,
        checkpoint,
        "model.diffusion_model.",
        device,
        memory_format=torch.channels_last,
    )
    unet_lora_targets = kohya_targets(denoiser, "lora_unet_")
    for index, lora in enumerate(loras):
        lora_counts[index] += apply_loras(unet_lora_targets, [lora])
        if not lora_counts[index]:
            checkpoint.close()
            raise InferenceError(f"{lora[0].name} is not compatible with SDXL.")
    profiler.mark("denoiser_load")
    context = context_cpu.to(device=device, dtype=dtype)
    pooled = pooled_cpu.to(device=device, dtype=dtype)
    labels = _size_condition(pooled, width, height)
    del context_cpu, pooled_cpu, pooled
    denoiser.prepare_context(context)
    profiler.mark("conditioning")

    latent = _sample_sdxl(denoiser, context, labels, width, height, steps, cfg, sampler, seed, device, dtype)
    profiler.mark("sampling")

    del denoiser, context, labels
    torch.cuda.empty_cache()

    decoder_dtype = torch.bfloat16 if preferred_dtype == torch.bfloat16 else torch.float32
    decoder = SDXLImageDecoder(device="meta", dtype=decoder_dtype)
    materialize_module(
        decoder,
        checkpoint,
        "first_stage_model.",
        device,
        memory_format=torch.channels_last,
    )
    profiler.mark("decoder_load")
    checkpoint.close()
    image = decoder(latent.to(dtype=decoder_dtype, memory_format=torch.channels_last))
    profiler.mark("decode")
    png = image_tensor_to_png(image)
    profiler.mark("png")
    del decoder, latent, image
    torch.cuda.empty_cache()
    profiler.finish()
    return png
