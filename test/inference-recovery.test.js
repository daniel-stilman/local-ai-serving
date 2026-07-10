'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const PYTHON = process.env.IMAGE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

const results = JSON.parse(execFileSync(PYTHON, ['-B', '-c', String.raw`
import io
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, 'inference')
import anima
import sdxl
import worker
from runtime import InferenceError, _asset_matches

def generation_error(**overrides):
    payload = {
        'kind': 'sdxl',
        'prompt': 'test',
        'negativePrompt': '',
        'width': 256,
        'height': 256,
        'steps': 1,
        'cfg': 1,
        'seed': 1,
        'modelPath': '',
        'loras': [],
    }
    payload.update(overrides)
    try:
        worker.generate(payload)
    except InferenceError as error:
        return str(error)
    return ''

sdxl_bad_size = generation_error(width=272, height=272)
anima_size_path_error = generation_error(kind='anima', width=272, height=272)
null_prompt = generation_error(prompt=None)
boolean_steps = generation_error(steps=True)
unbounded_cfg_values = [worker._finite_number(value, 'CFG') for value in (12345.678, -50.0)]

class CloseTracker:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True

tracker = CloseTracker()
worker._image_session = tracker
original_generate = worker.generate
original_stdin = sys.stdin
original_stdout = sys.stdout
try:
    def fail_generation(_payload, use_session=False):
        raise RuntimeError('synthetic corrupt CUDA session')
    worker.generate = fail_generation
    sys.stdin = io.StringIO('{}\n')
    sys.stdout = io.StringIO()
    worker.serve()
finally:
    worker.generate = original_generate
    sys.stdin = original_stdin
    sys.stdout = original_stdout

class FakeTokenizer:
    def encode(self, _text):
        return [0] * 77

class PartialContextDenoiser:
    def __init__(self):
        self.cached = None
        self.cleared = False
    def prepare_context(self, context):
        self.cached = context
        raise RuntimeError('synthetic partial context failure')
    def clear_context(self):
        self.cached = None
        self.cleared = True

session = sdxl.SDXLSession.__new__(sdxl.SDXLSession)
session.tokenizer = FakeTokenizer()
session.device = torch.device('cpu')
session.dtype = torch.float32
session.checkpoint = object()
session.loras = []
session.unet_lora_counts = []
session.denoiser = PartialContextDenoiser()
original_encode_prompts = sdxl._encode_prompts
try:
    sdxl._encode_prompts = lambda *_args: (
        torch.zeros((2, 77, 2), dtype=torch.float32),
        torch.zeros((2, 1280), dtype=torch.float32),
        [],
    )
    try:
        session.generate('private prompt', '', 256, 256, 1, 1.0, 'dpmpp_sde_karras', 1)
    except RuntimeError:
        pass
finally:
    sdxl._encode_prompts = original_encode_prompts

schedule = torch.tensor([1.0, 4.0])
continuous_timesteps = sdxl._sigma_to_timestep(torch.tensor([1.0, 2.0, 4.0]), schedule).tolist()

class FakeSDXLSamplerDenoiser:
    def __init__(self):
        self.calls = 0
    def __call__(self, latent, _timestep, _context, _labels):
        self.calls += 1
        return torch.zeros_like(latent)

sdxl_sampler_results = {}
for sampler_name in ('dpmpp_sde_karras', 'euler_ancestral_karras', 'euler_karras', 'dpmpp_2m_karras'):
    outputs = []
    call_counts = []
    for _repeat in range(2):
        fake_sdxl_sampler = FakeSDXLSamplerDenoiser()
        output = sdxl._sample_sdxl(
            fake_sdxl_sampler,
            torch.zeros((2, 1, 1), dtype=torch.float32),
            torch.zeros((2, 1), dtype=torch.float32),
            32,
            32,
            3,
            123.45,
            sampler_name,
            987,
            torch.device('cpu'),
            torch.float32,
        )
        outputs.append(output)
        call_counts.append(fake_sdxl_sampler.calls)
    sdxl_sampler_results[sampler_name] = {
        'finite': bool(torch.isfinite(outputs[0]).all()),
        'deterministic': bool(torch.equal(outputs[0], outputs[1])),
        'calls': call_counts,
    }

class FakeAnimaDenoiser:
    def __init__(self):
        self.calls = 0
    def prepare_context(self, _source, _target):
        self.calls += 1
        return torch.full((1, 512, 1024), 7.0)

class FakeT5Tokenizer:
    def encode(self, _text):
        return [1]

fake_anima = FakeAnimaDenoiser()
anima_context = anima._prepare_anima_context(
    fake_anima,
    torch.ones((1, 2, 1024)),
    torch.ones((1, 2, 1024)),
    FakeT5Tokenizer(),
    'positive prompt',
    '   ',
    torch.device('cpu'),
    torch.float32,
)
anima_blank_negative_zero = bool(
    fake_anima.calls == 1
    and torch.count_nonzero(anima_context[0]).item() == 0
    and torch.all(anima_context[1] == 7).item()
)

class FakeAnimaSamplerDenoiser:
    def __init__(self):
        self.input_dtypes = []
        self.timestep_dtypes = []
    def prepare_denoising(self, _context, _height, _width):
        return None, []
    def __call__(self, latent, timestep, _context, _rope, _context_kv):
        self.input_dtypes.append(str(latent.dtype))
        self.timestep_dtypes.append(str(timestep.dtype))
        return torch.ones_like(latent)

fake_sampler = FakeAnimaSamplerDenoiser()
anima_latent = anima._sample_anima(
    fake_sampler,
    torch.zeros((2, 512, 1024), dtype=torch.bfloat16),
    16,
    16,
    2,
    4.0,
    'flow_euler',
    123,
    torch.device('cpu'),
    torch.bfloat16,
)
anima_sampler_precision = bool(
    anima_latent.dtype == torch.float32
    and fake_sampler.input_dtypes == ['torch.bfloat16', 'torch.bfloat16']
    and fake_sampler.timestep_dtypes == ['torch.bfloat16', 'torch.bfloat16']
)

fake_heun_sampler = FakeAnimaSamplerDenoiser()
anima._sample_anima(
    fake_heun_sampler,
    torch.zeros((2, 512, 1024), dtype=torch.bfloat16),
    16,
    16,
    2,
    4.0,
    'flow_heun',
    123,
    torch.device('cpu'),
    torch.bfloat16,
)
anima_heun_evaluations = len(fake_heun_sampler.input_dtypes)

torch.manual_seed(7)
patch_embed = anima.PatchEmbed()
patch_input = torch.randn((1, 16, 1, 4, 4))
patch_optimized = patch_embed(patch_input)
patch_with_zero_channel = torch.cat((patch_input, torch.zeros_like(patch_input[:, :1])), dim=1)
batch, channels, frames, height, width = patch_with_zero_channel.shape
patch_reference_input = patch_with_zero_channel.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
patch_reference_input = patch_reference_input.permute(0, 2, 3, 5, 1, 4, 6).reshape(
    batch, frames, height // 2, width // 2, channels * 4
)
patch_reference = patch_embed.proj(patch_reference_input)
patch_embed_equivalent = torch.allclose(patch_optimized, patch_reference, atol=1e-6, rtol=1e-6)

sdxl_attention = sdxl.CrossAttention(64, 32)
sdxl_query = torch.randn((2, 3, 64))
sdxl_context = torch.randn((2, 5, 32))
sdxl_uncached = sdxl_attention(sdxl_query, sdxl_context)
sdxl_cached = sdxl_attention(
    sdxl_query,
    sdxl_context,
    sdxl_attention.project_context(sdxl_context),
)

anima_attention = anima.CosmosAttention(128, 32)
anima_query = torch.randn((2, 3, 128))
anima_attention_context = torch.randn((2, 5, 32))
anima_uncached = anima_attention(anima_query, anima_attention_context, None)
anima_cached = anima_attention(
    anima_query,
    anima_attention_context,
    None,
    anima_attention.project_context(anima_attention_context),
)
cached_attention_equivalent = bool(
    torch.allclose(sdxl_uncached, sdxl_cached, atol=1e-6, rtol=1e-6)
    and torch.allclose(anima_uncached, anima_cached, atol=1e-6, rtol=1e-6)
)

with tempfile.TemporaryDirectory() as directory:
    asset = Path(directory) / 'asset.bin'
    asset.write_bytes(b'verified tokenizer bytes')
    correct_hash = hashlib.sha256(asset.read_bytes()).hexdigest()
    tokenizer_hash_validation = (
        _asset_matches(asset, correct_hash)
        and not _asset_matches(asset, '0' * 64)
    )

print(json.dumps({
    'sdxl_bad_size': sdxl_bad_size,
    'anima_size_path_error': anima_size_path_error,
    'null_prompt': null_prompt,
    'boolean_steps': boolean_steps,
    'unbounded_cfg_values': unbounded_cfg_values,
    'generic_session_closed': tracker.closed,
    'partial_context_cleared': session.denoiser.cleared and session.denoiser.cached is None,
    'continuous_timesteps': continuous_timesteps,
    'sdxl_sampler_results': sdxl_sampler_results,
    'tokenizer_hash_validation': tokenizer_hash_validation,
    'anima_blank_negative_zero': anima_blank_negative_zero,
    'anima_sampler_precision': anima_sampler_precision,
    'anima_heun_evaluations': anima_heun_evaluations,
    'patch_embed_equivalent': bool(patch_embed_equivalent),
    'cached_attention_equivalent': cached_attention_equivalent,
}))
`], {
  cwd: ROOT,
  encoding: 'utf8',
  env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', PYTHONIOENCODING: 'utf-8' },
}));

test('worker rejects malformed types and SDXL-incompatible dimensions before model loading', () => {
  assert.match(results.sdxl_bad_size, /divisible by 32/i);
  assert.doesNotMatch(results.anima_size_path_error, /divisible/i);
  assert.match(results.null_prompt, /must be text/i);
  assert.match(results.boolean_steps, /steps is invalid/i);
});

test('worker accepts any finite CFG value without imposing a range', () => {
  assert.deepEqual(results.unbounded_cfg_values, [12345.678, -50]);
});

test('warm worker evicts a session after an unexpected inference exception', () => {
  assert.equal(results.generic_session_closed, true);
});

test('partial SDXL prompt context is cleared when context preparation fails', () => {
  assert.equal(results.partial_context_cleared, true);
});

test('SDXL maps Karras sigmas to continuous log-schedule timesteps', () => {
  assert.deepEqual(results.continuous_timesteps, [0, 0.5, 1]);
});

test('every SDXL sampler is finite and seed-deterministic, with SDE using its two-stage evaluations', () => {
  for (const [sampler, result] of Object.entries(results.sdxl_sampler_results)) {
    assert.equal(result.finite, true, `${sampler} produced a non-finite latent`);
    assert.equal(result.deterministic, true, `${sampler} ignored the supplied seed`);
    assert.deepEqual(result.calls, sampler === 'dpmpp_sde_karras' ? [5, 5] : [3, 3]);
  }
});

test('runtime readiness rejects corrupted tokenizer assets by checksum', () => {
  assert.equal(results.tokenizer_hash_validation, true);
});

test('Anima blank negative prompts use canonical zero unconditional conditioning', () => {
  assert.equal(results.anima_blank_negative_zero, true);
});

test('Anima keeps scheduler state in float32 while denoising in model precision', () => {
  assert.equal(results.anima_sampler_precision, true);
});

test('Anima Flow Heun uses a correction evaluation without cross-step history', () => {
  assert.equal(results.anima_heun_evaluations, 4);
});

test('optimized Anima patch embedding matches the explicit zero-channel reference', () => {
  assert.equal(results.patch_embed_equivalent, true);
});

test('cached Anima and SDXL context projections match uncached attention', () => {
  assert.equal(results.cached_attention_equivalent, true);
});
