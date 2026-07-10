'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const https = require('node:https');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const TEST_PORT = 33731;
const HTTPS_TEST_PORT = 33732;
const DNS_TEST_PORT = 33733;
const DIAGNOSTICS_TEST_PORT = 33734;
const BASE_URL = `http://127.0.0.1:${TEST_PORT}`;
const TEST_TOKEN = 'test-access-token';
const TEST_IMAGE = Buffer.from('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415408d763f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082', 'hex');

let serverProcess;
let serverOutput = '';
let imageRoot;
let fakeWorkerPath;

test.before(async () => {
  imageRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'local-chat-image-test-'));
  const modelsRoot = path.join(imageRoot, 'models');
  fs.mkdirSync(path.join(modelsRoot, 'diffusion_models'), { recursive: true });
  fs.mkdirSync(path.join(modelsRoot, 'checkpoints'), { recursive: true });
  fs.mkdirSync(path.join(modelsRoot, 'text_encoders'), { recursive: true });
  fs.mkdirSync(path.join(modelsRoot, 'vae'), { recursive: true });
  fs.mkdirSync(path.join(modelsRoot, 'loras'), { recursive: true });
  writeFakeAnimaSafetensors(path.join(modelsRoot, 'diffusion_models', 'anima-test.safetensors'), 'net.');
  writeFakeAnimaSafetensors(
    path.join(modelsRoot, 'diffusion_models', 'community-model.safetensors'),
    'model.diffusion_model.',
  );
  writeFakeSafetensorsFile(path.join(modelsRoot, 'diffusion_models', 'anima-lora.safetensors'), {
    'lora_unet_blocks_0_self_attn_q_proj.lora_down.weight': fakeTensor([8, 2048]),
  });
  writeFakeSafetensorsFile(path.join(modelsRoot, 'diffusion_models', 'anima-wrong-shape.safetensors'), {
    'net.llm_adapter.embed.weight': fakeTensor([32128, 1024]),
    'net.x_embedder.proj.1.weight': fakeTensor([2048, 64]),
    'net.blocks.0.self_attn.q_proj.weight': fakeTensor([2048, 2048]),
    'net.final_layer.linear.weight': fakeTensor([64, 2048]),
  });
  writeFakeSafetensors(path.join(modelsRoot, 'checkpoints', 'sdxl-test.safetensors'), 'stable-diffusion-xl-v1-base');
  writeFakeSafetensors(path.join(modelsRoot, 'checkpoints', 'not-an-image-model.safetensors'), 'ace-step-audio');
  const textEncoderPath = path.join(modelsRoot, 'text_encoders', 'encoder-test.safetensors');
  const vaePath = path.join(modelsRoot, 'vae', 'vae-test.safetensors');
  fs.writeFileSync(textEncoderPath, 'test');
  fs.writeFileSync(vaePath, 'test');
  writeFakeLora(path.join(modelsRoot, 'loras', 'anima-style.safetensors'), 'lora_unet_blocks_0_cross_attn_k_proj');
  writeFakeLora(path.join(modelsRoot, 'loras', 'sdxl-style.safetensors'), 'lora_unet_input_blocks_4_1_proj_in');
  writeFakeLora(path.join(modelsRoot, 'loras', 'unsupported-diffusers.safetensors'), 'lora_unet_down_blocks_0_attentions_0_proj_in');
  fakeWorkerPath = path.join(imageRoot, 'fake-image-worker.js');
  fs.writeFileSync(fakeWorkerPath, makeFakeImageWorkerSource());

  serverProcess = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      LOCAL_CONFIG_DISABLED: '1',
      HOST: '127.0.0.1',
      PORT: String(TEST_PORT),
      HTTPS: '0',
      ALLOW_LOCAL_BYPASS: '0',
      ACCESS_TOKEN: TEST_TOKEN,
      TEXT_BASE_URL: 'http://127.0.0.1:65535/v1',
      IMAGE_MODELS_ROOT: modelsRoot,
      IMAGE_PYTHON: process.execPath,
      ANIMA_TEXT_ENCODER_PATH: textEncoderPath,
      ANIMA_VAE_PATH: vaePath,
      IMAGE_WORKER_PATH: fakeWorkerPath,
      IMAGE_WORKER_PERSISTENT: '0',
      IMAGE_PROFILE: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  serverProcess.stdout.on('data', (chunk) => {
    serverOutput += chunk.toString('utf8');
  });
  serverProcess.stderr.on('data', (chunk) => {
    serverOutput += chunk.toString('utf8');
  });

  await waitForServer(BASE_URL);
});

test.after(async () => {
  if (serverProcess && !serverProcess.killed) {
    serverProcess.kill();
  }
  if (imageRoot) fs.rmSync(imageRoot, { recursive: true, force: true });
});

test('serves app shell and static assets', async () => {
  const shellResponse = await fetch(`${BASE_URL}/`);
  assert.equal(shellResponse.status, 200);
  assert.equal(shellResponse.headers.get('referrer-policy'), 'no-referrer');
  assert.equal(shellResponse.headers.get('x-frame-options'), 'DENY');
  assert.match(shellResponse.headers.get('content-security-policy'), /default-src 'self'/);
  assert.match(shellResponse.headers.get('content-security-policy'), /img-src 'self' data: blob:/);
  const html = await shellResponse.text();
  assert.match(html, /<title>Local Chat<\/title>/);
  assert.match(html, /id="promptInput"/);
  assert.match(html, /id="imageDialog"/);

  const cssResponse = await fetch(`${BASE_URL}/styles.css`);
  assert.equal(cssResponse.status, 200);
  assert.equal(cssResponse.headers.get('cache-control'), 'no-cache');
  const css = await cssResponse.text();
  assert.match(css, /--accent:/);

  const app = await fetchText('/app.js');
  assert.match(app, /function openSidebar\(\)/);
  assert.match(app, /function generateImage\(\)/);
});

test('discovers compatible local Anima and SDXL models', async () => {
  const response = await fetch(`${BASE_URL}/api/image/config`, { headers: accessHeaders() });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const body = await response.json();
  assert.equal(body.connected, true);
  assert.deepEqual(body.models.anima.map((model) => model.id), [
    'anima-test.safetensors',
    'community-model.safetensors',
  ]);
  assert.deepEqual(body.models.sdxl.map((model) => model.id), ['sdxl-test.safetensors']);
  assert.deepEqual(body.loras.anima.map((lora) => lora.id), ['anima-style.safetensors']);
  assert.deepEqual(body.loras.sdxl.map((lora) => lora.id), ['sdxl-style.safetensors']);
  assert.equal(body.dependencies.animaTextEncoder, true);
  assert.equal(body.dependencies.animaVae, true);
  assert.equal(body.dependencies.tokenizerAssets, true);
  assert.equal(body.runtime.engine, 'Direct test CUDA');
  assert.equal(body.runtime.gpu, 'Test GPU');
});

test('generates through the direct worker and returns only in-memory image bytes', async () => {
  const outputOffset = serverOutput.length;
  const response = await fetch(`${BASE_URL}/api/image/generate`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      kind: 'anima',
      model: 'anima-test.safetensors',
      prompt: 'private test prompt',
      size: 'square',
      seed: 42,
      loras: [{ id: 'anima-style.safetensors', strength: 0.75 }],
    }),
  });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.imageBase64, TEST_IMAGE.toString('base64'));
  assert.equal(body.seed, 42);
  assert.equal(body.model, 'anima-test.safetensors');
  assert.equal(body.mimeType, 'image/png');
  assert.equal(body.sampler, 'flow_euler');
  assert.deepEqual(body.loras, [{ id: 'anima-style.safetensors', strength: 0.75 }]);
  await new Promise((resolve) => setTimeout(resolve, 30));
  const diagnostics = serverOutput.slice(outputOffset);
  assert.match(diagnostics, /IMAGE_PROFILE {"pipeline":"anima-warm","stagesSeconds":{"sampling":1\.25},"totalSeconds":1\.5,"peakVramMiB":1234}/);
  assert.doesNotMatch(diagnostics, /UNTRUSTED_WORKER_DIAGNOSTIC|private-marker/);
});

test('rejects a LoRA from the wrong model family before inference', async () => {
  const response = await fetch(`${BASE_URL}/api/image/generate`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      kind: 'anima',
      model: 'anima-test.safetensors',
      prompt: 'wrong LoRA family test',
      size: 'square',
      seed: 45,
      loras: [{ id: 'sdxl-style.safetensors', strength: 1 }],
    }),
  });
  assert.equal(response.status, 400);
  const body = await response.json();
  assert.match(body.error, /not available for this model family/);
});

test('accepts the full-checkpoint Anima tensor prefix used by community models', async () => {
  const response = await fetch(`${BASE_URL}/api/image/generate`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      kind: 'anima',
      model: 'community-model.safetensors',
      prompt: 'community layout test',
      size: 'square',
      seed: 44,
      cfg: 123.45,
      sampler: 'flow_heun',
    }),
  });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.model, 'community-model.safetensors');
  assert.equal(body.cfg, 123.45);
  assert.equal(body.sampler, 'flow_heun');
});

test('rejects samplers from the wrong image-model family', async () => {
  const response = await fetch(`${BASE_URL}/api/image/generate`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      kind: 'anima',
      model: 'anima-test.safetensors',
      prompt: 'wrong sampler family test',
      size: 'square',
      sampler: 'dpmpp_sde_karras',
    }),
  });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /sampler is not available/i);
});

test('fails closed when the direct worker reports an inference error', async () => {
  const response = await fetch(`${BASE_URL}/api/image/generate`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      kind: 'anima',
      model: 'anima-test.safetensors',
      prompt: 'force-worker-error',
      size: 'square',
      seed: 43,
    }),
  });
  assert.equal(response.status, 502);
  const body = await response.json();
  assert.equal(body.error, 'Synthetic direct inference failure.');
  assert.equal(body.imageBase64, undefined);
});

test('requires the QR access token for API routes', async () => {
  const denied = await fetch(`${BASE_URL}/api/config`);
  assert.equal(denied.status, 401);
  assert.equal(denied.headers.get('x-local-access-required'), '1');

  const authorized = await fetch(`${BASE_URL}/api/config`, { headers: accessHeaders() });
  assert.equal(authorized.status, 200);
  assert.equal(authorized.headers.get('x-local-access-required'), null);
  assert.equal(authorized.headers.get('x-content-type-options'), 'nosniff');
});

test('does not label an upstream backend 401 as a local access-token failure', async () => {
  const upstream = http.createServer((_req, res) => {
    res.writeHead(401, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Synthetic upstream authorization failure.' }));
  });
  await listenOnRandomLocalPort(upstream);

  try {
    const response = await fetch(`${BASE_URL}/api/models`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ baseUrl: `http://127.0.0.1:${upstream.address().port}/v1` }),
    });
    assert.equal(response.status, 401);
    assert.equal(response.headers.get('x-local-access-required'), null);
    assert.match((await response.json()).error, /upstream authorization failure/i);
  } finally {
    await closeServer(upstream);
  }
});

test('reports local config without caching', async () => {
  const response = await fetch(`${BASE_URL}/api/config`, { headers: accessHeaders() });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('cache-control'), 'no-store');

  const body = await response.json();
  assert.equal(body.port, TEST_PORT);
  assert.equal(body.defaultBaseUrl, 'http://127.0.0.1:65535/v1');
  assert.equal(body.managedTextBackend.enabled, false);
  assert.ok(Array.isArray(body.lanUrls));
});

test('text load status is authenticated, path-free, and external backends do not fake progress', async () => {
  const denied = await fetch(`${BASE_URL}/api/text/status`);
  assert.equal(denied.status, 401);

  const statusResponse = await fetch(`${BASE_URL}/api/text/status`, { headers: accessHeaders() });
  assert.equal(statusResponse.status, 200);
  assert.equal(statusResponse.headers.get('cache-control'), 'no-store');
  assert.deepEqual(await statusResponse.json(), {
    managed: false,
    state: 'external',
    phase: 'idle',
  });

  const loadResponse = await fetch(`${BASE_URL}/api/text/load`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      baseUrl: 'http://127.0.0.1:65535/v1',
      model: 'synthetic-external-model',
    }),
  });
  assert.equal(loadResponse.status, 200);
  assert.deepEqual(await loadResponse.json(), {
    managed: false,
    state: 'external',
    phase: 'idle',
  });
});

test('locks host setup mutations when local configuration is disabled or environment-controlled', async () => {
  const statusResponse = await fetch(`${BASE_URL}/api/local-setup`);
  assert.equal(statusResponse.status, 200);
  assert.equal(statusResponse.headers.get('cache-control'), 'no-store');
  const status = await statusResponse.json();
  assert.equal(status.text.folderLockedByEnvironment, true);

  const mutationHeaders = {
    'Content-Type': 'application/json',
    Origin: BASE_URL,
    'X-Local-Setup': '1',
  };
  const saveResponse = await fetch(`${BASE_URL}/api/local-setup/text-folder`, {
    method: 'POST',
    headers: mutationHeaders,
    body: JSON.stringify({ path: imageRoot }),
  });
  assert.equal(saveResponse.status, 409);

  const pickerResponse = await fetch(`${BASE_URL}/api/local-setup/pick-folder`, {
    method: 'POST',
    headers: { Origin: BASE_URL, 'X-Local-Setup': '1' },
  });
  assert.equal(pickerResponse.status, 409);
});

test('host setup mutations require exact same-origin browser proof', async () => {
  for (const headers of [
    {},
    { Origin: BASE_URL },
    { 'X-Local-Setup': '1' },
    { Origin: 'http://cross-site.invalid', 'X-Local-Setup': '1' },
    { Origin: BASE_URL, 'X-Local-Setup': '1', 'Sec-Fetch-Site': 'cross-site' },
  ]) {
    const response = await fetch(`${BASE_URL}/api/local-setup/text-folder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify({ path: imageRoot }),
    });
    assert.equal(response.status, 403);
  }
});

test('rejects unsafe text backend URLs before making a network request', async () => {
  const response = await fetch(`${BASE_URL}/api/models`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ baseUrl: 'https://example.com/v1' }),
  });
  assert.equal(response.status, 400);
  const body = await response.json();
  assert.match(body.error, /private LAN address/);

  for (const hostname of ['127.attacker.example', '10.0.0.1evil', 'fcevil.example']) {
    const disguisedHost = await fetch(`${BASE_URL}/api/models`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ baseUrl: `http://${hostname}/v1` }),
    });
    assert.equal(disguisedHost.status, 400, `${hostname} must not pass private-address validation`);
  }

  for (const baseUrl of [
    'http://user:password@127.0.0.1:65535/v1',
    'http://user%2Dname:password@127.0.0.1:65535/v1',
    'http://169.254.169.254/v1',
    'http://[fe80::1]/v1',
    'http://[fd00:ec2::254]/v1',
    'http://[fd00:0ec2:0000:0000:0000:0000:0000:0254]/v1',
  ]) {
    const unsafeAddress = await fetch(`${BASE_URL}/api/models`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ baseUrl }),
    });
    assert.equal(unsafeAddress.status, 400);
    const unsafeBody = await unsafeAddress.json();
    assert.match(unsafeBody.error, /link-local and metadata endpoints are blocked/);
  }
});

test('pins DNS-approved private backends and bounds DNS resolution time', { timeout: 12_000 }, async () => {
  let upstreamRequests = 0;
  let upstreamHost = '';
  const upstream = http.createServer((req, res) => {
    upstreamRequests += 1;
    upstreamHost = req.headers.host || '';
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end('{"object":"list","data":[]}');
  });
  await listenOnRandomLocalPort(upstream);

  const dnsMockPath = path.join(imageRoot, 'dns-mock.js');
  fs.writeFileSync(dnsMockPath, makeDnsMockSource());
  const dnsProcess = spawn(process.execPath, ['--require', dnsMockPath, 'server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      LOCAL_CONFIG_DISABLED: '1',
      HOST: '127.0.0.1',
      PORT: String(DNS_TEST_PORT),
      HTTPS: '0',
      ALLOW_LOCAL_BYPASS: '0',
      ACCESS_TOKEN: TEST_TOKEN,
      TEXT_BASE_URL: 'http://127.0.0.1:65535/v1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    const dnsBaseUrl = `http://127.0.0.1:${DNS_TEST_PORT}`;
    await waitForStandaloneServer(dnsProcess, dnsBaseUrl);
    const upstreamPort = upstream.address().port;
    const requestModels = (hostname) => fetch(`${dnsBaseUrl}/api/models`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ baseUrl: `http://${hostname}:${upstreamPort}/v1` }),
    });

    const privateResponse = await requestModels('private-backend.test');
    assert.equal(privateResponse.status, 200);
    assert.equal(upstreamRequests, 1);
    assert.equal(upstreamHost, `private-backend.test:${upstreamPort}`);

    const mixedResponse = await requestModels('mixed-backend.test');
    assert.equal(mixedResponse.status, 400);
    assert.equal(upstreamRequests, 1);

    const localPublicResponse = await requestModels('untrusted-backend.local');
    assert.equal(localPublicResponse.status, 400);
    assert.equal(upstreamRequests, 1);

    const slowStartedAt = Date.now();
    const slowResponse = await requestModels('slow-backend.test');
    assert.equal(slowResponse.status, 504);
    assert.ok(Date.now() - slowStartedAt < 7_000);
    assert.equal(upstreamRequests, 1);
  } finally {
    dnsProcess.kill();
    await closeServer(upstream);
  }
});

test('refuses upstream redirects without contacting the redirect destination', async () => {
  let destinationRequests = 0;
  const destination = http.createServer((req, res) => {
    destinationRequests += 1;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end('{"object":"list","data":[]}');
  });
  const redirector = http.createServer((req, res) => {
    res.writeHead(307, {
      Location: `http://127.0.0.1:${destination.address().port}/v1/models`,
    });
    res.end();
  });
  await listenOnRandomLocalPort(destination);
  await listenOnRandomLocalPort(redirector);

  try {
    const response = await fetch(`${BASE_URL}/api/models`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ baseUrl: `http://127.0.0.1:${redirector.address().port}/v1` }),
    });
    assert.equal(response.status, 502);
    assert.equal(destinationRequests, 0);
  } finally {
    await closeServer(redirector);
    await closeServer(destination);
  }
});

test('returns useful validation errors for malformed chat requests', async () => {
  const invalidJson = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: '{',
  });
  assert.equal(invalidJson.status, 400);

  const missingModel = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      baseUrl: 'http://127.0.0.1:65535/v1',
      messages: [{ role: 'user', content: 'hello' }],
    }),
  });
  assert.equal(missingModel.status, 400);
  const body = await missingModel.json();
  assert.match(body.error, /Choose a model/);

  for (const malformedBody of ['null', '[]', '"text"']) {
    const invalidShape = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: malformedBody,
    });
    assert.equal(invalidShape.status, 400);
  }

  const nullMessage = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      baseUrl: 'http://127.0.0.1:65535/v1',
      model: 'local-test-model',
      messages: [null],
    }),
  });
  assert.equal(nullMessage.status, 400);
});

test('does not echo or log conversation content when upstream is unavailable', async () => {
  const secretPrompt = `private-prompt-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const response = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      baseUrl: 'http://127.0.0.1:65535/v1',
      model: 'local-test-model',
      messages: [{ role: 'user', content: secretPrompt }],
    }),
  });

  assert.equal(response.status, 502);
  const bodyText = await response.text();
  assert.equal(bodyText.includes(secretPrompt), false);

  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(serverOutput.includes(secretPrompt), false);
});

test('chat proxy strips previous assistant thinking before forwarding context', async () => {
  let upstreamRequestBody = '';
  const upstream = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      upstreamRequestBody = Buffer.concat(chunks).toString('utf8');
      res.writeHead(200, {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-store',
      });
      res.end('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n');
    });
  });

  await listenOnRandomLocalPort(upstream);

  try {
    const upstreamPort = upstream.address().port;
    const response = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        baseUrl: `http://127.0.0.1:${upstreamPort}/v1`,
        model: 'local-test-model',
        messages: [
          { role: 'system', content: 'System keeps literal <think>syntax</think>.' },
          { role: 'user', content: 'User keeps literal <think>syntax</think>.' },
          { role: 'assistant', content: '<think>secret plan</think>Public answer<think>second secret</think> tail' },
          { role: 'assistant', content: '<THINK>upper secret</THINK>Mixed case done' },
          { role: 'assistant', content: '<think>all secret</think>' },
          { role: 'assistant', content: 'Visible before <think>unfinished secret' },
          { role: 'assistant', content: 'Partial <thi tag remains visible' },
          { role: 'user', content: 'Next question' },
        ],
      }),
    });

    assert.equal(response.status, 200);
    await response.text();

    const forwarded = JSON.parse(upstreamRequestBody);
    assert.deepEqual(forwarded.messages, [
      { role: 'system', content: 'System keeps literal <think>syntax</think>.' },
      { role: 'user', content: 'User keeps literal <think>syntax</think>.' },
      { role: 'assistant', content: 'Public answer tail' },
      { role: 'assistant', content: 'Mixed case done' },
      { role: 'assistant', content: 'Visible before ' },
      { role: 'assistant', content: 'Partial <thi tag remains visible' },
      { role: 'user', content: 'Next question' },
    ]);
    assert.equal(upstreamRequestBody.includes('secret plan'), false);
    assert.equal(upstreamRequestBody.includes('second secret'), false);
    assert.equal(upstreamRequestBody.includes('upper secret'), false);
    assert.equal(upstreamRequestBody.includes('all secret'), false);
    assert.equal(upstreamRequestBody.includes('unfinished secret'), false);
  } finally {
    await closeServer(upstream);
  }
});

test('chat proxy applies selected thinking mode to the latest user message', async () => {
  const upstreamRequestBodies = [];
  const upstream = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      upstreamRequestBodies.push(Buffer.concat(chunks).toString('utf8'));
      res.writeHead(200, {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-store',
      });
      res.end('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n');
    });
  });

  await listenOnRandomLocalPort(upstream);

  try {
    const upstreamPort = upstream.address().port;
    const basePayload = {
      baseUrl: `http://127.0.0.1:${upstreamPort}/v1`,
      model: 'local-test-model',
      messages: [
        { role: 'user', content: 'Earlier question' },
        { role: 'assistant', content: 'Earlier answer' },
        { role: 'user', content: '/think\nCurrent question' },
      ],
    };

    const offResponse = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ ...basePayload, thinkingMode: 'off' }),
    });
    assert.equal(offResponse.status, 200);
    await offResponse.text();

    const onResponse = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ ...basePayload, thinkingMode: 'on' }),
    });
    assert.equal(onResponse.status, 200);
    await onResponse.text();

    const offForwarded = JSON.parse(upstreamRequestBodies[0]);
    const onForwarded = JSON.parse(upstreamRequestBodies[1]);
    assert.equal(offForwarded.messages[0].content, 'Earlier question');
    assert.equal(offForwarded.messages[2].content, '/no_think\nCurrent question');
    assert.equal(onForwarded.messages[2].content, '/think\nCurrent question');
  } finally {
    await closeServer(upstream);
  }
});

test('chat proxy forwards tool definitions, tool calls, and tool results', async () => {
  let upstreamRequestBody = '';
  const upstream = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      upstreamRequestBody = Buffer.concat(chunks).toString('utf8');
      res.writeHead(200, {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-store',
      });
      res.end('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n');
    });
  });

  await listenOnRandomLocalPort(upstream);

  try {
    const upstreamPort = upstream.address().port;
    const response = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        baseUrl: `http://127.0.0.1:${upstreamPort}/v1`,
        model: 'local-test-model',
        tools: [{
          type: 'function',
          function: {
            name: 'generate_image',
            description: 'Generate a picture.',
            parameters: { type: 'object', properties: { prompt: { type: 'string' } }, required: ['prompt'] },
          },
        }],
        messages: [
          { role: 'user', content: 'Draw a cat' },
          {
            role: 'assistant',
            content: '<think>tool secret</think>Making it now',
            tool_calls: [{
              id: 'call-1',
              type: 'function',
              function: { name: 'generate_image', arguments: '{"prompt":"a cat"}' },
            }],
          },
          { role: 'tool', tool_call_id: 'call-1', content: '{"ok":true}' },
        ],
      }),
    });

    assert.equal(response.status, 200);
    await response.text();

    const forwarded = JSON.parse(upstreamRequestBody);
    assert.equal(forwarded.tools.length, 1);
    assert.equal(forwarded.tools[0].function.name, 'generate_image');
    assert.equal(forwarded.tools[0].function.parameters.required[0], 'prompt');
    assert.equal(forwarded.tool_choice, 'auto');
    assert.deepEqual(forwarded.messages, [
      { role: 'user', content: 'Draw a cat' },
      {
        role: 'assistant',
        content: 'Making it now',
        tool_calls: [{
          id: 'call-1',
          type: 'function',
          function: { name: 'generate_image', arguments: '{"prompt":"a cat"}' },
        }],
      },
      { role: 'tool', content: '{"ok":true}', tool_call_id: 'call-1' },
    ]);
    assert.equal(upstreamRequestBody.includes('tool secret'), false);
  } finally {
    await closeServer(upstream);
  }
});

test('does not abort a healthy upstream request after the browser request body closes', async () => {
  const upstream = http.createServer((req, res) => {
    req.resume();
    req.on('end', () => {
      setTimeout(() => {
        res.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8' });
        res.end('data: {"choices":[{"delta":{"content":"delayed-ok"}}]}\n\ndata: [DONE]\n\n');
      }, 60);
    });
  });
  await listenOnRandomLocalPort(upstream);

  try {
    const response = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        baseUrl: `http://127.0.0.1:${upstream.address().port}/v1`,
        model: 'local-test-model',
        messages: [{ role: 'user', content: 'hello' }],
      }),
    });
    assert.equal(response.status, 200);
    assert.match(await response.text(), /delayed-ok/);
  } finally {
    await closeServer(upstream);
  }
});

test('local dashboard exposes tokenized access URLs without query-string tokens', async () => {
  const response = await fetch(`${BASE_URL}/api/access-info`);
  assert.equal(response.status, 200);

  const body = await response.json();
  assert.ok(body.accessUrls.length > 0);
  assert.match(body.accessUrls[0], /#access=/);
  assert.doesNotMatch(body.accessUrls[0], /\?access=/);
  assert.equal(body.tokenLength, TEST_TOKEN.length);
});

test('can serve the app over generated local HTTPS', async () => {
  const previousTlsSetting = process.env.NODE_TLS_REJECT_UNAUTHORIZED;
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
  const httpsProcess = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      LOCAL_CONFIG_DISABLED: '1',
      HOST: '127.0.0.1',
      PORT: String(HTTPS_TEST_PORT),
      HTTPS: '1',
      ALLOW_LOCAL_BYPASS: '0',
      ACCESS_TOKEN: TEST_TOKEN,
      TEXT_BASE_URL: 'http://127.0.0.1:65535/v1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    const baseUrl = `https://127.0.0.1:${HTTPS_TEST_PORT}`;
    await waitForStandaloneServer(httpsProcess, baseUrl);
    const response = await fetch(`${baseUrl}/api/config`, { headers: accessHeaders() });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.httpsEnabled, true);

    const tls = await inspectTlsConnection(`${baseUrl}/api/config`, accessHeaders());
    assert.equal(tls.statusCode, 200);
    assert.equal(tls.encrypted, true);
    assert.match(tls.protocol, /^TLSv1\.[23]$/);
    assert.ok(tls.cipher && tls.cipher.name);
    const subjectAltNames = tls.subjectAltName.split(', ');
    assert.equal(subjectAltNames.length, 3);
    assert.ok(subjectAltNames.includes('DNS:localhost'));
    assert.ok(subjectAltNames.includes('IP Address:127.0.0.1'));
    assert.ok(subjectAltNames.some((name) => /^IP Address:(?:::1|0:0:0:0:0:0:0:1)$/.test(name)));
  } finally {
    httpsProcess.kill();
    if (previousTlsSetting === undefined) {
      delete process.env.NODE_TLS_REJECT_UNAUTHORIZED;
    } else {
      process.env.NODE_TLS_REJECT_UNAUTHORIZED = previousTlsSetting;
    }
  }
});

test('logs unexpected request failures generically even when diagnostic flags are supplied', async () => {
  const outputOffset = serverOutput.length;
  const response = await requestWithHostHeader(TEST_PORT, '[');
  assert.equal(response.statusCode, 500);
  assert.deepEqual(JSON.parse(response.body), { error: 'Internal server error.' });
  await new Promise((resolve) => setTimeout(resolve, 30));

  const newOutput = serverOutput.slice(outputOffset);
  assert.equal(/Unexpected server error/.test(newOutput), true);
  assert.equal(/TypeError|ERR_INVALID_URL|server\.js:\d+|\\Users\\|\/home\//.test(newOutput), false);

  let diagnosticsOutput = '';
  const diagnosticsProcess = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      LOCAL_CONFIG_DISABLED: '1',
      PRIVATE_DIAGNOSTICS: '1',
      HOST: '127.0.0.1',
      PORT: String(DIAGNOSTICS_TEST_PORT),
      HTTPS: '0',
      ALLOW_LOCAL_BYPASS: '0',
      ACCESS_TOKEN: TEST_TOKEN,
      TEXT_BASE_URL: 'http://127.0.0.1:65535/v1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  diagnosticsProcess.stdout.on('data', (chunk) => {
    diagnosticsOutput += chunk.toString('utf8');
  });
  diagnosticsProcess.stderr.on('data', (chunk) => {
    diagnosticsOutput += chunk.toString('utf8');
  });

  try {
    const diagnosticsBaseUrl = `http://127.0.0.1:${DIAGNOSTICS_TEST_PORT}`;
    await waitForStandaloneServer(diagnosticsProcess, diagnosticsBaseUrl);
    const diagnosticsResponse = await requestWithHostHeader(DIAGNOSTICS_TEST_PORT, '[');
    assert.equal(diagnosticsResponse.statusCode, 500);
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.equal(/Unexpected server error/.test(diagnosticsOutput), true);
    assert.equal(/TypeError|ERR_INVALID_URL|server\.js:\d+|\\Users\\|\/home\//.test(diagnosticsOutput), false);
  } finally {
    diagnosticsProcess.kill();
  }
});

test('server source does not contain filesystem persistence or request-body logging paths', () => {
  const source = require('node:fs').readFileSync(require('node:path').join(ROOT, 'server.js'), 'utf8');
  assert.equal(/\b(writeFile|appendFile|createWriteStream|mkdir|rm|rename)\b/.test(source), false);
  assert.equal(/console\.(log|error|warn)\([^)]*(payload|messages|content|bodyText|apiKey)/.test(source), false);
  assert.doesNotMatch(source, /PRIVATE_DIAGNOSTICS/);
});

test('server uses the same default temperature as the browser UI', () => {
  const source = require('node:fs').readFileSync(require('node:path').join(ROOT, 'server.js'), 'utf8');
  assert.match(source, /normalizeNumber\(payload\.temperature,\s*0,\s*2,\s*0\.95\)/);
});

async function fetchText(pathname) {
  const response = await fetch(`${BASE_URL}${pathname}`);
  assert.equal(response.status, 200);
  return response.text();
}

function accessHeaders(headers = {}) {
  return {
    ...headers,
    'X-Access-Token': TEST_TOKEN,
  };
}

function inspectTlsConnection(url, headers = {}) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, { rejectUnauthorized: false, headers }, (res) => {
      const encrypted = res.socket.encrypted;
      const protocol = res.socket.getProtocol();
      const cipher = res.socket.getCipher();
      const subjectAltName = res.socket.getPeerCertificate().subjectaltname;

      res.resume();
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          encrypted,
          protocol,
          cipher,
          subjectAltName,
        });
      });
    });

    req.on('error', reject);
    req.end();
  });
}

function requestWithHostHeader(port, host) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1',
      port,
      path: '/',
      headers: { Host: host },
    }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve({
        statusCode: res.statusCode,
        body: Buffer.concat(chunks).toString('utf8'),
      }));
    });
    req.once('error', reject);
    req.end();
  });
}

function makeDnsMockSource() {
  return `'use strict';
const promises = require('node:dns').promises;
const originalLookup = promises.lookup.bind(promises);
const lookups = new Map();
promises.lookup = async (hostname, options) => {
  const count = (lookups.get(hostname) || 0) + 1;
  lookups.set(hostname, count);
  if (hostname === 'private-backend.test') {
    if (count > 1) return [{ address: '203.0.113.4', family: 4 }];
    return [{ address: '127.0.0.2', family: 4 }, { address: '127.0.0.1', family: 4 }];
  }
  if (hostname === 'mixed-backend.test') {
    return [{ address: '127.0.0.1', family: 4 }, { address: '203.0.113.2', family: 4 }];
  }
  if (hostname === 'untrusted-backend.local') {
    return [{ address: '203.0.113.3', family: 4 }];
  }
  if (hostname === 'slow-backend.test') return new Promise(() => {});
  return originalLookup(hostname, options);
};
`;
}

function writeFakeSafetensors(filePath, architecture) {
  const contents = {
    __metadata__: { 'modelspec.architecture': architecture },
  };
  if (architecture === 'stable-diffusion-xl-v1-base') {
    Object.assign(contents, {
      'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight': fakeTensor([49408, 768]),
      'conditioner.embedders.1.model.token_embedding.weight': fakeTensor([49408, 1280]),
      'model.diffusion_model.input_blocks.0.0.weight': fakeTensor([320, 4, 3, 3]),
      'model.diffusion_model.out.2.weight': fakeTensor([4, 320, 3, 3]),
      'first_stage_model.decoder.conv_out.weight': fakeTensor([3, 128, 3, 3]),
    });
  } else {
    contents['model.audio_model.input.weight'] = fakeTensor([1]);
  }
  writeFakeSafetensorsFile(filePath, contents);
}

function writeFakeAnimaSafetensors(filePath, prefix) {
  writeFakeSafetensorsFile(filePath, {
    [`${prefix}llm_adapter.embed.weight`]: fakeTensor([32128, 1024]),
    [`${prefix}x_embedder.proj.1.weight`]: fakeTensor([2048, 68]),
    [`${prefix}blocks.0.self_attn.q_proj.weight`]: fakeTensor([2048, 2048]),
    [`${prefix}final_layer.linear.weight`]: fakeTensor([64, 2048]),
  });
}

function writeFakeLora(filePath, base) {
  writeFakeSafetensorsFile(filePath, {
    [`${base}.alpha`]: { dtype: 'F32', shape: [], data_offsets: [0, 4] },
    [`${base}.lora_down.weight`]: fakeTensor([4, 2048]),
    [`${base}.lora_up.weight`]: fakeTensor([2048, 4]),
  });
}

function fakeTensor(shape) {
  return { dtype: 'F16', shape, data_offsets: [0, 2] };
}

function writeFakeSafetensorsFile(filePath, contents) {
  const header = Buffer.from(JSON.stringify(contents));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(filePath, Buffer.concat([length, header, Buffer.alloc(2)]));
}

function makeFakeImageWorkerSource() {
  return `'use strict';
const image = '${TEST_IMAGE.toString('base64')}';
if (process.argv.includes('--probe')) {
  process.stdout.write(JSON.stringify({
    ok: true,
    engine: 'Direct test CUDA',
    gpu: 'Test GPU',
    cuda: 'test',
    torch: 'test',
    cudaAvailable: true,
    tokenizerAssets: true
  }));
} else {
  const chunks = [];
  process.stdin.on('data', (chunk) => chunks.push(chunk));
  process.stdin.on('end', () => {
    const payload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    process.stderr.write('UNTRUSTED_WORKER_DIAGNOSTIC\\n');
    process.stderr.write('IMAGE_PROFILE {"pipeline":"anima-warm","stagesSeconds":{"sampling":"private-marker"},"totalSeconds":1.5,"peakVramMiB":1234}\\n');
    process.stderr.write('IMAGE_PROFILE {"pipeline":"anima-warm","stagesSeconds":{"sampling":1.25},"totalSeconds":1.5,"peakVramMiB":1234}\\n');
    if (payload.prompt === 'force-worker-error') {
      process.stdout.write(JSON.stringify({ ok: false, error: 'Synthetic direct inference failure.' }));
      process.exitCode = 1;
      return;
    }
    const validLoras = Array.isArray(payload.loras) && payload.loras.every((lora) =>
      lora.path.endsWith('.safetensors') && Number.isFinite(lora.strength));
    const valid = payload.modelPath.endsWith('.safetensors') && validLoras &&
      (payload.kind !== 'anima' || payload.textEncoderPath && payload.vaePath);
    process.stdout.write(JSON.stringify(valid
      ? { ok: true, imageBase64: image, mimeType: 'image/png' }
      : { ok: false, error: 'Invalid direct worker payload.' }));
  });
}
`;
}

async function waitForServer(url) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 5000) {
    if (serverProcess.exitCode !== null) {
      throw new Error(`Server exited early with code ${serverProcess.exitCode}`);
    }

    try {
      const response = await fetch(`${url}/api/config`, { headers: accessHeaders() });
      if (response.ok) return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  throw new Error('Timed out waiting for test server');
}

function listenOnRandomLocalPort(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject);
      resolve();
    });
  });
}

function closeServer(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

async function waitForStandaloneServer(processHandle, url) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 5000) {
    if (processHandle.exitCode !== null) {
      throw new Error(`HTTPS server exited early with code ${processHandle.exitCode}`);
    }

    try {
      const response = await fetch(`${url}/api/config`, { headers: accessHeaders() });
      if (response.ok) return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  throw new Error('Timed out waiting for HTTPS test server');
}
