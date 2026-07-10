'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const ACCESS_TOKEN = 'lifecycle-regression-token';
const TEST_IMAGE_BASE64 = Buffer.from(
  '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489' +
  '0000000d4944415408d763f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082',
  'hex',
).toString('base64');

let appProcess;
let appOutput = '';
let appBaseUrl;
let fixtureRoot;
let upstream;
let upstreamBaseUrl;
let imageEventFile;
const openChatResponses = new Set();
const stalledModelResponses = new Set();
let redirectTargetHits = 0;

test.before(async () => {
  fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'local-serving-lifecycle-'));
  const modelsRoot = path.join(fixtureRoot, 'models');
  const workerPath = path.join(fixtureRoot, 'fake-image-worker.js');
  imageEventFile = path.join(fixtureRoot, 'image-events.txt');

  fs.mkdirSync(path.join(modelsRoot, 'checkpoints'), { recursive: true });
  writeFakeSdxlCheckpoint(path.join(modelsRoot, 'checkpoints', 'sdxl-lifecycle-test.safetensors'));
  fs.writeFileSync(workerPath, makeFakeImageWorkerSource());

  upstream = http.createServer(handleUpstreamRequest);
  await listenOnRandomLocalPort(upstream);
  upstreamBaseUrl = `http://127.0.0.1:${upstream.address().port}/v1`;

  const appPort = await findAvailableLocalPort();
  appBaseUrl = `http://127.0.0.1:${appPort}`;
  appProcess = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      LOCAL_CONFIG_DISABLED: '1',
      HOST: '127.0.0.1',
      PORT: String(appPort),
      HTTPS: '0',
      ACCESS_TOKEN,
      TEXT_BASE_URL: upstreamBaseUrl,
      IMAGE_MODELS_ROOT: modelsRoot,
      IMAGE_PYTHON: process.execPath,
      IMAGE_WORKER_PATH: workerPath,
      IMAGE_WORKER_PERSISTENT: '0',
      FAKE_IMAGE_EVENT_FILE: imageEventFile,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  appProcess.stdout.on('data', (chunk) => {
    appOutput = (appOutput + chunk.toString('utf8')).slice(-16_000);
  });
  appProcess.stderr.on('data', (chunk) => {
    appOutput = (appOutput + chunk.toString('utf8')).slice(-16_000);
  });

  await waitForApp();
});

test.after(async () => {
  finishOpenChatResponses();
  for (const response of stalledModelResponses) response.destroy();
  stalledModelResponses.clear();
  await stopChild(appProcess);
  await closeServer(upstream);
  if (fixtureRoot) fs.rmSync(fixtureRoot, { recursive: true, force: true });
});

test('does not start image inference while a text stream is active', { timeout: 10_000 }, async () => {
  const chatResponse = await fetch(`${appBaseUrl}/api/chat`, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      model: 'shared-gpu-test-model',
      messages: [{ role: 'user', content: 'Keep this stream open while image generation is attempted.' }],
    }),
  });
  assert.equal(chatResponse.status, 200);
  const chatBodyPromise = chatResponse.text();

  let imageResponse;
  let imageBody;
  try {
    imageResponse = await fetch(`${appBaseUrl}/api/image/generate`, {
      method: 'POST',
      headers: apiHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        kind: 'sdxl',
        model: 'sdxl-lifecycle-test.safetensors',
        prompt: 'GPU mutual exclusion regression',
        size: 'square',
        seed: 7,
      }),
    });
    imageBody = await imageResponse.text();
  } finally {
    finishOpenChatResponses();
    await withDeadline(chatBodyPromise, 2_000, 'The held chat stream did not close cleanly.');
  }

  const imageEvents = fs.existsSync(imageEventFile) ? fs.readFileSync(imageEventFile, 'utf8') : '';
  assert.equal(
    imageResponse.status,
    409,
    `Image inference overlapped an active text stream (HTTP ${imageResponse.status}: ${imageBody}).`,
  );
  assert.equal(imageEvents, '', 'The image worker must not receive a generation job while text owns the GPU.');
});

test('default loopback bypass rejects cross-site browser requests without a token', async () => {
  const response = await fetch(`${appBaseUrl}/api/image/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'text/plain',
      Origin: 'https://attacker.example',
      'Sec-Fetch-Site': 'cross-site',
    },
    body: '{}',
  });

  assert.equal(response.status, 401);
  assert.match(await response.text(), /access token/i);
});

test('loopback bypass rejects a DNS-rebinding Host header', async () => {
  const result = await rawAppRequest('/api/image/generate', {
    'Content-Type': 'text/plain',
    Host: '127.attacker.example',
    Origin: 'http://127.attacker.example',
    'Sec-Fetch-Site': 'same-origin',
  }, '{}');

  assert.equal(result.status, 401);
  assert.match(result.body, /access token/i);
});

test('oversized JSON receives a 413 response instead of a reset connection', async () => {
  const response = await fetch(`${appBaseUrl}/api/chat`, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ padding: 'x'.repeat(2 * 1024 * 1024) }),
  });

  assert.equal(response.status, 413);
  assert.match(await response.text(), /too large/i);
});

test('model proxy timeout remains active while the upstream response body stalls', { timeout: 12_000 }, async () => {
  const controller = new AbortController();
  const startedAt = Date.now();
  let result;

  try {
    result = await Promise.race([
      fetch(`${appBaseUrl}/api/models`, {
        method: 'POST',
        headers: apiHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ baseUrl: upstreamBaseUrl }),
        signal: controller.signal,
      }).then(async (response) => ({
        kind: 'response',
        status: response.status,
        body: await response.text(),
      })),
      delay(7_000).then(() => ({ kind: 'hung' })),
    ]);
  } finally {
    controller.abort();
  }

  const elapsedMs = Date.now() - startedAt;
  assert.notEqual(
    result.kind,
    'hung',
    'The 5-second model deadline was cleared after headers and the stalled body remained open.',
  );
  assert.equal(result.status, 504, `Expected a timeout response, received HTTP ${result.status}: ${result.body}`);
  assert.match(result.body, /timed out|timeout/i);
  assert.ok(elapsedMs >= 4_000 && elapsedMs < 7_000, `Unexpected timeout duration: ${elapsedMs} ms`);
});

test('text proxy refuses chat and model redirects before forwarding data', async () => {
  redirectTargetHits = 0;
  const origin = new URL(upstreamBaseUrl).origin;
  const chatResponse = await fetch(`${appBaseUrl}/api/chat`, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      baseUrl: `${origin}/redirect-chat/v1`,
      model: 'redirect-test-model',
      messages: [{ role: 'user', content: 'redirect boundary regression' }],
    }),
  });
  assert.equal(chatResponse.status, 502);

  const modelsResponse = await fetch(`${appBaseUrl}/api/models`, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ baseUrl: `${origin}/redirect-models/v1` }),
  });
  assert.equal(modelsResponse.status, 502);
  assert.equal(redirectTargetHits, 0, 'The proxy must not follow an upstream redirect to another URL.');
});

test('successful proxied model responses retain the app security boundary', async () => {
  const origin = new URL(upstreamBaseUrl).origin;
  const response = await fetch(`${appBaseUrl}/api/models`, {
    method: 'POST',
    headers: apiHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ baseUrl: `${origin}/complete-models/v1` }),
  });

  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-security-policy') || '', /default-src 'self'/);
  assert.equal(response.headers.get('referrer-policy'), 'no-referrer');
  assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
  assert.equal(response.headers.get('x-frame-options'), 'DENY');
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.deepEqual(await response.json(), {
    object: 'list',
    data: [{ id: 'complete-model' }],
  });
});

function handleUpstreamRequest(req, res) {
  if (req.method === 'GET' && req.url === '/complete-models/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end('{"object":"list","data":[{"id":"complete-model"}]}');
    return;
  }

  if (req.url === '/redirect-chat/v1/chat/completions') {
    res.writeHead(307, { Location: `http://${req.headers.host}/redirect-target-chat` });
    res.end();
    return;
  }

  if (req.url === '/redirect-models/v1/models') {
    res.writeHead(307, { Location: `http://${req.headers.host}/redirect-target-models` });
    res.end();
    return;
  }

  if (req.url === '/redirect-target-chat' || req.url === '/redirect-target-models') {
    redirectTargetHits += 1;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(req.url.endsWith('models')
      ? '{"object":"list","data":[]}'
      : '{"choices":[{"message":{"content":"redirected"}}]}');
    return;
  }

  if (req.method === 'POST' && req.url === '/v1/chat/completions') {
    req.resume();
    req.on('end', () => {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-store',
      });
      res.write('data: {"choices":[{"delta":{"content":"held"}}]}\n\n');
      openChatResponses.add(res);
      res.once('close', () => openChatResponses.delete(res));
    });
    return;
  }

  if (req.method === 'GET' && req.url === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.write('{"object":"list","data":[');
    stalledModelResponses.add(res);
    res.once('close', () => stalledModelResponses.delete(res));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end('{"error":"not found"}');
}

function finishOpenChatResponses() {
  for (const response of openChatResponses) {
    if (!response.writableEnded) response.end('data: [DONE]\n\n');
  }
  openChatResponses.clear();
}

function apiHeaders(headers = {}) {
  return {
    ...headers,
    'X-Access-Token': ACCESS_TOKEN,
  };
}

function rawAppRequest(pathname, headers, body) {
  const target = new URL(appBaseUrl);
  return new Promise((resolve, reject) => {
    const request = http.request({
      hostname: target.hostname,
      port: target.port,
      path: pathname,
      method: 'POST',
      headers: {
        ...headers,
        'Content-Length': Buffer.byteLength(body),
      },
    }, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => resolve({
        status: response.statusCode,
        body: Buffer.concat(chunks).toString('utf8'),
      }));
    });
    request.on('error', reject);
    request.end(body);
  });
}

function writeFakeSdxlCheckpoint(filePath) {
  const tensor = (shape) => ({ dtype: 'F16', shape, data_offsets: [0, 2] });
  const header = Buffer.from(JSON.stringify({
    'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight': tensor([49408, 768]),
    'conditioner.embedders.1.model.token_embedding.weight': tensor([49408, 1280]),
    'model.diffusion_model.input_blocks.0.0.weight': tensor([320, 4, 3, 3]),
    'model.diffusion_model.out.2.weight': tensor([4, 320, 3, 3]),
    'first_stage_model.decoder.conv_out.weight': tensor([3, 128, 3, 3]),
  }));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(filePath, Buffer.concat([length, header, Buffer.alloc(2)]));
}

function makeFakeImageWorkerSource() {
  return `'use strict';
const fs = require('node:fs');
if (process.argv.includes('--probe')) {
  process.stdout.write(JSON.stringify({
    ok: true,
    engine: 'Lifecycle test CUDA',
    gpu: 'Shared test GPU',
    cuda: 'test',
    torch: 'test',
    cudaAvailable: true,
    tokenizerAssets: true
  }));
} else {
  const chunks = [];
  process.stdin.on('data', (chunk) => chunks.push(chunk));
  process.stdin.on('end', () => {
    JSON.parse(Buffer.concat(chunks).toString('utf8'));
    fs.appendFileSync(process.env.FAKE_IMAGE_EVENT_FILE, 'generate\\n');
    process.stdout.write(JSON.stringify({
      ok: true,
      imageBase64: '${TEST_IMAGE_BASE64}',
      mimeType: 'image/png'
    }));
  });
}
`;
}

async function waitForApp() {
  const deadline = Date.now() + 8_000;
  while (Date.now() < deadline) {
    if (hasExited(appProcess)) {
      throw new Error(`Lifecycle test server exited during startup.\n${appOutput}`);
    }
    try {
      const response = await fetch(`${appBaseUrl}/api/config`, { headers: apiHeaders() });
      if (response.ok) return;
    } catch {
      // The child has not opened its listener yet.
    }
    await delay(50);
  }
  throw new Error(`Timed out waiting for lifecycle test server.\n${appOutput}`);
}

async function findAvailableLocalPort() {
  const probe = http.createServer();
  await listenOnRandomLocalPort(probe);
  const port = probe.address().port;
  await closeServer(probe);
  return port;
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
  if (!server || !server.listening) return Promise.resolve();
  if (typeof server.closeAllConnections === 'function') server.closeAllConnections();
  return new Promise((resolve) => server.close(() => resolve()));
}

async function stopChild(child) {
  if (!child || hasExited(child)) return;
  child.kill();
  await Promise.race([
    new Promise((resolve) => child.once('exit', resolve)),
    delay(2_000),
  ]);
  if (!hasExited(child)) {
    child.kill('SIGKILL');
    await Promise.race([
      new Promise((resolve) => child.once('exit', resolve)),
      delay(2_000),
    ]);
  }
}

function hasExited(child) {
  return child.exitCode !== null || child.signalCode !== null;
}

function withDeadline(promise, milliseconds, message) {
  return Promise.race([
    promise,
    delay(milliseconds).then(() => {
      throw new Error(message);
    }),
  ]);
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
