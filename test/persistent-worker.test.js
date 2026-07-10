'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const TEST_TOKEN = 'persistent-worker-test-token';

test('persistent image worker reuses one process and accepts fragmented NDJSON responses', {
  timeout: 15_000,
}, async (t) => {
  const fixture = await startFixture({ idleMs: 5_000 });
  t.after(() => fixture.stop());

  const first = await fixture.generate('reuse-first');
  const second = await fixture.generate('fragmented-second');

  assert.equal(first.status, 200);
  assert.equal(second.status, 200);
  const firstImage = decodeWorkerImage(first.body);
  const secondImage = decodeWorkerImage(second.body);
  assert.equal(firstImage.prompt, 'reuse-first');
  assert.equal(secondImage.prompt, 'fragmented-second');
  assert.equal(secondImage.pid, firstImage.pid);
  assert.equal(secondImage.sequence, firstImage.sequence + 1);
  assert.equal(fixture.events().filter((event) => event.type === 'start').length, 1);
});

test('concurrent image requests share one in-flight runtime probe', {
  timeout: 15_000,
}, async (t) => {
  const fixture = await startFixture({ idleMs: 5_000 });
  t.after(() => fixture.stop());

  const responses = await Promise.all([
    fixture.generate('concurrent-probe-one'),
    fixture.generate('concurrent-probe-two'),
  ]);

  assert.deepEqual(responses.map((response) => response.status).sort(), [200, 409]);
  assert.equal(fixture.events().filter((event) => event.type === 'probe').length, 1);
});

test('concurrent text requests share one warm-worker GPU handoff', {
  timeout: 15_000,
}, async (t) => {
  const fixture = await startFixture({ idleMs: 5_000 });
  t.after(() => fixture.stop());
  const warmed = await fixture.generate('warm-before-text');
  assert.equal(warmed.status, 200);

  const upstream = http.createServer((request, response) => {
    if (request.url === '/v1/models') {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end('{"object":"list","data":[]}');
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise((resolve, reject) => {
    upstream.once('error', reject);
    upstream.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => new Promise((resolve) => upstream.close(resolve)));
  const upstreamUrl = `http://127.0.0.1:${upstream.address().port}/v1`;

  const responses = await Promise.all([
    fixture.models(upstreamUrl),
    fixture.models(upstreamUrl),
  ]);

  assert.deepEqual(responses.map((response) => response.status), [200, 200]);
});

test('coalesced extra NDJSON output never crosses into the next image request', {
  timeout: 15_000,
}, async (t) => {
  const fixture = await startFixture({ idleMs: 5_000 });
  t.after(() => fixture.stop());

  const first = await fixture.generate('coalesced-first');
  const second = await generateWhenGpuFree(fixture, 'after-coalesced');

  assert.equal(first.status, 200);
  assert.equal(second.status, 200);
  assert.equal(decodeWorkerImage(first.body).prompt, 'coalesced-first');
  assert.equal(
    decodeWorkerImage(second.body).prompt,
    'after-coalesced',
    'an unsolicited line from request A must not become request B\'s response',
  );
});

test('persistent image worker crash fails the request and the next request starts cleanly', {
  timeout: 15_000,
}, async (t) => {
  const fixture = await startFixture({ idleMs: 5_000 });
  t.after(() => fixture.stop());

  const crashed = await fixture.generate('crash-worker');
  assert.equal(crashed.status, 502);
  assert.match(crashed.body.error, /exited unexpectedly/i);

  const recovered = await fixture.generate('after-crash');
  assert.equal(recovered.status, 200);
  const recoveredImage = decodeWorkerImage(recovered.body);
  assert.equal(recoveredImage.prompt, 'after-crash');

  const starts = fixture.events().filter((event) => event.type === 'start');
  assert.equal(starts.length, 2);
  assert.notEqual(starts[0].pid, starts[1].pid);
  assert.equal(recoveredImage.pid, starts[1].pid);
});

test('client abort kills the warm worker and the next request starts cleanly', {
  timeout: 20_000,
}, async (t) => {
  const fixture = await startFixture({ idleMs: 5_000 });
  t.after(() => fixture.stop());
  const controller = new AbortController();
  const hanging = fixture.generate('hang-worker', { signal: controller.signal });
  await waitFor(
    () => fixture.events().some((event) => event.type === 'request' && event.prompt === 'hang-worker'),
    5_000,
    'the hanging request to reach the warm worker',
  );

  controller.abort();
  await assert.rejects(hanging, (error) => error.name === 'AbortError');
  await delay(150);

  const recovered = await fixture.generate('after-abort');
  assert.equal(recovered.status, 200);
  assert.equal(decodeWorkerImage(recovered.body).prompt, 'after-abort');
  assert.equal(fixture.events().filter((event) => event.type === 'start').length, 2);
});

test('persistent image worker restarts after its idle deadline', {
  timeout: 15_000,
}, async (t) => {
  const fixture = await startFixture({ idleMs: 75 });
  t.after(() => fixture.stop());

  const first = await fixture.generate('before-idle');
  assert.equal(first.status, 200);
  const firstImage = decodeWorkerImage(first.body);

  await waitFor(
    () => fixture.events().filter((event) => event.type === 'start').length === 1,
    1_000,
    'the first persistent worker to start',
  );
  await delay(300);

  const second = await fixture.generate('after-idle');
  assert.equal(second.status, 200);
  const secondImage = decodeWorkerImage(second.body);
  assert.equal(secondImage.prompt, 'after-idle');
  assert.notEqual(secondImage.pid, firstImage.pid);
  assert.equal(fixture.events().filter((event) => event.type === 'start').length, 2);
});

test('server shutdown waits for its idle warm worker to exit', {
  timeout: 15_000,
}, async () => {
  const fixture = await startFixture({ idleMs: 5_000 });
  const generated = await fixture.generate('before-shutdown');
  assert.equal(generated.status, 200);
  const workerPid = decodeWorkerImage(generated.body).pid;
  assert.equal(isProcessAlive(workerPid), true);

  const shutdown = await fixture.stop();
  assert.equal(shutdown.exitCode, 0);
  assert.equal(await isPortOpen(shutdown.port), false);
  await waitFor(() => !isProcessAlive(workerPid), 2_000, 'the warm worker process to exit');
});

async function startFixture({ idleMs }) {
  const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'persistent-image-worker-test-'));
  const modelsRoot = path.join(fixtureRoot, 'models');
  const workerPath = path.join(fixtureRoot, 'fake-ndjson-worker.js');
  const eventsPath = path.join(fixtureRoot, 'worker-events.ndjson');
  fs.mkdirSync(path.join(modelsRoot, 'checkpoints'), { recursive: true });
  writeFakeSdxlCheckpoint(path.join(modelsRoot, 'checkpoints', 'sdxl-test.safetensors'));
  fs.writeFileSync(workerPath, makeFakeWorkerSource());

  const port = await getFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  let output = '';
  const serverProcess = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      LOCAL_CONFIG_DISABLED: '1',
      HOST: '127.0.0.1',
      PORT: String(port),
      HTTPS: '0',
      ALLOW_LOCAL_BYPASS: '0',
      ACCESS_TOKEN: TEST_TOKEN,
      TEXT_BASE_URL: 'http://127.0.0.1:65535/v1',
      IMAGE_MODELS_ROOT: modelsRoot,
      IMAGE_PYTHON: process.execPath,
      IMAGE_WORKER_PATH: workerPath,
      IMAGE_WORKER_PERSISTENT: '1',
      IMAGE_WORKER_IDLE_MS: String(idleMs),
      FAKE_WORKER_EVENTS: eventsPath,
    },
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
  });
  serverProcess.stdout.on('data', (chunk) => {
    output += chunk.toString('utf8');
  });
  serverProcess.stderr.on('data', (chunk) => {
    output += chunk.toString('utf8');
  });

  try {
    await waitForServer(serverProcess, baseUrl, () => output);
  } catch (error) {
    await stopProcess(serverProcess);
    fs.rmSync(fixtureRoot, { recursive: true, force: true });
    throw error;
  }

  let stopped = false;
  return {
    async generate(prompt, options = {}) {
      const response = await fetch(`${baseUrl}/api/image/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Access-Token': TEST_TOKEN,
        },
        body: JSON.stringify({
          kind: 'sdxl',
          model: 'sdxl-test.safetensors',
          prompt,
          size: 'square',
          seed: 123,
          steps: 1,
        }),
        signal: options.signal,
      });
      return { status: response.status, body: await response.json() };
    },
    async models(textBaseUrl) {
      const response = await fetch(`${baseUrl}/api/models`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Access-Token': TEST_TOKEN,
        },
        body: JSON.stringify({ baseUrl: textBaseUrl }),
      });
      return { status: response.status, body: await response.text() };
    },
    events() {
      if (!fs.existsSync(eventsPath)) return [];
      return fs.readFileSync(eventsPath, 'utf8')
        .split(/\r?\n/)
        .filter(Boolean)
        .map((line) => JSON.parse(line));
    },
    async stop() {
      if (stopped) return;
      stopped = true;
      await stopProcess(serverProcess);
      fs.rmSync(fixtureRoot, { recursive: true, force: true });
      return { exitCode: serverProcess.exitCode, port };
    },
  };
}

function decodeWorkerImage(body) {
  assert.equal(typeof body.imageBase64, 'string');
  return JSON.parse(Buffer.from(body.imageBase64, 'base64').toString('utf8'));
}

async function generateWhenGpuFree(fixture, prompt) {
  const deadline = Date.now() + 2_000;
  while (Date.now() < deadline) {
    const response = await fixture.generate(prompt);
    if (response.status !== 409) return response;
    await delay(25);
  }
  throw new Error('Timed out waiting for the previous image worker to release the GPU.');
}

function makeFakeWorkerSource() {
  return `'use strict';
const fs = require('node:fs');
const readline = require('node:readline');
const eventsPath = process.env.FAKE_WORKER_EVENTS;

if (process.argv.includes('--probe')) {
  record({ type: 'probe', pid: process.pid });
  process.stdout.write(JSON.stringify({
    ok: true,
    engine: 'Persistent worker test engine',
    gpu: 'Test GPU',
    cuda: 'test',
    torch: 'test',
    cudaAvailable: true,
    tokenizerAssets: true
  }));
} else if (process.argv.includes('--serve')) {
  let sequence = 0;
  record({ type: 'start', pid: process.pid });
  const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  lines.on('line', (line) => {
    const request = JSON.parse(line);
    sequence += 1;
    record({ type: 'request', pid: process.pid, prompt: request.prompt, sequence });
    if (request.prompt === 'crash-worker') {
      process.exit(23);
      return;
    }
    if (request.prompt === 'hang-worker') return;
    const response = makeResponse(request.prompt, sequence);
    if (request.prompt === 'fragmented-second') {
      const split = Math.floor(response.length / 2);
      process.stdout.write(response.slice(0, split));
      setTimeout(() => process.stdout.write(response.slice(split) + '\\n'), 15);
      return;
    }
    if (request.prompt === 'coalesced-first') {
      process.stdout.write(response + '\\n' + makeResponse('stale-extra', sequence) + '\\n');
      return;
    }
    process.stdout.write(response + '\\n');
  });
} else {
  process.stdout.write(JSON.stringify({ ok: false, error: 'Expected --probe or --serve.' }));
}

function makeResponse(prompt, sequence) {
  const imageBase64 = Buffer.from(JSON.stringify({ prompt, sequence, pid: process.pid })).toString('base64');
  return JSON.stringify({ ok: true, imageBase64, mimeType: 'image/png' });
}

function record(event) {
  fs.appendFileSync(eventsPath, JSON.stringify(event) + '\\n');
}
`;
}

function writeFakeSdxlCheckpoint(filePath) {
  const header = Buffer.from(JSON.stringify({
    __metadata__: { 'modelspec.architecture': 'stable-diffusion-xl-v1-base' },
    'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight': fakeTensor([49408, 768]),
    'conditioner.embedders.1.model.token_embedding.weight': fakeTensor([49408, 1280]),
    'model.diffusion_model.input_blocks.0.0.weight': fakeTensor([320, 4, 3, 3]),
    'model.diffusion_model.out.2.weight': fakeTensor([4, 320, 3, 3]),
    'first_stage_model.decoder.conv_out.weight': fakeTensor([3, 128, 3, 3]),
  }));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(filePath, Buffer.concat([length, header, Buffer.alloc(2)]));
}

function fakeTensor(shape) {
  return { dtype: 'F16', shape, data_offsets: [0, 2] };
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const reservation = net.createServer();
    reservation.once('error', reject);
    reservation.listen(0, '127.0.0.1', () => {
      const address = reservation.address();
      reservation.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve(address.port);
      });
    });
  });
}

async function waitForServer(serverProcess, baseUrl, readOutput) {
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    if (serverProcess.exitCode !== null) {
      throw new Error(`Server exited early with code ${serverProcess.exitCode}: ${readOutput()}`);
    }
    try {
      const response = await fetch(`${baseUrl}/api/config`, {
        headers: { 'X-Access-Token': TEST_TOKEN },
      });
      if (response.ok) return;
    } catch {
      // The dedicated server is still binding its loopback port.
    }
    await delay(40);
  }
  throw new Error(`Timed out waiting for the dedicated server: ${readOutput()}`);
}

async function stopProcess(child) {
  if (child.exitCode !== null) return;
  const exited = once(child, 'exit');
  child.send({ type: 'shutdown' });
  const graceful = await Promise.race([
    exited.then(() => true),
    delay(2_500).then(() => false),
  ]);
  if (graceful || child.exitCode !== null) return;
  child.kill('SIGKILL');
  await Promise.race([exited, delay(1_000)]);
}

async function waitFor(predicate, timeoutMs, description) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await delay(20);
  }
  throw new Error(`Timed out waiting for ${description}.`);
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function isProcessAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function isPortOpen(port) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: '127.0.0.1', port });
    socket.once('connect', () => {
      socket.destroy();
      resolve(true);
    });
    socket.once('error', () => resolve(false));
    socket.setTimeout(500, () => {
      socket.destroy();
      resolve(false);
    });
  });
}
