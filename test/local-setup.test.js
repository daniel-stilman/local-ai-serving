'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const test = require('node:test');

const { FIELD_RULES } = require('../local-config');

const ROOT = path.resolve(__dirname, '..');
const ACCESS_TOKEN = 'setup-test-token';
const MODEL_IDS = ['synthetic-language-alpha', 'synthetic-language-beta'];

let fixtureRoot;
let configPath;
let modelsRoot;
let appBaseUrl;
let managedBaseUrl;
let serverProcess;
let serverOutput = '';

test.before(async () => {
  fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'local-setup-regression-'));
  configPath = path.join(fixtureRoot, 'config.local.json');
  modelsRoot = path.join(fixtureRoot, 'models');
  fs.mkdirSync(modelsRoot, { recursive: true });
  fs.writeFileSync(path.join(modelsRoot, `${MODEL_IDS[0]}.gguf`), 'synthetic-a');
  fs.writeFileSync(configPath, `${JSON.stringify({
    version: 1,
    textServerExecutable: process.execPath,
    textModelPath: path.join(modelsRoot, `${MODEL_IDS[0]}.gguf`),
    textModelsRoot: modelsRoot,
  }, null, 2)}\n`);

  const appPort = await reservePort();
  const textPort = await reservePort();
  appBaseUrl = `http://127.0.0.1:${appPort}`;
  managedBaseUrl = `http://127.0.0.1:${textPort}/v1`;
  const environment = makeHermeticEnvironment({
    HOST: '127.0.0.1',
    PORT: String(appPort),
    HTTPS: '0',
    ALLOW_LOCAL_BYPASS: '0',
    ACCESS_TOKEN,
    LOCAL_CONFIG_FILE: configPath,
    TEXT_PORT: String(textPort),
  });

  serverProcess = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: environment,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  serverProcess.stdout.on('data', (chunk) => { serverOutput += chunk.toString('utf8'); });
  serverProcess.stderr.on('data', (chunk) => { serverOutput += chunk.toString('utf8'); });
  await waitForServer();
});

test.after(async () => {
  await stopServer();
  if (fixtureRoot) fs.rmSync(fixtureRoot, { recursive: true, force: true });
});

test('host dashboard saves a model folder and exposes every discovered model to chat', async () => {
  const initialStatus = await fetch(`${appBaseUrl}/api/local-setup`);
  assert.equal(initialStatus.status, 200);
  assert.equal(initialStatus.headers.get('cache-control'), 'no-store');
  assert.equal(initialStatus.headers.get('x-frame-options'), 'DENY');
  assert.deepEqual(await initialStatus.json(), {
    text: {
      runtimeConfigured: true,
      folderConfigured: true,
      managedEnabled: true,
      modelCount: 1,
      models: [MODEL_IDS[0]],
      folderLockedByEnvironment: false,
    },
  });

  const saveResponse = await fetch(`${appBaseUrl}/api/local-setup/text-folder`, {
    method: 'POST',
    headers: setupHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ path: modelsRoot }),
  });
  assert.equal(saveResponse.status, 200);
  const savedStatus = await saveResponse.json();
  assert.equal(savedStatus.text.runtimeConfigured, true);
  assert.equal(savedStatus.text.folderConfigured, true);
  assert.equal(savedStatus.text.managedEnabled, true);
  assert.equal(savedStatus.text.modelCount, 1);
  assert.deepEqual(savedStatus.text.models, [MODEL_IDS[0]]);
  assert.equal(JSON.stringify(savedStatus).includes(modelsRoot), false);

  const configResponse = await fetch(`${appBaseUrl}/api/config`, { headers: accessHeaders() });
  assert.equal(configResponse.status, 200);
  const appConfig = await configResponse.json();
  assert.equal(appConfig.defaultBaseUrl, managedBaseUrl);
  assert.equal(appConfig.managedTextBackend.enabled, true);
  assert.equal(appConfig.localSetupAvailable, true);

  const modelsResponse = await fetch(`${appBaseUrl}/api/models`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ baseUrl: managedBaseUrl }),
  });
  assert.equal(modelsResponse.status, 200);
  const models = await modelsResponse.json();
  assert.deepEqual(models.data.map((model) => model.id), [MODEL_IDS[0]]);

  const savedConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  assert.equal(savedConfig.textModelsRoot, modelsRoot);
  assert.equal(savedConfig.textServerExecutable, process.execPath);
  assert.equal(Object.hasOwn(savedConfig, 'textModelPath'), false);
  assert.equal(Object.hasOwn(savedConfig, 'externalTextBaseUrl'), false);
  assert.equal(serverOutput.includes(modelsRoot), false);
  assert.ok(MODEL_IDS.every((id) => !serverOutput.includes(id)));
});

test('dashboard refresh discovers models added after startup without a restart', async () => {
  fs.writeFileSync(path.join(modelsRoot, `${MODEL_IDS[1]}.gguf`), 'synthetic-model-b');
  const refreshResponse = await fetch(`${appBaseUrl}/api/local-setup/refresh-text-models`, {
    method: 'POST',
    headers: setupHeaders(),
  });
  assert.equal(refreshResponse.status, 200);
  const status = await refreshResponse.json();
  assert.equal(status.text.modelCount, MODEL_IDS.length);
  assert.deepEqual(new Set(status.text.models), new Set(MODEL_IDS));

  const configResponse = await fetch(`${appBaseUrl}/api/config`, { headers: accessHeaders() });
  const config = await configResponse.json();
  const modelsResponse = await fetch(`${appBaseUrl}/api/models`, {
    method: 'POST',
    headers: accessHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ baseUrl: config.defaultBaseUrl }),
  });
  assert.equal(modelsResponse.status, 200);
  const models = await modelsResponse.json();
  assert.deepEqual(new Set(models.data.map((model) => model.id)), new Set(MODEL_IDS));
});

test('local setup rejects invalid folders without changing the saved configuration', async () => {
  const before = fs.readFileSync(configPath, 'utf8');
  const invalidValues = [
    '',
    path.join(fixtureRoot, 'missing'),
    path.join(modelsRoot, `${MODEL_IDS[0]}.gguf`),
  ];
  for (const candidate of invalidValues) {
    const response = await fetch(`${appBaseUrl}/api/local-setup/text-folder`, {
      method: 'POST',
      headers: setupHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ path: candidate }),
    });
    assert.equal(response.status, 400);
    const errorText = await response.text();
    if (candidate) assert.equal(errorText.includes(candidate), false);
    assert.equal(fs.readFileSync(configPath, 'utf8'), before);
  }

  const emptyRoot = path.join(fixtureRoot, 'empty');
  fs.mkdirSync(emptyRoot);
  const noModelsResponse = await fetch(`${appBaseUrl}/api/local-setup/text-folder`, {
    method: 'POST',
    headers: setupHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ path: emptyRoot }),
  });
  assert.equal(noModelsResponse.status, 400);
  assert.equal((await noModelsResponse.text()).includes(emptyRoot), false);
  assert.equal(fs.readFileSync(configPath, 'utf8'), before);
});

test('model-folder changes cannot interrupt an active text response', async () => {
  let upstreamResponse;
  let markRequestStarted;
  const requestStarted = new Promise((resolve) => { markRequestStarted = resolve; });
  const upstream = http.createServer((_request, response) => {
    upstreamResponse = response;
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.write('data: {"choices":[{"delta":{"content":"synthetic"}}]}\n\n');
    markRequestStarted();
  });
  await listen(upstream);
  const before = fs.readFileSync(configPath, 'utf8');

  try {
    const chatPromise = fetch(`${appBaseUrl}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        baseUrl: `http://127.0.0.1:${upstream.address().port}/v1`,
        model: 'synthetic-external-model',
        messages: [{ role: 'user', content: 'synthetic request' }],
      }),
    });
    await requestStarted;
    const chatResponse = await chatPromise;
    assert.equal(chatResponse.status, 200);

    const switchResponse = await fetch(`${appBaseUrl}/api/chat`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        baseUrl: managedBaseUrl,
        model: MODEL_IDS[1],
        messages: [{ role: 'user', content: 'synthetic competing request' }],
      }),
    });
    assert.equal(switchResponse.status, 409);

    const mutationResponse = await fetch(`${appBaseUrl}/api/local-setup/text-folder`, {
      method: 'POST',
      headers: setupHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ path: modelsRoot }),
    });
    assert.equal(mutationResponse.status, 409);
    assert.equal(fs.readFileSync(configPath, 'utf8'), before);

    const refreshResponse = await fetch(`${appBaseUrl}/api/local-setup/refresh-text-models`, {
      method: 'POST',
      headers: setupHeaders(),
    });
    assert.equal(refreshResponse.status, 409);
    assert.equal(fs.readFileSync(configPath, 'utf8'), before);

    upstreamResponse.end('data: [DONE]\n\n');
    assert.match(await chatResponse.text(), /\[DONE\]/);
  } finally {
    if (upstreamResponse && !upstreamResponse.writableEnded) upstreamResponse.end();
    await close(upstream);
  }
});

test('phone-style hosts and access tokens cannot reach or mutate host setup', async () => {
  fs.writeFileSync(path.join(modelsRoot, `${MODEL_IDS[1]}.gguf`), 'synthetic-model-b');
  const refreshResponse = await fetch(`${appBaseUrl}/api/local-setup/refresh-text-models`, {
    method: 'POST',
    headers: setupHeaders(),
  });
  assert.equal(refreshResponse.status, 200);

  const remoteHeaders = {
    Host: '192.0.2.10',
    Origin: 'http://192.0.2.10',
    'X-Local-Setup': '1',
  };
  const statusResponse = await fetch(`${appBaseUrl}/api/local-setup`, { headers: remoteHeaders });
  assert.equal(statusResponse.status, 403);

  const before = fs.readFileSync(configPath, 'utf8');
  const mutationResponse = await fetch(`${appBaseUrl}/api/local-setup/text-folder`, {
    method: 'POST',
    headers: {
      ...remoteHeaders,
      'X-Access-Token': ACCESS_TOKEN,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ path: modelsRoot }),
  });
  assert.equal(mutationResponse.status, 403);
  assert.equal(fs.readFileSync(configPath, 'utf8'), before);

  const configResponse = await fetch(`${appBaseUrl}/api/config`, {
    headers: {
      Host: '192.0.2.10',
      Origin: 'http://192.0.2.10',
      'X-Access-Token': ACCESS_TOKEN,
    },
  });
  assert.equal(configResponse.status, 200);
  const phoneConfig = await configResponse.json();
  assert.equal(phoneConfig.localSetupAvailable, false);

  const modelsResponse = await fetch(`${appBaseUrl}/api/models`, {
    method: 'POST',
    headers: {
      Host: '192.0.2.10',
      Origin: 'http://192.0.2.10',
      'X-Access-Token': ACCESS_TOKEN,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ baseUrl: phoneConfig.defaultBaseUrl }),
  });
  assert.equal(modelsResponse.status, 200);
  const phoneModels = await modelsResponse.json();
  assert.deepEqual(new Set(phoneModels.data.map((model) => model.id)), new Set(MODEL_IDS));
});

test('an explicit model-root environment override wins and locks host folder changes', {
  timeout: 15_000,
}, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-setup-precedence-'));
  const configModelsRoot = path.join(root, 'config-models');
  const overrideModelsRoot = path.join(root, 'override-models');
  const isolatedConfigPath = path.join(root, 'config.local.json');
  const configModelId = 'synthetic-config-language';
  const overrideModelId = 'synthetic-override-language';
  fs.mkdirSync(configModelsRoot);
  fs.mkdirSync(overrideModelsRoot);
  fs.writeFileSync(path.join(configModelsRoot, `${configModelId}.gguf`), 'synthetic-config');
  fs.writeFileSync(path.join(overrideModelsRoot, `${overrideModelId}.gguf`), 'synthetic-override');
  fs.writeFileSync(isolatedConfigPath, `${JSON.stringify({
    version: 1,
    textServerExecutable: process.execPath,
    textModelsRoot: configModelsRoot,
  }, null, 2)}\n`);
  const savedBefore = fs.readFileSync(isolatedConfigPath, 'utf8');
  const fixture = await launchIsolatedServer(isolatedConfigPath, {
    TEXT_MODELS_ROOT: overrideModelsRoot,
  });

  try {
    const statusResponse = await fetch(`${fixture.baseUrl}/api/local-setup`);
    assert.equal(statusResponse.status, 200);
    const status = await statusResponse.json();
    assert.equal(status.text.folderLockedByEnvironment, true);
    assert.deepEqual(status.text.models, [overrideModelId]);

    const configResponse = await fetch(`${fixture.baseUrl}/api/config`, {
      headers: accessHeaders(),
    });
    assert.equal(configResponse.status, 200);
    const config = await configResponse.json();
    assert.equal(config.managedTextBackend.enabled, true);

    const modelsResponse = await fetch(`${fixture.baseUrl}/api/models`, {
      method: 'POST',
      headers: accessHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ baseUrl: config.defaultBaseUrl }),
    });
    assert.equal(modelsResponse.status, 200);
    assert.deepEqual((await modelsResponse.json()).data.map((model) => model.id), [overrideModelId]);

    const mutationResponse = await fetch(`${fixture.baseUrl}/api/local-setup/text-folder`, {
      method: 'POST',
      headers: isolatedSetupHeaders(fixture.baseUrl, { 'Content-Type': 'application/json' }),
      body: JSON.stringify({ path: configModelsRoot }),
    });
    assert.equal(mutationResponse.status, 409);
    assert.equal(fs.readFileSync(isolatedConfigPath, 'utf8'), savedBefore);
    assert.equal(fixture.output().includes(configModelsRoot), false);
    assert.equal(fixture.output().includes(overrideModelsRoot), false);
    assert.equal(fixture.output().includes(configModelId), false);
    assert.equal(fixture.output().includes(overrideModelId), false);
  } finally {
    await fixture.stop();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('server startup tolerates stale ignored resources while retaining valid local settings', {
  timeout: 15_000,
}, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-setup-stale-'));
  const isolatedConfigPath = path.join(root, 'config.local.json');
  const staleExecutable = path.join(root, 'missing-runtime');
  const staleModelsRoot = path.join(root, 'missing-models');
  const externalTextBaseUrl = 'http://127.0.0.1:65431/v1';
  fs.writeFileSync(isolatedConfigPath, `${JSON.stringify({
    version: 1,
    textServerExecutable: staleExecutable,
    textModelsRoot: staleModelsRoot,
    externalTextBaseUrl,
  }, null, 2)}\n`);
  const fixture = await launchIsolatedServer(isolatedConfigPath);

  try {
    const configResponse = await fetch(`${fixture.baseUrl}/api/config`, {
      headers: accessHeaders(),
    });
    assert.equal(configResponse.status, 200);
    const config = await configResponse.json();
    assert.equal(config.defaultBaseUrl, externalTextBaseUrl);
    assert.equal(config.managedTextBackend.enabled, false);
    assert.equal(config.setupRequired, false);

    const statusResponse = await fetch(`${fixture.baseUrl}/api/local-setup`);
    assert.equal(statusResponse.status, 200);
    const status = await statusResponse.json();
    assert.equal(status.text.runtimeConfigured, false);
    assert.equal(status.text.folderConfigured, false);
    assert.equal(status.text.managedEnabled, false);
    assert.equal(status.text.modelCount, 0);
    assert.equal(fixture.output().includes(staleExecutable), false);
    assert.equal(fixture.output().includes(staleModelsRoot), false);
  } finally {
    await fixture.stop();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

function setupHeaders(extra = {}) {
  return {
    Origin: appBaseUrl,
    'X-Local-Setup': '1',
    ...extra,
  };
}

function accessHeaders(extra = {}) {
  return { 'X-Access-Token': ACCESS_TOKEN, ...extra };
}

function isolatedSetupHeaders(baseUrl, extra = {}) {
  return {
    Origin: baseUrl,
    'X-Local-Setup': '1',
    ...extra,
  };
}

function makeHermeticEnvironment(overrides) {
  const environment = { ...process.env };
  const sensitivePrefixes = ['TEXT_', 'IMAGE_', 'ANIMA_', 'SMOKE_'];
  for (const key of Object.keys(environment)) {
    const normalized = key.toUpperCase();
    if (
      sensitivePrefixes.some((prefix) => normalized.startsWith(prefix))
      || normalized.startsWith('LOCAL_CONFIG')
      || ['HOST', 'PORT', 'HTTPS', 'ALLOW_LOCAL_BYPASS', 'ACCESS_TOKEN', 'PRIVATE_DIAGNOSTICS'].includes(normalized)
    ) delete environment[key];
  }
  for (const rule of Object.values(FIELD_RULES)) delete environment[rule.environment];
  return Object.assign(environment, overrides);
}

async function launchIsolatedServer(isolatedConfigPath, overrides = {}) {
  const appPort = await reservePort();
  const textPort = await reservePort();
  const baseUrl = `http://127.0.0.1:${appPort}`;
  let output = '';
  const child = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: makeHermeticEnvironment({
      HOST: '127.0.0.1',
      PORT: String(appPort),
      HTTPS: '0',
      ALLOW_LOCAL_BYPASS: '0',
      ACCESS_TOKEN,
      LOCAL_CONFIG_FILE: isolatedConfigPath,
      TEXT_PORT: String(textPort),
      ...overrides,
    }),
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  child.stdout.on('data', (chunk) => { output += chunk.toString('utf8'); });
  child.stderr.on('data', (chunk) => { output += chunk.toString('utf8'); });
  try {
    await waitForStandaloneServer(child, baseUrl);
  } catch (error) {
    await stopChild(child);
    throw error;
  }
  return {
    baseUrl,
    output: () => output,
    stop: () => stopChild(child),
  };
}

async function reservePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
}

async function close(server) {
  if (!server.listening) return;
  await new Promise((resolve) => server.close(resolve));
}

async function waitForServer() {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (serverProcess.exitCode !== null) throw new Error('Synthetic local setup server exited before readiness.');
    try {
      const response = await fetch(`${appBaseUrl}/`);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error('Synthetic local setup server did not become ready.');
}

async function waitForStandaloneServer(child, baseUrl) {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error('An isolated synthetic setup server exited before readiness.');
    }
    try {
      const response = await fetch(`${baseUrl}/`);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error('An isolated synthetic setup server did not become ready.');
}

async function stopChild(child) {
  if (!child || child.exitCode !== null) return;
  const exited = new Promise((resolve) => child.once('exit', resolve));
  child.kill();
  await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 3_000))]);
  if (child.exitCode === null) {
    child.kill('SIGKILL');
    await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 1_000))]);
  }
}

async function stopServer() {
  if (!serverProcess || serverProcess.exitCode !== null) return;
  const exited = new Promise((resolve) => serverProcess.once('exit', resolve));
  serverProcess.kill();
  await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 3_000))]);
  if (serverProcess.exitCode === null) serverProcess.kill('SIGKILL');
}
