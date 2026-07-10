'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const ROOT = path.resolve(__dirname, '..');
const PUBLIC_ROOT = path.join(ROOT, 'public');
const STATE_STORAGE_KEY = 'local-ai-serving-state-v1';
const CURRENT_TOKEN = 'browser-smoke-current-token';
const EXTERNAL_TOKEN = 'browser-smoke-external-token';
const STALE_TOKEN = 'browser-smoke-stale-token';
const REAL_SERVER_TOKEN = 'browser-smoke-real-server-token';
const VALID_TOKENS = new Set([CURRENT_TOKEN, EXTERNAL_TOKEN]);
const MANAGED_BASE_URL = 'http://synthetic-managed.invalid/v1';
const LEGACY_BASE_URL = 'http://synthetic-legacy.invalid/v1';
const EXTERNAL_BASE_URL = 'http://synthetic-external.invalid/v1';
const TEXT_MODELS = ['synthetic-text-alpha', 'synthetic-text-beta'];
const STALE_TEXT_MODEL = 'synthetic-text-stale';
const CHAT_TURN_ONE = Object.freeze({
  prompt: 'Remember the first synthetic detail.',
  response: 'The first synthetic detail is remembered.',
});
const CHAT_TURN_TWO = Object.freeze({
  prompt: 'Use that detail in a second synthetic turn.',
  response: 'The second synthetic turn used the remembered detail.',
});
const CHAT_IMAGE_TOOL = Object.freeze({
  prompt: 'Create a synthetic picture with the image tool.',
  imagePrompt: 'A small synthetic landscape used for browser regression testing.',
  response: 'The synthetic tool image is ready.',
});
const MANUAL_IMAGE_PROMPT = 'A synthetic square made through Image studio.';
const EXPECTED_AUTO_NEGATIVE = 'bad A synthetic bad square made bad through Image bad studio.';
const CHAT_AFTER_IMAGE = Object.freeze({
  prompt: 'Continue the conversation after both images.',
  response: 'The conversation continued after both images.',
  regeneratedResponse: 'The conversation was regenerated after both images.',
});
const CHAT_EDITED_TURN = Object.freeze({
  prompt: 'Continue with an edited follow-up after both images.',
  response: 'The edited follow-up kept the earlier mixed conversation.',
});
const RENAMED_CONVERSATION = 'Synthetic mixed conversation';
const SYNTHETIC_TOOL_CALL_ID = 'synthetic-image-tool-call';
const SYNTHETIC_IMAGE_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
const SYNTHETIC_PHONE_HOST = 'synthetic-phone.invalid';
const ANIMA_IMAGE_MODELS = ['synthetic-image-anima', 'synthetic-image-anima-variant'];
const SDXL_IMAGE_MODELS = ['synthetic-image-sdxl', 'synthetic-image-sdxl-variant'];
const IMAGE_MODELS = [...ANIMA_IMAGE_MODELS, ...SDXL_IMAGE_MODELS];
const API_PATHS = new Set(['/api/config', '/api/models', '/api/image/config']);
const FIXTURE_API_PATHS = new Set([...API_PATHS, '/api/chat', '/api/image/generate']);
const TEXT_LOAD_API_PATHS = new Set(['/api/text/load', '/api/text/status']);
for (const apiPath of TEXT_LOAD_API_PATHS) FIXTURE_API_PATHS.add(apiPath);
const ACCESS_REQUIRED_ID = 'accessRequired';
const OPTIONAL = process.argv.includes('--optional');
const BROWSER_TIMEOUT_MS = 30_000;
const PAGE_SETTLE_TIMEOUT_MS = 15_000;
const REAL_SERVER_TIMEOUT_MS = 15_000;
const REAL_SERVER_TEMP_PREFIX = 'local-ai-real-browser-';
const PHONE_VIEWPORT = Object.freeze({ width: 390, height: 844, mobile: true });
const DESKTOP_VIEWPORT = Object.freeze({ width: 1440, height: 900, mobile: false });
const TEXT_CONFIG_ENVIRONMENT_KEYS = [
  'TEXT_SERVER_EXE',
  'TEXT_MODEL_PATH',
  'TEXT_MODELS_ROOT',
  'TEXT_BASE_URL',
  'TEXT_BACKEND',
  'TEXT_MODEL_ALIAS',
  'TEXT_MODEL_MAX_GIB',
  'TEXT_PORT',
];
const MACHINE_CONFIG_ENVIRONMENT_KEYS = [
  ...TEXT_CONFIG_ENVIRONMENT_KEYS,
  'IMAGE_MODELS_ROOT',
  'IMAGE_PYTHON',
  'ANIMA_TEXT_ENCODER_PATH',
  'ANIMA_VAE_PATH',
  'LOCAL_CONFIG_DISABLED',
];

const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
};

main().catch((error) => {
  console.error(`Browser regression failed: ${safeMessage(error)}`);
  process.exitCode = 1;
});

async function main() {
  const browserPath = findBrowser();
  if (!browserPath) {
    if (OPTIONAL) {
      console.log('Browser regression skipped: no supported Edge/Chrome executable was found.');
      return;
    }
    throw new Error('No supported Edge/Chrome executable was found (use --optional to allow a skip).');
  }
  const requests = [];
  const server = createFixtureServer(requests);
  await listen(server);
  const address = server.address();
  const origin = `http://127.0.0.1:${address.port}`;

  try {
    const scenarios = [
      {
        name: 'current fragment token with legacy saved backend',
        fragment: `#access=${CURRENT_TOKEN}`,
        accessRequired: false,
        accessToken: CURRENT_TOKEN,
        localSetupAvailable: true,
        initialState: { settings: { baseUrl: LEGACY_BASE_URL } },
        expectedBaseUrl: MANAGED_BASE_URL,
        expectedBaseUrlOverride: false,
        exerciseChat: true,
        viewport: PHONE_VIEWPORT,
      },
      {
        name: 'current explicit external override',
        fragment: `#access=${EXTERNAL_TOKEN}`,
        accessRequired: false,
        accessToken: EXTERNAL_TOKEN,
        localSetupAvailable: false,
        initialState: { settings: { baseUrl: EXTERNAL_BASE_URL, baseUrlOverride: true } },
        expectedBaseUrl: EXTERNAL_BASE_URL,
        expectedBaseUrlOverride: true,
        viewport: DESKTOP_VIEWPORT,
      },
      {
        name: 'current stale saved model replacement',
        fragment: `#access=${CURRENT_TOKEN}`,
        accessRequired: false,
        accessToken: CURRENT_TOKEN,
        localSetupAvailable: true,
        initialState: { settings: { baseUrl: LEGACY_BASE_URL, model: STALE_TEXT_MODEL } },
        expectedBaseUrl: MANAGED_BASE_URL,
        expectedBaseUrlOverride: false,
        expectedReplacementModel: TEXT_MODELS[0],
        viewport: PHONE_VIEWPORT,
      },
      { name: 'missing token', fragment: '', accessRequired: true, viewport: PHONE_VIEWPORT },
      { name: 'stale fragment token', fragment: `#access=${STALE_TOKEN}`, accessRequired: true, viewport: PHONE_VIEWPORT },
    ];

    for (const scenario of scenarios) {
      const before = requests.length;
      const page = await inspectPage(browserPath, `${origin}/${scenario.fragment}`, scenario);
      const scenarioRequests = requests.slice(before);
      assertScenario(page, scenario, scenarioRequests);
    }

  } finally {
    await closeServer(server);
  }

  await runRealServerModelSetupScenario(browserPath);
  console.log('Browser regression passed: full mixed-conversation lifecycle, auth/model flows, and real-server setup are covered.');
}

async function runRealServerModelSetupScenario(browserPath) {
  const fixture = await createRealServerFixture();
  let serverProcess = null;
  try {
    const [appPort, textPort] = await allocateDistinctPorts(2);
    serverProcess = startRealServer(fixture.configPath, appPort, textPort);
    const origin = `http://127.0.0.1:${appPort}`;
    await waitForRealServer(origin, serverProcess);

    const dashboardUrl = `${origin}/dashboard#models`;
    const dashboard = await configureModelsThroughDashboard(
      browserPath,
      dashboardUrl,
      fixture.modelsRoot,
      [TEXT_MODELS[0]],
    );
    assert.deepEqual(dashboard.models, [TEXT_MODELS[0]],
      'real server: initial dashboard setup did not start with one synthetic text model');
    assert.equal(dashboard.count, '1',
      'real server: initial dashboard setup reported the wrong selectable-model count');
    assert.match(dashboard.message, /1 selectable model ready/i,
      'real server: dashboard did not confirm the manual folder save');

    await writeSyntheticTextModel(fixture.modelsRoot, TEXT_MODELS[1]);
    const serverPid = serverProcess.child.pid;
    const refreshedDashboard = await refreshModelsThroughDashboard(browserPath, dashboardUrl, TEXT_MODELS);
    assert.deepEqual(refreshedDashboard.models, TEXT_MODELS,
      'real server: dashboard Refresh did not discover the second synthetic model');
    assert.equal(refreshedDashboard.count, String(TEXT_MODELS.length),
      'real server: dashboard Refresh reported the wrong model count');
    assert.equal(serverProcess.child.pid, serverPid,
      'real server: model refresh unexpectedly replaced the server process');
    assert.equal(hasExited(serverProcess.child), false,
      'real server: model refresh unexpectedly stopped the server');

    const expectedBaseUrl = `http://127.0.0.1:${textPort}/v1`;
    const scenario = {
      name: 'real server legacy saved backend migration',
      accessRequired: false,
      initialState: { settings: { baseUrl: LEGACY_BASE_URL } },
      expectedTextModels: TEXT_MODELS,
      requireImageModels: false,
    };
    const page = await inspectPage(browserPath, `${origin}/#access=${REAL_SERVER_TOKEN}`, scenario);
    assert.deepEqual(optionValues(page.dom, 'modelSelect'), TEXT_MODELS,
      'real server: managed text models did not populate after legacy state migration');
    assert.equal(page.textModelDisabled, false,
      'real server: managed text model selector remained disabled');
    assert.equal(page.selectedTextModel, TEXT_MODELS[0],
      'real server: first discovered text model was not selected');
    assert.equal(page.baseUrlValue, expectedBaseUrl,
      'real server: connection form did not migrate to the managed backend URL');
    assert.equal(page.storedState.settings.baseUrl, expectedBaseUrl,
      'real server: migrated managed backend URL was not persisted');
    assert.equal(page.storedState.settings.baseUrlOverride, false,
      'real server: legacy backend state became an explicit override');
    assert.equal(page.storedAccessToken, REAL_SERVER_TOKEN,
      'real server: access fragment was not retained in tab-scoped storage');
    assert.equal(page.locationHash, '',
      'real server: access fragment was not scrubbed');
    assert.equal(page.localSetupLinkVisible, true,
      'real server: host-local model setup link is not visible');

    const phoneScenario = {
      name: 'real server phone-style authorized model discovery',
      accessRequired: false,
      initialState: { settings: { baseUrl: LEGACY_BASE_URL } },
      expectedTextModels: TEXT_MODELS,
      requireImageModels: false,
      viewport: PHONE_VIEWPORT,
      browserArgs: [
        '--no-proxy-server',
        `--host-resolver-rules=MAP ${SYNTHETIC_PHONE_HOST} 127.0.0.1`,
      ],
    };
    const phoneOrigin = `http://${SYNTHETIC_PHONE_HOST}:${appPort}`;
    const phonePage = await inspectPage(
      browserPath,
      `${phoneOrigin}/#access=${REAL_SERVER_TOKEN}`,
      phoneScenario,
    );
    assert.deepEqual(optionValues(phonePage.dom, 'modelSelect'), TEXT_MODELS,
      'real server: phone-style authorized browser did not load both text models');
    assert.equal(phonePage.textModelDisabled, false,
      'real server: phone-style text model selector remained disabled');
    assert.equal(phonePage.localSetupLinkVisible, false,
      'real server: phone-style browser exposed host-local setup');
    assert.equal(phonePage.localSetupGuidanceVisible, true,
      'real server: phone-style browser omitted remote setup guidance');
    assertAppearance(phonePage.appearance, phoneScenario.name, PHONE_VIEWPORT);

    const savedConfig = JSON.parse(await fs.promises.readFile(fixture.configPath, 'utf8'));
    assert.deepEqual(Object.keys(savedConfig).sort(), ['textModelsRoot', 'textServerExecutable', 'version'],
      'real server: dashboard saved unexpected local configuration fields');
    assert.equal(savedConfig.textModelsRoot, fixture.modelsRoot,
      'real server: dashboard did not persist the manually entered model folder');
    assert.doesNotMatch(serverProcess.getOutput(), /Direct text engine ready/i,
      'real server: browser regression unexpectedly launched the text runtime');
  } finally {
    try {
      await stopRealServer(serverProcess);
    } finally {
      await removeRealServerFixture(fixture.root);
    }
  }
}

async function createRealServerFixture() {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), REAL_SERVER_TEMP_PREFIX));
  try {
    const modelsRoot = path.join(root, 'synthetic-models');
    const runtimePath = path.join(root, process.platform === 'win32'
      ? 'synthetic-text-runtime.exe'
      : 'synthetic-text-runtime');
    const configPath = path.join(root, 'synthetic-local-config.json');
    await fs.promises.mkdir(modelsRoot);
    await fs.promises.writeFile(runtimePath, Buffer.from('synthetic text runtime - browser regression\n'), { mode: 0o700 });
    await writeSyntheticTextModel(modelsRoot, TEXT_MODELS[0]);
    const initialConfig = { version: 1, textServerExecutable: runtimePath };
    await fs.promises.writeFile(configPath, `${JSON.stringify(initialConfig, null, 2)}\n`, { mode: 0o600 });
    assert.deepEqual(Object.keys(JSON.parse(await fs.promises.readFile(configPath, 'utf8'))).sort(),
      ['textServerExecutable', 'version'],
      'real server: initial local config contains more than the synthetic runtime path');
    return { root, modelsRoot, configPath };
  } catch (error) {
    await removeRealServerFixture(root);
    throw error;
  }
}

async function writeSyntheticTextModel(modelsRoot, id) {
  await fs.promises.writeFile(
    path.join(modelsRoot, `${id}.gguf`),
    Buffer.from(`GGUF synthetic ${id}\n`),
  );
}

function startRealServer(configPath, appPort, textPort) {
  const environment = { ...process.env };
  for (const key of Object.keys(environment)) {
    if (key.startsWith('TEXT_')) delete environment[key];
  }
  for (const key of MACHINE_CONFIG_ENVIRONMENT_KEYS) delete environment[key];
  Object.assign(environment, {
    ACCESS_TOKEN: REAL_SERVER_TOKEN,
    ALLOW_LOCAL_BYPASS: '1',
    HOST: '127.0.0.1',
    HTTPS: '0',
    LOCAL_CONFIG_FILE: configPath,
    PORT: String(appPort),
    PRIVATE_DIAGNOSTICS: '0',
    TEXT_PORT: String(textPort),
  });
  const child = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: environment,
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    windowsHide: true,
  });
  let output = '';
  const capture = (chunk) => { output = `${output}${chunk}`.slice(-8_000); };
  child.stdout.on('data', capture);
  child.stderr.on('data', capture);
  child.on('error', capture);
  return { child, getOutput: () => output };
}

async function waitForRealServer(origin, serverProcess) {
  const deadline = Date.now() + REAL_SERVER_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (hasExited(serverProcess.child)) throw new Error('The isolated real server exited during startup.');
    try {
      const response = await fetch(`${origin}/dashboard`, {
        headers: { 'Sec-Fetch-Site': 'same-origin' },
        signal: AbortSignal.timeout(1_000),
      });
      if (response.ok) return;
    } catch {
      // The isolated server may still be binding its loopback port.
    }
    await delay(50);
  }
  throw new Error('Timed out waiting for the isolated real server.');
}

async function stopRealServer(serverProcess) {
  if (!serverProcess || hasExited(serverProcess.child)) return;
  const exited = new Promise((resolve) => serverProcess.child.once('exit', resolve));
  try { serverProcess.child.send({ type: 'shutdown' }); } catch { serverProcess.child.kill(); }
  if (await settlesWithin(exited, 10_000)) return;
  serverProcess.child.kill();
  if (await settlesWithin(exited, 3_000)) return;
  serverProcess.child.kill('SIGKILL');
  await settlesWithin(exited, 2_000);
  if (!hasExited(serverProcess.child)) throw new Error('The isolated real server could not be stopped.');
}

async function removeRealServerFixture(root) {
  const resolvedRoot = path.resolve(root);
  const resolvedTemp = path.resolve(os.tmpdir());
  const safe = path.dirname(resolvedRoot) === resolvedTemp
    && path.basename(resolvedRoot).startsWith(REAL_SERVER_TEMP_PREFIX);
  if (!safe) throw new Error('Refusing to remove an unexpected real-server fixture path.');
  await fs.promises.rm(resolvedRoot, { recursive: true, force: true });
}

function createFixtureServer(requests) {
  const textLoadStates = new Map();
  const chatPromptCounts = new Map();
  return http.createServer(async (request, response) => {
    try {
      const url = new URL(request.url || '/', 'http://fixture.invalid');
      if (FIXTURE_API_PATHS.has(url.pathname)) {
        const body = await readRequestBody(request);
        const token = String(request.headers['x-access-token'] || '');
        requests.push({ path: url.pathname, method: request.method, token, body });
        if (!VALID_TOKENS.has(token)) {
          sendJson(response, 401, { error: 'Access required' }, { 'X-Local-Access-Required': '1' });
          return;
        }
        await serveApi(url.pathname, response, token, body, textLoadStates, chatPromptCounts);
        return;
      }

      if (request.method !== 'GET' && request.method !== 'HEAD') {
        response.writeHead(405, { Allow: 'GET, HEAD' });
        response.end();
        return;
      }

      const relativePath = url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname.slice(1));
      const filePath = path.resolve(PUBLIC_ROOT, relativePath);
      const relativeToPublic = path.relative(PUBLIC_ROOT, filePath);
      if (relativeToPublic.startsWith('..') || path.isAbsolute(relativeToPublic)) {
        response.writeHead(404);
        response.end();
        return;
      }

      const stat = await fs.promises.stat(filePath).catch(() => null);
      if (!stat || !stat.isFile()) {
        response.writeHead(404);
        response.end();
        return;
      }

      response.writeHead(200, {
        'Cache-Control': 'no-store',
        'Content-Type': MIME_TYPES[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
        'X-Content-Type-Options': 'nosniff',
      });
      if (request.method === 'HEAD') {
        response.end();
      } else {
        fs.createReadStream(filePath).pipe(response);
      }
    } catch {
      if (!response.headersSent) response.writeHead(500);
      response.end();
    }
  });
}

async function serveApi(apiPath, response, token, body, textLoadStates, chatPromptCounts) {
  if (apiPath === '/api/config') {
    sendJson(response, 200, {
      defaultBaseUrl: MANAGED_BASE_URL,
      managedTextBackend: { enabled: true },
      localSetupAvailable: token === CURRENT_TOKEN,
    });
    return;
  }

  if (apiPath === '/api/models') {
    sendJson(response, 200, {
      object: 'list',
      data: TEXT_MODELS.map((id) => ({ id, object: 'model' })),
    });
    return;
  }

  if (apiPath === '/api/text/status') {
    sendJson(response, 200, {
      managed: true,
      ...(textLoadStates.get(token) || { state: 'idle', phase: 'idle' }),
    });
    return;
  }

  if (apiPath === '/api/text/load') {
    const payload = parseRequestJson(body);
    if (!TEXT_MODELS.includes(payload.model)) {
      sendJson(response, 400, { error: 'Synthetic model selection is unavailable.' });
      return;
    }
    textLoadStates.set(token, { state: 'loading', phase: 'loading' });
    await delay(600);
    textLoadStates.set(token, { state: 'ready', phase: 'ready' });
    sendJson(response, 200, { managed: true, state: 'ready', phase: 'ready' });
    return;
  }

  if (apiPath === '/api/chat') {
    const payload = parseRequestJson(body);
    sendSyntheticChatStream(response, payload, chatPromptCounts);
    return;
  }

  if (apiPath === '/api/image/generate') {
    const payload = parseRequestJson(body);
    const model = IMAGE_MODELS.includes(payload.model) ? payload.model : IMAGE_MODELS[0];
    sendJson(response, 200, {
      imageBase64: SYNTHETIC_IMAGE_BASE64,
      mimeType: 'image/png',
      model,
      seed: 4242,
      width: 1,
      height: 1,
      steps: payload.steps,
      cfg: payload.cfg,
      sampler: payload.sampler,
      loras: Array.isArray(payload.loras) ? payload.loras : [],
    });
    return;
  }

  sendJson(response, 200, {
    connected: true,
    models: {
      anima: [
        syntheticImageModel(ANIMA_IMAGE_MODELS[0], 'Synthetic Anima'),
        syntheticImageModel(ANIMA_IMAGE_MODELS[1], 'Synthetic Anima Variant'),
      ],
      sdxl: [
        syntheticImageModel(SDXL_IMAGE_MODELS[0], 'Synthetic SDXL'),
        syntheticImageModel(SDXL_IMAGE_MODELS[1], 'Synthetic SDXL Variant'),
      ],
    },
    loras: { anima: [], sdxl: [] },
    runtime: {},
    dependencies: { animaTextEncoder: true, animaVae: true },
  });
}

function parseRequestJson(value) {
  try { return JSON.parse(value || '{}'); } catch { return {}; }
}

function sendSyntheticChatStream(response, payload, chatPromptCounts) {
  const messages = Array.isArray(payload.messages) ? payload.messages : [];
  const lastUser = [...messages].reverse().find((message) => message.role === 'user');
  const hasSyntheticToolResult = messages.some((message) => (
    message.role === 'tool' && message.tool_call_id === SYNTHETIC_TOOL_CALL_ID
  ));
  const prompt = typeof lastUser?.content === 'string' ? lastUser.content : '';
  const count = (chatPromptCounts.get(prompt) || 0) + 1;
  chatPromptCounts.set(prompt, count);

  let delta;
  if (payload.model !== TEXT_MODELS[1]) {
    delta = { content: 'Synthetic fixture received the wrong model.' };
  } else if (prompt === CHAT_IMAGE_TOOL.prompt && !hasSyntheticToolResult) {
    delta = {
      tool_calls: [{
        index: 0,
        id: SYNTHETIC_TOOL_CALL_ID,
        type: 'function',
        function: {
          name: 'generate_image',
          arguments: JSON.stringify({ prompt: CHAT_IMAGE_TOOL.imagePrompt, orientation: 'landscape' }),
        },
      }],
    };
  } else {
    const content = prompt === CHAT_TURN_ONE.prompt ? CHAT_TURN_ONE.response
      : prompt === CHAT_TURN_TWO.prompt ? CHAT_TURN_TWO.response
        : prompt === CHAT_IMAGE_TOOL.prompt && hasSyntheticToolResult ? CHAT_IMAGE_TOOL.response
          : prompt === CHAT_AFTER_IMAGE.prompt && count > 1 ? CHAT_AFTER_IMAGE.regeneratedResponse
            : prompt === CHAT_AFTER_IMAGE.prompt ? CHAT_AFTER_IMAGE.response
              : prompt === CHAT_EDITED_TURN.prompt ? CHAT_EDITED_TURN.response
                : 'Synthetic fixture received an unexpected conversation turn.';
    delta = { content };
  }
  const events = [
    `data: ${JSON.stringify({ choices: [{ delta }] })}\n\n`,
    'data: [DONE]\n\n',
  ];
  response.writeHead(200, {
    'Cache-Control': 'no-store',
    'Content-Type': 'text/event-stream; charset=utf-8',
  });
  response.end(events.join(''));
}

function syntheticImageModel(id, label) {
  return {
    id,
    label,
    recommendedSteps: 4,
    recommendedCfg: 1,
  };
}

function sendJson(response, statusCode, value, headers = {}) {
  const body = JSON.stringify(value);
  response.writeHead(statusCode, {
    'Cache-Control': 'no-store',
    'Content-Length': Buffer.byteLength(body),
    'Content-Type': 'application/json; charset=utf-8',
    ...headers,
  });
  response.end(body);
}

async function readRequestBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf8');
}

function assertScenario(page, scenario, requests) {
  const { dom, viewport } = page;
  const expectedViewport = scenario.viewport || PHONE_VIEWPORT;
  assert.deepEqual(viewport, { width: expectedViewport.width, height: expectedViewport.height },
    `${scenario.name}: browser did not use the expected viewport`);
  assert.match(dom, /<meta\b(?=[^>]*\bname=["']viewport["'])(?=[^>]*\bcontent=["'][^"']*width=device-width)[^>]*>/i,
    `${scenario.name}: responsive viewport metadata is missing`);

  const accessTag = findOpeningTagById(dom, ACCESS_REQUIRED_ID);
  assert.ok(accessTag, `${scenario.name}: #${ACCESS_REQUIRED_ID} is missing`);
  const accessHidden = /\shidden(?:\s|=|>)/i.test(accessTag);
  assert.equal(accessHidden, !scenario.accessRequired,
    `${scenario.name}: access-required state visibility is incorrect`);
  assert.equal(page.appearance.accessDisplay, scenario.accessRequired ? 'grid' : 'none',
    `${scenario.name}: access-required state is not visually enforced`);
  assert.equal(page.locationHash, '', `${scenario.name}: the access fragment was not scrubbed`);

  const textOptions = optionValues(dom, 'modelSelect');
  const imageOptions = optionValues(dom, 'imageModelSelect');
  const apiRequests = requests.filter((entry) => API_PATHS.has(entry.path));
  if (!scenario.accessRequired) {
    assertAppearance(page.appearance, scenario.name, expectedViewport);
    assert.ok(apiRequests.length > 0, `${scenario.name}: frontend made no API requests`);
    assert.deepEqual(textOptions, TEXT_MODELS, `${scenario.name}: text models did not populate`);
    assert.equal(page.textModelDisabled, false, `${scenario.name}: discovered text models are not selectable`);
    assert.equal(page.selectedTextModel, scenario.exerciseChat ? TEXT_MODELS[1] : TEXT_MODELS[0],
      `${scenario.name}: the expected discovered text model was not selected`);
    assert.deepEqual(imageOptions, ANIMA_IMAGE_MODELS, `${scenario.name}: image models did not populate`);
    assert.deepEqual(page.imageOptionsByKind, {
      anima: ANIMA_IMAGE_MODELS,
      sdxl: SDXL_IMAGE_MODELS,
    }, `${scenario.name}: switching image families did not expose both model lists`);
    assert.deepEqual(page.samplerOptionsByKind, {
      anima: ['flow_euler', 'flow_heun'],
      sdxl: ['dpmpp_sde_karras', 'euler_ancestral_karras', 'euler_karras', 'dpmpp_2m_karras'],
    }, `${scenario.name}: switching image families did not expose the compatible sampler lists`);
    assert.deepEqual(new Set(apiRequests.map((entry) => entry.path)), API_PATHS,
      `${scenario.name}: frontend did not cover all model/config endpoints`);
    assert.ok(apiRequests.every((entry) => entry.token === scenario.accessToken),
      `${scenario.name}: an API request omitted the current access token`);
    assert.equal(page.storedAccessToken, scenario.accessToken,
      `${scenario.name}: the current token was not retained in tab-scoped storage`);
    assert.equal(page.baseUrlValue, scenario.expectedBaseUrl,
      `${scenario.name}: the connection form has the wrong backend URL`);
    assert.equal(page.storedState.settings.baseUrl, scenario.expectedBaseUrl,
      `${scenario.name}: the persisted backend URL has the wrong migration result`);
    assert.equal(page.storedState.settings.baseUrlOverride, scenario.expectedBaseUrlOverride,
      `${scenario.name}: the explicit backend override flag has the wrong migration result`);
    const modelRequest = apiRequests.find((entry) => entry.path === '/api/models');
    assert.ok(modelRequest, `${scenario.name}: frontend did not discover text models`);
    assert.equal(JSON.parse(modelRequest.body).baseUrl, scenario.expectedBaseUrl,
      `${scenario.name}: text discovery used the wrong backend URL`);
    assert.equal(page.localSetupLinkVisible, scenario.localSetupAvailable,
      `${scenario.name}: host-local model setup link visibility is incorrect`);
    assert.equal(page.localSetupGuidanceVisible, !scenario.localSetupAvailable,
      `${scenario.name}: remote model setup guidance visibility is incorrect`);
    assert.match(page.localSetupLinkHref, /\/dashboard#models$/,
      `${scenario.name}: local model setup link targets the wrong dashboard section`);
    if (!scenario.localSetupAvailable) {
      assert.match(page.localSetupGuidanceText, /computer running this server/i,
        `${scenario.name}: remote setup guidance is unclear`);
    }
    if (scenario.expectedReplacementModel) {
      assert.equal(page.storedState.settings.model, scenario.expectedReplacementModel,
        `${scenario.name}: unavailable saved model was not replaced and persisted`);
      assert.notEqual(page.storedState.settings.model, STALE_TEXT_MODEL,
        `${scenario.name}: stale saved model survived discovery`);
    }
    if (scenario.exerciseChat) {
      const chatRequests = requests.filter((entry) => entry.path === '/api/chat');
      const imageRequests = requests.filter((entry) => entry.path === '/api/image/generate');
      const loadRequests = requests.filter((entry) => entry.path === '/api/text/load');
      const statusRequests = requests.filter((entry) => entry.path === '/api/text/status');
      assert.equal(chatRequests.length, 7,
        `${scenario.name}: full mixed conversation produced the wrong chat request count`);
      assert.equal(imageRequests.length, 2,
        `${scenario.name}: assistant-tool and Image-studio flows did not both generate an image`);
      assert.ok(loadRequests.length >= 8,
        `${scenario.name}: model selection and every chat round did not verify managed readiness`);
      assert.ok(statusRequests.length >= 2,
        `${scenario.name}: the loading bar did not poll observable backend status`);
      assert.ok(loadRequests.every((entry) => entry.token === scenario.accessToken),
        `${scenario.name}: a model-load request omitted the current access token`);
      assert.ok(loadRequests.every((entry) => parseRequestJson(entry.body).model === TEXT_MODELS[1]),
        `${scenario.name}: a model-load request used the wrong selected model`);
      assert.ok(chatRequests.every((entry) => entry.token === scenario.accessToken),
        `${scenario.name}: a chat request omitted the current access token`);
      const chatPayloads = chatRequests.map((entry) => parseRequestJson(entry.body));
      assert.ok(chatPayloads.every((payload) => payload.model === TEXT_MODELS[1]),
        `${scenario.name}: a chat payload did not use the persisted second model`);
      assertTextTurnHistory(chatPayloads, scenario.name);
      assertMixedMediaHistory(chatPayloads, imageRequests, scenario.name, scenario.accessToken);
      assert.equal(page.chatExercise?.selectedModel, TEXT_MODELS[1],
        `${scenario.name}: second model was not restored after reload`);
      for (const expected of [
        CHAT_TURN_ONE.response,
        CHAT_TURN_TWO.response,
        CHAT_IMAGE_TOOL.response,
        CHAT_EDITED_TURN.response,
      ]) {
        assert.match(page.chatExercise?.assistantText || '', new RegExp(escapeRegExp(expected)),
          `${scenario.name}: an expected multi-turn response was not rendered`);
      }
      assert.equal(page.chatExercise?.selectionLoad?.visible, true,
        `${scenario.name}: switching models did not reveal the loading bar`);
      assert.equal(page.chatExercise?.selectionLoad?.indeterminate, true,
        `${scenario.name}: switching models did not use an honest indeterminate progress state`);
      assert.equal(page.chatExercise?.selectionLoad?.modelDisabled, true,
        `${scenario.name}: the selector remained interactive during a model handoff`);
      assert.equal(page.chatExercise?.selectionReady?.complete, true,
        `${scenario.name}: the loading bar never reached its ready state`);
      assert.equal(page.chatExercise?.chatLoad?.visible, true,
        `${scenario.name}: chat did not expose model readiness work before streaming`);
      assert.equal(page.storedState.settings.model, TEXT_MODELS[1],
        `${scenario.name}: second model selection was not retained after chat`);
      assertConversationLifecycle(page.chatExercise, scenario.name);
    }
    return;
  }

  assert.equal(page.storedAccessToken, '',
    `${scenario.name}: a missing or stale token remained in tab-scoped storage`);
  assert.equal(page.localSetupLinkVisible, false,
    `${scenario.name}: host-local setup was offered without authenticated config`);
  assert.equal(page.localSetupGuidanceVisible, true,
    `${scenario.name}: unauthenticated setup guidance is missing`);

  assert.ok(!textOptions.some((value) => TEXT_MODELS.includes(value)),
    `${scenario.name}: text models populated without valid access`);
  assert.ok(!imageOptions.some((value) => IMAGE_MODELS.includes(value)),
    `${scenario.name}: image models populated without valid access`);
  assert.doesNotMatch(dom, /synthetic-(?:text|image)-/,
    `${scenario.name}: protected model metadata leaked into the page`);

  if (scenario.fragment) {
    assert.ok(apiRequests.some((entry) => entry.token === STALE_TOKEN),
      `${scenario.name}: stale access was not rejected by the API fixture`);
    assert.ok(apiRequests.every((entry) => entry.token === STALE_TOKEN || entry.token === ''),
      `${scenario.name}: frontend sent an unexpected access credential`);
  } else {
    assert.ok(apiRequests.every((entry) => entry.token === ''),
      `${scenario.name}: frontend sent an unexpected access credential`);
  }
}

function assertTextTurnHistory(payloads, scenarioName) {
  const expectedUserTurns = [
    [CHAT_TURN_ONE.prompt],
    [CHAT_TURN_ONE.prompt, CHAT_TURN_TWO.prompt],
    [CHAT_TURN_ONE.prompt, CHAT_TURN_TWO.prompt, CHAT_IMAGE_TOOL.prompt],
    [CHAT_TURN_ONE.prompt, CHAT_TURN_TWO.prompt, CHAT_IMAGE_TOOL.prompt],
    [CHAT_TURN_ONE.prompt, CHAT_TURN_TWO.prompt, CHAT_IMAGE_TOOL.prompt, CHAT_AFTER_IMAGE.prompt],
    [CHAT_TURN_ONE.prompt, CHAT_TURN_TWO.prompt, CHAT_IMAGE_TOOL.prompt, CHAT_AFTER_IMAGE.prompt],
    [CHAT_TURN_ONE.prompt, CHAT_TURN_TWO.prompt, CHAT_IMAGE_TOOL.prompt, CHAT_EDITED_TURN.prompt],
  ];
  payloads.forEach((payload, index) => {
    assert.equal(payload.messages?.[0]?.role, 'system',
      `${scenarioName}: chat round ${index + 1} lost the system prompt`);
    assert.deepEqual(
      (payload.messages || []).filter((message) => message.role === 'user').map((message) => message.content),
      expectedUserTurns[index],
      `${scenarioName}: chat round ${index + 1} sent the wrong ordered user history`,
    );
  });
  assert.ok(payloads[1].messages.some((message) => (
    message.role === 'assistant' && message.content === CHAT_TURN_ONE.response
  )), `${scenarioName}: the second turn omitted the first assistant answer`);
  assert.ok(payloads[2].messages.some((message) => (
    message.role === 'assistant' && message.content === CHAT_TURN_TWO.response
  )), `${scenarioName}: the third turn omitted the second assistant answer`);
  assert.ok(!payloads[6].messages.some((message) => (
    message.content === CHAT_AFTER_IMAGE.prompt
      || message.content === CHAT_AFTER_IMAGE.response
      || message.content === CHAT_AFTER_IMAGE.regeneratedResponse
  )), `${scenarioName}: editing did not truncate the replaced user turn and its answer from model context`);
}

function assertMixedMediaHistory(payloads, imageRequests, scenarioName, accessToken) {
  const toolFollowUp = payloads[3].messages || [];
  const toolCall = toolFollowUp.find((message) => (
    message.role === 'assistant'
      && message.tool_calls?.[0]?.id === SYNTHETIC_TOOL_CALL_ID
      && message.tool_calls?.[0]?.function?.name === 'generate_image'
  ));
  const toolResult = toolFollowUp.find((message) => (
    message.role === 'tool' && message.tool_call_id === SYNTHETIC_TOOL_CALL_ID
  ));
  assert.ok(toolCall, `${scenarioName}: assistant image-tool call was not retained in follow-up context`);
  assert.ok(toolResult, `${scenarioName}: generated image result was not returned to the text model`);
  assert.deepEqual(JSON.parse(toolCall.tool_calls[0].function.arguments), {
    prompt: CHAT_IMAGE_TOOL.imagePrompt,
    orientation: 'landscape',
  }, `${scenarioName}: assistant image-tool arguments changed before follow-up`);
  assert.equal(JSON.parse(toolResult.content).ok, true,
    `${scenarioName}: successful image generation was not reported as a successful tool result`);

  const mixedTextPayloads = payloads.slice(4);
  assert.ok(mixedTextPayloads.every((payload) => !(payload.messages || []).some((message) => (
    message.content === MANUAL_IMAGE_PROMPT || message.role === 'image'
  ))), `${scenarioName}: manual image cards leaked into the text-only backend history`);
  assert.ok(mixedTextPayloads.every((payload) => (payload.messages || []).some((message) => (
    message.role === 'tool' && message.tool_call_id === SYNTHETIC_TOOL_CALL_ID
  ))), `${scenarioName}: mixed follow-up lost the earlier assistant image-tool result`);

  const imagePayloads = imageRequests.map((entry) => parseRequestJson(entry.body));
  assert.deepEqual(imagePayloads.map((payload) => payload.prompt), [
    CHAT_IMAGE_TOOL.imagePrompt,
    MANUAL_IMAGE_PROMPT,
  ], `${scenarioName}: image-tool and Image-studio prompts reached the backend in the wrong order`);
  assert.equal(imagePayloads[0].sampler, 'flow_euler',
    `${scenarioName}: assistant image tool did not use the persisted Anima sampler`);
  assert.equal(imagePayloads[1].kind, 'sdxl',
    `${scenarioName}: manual image generation did not switch model families`);
  assert.equal(imagePayloads[1].model, SDXL_IMAGE_MODELS[1],
    `${scenarioName}: manual image generation did not use the switched image model`);
  assert.equal(imagePayloads[1].steps, 5,
    `${scenarioName}: image model switching reset the chosen step count`);
  assert.equal(imagePayloads[1].cfg, 123.45,
    `${scenarioName}: image model switching reset the chosen CFG`);
  assert.equal(imagePayloads[1].sampler, 'dpmpp_sde_karras',
    `${scenarioName}: the preferred single-step-history SDXL sampler did not reach generation`);
  assert.equal(imagePayloads[1].negativePrompt, EXPECTED_AUTO_NEGATIVE,
    `${scenarioName}: automatic negative prompt did not reach image generation unchanged`);
  assert.ok(imageRequests.every((entry) => entry.token === accessToken),
    `${scenarioName}: an image-generation request omitted current access`);
}

function assertConversationLifecycle(exercise, scenarioName) {
  assert.deepEqual(exercise.imageControls, {
    generated: true,
    kind: 'sdxl',
    model: SDXL_IMAGE_MODELS[1],
    steps: '5',
    cfg: '123.45',
    sampler: 'dpmpp_sde_karras',
    negativePrompt: EXPECTED_AUTO_NEGATIVE,
    scrollBefore: 0,
  }, `${scenarioName}: image controls did not survive the checkpoint switch`);
  assert.equal(exercise.imageGallery?.itemCount, 2,
    `${scenarioName}: gallery did not include every image in the active conversation`);
  assert.equal(exercise.imageGallery?.galleryScrollable, true,
    `${scenarioName}: gallery viewport did not allow scrolling through the conversation images`);
  assert.ok(exercise.imageGallery?.initialGalleryScroll > 0,
    `${scenarioName}: gallery did not open at the clicked conversation image`);
  assert.ok(exercise.imageGallery?.galleryScroll < exercise.imageGallery?.initialGalleryScroll,
    `${scenarioName}: gallery did not scroll from the clicked image back through the conversation`);
  assert.ok(Math.abs(exercise.imageGallery?.scrollAfter - exercise.imageGallery?.scrollBefore) <= 1,
    `${scenarioName}: image generation or gallery interaction moved the conversation viewport`);
  for (const key of ['mixedBeforeReload', 'mixedAfterReload']) {
    const snapshot = exercise?.[key] || {};
    assert.equal(snapshot.messageCount, 11,
      `${scenarioName}: ${key} stored the wrong mixed-conversation message count`);
    assert.equal(snapshot.visibleMessageCount, 9,
      `${scenarioName}: ${key} rendered hidden tool protocol messages as chat cards`);
    assert.equal(snapshot.imageCount, 2,
      `${scenarioName}: ${key} did not retain both generated image messages`);
    assert.equal(snapshot.generatedImageCount, 2,
      `${scenarioName}: ${key} did not hydrate both generated image blobs`);
    assert.equal(snapshot.kinds.filter((kind) => kind === 'tool-result').length, 1,
      `${scenarioName}: ${key} lost the image tool result`);
    assert.equal(snapshot.kinds.filter((kind) => kind === 'image-prompt').length, 1,
      `${scenarioName}: ${key} lost the manual Image-studio prompt card`);
  }
  assert.deepEqual(exercise.mixedAfterReload.kinds, exercise.mixedBeforeReload.kinds,
    `${scenarioName}: mixed message types changed across reload`);
  assert.deepEqual(exercise.mixedAfterReload.contents, exercise.mixedBeforeReload.contents,
    `${scenarioName}: mixed message content changed across reload`);
  assert.equal(exercise.mixedAfterReload.imageSettings?.modelByKind?.sdxl, SDXL_IMAGE_MODELS[1],
    `${scenarioName}: switched image model was not retained across reload`);
  assert.equal(exercise.mixedAfterReload.imageSettings?.stepsByKind?.sdxl, '5',
    `${scenarioName}: chosen image steps were not retained across reload`);
  assert.equal(exercise.mixedAfterReload.imageSettings?.cfgByKind?.sdxl, '123.45',
    `${scenarioName}: chosen image CFG was not retained across reload`);
  assert.equal(exercise.mixedAfterReload.imageSettings?.samplerByKind?.sdxl, 'dpmpp_sde_karras',
    `${scenarioName}: chosen image sampler was not retained across reload`);
  assert.equal(exercise.mixedAfterReload.imageSettings?.autoNegativeEvery, 2,
    `${scenarioName}: auto-negative interval was not retained across reload`);
  assert.equal(exercise.mixedAfterReload.imageSettings?.negativePrompt, EXPECTED_AUTO_NEGATIVE,
    `${scenarioName}: generated negative prompt was not retained across reload`);
  assert.equal(exercise.afterRegenerate?.messageCount, 13,
    `${scenarioName}: regeneration appended instead of replacing the last answer`);
  assert.ok(exercise.afterRegenerate?.contents.includes(CHAT_AFTER_IMAGE.regeneratedResponse),
    `${scenarioName}: regenerated answer was not persisted`);
  assert.ok(!exercise.afterRegenerate?.contents.includes(CHAT_AFTER_IMAGE.response),
    `${scenarioName}: replaced answer remained after regeneration`);
  assert.equal(exercise.afterEdit?.messageCount, 11,
    `${scenarioName}: editing did not truncate from the selected user turn`);
  assert.equal(exercise.afterEdit?.promptValue, CHAT_AFTER_IMAGE.prompt,
    `${scenarioName}: editing did not restore the selected user text to the composer`);
  assert.equal(exercise.afterEditedTurn?.messageCount, 13,
    `${scenarioName}: edited follow-up did not rebuild the mixed conversation cleanly`);
  assert.ok(exercise.afterEditedTurn?.contents.includes(CHAT_EDITED_TURN.response),
    `${scenarioName}: edited follow-up answer was not persisted`);
  assert.deepEqual(exercise.conversationIsolation?.created, {
    title: 'New chat', conversationCount: 2, messageCount: 0,
  }, `${scenarioName}: new conversation was not isolated`);
  assert.deepEqual(exercise.conversationIsolation?.switchedAndDeleted, {
    title: RENAMED_CONVERSATION, conversationCount: 1, imageCount: 2,
  }, `${scenarioName}: switching/deleting conversations damaged the mixed chat`);
  assert.equal(exercise.afterClear?.title, RENAMED_CONVERSATION,
    `${scenarioName}: clear unexpectedly renamed the conversation`);
  assert.equal(exercise.afterClear?.messageCount, 0,
    `${scenarioName}: clear left persisted messages behind`);
  assert.equal(exercise.afterClear?.visibleMessageCount, 0,
    `${scenarioName}: clear left visible message cards behind`);
  assert.equal(exercise.afterClear?.conversationCount, 1,
    `${scenarioName}: clear removed the conversation container`);
}

function assertAppearance(appearance, scenarioName, viewport) {
  assert.ok(appearance.stylesheets >= 1, `${scenarioName}: the recovered stylesheet did not load`);
  assert.equal(appearance.bodyBackgroundColor, 'rgb(0, 128, 128)',
    `${scenarioName}: the recovered teal desktop background is missing`);
  assert.match(appearance.bodyBackgroundImage, /repeating-conic-gradient/i,
    `${scenarioName}: the recovered patterned desktop background is missing`);
  assert.match(appearance.bodyFontFamily, /Tahoma|MS Sans Serif|Segoe UI/i,
    `${scenarioName}: the recovered desktop font stack is missing`);
  assert.equal(appearance.bodyOverflowX, 'hidden', `${scenarioName}: body horizontal overflow regressed`);
  assert.equal(appearance.bodyOverflowY, 'hidden', `${scenarioName}: body vertical overflow regressed`);
  assert.equal(appearance.appDisplay, 'grid', `${scenarioName}: application grid is missing`);
  assert.equal(appearance.appWidth, viewport.width, `${scenarioName}: application width does not fill its viewport`);
  assert.equal(appearance.appHeight, viewport.height, `${scenarioName}: application height does not fill its viewport`);
  assert.match(appearance.topbarBackgroundImage,
    /rgb\(23,\s*23,\s*168\).*rgb\(0,\s*0,\s*128\)/,
    `${scenarioName}: the recovered navy titlebar is missing`);
  assert.equal(appearance.topbarBorderRadius, '0px',
    `${scenarioName}: the recovered square desktop geometry is missing`);
  assert.equal(appearance.chatDisplay, 'grid', `${scenarioName}: chat layout is not visible`);
  assert.equal(appearance.chatBackgroundColor, 'rgb(192, 192, 192)',
    `${scenarioName}: chat window surface reverted`);
  assert.notEqual(appearance.chatBoxShadow, 'none', `${scenarioName}: chat window bevel/shadow is missing`);
  assert.equal(appearance.messagesOverflowY, 'auto', `${scenarioName}: message viewport is no longer scrollable`);
  assert.match(appearance.messagesBackgroundImage, /repeating-conic-gradient/i,
    `${scenarioName}: message-workspace pattern is missing`);
  assert.equal(appearance.composerBackgroundColor, 'rgb(192, 192, 192)',
    `${scenarioName}: composer surface reverted`);
  assert.ok(appearance.promptHeight >= 46, `${scenarioName}: prompt control is too short`);
  assert.ok(appearance.sendHeight >= 46, `${scenarioName}: send control is too short`);
  assert.ok(appearance.composerHeight <= 66,
    `${scenarioName}: composer consumes too much conversation height`);
  assert.ok(
    appearance.messagesHeight >= appearance.chatHeight
      - appearance.topbarHeight
      - appearance.modelLoadingHeight
      - appearance.composerHeight
      - 10,
    `${scenarioName}: conversation viewport does not receive the remaining chat height`,
  );
  assert.ok(Math.abs(appearance.messagesBottom - appearance.composerTop) <= 2,
    `${scenarioName}: composer is not directly below the conversation viewport`);
  assert.ok(Math.abs(appearance.composerBottom - (appearance.chatBottom - 4)) <= 2,
    `${scenarioName}: composer is no longer anchored to the chat window bottom`);

  if (viewport.mobile) {
    assert.equal(appearance.appPaddingLeft, 5, `${scenarioName}: phone shell padding reverted`);
    assert.ok(appearance.chatWidth >= viewport.width - 12 && appearance.chatWidth <= viewport.width - 8,
      `${scenarioName}: chat layout does not fit the phone width`);
    assert.ok(appearance.chatHeight >= viewport.height - 12 && appearance.chatHeight <= viewport.height - 8,
      `${scenarioName}: chat layout does not fill the phone height`);
    assert.ok(appearance.chatLeft >= 4 && appearance.chatLeft <= 6,
      `${scenarioName}: phone chat window is horizontally displaced`);
    assert.equal(appearance.sidebarPosition, 'fixed', `${scenarioName}: phone sidebar is not an overlay drawer`);
    assert.ok(appearance.sidebarLeft < -300, `${scenarioName}: closed phone drawer remains onscreen`);
    assert.ok(appearance.sidebarWidth >= 330 && appearance.sidebarWidth <= 345,
      `${scenarioName}: phone drawer width regressed`);
    assert.equal(appearance.mobileMenuDisplay, 'grid',
      `${scenarioName}: mobile navigation did not activate at the phone viewport`);
    assert.ok(appearance.topbarHeight >= 58 && appearance.topbarHeight <= 64,
      `${scenarioName}: phone titlebar height regressed`);
    assert.ok(appearance.composerHeight <= 60,
      `${scenarioName}: phone composer is not compact`);
    return;
  }

  assert.equal(appearance.appPaddingLeft, 12, `${scenarioName}: desktop shell padding reverted`);
  assert.match(appearance.appGridColumns, /^292px\s+/, `${scenarioName}: desktop sidebar grid track regressed`);
  assert.equal(appearance.sidebarPosition, 'static', `${scenarioName}: desktop sidebar became an overlay`);
  assert.equal(appearance.sidebarDisplay, 'flex', `${scenarioName}: desktop sidebar is hidden`);
  assert.ok(appearance.sidebarWidth >= 290 && appearance.sidebarWidth <= 294,
    `${scenarioName}: desktop sidebar width regressed`);
  assert.ok(appearance.sidebarHeight >= viewport.height - 26 && appearance.sidebarHeight <= viewport.height - 22,
    `${scenarioName}: desktop sidebar height regressed`);
  assert.ok(appearance.chatLeft - appearance.sidebarRight >= 8
    && appearance.chatLeft - appearance.sidebarRight <= 12,
  `${scenarioName}: desktop window gap regressed`);
  assert.ok(appearance.chatWidth > 1000, `${scenarioName}: desktop chat workspace collapsed`);
  assert.ok(appearance.chatHeight >= viewport.height - 26 && appearance.chatHeight <= viewport.height - 22,
    `${scenarioName}: desktop chat height regressed`);
  assert.equal(appearance.mobileMenuDisplay, 'none',
    `${scenarioName}: mobile navigation leaked into desktop layout`);
  assert.ok(appearance.topbarHeight >= 60 && appearance.topbarHeight <= 64,
    `${scenarioName}: desktop titlebar height regressed`);
}

function findOpeningTagById(html, id) {
  const escapedId = escapeRegExp(id);
  const match = html.match(new RegExp(`<[^>]+\\sid=["']${escapedId}["'][^>]*>`, 'i'));
  return match ? match[0] : '';
}

function optionValues(html, selectId) {
  const escapedId = escapeRegExp(selectId);
  const select = html.match(new RegExp(`<select[^>]+id=["']${escapedId}["'][^>]*>([\\s\\S]*?)<\\/select>`, 'i'));
  assert.ok(select, `#${selectId} is missing from browser DOM`);
  return [...select[1].matchAll(/<option\b[^>]*\bvalue=["']([^"']*)["'][^>]*>/gi)]
    .map((match) => decodeHtmlAttribute(match[1]));
}

function decodeHtmlAttribute(value) {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function inspectPage(browserPath, url, scenario) {
  return withHeadlessBrowser(
    browserPath,
    (cdp) => inspectWithCdp(cdp, url, scenario),
    scenario.browserArgs || [],
  );
}

async function withHeadlessBrowser(browserPath, operation, extraArguments = []) {
  const profilePath = path.join(os.tmpdir(), `local-ai-browser-smoke-${process.pid}-${crypto.randomUUID()}`);
  await fs.promises.mkdir(profilePath, { recursive: true });
  let browser = null;
  let cdp = null;
  try {
    const args = [
      '--headless',
      '--disable-background-networking',
      '--disable-component-update',
      '--disable-default-apps',
      '--disable-extensions',
      '--disable-features=msEdgeSidebarV2,Translate',
      '--disable-gpu',
      '--disable-sync',
      '--hide-scrollbars',
      '--metrics-recording-only',
      '--mute-audio',
      '--no-default-browser-check',
      '--no-first-run',
      '--remote-allow-origins=*',
      '--remote-debugging-port=0',
      `--user-data-dir=${profilePath}`,
      '--window-size=390,844',
      ...extraArguments,
      'about:blank',
    ];
    if (typeof process.getuid === 'function' && process.getuid() === 0) args.unshift('--no-sandbox');
    browser = launchBrowser(browserPath, args);
    const endpoint = await waitForDevToolsEndpoint(profilePath, browser);
    cdp = await CdpWebSocket.connect(endpoint);
    return await operation(cdp);
  } catch (error) {
    if (process.env.BROWSER_SMOKE_DEBUG === '1' && browser) {
      console.error(`Browser process exit: ${browser.child.exitCode === null ? 'running' : browser.child.exitCode}`);
      if (browser.getStderr()) console.error(safeDiagnostic(browser.getStderr()));
    }
    throw error;
  } finally {
    if (cdp) {
      await cdp.send('Browser.close').catch(() => {});
      cdp.close();
    }
    if (browser) await stopBrowserProcess(browser.child);
    await removeBrowserProfile(profilePath);
  }
}

async function stopBrowserProcess(child) {
  if (!child || hasExited(child)) return;
  const exited = new Promise((resolve) => child.once('exit', resolve));
  if (process.platform === 'win32') {
    const killer = spawn('taskkill', ['/pid', String(child.pid), '/t', '/f'], {
      stdio: 'ignore',
      windowsHide: true,
    });
    await new Promise((resolve) => killer.once('exit', resolve));
    await settlesWithin(exited, 5_000);
    if (!hasExited(child)) throw new Error('The headless browser process tree could not be stopped.');
    return;
  }
  if (!child.killed) child.kill();
  if (await settlesWithin(exited, 5_000)) return;
  child.kill('SIGKILL');
  await settlesWithin(exited, 2_000);
  if (!hasExited(child)) throw new Error('The headless browser process could not be stopped.');
}

function launchBrowser(browserPath, args) {
  const child = spawn(browserPath, args, {
    stdio: ['ignore', 'ignore', 'pipe'],
    windowsHide: true,
  });
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => { stderr = `${stderr}${chunk}`.slice(-4_000); });
  child.on('error', () => {});
  return { child, getStderr: () => stderr };
}

async function waitForDevToolsEndpoint(profilePath, browser) {
  const portFile = path.join(profilePath, 'DevToolsActivePort');
  const deadline = Date.now() + BROWSER_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const value = await fs.promises.readFile(portFile, 'utf8').catch(() => '');
    const [portText, socketPath] = value.split(/\r?\n/);
    const port = Number.parseInt(portText, 10);
    if (Number.isInteger(port) && port > 0 && socketPath && socketPath.startsWith('/devtools/browser/')) {
      return `ws://127.0.0.1:${port}${socketPath}`;
    }
    await delay(50);
  }
  if (process.env.BROWSER_SMOKE_DEBUG === '1' && browser.getStderr()) {
    console.error(safeDiagnostic(browser.getStderr()));
  }
  throw new Error('Headless browser did not expose its local debugging endpoint.');
}

async function configureModelsThroughDashboard(browserPath, url, modelsRoot, expectedModels) {
  return withHeadlessBrowser(
    browserPath,
    (cdp) => configureModelsWithCdp(cdp, url, modelsRoot, expectedModels),
  );
}

async function configureModelsWithCdp(cdp, url, modelsRoot, expectedModels) {
  const target = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const attached = await cdp.send('Target.attachToTarget', { targetId: target.targetId, flatten: true });
  const sessionId = attached.sessionId;
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Runtime.enable', {}, sessionId);
  await cdp.send('Page.navigate', { url }, sessionId);

  await waitForBrowserExpression(cdp, sessionId, `(() => {
    if (document.readyState !== 'complete') return false;
    const input = document.getElementById('textModelsRootInput');
    const save = document.getElementById('saveTextModelsRootButton');
    const runtime = document.getElementById('textRuntimeStatus');
    return Boolean(input && save && !save.disabled && runtime && runtime.textContent.trim() === 'Configured');
  })()`, 'The real dashboard did not become ready for manual model-folder setup.');

  const submitted = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const input = document.getElementById('textModelsRootInput');
      const save = document.getElementById('saveTextModelsRootButton');
      if (!input || !save || save.disabled) return false;
      input.value = ${JSON.stringify(modelsRoot)};
      input.dispatchEvent(new Event('input', { bubbles: true }));
      save.click();
      return true;
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.equal(submitted.result?.value, true,
    'real server: dashboard manual folder form could not be submitted');

  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const count = document.getElementById('textModelCount');
    const badge = document.getElementById('modelSetupBadge');
    const models = Array.from(document.querySelectorAll('#textModelList li'), item => item.textContent.trim());
    return count && count.textContent.trim() === ${JSON.stringify(String(expectedModels.length))}
      && badge && badge.textContent.trim() === 'Ready'
      && ${JSON.stringify(expectedModels)}.every(model => models.includes(model));
  })()`, 'The real dashboard did not finish scanning the manually entered model folder.');

  const result = await cdp.send('Runtime.evaluate', {
    expression: `(() => ({
      count: document.getElementById('textModelCount').textContent.trim(),
      message: document.getElementById('modelSetupMessage').textContent.trim(),
      models: Array.from(document.querySelectorAll('#textModelList li'), item => item.textContent.trim()),
    }))()`,
    returnByValue: true,
  }, sessionId);
  return result.result?.value || {};
}

async function refreshModelsThroughDashboard(browserPath, url, expectedModels) {
  return withHeadlessBrowser(browserPath, async (cdp) => {
    const target = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const attached = await cdp.send('Target.attachToTarget', { targetId: target.targetId, flatten: true });
    const sessionId = attached.sessionId;
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send('Page.navigate', { url }, sessionId);
    await waitForBrowserExpression(cdp, sessionId, `(() => {
      const refresh = document.getElementById('refreshModelSetupButton');
      const count = document.getElementById('textModelCount');
      return document.readyState === 'complete' && refresh && !refresh.disabled && count && count.textContent.trim() === '1';
    })()`, 'The real dashboard did not become ready for model refresh.');
    const clicked = await cdp.send('Runtime.evaluate', {
      expression: `(() => {
        const refresh = document.getElementById('refreshModelSetupButton');
        if (!refresh || refresh.disabled) return false;
        refresh.click();
        return true;
      })()`,
      returnByValue: true,
    }, sessionId);
    assert.equal(clicked.result?.value, true,
      'real server: dashboard Refresh button could not be clicked');
    await waitForBrowserExpression(cdp, sessionId, `(() => {
      const count = document.getElementById('textModelCount');
      const models = Array.from(document.querySelectorAll('#textModelList li'), item => item.textContent.trim());
      return count && count.textContent.trim() === ${JSON.stringify(String(expectedModels.length))}
        && ${JSON.stringify(expectedModels)}.every(model => models.includes(model));
    })()`, 'The real dashboard Refresh did not expose the rescanned text models.');
    const result = await cdp.send('Runtime.evaluate', {
      expression: `(() => ({
        count: document.getElementById('textModelCount').textContent.trim(),
        models: Array.from(document.querySelectorAll('#textModelList li'), item => item.textContent.trim()),
      }))()`,
      returnByValue: true,
    }, sessionId);
    return result.result?.value || {};
  });
}

async function waitForBrowserExpression(cdp, sessionId, expression, timeoutMessage) {
  const deadline = Date.now() + PAGE_SETTLE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const evaluated = await cdp.send('Runtime.evaluate', { expression, returnByValue: true }, sessionId);
    if (evaluated.result?.value) return;
    await delay(75);
  }
  throw new Error(timeoutMessage);
}

async function inspectWithCdp(cdp, url, scenario) {
  const viewport = scenario.viewport || PHONE_VIEWPORT;
  const target = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const attached = await cdp.send('Target.attachToTarget', { targetId: target.targetId, flatten: true });
  const sessionId = attached.sessionId;
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: viewport.width,
    height: viewport.height,
    deviceScaleFactor: 1,
    mobile: viewport.mobile,
    screenWidth: viewport.width,
    screenHeight: viewport.height,
  }, sessionId);
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Runtime.enable', {}, sessionId);
  if (scenario.initialState) {
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `try {
        if (!localStorage.getItem(${JSON.stringify(STATE_STORAGE_KEY)})) {
          localStorage.setItem(${JSON.stringify(STATE_STORAGE_KEY)}, ${JSON.stringify(JSON.stringify(scenario.initialState))});
        }
      } catch {}`,
    }, sessionId);
  }
  await cdp.send('Page.navigate', { url }, sessionId);

  const deadline = Date.now() + PAGE_SETTLE_TIMEOUT_MS;
  const readinessExpression = pageReadinessExpression(scenario);
  let ready = false;
  while (Date.now() < deadline) {
    const evaluated = await cdp.send('Runtime.evaluate', {
      expression: readinessExpression,
      returnByValue: true,
    }, sessionId);
    ready = Boolean(evaluated.result && evaluated.result.value);
    if (ready) break;
    await delay(75);
  }
  if (!ready) throw new Error('Frontend did not settle before the browser regression deadline.');

  const chatExercise = scenario.exerciseChat
    ? await exercisePersistedChatSelection(cdp, sessionId, scenario)
    : null;
  await delay(100);
  const evaluated = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const bodyStyle = getComputedStyle(document.body);
      const appShell = document.querySelector('.app-shell');
      const sidebar = document.getElementById('sidebar');
      const topbar = document.querySelector('.topbar');
      const chat = document.querySelector('.chat-layout');
      const messages = document.getElementById('messages');
      const modelLoading = document.getElementById('modelLoading');
      const composer = document.querySelector('.composer');
      const prompt = document.getElementById('promptInput');
      const send = document.getElementById('sendButton');
      const mobileMenu = document.getElementById('openSidebarButton');
      const access = document.getElementById(${JSON.stringify(ACCESS_REQUIRED_ID)});
      const textModel = document.getElementById('modelSelect');
      const baseUrl = document.getElementById('baseUrlInput');
      const localSetupLink = document.getElementById('localModelSetupLink');
      const localSetupGuidance = document.getElementById('localModelSetupGuidance');
      const topbarStyle = topbar ? getComputedStyle(topbar) : null;
      const appShellStyle = appShell ? getComputedStyle(appShell) : null;
      const sidebarStyle = sidebar ? getComputedStyle(sidebar) : null;
      const chatStyle = chat ? getComputedStyle(chat) : null;
      const messagesStyle = messages ? getComputedStyle(messages) : null;
      const modelLoadingStyle = modelLoading ? getComputedStyle(modelLoading) : null;
      const composerStyle = composer ? getComputedStyle(composer) : null;
      const menuStyle = mobileMenu ? getComputedStyle(mobileMenu) : null;
      const accessStyle = access ? getComputedStyle(access) : null;
      const appShellRect = appShell ? appShell.getBoundingClientRect() : null;
      const sidebarRect = sidebar ? sidebar.getBoundingClientRect() : null;
      const chatRect = chat ? chat.getBoundingClientRect() : null;
      const topbarRect = topbar ? topbar.getBoundingClientRect() : null;
      const messagesRect = messages ? messages.getBoundingClientRect() : null;
      const modelLoadingRect = modelLoading ? modelLoading.getBoundingClientRect() : null;
      const composerRect = composer ? composer.getBoundingClientRect() : null;
      const promptRect = prompt ? prompt.getBoundingClientRect() : null;
      const sendRect = send ? send.getBoundingClientRect() : null;
      const imageOptionsByKind = {};
      const samplerOptionsByKind = {};
      if (!${scenario.accessRequired ? 'true' : 'false'} && ${scenario.requireImageModels === false ? 'false' : 'true'}) {
        const imageKind = document.getElementById('imageKindSelect');
        const imageModels = document.getElementById('imageModelSelect');
        const imageSamplers = document.getElementById('imageSamplerSelect');
        if (imageKind && imageModels && imageSamplers) {
          for (const kind of ['anima', 'sdxl']) {
            imageKind.value = kind;
            imageKind.dispatchEvent(new Event('change', { bubbles: true }));
            imageOptionsByKind[kind] = Array.from(imageModels.options, option => option.value);
            samplerOptionsByKind[kind] = Array.from(imageSamplers.options, option => option.value);
          }
          imageKind.value = 'anima';
          imageKind.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
      let storedState = {};
      try { storedState = JSON.parse(localStorage.getItem(${JSON.stringify(STATE_STORAGE_KEY)}) || '{}'); } catch {}
      return {
        dom: document.documentElement.outerHTML,
        viewport: { width: innerWidth, height: innerHeight },
        locationHash: location.hash,
        storedAccessToken: sessionStorage.getItem('local-ai-serving-access-token-v1') || '',
        storedState,
        baseUrlValue: baseUrl ? baseUrl.value : '',
        selectedTextModel: textModel ? textModel.value : '',
        textModelDisabled: textModel ? textModel.disabled : true,
        localSetupLinkVisible: Boolean(localSetupLink && !localSetupLink.hidden),
        localSetupLinkHref: localSetupLink ? localSetupLink.href : '',
        localSetupGuidanceVisible: Boolean(localSetupGuidance && !localSetupGuidance.hidden),
        localSetupGuidanceText: localSetupGuidance ? localSetupGuidance.textContent.trim() : '',
        imageOptionsByKind,
        samplerOptionsByKind,
        appearance: {
          stylesheets: document.styleSheets.length,
          bodyBackgroundColor: bodyStyle.backgroundColor,
          bodyBackgroundImage: bodyStyle.backgroundImage,
          bodyFontFamily: bodyStyle.fontFamily,
          bodyOverflowX: bodyStyle.overflowX,
          bodyOverflowY: bodyStyle.overflowY,
          appDisplay: appShellStyle ? appShellStyle.display : '',
          appGridColumns: appShellStyle ? appShellStyle.gridTemplateColumns : '',
          appPaddingLeft: appShellStyle ? Math.round(parseFloat(appShellStyle.paddingLeft)) : 0,
          appWidth: appShellRect ? Math.round(appShellRect.width) : 0,
          appHeight: appShellRect ? Math.round(appShellRect.height) : 0,
          sidebarDisplay: sidebarStyle ? sidebarStyle.display : '',
          sidebarPosition: sidebarStyle ? sidebarStyle.position : '',
          sidebarWidth: sidebarRect ? Math.round(sidebarRect.width) : 0,
          sidebarHeight: sidebarRect ? Math.round(sidebarRect.height) : 0,
          sidebarLeft: sidebarRect ? Math.round(sidebarRect.left) : 0,
          sidebarRight: sidebarRect ? Math.round(sidebarRect.right) : 0,
          topbarBackgroundImage: topbarStyle ? topbarStyle.backgroundImage : '',
          topbarBorderRadius: topbarStyle ? topbarStyle.borderRadius : '',
          topbarHeight: topbarRect ? Math.round(topbarRect.height) : 0,
          chatDisplay: chatStyle ? chatStyle.display : '',
          chatBackgroundColor: chatStyle ? chatStyle.backgroundColor : '',
          chatBoxShadow: chatStyle ? chatStyle.boxShadow : '',
          chatWidth: chatRect ? Math.round(chatRect.width) : 0,
          chatHeight: chatRect ? Math.round(chatRect.height) : 0,
          chatLeft: chatRect ? Math.round(chatRect.left) : 0,
          chatRight: chatRect ? Math.round(chatRect.right) : 0,
          chatTop: chatRect ? Math.round(chatRect.top) : 0,
          chatBottom: chatRect ? Math.round(chatRect.bottom) : 0,
          messagesOverflowY: messagesStyle ? messagesStyle.overflowY : '',
          messagesBackgroundImage: messagesStyle ? messagesStyle.backgroundImage : '',
          messagesHeight: messagesRect ? Math.round(messagesRect.height) : 0,
          messagesBottom: messagesRect ? Math.round(messagesRect.bottom) : 0,
          modelLoadingHeight: modelLoadingRect && modelLoadingStyle?.display !== 'none'
            ? Math.round(modelLoadingRect.height)
            : 0,
          composerBackgroundColor: composerStyle ? composerStyle.backgroundColor : '',
          composerWidth: composerRect ? Math.round(composerRect.width) : 0,
          composerHeight: composerRect ? Math.round(composerRect.height) : 0,
          composerLeft: composerRect ? Math.round(composerRect.left) : 0,
          composerTop: composerRect ? Math.round(composerRect.top) : 0,
          composerBottom: composerRect ? Math.round(composerRect.bottom) : 0,
          promptHeight: promptRect ? Math.round(promptRect.height) : 0,
          sendHeight: sendRect ? Math.round(sendRect.height) : 0,
          mobileMenuDisplay: menuStyle ? menuStyle.display : '',
          accessDisplay: accessStyle ? accessStyle.display : '',
        },
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  const value = evaluated.result && evaluated.result.value;
  if (!value || typeof value.dom !== 'string') throw new Error('Browser did not return the rendered frontend DOM.');
  value.chatExercise = chatExercise;
  return value;
}

async function exercisePersistedChatSelection(cdp, sessionId, scenario) {
  const selectedModel = TEXT_MODELS[1];
  const selected = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const select = document.getElementById('modelSelect');
      if (!select || !Array.from(select.options, option => option.value).includes(${JSON.stringify(selectedModel)})) {
        return { selected: false };
      }
      select.value = ${JSON.stringify(selectedModel)};
      select.dispatchEvent(new Event('change', { bubbles: true }));
      const loading = document.getElementById('modelLoading');
      const progress = document.getElementById('modelLoadingProgress');
      return {
        selected: true,
        visible: Boolean(loading && !loading.hidden),
        indeterminate: Boolean(progress && !progress.hasAttribute('value')),
        modelDisabled: select.disabled,
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  const selectionLoad = selected.result?.value || {};
  assert.equal(selectionLoad.selected, true,
    'stream fixture: second synthetic text model could not be selected');
  assert.equal(selectionLoad.visible, true,
    'stream fixture: selecting the second model did not reveal loading state');
  assert.equal(selectionLoad.indeterminate, true,
    'stream fixture: the model loading progress was not indeterminate');
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const select = document.getElementById('modelSelect');
    const loading = document.getElementById('modelLoading');
    const progress = document.getElementById('modelLoadingProgress');
    return select && !select.disabled && loading && !loading.hidden && progress
      && progress.hasAttribute('value') && progress.value === progress.max;
  })()`, 'The selected second text model never reached its ready UI state.');
  const readySnapshot = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const progress = document.getElementById('modelLoadingProgress');
      return { complete: Boolean(progress && progress.hasAttribute('value') && progress.value === progress.max) };
    })()`,
    returnByValue: true,
  }, sessionId);
  const selectionReady = readySnapshot.result?.value || {};
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    try {
      const state = JSON.parse(localStorage.getItem(${JSON.stringify(STATE_STORAGE_KEY)}) || '{}');
      return state.settings && state.settings.model === ${JSON.stringify(selectedModel)};
    } catch { return false; }
  })()`, 'The selected second text model was not persisted before reload.');

  await cdp.send('Runtime.evaluate', {
    expression: "window.__browserSmokeReloadMarker = 'before'",
  }, sessionId);
  await cdp.send('Page.reload', { ignoreCache: true }, sessionId);
  await waitForBrowserExpression(cdp, sessionId, `(() => (
    document.readyState === 'complete' && window.__browserSmokeReloadMarker !== 'before'
  ))()`, 'The frontend document did not reload.');
  await waitForBrowserExpression(
    cdp,
    sessionId,
    pageReadinessExpression(scenario),
    'The frontend did not settle after reloading the persisted model selection.',
  );
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const select = document.getElementById('modelSelect');
    return select && !select.disabled && select.value === ${JSON.stringify(selectedModel)};
  })()`, 'The second text model was not restored after reload.');

  const firstTurn = await submitConversationPrompt(cdp, sessionId, CHAT_TURN_ONE.prompt, CHAT_TURN_ONE.response);
  const chatLoad = firstTurn.loading;
  assert.equal(chatLoad.visible, true,
    'conversation fixture: chat did not reveal managed model loading state');
  await submitConversationPrompt(cdp, sessionId, CHAT_TURN_TWO.prompt, CHAT_TURN_TWO.response);
  await submitConversationPrompt(cdp, sessionId, CHAT_IMAGE_TOOL.prompt, CHAT_IMAGE_TOOL.response);
  await waitForGeneratedImages(cdp, sessionId, 1, 'The assistant image tool did not render its generated image.');

  const imageControls = await generateManualImageThroughUi(cdp, sessionId);
  await waitForGeneratedImages(cdp, sessionId, 2, 'Image studio did not render its generated image.');
  const imageGallery = await exerciseImageGalleryAndViewport(cdp, sessionId);
  const mixedBeforeReload = await readConversationSnapshot(cdp, sessionId);

  await reloadConversationPage(cdp, sessionId, scenario);
  await waitForGeneratedImages(cdp, sessionId, 2, 'Generated images did not survive a full-page reload.');
  const mixedAfterReload = await readConversationSnapshot(cdp, sessionId);

  await submitConversationPrompt(cdp, sessionId, CHAT_AFTER_IMAGE.prompt, CHAT_AFTER_IMAGE.response);
  await clickConversationAction(cdp, sessionId, 'regenerateButton');
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const text = document.getElementById('messages')?.textContent || '';
    return text.includes(${JSON.stringify(CHAT_AFTER_IMAGE.regeneratedResponse)})
      && !text.includes(${JSON.stringify(CHAT_AFTER_IMAGE.response)});
  })()`, 'Regenerating did not replace the last assistant answer.');
  const afterRegenerate = await readConversationSnapshot(cdp, sessionId);

  const edited = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const message = Array.from(document.querySelectorAll('.message.user')).find(node => (
        node.querySelector('.message-content')?.textContent.trim() === ${JSON.stringify(CHAT_AFTER_IMAGE.prompt)}
      ));
      const edit = Array.from(message?.querySelectorAll('button') || []).find(button => button.textContent.trim() === 'Edit');
      if (!edit) return false;
      edit.click();
      return true;
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.equal(edited.result?.value, true, 'conversation fixture: the last user turn could not be edited');
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const prompt = document.getElementById('promptInput');
    const text = document.getElementById('messages')?.textContent || '';
    return prompt?.value === ${JSON.stringify(CHAT_AFTER_IMAGE.prompt)}
      && !text.includes(${JSON.stringify(CHAT_AFTER_IMAGE.regeneratedResponse)});
  })()`, 'Editing did not restore and truncate the selected user turn.');
  const afterEdit = await readConversationSnapshot(cdp, sessionId);
  await submitConversationPrompt(cdp, sessionId, CHAT_EDITED_TURN.prompt, CHAT_EDITED_TURN.response);
  const afterEditedTurn = await readConversationSnapshot(cdp, sessionId);

  await cdp.send('Runtime.evaluate', {
    expression: `window.prompt = () => ${JSON.stringify(RENAMED_CONVERSATION)}`,
  }, sessionId);
  await clickConversationAction(cdp, sessionId, 'renameButton');
  await waitForBrowserExpression(cdp, sessionId, `(() => (
    document.getElementById('conversationTitle')?.textContent.trim() === ${JSON.stringify(RENAMED_CONVERSATION)}
  ))()`, 'Renaming did not update the active conversation title.');
  await reloadConversationPage(cdp, sessionId, scenario);
  await waitForGeneratedImages(cdp, sessionId, 2, 'Mixed conversation images disappeared after the renamed chat reloaded.');

  const conversationIsolation = await exerciseConversationIsolation(cdp, sessionId);
  await cdp.send('Runtime.evaluate', { expression: 'window.confirm = () => true' }, sessionId);
  await clickConversationAction(cdp, sessionId, 'clearButton');
  await waitForBrowserExpression(cdp, sessionId, `(() => (
    document.querySelectorAll('.message').length === 0
      && Boolean(document.querySelector('.empty-state'))
  ))()`, 'Clearing did not return the active conversation to its empty state.');
  const afterClear = await readConversationSnapshot(cdp, sessionId);

  const result = await cdp.send('Runtime.evaluate', {
    expression: `(() => ({
      selectedModel: document.getElementById('modelSelect').value,
    }))()`,
    returnByValue: true,
  }, sessionId);
  return {
    ...(result.result?.value || {}),
    assistantText: afterEditedTurn.contents.join(' '),
    selectionLoad,
    selectionReady,
    chatLoad,
    imageControls,
    imageGallery,
    mixedBeforeReload,
    mixedAfterReload,
    afterRegenerate,
    afterEdit,
    afterEditedTurn,
    conversationIsolation,
    afterClear,
  };
}

async function submitConversationPrompt(cdp, sessionId, promptText, expectedResponse) {
  const submitted = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const prompt = document.getElementById('promptInput');
      const send = document.getElementById('sendButton');
      if (!prompt || !send || send.disabled) return { submitted: false };
      prompt.value = ${JSON.stringify(promptText)};
      prompt.dispatchEvent(new Event('input', { bubbles: true }));
      send.click();
      const loading = document.getElementById('modelLoading');
      const progress = document.getElementById('modelLoadingProgress');
      return {
        submitted: true,
        loading: {
          visible: Boolean(loading && !loading.hidden),
          indeterminate: Boolean(progress && !progress.hasAttribute('value')),
        },
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.equal(submitted.result?.value?.submitted, true,
    `conversation fixture: could not submit ${JSON.stringify(promptText)}`);
  await waitForBrowserExpression(cdp, sessionId, `(() => (
    Array.from(document.querySelectorAll('.message.assistant .message-content'))
      .some(message => message.textContent.includes(${JSON.stringify(expectedResponse)}))
      && !document.getElementById('promptInput')?.disabled
  ))()`, `The expected response for ${JSON.stringify(promptText)} did not render.`);
  return submitted.result?.value || {};
}

async function generateManualImageThroughUi(cdp, sessionId) {
  await cdp.send('Runtime.evaluate', {
    expression: `document.getElementById('messages').scrollTop = 0`,
  }, sessionId);
  const opened = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const open = document.getElementById('imageButton');
      if (!open || open.disabled) return false;
      open.click();
      return Boolean(document.getElementById('imageDialog')?.open);
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.equal(opened.result?.value, true,
    'conversation fixture: Image studio could not be opened');
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const dialog = document.getElementById('imageDialog');
    const generate = document.getElementById('generateImageButton');
    return Boolean(dialog?.open && generate && !generate.disabled);
  })()`, 'Image studio did not finish refreshing its model list.');
  const generated = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const prompt = document.getElementById('imagePromptInput');
      const generate = document.getElementById('generateImageButton');
      const kind = document.getElementById('imageKindSelect');
      const model = document.getElementById('imageModelSelect');
      const steps = document.getElementById('imageStepsInput');
      const cfg = document.getElementById('imageCfgInput');
      const sampler = document.getElementById('imageSamplerSelect');
      const every = document.getElementById('autoNegativeEverySelect');
      const auto = document.getElementById('autoNegativeButton');
      if (!prompt || !generate || generate.disabled || !kind || !model || !steps || !cfg || !sampler || !every || !auto) return { generated: false };
      kind.value = 'sdxl';
      kind.dispatchEvent(new Event('change', { bubbles: true }));
      steps.value = '5';
      steps.dispatchEvent(new Event('input', { bubbles: true }));
      cfg.value = '123.45';
      cfg.dispatchEvent(new Event('input', { bubbles: true }));
      model.value = ${JSON.stringify(SDXL_IMAGE_MODELS[1])};
      model.dispatchEvent(new Event('change', { bubbles: true }));
      prompt.value = ${JSON.stringify(MANUAL_IMAGE_PROMPT)};
      prompt.dispatchEvent(new Event('input', { bubbles: true }));
      every.value = '2';
      every.dispatchEvent(new Event('change', { bubbles: true }));
      const originalRandom = Math.random;
      Math.random = () => 0;
      auto.click();
      Math.random = originalRandom;
      const negativePrompt = document.getElementById('negativePromptInput').value;
      const snapshot = {
        generated: true,
        kind: kind.value,
        model: model.value,
        steps: steps.value,
        cfg: cfg.value,
        sampler: sampler.value,
        negativePrompt,
        scrollBefore: document.getElementById('messages').scrollTop,
      };
      generate.click();
      return snapshot;
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.equal(generated.result?.value?.generated, true,
    'conversation fixture: Image studio could not submit a manual image prompt');
  return generated.result?.value || {};
}

async function exerciseImageGalleryAndViewport(cdp, sessionId) {
  const opened = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const messages = document.getElementById('messages');
      const imageFrames = Array.from(document.querySelectorAll('.generated-image-frame'));
      const clickedImage = imageFrames[imageFrames.length - 1];
      if (!messages || !clickedImage) return { opened: false };
      const scrollBefore = messages.scrollTop;
      clickedImage.click();
      return {
        opened: document.getElementById('imageGalleryDialog')?.open === true,
        scrollBefore,
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.equal(opened.result?.value?.opened, true,
    'conversation fixture: clicking a generated image did not open the gallery');
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const dialog = document.getElementById('imageGalleryDialog');
    const items = Array.from(document.querySelectorAll('.gallery-item'));
    const images = Array.from(document.querySelectorAll('.gallery-image'));
    return dialog?.open && items.length === 2 && images.length === 2
      && images.every(image => image.complete && image.naturalWidth > 0);
  })()`, 'The conversation gallery did not hydrate both generated images.');
  const inspected = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const messages = document.getElementById('messages');
      const viewport = document.getElementById('imageGalleryViewport');
      const initialGalleryScroll = viewport.scrollTop;
      viewport.scrollTop = 0;
      const galleryScroll = viewport.scrollTop;
      const itemCount = document.querySelectorAll('.gallery-item').length;
      const galleryScrollable = viewport.scrollHeight > viewport.clientHeight;
      document.getElementById('closeImageGalleryButton').click();
      return {
        itemCount,
        initialGalleryScroll,
        galleryScroll,
        galleryScrollable,
        scrollBefore: ${JSON.stringify(opened.result?.value?.scrollBefore || 0)},
        scrollAfter: messages.scrollTop,
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  return inspected.result?.value || {};
}

async function waitForGeneratedImages(cdp, sessionId, expectedCount, timeoutMessage) {
  await waitForBrowserExpression(cdp, sessionId, `(() => {
    const imageMessages = Array.from(document.querySelectorAll('.message.kind-image'));
    const images = Array.from(document.querySelectorAll('.generated-image'));
    return imageMessages.length === ${expectedCount}
      && images.length === ${expectedCount}
      && images.every(image => image.complete && image.naturalWidth > 0);
  })()`, timeoutMessage);
}

async function reloadConversationPage(cdp, sessionId, scenario) {
  await cdp.send('Runtime.evaluate', {
    expression: "window.__browserSmokeReloadMarker = 'before'",
  }, sessionId);
  await cdp.send('Page.reload', { ignoreCache: true }, sessionId);
  await waitForBrowserExpression(cdp, sessionId, `(() => (
    document.readyState === 'complete' && window.__browserSmokeReloadMarker !== 'before'
  ))()`, 'The mixed conversation page did not reload.');
  await waitForBrowserExpression(cdp, sessionId, pageReadinessExpression(scenario),
    'The mixed conversation page did not settle after reload.');
}

async function clickConversationAction(cdp, sessionId, actionId) {
  const clicked = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const menu = document.getElementById('conversationMenuButton');
      const action = document.getElementById(${JSON.stringify(actionId)});
      if (!menu || !action || action.disabled) return false;
      menu.click();
      if (!document.getElementById('conversationActionsDialog')?.open) return false;
      action.click();
      return true;
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.equal(clicked.result?.value, true,
    `conversation fixture: action ${actionId} could not be invoked through the app menu`);
}

async function exerciseConversationIsolation(cdp, sessionId) {
  const created = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      document.getElementById('newChatButton')?.click();
      return {
        title: document.getElementById('conversationTitle')?.textContent.trim(),
        conversationCount: document.querySelectorAll('.conversation-item').length,
        messageCount: document.querySelectorAll('.message').length,
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.deepEqual(created.result?.value, { title: 'New chat', conversationCount: 2, messageCount: 0 },
    'conversation fixture: creating a new chat did not isolate it from the mixed conversation');

  const switchedAndDeleted = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const items = Array.from(document.querySelectorAll('.conversation-item'));
      const oldItem = items.find(item => item.querySelector('.conversation-name')?.textContent.trim() === ${JSON.stringify(RENAMED_CONVERSATION)});
      oldItem?.querySelector('.conversation-open')?.click();
      const newItem = Array.from(document.querySelectorAll('.conversation-item')).find(item => (
        item.querySelector('.conversation-name')?.textContent.trim() === 'New chat'
      ));
      newItem?.querySelector('.delete-chat')?.click();
      return {
        title: document.getElementById('conversationTitle')?.textContent.trim(),
        conversationCount: document.querySelectorAll('.conversation-item').length,
        imageCount: document.querySelectorAll('.message.kind-image').length,
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  assert.deepEqual(switchedAndDeleted.result?.value, {
    title: RENAMED_CONVERSATION,
    conversationCount: 1,
    imageCount: 2,
  }, 'conversation fixture: switching back or deleting the isolated new chat changed the original conversation');
  return { created: created.result?.value, switchedAndDeleted: switchedAndDeleted.result?.value };
}

async function readConversationSnapshot(cdp, sessionId) {
  const snapshot = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      let saved = {};
      try { saved = JSON.parse(localStorage.getItem(${JSON.stringify(STATE_STORAGE_KEY)}) || '{}'); } catch {}
      const active = (saved.conversations || []).find(item => item.id === saved.activeConversationId) || {};
      return {
        title: active.title || '',
        roles: (active.messages || []).map(message => message.role),
        kinds: (active.messages || []).map(message => message.kind || 'text'),
        contents: (active.messages || []).map(message => message.content || ''),
        messageCount: (active.messages || []).length,
        visibleMessageCount: document.querySelectorAll('.message').length,
        imageCount: document.querySelectorAll('.message.kind-image').length,
        generatedImageCount: document.querySelectorAll('.generated-image').length,
        promptValue: document.getElementById('promptInput')?.value || '',
        conversationCount: (saved.conversations || []).length,
        imageSettings: saved.image || {},
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  return snapshot.result?.value || {};
}

function pageReadinessExpression(scenario) {
  const expectAccessRequired = Boolean(scenario.accessRequired);
  const textModels = JSON.stringify(scenario.expectedTextModels || TEXT_MODELS);
  const imageModels = JSON.stringify(IMAGE_MODELS);
  const requireImageModels = scenario.requireImageModels !== false;
  return `(() => {
    if (document.readyState !== 'complete') return false;
    const access = document.getElementById(${JSON.stringify(ACCESS_REQUIRED_ID)});
    const text = document.getElementById('modelSelect');
    const image = document.getElementById('imageModelSelect');
    if (!access || !text || !image) return false;
    if (${expectAccessRequired ? 'true' : 'false'}) return !access.hidden;
    const textValues = Array.from(text.options, option => option.value);
    const imageValues = Array.from(image.options, option => option.value);
    return access.hidden
      && ${textModels}.every(value => textValues.includes(value))
      && (${requireImageModels ? imageModels + '.some(value => imageValues.includes(value))' : 'true'});
  })()`;
}

async function removeBrowserProfile(profilePath) {
  const resolvedTemp = path.resolve(os.tmpdir());
  const resolvedProfile = path.resolve(profilePath);
  const safeProfile = path.dirname(resolvedProfile) === resolvedTemp
    && path.basename(resolvedProfile).startsWith('local-ai-browser-smoke-');
  if (!safeProfile) throw new Error('Refusing to remove an unexpected browser profile path.');
  if (process.platform === 'win32') {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      const cleaner = spawn('powershell.exe', [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        'Remove-Item -LiteralPath $env:LOCAL_AI_BROWSER_PROFILE -Recurse -Force -ErrorAction Stop',
      ], {
        env: { ...process.env, LOCAL_AI_BROWSER_PROFILE: resolvedProfile },
        shell: false,
        stdio: 'ignore',
        windowsHide: true,
      });
      await new Promise((resolve) => cleaner.once('exit', resolve));
      if (!fs.existsSync(resolvedProfile)) return;
      await delay(100);
    }
    throw new Error('The headless browser profile could not be removed.');
  }
  let lastError = null;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      await fs.promises.rm(resolvedProfile, { recursive: true, force: true });
      if (!fs.existsSync(resolvedProfile)) return;
    } catch (error) {
      lastError = error;
    }
    await delay(100);
  }
  throw new Error(lastError ? 'The headless browser profile could not be removed.' : 'The headless browser profile remained after cleanup.');
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

class CdpWebSocket {
  static connect(url) {
    if (typeof globalThis.WebSocket !== 'function' || process.env.BROWSER_SMOKE_FORCE_RAW_WEBSOCKET === '1') {
      return RawWebSocket.connect(url).then((socket) => new CdpWebSocket(socket));
    }
    return new Promise((resolve, reject) => {
      const socket = new globalThis.WebSocket(url);
      const timer = setTimeout(() => {
        socket.close();
        reject(new Error('Timed out connecting to the browser debugging endpoint.'));
      }, 10_000);
      socket.addEventListener('open', () => {
        clearTimeout(timer);
        resolve(new CdpWebSocket(socket));
      }, { once: true });
      socket.addEventListener('error', () => {
        clearTimeout(timer);
        reject(new Error('Could not connect to the browser debugging endpoint.'));
      }, { once: true });
    });
  }

  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.closed = false;
    socket.addEventListener('message', (event) => this.onMessage(event.data));
    socket.addEventListener('close', () => this.close());
    socket.addEventListener('error', () => this.close());
  }

  send(method, params = {}, sessionId = '') {
    if (this.closed) return Promise.reject(new Error('Browser debugging connection closed unexpectedly.'));
    const id = this.nextId++;
    const message = { id, method, params };
    if (sessionId) message.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Browser command timed out: ${method}`));
      }, 10_000);
      this.pending.set(id, { resolve, reject, timer, method });
      try {
        this.socket.send(JSON.stringify(message));
      } catch {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new Error('Could not write to the browser debugging connection.'));
      }
    });
  }

  onMessage(value) {
    let message;
    try {
      const text = typeof value === 'string' ? value : Buffer.from(value).toString('utf8');
      message = JSON.parse(text);
    } catch {
      return;
    }
    if (!message.id) return;
    const pending = this.pending.get(message.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(message.id);
    if (message.error) pending.reject(new Error(`Browser command failed: ${message.error.message || 'unknown error'}`));
    else pending.resolve(message.result || {});
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    try { this.socket.close(); } catch { /* Already closed. */ }
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error(`Browser debugging connection closed while awaiting ${pending.method}.`));
    }
    this.pending.clear();
  }
}

class RawWebSocket {
  static connect(value) {
    const url = new URL(value);
    if (url.protocol !== 'ws:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) {
      return Promise.reject(new Error('Refusing a non-local browser debugging endpoint.'));
    }
    const port = Number.parseInt(url.port, 10);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      return Promise.reject(new Error('Browser debugging endpoint has an invalid port.'));
    }

    return new Promise((resolve, reject) => {
      const key = crypto.randomBytes(16).toString('base64');
      const expectedAccept = crypto.createHash('sha1')
        .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
        .digest('base64');
      const socket = net.createConnection({ host: url.hostname, port });
      let buffer = Buffer.alloc(0);
      let settled = false;
      const timer = setTimeout(() => fail(new Error('Timed out connecting to the browser debugging endpoint.')), 10_000);

      socket.setNoDelay(true);
      socket.once('connect', () => {
        const requestPath = `${url.pathname}${url.search}`;
        socket.write([
          `GET ${requestPath} HTTP/1.1`,
          `Host: ${url.host}`,
          'Connection: Upgrade',
          'Upgrade: websocket',
          `Sec-WebSocket-Key: ${key}`,
          'Sec-WebSocket-Version: 13',
          'Origin: http://127.0.0.1',
          '',
          '',
        ].join('\r\n'));
      });
      socket.on('data', onHandshakeData);
      socket.once('error', () => fail(new Error('Could not connect to the browser debugging endpoint.')));
      socket.once('close', () => {
        if (!settled) fail(new Error('Browser debugging endpoint closed during the handshake.'));
      });

      function onHandshakeData(chunk) {
        buffer = Buffer.concat([buffer, chunk]);
        const headerEnd = buffer.indexOf('\r\n\r\n');
        if (headerEnd === -1) {
          if (buffer.length > 16 * 1024) fail(new Error('Browser debugging handshake was too large.'));
          return;
        }
        const head = buffer.subarray(0, headerEnd).toString('latin1');
        const lines = head.split('\r\n');
        const headers = new Map();
        for (const line of lines.slice(1)) {
          const separator = line.indexOf(':');
          if (separator > 0) headers.set(line.slice(0, separator).trim().toLowerCase(), line.slice(separator + 1).trim());
        }
        if (!/^HTTP\/1\.1 101\b/.test(lines[0]) || headers.get('sec-websocket-accept') !== expectedAccept) {
          fail(new Error('Browser debugging endpoint rejected the WebSocket handshake.'));
          return;
        }
        settled = true;
        clearTimeout(timer);
        socket.off('data', onHandshakeData);
        const remainder = buffer.subarray(headerEnd + 4);
        const raw = new RawWebSocket(socket);
        if (remainder.length) raw.onData(remainder);
        resolve(raw);
      }

      function fail(error) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        socket.destroy();
        reject(error);
      }
    });
  }

  constructor(socket) {
    this.socket = socket;
    this.buffer = Buffer.alloc(0);
    this.fragments = [];
    this.fragmentOpcode = 0;
    this.listeners = new Map();
    this.closed = false;
    socket.on('data', (chunk) => this.onData(chunk));
    socket.on('error', () => this.emit('error', {}));
    socket.on('close', () => this.finishClose());
  }

  addEventListener(type, callback, options = {}) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push({ callback, once: Boolean(options.once) });
  }

  send(value) {
    if (this.closed) throw new Error('Browser debugging connection is closed.');
    this.writeFrame(0x1, Buffer.from(String(value), 'utf8'));
  }

  close() {
    if (this.closed) return;
    try { this.writeFrame(0x8, Buffer.alloc(0)); } catch { /* The socket may already be closed. */ }
    this.closed = true;
    this.socket.end();
    this.emit('close', {});
  }

  writeFrame(opcode, payload) {
    const length = payload.length;
    const extendedLengthBytes = length < 126 ? 0 : length <= 0xffff ? 2 : 8;
    const header = Buffer.alloc(2 + extendedLengthBytes + 4);
    header[0] = 0x80 | opcode;
    if (extendedLengthBytes === 0) header[1] = 0x80 | length;
    else if (extendedLengthBytes === 2) {
      header[1] = 0x80 | 126;
      header.writeUInt16BE(length, 2);
    } else {
      header[1] = 0x80 | 127;
      header.writeBigUInt64BE(BigInt(length), 2);
    }
    const maskOffset = 2 + extendedLengthBytes;
    const mask = crypto.randomBytes(4);
    mask.copy(header, maskOffset);
    const masked = Buffer.allocUnsafe(length);
    for (let index = 0; index < length; index += 1) masked[index] = payload[index] ^ mask[index % 4];
    this.socket.write(Buffer.concat([header, masked]));
  }

  onData(chunk) {
    if (this.closed) return;
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (this.readFrame()) {
      // Continue until the current network buffer no longer contains a full frame.
    }
  }

  readFrame() {
    if (this.buffer.length < 2) return false;
    const first = this.buffer[0];
    const second = this.buffer[1];
    const final = Boolean(first & 0x80);
    const opcode = first & 0x0f;
    const masked = Boolean(second & 0x80);
    let length = second & 0x7f;
    let offset = 2;
    if (length === 126) {
      if (this.buffer.length < 4) return false;
      length = this.buffer.readUInt16BE(2);
      offset = 4;
    } else if (length === 127) {
      if (this.buffer.length < 10) return false;
      const largeLength = this.buffer.readBigUInt64BE(2);
      if (largeLength > 4n * 1024n * 1024n) {
        this.emit('error', {});
        this.close();
        return false;
      }
      length = Number(largeLength);
      offset = 10;
    }
    const maskBytes = masked ? 4 : 0;
    if (length > 4 * 1024 * 1024) {
      this.emit('error', {});
      this.close();
      return false;
    }
    if (this.buffer.length < offset + maskBytes + length) return false;
    const mask = masked ? this.buffer.subarray(offset, offset + 4) : null;
    offset += maskBytes;
    const payload = Buffer.from(this.buffer.subarray(offset, offset + length));
    this.buffer = this.buffer.subarray(offset + length);
    if (mask) {
      for (let index = 0; index < payload.length; index += 1) payload[index] ^= mask[index % 4];
    }
    this.handleFrame(opcode, final, payload);
    return this.buffer.length >= 2;
  }

  handleFrame(opcode, final, payload) {
    if (opcode === 0x8) {
      this.finishClose();
      this.socket.destroy();
      return;
    }
    if (opcode === 0x9) {
      this.writeFrame(0xA, payload);
      return;
    }
    if (opcode === 0xA) return;
    if (opcode === 0x1 || opcode === 0x2) {
      this.fragments = [payload];
      this.fragmentOpcode = opcode;
    } else if (opcode === 0x0 && this.fragments.length) {
      this.fragments.push(payload);
    } else {
      return;
    }
    if (!final) return;
    const message = Buffer.concat(this.fragments);
    const messageOpcode = this.fragmentOpcode;
    this.fragments = [];
    this.fragmentOpcode = 0;
    this.emit('message', { data: messageOpcode === 0x1 ? message.toString('utf8') : message });
  }

  finishClose() {
    if (this.closed) return;
    this.closed = true;
    this.emit('close', {});
  }

  emit(type, event) {
    const listeners = this.listeners.get(type) || [];
    for (const listener of [...listeners]) {
      try { listener.callback(event); } catch { /* Listener failures do not corrupt framing. */ }
      if (listener.once) {
        const index = listeners.indexOf(listener);
        if (index >= 0) listeners.splice(index, 1);
      }
    }
  }
}

function findBrowser() {
  const explicit = process.env.BROWSER_BIN || process.env.EDGE_BIN || process.env.CHROME_BIN;
  if (explicit && isFile(explicit)) return explicit;

  const candidates = process.platform === 'win32'
    ? windowsBrowserCandidates()
    : process.platform === 'darwin'
      ? [
          '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
          '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          '/Applications/Chromium.app/Contents/MacOS/Chromium',
        ]
      : [
          '/usr/bin/microsoft-edge',
          '/usr/bin/microsoft-edge-stable',
          '/usr/bin/google-chrome',
          '/usr/bin/google-chrome-stable',
          '/usr/bin/chromium',
          '/usr/bin/chromium-browser',
        ];

  return candidates.find(isFile) || '';
}

function windowsBrowserCandidates() {
  const programFiles = [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)']].filter(Boolean);
  const candidates = [];
  for (const root of programFiles) {
    candidates.push(
      path.join(root, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
      path.join(root, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    );
  }
  if (process.env.LOCALAPPDATA) {
    candidates.push(
      path.join(process.env.LOCALAPPDATA, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
      path.join(process.env.LOCALAPPDATA, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    );
  }
  return candidates;
}

function isFile(candidate) {
  if (!candidate) return false;
  try {
    return fs.statSync(candidate).isFile();
  } catch {
    return false;
  }
}

async function allocateDistinctPorts(count) {
  const reservations = [];
  try {
    for (let index = 0; index < count; index += 1) {
      const reservation = net.createServer();
      await new Promise((resolve, reject) => {
        reservation.once('error', reject);
        reservation.listen(0, '127.0.0.1', () => {
          reservation.off('error', reject);
          resolve();
        });
      });
      reservations.push(reservation);
    }
    return reservations.map((reservation) => reservation.address().port);
  } finally {
    await Promise.all(reservations.map((reservation) => closeServer(reservation)));
  }
}

function hasExited(child) {
  return !child || child.exitCode !== null || child.signalCode !== null;
}

function settlesWithin(promise, milliseconds) {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(false), milliseconds);
    promise.then(() => {
      clearTimeout(timeout);
      resolve(true);
    }, () => {
      clearTimeout(timeout);
      resolve(true);
    });
  });
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject);
      resolve();
    });
  });
}

function closeServer(server) {
  return new Promise((resolve) => server.close(() => resolve()));
}

function safeMessage(error) {
  const message = error && error.message ? String(error.message) : 'Unknown error.';
  return message.replace(/(?:[A-Za-z]:)?[\\/][^\s"']+/g, '[path]');
}

function safeDiagnostic(value) {
  return String(value)
    .replace(/browser-smoke-(?:current|external|stale|real-server)-token/g, '[synthetic-token]')
    .replace(/https?:\/\/[^\s"']+/g, '[url]')
    .replace(/(?:[A-Za-z]:)?[\\/][^\r\n"']+/g, '[path]')
    .slice(0, 2_000);
}
