'use strict';

let handlingFatalError = false;
process.once('uncaughtException', handleFatalError);
process.once('unhandledRejection', handleFatalError);

const {
  CONFIG_VERSION,
  FIELD_RULES,
  applyLocalConfig,
  normalizeLocalConfig,
  readLocalConfig,
  saveLocalConfig,
} = require('./local-config');
const TEXT_CONFIG_ENVIRONMENT_KEYS = Object.freeze([
  'TEXT_SERVER_EXE',
  'TEXT_MODEL_PATH',
  'TEXT_MODELS_ROOT',
  'TEXT_BASE_URL',
]);
const explicitTextEnvironmentKeys = new Set(
  TEXT_CONFIG_ENVIRONMENT_KEYS.filter((key) => Object.prototype.hasOwnProperty.call(process.env, key)),
);
try {
  applyLocalConfig(process.env, { ignoreUnavailable: true });
} catch (error) {
  console.error(error.code === 'LOCAL_CONFIG_ERROR' ? error.message : 'Local configuration failed.');
  process.exit(1);
}

const fs = require('fs');
const http = require('http');
const https = require('https');
const crypto = require('crypto');
const dns = require('dns');
const net = require('net');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');
const { URL } = require('url');
const { createManagedTextBackend, discoverManagedTextCatalog } = require('./text-backend');
const { pickFolder } = require('./folder-picker');

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const PORT = Number.parseInt(process.env.PORT || '3000', 10);
const HOST = process.env.HOST || '0.0.0.0';
const HTTPS_ENABLED = normalizeBoolean(process.env.HTTPS, true);
const ACCESS_TOKEN = process.env.ACCESS_TOKEN || makeAccessToken();
const ALLOW_LOCAL_BYPASS = process.env.ALLOW_LOCAL_BYPASS !== '0';
let managedTextBackend = createInitialManagedTextBackend();
let DEFAULT_TEXT_BASE_URL = resolveConfiguredTextBaseUrl(process.env, managedTextBackend);
const IMAGE_MODELS_ROOT = process.env.IMAGE_MODELS_ROOT
  ? path.resolve(process.env.IMAGE_MODELS_ROOT)
  : '';
const IMAGE_WORKER_PATH = path.resolve(process.env.IMAGE_WORKER_PATH || path.join(ROOT, 'inference', 'worker.py'));
const IMAGE_PYTHON = process.env.IMAGE_PYTHON || defaultPythonCommand();
const IMAGE_CONFIGURED = Boolean(IMAGE_MODELS_ROOT && process.env.IMAGE_PYTHON);
const ANIMA_TEXT_ENCODER_PATH = process.env.ANIMA_TEXT_ENCODER_PATH
  ? path.resolve(process.env.ANIMA_TEXT_ENCODER_PATH)
  : '';
const ANIMA_VAE_PATH = process.env.ANIMA_VAE_PATH
  ? path.resolve(process.env.ANIMA_VAE_PATH)
  : '';
const REQUEST_LIMIT_BYTES = 2 * 1024 * 1024;
const MODEL_REQUEST_TIMEOUT_MS = 5000;
const PROXY_TIMEOUT_MS = 10 * 60 * 1000;
const IMAGE_GENERATION_TIMEOUT_MS = 20 * 60 * 1000;
const IMAGE_WORKER_IDLE_MS = normalizeInteger(process.env.IMAGE_WORKER_IDLE_MS, 0, 60 * 60 * 1000, 60_000);
const USE_PERSISTENT_IMAGE_WORKER = process.env.IMAGE_WORKER_PERSISTENT === '1'
  || (process.env.IMAGE_WORKER_PERSISTENT !== '0' && !process.env.IMAGE_WORKER_PATH);
const IMAGE_RESPONSE_LIMIT_BYTES = 45 * 1024 * 1024;
const MAX_IMAGE_LORAS = 4;
const FETCH_TIMEOUT_STATE = Symbol('fetchTimeoutState');
const IMAGE_FILE_EXTENSIONS = new Set(['.safetensors']);
const ANIMA_MODEL_PREFIXES = ['net.', 'model.diffusion_model.'];
const ANIMA_MODEL_SIGNATURE = new Map([
  ['llm_adapter.embed.weight', [32128, 1024]],
  ['x_embedder.proj.1.weight', [2048, 68]],
  ['blocks.0.self_attn.q_proj.weight', [2048, 2048]],
  ['final_layer.linear.weight', [64, 2048]],
]);
const SDXL_MODEL_SIGNATURE = new Map([
  ['conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight', [49408, 768]],
  ['conditioner.embedders.1.model.token_embedding.weight', [49408, 1280]],
  ['model.diffusion_model.input_blocks.0.0.weight', [320, 4, 3, 3]],
  ['model.diffusion_model.out.2.weight', [4, 320, 3, 3]],
  ['first_stage_model.decoder.conv_out.weight', [3, 128, 3, 3]],
]);
const DIRECT_WEIGHT_DTYPES = new Set(['F16', 'BF16', 'F32']);
const IMAGE_SIZES = new Map([
  ['portrait', { width: 832, height: 1216 }],
  ['square', { width: 1024, height: 1024 }],
  ['landscape', { width: 1216, height: 832 }],
]);
const modelTypeCache = new Map();
let activeImageProcess = null;
let imageGpuReserved = false;
let activeTextRequests = 0;
let activeTextBackendKey = '';
let textBackendReconfiguring = false;
let activeFolderPicker = null;
let imageRuntimeProbeCache = null;
let imageRuntimeProbePromise = null;
let persistentImageWorker = null;
let persistentImageWorkerStopPromise = null;
let persistentImageWorkerStoppingChild = null;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.ico': 'image/x-icon',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webmanifest': 'application/manifest+json',
};

const protocol = HTTPS_ENABLED ? 'https' : 'http';
const tlsOptions = HTTPS_ENABLED ? createTlsOptions() : null;

const requestHandler = async (req, res) => {
  try {
    await routeRequest(req, res);
  } catch (error) {
    const statusCode = Number.isInteger(error.statusCode) ? error.statusCode : 500;
    if (statusCode === 500) {
      logUnexpectedServerError(error);
    }
    if (!res.headersSent && !res.destroyed) {
      const extraHeaders = error.localAccessRequired ? { 'X-Local-Access-Required': '1' } : {};
      sendJson(res, statusCode, { error: statusCode === 500 ? 'Internal server error.' : error.message }, extraHeaders);
    } else if (!res.destroyed) {
      res.end();
    }
  }
};

function logUnexpectedServerError(error) {
  if (process.env.PRIVATE_DIAGNOSTICS === '1') {
    console.error(error);
    return;
  }
  console.error('Unexpected server error. Set PRIVATE_DIAGNOSTICS=1 for private local details.');
}

function handleFatalError(error) {
  if (handlingFatalError) process.exit(1);
  handlingFatalError = true;
  logUnexpectedServerError(error);
  process.exit(1);
}

const server = HTTPS_ENABLED ? https.createServer(tlsOptions, requestHandler) : http.createServer(requestHandler);

server.listen(PORT, HOST, () => {
  const localUrl = `${protocol}://localhost:${PORT}`;
  console.log(`Local chat is running at ${localUrl}`);
  if (managedTextBackend.enabled) {
    console.log('Managed text backend is configured (loads on demand).');
  }
  console.log(`Local access dashboard: ${localUrl}/dashboard`);
  if (HTTPS_ENABLED) {
    console.log('HTTPS is enabled with a local self-signed certificate.');
  }
  if (!['127.0.0.1', '::1', 'localhost'].includes(String(HOST).toLowerCase())) {
    console.log('LAN access: open the local dashboard and scan its current QR code.');
  }
});

async function routeRequest(req, res) {
  const requestUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  let pathname;
  try {
    pathname = decodeURIComponent(requestUrl.pathname);
  } catch {
    throw Object.assign(new Error('The request path is malformed.'), { statusCode: 400 });
  }

  if (req.method === 'GET' && pathname === '/api/access-info') {
    assertLocalDashboardRequest(req);
    sendJson(res, 200, {
      accessUrls: getLanAddresses().map((address) => makeAccessUrl(address)),
      dashboardUrl: `${protocol}://localhost:${PORT}/dashboard`,
      httpsEnabled: HTTPS_ENABLED,
      certificateFingerprint: tlsOptions ? tlsOptions.fingerprint256 : '',
      tokenLength: ACCESS_TOKEN.length,
    });
    return;
  }

  if (req.method === 'GET' && pathname === '/api/local-setup') {
    assertLocalDashboardRequest(req);
    sendJson(res, 200, getLocalSetupStatus());
    return;
  }

  if (req.method === 'POST' && pathname === '/api/local-setup/pick-folder') {
    assertLocalSetupMutationRequest(req);
    if (isTextFolderLocked()) {
      throw Object.assign(new Error('The text model location is controlled by an environment override.'), {
        statusCode: 409,
      });
    }
    const selectedPath = await pickLocalFolder(req, res);
    sendJson(res, 200, selectedPath ? { cancelled: false, path: selectedPath } : { cancelled: true });
    return;
  }

  if (req.method === 'POST' && pathname === '/api/local-setup/text-folder') {
    assertLocalSetupMutationRequest(req);
    const payload = await readJsonBody(req);
    const status = await updateLocalTextModelsRoot(payload.path);
    sendJson(res, 200, status);
    return;
  }

  if (req.method === 'POST' && pathname === '/api/local-setup/refresh-text-models') {
    assertLocalSetupMutationRequest(req);
    const status = await refreshLocalTextModels();
    sendJson(res, 200, status);
    return;
  }

  if (req.method === 'GET' && pathname === '/api/config') {
    assertAuthorized(req);
    sendJson(res, 200, {
      defaultBaseUrl: DEFAULT_TEXT_BASE_URL,
      managedTextBackend: managedTextBackend.enabled ? {
        enabled: true,
        model: managedTextBackend.alias,
      } : { enabled: false },
      setupRequired: !DEFAULT_TEXT_BASE_URL,
      localSetupAvailable: isLocalRequest(req) && isTrustedLocalBrowserRequest(req),
      port: PORT,
      lanUrls: getLanAddresses().map((address) => `${protocol}://${address}:${PORT}`),
      httpsEnabled: HTTPS_ENABLED,
      accessRequired: true,
    });
    return;
  }

  if (req.method === 'POST' && pathname === '/api/models') {
    assertAuthorized(req);
    await handleModelsRequest(req, res);
    return;
  }

  if (req.method === 'GET' && pathname === '/api/text/status') {
    assertAuthorized(req);
    sendJson(res, 200, getTextBackendLoadStatus());
    return;
  }

  if (req.method === 'POST' && pathname === '/api/text/load') {
    assertAuthorized(req);
    await handleTextLoadRequest(req, res);
    return;
  }

  if (req.method === 'POST' && pathname === '/api/chat') {
    assertAuthorized(req);
    await handleChatRequest(req, res);
    return;
  }

  if (req.method === 'GET' && pathname === '/api/image/config') {
    assertAuthorized(req);
    await handleImageConfigRequest(req, res);
    return;
  }

  if (req.method === 'POST' && pathname === '/api/image/generate') {
    assertAuthorized(req);
    await handleImageGenerationRequest(req, res);
    return;
  }

  if (pathname.startsWith('/api/')) {
    sendJson(res, 404, { error: 'API route not found.' });
    return;
  }

  if (req.method === 'GET' || req.method === 'HEAD') {
    if (pathname === '/dashboard' || pathname.startsWith('/dashboard.')) {
      assertLocalDashboardRequest(req);
    }
    if (pathname === '/dashboard') {
      await serveStatic('/dashboard.html', req, res);
      return;
    }
    await serveStatic(pathname, req, res);
    return;
  }

  sendJson(res, 405, { error: 'Method not allowed.' }, { Allow: 'GET, HEAD, POST' });
}

async function handleModelsRequest(req, res) {
  const payload = await readJsonBody(req);
  const clientAbortSignal = abortWhenClientDisconnects(req, res);
  const baseUrl = resolveTextBaseUrl(payload.baseUrl || DEFAULT_TEXT_BASE_URL);
  const apiKey = typeof payload.apiKey === 'string' ? payload.apiKey : '';
  assertAllowedBaseUrl(baseUrl);
  if (managedTextBackend.matches(baseUrl)) {
    sendJson(res, 200, {
      object: 'list',
      data: managedTextBackend.models.map((model) => ({
        id: model.id,
        object: 'model',
        created: 0,
        owned_by: 'local',
      })),
    });
    return;
  }
  const releaseTextBackend = await prepareTextBackend(baseUrl);
  let upstreamResponse = null;
  try {
    upstreamResponse = await fetchTextBackend(
      `${baseUrl}/models`,
      {
        method: 'GET',
        headers: makeUpstreamHeaders(apiKey),
        signal: clientAbortSignal,
      },
      MODEL_REQUEST_TIMEOUT_MS,
    );

    const bodyText = await readUpstreamText(upstreamResponse);
    res.writeHead(upstreamResponse.status, {
      'Content-Type': upstreamResponse.headers.get('content-type') || 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      ...securityHeaders(),
    });
    res.end(bodyText);
  } finally {
    releaseFetchTimeout(upstreamResponse);
    releaseTextBackend();
  }
}

async function handleTextLoadRequest(req, res) {
  const payload = await readJsonBody(req);
  const baseUrl = resolveTextBaseUrl(payload.baseUrl || DEFAULT_TEXT_BASE_URL);
  const model = typeof payload.model === 'string' ? payload.model.trim() : '';
  assertAllowedBaseUrl(baseUrl);

  if (!managedTextBackend.matches(baseUrl)) {
    sendJson(res, 200, { managed: false, state: 'external', phase: 'idle' });
    return;
  }
  if (!model) {
    sendJson(res, 400, { error: 'Choose a model before loading it.' });
    return;
  }

  const releaseTextBackend = await prepareTextBackend(baseUrl, model);
  releaseTextBackend();
  sendJson(res, 200, getTextBackendLoadStatus());
}

function getTextBackendLoadStatus() {
  if (!managedTextBackend.enabled) {
    return { managed: false, state: 'external', phase: 'idle' };
  }
  return { managed: true, ...managedTextBackend.getStatus() };
}

async function handleChatRequest(req, res) {
  const payload = await readJsonBody(req);
  const clientAbortSignal = abortWhenClientDisconnects(req, res);
  const baseUrl = resolveTextBaseUrl(payload.baseUrl || DEFAULT_TEXT_BASE_URL);
  const messages = applyThinkingMode(normalizeMessages(payload.messages), normalizeThinkingMode(payload.thinkingMode));
  const model = typeof payload.model === 'string' ? payload.model.trim() : '';
  const apiKey = typeof payload.apiKey === 'string' ? payload.apiKey : '';
  const temperature = normalizeNumber(payload.temperature, 0, 2, 0.95);
  const maxTokens = normalizeInteger(payload.max_tokens, 1, 131072, 2048);
  const tools = normalizeChatTools(payload.tools);

  assertAllowedBaseUrl(baseUrl);

  if (!model) {
    sendJson(res, 400, { error: 'Choose a model before sending a message.' });
    return;
  }

  if (!messages.length) {
    sendJson(res, 400, { error: 'At least one message is required.' });
    return;
  }

  const releaseTextBackend = await prepareTextBackend(baseUrl, model);
  let upstreamResponse = null;
  try {
    if (clientAbortSignal.aborted) return;
    upstreamResponse = await fetchTextBackend(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        ...makeUpstreamHeaders(apiKey),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        stream: true,
        ...(tools.length ? { tools, tool_choice: 'auto' } : {}),
      }),
      signal: clientAbortSignal,
    });

    if (!upstreamResponse.ok || !upstreamResponse.body) {
      const errorText = await readUpstreamText(upstreamResponse);
      sendJson(res, upstreamResponse.status || 502, {
        error: extractUpstreamError(errorText) || `The text backend returned HTTP ${upstreamResponse.status}.`,
      });
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
      ...securityHeaders(),
    });

    try {
      for await (const chunk of upstreamResponse.body) {
        if (res.destroyed) break;
        res.write(chunk);
      }
    } catch (error) {
      if (!res.destroyed && (error.name !== 'AbortError' || didFetchTimeOut(upstreamResponse))) {
        const message = didFetchTimeOut(upstreamResponse)
          ? 'The text generation request timed out.'
          : 'The text backend stream ended unexpectedly.';
        res.write(`event: error\ndata: ${JSON.stringify({ error: message })}\n\n`);
      }
    } finally {
      releaseFetchTimeout(upstreamResponse);
      res.end();
    }
  } finally {
    releaseFetchTimeout(upstreamResponse);
    releaseTextBackend();
  }
}

async function serveStatic(urlPathname, req, res) {
  const relativePath = urlPathname === '/' ? 'index.html' : urlPathname.replace(/^\/+/, '');
  const resolvedPath = path.resolve(PUBLIC_DIR, relativePath);

  if (!resolvedPath.startsWith(PUBLIC_DIR + path.sep) && resolvedPath !== PUBLIC_DIR) {
    sendJson(res, 403, { error: 'Forbidden.' });
    return;
  }

  let filePath = resolvedPath;
  let stats;

  try {
    stats = await fs.promises.stat(filePath);
    if (stats.isDirectory()) {
      filePath = path.join(filePath, 'index.html');
      stats = await fs.promises.stat(filePath);
    }
  } catch {
    filePath = path.join(PUBLIC_DIR, 'index.html');
    stats = await fs.promises.stat(filePath);
  }

  const ext = path.extname(filePath).toLowerCase();
  const headers = {
    'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
    'Content-Length': stats.size,
    'Cache-Control': ext === '.html' ? 'no-store' : 'no-cache',
    ...securityHeaders(),
  };

  res.writeHead(200, headers);
  if (req.method === 'HEAD') {
    res.end();
    return;
  }

  const stream = fs.createReadStream(filePath);
  stream.once('error', () => {
    if (!res.destroyed) res.destroy();
  });
  stream.pipe(res);
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const declaredLength = Number(req.headers['content-length']);
    if (Number.isFinite(declaredLength) && declaredLength > REQUEST_LIMIT_BYTES) {
      req.resume();
      reject(Object.assign(new Error('Request body is too large.'), { statusCode: 413 }));
      return;
    }
    let totalBytes = 0;
    const chunks = [];
    let tooLarge = false;

    req.on('data', (chunk) => {
      if (tooLarge) return;
      totalBytes += chunk.length;
      if (totalBytes > REQUEST_LIMIT_BYTES) {
        tooLarge = true;
        chunks.length = 0;
        reject(Object.assign(new Error('Request body is too large.'), { statusCode: 413 }));
        return;
      }
      chunks.push(chunk);
    });

    req.on('end', () => {
      if (tooLarge) return;
      try {
        const raw = Buffer.concat(chunks).toString('utf8');
        const value = raw ? JSON.parse(raw) : {};
        if (!value || typeof value !== 'object' || Array.isArray(value)) {
          throw Object.assign(new Error('The JSON request body must be an object.'), { statusCode: 400 });
        }
        resolve(value);
      } catch (error) {
        reject(error && error.statusCode
          ? error
          : Object.assign(new Error('Invalid JSON request body.'), { statusCode: 400 }));
      }
    });

    req.on('error', reject);
  });
}

function assertAuthorized(req) {
  if (ALLOW_LOCAL_BYPASS && isLocalRequest(req) && isTrustedLocalBrowserRequest(req)) {
    return;
  }

  const token = getRequestToken(req);
  if (!isValidAccessToken(token)) {
    throw Object.assign(new Error('Access token required. Open the dashboard on the host machine and scan the QR code.'), {
      statusCode: 401,
      localAccessRequired: true,
    });
  }
}

function assertLocalDashboardRequest(req) {
  if (!isLocalRequest(req) || !isTrustedLocalBrowserRequest(req)) {
    throw Object.assign(new Error('The access dashboard is only available from the host machine.'), { statusCode: 403 });
  }
}

function assertLocalSetupMutationRequest(req) {
  assertLocalDashboardRequest(req);
  const host = String(req.headers.host || '').trim();
  const origin = String(req.headers.origin || '').trim();
  let expectedOrigin;
  try {
    expectedOrigin = new URL(`${protocol}://${host}`).origin;
  } catch {
    expectedOrigin = '';
  }
  if (
    !expectedOrigin
    || origin !== expectedOrigin
    || req.headers['x-local-setup'] !== '1'
    || !['', 'same-origin'].includes(String(req.headers['sec-fetch-site'] || '').trim().toLowerCase())
  ) {
    throw Object.assign(new Error('Local setup changes require the host dashboard.'), { statusCode: 403 });
  }
}

async function pickLocalFolder(req, res) {
  if (activeFolderPicker) {
    throw Object.assign(new Error('A folder chooser is already open on this computer.'), { statusCode: 409 });
  }
  const abortController = new AbortController();
  const abortPicker = () => {
    if (!res.writableEnded) abortController.abort();
  };
  res.once('close', abortPicker);
  const operation = pickFolder({ signal: abortController.signal });
  activeFolderPicker = operation;
  try {
    return await operation;
  } finally {
    res.off('close', abortPicker);
    if (activeFolderPicker === operation) activeFolderPicker = null;
  }
}

function getLocalSetupStatus() {
  const modelIds = managedTextBackend.models.map((model) => model.id);
  return {
    text: {
      runtimeConfigured: isExistingFile(process.env.TEXT_SERVER_EXE),
      folderConfigured: Boolean(process.env.TEXT_MODEL_PATH || process.env.TEXT_MODELS_ROOT),
      managedEnabled: managedTextBackend.enabled,
      modelCount: modelIds.length,
      models: modelIds,
      folderLockedByEnvironment: isTextFolderLocked(),
    },
  };
}

async function updateLocalTextModelsRoot(selectedPath) {
  if (isTextFolderLocked()) {
    throw Object.assign(new Error('The text model location is controlled by an environment override.'), {
      statusCode: 409,
    });
  }
  if (typeof selectedPath !== 'string' || !selectedPath.trim()) {
    throw Object.assign(new Error('Choose a model folder first.'), { statusCode: 400 });
  }

  let selected;
  try {
    selected = normalizeLocalConfig(
      { version: CONFIG_VERSION, textModelsRoot: selectedPath.trim() },
      { checkExisting: true },
    ).textModelsRoot;
  } catch (error) {
    if (error?.code === 'LOCAL_CONFIG_ERROR') error.statusCode = 400;
    throw error;
  }
  const currentConfig = readLocalConfig();
  const nextConfig = { ...currentConfig, textModelsRoot: selected };
  delete nextConfig.textModelPath;
  delete nextConfig.externalTextBaseUrl;

  const candidateEnvironment = makeTextEnvironment(nextConfig);
  if (activeTextRequests || textBackendReconfiguring) {
    throw Object.assign(new Error('Wait for the current text request to finish before changing model folders.'), {
      statusCode: 409,
    });
  }

  textBackendReconfiguring = true;
  const previousBackend = managedTextBackend;
  try {
    const catalogEntries = await discoverManagedTextCatalog(candidateEnvironment);
    if (!catalogEntries.length) {
      throw Object.assign(new Error('No compatible GGUF models were found in the selected folder.'), {
        statusCode: 400,
      });
    }
    try {
      normalizeLocalConfig(
        { version: CONFIG_VERSION, textModelsRoot: selected },
        { checkExisting: true },
      );
    } catch (error) {
      if (error?.code === 'LOCAL_CONFIG_ERROR') error.statusCode = 400;
      throw error;
    }
    const candidateBackend = createManagedTextBackend({
      env: candidateEnvironment,
      catalogEntries,
    });
    saveLocalConfig(nextConfig, { checkExisting: false });
    try {
      await previousBackend.stop();
    } catch {
      try { saveLocalConfig(currentConfig, { checkExisting: false }); } catch {}
      throw Object.assign(new Error('The current text engine could not be stopped safely.'), { statusCode: 503 });
    }
    applyTextEnvironment(candidateEnvironment);
    managedTextBackend = candidateBackend;
    DEFAULT_TEXT_BASE_URL = resolveConfiguredTextBaseUrl(candidateEnvironment, candidateBackend);
    activeTextBackendKey = '';
    return getLocalSetupStatus();
  } finally {
    textBackendReconfiguring = false;
  }
}

async function refreshLocalTextModels() {
  if (activeTextRequests || textBackendReconfiguring) {
    throw Object.assign(new Error('Wait for the current text request to finish before refreshing models.'), {
      statusCode: 409,
    });
  }
  const candidateEnvironment = makeInitialManagedTextEnvironment();
  if (
    (!candidateEnvironment.TEXT_MODEL_PATH && !candidateEnvironment.TEXT_MODELS_ROOT)
    || candidateEnvironment.TEXT_BASE_URL
    || String(candidateEnvironment.TEXT_BACKEND || '').trim().toLowerCase() === 'external'
  ) {
    throw Object.assign(new Error('A managed text model location is not configured.'), { statusCode: 409 });
  }

  textBackendReconfiguring = true;
  const previousBackend = managedTextBackend;
  try {
    const catalogEntries = await discoverManagedTextCatalog(candidateEnvironment);
    const candidateBackend = createManagedTextBackend({
      env: candidateEnvironment,
      catalogEntries,
    });
    try {
      await previousBackend.stop();
    } catch {
      throw Object.assign(new Error('The current text engine could not be stopped safely.'), { statusCode: 503 });
    }
    managedTextBackend = candidateBackend;
    DEFAULT_TEXT_BASE_URL = resolveConfiguredTextBaseUrl(candidateEnvironment, candidateBackend);
    activeTextBackendKey = '';
    return getLocalSetupStatus();
  } finally {
    textBackendReconfiguring = false;
  }
}

function makeTextEnvironment(config) {
  const environment = { ...process.env };
  for (const key of TEXT_CONFIG_ENVIRONMENT_KEYS) {
    if (!explicitTextEnvironmentKeys.has(key)) delete environment[key];
  }
  for (const [field, rule] of Object.entries(FIELD_RULES)) {
    if (!TEXT_CONFIG_ENVIRONMENT_KEYS.includes(rule.environment)) continue;
    if (!Object.prototype.hasOwnProperty.call(environment, rule.environment) && config[field]) {
      environment[rule.environment] = config[field];
    }
  }
  return environment;
}

function makeInitialManagedTextEnvironment() {
  const environment = { ...process.env };
  if (
    environment.TEXT_MODELS_ROOT
    && environment.TEXT_MODEL_PATH
    && !explicitTextEnvironmentKeys.has('TEXT_MODEL_PATH')
  ) {
    delete environment.TEXT_MODEL_PATH;
  }
  return environment;
}

function createInitialManagedTextBackend() {
  const environment = makeInitialManagedTextEnvironment();
  try {
    return createManagedTextBackend({ env: environment });
  } catch (error) {
    if (error?.code !== 'TEXT_MODEL_DISCOVERY_LIMIT') throw error;
    const degradedEnvironment = { ...environment };
    delete degradedEnvironment.TEXT_MODEL_PATH;
    delete degradedEnvironment.TEXT_MODELS_ROOT;
    console.error('Local text model discovery exceeded its safety limits; use the host dashboard to choose a narrower folder.');
    return createManagedTextBackend({ env: degradedEnvironment });
  }
}

function applyTextEnvironment(environment) {
  for (const key of TEXT_CONFIG_ENVIRONMENT_KEYS) {
    if (explicitTextEnvironmentKeys.has(key)) continue;
    if (Object.prototype.hasOwnProperty.call(environment, key)) {
      process.env[key] = environment[key];
    } else {
      delete process.env[key];
    }
  }
}

function isTextFolderLocked() {
  return isEnabledEnvironmentValue(process.env.LOCAL_CONFIG_DISABLED)
    || ['TEXT_MODEL_PATH', 'TEXT_MODELS_ROOT', 'TEXT_BASE_URL'].some((key) => explicitTextEnvironmentKeys.has(key))
    || String(process.env.TEXT_BACKEND || '').trim().toLowerCase() === 'external';
}

function isExistingFile(candidate) {
  if (!candidate) return false;
  try {
    return fs.statSync(path.resolve(candidate)).isFile();
  } catch {
    return false;
  }
}

function isEnabledEnvironmentValue(value) {
  return ['1', 'true', 'on', 'yes'].includes(String(value || '').trim().toLowerCase());
}

function getRequestToken(req) {
  const headerToken = req.headers['x-access-token'];
  if (typeof headerToken === 'string' && headerToken.trim()) {
    return headerToken.trim();
  }

  const authorization = req.headers.authorization;
  if (typeof authorization === 'string' && authorization.toLowerCase().startsWith('bearer ')) {
    return authorization.slice(7).trim();
  }

  return '';
}

function isValidAccessToken(token) {
  if (!token || token.length !== ACCESS_TOKEN.length) {
    return false;
  }

  const tokenBuffer = Buffer.from(token);
  const expectedBuffer = Buffer.from(ACCESS_TOKEN);
  if (tokenBuffer.length !== expectedBuffer.length) {
    return false;
  }

  return crypto.timingSafeEqual(tokenBuffer, expectedBuffer);
}

function isLocalRequest(req) {
  const remoteAddress = normalizeRemoteAddress(req.socket.remoteAddress || '');
  return remoteAddress === '::1' || remoteAddress === 'localhost' || remoteAddress.startsWith('127.');
}

function normalizeRemoteAddress(address) {
  if (address.startsWith('::ffff:')) {
    return address.slice(7);
  }
  return address.toLowerCase();
}

function makeAccessUrl(address) {
  return `${protocol}://${address}:${PORT}/#access=${ACCESS_TOKEN}`;
}

function makeAccessToken() {
  return crypto.randomBytes(24).toString('base64url');
}

function normalizeBoolean(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  return !['0', 'false', 'off', 'no'].includes(String(value).trim().toLowerCase());
}

function normalizeMessages(value) {
  if (!Array.isArray(value)) return [];
  if (value.some((message) => !message || typeof message !== 'object' || Array.isArray(message))) {
    throw Object.assign(new Error('Every chat message must be an object.'), { statusCode: 400 });
  }
  return value
    .map((message) => normalizeMessage(message))
    .filter(Boolean);
}

function normalizeChatTools(value) {
  if (!Array.isArray(value)) return [];
  const tools = [];
  for (const item of value.slice(0, 4)) {
    if (!item || item.type !== 'function' || !item.function || typeof item.function !== 'object') continue;
    const name = typeof item.function.name === 'string' ? item.function.name.trim() : '';
    if (!name) continue;
    const tool = { type: 'function', function: { name } };
    if (typeof item.function.description === 'string') {
      tool.function.description = item.function.description.slice(0, 2000);
    }
    if (item.function.parameters && typeof item.function.parameters === 'object' && !Array.isArray(item.function.parameters)) {
      tool.function.parameters = item.function.parameters;
    }
    tools.push(tool);
  }
  return tools;
}

function normalizeToolCalls(value) {
  if (!Array.isArray(value)) return [];
  const calls = [];
  for (const item of value.slice(0, 8)) {
    const id = item && typeof item.id === 'string' ? item.id : '';
    const fn = item && item.function && typeof item.function === 'object' ? item.function : null;
    const name = fn && typeof fn.name === 'string' ? fn.name : '';
    if (!id || !name) continue;
    calls.push({
      id,
      type: 'function',
      function: {
        name,
        arguments: fn && typeof fn.arguments === 'string' ? fn.arguments : '',
      },
    });
  }
  return calls;
}

function applyThinkingMode(messages, thinkingMode) {
  if (!['on', 'off'].includes(thinkingMode)) return messages;

  const lastUserIndex = findLastUserMessageIndex(messages);
  if (lastUserIndex === -1) return messages;

  const directive = thinkingMode === 'on' ? '/think' : '/no_think';
  return messages.map((message, index) => {
    if (index !== lastUserIndex) return message;
    return {
      ...message,
      content: `${directive}\n${stripLeadingThinkingDirective(message.content)}`,
    };
  });
}

function findLastUserMessageIndex(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user') return index;
  }
  return -1;
}

function stripLeadingThinkingDirective(content) {
  return content.replace(/^\s*\/(?:no_)?think(?:\s*\r?\n|\s+)/i, '').trimStart();
}

function normalizeThinkingMode(value) {
  const mode = String(value || '').trim().toLowerCase();
  return ['auto', 'on', 'off'].includes(mode) ? mode : 'auto';
}

function normalizeMessage(message) {
  if (!message || typeof message !== 'object') return null;
  const role = typeof message.role === 'string' ? message.role : '';
  const content = typeof message.content === 'string' ? message.content : '';

  if (role === 'tool') {
    const toolCallId = typeof message.tool_call_id === 'string' ? message.tool_call_id.trim() : '';
    if (!toolCallId || !content.trim()) return null;
    return { role, content, tool_call_id: toolCallId };
  }

  if (!['system', 'user', 'assistant'].includes(role)) return null;

  if (role === 'assistant') {
    const cleaned = stripThinkingFromText(content);
    const toolCalls = normalizeToolCalls(message.tool_calls);
    if (toolCalls.length) return { role, content: cleaned, tool_calls: toolCalls };
    if (!cleaned.trim()) return null;
    return { role, content: cleaned };
  }

  if (!content.trim()) return null;
  return { role, content };
}

function stripThinkingFromText(text) {
  const parser = createThinkingParser();
  const parsed = parser.append(text);
  const finalDelta = parser.flush();
  return parsed.content + finalDelta.content;
}

function createThinkingParser() {
  const startTag = '<think>';
  const endTag = '</think>';
  let mode = 'content';
  let tagBuffer = '';

  return {
    append(text) {
      const result = { content: '', reasoning: '' };

      for (const character of text) {
        if (tagBuffer || character === '<') {
          tagBuffer += character;
          const normalized = tagBuffer.toLowerCase();

          if (normalized === startTag) {
            mode = 'reasoning';
            tagBuffer = '';
            continue;
          }

          if (normalized === endTag) {
            mode = 'content';
            tagBuffer = '';
            continue;
          }

          if (startTag.startsWith(normalized) || endTag.startsWith(normalized)) {
            continue;
          }

          appendThinkingPart(result, mode, tagBuffer);
          tagBuffer = '';
          continue;
        }

        appendThinkingPart(result, mode, character);
      }

      return result;
    },
    flush() {
      const result = { content: '', reasoning: '' };
      if (tagBuffer) {
        appendThinkingPart(result, mode, tagBuffer);
        tagBuffer = '';
      }
      return result;
    },
  };
}

function appendThinkingPart(result, mode, text) {
  if (mode === 'reasoning') {
    result.reasoning += text;
  } else {
    result.content += text;
  }
}

async function handleImageConfigRequest(req, res) {
  const [models, loras, runtime, animaTextEncoder, animaVae] = await Promise.all([
    discoverImageModels(),
    discoverImageLoras(),
    getImageRuntimeStatus(),
    localFileExists(ANIMA_TEXT_ENCODER_PATH),
    localFileExists(ANIMA_VAE_PATH),
  ]);

  sendJson(res, 200, {
    connected: Boolean(runtime.ok),
    models,
    loras,
    runtime,
    dependencies: {
      animaTextEncoder,
      animaVae,
      tokenizerAssets: Boolean(runtime.tokenizerAssets),
    },
  });
}

async function handleImageGenerationRequest(req, res) {
  const payload = await readJsonBody(req);
  const signal = abortWhenClientDisconnects(req, res);
  const kind = payload.kind === 'anima' ? 'anima' : payload.kind === 'sdxl' ? 'sdxl' : '';
  const prompt = normalizePromptText(payload.prompt, 5000);
  const negativePrompt = normalizePromptText(payload.negativePrompt, 3000, true);
  const size = IMAGE_SIZES.get(String(payload.size || 'square')) || IMAGE_SIZES.get('square');
  const seed = normalizeSeed(payload.seed);

  if (!kind) {
    sendJson(res, 400, { error: 'Choose Anima or SDXL image generation.' });
    return;
  }
  if (!prompt) {
    sendJson(res, 400, { error: 'Describe the image you want to generate.' });
    return;
  }

  const [models, availableLoras] = await Promise.all([discoverImageModels(), discoverImageLoras()]);
  const allowedModels = models[kind];
  const requestedModel = String(payload.model || '');
  const selectedModel = allowedModels.find((model) => model.id === requestedModel);
  if (!selectedModel) {
    sendJson(res, 400, { error: 'The selected image model is not available.' });
    return;
  }
  const selectedLoras = normalizeLoraSelections(payload.loras, availableLoras[kind]);

  if (kind === 'anima') {
    const dependenciesReady =
      (await localFileExists(ANIMA_TEXT_ENCODER_PATH)) &&
      (await localFileExists(ANIMA_VAE_PATH));
    if (!dependenciesReady) {
      sendJson(res, 400, { error: 'Anima text-encoder and VAE files are not configured.' });
      return;
    }
  }

  const steps = normalizeInteger(payload.steps, 1, 80, selectedModel.recommendedSteps);
  const cfg = normalizeNumber(payload.cfg, 0, 20, selectedModel.recommendedCfg);
  const runtime = await getImageRuntimeStatus();
  if (!runtime.ok) {
    sendJson(res, 503, { error: runtime.error || 'The direct CUDA image engine is not ready.' });
    return;
  }
  if (activeTextRequests > 0) {
    sendJson(res, 409, { error: 'Image generation will be available when the current text request finishes.' });
    return;
  }
  if (activeImageProcess || imageGpuReserved) {
    sendJson(res, 409, { error: 'Another image is already generating on this GPU.' });
    return;
  }

  const workerPayload = {
    kind,
    modelPath: selectedModel.filePath,
    prompt,
    negativePrompt: negativePrompt || (kind === 'anima'
      ? 'worst quality, low quality, blurry, jpeg artifacts'
      : 'worst quality, low quality, blurry, jpeg artifacts, watermark, text'),
    width: size.width,
    height: size.height,
    steps,
    cfg,
    seed,
    loras: selectedLoras.map((lora) => ({ path: lora.filePath, strength: lora.strength })),
  };
  if (kind === 'anima') {
    workerPayload.textEncoderPath = ANIMA_TEXT_ENCODER_PATH;
    workerPayload.vaePath = ANIMA_VAE_PATH;
  }

  if (signal.aborted) return;
  imageGpuReserved = true;
  try {
    await managedTextBackend.stop();
    const image = await runImageWorker(workerPayload, { signal, exclusive: true });
    if (!res.destroyed) {
      sendJson(res, 200, {
        imageBase64: image.imageBase64,
        mimeType: image.mimeType || 'image/png',
        seed,
        model: selectedModel.id,
        kind,
        width: size.width,
        height: size.height,
        steps,
        cfg,
        loras: selectedLoras.map((lora) => ({ id: lora.id, strength: lora.strength })),
      });
    }
  } catch (error) {
    if (signal.aborted || res.destroyed) return;
    throw error;
  } finally {
    imageGpuReserved = false;
  }
}

function isTrustedLocalBrowserRequest(req) {
  const host = String(req.headers.host || '').trim();
  let hostname;
  try {
    hostname = new URL(`http://${host}`).hostname.toLowerCase().replace(/\.$/, '');
  } catch {
    return false;
  }
  const bareHostname = hostname.replace(/^\[|\]$/g, '');
  if (hostname !== 'localhost' && !isLoopbackHost(bareHostname)) {
    return false;
  }
  const fetchSite = String(req.headers['sec-fetch-site'] || '').toLowerCase();
  if (fetchSite === 'cross-site') return false;
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    return new URL(origin).origin === new URL(`${protocol}://${host}`).origin;
  } catch {
    return false;
  }
}

async function discoverImageModels() {
  if (!IMAGE_MODELS_ROOT) return { anima: [], sdxl: [] };
  const [animaFiles, checkpointFiles] = await Promise.all([
    listModelFiles(path.join(IMAGE_MODELS_ROOT, 'diffusion_models')),
    listModelFiles(path.join(IMAGE_MODELS_ROOT, 'checkpoints')),
  ]);

  const anima = [];
  for (const file of animaFiles) {
    if (await isAnimaCheckpoint(file.path)) anima.push(makeImageModelDescriptor('anima', file));
  }
  const sdxl = [];
  for (const file of checkpointFiles) {
    if (await isSdxlCheckpoint(file.path)) sdxl.push(makeImageModelDescriptor('sdxl', file));
  }

  return { anima, sdxl };
}

async function discoverImageLoras() {
  if (!IMAGE_MODELS_ROOT) return { anima: [], sdxl: [] };
  const files = await listModelFiles(path.join(IMAGE_MODELS_ROOT, 'loras'));
  const loras = { anima: [], sdxl: [] };
  for (const file of files) {
    const families = await classifyImageLora(file.path);
    for (const family of families) loras[family].push(makeImageLoraDescriptor(file));
  }
  return loras;
}

async function classifyImageLora(filePath) {
  let stats;
  try {
    stats = await fs.promises.stat(filePath);
  } catch {
    return [];
  }
  const cacheKey = `lora:${filePath}:${stats.size}:${stats.mtimeMs}`;
  if (modelTypeCache.has(cacheKey)) return modelTypeCache.get(cacheKey);

  const families = [];
  try {
    const headerBuffer = await readSafetensorsHeader(filePath);
    if (!headerBuffer) return [];
    const header = JSON.parse(headerBuffer.toString('utf8'));
    const keys = Object.keys(header).filter((key) => key !== '__metadata__');
    if (keys.some((key) => key.endsWith('.dora_scale'))) {
      modelTypeCache.set(cacheKey, families);
      return families;
    }
    const hasPair = (base, downSuffix, upSuffix) =>
      isComposableLoraPair(header[base + downSuffix], header[base + upSuffix]);
    const kohyaBases = keys
      .filter((key) => key.endsWith('.lora_down.weight'))
      .map((key) => key.slice(0, -'.lora_down.weight'.length));
    const peftBases = keys
      .filter((key) => key.endsWith('.lora_A.weight'))
      .map((key) => key.slice(0, -'.lora_A.weight'.length));

    const animaKohya = kohyaBases.some((base) =>
      base.startsWith('lora_unet_blocks_') && hasPair(base, '.lora_down.weight', '.lora_up.weight'),
    );
    const animaPeft = peftBases.some((base) =>
      base.startsWith('diffusion_model.blocks.') &&
      (base.includes('.adaln_modulation_') || base.includes('.self_attn.q_proj')) &&
      hasPair(base, '.lora_A.weight', '.lora_B.weight'),
    );
    if (animaKohya || animaPeft) families.push('anima');

    const sdxlKohya = kohyaBases.some((base) =>
      (
        base.startsWith('lora_unet_input_blocks_') ||
        base.startsWith('lora_unet_middle_block_') ||
        base.startsWith('lora_unet_output_blocks_') ||
        base.startsWith('lora_te1_text_model_encoder_layers_') ||
        base.startsWith('lora_te2_text_model_encoder_layers_')
      ) && hasPair(base, '.lora_down.weight', '.lora_up.weight'),
    );
    if (sdxlKohya) families.push('sdxl');
  } catch {
    // Unsupported or damaged files are omitted from the picker.
  }
  modelTypeCache.set(cacheKey, families);
  return families;
}

function hasLoraTensor(entry) {
  return Boolean(
    entry &&
    DIRECT_WEIGHT_DTYPES.has(entry.dtype) &&
    Array.isArray(entry.shape) &&
    (entry.shape.length === 2 || entry.shape.length === 4) &&
    Array.isArray(entry.data_offsets) &&
    entry.data_offsets.length === 2
  );
}

function isComposableLoraPair(down, up) {
  if (!hasLoraTensor(down) || !hasLoraTensor(up)) return false;
  if (down.shape.length !== up.shape.length || up.shape[1] !== down.shape[0]) return false;
  if (down.shape.length === 2) return true;
  const downPointwise = down.shape[2] === 1 && down.shape[3] === 1;
  const upPointwise = up.shape[2] === 1 && up.shape[3] === 1;
  return downPointwise || upPointwise;
}

async function listModelFiles(directory, relativeDirectory = '') {
  let entries;
  try {
    entries = await fs.promises.readdir(path.join(directory, relativeDirectory), { withFileTypes: true });
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }

  const files = [];
  for (const entry of entries) {
    const relativePath = path.join(relativeDirectory, entry.name);
    if (entry.isDirectory()) {
      files.push(...await listModelFiles(directory, relativePath));
      continue;
    }
    if (!entry.isFile() || !IMAGE_FILE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) continue;
    files.push({
      id: relativePath.split(path.sep).join('/'),
      path: path.join(directory, relativePath),
    });
  }
  return files.sort((a, b) => a.id.localeCompare(b.id, undefined, { sensitivity: 'base' }));
}

async function isAnimaCheckpoint(filePath) {
  let stats;
  try {
    stats = await fs.promises.stat(filePath);
  } catch {
    return false;
  }

  const cacheKey = `anima:${filePath}:${stats.size}:${stats.mtimeMs}`;
  if (modelTypeCache.has(cacheKey)) return modelTypeCache.get(cacheKey);

  let compatible = false;
  try {
    const headerBuffer = await readSafetensorsHeader(filePath);
    if (!headerBuffer) return false;
    const header = JSON.parse(headerBuffer.toString('utf8'));
    compatible = ANIMA_MODEL_PREFIXES.some((prefix) =>
      [...ANIMA_MODEL_SIGNATURE].every(([name, shape]) =>
        hasTensorSignature(header[prefix + name], shape),
      ),
    );
  } catch {
    compatible = false;
  }
  modelTypeCache.set(cacheKey, compatible);
  return compatible;
}

function hasTensorSignature(entry, expectedShape) {
  return Boolean(
    entry &&
    DIRECT_WEIGHT_DTYPES.has(entry.dtype) &&
    Array.isArray(entry.shape) &&
    entry.shape.length === expectedShape.length &&
    entry.shape.every((value, index) => value === expectedShape[index]) &&
    Array.isArray(entry.data_offsets) &&
    entry.data_offsets.length === 2
  );
}

async function isSdxlCheckpoint(filePath) {
  let stats;
  try {
    stats = await fs.promises.stat(filePath);
  } catch {
    return false;
  }

  const cacheKey = `sdxl:${filePath}:${stats.size}:${stats.mtimeMs}`;
  if (modelTypeCache.has(cacheKey)) return modelTypeCache.get(cacheKey);
  if (path.extname(filePath).toLowerCase() !== '.safetensors') {
    modelTypeCache.set(cacheKey, true);
    return true;
  }

  try {
    const headerBuffer = await readSafetensorsHeader(filePath);
    if (!headerBuffer) return false;
    const header = JSON.parse(headerBuffer.toString('utf8'));
    const isSdxl = [...SDXL_MODEL_SIGNATURE].every(([name, shape]) =>
      hasTensorSignature(header[name], shape),
    );
    modelTypeCache.set(cacheKey, isSdxl);
    return isSdxl;
  } catch {
    modelTypeCache.set(cacheKey, false);
    return false;
  }
}

async function readSafetensorsHeader(filePath) {
  let handle;
  try {
    handle = await fs.promises.open(filePath, 'r');
    const lengthBuffer = Buffer.alloc(8);
    if (!await readFileChunk(handle, lengthBuffer, 0)) return null;
    const headerLength = Number(lengthBuffer.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > 32 * 1024 * 1024) {
      return null;
    }
    const headerBuffer = Buffer.alloc(headerLength);
    if (!await readFileChunk(handle, headerBuffer, 8)) return null;
    return headerBuffer;
  } finally {
    if (handle) await handle.close();
  }
}

async function readFileChunk(handle, buffer, position) {
  let offset = 0;
  while (offset < buffer.length) {
    const { bytesRead } = await handle.read(buffer, offset, buffer.length - offset, position + offset);
    if (!bytesRead) return false;
    offset += bytesRead;
  }
  return true;
}

function makeImageModelDescriptor(kind, file) {
  const accelerated = /(?:turbo|lightning|dmd|lcm|hyper)/i.test(file.id);
  const descriptor = {
    id: file.id,
    label: path.basename(file.id, path.extname(file.id)),
    recommendedSteps: accelerated ? 8 : kind === 'anima' ? 30 : 28,
    recommendedCfg: accelerated ? 1.5 : kind === 'anima' ? 4 : 6,
  };
  Object.defineProperty(descriptor, 'filePath', { value: file.path, enumerable: false });
  return descriptor;
}

function makeImageLoraDescriptor(file) {
  const descriptor = {
    id: file.id,
    label: path.basename(file.id, path.extname(file.id)),
  };
  Object.defineProperty(descriptor, 'filePath', { value: file.path, enumerable: false });
  return descriptor;
}

async function localFileExists(filePath) {
  if (!filePath) return false;
  try {
    const stats = await fs.promises.stat(filePath);
    return stats.isFile();
  } catch {
    return false;
  }
}

function defaultPythonCommand() {
  return process.platform === 'win32' ? 'python' : 'python3';
}

async function getImageRuntimeStatus() {
  if (!IMAGE_CONFIGURED) {
    return {
      ok: false,
      engine: 'Direct PyTorch/CUDA',
      gpu: '',
      cuda: '',
      torch: '',
      tokenizerAssets: false,
      error: 'Image generation is not configured. Run npm run configure.',
    };
  }
  const now = Date.now();
  if (imageRuntimeProbeCache && imageRuntimeProbeCache.expiresAt > now) {
    return imageRuntimeProbeCache.value;
  }
  if (imageRuntimeProbePromise) return imageRuntimeProbePromise;
  imageRuntimeProbePromise = probeImageRuntime();
  try {
    return await imageRuntimeProbePromise;
  } finally {
    imageRuntimeProbePromise = null;
  }
}

async function probeImageRuntime() {
  try {
    const result = await runImageWorker(null, { args: ['--probe'], timeoutMs: 20_000, allowFailure: true });
    const value = {
      ok: Boolean(result.ok),
      engine: result.engine || 'Direct PyTorch/CUDA',
      gpu: result.gpu || '',
      cuda: result.cuda || '',
      torch: result.torch || '',
      tokenizerAssets: Boolean(result.tokenizerAssets),
      error: result.ok ? '' : makeImageRuntimeError(result),
    };
    imageRuntimeProbeCache = { expiresAt: Date.now() + 30_000, value };
    return value;
  } catch {
    const value = {
      ok: false,
      engine: 'Direct PyTorch/CUDA',
      gpu: '',
      cuda: '',
      torch: '',
      tokenizerAssets: false,
      error: 'The direct image worker could not start.',
    };
    imageRuntimeProbeCache = { expiresAt: Date.now() + 5_000, value };
    return value;
  }
}

function makeImageRuntimeError(result) {
  if (!result.tokenizerAssets) return 'Run npm run setup:image once to install the verified tokenizer tables.';
  if (!result.cudaAvailable) return 'The configured Python runtime cannot access an NVIDIA CUDA GPU.';
  return result.error || 'The direct image engine is not ready.';
}

function runImageWorker(payload, options = {}) {
  if (payload !== null && options.exclusive && USE_PERSISTENT_IMAGE_WORKER) {
    return runPersistentImageWorker(payload, options);
  }
  const args = [IMAGE_WORKER_PATH, ...(options.args || [])];
  const timeoutMs = options.timeoutMs || IMAGE_GENERATION_TIMEOUT_MS;
  const signal = options.signal;
  const exclusive = Boolean(options.exclusive);
  const allowFailure = Boolean(options.allowFailure);

  return new Promise((resolve, reject) => {
    let settled = false;
    let outputBytes = 0;
    const output = [];
    const child = spawn(IMAGE_PYTHON, args, {
      cwd: ROOT,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONDONTWRITEBYTECODE: '1',
        PYTHONUNBUFFERED: '1',
        HF_HUB_OFFLINE: '1',
        TRANSFORMERS_OFFLINE: '1',
        TOKENIZERS_PARALLELISM: 'false',
      },
    });
    if (exclusive) activeImageProcess = child;

    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (signal) signal.removeEventListener('abort', abort);
      callback(value);
    };
    const abort = () => {
      child.kill();
      finish(reject, makeHttpError('Image generation was stopped.', 499));
    };
    const timeout = setTimeout(() => {
      child.kill();
      finish(reject, makeHttpError('Direct image generation timed out.', 504));
    }, timeoutMs);

    if (signal) {
      if (signal.aborted) {
        abort();
        return;
      }
      signal.addEventListener('abort', abort, { once: true });
    }

    child.stdout.on('data', (chunk) => {
      outputBytes += chunk.length;
      if (outputBytes > IMAGE_RESPONSE_LIMIT_BYTES) {
        child.kill();
        finish(reject, makeHttpError('The generated image was too large to return safely.', 502));
        return;
      }
      output.push(chunk);
    });
    consumeImageWorkerDiagnostics(child.stderr);
    child.once('exit', () => {
      if (exclusive && activeImageProcess === child) activeImageProcess = null;
    });
    child.on('error', () => {
      finish(reject, makeHttpError('The direct image worker could not start. Check IMAGE_PYTHON.', 502));
    });
    child.on('close', () => {
      if (exclusive && activeImageProcess === child) activeImageProcess = null;
      if (settled) return;
      let result;
      try {
        result = JSON.parse(Buffer.concat(output).toString('utf8'));
      } catch {
        finish(reject, makeHttpError('The direct image worker returned an invalid response.', 502));
        return;
      }
      if (!result || (!result.ok && !allowFailure)) {
        finish(reject, makeHttpError(result && result.error || 'Direct image generation failed.', 502));
        return;
      }
      finish(resolve, result);
    });

    if (payload === null) {
      child.stdin.end();
    } else {
      child.stdin.end(JSON.stringify(payload));
    }
  });
}

function runPersistentImageWorker(payload, options = {}) {
  const timeoutMs = options.timeoutMs || IMAGE_GENERATION_TIMEOUT_MS;
  const signal = options.signal;
  const worker = getPersistentImageWorker();
  if (worker.pending) {
    return Promise.reject(makeHttpError('Another image is already generating on this GPU.', 409));
  }

  return new Promise((resolve, reject) => {
    const finish = (callback, value, keepWorker = true) => {
      if (!worker.pending) return;
      clearTimeout(worker.pending.timeout);
      if (signal) signal.removeEventListener('abort', abort);
      worker.pending = null;
      if (keepWorker) {
        if (activeImageProcess === worker.child) activeImageProcess = null;
        scheduleImageWorkerIdleStop(worker);
      }
      callback(value);
    };
    const abort = () => {
      finish(reject, makeHttpError('Image generation was stopped.', 499), false);
      stopPersistentImageWorkerNow(worker);
    };
    const timeout = setTimeout(() => {
      finish(reject, makeHttpError('Direct image generation timed out.', 504), false);
      stopPersistentImageWorkerNow(worker);
    }, timeoutMs);
    worker.pending = { resolve, reject, finish, timeout };
    activeImageProcess = worker.child;

    if (signal) {
      if (signal.aborted) {
        abort();
        return;
      }
      signal.addEventListener('abort', abort, { once: true });
    }
    worker.child.stdin.write(`${JSON.stringify(payload)}\n`, (error) => {
      if (error) {
        finish(reject, makeHttpError('The direct image worker could not accept the request.', 502), false);
        stopPersistentImageWorkerNow(worker);
      }
    });
  });
}

function getPersistentImageWorker() {
  if (persistentImageWorker && isProcessRunning(persistentImageWorker.child)) {
    clearTimeout(persistentImageWorker.idleTimer);
    return persistentImageWorker;
  }

  const child = spawn(IMAGE_PYTHON, [IMAGE_WORKER_PATH, '--serve'], {
    cwd: ROOT,
    windowsHide: true,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONDONTWRITEBYTECODE: '1',
      PYTHONUNBUFFERED: '1',
      HF_HUB_OFFLINE: '1',
      TRANSFORMERS_OFFLINE: '1',
      TOKENIZERS_PARALLELISM: 'false',
    },
  });
  const worker = {
    child,
    buffer: '',
    outputBytes: 0,
    pending: null,
    idleTimer: null,
  };
  persistentImageWorker = worker;
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => handlePersistentImageOutput(worker, chunk));
  consumeImageWorkerDiagnostics(child.stderr);
  child.on('error', () => failPersistentImageWorker(worker, 'The direct image worker could not start. Check IMAGE_PYTHON.'));
  child.once('exit', () => {
    if (activeImageProcess === child) activeImageProcess = null;
    failPersistentImageWorker(worker, 'The direct image worker exited unexpectedly.');
  });
  child.once('close', () => {
    if (activeImageProcess === child) activeImageProcess = null;
  });
  return worker;
}

function consumeImageWorkerDiagnostics(stream) {
  if (process.env.IMAGE_PROFILE !== '1') {
    stream.resume();
    return;
  }
  stream.setEncoding('utf8');
  let buffer = '';
  stream.on('data', (chunk) => {
    buffer = `${buffer}${chunk}`;
    if (buffer.length > 16_384) buffer = buffer.slice(-16_384);
    let newline = buffer.indexOf('\n');
    while (newline >= 0) {
      const line = buffer.slice(0, newline).replace(/\r$/, '');
      buffer = buffer.slice(newline + 1);
      const profile = normalizeImageProfileLine(line);
      if (profile) console.error(`IMAGE_PROFILE ${JSON.stringify(profile)}`);
      newline = buffer.indexOf('\n');
    }
  });
}

function normalizeImageProfileLine(line) {
  if (!line.startsWith('IMAGE_PROFILE ')) return null;
  let value;
  try {
    value = JSON.parse(line.slice('IMAGE_PROFILE '.length));
  } catch {
    return null;
  }
  const pipelines = new Set(['anima', 'anima-warm', 'sdxl', 'sdxl-warm']);
  const stageNames = new Set([
    'text_encoder', 'text_encoders', 'denoiser_load', 'conditioning',
    'clip_l_load', 'clip_l_encode', 'clip_g_load', 'clip_g_encode',
    'sampling', 'decoder_load', 'decode', 'png',
  ]);
  if (!value || !pipelines.has(value.pipeline) || !value.stagesSeconds || typeof value.stagesSeconds !== 'object') {
    return null;
  }
  const stagesSeconds = {};
  for (const [name, duration] of Object.entries(value.stagesSeconds)) {
    if (!stageNames.has(name) || !Number.isFinite(duration) || duration < 0 || duration > 7_200) return null;
    stagesSeconds[name] = duration;
  }
  if (!Object.keys(stagesSeconds).length) return null;
  const stagePeakVramMiB = normalizeProfileStageNumbers(value.stagePeakVramMiB, stageNames);
  const stageEndVramMiB = normalizeProfileStageNumbers(value.stageEndVramMiB, stageNames);
  if (value.stagePeakVramMiB && !stagePeakVramMiB) return null;
  if (value.stageEndVramMiB && !stageEndVramMiB) return null;
  if (!Number.isFinite(value.totalSeconds) || value.totalSeconds < 0 || value.totalSeconds > 7_200) return null;
  if (!Number.isFinite(value.peakVramMiB) || value.peakVramMiB < 0 || value.peakVramMiB > 1_000_000) return null;
  return {
    pipeline: value.pipeline,
    stagesSeconds,
    ...(stagePeakVramMiB ? { stagePeakVramMiB } : {}),
    ...(stageEndVramMiB ? { stageEndVramMiB } : {}),
    totalSeconds: value.totalSeconds,
    peakVramMiB: value.peakVramMiB,
  };
}

function normalizeProfileStageNumbers(value, stageNames) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const result = {};
  for (const [name, amount] of Object.entries(value)) {
    if (!stageNames.has(name) || !Number.isFinite(amount) || amount < 0 || amount > 1_000_000) return null;
    result[name] = amount;
  }
  return Object.keys(result).length ? result : null;
}

function handlePersistentImageOutput(worker, chunk) {
  worker.outputBytes += Buffer.byteLength(chunk);
  if (worker.outputBytes > IMAGE_RESPONSE_LIMIT_BYTES) {
    failPersistentImageWorker(worker, 'The generated image was too large to return safely.');
    stopPersistentImageWorkerNow(worker);
    return;
  }
  worker.buffer += chunk;
  let newline = worker.buffer.indexOf('\n');
  while (newline >= 0) {
    const line = worker.buffer.slice(0, newline).replace(/\r$/, '');
    worker.buffer = worker.buffer.slice(newline + 1);
    worker.outputBytes = Buffer.byteLength(worker.buffer);
    if (!worker.pending) {
      failPersistentImageWorker(worker, 'The direct image worker returned an unexpected extra response.');
      stopPersistentImageWorkerNow(worker);
      return;
    }
    let result;
    try {
      result = JSON.parse(line);
    } catch {
      failPersistentImageWorker(worker, 'The direct image worker returned an invalid response.');
      stopPersistentImageWorkerNow(worker);
      return;
    }
    if (!result || !result.ok) {
      worker.pending.finish(worker.pending.reject, makeHttpError(result && result.error || 'Direct image generation failed.', 502));
    } else {
      worker.pending.finish(worker.pending.resolve, result);
    }
    newline = worker.buffer.indexOf('\n');
  }
}

function failPersistentImageWorker(worker, message) {
  if (worker.pending) {
    worker.pending.finish(worker.pending.reject, makeHttpError(message, 502), false);
  }
  if (persistentImageWorker === worker) persistentImageWorker = null;
}

function scheduleImageWorkerIdleStop(worker) {
  clearTimeout(worker.idleTimer);
  if (IMAGE_WORKER_IDLE_MS === 0) {
    void stopPersistentImageWorker(worker).catch(() => {});
    return;
  }
  worker.idleTimer = setTimeout(() => {
    void stopPersistentImageWorker(worker).catch(() => {});
  }, IMAGE_WORKER_IDLE_MS);
  worker.idleTimer.unref();
}

async function stopPersistentImageWorker(expectedWorker = null) {
  if (persistentImageWorkerStopPromise) return persistentImageWorkerStopPromise;
  const worker = persistentImageWorker;
  if (!worker || (expectedWorker && worker !== expectedWorker)) return;
  persistentImageWorkerStoppingChild = worker.child;
  const stopping = stopPersistentImageWorkerInstance(worker);
  persistentImageWorkerStopPromise = stopping;
  try {
    await stopping;
  } finally {
    if (persistentImageWorkerStopPromise === stopping) persistentImageWorkerStopPromise = null;
    if (persistentImageWorkerStoppingChild === worker.child) persistentImageWorkerStoppingChild = null;
  }
}

async function stopPersistentImageWorkerInstance(worker) {
  if (isProcessRunning(worker.child)) activeImageProcess = worker.child;
  const exited = !isProcessRunning(worker.child)
    ? Promise.resolve(true)
    : new Promise((resolve) => worker.child.once('exit', () => resolve(true)));
  stopPersistentImageWorkerNow(worker);
  if (!isProcessRunning(worker.child)) {
    if (activeImageProcess === worker.child) activeImageProcess = null;
    return;
  }
  const stoppedGracefully = await Promise.race([
    exited,
    unrefDelay(2000, false),
  ]);
  if (!stoppedGracefully && isProcessRunning(worker.child)) {
    worker.child.kill('SIGKILL');
    await Promise.race([
      exited,
      unrefDelay(1000),
    ]);
  }
  if (isProcessRunning(worker.child)) {
    throw makeHttpError('The image engine could not release the GPU safely.', 503);
  }
  if (activeImageProcess === worker.child) activeImageProcess = null;
}

function stopPersistentImageWorkerNow(worker = persistentImageWorker) {
  if (!worker) return;
  clearTimeout(worker.idleTimer);
  if (isProcessRunning(worker.child)) {
    activeImageProcess = worker.child;
    worker.child.kill();
  }
  if (persistentImageWorker === worker) persistentImageWorker = null;
}

function isProcessRunning(processHandle) {
  return Boolean(processHandle)
    && processHandle.exitCode === null
    && processHandle.signalCode == null;
}

function unrefDelay(milliseconds, value) {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(value), milliseconds);
    timeout.unref?.();
  });
}

function normalizePromptText(value, maxLength, allowEmpty = false) {
  const text = typeof value === 'string' ? value.trim() : '';
  if (!text && allowEmpty) return '';
  return text.slice(0, maxLength);
}

function normalizeSeed(value) {
  const randomSeed = () => crypto.randomInt(0, 2 ** 48 - 1);
  if (value === '' || value === undefined || value === null) return randomSeed();
  return normalizeInteger(value, 0, Number.MAX_SAFE_INTEGER, randomSeed());
}

function normalizeLoraSelections(value, availableLoras) {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || value.length > MAX_IMAGE_LORAS) {
    throw makeHttpError(`Choose no more than ${MAX_IMAGE_LORAS} LoRAs.`, 400);
  }
  const selected = [];
  const seen = new Set();
  for (const item of value) {
    const id = item && typeof item.id === 'string' ? item.id : '';
    const strength = Number(item && item.strength);
    const descriptor = availableLoras.find((lora) => lora.id === id);
    if (!descriptor) throw makeHttpError('A selected LoRA is not available for this model family.', 400);
    if (seen.has(id)) throw makeHttpError('Choose each LoRA only once.', 400);
    if (!Number.isFinite(strength) || strength < -2 || strength > 2) {
      throw makeHttpError('LoRA strength must be between -2 and 2.', 400);
    }
    seen.add(id);
    if (strength) selected.push({
      id: descriptor.id,
      label: descriptor.label,
      filePath: descriptor.filePath,
      strength,
    });
  }
  return selected;
}

function makeHttpError(message, statusCode) {
  return Object.assign(new Error(message), { statusCode });
}

function normalizeBaseUrl(value) {
  const input = String(value || '').trim();
  if (!input) {
    throw Object.assign(new Error('Configure a text backend before using text generation.'), { statusCode: 400 });
  }
  try {
    const parsed = new URL(input);
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      throw Object.assign(new Error('Only HTTP and HTTPS text backend URLs are supported.'), { statusCode: 400 });
    }
    if (parsed.username || parsed.password) {
      throw unsafeTextBackendUrlError();
    }
    parsed.hash = '';
    parsed.search = '';
    parsed.pathname = parsed.pathname.replace(/\/+$/, '') || '/v1';
    return parsed.toString().replace(/\/$/, '');
  } catch (error) {
    if (error.statusCode === 400) throw error;
    throw Object.assign(new Error('The text backend base URL is invalid.'), { statusCode: 400 });
  }
}

function assertAllowedBaseUrl(baseUrl) {
  const parsed = new URL(baseUrl);

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw Object.assign(new Error('Only HTTP and HTTPS text backend URLs are supported.'), { statusCode: 400 });
  }

  if (parsed.username || parsed.password) {
    throw unsafeTextBackendUrlError();
  }

  const hostname = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');
  const ipVersion = net.isIP(hostname);
  if (!hostname || (ipVersion && !isAllowedBackendAddress(hostname))) {
    throw unsafeTextBackendUrlError();
  }
}

function unsafeTextBackendUrlError() {
  return Object.assign(
    new Error('For safety, the text backend URL must resolve only to loopback or private LAN addresses; link-local and metadata endpoints are blocked.'),
    { statusCode: 400 },
  );
}

async function resolveAllowedBackendTarget(url, signal) {
  const parsed = url instanceof URL ? url : new URL(url);
  assertAllowedBaseUrl(parsed.toString());
  const hostname = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');
  const literalFamily = net.isIP(hostname);
  let addresses;

  if (literalFamily) {
    addresses = [{ address: hostname, family: literalFamily }];
  } else {
    try {
      addresses = await raceWithAbort(
        dns.promises.lookup(hostname, { all: true, verbatim: true }),
        signal,
      );
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      throw unsafeTextBackendUrlError();
    }
  }

  if (!addresses.length || addresses.some(({ address, family }) => (
    net.isIP(address) !== family || !isAllowedBackendAddress(address.toLowerCase())
  ))) {
    throw unsafeTextBackendUrlError();
  }

  return {
    addresses: addresses.map(({ address, family }) => ({ address, family })),
    hostname,
  };
}

function isAllowedBackendAddress(address) {
  return isLoopbackHost(address) || isPrivateIpv4(address) || isPrivateIpv6(address);
}

function isLoopbackHost(hostname) {
  return hostname === '::1'
    || (net.isIP(hostname) === 4 && Number(hostname.split('.')[0]) === 127);
}

function isPrivateIpv4(hostname) {
  if (net.isIP(hostname) !== 4) return false;
  const parts = hostname.split('.').map((part) => Number.parseInt(part, 10));
  const [a, b] = parts;
  return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
}

function isPrivateIpv6(hostname) {
  if (net.isIP(hostname) !== 6) return false;
  let normalized;
  try {
    normalized = new URL(`http://[${hostname}]/`).hostname.slice(1, -1).toLowerCase();
  } catch {
    return false;
  }
  if (normalized === 'fd00:ec2::254') return false;
  const firstHextet = Number.parseInt(normalized.split(':')[0], 16);
  return firstHextet >= 0xfc00 && firstHextet <= 0xfdff;
}

function raceWithAbort(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(makeAbortError());
  return new Promise((resolve, reject) => {
    const abort = () => reject(makeAbortError());
    signal.addEventListener('abort', abort, { once: true });
    Promise.resolve(promise).then(
      (value) => {
        signal.removeEventListener('abort', abort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener('abort', abort);
        reject(error);
      },
    );
  });
}

function makeAbortError() {
  return Object.assign(new Error('The text backend request was stopped.'), { name: 'AbortError' });
}

function normalizeNumber(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function normalizeInteger(value, min, max, fallback) {
  return Math.round(normalizeNumber(value, min, max, fallback));
}

function makeUpstreamHeaders(apiKey) {
  const headers = {
    Accept: 'application/json, text/event-stream',
  };

  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  return headers;
}

async function prepareTextBackend(baseUrl, model = '') {
  if (textBackendReconfiguring) {
    throw Object.assign(new Error('The local text configuration is being updated. Try again shortly.'), {
      statusCode: 409,
    });
  }
  if (imageGpuReserved || (activeImageProcess && activeImageProcess !== persistentImageWorkerStoppingChild)) {
    throw Object.assign(new Error('Text generation will be available when the current image finishes.'), {
      statusCode: 409,
    });
  }
  const usesManagedBackend = managedTextBackend.matches(baseUrl);
  const requestKey = usesManagedBackend ? `managed:${model}` : `external:${baseUrl}`;
  if (activeTextRequests && activeTextBackendKey !== requestKey) {
    throw Object.assign(new Error('Another text model is currently generating. Wait for it to finish before switching.'), {
      statusCode: 409,
    });
  }
  if (!activeTextRequests) activeTextBackendKey = requestKey;
  activeTextRequests += 1;
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    activeTextRequests -= 1;
    if (!activeTextRequests) activeTextBackendKey = '';
  };
  try {
    await stopPersistentImageWorker();
    if (usesManagedBackend) {
      await managedTextBackend.ensureReady(model);
    }
    return release;
  } catch (error) {
    release();
    throw error;
  }
}

function resolveTextBaseUrl(value) {
  return normalizeBaseUrl(value);
}

function resolveConfiguredTextBaseUrl(environment, backend) {
  const configured = environment.TEXT_BASE_URL || backend.baseUrl || '';
  return configured ? normalizeBaseUrl(configured) : '';
}

async function fetchTextBackend(url, options, timeoutMs = PROXY_TIMEOUT_MS) {
  try {
    return await fetchWithTimeout(url, options, timeoutMs);
  } catch (error) {
    if (error.statusCode === 400) throw error;
    if (error.name === 'AbortError') {
      throw Object.assign(new Error('The text generation request timed out or was stopped.'), { statusCode: 504 });
    }

    const parsed = new URL(url);
    throw Object.assign(
      new Error(`Could not reach the text backend at ${parsed.origin}. Make sure the configured local server is running.`),
      { statusCode: 502 },
    );
  }
}

async function shutdownManagedProcesses() {
  const imageProcess = activeImageProcess;
  const persistentProcess = persistentImageWorker?.child;
  await Promise.allSettled([
    stopPersistentImageWorker(),
    imageProcess && imageProcess !== persistentProcess
      ? stopChildProcess(imageProcess)
      : Promise.resolve(),
    managedTextBackend.stop(),
  ]);
}

function shutdownManagedProcessesNow() {
  if (isProcessRunning(activeImageProcess)) activeImageProcess.kill();
  stopPersistentImageWorkerNow();
  managedTextBackend.stopNow();
}

async function stopChildProcess(child) {
  if (!isProcessRunning(child)) return;
  const exited = new Promise((resolve) => child.once('exit', resolve));
  child.kill();
  await Promise.race([exited, unrefDelay(2000)]);
  if (isProcessRunning(child)) {
    child.kill('SIGKILL');
    await Promise.race([exited, unrefDelay(1000)]);
  }
}

let shutdownStarted = false;
process.once('exit', shutdownManagedProcessesNow);
async function requestGracefulShutdown() {
  if (shutdownStarted) return;
  shutdownStarted = true;
  const serverClosed = new Promise((resolve) => server.close(resolve));
  await Promise.race([
    Promise.allSettled([shutdownManagedProcesses(), serverClosed]),
    new Promise((resolve) => setTimeout(resolve, 8000)),
  ]);
  shutdownManagedProcessesNow();
  process.exit(0);
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, requestGracefulShutdown);
}
process.on('message', (message) => {
  if (message?.type === 'shutdown') void requestGracefulShutdown();
});
process.once('disconnect', () => {
  if (process.connected === false) void requestGracefulShutdown();
});

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const outerSignal = options.signal;
  const state = { timedOut: false, timeout: null, outerSignal, outerAbort: null };
  state.timeout = setTimeout(() => {
    state.timedOut = true;
    controller.abort();
  }, timeoutMs);
  state.timeout.unref?.();

  if (outerSignal) {
    if (outerSignal.aborted) controller.abort();
    state.outerAbort = () => controller.abort();
    outerSignal.addEventListener('abort', state.outerAbort, { once: true });
  }

  try {
    const parsed = new URL(url);
    const target = await resolveAllowedBackendTarget(parsed, controller.signal);
    const response = await requestPinnedTextBackend(parsed, target, {
      ...options,
      signal: controller.signal,
    });
    Object.defineProperty(response, FETCH_TIMEOUT_STATE, { value: state });
    return response;
  } catch (error) {
    releaseFetchTimeoutState(state);
    throw error;
  }
}

function requestPinnedTextBackend(parsed, target, options) {
  const attempt = (addressIndex) => new Promise((resolve, reject) => {
    const transport = parsed.protocol === 'https:' ? https : http;
    const targetAddress = target.addresses[addressIndex];
    const headers = { ...options.headers };
    for (const name of Object.keys(headers)) {
      if (name.toLowerCase() === 'host') delete headers[name];
    }
    headers.Host = parsed.host;

    const requestOptions = {
      protocol: parsed.protocol,
      hostname: targetAddress.address,
      family: targetAddress.family,
      port: parsed.port || (parsed.protocol === 'https:' ? 443 : 80),
      method: options.method || 'GET',
      path: `${parsed.pathname}${parsed.search}`,
      headers,
      signal: options.signal,
      agent: false,
    };
    if (parsed.protocol === 'https:' && net.isIP(target.hostname) === 0) {
      requestOptions.servername = target.hostname;
    }

    let responseStarted = false;
    const request = transport.request(requestOptions, (message) => {
      responseStarted = true;
      const status = message.statusCode || 502;
      if (status >= 300 && status < 400) {
        message.resume();
        reject(new Error('Text backend redirects are not allowed.'));
        return;
      }
      resolve(wrapUpstreamResponse(message));
    });
    request.once('error', (error) => {
      const canRetry = !responseStarted
        && !options.signal?.aborted
        && addressIndex + 1 < target.addresses.length;
      if (canRetry) {
        resolve(attempt(addressIndex + 1));
      } else {
        reject(error);
      }
    });
    request.end(options.body);
  });

  return attempt(0);
}

function wrapUpstreamResponse(message) {
  const status = message.statusCode || 502;
  return {
    status,
    ok: status >= 200 && status < 300,
    body: message,
    headers: {
      get(name) {
        const value = message.headers[String(name).toLowerCase()];
        if (Array.isArray(value)) return value.join(', ');
        return value === undefined ? null : String(value);
      },
    },
    async text() {
      const chunks = [];
      for await (const chunk of message) chunks.push(Buffer.from(chunk));
      return Buffer.concat(chunks).toString('utf8');
    },
  };
}

async function readUpstreamText(response) {
  try {
    return await response.text();
  } catch (error) {
    if (didFetchTimeOut(response) || error.name === 'AbortError') {
      throw Object.assign(new Error('The text backend response timed out or was stopped.'), { statusCode: 504 });
    }
    throw Object.assign(new Error('The text backend response ended unexpectedly.'), { statusCode: 502 });
  } finally {
    releaseFetchTimeout(response);
  }
}

function didFetchTimeOut(response) {
  return Boolean(response?.[FETCH_TIMEOUT_STATE]?.timedOut);
}

function releaseFetchTimeout(response) {
  const state = response?.[FETCH_TIMEOUT_STATE];
  if (state) releaseFetchTimeoutState(state);
}

function releaseFetchTimeoutState(state) {
  if (!state.timeout) return;
  clearTimeout(state.timeout);
  state.timeout = null;
  if (state.outerSignal && state.outerAbort) {
    state.outerSignal.removeEventListener('abort', state.outerAbort);
  }
}

function abortWhenClientDisconnects(req, res) {
  const controller = new AbortController();
  if (req.aborted || res.destroyed) controller.abort();
  req.on('aborted', () => controller.abort());
  res.on('close', () => {
    if (!res.writableEnded) controller.abort();
  });
  return controller.signal;
}

function extractUpstreamError(errorText) {
  if (!errorText) return '';
  try {
    const parsed = JSON.parse(errorText);
    if (typeof parsed.error === 'string') return parsed.error;
    if (parsed.error && typeof parsed.error.message === 'string') return parsed.error.message;
    if (typeof parsed.message === 'string') return parsed.message;
  } catch {
    return errorText.slice(0, 500);
  }
  return errorText.slice(0, 500);
}

function sendJson(res, statusCode, body, extraHeaders = {}) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    ...securityHeaders(),
    ...extraHeaders,
  });
  res.end(JSON.stringify(body));
}

function securityHeaders() {
  return {
    'Content-Security-Policy':
      "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    'Referrer-Policy': 'no-referrer',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
  };
}

function createTlsOptions() {
  if (process.env.TLS_CERT_FILE || process.env.TLS_KEY_FILE) {
    if (!process.env.TLS_CERT_FILE || !process.env.TLS_KEY_FILE) {
      throw new Error('TLS_CERT_FILE and TLS_KEY_FILE must be set together.');
    }

    const cert = fs.readFileSync(path.resolve(process.env.TLS_CERT_FILE), 'utf8');
    const key = fs.readFileSync(path.resolve(process.env.TLS_KEY_FILE), 'utf8');
    const certificate = new crypto.X509Certificate(cert);
    return {
      cert,
      key,
      fingerprint256: certificate.fingerprint256,
    };
  }

  return createSelfSignedCertificate();
}

function createSelfSignedCertificate() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
  const subjectPublicKeyInfo = publicKey.export({ type: 'spki', format: 'der' });
  const privateKeyPem = privateKey.export({ type: 'pkcs8', format: 'pem' });
  const signatureAlgorithm = derSequence(derOid('1.2.840.113549.1.1.11'), derNull());
  const name = derName('Local LLM Serve');
  const notBefore = new Date(Date.now() - 60 * 60 * 1000);
  const notAfter = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);
  const extensions = derExplicit(
    3,
    derSequence(
      derExtension('2.5.29.19', true, derSequence()),
      derExtension('2.5.29.15', true, derBitString(Buffer.from([0xa0]), 5)),
      derExtension('2.5.29.37', false, derSequence(derOid('1.3.6.1.5.5.7.3.1'))),
      derExtension('2.5.29.17', false, derSubjectAltNames()),
    ),
  );

  const tbsCertificate = derSequence(
    derExplicit(0, derInteger(2)),
    derInteger(crypto.randomBytes(16)),
    signatureAlgorithm,
    name,
    derSequence(derUtcTime(notBefore), derUtcTime(notAfter)),
    name,
    subjectPublicKeyInfo,
    extensions,
  );

  const signature = crypto.createSign('RSA-SHA256').update(tbsCertificate).sign(privateKey);
  const certificate = derSequence(tbsCertificate, signatureAlgorithm, derBitString(signature, 0));
  const cert = toPem('CERTIFICATE', certificate);

  return {
    key: privateKeyPem,
    cert,
    fingerprint256: crypto.createHash('sha256').update(certificate).digest('hex').match(/.{2}/g).join(':').toUpperCase(),
  };
}

function derSubjectAltNames() {
  return derSequence(
    derContextPrimitive(2, Buffer.from('localhost', 'ascii')),
    derContextPrimitive(7, Buffer.from([127, 0, 0, 1])),
    derContextPrimitive(7, Buffer.from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),
  );
}

function derExtension(oid, critical, value) {
  const parts = [derOid(oid)];
  if (critical) {
    parts.push(derBoolean(true));
  }
  parts.push(derOctetString(value));
  return derSequence(...parts);
}

function derName(commonName) {
  return derSequence(derSet(derSequence(derOid('2.5.4.3'), derUtf8String(commonName))));
}

function derSequence(...parts) {
  return der(0x30, Buffer.concat(parts));
}

function derSet(...parts) {
  return der(0x31, Buffer.concat(parts));
}

function derExplicit(tagNumber, content) {
  return der(0xa0 + tagNumber, content);
}

function derContextPrimitive(tagNumber, content) {
  return der(0x80 + tagNumber, content);
}

function derInteger(value) {
  let bytes;
  if (Buffer.isBuffer(value)) {
    bytes = Buffer.from(value);
  } else {
    let hex = Number(value).toString(16);
    if (hex.length % 2) hex = `0${hex}`;
    bytes = Buffer.from(hex, 'hex');
  }

  while (bytes.length > 1 && bytes[0] === 0 && (bytes[1] & 0x80) === 0) {
    bytes = bytes.subarray(1);
  }
  if (bytes[0] & 0x80) {
    bytes = Buffer.concat([Buffer.from([0]), bytes]);
  }

  return der(0x02, bytes);
}

function derBoolean(value) {
  return der(0x01, Buffer.from([value ? 0xff : 0x00]));
}

function derBitString(bytes, unusedBits) {
  return der(0x03, Buffer.concat([Buffer.from([unusedBits]), bytes]));
}

function derOctetString(bytes) {
  return der(0x04, bytes);
}

function derNull() {
  return der(0x05, Buffer.alloc(0));
}

function derOid(value) {
  const parts = value.split('.').map((part) => Number.parseInt(part, 10));
  const bytes = [parts[0] * 40 + parts[1]];
  for (const part of parts.slice(2)) {
    const stack = [part & 0x7f];
    let remaining = part >> 7;
    while (remaining > 0) {
      stack.unshift((remaining & 0x7f) | 0x80);
      remaining >>= 7;
    }
    bytes.push(...stack);
  }
  return der(0x06, Buffer.from(bytes));
}

function derUtf8String(value) {
  return der(0x0c, Buffer.from(value, 'utf8'));
}

function derUtcTime(date) {
  const year = String(date.getUTCFullYear()).slice(-2);
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  const hour = String(date.getUTCHours()).padStart(2, '0');
  const minute = String(date.getUTCMinutes()).padStart(2, '0');
  const second = String(date.getUTCSeconds()).padStart(2, '0');
  return der(0x17, Buffer.from(`${year}${month}${day}${hour}${minute}${second}Z`, 'ascii'));
}

function der(tag, content) {
  return Buffer.concat([Buffer.from([tag]), derLength(content.length), content]);
}

function derLength(length) {
  if (length < 0x80) {
    return Buffer.from([length]);
  }

  let hex = length.toString(16);
  if (hex.length % 2) hex = `0${hex}`;
  const bytes = Buffer.from(hex, 'hex');
  return Buffer.concat([Buffer.from([0x80 | bytes.length]), bytes]);
}

function toPem(label, derBytes) {
  const base64 = derBytes.toString('base64').match(/.{1,64}/g).join('\n');
  return `-----BEGIN ${label}-----\n${base64}\n-----END ${label}-----\n`;
}

function getLanAddresses() {
  const addresses = [];
  const interfaces = os.networkInterfaces();

  for (const entries of Object.values(interfaces)) {
    for (const entry of entries || []) {
      if (entry.family === 'IPv4' && !entry.internal) {
        addresses.push(entry.address);
      }
    }
  }

  return addresses;
}
