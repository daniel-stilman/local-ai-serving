'use strict';

const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const { performance } = require('perf_hooks');
const { spawn } = require('child_process');

const GIB = 1024 ** 3;
const TEXT_START_TIMEOUT_BASE_MS = 3 * 60_000;
const TEXT_START_TIMEOUT_PER_GIB_MS = 15_000;
const TEXT_START_TIMEOUT_MAX_MS = 15 * 60_000;
const TEXT_START_TIMEOUT_EXPLICIT_MAX_MS = 30 * 60_000;
const TEXT_MODEL_SCAN_LIMITS = Object.freeze({
  maxDepth: 12,
  maxEntries: 10_000,
  maxMilliseconds: 1_000,
});

function createManagedTextBackend(options = {}) {
  const environment = options.env || process.env;
  const spawnProcess = options.spawn || spawn;
  const fetchImpl = options.fetch || globalThis.fetch;
  const wait = options.delay || delay;
  const now = options.now || Date.now;
  const portInUse = options.isPortInUse || isPortInUse;
  const explicitlyExternal = Boolean(environment.TEXT_BASE_URL)
    || String(environment.TEXT_BACKEND || '').toLowerCase() === 'external';
  const executable = explicitlyExternal ? '' : findLlamaServer(environment.TEXT_SERVER_EXE);
  const catalogEntries = explicitlyExternal
    ? Object.freeze([])
    : (options.catalogEntries === undefined
      ? makeTextModelCatalog(environment)
      : normalizeInternalTextModelCatalog(options.catalogEntries));
  const catalogById = new Map(catalogEntries.map((entry) => [entry.id, entry]));
  const defaultEntry = catalogEntries.find((entry) => entry.default) || catalogEntries[0] || null;
  const modelPath = defaultEntry?.path || '';
  const enabled = Boolean(executable && defaultEntry);
  const port = normalizeInteger(environment.TEXT_PORT, 1024, 65535, 1235);
  const baseUrl = enabled ? `http://127.0.0.1:${port}/v1` : '';
  const alias = enabled ? defaultEntry.id : '';
  const models = Object.freeze(catalogEntries.map((entry) => Object.freeze({ id: entry.id })));
  const modelIds = Object.freeze(models.map((model) => model.id));
  const logger = options.logger || console;

  let child = null;
  let ready = false;
  let startPromise = null;
  let startModelId = '';
  let activeModelId = '';
  let stderrTail = '';
  let diagnosticsCarry = '';
  let gpuOffload = null;
  let lifecycleGeneration = 0;
  let shutdownGeneration = 0;
  let loadStatus = enabled
    ? { state: 'idle', phase: 'idle' }
    : { state: 'disabled', phase: 'idle' };

  function matches(candidate) {
    return enabled && stripTrailingSlash(candidate) === baseUrl;
  }

  async function ensureReady(modelId = alias) {
    if (!enabled) return;
    const selected = typeof modelId === 'string' ? catalogById.get(modelId) : null;
    if (!selected) throw modelSelectionError();
    if (!modelFilesAvailable(selected)) throw modelUnavailableError();
    if (ready && activeModelId === selected.id && isChildRunning(child)) return;
    if (startPromise) {
      if (startModelId === selected.id) return startPromise;
      const pending = startPromise;
      const requestedShutdownGeneration = shutdownGeneration;
      return pending.catch((error) => {
        if (shutdownGeneration !== requestedShutdownGeneration) throw backendStoppedError();
        return error;
      }).then(() => {
        if (shutdownGeneration !== requestedShutdownGeneration) throw backendStoppedError();
        return ensureReady(selected.id);
      });
    }

    const requestedShutdownGeneration = shutdownGeneration;
    setLoadStatus('loading', child ? 'stopping' : 'starting');
    const operation = switchTo(selected, requestedShutdownGeneration);
    let trackedPromise;
    trackedPromise = operation
      .catch((error) => {
        if (shutdownGeneration === requestedShutdownGeneration && !ready) {
          setLoadStatus('error', 'error');
        }
        throw error;
      })
      .finally(() => {
        if (startPromise === trackedPromise) {
          startPromise = null;
          startModelId = '';
        }
      });
    startModelId = selected.id;
    startPromise = trackedPromise;
    return trackedPromise;
  }

  async function switchTo(selected, requestedShutdownGeneration) {
    if (child) {
      setLoadStatus('loading', 'stopping');
      await stopRunning();
    }
    if (shutdownGeneration !== requestedShutdownGeneration) throw backendStoppedError();
    setLoadStatus('loading', 'starting');
    return start(selected, requestedShutdownGeneration);
  }

  async function start(selected, requestedShutdownGeneration) {
    const startGeneration = ++lifecycleGeneration;
    stderrTail = '';
    diagnosticsCarry = '';
    gpuOffload = null;
    if (await portInUse(port)) {
      throw backendStartError(`The configured text port ${port} is already in use.`);
    }
    if (
      startGeneration !== lifecycleGeneration
      || shutdownGeneration !== requestedShutdownGeneration
    ) {
      throw backendStartError('The direct text engine was stopped before it could start.');
    }
    if (!modelFilesAvailable(selected)) throw modelUnavailableError();
    const args = makeServerArgs(selected.path, selected.id, port, environment);
    let launched;
    try {
      launched = spawnProcess(executable, args, {
        cwd: path.dirname(executable),
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: environment,
      });
    } catch (error) {
      stderrTail = error && error.message || '';
      throw backendStartError('The direct text engine could not start.');
    }
    child = launched;
    activeModelId = selected.id;
    ready = false;
    setLoadStatus('loading', 'loading');
    let spawnError = null;
    let exitState = null;
    launched.on('error', (error) => {
      spawnError = error;
    });
    const captureDiagnostics = (chunk) => {
      const text = chunk.toString('utf8');
      stderrTail = (stderrTail + text).slice(-8000);
      const diagnostics = diagnosticsCarry + text;
      for (const match of diagnostics.matchAll(/offloaded\s+(\d+)\s*\/\s*(\d+)\s+layers?\s+to\s+GPU/gi)) {
        gpuOffload = { offloaded: Number(match[1]), total: Number(match[2]) };
      }
      diagnosticsCarry = diagnostics.slice(-256);
    };
    launched.stdout?.on('data', captureDiagnostics);
    launched.stderr?.on('data', captureDiagnostics);
    launched.once('exit', (code, signal) => {
      exitState = { code, signal };
      if (child === launched) {
        child = null;
        activeModelId = '';
        ready = false;
        if (
          startGeneration === lifecycleGeneration
          && shutdownGeneration === requestedShutdownGeneration
        ) setLoadStatus('error', 'error');
      }
    });

    const deadline = now() + resolveTextStartTimeoutMs(environment, selected);
    while (now() < deadline) {
      if (spawnError) {
        if (child === launched) {
          child = null;
          activeModelId = '';
        }
        stderrTail = spawnError.message || stderrTail;
        throw backendStartError('The direct text engine could not start.');
      }
      if (
        startGeneration !== lifecycleGeneration
        || shutdownGeneration !== requestedShutdownGeneration
      ) {
        throw backendStartError('The direct text engine was stopped before it became ready.');
      }
      if (exitState || !isChildRunning(launched)) {
        const reason = exitState && exitState.code !== null
          ? `code ${exitState.code}`
          : `signal ${exitState?.signal || launched.signalCode || 'unknown'}`;
        throw backendStartError(`The direct text engine exited with ${reason}.`);
      }
      if (child !== launched) {
        throw backendStartError('The direct text engine was stopped before it became ready.');
      }
      try {
        if (await isExpectedServer(fetchImpl, port, selected.id)) {
          if (
            spawnError
            || startGeneration !== lifecycleGeneration
            || shutdownGeneration !== requestedShutdownGeneration
            || child !== launched
            || !isChildRunning(launched)
          ) continue;
          ready = true;
          setLoadStatus('ready', 'ready');
          const gpuStatus = gpuOffload?.offloaded > 0
            ? ` (GPU offload confirmed: ${gpuOffload.offloaded}/${gpuOffload.total} layers)`
            : ' (GPU offload not confirmed)';
          logger.log(`Direct text engine ready${gpuStatus}`);
          return;
        }
      } catch {
        // The CUDA model is still loading.
      }
      await wait(100);
    }

    await stopRunning();
    throw backendStartError('The direct text engine did not become ready in time.');
  }

  async function stop() {
    shutdownGeneration += 1;
    if (child) setLoadStatus('loading', 'stopping');
    try {
      return await stopRunning();
    } finally {
      if (!child) setLoadStatus(enabled ? 'idle' : 'disabled', 'idle');
    }
  }

  async function stopRunning() {
    lifecycleGeneration += 1;
    const running = child;
    ready = false;
    if (!running) {
      activeModelId = '';
      return;
    }
    const exited = !isChildRunning(running)
      ? Promise.resolve()
      : new Promise((resolve) => running.once('exit', resolve));
    if (isChildRunning(running)) running.kill();
    await Promise.race([exited, wait(5000)]);
    if (isChildRunning(running)) {
      running.kill('SIGKILL');
      await Promise.race([exited, wait(1000)]);
    }
    if (isChildRunning(running)) {
      throw backendStartError('The direct text engine could not be stopped safely.');
    }
    if (child === running) {
      child = null;
      activeModelId = '';
    }
  }

  function stopNow() {
    shutdownGeneration += 1;
    lifecycleGeneration += 1;
    ready = false;
    if (isChildRunning(child)) child.kill();
    child = null;
    activeModelId = '';
    setLoadStatus(enabled ? 'idle' : 'disabled', 'idle');
  }

  function setLoadStatus(state, phase) {
    loadStatus = { state, phase };
  }

  function getStatus() {
    return { state: loadStatus.state, phase: loadStatus.phase };
  }

  function backendStartError(message) {
    const detail = stderrTail
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .slice(-3)
      .join(' ');
    if (detail) {
      logger.error?.('Direct text engine diagnostics were redacted.');
    }
    return Object.assign(new Error(message), { statusCode: 503 });
  }

  function modelSelectionError() {
    return Object.assign(new Error('The selected local text model is unavailable.'), { statusCode: 400 });
  }

  function modelUnavailableError() {
    return Object.assign(new Error('The selected local text model is no longer available.'), { statusCode: 503 });
  }

  function backendStoppedError() {
    return Object.assign(new Error('The direct text engine was stopped before it could switch models.'), { statusCode: 503 });
  }

  return {
    enabled,
    baseUrl,
    alias,
    models,
    modelIds,
    modelCount: models.length,
    executable,
    modelPath,
    matches,
    getStatus,
    ensureReady,
    stop,
    stopNow,
  };
}

function resolveTextStartTimeoutMs(environment, selected) {
  const explicit = String(environment.TEXT_START_TIMEOUT_MS || '').trim();
  if (explicit) {
    return normalizeInteger(
      explicit,
      1000,
      TEXT_START_TIMEOUT_EXPLICIT_MAX_MS,
      TEXT_START_TIMEOUT_BASE_MS,
    );
  }

  let totalBytes = 0;
  for (const filePath of selected.files || []) {
    try {
      totalBytes += fs.statSync(filePath).size;
    } catch {
      return TEXT_START_TIMEOUT_BASE_MS;
    }
  }
  const sizeAllowance = Math.ceil(totalBytes / GIB) * TEXT_START_TIMEOUT_PER_GIB_MS;
  return Math.min(TEXT_START_TIMEOUT_MAX_MS, TEXT_START_TIMEOUT_BASE_MS + sizeAllowance);
}

async function isExpectedServer(fetchImpl, port, alias) {
  const health = await fetchImpl(`http://127.0.0.1:${port}/health`, {
    signal: AbortSignal.timeout(750),
  });
  if (!health.ok) return false;
  const models = await fetchImpl(`http://127.0.0.1:${port}/v1/models`, {
    signal: AbortSignal.timeout(750),
  });
  if (!models.ok) return false;
  const payload = await models.json();
  return Array.isArray(payload.data) && payload.data.some((model) => model && model.id === alias);
}

function makeServerArgs(modelPath, alias, port, environment = process.env) {
  const logicalCpus = os.cpus().length;
  const generationThreads = normalizeInteger(environment.TEXT_THREADS, 1, logicalCpus, Math.min(8, logicalCpus));
  const batchThreads = normalizeInteger(environment.TEXT_BATCH_THREADS, 1, logicalCpus, Math.min(16, logicalCpus));
  return [
    '--model', modelPath,
    '--alias', alias,
    '--host', '127.0.0.1',
    '--port', String(port),
    '--ctx-size', String(normalizeInteger(environment.TEXT_CONTEXT_SIZE, 4096, 131072, 8192)),
    '--gpu-layers', 'all',
    '--fit', 'on',
    '--fit-target', String(normalizeInteger(environment.TEXT_GPU_MARGIN_MIB, 256, 4096, 1536)),
    '--fit-ctx', '4096',
    '--flash-attn', 'on',
    '--cache-type-k', environment.TEXT_KV_DTYPE || 'q8_0',
    '--cache-type-v', environment.TEXT_KV_DTYPE || 'q8_0',
    '--parallel', '1',
    '--threads', String(generationThreads),
    '--threads-batch', String(batchThreads),
    '--batch-size', '2048',
    '--ubatch-size', '512',
    '--cache-prompt',
    '--sleep-idle-seconds', String(normalizeInteger(environment.TEXT_SLEEP_IDLE_SECONDS, 30, 3600, 300)),
    '--offline',
    '--no-webui',
    '--log-verbosity', String(normalizeInteger(environment.TEXT_LOG_VERBOSITY, 0, 5, 1)),
  ];
}

function findLlamaServer(configuredPath) {
  if (!configuredPath) return '';
  const resolved = path.resolve(configuredPath);
  return isFile(resolved) ? resolved : '';
}

function findTextModel(
  configuredPath,
  environment = process.env,
  modelsRoot = environment.TEXT_MODELS_ROOT || '',
) {
  if (configuredPath) {
    const resolved = path.resolve(configuredPath);
    return isFile(resolved) && path.extname(resolved).toLowerCase() === '.gguf' ? resolved : '';
  }

  if (!modelsRoot) return '';
  const candidates = discoverTextModelCandidates(path.resolve(modelsRoot));
  if (!candidates.length) return '';

  const maximum = normalizeNumber(environment.TEXT_MODEL_MAX_GIB, 1, 100, 10) * GIB;
  return selectTextModel(candidates, maximum);
}

function makeTextModelCatalog(environment = process.env) {
  const configuredPath = environment.TEXT_MODEL_PATH || '';
  if (configuredPath) {
    const resolved = findTextModel(configuredPath, environment);
    return normalizeInternalTextModelCatalog(resolved ? [makeExplicitTextModelEntry(environment, resolved)] : []);
  }

  const modelsRoot = environment.TEXT_MODELS_ROOT || '';
  if (!modelsRoot) return Object.freeze([]);
  const maximum = normalizeNumber(environment.TEXT_MODEL_MAX_GIB, 1, 100, 10) * GIB;
  const resolvedRoot = path.resolve(modelsRoot);
  const candidates = discoverTextModelCandidates(resolvedRoot);
  return buildTextModelCatalog(environment, resolvedRoot, candidates, maximum);
}

async function discoverManagedTextCatalog(environment = process.env, scanOptions = {}) {
  if (isExternalTextEnvironment(environment)) return Object.freeze([]);
  const configuredPath = environment.TEXT_MODEL_PATH || '';
  if (configuredPath) {
    const resolved = path.resolve(configuredPath);
    const valid = path.extname(resolved).toLowerCase() === '.gguf' && await isFileAsync(resolved);
    return normalizeInternalTextModelCatalog(valid ? [makeExplicitTextModelEntry(environment, resolved)] : []);
  }

  const modelsRoot = environment.TEXT_MODELS_ROOT || '';
  if (!modelsRoot) return Object.freeze([]);
  const maximum = normalizeNumber(environment.TEXT_MODEL_MAX_GIB, 1, 100, 10) * GIB;
  const resolvedRoot = path.resolve(modelsRoot);
  const candidates = await discoverTextModelCandidatesAsync(resolvedRoot, scanOptions);
  return buildTextModelCatalog(environment, resolvedRoot, candidates, maximum);
}

function isExternalTextEnvironment(environment) {
  return Boolean(environment.TEXT_BASE_URL)
    || String(environment.TEXT_BACKEND || '').toLowerCase() === 'external';
}

function makeExplicitTextModelEntry(environment, resolved) {
  const configuredAlias = String(environment.TEXT_MODEL_ALIAS || '');
  const id = configuredAlias || path.basename(resolved, path.extname(resolved));
  return { id, path: resolved, files: [resolved], default: true };
}

function buildTextModelCatalog(environment, resolvedRoot, candidates, maximum) {
  if (!candidates.length) return Object.freeze([]);

  const defaultPath = selectCatalogDefaultTextModel(candidates, maximum);
  const configuredAlias = String(environment.TEXT_MODEL_ALIAS || '');
  const names = candidates.map((candidate) => describeTextModelCandidate(resolvedRoot, candidate.path));
  const basenameCounts = new Map();
  for (const name of names) {
    const key = name.basename.toLowerCase();
    basenameCounts.set(key, (basenameCounts.get(key) || 0) + 1);
  }
  const indexes = candidates.map((_candidate, index) => index)
    .sort((left, right) => names[left].relative.localeCompare(
      names[right].relative,
      undefined,
      { numeric: true, sensitivity: 'base' },
    ));
  const usedIds = new Set(configuredAlias ? [configuredAlias.toLowerCase()] : []);
  const entries = [];
  for (const index of indexes) {
    const candidate = candidates[index];
    const name = names[index];
    const isDefault = candidate.path === defaultPath;
    const proposed = basenameCounts.get(name.basename.toLowerCase()) === 1 ? name.basename : name.relative;
    const id = isDefault && configuredAlias
      ? configuredAlias
      : allocateUniqueModelId(proposed, usedIds);
    entries.push({
      id,
      path: candidate.path,
      files: candidate.files || [candidate.path],
      default: isDefault,
    });
  }
  entries.sort((left, right) => left.id.localeCompare(
    right.id,
    undefined,
    { numeric: true, sensitivity: 'base' },
  ));
  return normalizeInternalTextModelCatalog(entries);
}

function normalizeInternalTextModelCatalog(value) {
  if (!Array.isArray(value) || value.length > TEXT_MODEL_SCAN_LIMITS.maxEntries) {
    throw invalidInternalTextModelCatalogError();
  }
  const ids = new Set();
  let defaultCount = 0;
  const entries = value.map((entry) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      throw invalidInternalTextModelCatalogError();
    }
    const id = typeof entry.id === 'string' ? entry.id : '';
    const modelPath = typeof entry.path === 'string' ? entry.path : '';
    const files = Array.isArray(entry.files) ? entry.files : [];
    const idKey = id.toLowerCase();
    if (
      !id
      || id.length > 512
      || ids.has(idKey)
      || typeof entry.default !== 'boolean'
      || !path.isAbsolute(modelPath)
      || path.extname(modelPath).toLowerCase() !== '.gguf'
      || files.length < 1
      || files.length > TEXT_MODEL_SCAN_LIMITS.maxEntries
      || !files.every((filePath) => (
        typeof filePath === 'string'
        && path.isAbsolute(filePath)
        && path.extname(filePath).toLowerCase() === '.gguf'
      ))
      || !files.some((filePath) => sameLocalPath(filePath, modelPath))
    ) {
      throw invalidInternalTextModelCatalogError();
    }
    ids.add(idKey);
    const isDefault = entry.default === true;
    if (isDefault) defaultCount += 1;
    const normalized = {};
    Object.defineProperties(normalized, {
      id: { value: id, enumerable: true },
      default: { value: isDefault, enumerable: true },
      path: { value: modelPath, enumerable: false },
      files: { value: Object.freeze([...files]), enumerable: false },
    });
    return Object.freeze(normalized);
  });
  if (entries.length && defaultCount !== 1) throw invalidInternalTextModelCatalogError();
  return Object.freeze(entries);
}

function invalidInternalTextModelCatalogError() {
  const error = Object.assign(new Error('The local text model catalog is invalid.'), {
    code: 'TEXT_MODEL_CATALOG_INVALID',
    statusCode: 503,
  });
  error.stack = `${error.name}: ${error.message}`;
  return error;
}

function sameLocalPath(left, right) {
  const normalizedLeft = path.resolve(left);
  const normalizedRight = path.resolve(right);
  return process.platform === 'win32'
    ? normalizedLeft.toLowerCase() === normalizedRight.toLowerCase()
    : normalizedLeft === normalizedRight;
}

function describeTextModelCandidate(modelsRoot, modelPath) {
  const relativePath = path.relative(modelsRoot, modelPath).split(path.sep).join('/');
  const relative = relativePath
    .replace(/\.gguf$/i, '')
    .replace(/-00001-of-\d{5}$/i, '');
  return { relative, basename: relative.split('/').at(-1) };
}

function allocateUniqueModelId(proposed, usedIds) {
  const base = String(proposed || 'local-text-model');
  let id = base;
  let suffix = 2;
  while (usedIds.has(id.toLowerCase())) {
    id = `${base} (${suffix})`;
    suffix += 1;
  }
  usedIds.add(id.toLowerCase());
  return id;
}

function modelFilesAvailable(model) {
  return Array.isArray(model.files)
    && model.files.length > 0
    && model.files.every((filePath) => isFile(filePath));
}

function isAutomaticTextModelCandidate(filePath) {
  const name = path.basename(filePath).toLowerCase();
  if (isExcludedTextModelName(name)) return false;
  const split = name.match(/-(\d{5})-of-\d{5}\.gguf$/);
  if (split && split[1] !== '00001') return false;
  return true;
}

function discoverTextModelCandidates(modelsRoot, scanOptions = {}) {
  const candidates = [];
  const splitGroups = new Map();
  walkFiles(modelsRoot, (filePath, stats) => {
    collectTextModelCandidate(candidates, splitGroups, filePath, stats);
  }, scanOptions);
  finishSplitTextModelCandidates(candidates, splitGroups);
  return candidates;
}

async function discoverTextModelCandidatesAsync(modelsRoot, scanOptions = {}) {
  const candidates = [];
  const splitGroups = new Map();
  await walkFilesAsync(modelsRoot, (filePath, stats) => {
    collectTextModelCandidate(candidates, splitGroups, filePath, stats);
  }, scanOptions);
  finishSplitTextModelCandidates(candidates, splitGroups);
  return candidates;
}

function collectTextModelCandidate(candidates, splitGroups, filePath, stats) {
  const name = path.basename(filePath).toLowerCase();
  if (isExcludedTextModelName(name)) return;
  const split = name.match(/^(.*)-(\d{5})-of-(\d{5})\.gguf$/);
  if (!split) {
    candidates.push({ path: filePath, size: stats.size, shards: 1, files: [filePath] });
    return;
  }

  const index = Number(split[2]);
  const total = Number(split[3]);
  if (!index || !total || index > total) return;
  const key = `${path.dirname(filePath).toLowerCase()}\0${split[1]}\0${total}`;
  let group = splitGroups.get(key);
  if (!group) {
    group = { total, shards: new Map() };
    splitGroups.set(key, group);
  }
  group.shards.set(index, { path: filePath, size: stats.size });
}

function finishSplitTextModelCandidates(candidates, splitGroups) {
  for (const group of splitGroups.values()) {
    if (group.shards.size !== group.total) continue;
    let size = 0;
    let complete = true;
    for (let index = 1; index <= group.total; index += 1) {
      const shard = group.shards.get(index);
      if (!shard) {
        complete = false;
        break;
      }
      size += shard.size;
    }
    if (!complete) continue;
    candidates.push({
      path: group.shards.get(1).path,
      size,
      shards: group.total,
      files: Array.from({ length: group.total }, (_value, offset) => group.shards.get(offset + 1).path),
    });
  }
}

function isExcludedTextModelName(name) {
  return !name.endsWith('.gguf')
    || name.includes('mmproj')
    || /(?:^|[-_.])(?:embed(?:ding|dings)?|rerank(?:er)?)(?:[-_.]|$)/i.test(name);
}

function selectTextModel(candidates, maximumBytes) {
  const fitting = candidates.filter((candidate) => candidate.size <= maximumBytes);
  if (!fitting.length) return '';
  fitting.sort((left, right) => left.size - right.size || left.path.localeCompare(right.path));
  return fitting.at(-1).path;
}

function selectCatalogDefaultTextModel(candidates, maximumBytes) {
  const fitting = selectTextModel(candidates, maximumBytes);
  if (fitting) return fitting;
  return [...candidates]
    .sort((left, right) => left.size - right.size || left.path.localeCompare(right.path))[0]?.path || '';
}

function walkFiles(directory, visitor, options = {}) {
  const state = makeTextModelScanState(options);
  walkDirectory(directory, visitor, state, 0);
}

function walkDirectory(directory, visitor, state, depth) {
  assertTextModelScanBudget(state, depth);
  let handle;
  try {
    handle = fs.opendirSync(directory);
  } catch {
    assertTextModelScanBudget(state, depth);
    return;
  }

  try {
    while (true) {
      assertTextModelScanBudget(state, depth);
      let entry;
      try {
        entry = handle.readSync();
      } catch {
        assertTextModelScanBudget(state, depth);
        return;
      }
      assertTextModelScanBudget(state, depth);
      if (!entry) return;
      state.visitedEntries += 1;
      assertTextModelScanBudget(state, depth);
      const entryPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        assertTextModelScanBudget(state, depth + 1);
        walkDirectory(entryPath, visitor, state, depth + 1);
        assertTextModelScanBudget(state, depth);
      } else if (entry.isFile()) {
        let stats;
        try {
          stats = fs.statSync(entryPath);
        } catch {
          assertTextModelScanBudget(state, depth);
          continue;
        }
        assertTextModelScanBudget(state, depth);
        visitor(entryPath, stats);
        assertTextModelScanBudget(state, depth);
      }
    }
  } finally {
    try { handle.closeSync(); } catch {}
    assertTextModelScanBudget(state, depth);
  }
}

async function walkFilesAsync(directory, visitor, options = {}) {
  const state = makeTextModelScanState(options);
  await walkDirectoryAsync(directory, visitor, state, 0);
}

async function walkDirectoryAsync(directory, visitor, state, depth) {
  assertTextModelScanBudget(state, depth);
  let handle;
  try {
    handle = await fs.promises.opendir(directory);
  } catch {
    assertTextModelScanBudget(state, depth);
    return;
  }

  try {
    while (true) {
      assertTextModelScanBudget(state, depth);
      let entry;
      try {
        entry = await handle.read();
      } catch {
        assertTextModelScanBudget(state, depth);
        return;
      }
      assertTextModelScanBudget(state, depth);
      if (!entry) return;
      state.visitedEntries += 1;
      assertTextModelScanBudget(state, depth);
      const entryPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        assertTextModelScanBudget(state, depth + 1);
        await walkDirectoryAsync(entryPath, visitor, state, depth + 1);
        assertTextModelScanBudget(state, depth);
      } else if (entry.isFile()) {
        let stats;
        try {
          stats = await fs.promises.stat(entryPath);
        } catch {
          assertTextModelScanBudget(state, depth);
          continue;
        }
        assertTextModelScanBudget(state, depth);
        await visitor(entryPath, stats);
        assertTextModelScanBudget(state, depth);
      }
    }
  } finally {
    try { await handle.close(); } catch {}
    assertTextModelScanBudget(state, depth);
  }
}

function makeTextModelScanState(options) {
  const now = typeof options.now === 'function' ? options.now : () => performance.now();
  return {
    maxDepth: normalizeInteger(options.maxDepth, 0, TEXT_MODEL_SCAN_LIMITS.maxDepth, TEXT_MODEL_SCAN_LIMITS.maxDepth),
    maxEntries: normalizeInteger(options.maxEntries, 1, TEXT_MODEL_SCAN_LIMITS.maxEntries, TEXT_MODEL_SCAN_LIMITS.maxEntries),
    maxMilliseconds: normalizeInteger(
      options.maxMilliseconds,
      1,
      TEXT_MODEL_SCAN_LIMITS.maxMilliseconds,
      TEXT_MODEL_SCAN_LIMITS.maxMilliseconds,
    ),
    now,
    startedAt: now(),
    visitedEntries: 0,
  };
}

function assertTextModelScanBudget(state, depth) {
  if (
    depth > state.maxDepth
    || state.visitedEntries > state.maxEntries
    || state.now() - state.startedAt > state.maxMilliseconds
  ) {
    const error = Object.assign(new Error('Local text model discovery exceeded its safety limits.'), {
      code: 'TEXT_MODEL_DISCOVERY_LIMIT',
      statusCode: 503,
    });
    error.stack = `${error.name}: ${error.message}`;
    throw error;
  }
}

function isFile(filePath) {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

async function isFileAsync(filePath) {
  try {
    return (await fs.promises.stat(filePath)).isFile();
  } catch {
    return false;
  }
}

function isChildRunning(processHandle) {
  return Boolean(processHandle)
    && processHandle.exitCode === null
    && processHandle.signalCode == null;
}

function stripTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '');
}

function normalizeNumber(value, minimum, maximum, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
}

function normalizeInteger(value, minimum, maximum, fallback) {
  return Math.round(normalizeNumber(value, minimum, maximum, fallback));
}

function normalizeBoolean(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  return !['0', 'false', 'off', 'no'].includes(String(value).trim().toLowerCase());
}

function delay(milliseconds) {
  return new Promise((resolve) => {
    const timeout = setTimeout(resolve, milliseconds);
    timeout.unref?.();
  });
}

function isPortInUse(port) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: '127.0.0.1', port });
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(value);
    };
    socket.once('connect', () => finish(true));
    socket.once('error', () => finish(false));
    socket.setTimeout(500, () => finish(false));
  });
}

module.exports = {
  TEXT_MODEL_SCAN_LIMITS,
  createManagedTextBackend,
  discoverManagedTextCatalog,
  findLlamaServer,
  findTextModel,
  makeServerArgs,
  isPortInUse,
  isAutomaticTextModelCandidate,
  discoverTextModelCandidates,
  selectTextModel,
};
