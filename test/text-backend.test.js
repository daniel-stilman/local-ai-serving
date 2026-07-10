'use strict';

const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { PassThrough } = require('node:stream');
const test = require('node:test');

const {
  TEXT_MODEL_SCAN_LIMITS,
  createManagedTextBackend,
  discoverManagedTextCatalog,
  discoverTextModelCandidates,
  findLlamaServer,
  findTextModel,
  isAutomaticTextModelCandidate,
  makeServerArgs,
  selectTextModel,
} = require('../text-backend');

test('managed text backend accepts explicit local runtime and model paths', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-text-backend-test-'));
  try {
    const executable = path.join(root, 'llama-server.exe');
    const model = path.join(root, 'model.gguf');
    fs.writeFileSync(executable, 'test');
    fs.writeFileSync(model, 'test');
    assert.equal(findLlamaServer(executable), executable);
    assert.equal(findTextModel(model), model);
    assert.equal(findTextModel(path.join(root, 'missing.gguf')), '');
    assert.equal(findLlamaServer(), '');
    assert.equal(findTextModel('', {}), '');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('managed llama.cpp arguments encode the measured single-GPU policy', () => {
  const args = makeServerArgs('model.gguf', 'model-alias', 1235, {});
  assert.deepEqual(args.slice(0, 6), ['--model', 'model.gguf', '--alias', 'model-alias', '--host', '127.0.0.1']);
  assert.ok(hasPair(args, '--port', '1235'));
  assert.ok(hasPair(args, '--ctx-size', '8192'));
  assert.ok(hasPair(args, '--gpu-layers', 'all'));
  assert.ok(hasPair(args, '--flash-attn', 'on'));
  assert.ok(hasPair(args, '--cache-type-k', 'q8_0'));
  assert.ok(hasPair(args, '--cache-type-v', 'q8_0'));
  assert.ok(hasPair(args, '--parallel', '1'));
  assert.ok(args.includes('--no-cache-prompt'));
  assert.ok(hasPair(args, '--cache-ram', '0'));
  assert.ok(args.includes('--no-cache-idle-slots'));
  assert.ok(args.includes('--log-disable'));
  assert.equal(args.includes('--cache-prompt'), false);
  assert.equal(args.includes('--log-verbosity'), false);
  assert.ok(args.includes('--offline'));
  assert.ok(args.includes('--no-webui'));
});

test('automatic model selection never violates the configured size cap', () => {
  const gib = 1024 ** 3;
  const candidates = [
    { path: 'small.gguf', size: 4 * gib },
    { path: 'best-fit.gguf', size: 9 * gib },
    { path: 'too-large.gguf', size: 20 * gib },
  ];

  assert.equal(selectTextModel(candidates, 10 * gib), 'best-fit.gguf');
  assert.equal(selectTextModel(candidates.slice(2), 10 * gib), '');
});

test('automatic discovery skips projectors, embedding models, rerankers, and non-first split shards', () => {
  for (const filename of [
    'mmproj-model-f16.gguf',
    'nomic-embed-text.gguf',
    'bge-reranker-v2.gguf',
    'chat-model-00002-of-00002.gguf',
  ]) {
    assert.equal(isAutomaticTextModelCandidate(filename), false, filename);
  }
  assert.equal(isAutomaticTextModelCandidate('chat-model-00001-of-00002.gguf'), true);
  assert.equal(isAutomaticTextModelCandidate('ordinary-chat-model.gguf'), true);
});

test('automatic model scanning chooses the first shard instead of a larger invalid candidate', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'text-model-discovery-test-'));
  try {
    const files = new Map([
      ['ordinary-chat.gguf', 10],
      ['nomic-embed-text.gguf', 80],
      ['large-chat-00001-of-00002.gguf', 30],
      ['large-chat-00002-of-00002.gguf', 40],
    ]);
    for (const [filename, size] of files) fs.writeFileSync(path.join(root, filename), Buffer.alloc(size));
    assert.equal(
      findTextModel('', { TEXT_MODEL_MAX_GIB: '10' }, root),
      path.join(root, 'large-chat-00001-of-00002.gguf'),
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('split model discovery enforces complete shards and aggregate size', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'split-text-model-discovery-test-'));
  try {
    const files = new Map([
      ['complete-00001-of-00002.gguf', 60],
      ['complete-00002-of-00002.gguf', 50],
      ['incomplete-00001-of-00002.gguf', 90],
      ['single.gguf', 25],
    ]);
    for (const [filename, size] of files) fs.writeFileSync(path.join(root, filename), Buffer.alloc(size));
    const candidates = discoverTextModelCandidates(root)
      .sort((left, right) => left.path.localeCompare(right.path));
    assert.deepEqual(candidates.map((candidate) => ({
      name: path.basename(candidate.path),
      size: candidate.size,
      shards: candidate.shards,
    })), [
      { name: 'complete-00001-of-00002.gguf', size: 110, shards: 2 },
      { name: 'single.gguf', size: 25, shards: 1 },
    ]);
    assert.equal(selectTextModel(candidates, 100), path.join(root, 'single.gguf'));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('automatic model scanning has conservative public traversal limits', () => {
  assert.deepEqual(TEXT_MODEL_SCAN_LIMITS, {
    maxDepth: 12,
    maxEntries: 10_000,
    maxMilliseconds: 1_000,
  });
  assert.equal(Object.isFrozen(TEXT_MODEL_SCAN_LIMITS), true);
});

test('automatic model scanning fails generically before excessive recursion', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'text-model-depth-limit-test-'));
  try {
    const nested = path.join(root, 'private-level-one', 'private-level-two');
    fs.mkdirSync(nested, { recursive: true });
    fs.writeFileSync(path.join(nested, 'private-model.gguf'), 'test');
    assertGenericScanLimitFailure(
      () => discoverTextModelCandidates(root, { maxDepth: 1 }),
      [root, 'private-level-one', 'private-model'],
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('automatic model scanning fails generically when the entry budget is exhausted', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'text-model-entry-limit-test-'));
  try {
    for (const name of ['private-one.gguf', 'private-two.gguf', 'private-three.gguf']) {
      fs.writeFileSync(path.join(root, name), 'test');
    }
    assertGenericScanLimitFailure(
      () => discoverTextModelCandidates(root, { maxEntries: 2 }),
      [root, 'private-one', 'private-two', 'private-three'],
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('automatic model scanning fails generically when its elapsed-time budget is exhausted', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'text-model-time-limit-test-'));
  let clock = 0;
  try {
    fs.writeFileSync(path.join(root, 'private-model.gguf'), 'test');
    assertGenericScanLimitFailure(
      () => discoverTextModelCandidates(root, {
        maxMilliseconds: 5,
        now: () => {
          clock += 2;
          return clock;
        },
      }),
      [root, 'private-model'],
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('an explicit model path bypasses recursive root discovery limits', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'text-model-explicit-bypass-test-'));
  try {
    const executable = path.join(root, 'llama-server.exe');
    const explicitModel = path.join(root, 'explicit.gguf');
    let nested = path.join(root, 'automatic-root');
    fs.writeFileSync(executable, 'test');
    fs.writeFileSync(explicitModel, 'test');
    for (let depth = 0; depth <= TEXT_MODEL_SCAN_LIMITS.maxDepth; depth += 1) {
      nested = path.join(nested, `level-${depth}`);
    }
    fs.mkdirSync(nested, { recursive: true });
    fs.writeFileSync(path.join(nested, 'unreachable.gguf'), 'test');

    const backend = createManagedTextBackend({
      env: {
        TEXT_SERVER_EXE: executable,
        TEXT_MODEL_PATH: explicitModel,
        TEXT_MODELS_ROOT: path.join(root, 'automatic-root'),
      },
    });
    assert.equal(backend.enabled, true);
    assert.equal(backend.modelPath, explicitModel);
    assert.equal(backend.modelCount, 1);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('asynchronous catalog discovery matches startup discovery and yields immutable handoff entries', async () => {
  const fixture = makeMultiModelFixture({ duplicateBasenames: true });
  try {
    const catalogPromise = discoverManagedTextCatalog(fixture.options.env);
    await new Promise((resolve) => setImmediate(resolve));
    const catalogEntries = await catalogPromise;
    const startupBackend = createManagedTextBackend(fixture.options);
    const precomputedBackend = createManagedTextBackend({ ...fixture.options, catalogEntries });

    assert.equal(Object.isFrozen(catalogEntries), true);
    assert.equal(catalogEntries.every((entry) => (
      Object.isFrozen(entry) && Object.isFrozen(entry.files)
    )), true);
    assert.equal(catalogEntries.every((entry) => (
      Object.keys(entry).join(',') === 'id,default'
    )), true);
    assert.equal(JSON.stringify(catalogEntries).includes(fixture.root), false);
    assert.deepEqual(precomputedBackend.modelIds, startupBackend.modelIds);
    assert.deepEqual(precomputedBackend.models, startupBackend.models);
    assert.equal(precomputedBackend.alias, startupBackend.alias);
    assert.equal(precomputedBackend.modelCount, startupBackend.modelCount);
    assert.equal(precomputedBackend.models.every((model) => (
      Object.isFrozen(model) && Object.keys(model).length === 1 && typeof model.id === 'string'
    )), true);
    assert.equal(JSON.stringify(precomputedBackend.models).includes(fixture.root), false);
    assert.throws(() => { catalogEntries[0].id = 'mutated'; }, TypeError);
  } finally {
    fixture.cleanup();
  }
});

test('a precomputed catalog bypasses synchronous root rescanning', async () => {
  const fixture = makeMultiModelFixture();
  try {
    const catalogEntries = await discoverManagedTextCatalog(fixture.options.env);
    let excessiveRoot = path.join(fixture.root, 'replacement-root');
    for (let depth = 0; depth <= TEXT_MODEL_SCAN_LIMITS.maxDepth; depth += 1) {
      excessiveRoot = path.join(excessiveRoot, `private-level-${depth}`);
    }
    fs.mkdirSync(excessiveRoot, { recursive: true });
    fs.writeFileSync(path.join(excessiveRoot, 'private-model.gguf'), 'test');

    const backend = createManagedTextBackend({
      ...fixture.options,
      env: { ...fixture.options.env, TEXT_MODELS_ROOT: path.join(fixture.root, 'replacement-root') },
      catalogEntries,
    });
    assert.deepEqual(backend.modelIds, ['complete-chat', 'ordinary-chat']);
    assert.equal(backend.modelCount, 2);
  } finally {
    fixture.cleanup();
  }
});

test('a malformed precomputed catalog fails generically without exposing its contents', () => {
  const privateMarker = 'private-catalog-marker.gguf';
  assert.throws(() => createManagedTextBackend({
    env: {},
    catalogEntries: [{
      id: 'private-model-id',
      path: privateMarker,
      files: [privateMarker],
      default: true,
    }],
  }), (error) => {
    assert.equal(error.code, 'TEXT_MODEL_CATALOG_INVALID');
    assert.equal(error.statusCode, 503);
    assert.equal(error.message, 'The local text model catalog is invalid.');
    assert.equal(error.stack, `Error: ${error.message}`);
    assert.equal(error.message.includes(privateMarker), false);
    assert.equal(error.message.includes('private-model-id'), false);
    return true;
  });
});

test('asynchronous discovery enforces depth, entry, and elapsed-time limits generically', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'async-text-model-limits-test-'));
  try {
    const nested = path.join(root, 'private-one', 'private-two');
    fs.mkdirSync(nested, { recursive: true });
    fs.writeFileSync(path.join(nested, 'private-nested.gguf'), 'test');
    fs.writeFileSync(path.join(root, 'private-one.gguf'), 'test');
    fs.writeFileSync(path.join(root, 'private-two.gguf'), 'test');
    fs.writeFileSync(path.join(root, 'private-three.gguf'), 'test');

    await assertGenericScanLimitRejection(
      discoverManagedTextCatalog({ TEXT_MODELS_ROOT: root }, { maxDepth: 1 }),
      [root, 'private-one', 'private-nested'],
    );
    await assertGenericScanLimitRejection(
      discoverManagedTextCatalog({ TEXT_MODELS_ROOT: root }, { maxEntries: 2 }),
      [root, 'private-one', 'private-two', 'private-three'],
    );
    let clock = 0;
    await assertGenericScanLimitRejection(
      discoverManagedTextCatalog({ TEXT_MODELS_ROOT: root }, {
        maxMilliseconds: 5,
        now: () => {
          clock += 2;
          return clock;
        },
      }),
      [root, 'private-one', 'private-two', 'private-three'],
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('asynchronous explicit-file discovery does not traverse the candidate root', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'async-explicit-text-model-test-'));
  try {
    const explicitModel = path.join(root, 'explicit.gguf');
    let excessiveRoot = path.join(root, 'automatic-root');
    fs.writeFileSync(explicitModel, 'test');
    for (let depth = 0; depth <= TEXT_MODEL_SCAN_LIMITS.maxDepth; depth += 1) {
      excessiveRoot = path.join(excessiveRoot, `private-level-${depth}`);
    }
    fs.mkdirSync(excessiveRoot, { recursive: true });
    fs.writeFileSync(path.join(excessiveRoot, 'private-model.gguf'), 'test');

    const catalogEntries = await discoverManagedTextCatalog({
      TEXT_MODEL_PATH: explicitModel,
      TEXT_MODEL_ALIAS: 'explicit-model',
      TEXT_MODELS_ROOT: path.join(root, 'automatic-root'),
    });
    assert.equal(catalogEntries.length, 1);
    assert.equal(catalogEntries[0].id, 'explicit-model');
    assert.equal(catalogEntries[0].path, explicitModel);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('sync and async root catalogs retain over-cap models while defaulting to the largest fitting model', async () => {
  const fixture = makeSizeCapCatalogFixture({ includeFitting: true });
  try {
    const startupBackend = createManagedTextBackend(fixture.options);
    const catalogEntries = await discoverManagedTextCatalog(fixture.options.env);
    const precomputedBackend = createManagedTextBackend({ ...fixture.options, catalogEntries });

    assert.deepEqual(startupBackend.modelIds, ['larger-model', 'small-model']);
    assert.deepEqual(precomputedBackend.modelIds, startupBackend.modelIds);
    assert.equal(startupBackend.alias, 'small-model');
    assert.equal(precomputedBackend.alias, 'small-model');
    assert.equal(startupBackend.modelCount, 2);
    assert.equal(findTextModel('', fixture.options.env, fixture.modelsRoot), fixture.smallModel);

    await precomputedBackend.ensureReady('larger-model');
    assert.equal(fixture.launches.length, 1);
    assert.equal(argumentValue(fixture.launches[0].args, '--model'), fixture.largerModel);
    await precomputedBackend.stop();
  } finally {
    fixture.cleanup();
  }
});

test('sync and async root catalogs choose the smallest model when every candidate exceeds the cap', async () => {
  const fixture = makeSizeCapCatalogFixture({ includeFitting: false });
  try {
    const startupBackend = createManagedTextBackend(fixture.options);
    const catalogEntries = await discoverManagedTextCatalog(fixture.options.env);
    const precomputedBackend = createManagedTextBackend({ ...fixture.options, catalogEntries });

    assert.deepEqual(startupBackend.modelIds, ['larger-model', 'smallest-model']);
    assert.deepEqual(precomputedBackend.modelIds, startupBackend.modelIds);
    assert.equal(startupBackend.alias, 'smallest-model');
    assert.equal(precomputedBackend.alias, 'smallest-model');
    assert.equal(startupBackend.modelCount, 2);
    assert.equal(findTextModel('', fixture.options.env, fixture.modelsRoot), '');
  } finally {
    fixture.cleanup();
  }
});

test('managed root discovery exposes every compatible model through readable path-free IDs', () => {
  const fixture = makeMultiModelFixture();
  try {
    const backend = createManagedTextBackend(fixture.options);

    assert.equal(backend.enabled, true);
    assert.deepEqual(backend.modelIds, ['complete-chat', 'ordinary-chat']);
    assert.deepEqual(backend.models, [{ id: 'complete-chat' }, { id: 'ordinary-chat' }]);
    assert.equal(backend.modelCount, 2);
    assert.equal(backend.alias, 'complete-chat');
    assert.equal(backend.modelPath, fixture.modelPaths[1]);
    assert.deepEqual(backend.models.map((model) => Object.keys(model)), [['id'], ['id']]);
    assert.equal(JSON.stringify(backend.models).includes(fixture.root), false);
    assert.doesNotMatch(JSON.stringify(backend.models), /\.gguf|\\/i);
  } finally {
    fixture.cleanup();
  }
});

test('duplicate model basenames receive deterministic relative-path IDs', () => {
  const fixture = makeMultiModelFixture({ duplicateBasenames: true });
  try {
    const backend = createManagedTextBackend(fixture.options);
    assert.deepEqual(backend.modelIds, [
      'complete-chat',
      'first/shared-chat',
      'ordinary-chat',
      'second/shared-chat',
    ]);
    assert.equal(new Set(backend.modelIds.map((id) => id.toLowerCase())).size, backend.modelCount);
  } finally {
    fixture.cleanup();
  }
});

test('an explicit model path remains a single-model override and honors its alias', () => {
  const fixture = makeMultiModelFixture();
  try {
    const explicitPath = fixture.modelPaths[0];
    const backend = createManagedTextBackend({
      ...fixture.options,
      env: {
        ...fixture.options.env,
        TEXT_MODEL_PATH: explicitPath,
        TEXT_MODEL_ALIAS: 'configured-local-model',
      },
    });

    assert.equal(backend.alias, 'configured-local-model');
    assert.equal(backend.modelPath, explicitPath);
    assert.deepEqual(backend.modelIds, ['configured-local-model']);
    assert.deepEqual(backend.models, [{ id: 'configured-local-model' }]);
    assert.equal(backend.modelCount, 1);
  } finally {
    fixture.cleanup();
  }
});

test('selecting a different managed model safely replaces the warm child', async () => {
  const fixture = makeMultiModelFixture();
  try {
    const backend = createManagedTextBackend(fixture.options);
    const [firstId, secondId] = backend.modelIds;

    await backend.ensureReady(firstId);
    await Promise.all([backend.ensureReady(secondId), backend.ensureReady(secondId)]);
    await backend.ensureReady(secondId);

    assert.equal(fixture.launches.length, 2);
    assert.equal(fixture.launches[0].child.killed, true);
    assert.equal(argumentValue(fixture.launches[0].args, '--alias'), firstId);
    assert.equal(argumentValue(fixture.launches[0].args, '--model'), fixture.modelPaths[1]);
    assert.equal(argumentValue(fixture.launches[1].args, '--alias'), secondId);
    assert.equal(argumentValue(fixture.launches[1].args, '--model'), fixture.modelPaths[0]);
    await backend.stop();
  } finally {
    fixture.cleanup();
  }
});

test('production startup budget and path-free status cover a model that needs more than thirty seconds', async () => {
  await withManagedBackend(async ({ options }) => {
    let clock = 0;
    delete options.env.TEXT_START_TIMEOUT_MS;
    options.now = () => clock;
    options.delay = async () => {
      clock += 5_000;
    };
    options.fetch = async (url) => {
      if (clock < 35_000) throw new Error('synthetic engine is still loading');
      if (url.endsWith('/health')) return { ok: true };
      return {
        ok: true,
        json: async () => ({ data: [{ id: options.env.TEXT_MODEL_ALIAS }] }),
      };
    };
    const backend = createManagedTextBackend(options);
    const starting = backend.ensureReady();
    await Promise.resolve();

    const loadingStatus = backend.getStatus();
    assert.equal(loadingStatus.state, 'loading');
    assert.ok(['starting', 'loading'].includes(loadingStatus.phase));
    await starting;
    assert.ok(clock >= 35_000);
    assert.deepEqual(backend.getStatus(), { state: 'ready', phase: 'ready' });
    assert.equal(JSON.stringify(backend.getStatus()).includes(options.env.TEXT_MODEL_PATH), false);
    assert.equal(JSON.stringify(backend.getStatus()).includes(options.env.TEXT_MODEL_ALIAS), false);

    await backend.stop();
    assert.deepEqual(backend.getStatus(), { state: 'idle', phase: 'idle' });
  });
});

test('concurrent requests for different managed models serialize their handoff', async () => {
  const fixture = makeMultiModelFixture();
  try {
    const backend = createManagedTextBackend(fixture.options);
    const [firstId, secondId] = backend.modelIds;

    await Promise.all([backend.ensureReady(firstId), backend.ensureReady(secondId)]);

    assert.equal(fixture.launches.length, 2);
    assert.equal(fixture.launches[0].child.killed, true);
    assert.equal(argumentValue(fixture.launches[1].args, '--alias'), secondId);
    await backend.stop();
  } finally {
    fixture.cleanup();
  }
});

test('stopping a pending switch cancels queued model requests without resurrection', async () => {
  const fixture = makeMultiModelFixture();
  const health = deferred();
  try {
    fixture.options.fetch = async (url) => {
      if (url.endsWith('/health')) return health.promise;
      const latest = fixture.launches.at(-1);
      return {
        ok: true,
        json: async () => ({ data: [{ id: argumentValue(latest?.args || [], '--alias') }] }),
      };
    };
    const backend = createManagedTextBackend(fixture.options);
    const first = backend.ensureReady(backend.modelIds[0]);
    const queued = backend.ensureReady(backend.modelIds[1]);
    await Promise.resolve();
    await Promise.resolve();

    await backend.stop();
    health.resolve({ ok: true });

    await assert.rejects(first, (error) => error.statusCode === 503 && /stopped before/.test(error.message));
    await assert.rejects(queued, (error) => error.statusCode === 503 && /stopped before/.test(error.message));
    assert.equal(fixture.launches.length, 1);
  } finally {
    health.resolve({ ok: true });
    fixture.cleanup();
  }
});

test('an unknown managed model never stops or replaces the ready child', async () => {
  const fixture = makeMultiModelFixture();
  try {
    const backend = createManagedTextBackend(fixture.options);
    await backend.ensureReady(backend.modelIds[0]);

    await assert.rejects(
      backend.ensureReady('unknown-local-model'),
      (error) => error.statusCode === 400 && !error.message.includes('unknown-local-model'),
    );
    assert.equal(fixture.launches.length, 1);
    assert.equal(fixture.launches[0].child.killed, false);
    await backend.stop();
  } finally {
    fixture.cleanup();
  }
});

test('a disappeared split shard fails generically before launching a child', async () => {
  const fixture = makeMultiModelFixture();
  try {
    const backend = createManagedTextBackend(fixture.options);
    fs.rmSync(fixture.requiredModelFiles.split[1]);

    await assert.rejects(
      backend.ensureReady(backend.alias),
      (error) => error.statusCode === 503
        && /no longer available/.test(error.message)
        && !error.message.includes(backend.alias)
        && !error.message.includes(fixture.root),
    );
    assert.equal(fixture.launches.length, 0);
    assert.deepEqual(fixture.logs, []);
  } finally {
    fixture.cleanup();
  }
});

test('a model disappearing after asynchronous discovery is rejected by the consuming backend', async () => {
  const fixture = makeMultiModelFixture();
  try {
    const catalogEntries = await discoverManagedTextCatalog(fixture.options.env);
    const backend = createManagedTextBackend({ ...fixture.options, catalogEntries });
    fs.rmSync(fixture.requiredModelFiles.split[1]);

    await assert.rejects(
      backend.ensureReady(backend.alias),
      (error) => error.statusCode === 503
        && /no longer available/.test(error.message)
        && !error.message.includes(backend.alias)
        && !error.message.includes(fixture.root),
    );
    assert.equal(fixture.launches.length, 0);
    assert.deepEqual(fixture.logs, []);
  } finally {
    fixture.cleanup();
  }
});

test('a disappeared switch target never stops the currently warm child', async () => {
  const fixture = makeMultiModelFixture();
  try {
    const backend = createManagedTextBackend(fixture.options);
    const ordinaryId = backend.modelIds.find((id) => id === 'ordinary-chat');
    const splitId = backend.modelIds.find((id) => id === 'complete-chat');
    await backend.ensureReady(ordinaryId);
    fs.rmSync(fixture.requiredModelFiles.split[1]);

    await assert.rejects(
      backend.ensureReady(splitId),
      (error) => error.statusCode === 503 && /no longer available/.test(error.message),
    );
    assert.equal(fixture.launches.length, 1);
    assert.equal(fixture.launches[0].child.killed, false);
    await backend.stop();
  } finally {
    fixture.cleanup();
  }
});

test('model disappearance during port preflight is rechecked before process launch', async () => {
  await withManagedBackend(async ({ options, children }) => {
    const portCheck = deferred();
    options.isPortInUse = () => portCheck.promise;
    const backend = createManagedTextBackend(options);
    const starting = backend.ensureReady();
    await Promise.resolve();

    fs.rmSync(options.env.TEXT_MODEL_PATH);
    portCheck.resolve(false);

    await assert.rejects(
      starting,
      (error) => error.statusCode === 503
        && /no longer available/.test(error.message)
        && !error.message.includes(options.env.TEXT_MODEL_PATH)
        && !error.message.includes(options.env.TEXT_MODEL_ALIAS),
    );
    assert.equal(children.length, 0);
  });
});

test('a managed model switch fails closed if the old child cannot stop', async () => {
  const fixture = makeMultiModelFixture({ stubbornFirstChild: true });
  try {
    const backend = createManagedTextBackend(fixture.options);
    await backend.ensureReady(backend.modelIds[0]);

    await assert.rejects(
      backend.ensureReady(backend.modelIds[1]),
      (error) => error.statusCode === 503 && /stopped safely/.test(error.message),
    );
    assert.equal(fixture.launches.length, 1);
  } finally {
    fixture.cleanup();
  }
});

test('concurrent readiness checks launch only one managed text process', async () => {
  await withManagedBackend(async ({ options, children }) => {
    const backend = createManagedTextBackend(options);

    await Promise.all([backend.ensureReady(), backend.ensureReady(), backend.ensureReady()]);

    assert.equal(children.length, 1);
    await backend.stop();
  });
});

test('managed readiness reports a generic logging-disabled state', async () => {
  await withManagedBackend(async ({ options, children }) => {
    const logs = [];
    options.logger = { log(message) { logs.push(message); } };
    options.spawn = () => {
      const child = new FakeChild();
      children.push(child);
      return child;
    };
    const backend = createManagedTextBackend(options);
    await backend.ensureReady();
    assert.equal(logs.length, 1);
    assert.equal(logs[0], 'Direct text engine ready (request logging disabled).');
    assert.doesNotMatch(logs[0], /expected-model/);
    assert.equal(logs.some((message) => message.includes(options.env.TEXT_MODEL_PATH)), false);
    await backend.stop();
  });
});

test('a stopped managed text backend can be started again cleanly', async () => {
  await withManagedBackend(async ({ options, children }) => {
    const backend = createManagedTextBackend(options);

    await backend.ensureReady();
    await backend.stop();
    await backend.ensureReady();

    assert.equal(children.length, 2);
    await backend.stop();
  });
});

test('managed text handoff fails closed when a child cannot be killed', async () => {
  await withManagedBackend(async ({ options, children }) => {
    options.delay = async () => {};
    options.spawn = () => {
      const child = new StubbornChild();
      children.push(child);
      return child;
    };
    const backend = createManagedTextBackend(options);
    await backend.ensureReady();

    await assert.rejects(
      backend.stop(),
      (error) => error.statusCode === 503 && /stopped safely/.test(error.message),
    );
  });
});

test('an asynchronous spawn error becomes a 503 instead of an uncaught process error', async () => {
  await withManagedBackend(async ({ options }) => {
    options.spawn = () => {
      const child = new FakeChild();
      queueMicrotask(() => child.emit('error', new Error('synthetic spawn failure')));
      return child;
    };
    options.fetch = async () => {
      throw new Error('not listening');
    };
    const backend = createManagedTextBackend(options);

    await assert.rejects(
      backend.ensureReady(),
      (error) => error.statusCode === 503
        && /could not start/.test(error.message)
        && !/synthetic spawn failure/.test(error.message),
    );
  });
});

test('managed startup output is discarded even when verbose flags are requested', async () => {
  await withManagedBackend(async ({ options }) => {
    const sentinel = 'SYNTHETIC_SENSITIVE_LOCAL_DIAGNOSTIC';
    const diagnostics = [];
    options.env.TEXT_LOG_PRIVATE_DIAGNOSTICS = '1';
    options.env.TEXT_LOG_VERBOSITY = '5';
    options.logger = {
      log() {},
      error(message) {
        diagnostics.push(message);
      },
    };
    options.spawn = () => {
      const child = new FakeChild();
      queueMicrotask(() => {
        child.stderr.write(
          `${sentinel}: failed to load ${options.env.TEXT_MODEL_PATH} as ${options.env.TEXT_MODEL_ALIAS}\n`,
        );
        child.exitWithCode(1);
      });
      return child;
    };
    options.fetch = async () => {
      throw new Error('not listening');
    };
    const backend = createManagedTextBackend(options);

    await assert.rejects(
      backend.ensureReady(),
      (error) => error.statusCode === 503 && !error.message.includes(sentinel),
    );
    assert.equal(diagnostics.length, 0);
  });
});

test('a signal-only child exit fails readiness immediately instead of polling to timeout', async () => {
  await withManagedBackend(async ({ options }) => {
    let clock = 0;
    let child;
    options.now = () => clock;
    options.delay = async (milliseconds) => {
      clock += milliseconds;
      child.exitWithSignal('SIGTERM');
    };
    options.spawn = () => {
      child = new FakeChild();
      return child;
    };
    options.fetch = async () => {
      throw new Error('not listening');
    };
    const backend = createManagedTextBackend(options);

    await assert.rejects(
      backend.ensureReady(),
      (error) => error.statusCode === 503 && /SIGTERM/.test(error.message),
    );
    assert.equal(clock, 100);
  });
});

test('stopping during a readiness probe cannot resurrect a killed backend', async () => {
  await withManagedBackend(async ({ options }) => {
    const health = deferred();
    options.fetch = async (url) => {
      if (url.endsWith('/health')) return health.promise;
      return {
        ok: true,
        json: async () => ({ data: [{ id: options.env.TEXT_MODEL_ALIAS }] }),
      };
    };
    const backend = createManagedTextBackend(options);
    const starting = backend.ensureReady();
    await Promise.resolve();

    await backend.stop();
    health.resolve({ ok: true });

    await assert.rejects(
      starting,
      (error) => error.statusCode === 503 && /stopped before/.test(error.message),
    );
  });
});

test('stopping during the port preflight prevents a late process launch', async () => {
  await withManagedBackend(async ({ options, children }) => {
    const portCheck = deferred();
    options.isPortInUse = () => portCheck.promise;
    const backend = createManagedTextBackend(options);
    const starting = backend.ensureReady();
    await Promise.resolve();

    await backend.stop();
    portCheck.resolve(false);

    await assert.rejects(
      starting,
      (error) => error.statusCode === 503 && /stopped before/.test(error.message),
    );
    assert.equal(children.length, 0);
  });
});

test('a different service on the configured port is never accepted as the text backend', async () => {
  await withManagedBackend(async ({ options, children }) => {
    let clock = 0;
    options.now = () => clock;
    options.delay = async (milliseconds) => {
      clock += milliseconds;
    };
    options.fetch = async (url) => {
      if (url.endsWith('/health')) return { ok: true };
      return {
        ok: true,
        json: async () => ({ data: [{ id: 'some-other-model' }] }),
      };
    };
    const backend = createManagedTextBackend(options);

    await assert.rejects(
      backend.ensureReady(),
      (error) => error.statusCode === 503 && /did not become ready/.test(error.message),
    );
    assert.equal(children.length, 1);
    assert.equal(children[0].killed, true);
  });
});

test('an occupied managed port is rejected before launching a second server', async () => {
  await withManagedBackend(async ({ options, children }) => {
    options.isPortInUse = async () => true;
    const backend = createManagedTextBackend(options);

    await assert.rejects(
      backend.ensureReady(),
      (error) => error.statusCode === 503 && /already in use/.test(error.message),
    );
    assert.equal(children.length, 0);
  });
});

test('explicit external text mode never launches the bundled backend', () => {
  let launches = 0;
  const backend = createManagedTextBackend({
    env: { TEXT_BACKEND: 'EXTERNAL' },
    spawn: () => {
      launches += 1;
      return new FakeChild();
    },
  });

  assert.equal(backend.enabled, false);
  assert.deepEqual(backend.getStatus(), { state: 'disabled', phase: 'idle' });
  assert.equal(backend.matches('http://127.0.0.1:1235/v1'), false);
  assert.equal(launches, 0);
});

async function withManagedBackend(callback) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-text-lifecycle-test-'));
  const executable = path.join(root, 'llama-server.exe');
  const model = path.join(root, 'model.gguf');
  fs.writeFileSync(executable, 'test');
  fs.writeFileSync(model, 'test');
  const children = [];
  const alias = 'expected-model';
  const options = {
    env: {
      TEXT_SERVER_EXE: executable,
      TEXT_MODEL_PATH: model,
      TEXT_MODEL_ALIAS: alias,
      TEXT_START_TIMEOUT_MS: '1000',
    },
    logger: { log() {} },
    isPortInUse: async () => false,
    spawn: () => {
      const child = new FakeChild();
      children.push(child);
      return child;
    },
    fetch: async (url) => {
      if (url.endsWith('/health')) return { ok: true };
      return {
        ok: true,
        json: async () => ({ data: [{ id: alias }] }),
      };
    },
  };

  try {
    await callback({ options, children });
  } finally {
    for (const child of children) child.kill();
    fs.rmSync(root, { recursive: true, force: true });
  }
}

function makeMultiModelFixture({ duplicateBasenames = false, stubbornFirstChild = false } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-text-catalog-test-'));
  const executable = path.join(root, 'llama-server.exe');
  const modelsRoot = path.join(root, 'models');
  const nested = path.join(modelsRoot, 'nested');
  fs.mkdirSync(nested, { recursive: true });
  fs.writeFileSync(executable, 'test');
  const ordinary = path.join(modelsRoot, 'ordinary-chat.gguf');
  const splitFirst = path.join(nested, 'complete-chat-00001-of-00002.gguf');
  const splitSecond = path.join(nested, 'complete-chat-00002-of-00002.gguf');
  fs.writeFileSync(ordinary, Buffer.alloc(10));
  fs.writeFileSync(splitFirst, Buffer.alloc(30));
  fs.writeFileSync(splitSecond, Buffer.alloc(40));
  fs.writeFileSync(path.join(modelsRoot, 'incomplete-chat-00001-of-00002.gguf'), Buffer.alloc(90));
  fs.writeFileSync(path.join(modelsRoot, 'mmproj-private.gguf'), Buffer.alloc(100));
  if (duplicateBasenames) {
    const first = path.join(modelsRoot, 'first');
    const second = path.join(modelsRoot, 'second');
    fs.mkdirSync(first, { recursive: true });
    fs.mkdirSync(second, { recursive: true });
    fs.writeFileSync(path.join(first, 'shared-chat.gguf'), Buffer.alloc(7));
    fs.writeFileSync(path.join(second, 'shared-chat.gguf'), Buffer.alloc(8));
  }

  const launches = [];
  const logs = [];
  let servedAlias = '';
  const options = {
    env: {
      TEXT_SERVER_EXE: executable,
      TEXT_MODELS_ROOT: modelsRoot,
      TEXT_MODEL_MAX_GIB: '1',
      TEXT_START_TIMEOUT_MS: '1000',
    },
    logger: {
      log(message) { logs.push(message); },
      error(message) { logs.push(message); },
    },
    delay: stubbornFirstChild ? async () => {} : undefined,
    isPortInUse: async () => false,
    spawn(_command, args) {
      const child = stubbornFirstChild && launches.length === 0 ? new StubbornChild() : new FakeChild();
      servedAlias = argumentValue(args, '--alias');
      launches.push({ args, child });
      return child;
    },
    fetch: async (url) => {
      if (url.endsWith('/health')) return { ok: true };
      return {
        ok: true,
        json: async () => ({ data: [{ id: servedAlias }] }),
      };
    },
  };
  if (!options.delay) delete options.delay;

  return {
    root,
    options,
    launches,
    logs,
    modelPaths: [ordinary, splitFirst],
    requiredModelFiles: { ordinary: [ordinary], split: [splitFirst, splitSecond] },
    cleanup() {
      for (const launch of launches) launch.child.kill();
      fs.rmSync(root, { recursive: true, force: true });
    },
  };
}

function makeSizeCapCatalogFixture({ includeFitting }) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-text-size-cap-test-'));
  const executable = path.join(root, 'llama-server.exe');
  const modelsRoot = path.join(root, 'models');
  const smallModel = path.join(modelsRoot, includeFitting ? 'small-model.gguf' : 'smallest-model.gguf');
  const largerModel = path.join(modelsRoot, 'larger-model.gguf');
  fs.mkdirSync(modelsRoot, { recursive: true });
  fs.writeFileSync(executable, 'test');
  if (includeFitting) fs.writeFileSync(smallModel, 'test');
  else makeSparseFile(smallModel, 1024 ** 3 + 1);
  makeSparseFile(largerModel, 1024 ** 3 + 2);

  const launches = [];
  let servedAlias = '';
  const options = {
    env: {
      TEXT_SERVER_EXE: executable,
      TEXT_MODELS_ROOT: modelsRoot,
      TEXT_MODEL_MAX_GIB: '1',
      TEXT_START_TIMEOUT_MS: '1000',
    },
    logger: { log() {}, error() {} },
    isPortInUse: async () => false,
    spawn(_command, args) {
      const child = new FakeChild();
      servedAlias = argumentValue(args, '--alias');
      launches.push({ args, child });
      return child;
    },
    fetch: async (url) => {
      if (url.endsWith('/health')) return { ok: true };
      return { ok: true, json: async () => ({ data: [{ id: servedAlias }] }) };
    },
  };
  return {
    root,
    modelsRoot,
    smallModel,
    largerModel,
    launches,
    options,
    cleanup() {
      for (const launch of launches) launch.child.kill();
      fs.rmSync(root, { recursive: true, force: true });
    },
  };
}

function makeSparseFile(filePath, size) {
  const handle = fs.openSync(filePath, 'w');
  try {
    fs.ftruncateSync(handle, size);
  } finally {
    fs.closeSync(handle);
  }
}

class FakeChild extends EventEmitter {
  constructor() {
    super();
    this.stdout = new PassThrough();
    this.stderr = new PassThrough();
    this.exitCode = null;
    this.signalCode = null;
    this.killed = false;
  }

  kill(signal = 'SIGTERM') {
    if (this.exitCode !== null) return false;
    this.killed = true;
    this.exitCode = 0;
    queueMicrotask(() => this.emit('exit', 0, signal));
    return true;
  }

  exitWithCode(code) {
    if (this.exitCode !== null || this.signalCode !== null) return;
    this.exitCode = code;
    this.emit('exit', code, null);
  }

  exitWithSignal(signal) {
    if (this.exitCode !== null || this.signalCode !== null) return;
    this.signalCode = signal;
    this.emit('exit', null, signal);
  }
}

class StubbornChild extends FakeChild {
  kill() {
    this.killed = true;
    return true;
  }
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function hasPair(values, key, expected) {
  const index = values.indexOf(key);
  return index >= 0 && values[index + 1] === expected;
}

function argumentValue(values, key) {
  const index = values.indexOf(key);
  return index >= 0 ? values[index + 1] : '';
}

function assertGenericScanLimitFailure(callback, forbiddenValues) {
  assert.throws(callback, (error) => {
    assert.equal(error.code, 'TEXT_MODEL_DISCOVERY_LIMIT');
    assert.equal(error.statusCode, 503);
    assert.equal(error.message, 'Local text model discovery exceeded its safety limits.');
    assert.equal(error.stack, `Error: ${error.message}`);
    for (const value of forbiddenValues) {
      assert.equal(error.message.includes(value), false);
      assert.equal(error.stack.includes(value), false);
    }
    return true;
  });
}

async function assertGenericScanLimitRejection(promise, forbiddenValues) {
  await assert.rejects(promise, (error) => {
    assert.equal(error.code, 'TEXT_MODEL_DISCOVERY_LIMIT');
    assert.equal(error.statusCode, 503);
    assert.equal(error.message, 'Local text model discovery exceeded its safety limits.');
    assert.equal(error.stack, `Error: ${error.message}`);
    for (const value of forbiddenValues) {
      assert.equal(error.message.includes(value), false);
      assert.equal(error.stack.includes(value), false);
    }
    return true;
  });
}
