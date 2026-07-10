'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');
const zlib = require('node:zlib');

const {
  chooseImageModel,
  DEFAULT_SMOKE_BUDGETS,
  discoverTextModels,
  enforceMaximumDuration,
  enforceWarmImprovement,
  extractImageProfiles,
  inspectPngBase64,
  parseSseTranscript,
  redactLocalDiagnostics,
  assertManagedStartCount,
  resolveImageBudget,
  resolveImageKinds,
  resolveManagedTextRoot,
  resolveTextModels,
  validateTextCompletion,
  verifyManagedCatalog,
} = require('../scripts/smoke-hardware');

const SMOKE_SCRIPT = path.resolve(__dirname, '..', 'scripts', 'smoke-hardware.js');
const SMOKE_SOURCE = fs.readFileSync(SMOKE_SCRIPT, 'utf8');

const TEST_PNG = makeRgbaPng(8, 8, (x, y) => [x * 31, y * 29, (x + y) * 17, 255]);

test('hardware smoke parser requires valid SSE data and a DONE terminator', () => {
  const parsed = parseSseTranscript([
    'data: {"choices":[{"delta":{"reasoningContent":"hidden"}}]}',
    '',
    'data: {"choices":[{"delta":{"content":"SMOKE_"}}]}',
    '',
    'data: {"choices":[{"delta":{"content":"TEXT_OK"}}]}',
    '',
    'data: [DONE]',
    '',
  ].join('\r\n'));

  assert.equal(parsed.done, true);
  assert.equal(parsed.events, 3);
  assert.equal(parsed.content, 'SMOKE_TEXT_OK');
  assert.equal(parsed.reasoning, 'hidden');
  assert.doesNotThrow(() => validateTextCompletion(parsed));
  assert.equal(parseSseTranscript('data: {"choices":[]}\n\n').done, false);
  assert.throws(() => parseSseTranscript('data: {invalid}\n\n'), /invalid SSE JSON/);
  assert.throws(
    () => parseSseTranscript('data: {"error":"private-model-identifier"}\n\n'),
    (error) => {
      assert.match(error.message, /stream error/);
      assert.doesNotMatch(error.message, /private-model-identifier/);
      return true;
    },
  );
  assert.throws(() => validateTextCompletion({
    done: true,
    events: 1,
    content: '',
    reasoning: 'SMOKE_TEXT_OK',
  }), /no visible response text/);
  assert.throws(() => validateTextCompletion({
    done: true,
    events: 1,
    content: 'different visible answer',
    reasoning: 'SMOKE_TEXT_OK',
  }), /visible generated text.*smoke marker/);
});

test('hardware smoke PNG inspection validates structure and dimensions', () => {
  const inspected = inspectPngBase64(TEST_PNG.toString('base64'));
  assert.equal(inspected.width, 8);
  assert.equal(inspected.height, 8);
  assert.equal(inspected.bytes, TEST_PNG.length);
  assert.match(inspected.sha256, /^[a-f0-9]{64}$/);
  assert.equal(inspected.visibleFraction, 1);
  assert.ok(inspected.luminanceVariance > 0.5);
  assert.ok(inspected.channelRange >= 4);
  assert.throws(() => inspectPngBase64(Buffer.from('not-png').toString('base64')), /valid PNG/);
  assert.throws(() => inspectPngBase64('not base64'), /invalid base64/);
  assert.throws(() => inspectPngBase64(TEST_PNG.subarray(0, -3).toString('base64')), /truncated|missing/);
  assert.throws(() => inspectPngBase64(Buffer.concat([TEST_PNG, Buffer.from('tail')]).toString('base64')), /trailing data/);

  const corrupt = Buffer.from(TEST_PNG);
  corrupt[42] ^= 0xff;
  assert.throws(() => inspectPngBase64(corrupt.toString('base64')), /corrupt chunk/);
  const uniform = makeRgbaPng(8, 8, () => [24, 24, 24, 255]);
  assert.throws(() => inspectPngBase64(uniform.toString('base64')), /blank or near-uniform/);
  const transparent = makeRgbaPng(8, 8, (x, y) => [x * 31, y * 29, 64, 0]);
  assert.throws(() => inspectPngBase64(transparent.toString('base64')), /predominantly transparent/);
});

test('hardware smoke chooses explicit models or the fastest discovered descriptor', () => {
  const models = [
    { id: 'slow.safetensors', recommendedSteps: 28 },
    { id: 'fast-b.safetensors', recommendedSteps: 8 },
    { id: 'fast-a.safetensors', recommendedSteps: 8 },
  ];
  assert.equal(chooseImageModel(models).id, 'fast-a.safetensors');
  assert.equal(chooseImageModel(models, 'slow.safetensors').id, 'slow.safetensors');
  assert.throws(() => chooseImageModel(models, 'missing.safetensors'), /unavailable/);
});

test('hardware smoke all-model discovery mirrors safe automatic GGUF filtering', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hardware-smoke-models-'));
  try {
    const files = new Map([
      ['chat-small.gguf', 10],
      ['chat-large-00001-of-00002.gguf', 30],
      ['chat-large-00002-of-00002.gguf', 40],
      ['mmproj-chat.gguf', 50],
      ['nomic-embed-text.gguf', 60],
      ['chat-over-cap.gguf', 101],
      ['incomplete-00001-of-00002.gguf', 80],
    ]);
    for (const [filename, size] of files) fs.writeFileSync(path.join(root, filename), Buffer.alloc(size));
    assert.deepEqual(
      discoverTextModels(root, 100).map((candidate) => path.basename(candidate.path)),
      ['chat-large-00001-of-00002.gguf', 'chat-small.gguf'],
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('hardware smoke requires both image families unless partial coverage is explicit', () => {
  const complete = { models: { anima: [{ id: 'a' }], sdxl: [{ id: 's' }] } };
  const partial = { models: { anima: [{ id: 'a' }], sdxl: [] } };
  withEnvironment({ SMOKE_IMAGE_KINDS: null, SMOKE_ALLOW_PARTIAL: null }, () => {
    assert.deepEqual(resolveImageKinds(complete), ['anima', 'sdxl']);
    assert.throws(() => resolveImageKinds(partial), /No compatible sdxl/);
  });
  withEnvironment({ SMOKE_IMAGE_KINDS: 'anima,sdxl', SMOKE_ALLOW_PARTIAL: '1' }, () => {
    assert.deepEqual(resolveImageKinds(partial), ['anima']);
  });
  withEnvironment({ SMOKE_IMAGE_KINDS: 'unknown', SMOKE_ALLOW_PARTIAL: null }, () => {
    assert.throws(() => resolveImageKinds(complete), /Unsupported/);
  });
});

test('hardware smoke CLI validates options before touching hardware', () => {
  const help = spawnSync(process.execPath, [SMOKE_SCRIPT, '--help'], {
    cwd: path.resolve(__dirname, '..'),
    encoding: 'utf8',
    timeout: 5_000,
  });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /--text-only\|--all-text-models/);

  const invalid = spawnSync(process.execPath, [SMOKE_SCRIPT, '--definitely-invalid'], {
    cwd: path.resolve(__dirname, '..'),
    encoding: 'utf8',
    timeout: 5_000,
  });
  assert.equal(invalid.status, 1);
  assert.match(invalid.stderr, /Unknown smoke option/);
});

test('hardware smoke logs use anonymous runtime aliases and aggregate skip counts', () => {
  assert.match(SMOKE_SOURCE, /TEXT_MODEL_ALIAS: 'smoke-text-default'/);
  assert.match(SMOKE_SOURCE, /managed text selection\(s\) in one app process/);
  assert.doesNotMatch(SMOKE_SOURCE, /const baseName = path\.basename\(modelPath/);
  assert.doesNotMatch(SMOKE_SOURCE, /skipped\.map\(.*path\.basename/);
  assert.match(SMOKE_SOURCE, /redactLocalDiagnostics\(error\.message\)/);
  assert.match(SMOKE_SOURCE, /PRIVATE_DIAGNOSTICS: '0'/);
  assert.match(SMOKE_SOURCE, /failed with HTTP \$\{response\.status\}/);
  assert.doesNotMatch(SMOKE_SOURCE, /typeof payload\.error === 'string'/);
  assert.match(SMOKE_SOURCE, /\/api\/text\/load/);
  assert.match(SMOKE_SOURCE, /\/api\/text\/status/);
  assert.match(SMOKE_SOURCE, /never exposed its loading state/);
  assert.match(SMOKE_SOURCE, /TEXT_START_TIMEOUT_MS: process\.env\.SMOKE_TEXT_START_TIMEOUT_MS \|\| ''/);
  assert.doesNotMatch(SMOKE_SOURCE, /SMOKE_TEXT_START_TIMEOUT_MS \|\| '120000'/);
});

test('hardware smoke diagnostics redact runtime-discovered IDs and filenames', () => {
  const identifier = 'synthetic-private-model-id';
  const filename = 'synthetic-private-file.gguf';
  const redacted = redactLocalDiagnostics(
    `engine ${identifier}\nfile ${filename}\nlocal C:\\private\\${filename}`,
    [identifier, filename],
  );
  assert.doesNotMatch(redacted, new RegExp(identifier));
  assert.doesNotMatch(redacted, new RegExp(filename.replace('.', '\\.')));
  assert.match(redacted, /\[private-model\]|\[local-path\]/);
});

test('hardware smoke requires an exact, unique managed catalog without exposing identifiers', () => {
  const expected = [{ id: 'synthetic-a' }, { id: 'synthetic-b' }];
  assert.doesNotThrow(() => verifyManagedCatalog({
    data: [{ id: 'synthetic-b' }, { id: 'synthetic-a' }],
  }, expected));
  assert.throws(() => verifyManagedCatalog({ data: [{ id: 'synthetic-a' }] }, expected), /complete managed text catalog/);
  assert.throws(() => verifyManagedCatalog({
    data: [{ id: 'synthetic-a' }, { id: 'synthetic-a' }],
  }, expected), /invalid managed text model catalog/);
});

test('hardware smoke extracts only structured anonymous image profiles', () => {
  const profiles = extractImageProfiles([
    'noise',
    'IMAGE_PROFILE {"pipeline":"anima-warm","stagesSeconds":{"sampling":1.25},"totalSeconds":1.5,"peakVramMiB":1234}',
    'IMAGE_PROFILE {"pipeline":"private-model-identifier","stagesSeconds":{"sampling":1.25},"totalSeconds":1.5,"peakVramMiB":1234}',
    'IMAGE_PROFILE not-json',
  ].join('\n'));
  assert.deepEqual(profiles, [{
    pipeline: 'anima-warm',
    stagesSeconds: { sampling: 1.25 },
    totalSeconds: 1.5,
    peakVramMiB: 1234,
  }]);
});

test('hardware smoke performance gates catch absolute and warm-session regressions', () => {
  assert.deepEqual(DEFAULT_SMOKE_BUDGETS, {
    textColdMs: 30_000,
    textWarmMs: 15_000,
    textRestartMs: 30_000,
    imageColdMs: 45_000,
    imageWarmMs: 20_000,
  });
  assert.equal(resolveImageBudget('anima', 'COLD', {}), 45_000);
  assert.equal(resolveImageBudget('sdxl', 'WARM', {}), 20_000);
  assert.equal(resolveImageBudget('anima', 'WARM', {
    SMOKE_MAX_IMAGE_WARM_MS: '17000',
    SMOKE_MAX_ANIMA_WARM_MS: '13000',
  }), 13_000);
  assert.doesNotThrow(() => enforceMaximumDuration(12_000, 30_000, 'cold stage'));
  assert.throws(() => enforceMaximumDuration(30_001, 30_000, 'cold stage'), /regression ceiling/);
  withEnvironment({ SMOKE_MAX_WARM_RATIO: null }, () => {
    assert.doesNotThrow(() => enforceWarmImprovement(10_000, 8_999, 'session'));
    assert.throws(() => enforceWarmImprovement(10_000, 9_001, 'session'), /warm time/);
  });
});

test('hardware smoke enforces exact managed starts and reported GPU offload', () => {
  const full = 'Direct text engine ready (GPU offload confirmed: 49/49 layers)';
  assert.doesNotThrow(() => assertManagedStartCount(full, 1));
  assert.throws(() => assertManagedStartCount('', 1), /exactly 1 managed text start/);
  assert.throws(() => assertManagedStartCount(`${full}\n${full}`, 1), /exactly 1 managed text start/);
  const partial = 'Direct text engine ready (GPU offload confirmed: 40/49 layers)';
  withEnvironment({ SMOKE_REQUIRE_FULL_TEXT_GPU: null }, () => {
    assert.throws(() => assertManagedStartCount(partial, 1), /Only 40\/49/);
  });
  withEnvironment({ SMOKE_REQUIRE_FULL_TEXT_GPU: '0' }, () => {
    assert.doesNotThrow(() => assertManagedStartCount(partial, 1));
  });
});

test('hardware smoke text-model resolution preserves explicit order and deduplicates all-mode discovery', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hardware-smoke-resolution-'));
  try {
    const first = path.join(root, 'explicit-first.gguf');
    const second = path.join(root, 'configured-second.gguf');
    fs.writeFileSync(first, Buffer.alloc(10));
    fs.writeFileSync(second, Buffer.alloc(20));
    withEnvironment({
      SMOKE_TEXT_MODEL_PATH: first,
      TEXT_MODEL_PATH: null,
      SMOKE_TEXT_MODELS: `${second};${first}`,
      SMOKE_TEXT_MODELS_ROOT: root,
      SMOKE_MAX_TEXT_MODELS: '2',
      TEXT_MODEL_MAX_GIB: '10',
    }, () => {
      assert.deepEqual(resolveTextModels(true), [first, second]);
      process.env.SMOKE_MAX_TEXT_MODELS = '1';
      assert.deepEqual(resolveTextModels(true), [first]);
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('hardware smoke bounds default managed-root coverage to two models and reserves breadth for all-mode', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hardware-smoke-bounded-'));
  try {
    const small = path.join(root, 'synthetic-small.gguf');
    const medium = path.join(root, 'synthetic-medium.gguf');
    const large = path.join(root, 'synthetic-large.gguf');
    const overCap = path.join(root, 'synthetic-over-cap.gguf');
    const largerOverCap = path.join(root, 'synthetic-larger-over-cap.gguf');
    fs.writeFileSync(small, Buffer.alloc(10));
    fs.writeFileSync(medium, Buffer.alloc(20));
    fs.writeFileSync(large, Buffer.alloc(30));
    fs.closeSync(fs.openSync(overCap, 'w'));
    fs.truncateSync(overCap, 1024 ** 3 + 1);
    fs.closeSync(fs.openSync(largerOverCap, 'w'));
    fs.truncateSync(largerOverCap, 1024 ** 3 + 2);
    withEnvironment({
      SMOKE_TEXT_MODEL_PATH: null,
      TEXT_MODEL_PATH: null,
      SMOKE_TEXT_MODELS: null,
      SMOKE_TEXT_MODELS_ROOT: root,
      TEXT_MODELS_ROOT: null,
      SMOKE_MAX_TEXT_MODELS: null,
      TEXT_MODEL_MAX_GIB: '1',
    }, () => {
      assert.deepEqual(resolveTextModels(false), [large, medium]);
      assert.deepEqual(resolveTextModels(true), [large, overCap, largerOverCap, medium, small]);
      assert.equal(resolveManagedTextRoot(resolveTextModels(false)), root);
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

function makeRgbaPng(width, height, pixelAt) {
  const bytesPerPixel = 4;
  const rawRows = [];
  for (let y = 0; y < height; y += 1) {
    const row = Buffer.alloc(width * bytesPerPixel);
    for (let x = 0; x < width; x += 1) {
      const pixel = pixelAt(x, y);
      for (let channel = 0; channel < bytesPerPixel; channel += 1) {
        row[x * bytesPerPixel + channel] = pixel[channel];
      }
    }
    rawRows.push(row);
  }
  const filteredRows = rawRows.map((row, y) => {
    const filter = y % 5;
    const encoded = Buffer.alloc(row.length + 1);
    encoded[0] = filter;
    for (let x = 0; x < row.length; x += 1) {
      const left = x >= bytesPerPixel ? row[x - bytesPerPixel] : 0;
      const up = y > 0 ? rawRows[y - 1][x] : 0;
      const upperLeft = y > 0 && x >= bytesPerPixel ? rawRows[y - 1][x - bytesPerPixel] : 0;
      let predictor = 0;
      if (filter === 1) predictor = left;
      else if (filter === 2) predictor = up;
      else if (filter === 3) predictor = Math.floor((left + up) / 2);
      else if (filter === 4) predictor = testPaeth(left, up, upperLeft);
      encoded[x + 1] = (row[x] - predictor) & 0xff;
    }
    return encoded;
  });
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  return Buffer.concat([
    Buffer.from('89504e470d0a1a0a', 'hex'),
    makePngChunk('IHDR', header),
    makePngChunk('IDAT', zlib.deflateSync(Buffer.concat(filteredRows))),
    makePngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function makePngChunk(type, data) {
  const typeBytes = Buffer.from(type, 'ascii');
  const chunk = Buffer.alloc(12 + data.length);
  chunk.writeUInt32BE(data.length, 0);
  typeBytes.copy(chunk, 4);
  data.copy(chunk, 8);
  chunk.writeUInt32BE(testCrc32(Buffer.concat([typeBytes, data])), 8 + data.length);
  return chunk;
}

function testCrc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function testPaeth(left, up, upperLeft) {
  const estimate = left + up - upperLeft;
  const leftDistance = Math.abs(estimate - left);
  const upDistance = Math.abs(estimate - up);
  const upperLeftDistance = Math.abs(estimate - upperLeft);
  if (leftDistance <= upDistance && leftDistance <= upperLeftDistance) return left;
  return upDistance <= upperLeftDistance ? up : upperLeft;
}

function withEnvironment(values, callback) {
  const previous = new Map();
  for (const [name, value] of Object.entries(values)) {
    previous.set(name, process.env[name]);
    if (value === null) delete process.env[name];
    else process.env[name] = value;
  }
  try {
    return callback();
  } finally {
    for (const [name, value] of previous) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
}
