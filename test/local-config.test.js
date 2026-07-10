'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const CONFIGURE_SCRIPT = path.join(ROOT, 'scripts', 'configure.js');
const {
  applyLocalConfig,
  normalizeLocalConfig,
  readLocalConfig,
  saveLocalConfig,
} = require('../local-config');

test('local configuration is strict, private, and environment-overridable', () => {
  const fixture = makeFixture();
  try {
    const configPath = path.join(fixture.root, 'config.local.json');
    const saved = saveLocalConfig({
      version: 1,
      textServerExecutable: fixture.textServer,
      textModelPath: fixture.textModel,
      textModelsRoot: fixture.textModelsRoot,
      imageModelsRoot: fixture.imageModelsRoot,
      imagePythonExecutable: fixture.python,
      animaTextEncoderPath: fixture.encoder,
      animaVaePath: fixture.vae,
    }, { filePath: configPath, checkExisting: true });

    assert.deepEqual(readLocalConfig({ filePath: configPath }), saved);
    const environment = { TEXT_MODEL_PATH: '' };
    applyLocalConfig(environment, { filePath: configPath });
    assert.equal(environment.TEXT_MODEL_PATH, '', 'an explicitly present environment value must win');
    assert.equal(environment.TEXT_SERVER_EXE, fixture.textServer);
    assert.equal(environment.IMAGE_MODELS_ROOT, fixture.imageModelsRoot);
    assert.equal(environment.ANIMA_TEXT_ENCODER_PATH, fixture.encoder);
    assert.doesNotMatch(fs.readFileSync(configPath, 'utf8'), /access.?token|api.?key/i);
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true });
  }
});

test('local configuration rejects typos, relative paths, wrong types, and unsupported versions', () => {
  assert.throws(() => normalizeLocalConfig({ version: 2 }), /version must be 1/);
  assert.throws(() => normalizeLocalConfig({ version: 1, surprise: 'value' }), /Unknown/);
  assert.throws(() => normalizeLocalConfig({ version: 1, textModelPath: 'relative.gguf' }), /absolute path/);
  assert.throws(() => normalizeLocalConfig({ version: 1, textModelPath: 42 }), /must be text/);
  assert.throws(() => normalizeLocalConfig({ version: 1, textModelPath: path.resolve('wrong.bin') }), /GGUF/);
  assert.throws(() => normalizeLocalConfig({ version: 1, externalTextBaseUrl: 'file:///private' }), /HTTP/);
  assert.doesNotThrow(() => normalizeLocalConfig({ version: 1, imagePythonExecutable: 'python3' }));
});

test('external text URLs reject embedded credentials without echoing them', () => {
  const credentialUrl = new URL('https://example.invalid/v1');
  credentialUrl.username = 'fixture-user';
  credentialUrl.password = 'fixture-secret';
  const privateUrl = credentialUrl.toString();
  assert.throws(
    () => normalizeLocalConfig({ version: 1, externalTextBaseUrl: privateUrl }),
    (error) => {
      assert.equal(error.code, 'LOCAL_CONFIG_ERROR');
      assert.match(error.message, /must not contain credentials/);
      assert.doesNotMatch(error.message, /fixture-user|fixture-secret|example\.invalid/);
      assert.equal(error.message.includes(privateUrl), false);
      return true;
    },
  );
  const passwordOnlyUrl = new URL('http://example.invalid');
  passwordOnlyUrl.password = 'fixture-secret';
  assert.throws(() => normalizeLocalConfig(
    { version: 1, externalTextBaseUrl: passwordOnlyUrl.toString() },
  ), /must not contain credentials/);
});

test('stale lower-priority paths do not break an explicit environment override', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-config-override-'));
  try {
    const configPath = path.join(root, 'config.local.json');
    saveLocalConfig({
      version: 1,
      textModelPath: path.join(root, 'missing.gguf'),
    }, { filePath: configPath, checkExisting: false });
    const environment = { TEXT_MODEL_PATH: '' };
    assert.doesNotThrow(() => applyLocalConfig(environment, { filePath: configPath }));
    assert.equal(environment.TEXT_MODEL_PATH, '');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('server-tolerant loading skips stale resources while preserving valid local settings', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-config-stale-'));
  try {
    const configPath = path.join(root, 'config.local.json');
    saveLocalConfig({
      version: 1,
      textModelPath: path.join(root, 'missing.gguf'),
      textModelsRoot: path.join(root, 'missing-models'),
      externalTextBaseUrl: 'http://127.0.0.1:65530/v1',
    }, { filePath: configPath, checkExisting: false });

    assert.throws(() => applyLocalConfig({}, { filePath: configPath }), /does not exist or has the wrong type/);
    const environment = {};
    assert.doesNotThrow(() => applyLocalConfig(environment, {
      filePath: configPath,
      ignoreUnavailable: true,
    }));
    assert.equal(environment.TEXT_MODEL_PATH, undefined);
    assert.equal(environment.TEXT_MODELS_ROOT, undefined);
    assert.equal(environment.TEXT_BASE_URL, 'http://127.0.0.1:65530/v1');

    fs.writeFileSync(configPath, '{invalid');
    assert.throws(() => applyLocalConfig(environment, {
      filePath: configPath,
      ignoreUnavailable: true,
    }), /valid JSON/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('missing or explicitly disabled local configuration is harmless', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-config-disabled-'));
  try {
    const missing = path.join(root, 'missing.json');
    assert.deepEqual(readLocalConfig({ filePath: missing }), {});
    const invalid = path.join(root, 'invalid.json');
    fs.writeFileSync(invalid, '{invalid');
    assert.deepEqual(readLocalConfig({
      filePath: invalid,
      environment: { LOCAL_CONFIG_DISABLED: '1' },
    }), {});
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('configure CLI documents anonymous local storage and the file is ignored', () => {
  const result = spawnSync(process.execPath, [CONFIGURE_SCRIPT, '--help'], {
    cwd: ROOT,
    encoding: 'utf8',
    timeout: 5_000,
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /ignored config\.local\.json/);
  const ignore = fs.readFileSync(path.join(ROOT, '.gitignore'), 'utf8');
  assert.match(ignore, /^config\.local\.json$/m);
  const example = JSON.parse(fs.readFileSync(path.join(ROOT, 'config.example.json'), 'utf8'));
  assert.ok(Object.values(example).every((value) => value === 1 || value === ''));
});

function makeFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-config-test-'));
  const textModelsRoot = path.join(root, 'text-models');
  const imageModelsRoot = path.join(root, 'image-models');
  fs.mkdirSync(textModelsRoot);
  fs.mkdirSync(imageModelsRoot);
  const files = {
    textServer: path.join(root, 'text-server.bin'),
    textModel: path.join(textModelsRoot, 'chat.gguf'),
    python: path.join(root, 'python.bin'),
    encoder: path.join(root, 'encoder.safetensors'),
    vae: path.join(root, 'vae.safetensors'),
  };
  for (const filePath of Object.values(files)) fs.writeFileSync(filePath, 'fixture');
  return { root, textModelsRoot, imageModelsRoot, ...files };
}
