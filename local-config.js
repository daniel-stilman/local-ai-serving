'use strict';

const fs = require('node:fs');
const path = require('node:path');

const CONFIG_VERSION = 1;
const DEFAULT_CONFIG_PATH = path.join(__dirname, 'config.local.json');
const FIELD_RULES = Object.freeze({
  textServerExecutable: { environment: 'TEXT_SERVER_EXE', type: 'file' },
  textModelPath: { environment: 'TEXT_MODEL_PATH', type: 'gguf' },
  textModelsRoot: { environment: 'TEXT_MODELS_ROOT', type: 'directory' },
  externalTextBaseUrl: { environment: 'TEXT_BASE_URL', type: 'url' },
  imageModelsRoot: { environment: 'IMAGE_MODELS_ROOT', type: 'directory' },
  imagePythonExecutable: { environment: 'IMAGE_PYTHON', type: 'command' },
  animaTextEncoderPath: { environment: 'ANIMA_TEXT_ENCODER_PATH', type: 'safetensors' },
  animaVaePath: { environment: 'ANIMA_VAE_PATH', type: 'safetensors' },
});

function getLocalConfigPath(environment = process.env) {
  return environment.LOCAL_CONFIG_FILE
    ? path.resolve(environment.LOCAL_CONFIG_FILE)
    : DEFAULT_CONFIG_PATH;
}

function readLocalConfig(options = {}) {
  const environment = options.environment || process.env;
  if (isEnabled(environment.LOCAL_CONFIG_DISABLED)) return {};
  const filePath = options.filePath || getLocalConfigPath(environment);
  let source;
  try {
    source = fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    if (error.code === 'ENOENT') return {};
    throw configError('The local configuration could not be read.');
  }

  let parsed;
  try {
    parsed = JSON.parse(source);
  } catch {
    throw configError('The local configuration is not valid JSON.');
  }
  return normalizeLocalConfig(parsed, { checkExisting: false });
}

function applyLocalConfig(environment = process.env, options = {}) {
  const config = readLocalConfig({ ...options, environment });
  for (const [field, rule] of Object.entries(FIELD_RULES)) {
    if (
      !Object.prototype.hasOwnProperty.call(environment, rule.environment)
      && config[field]
    ) {
      let validated;
      try {
        validated = normalizeLocalConfig(
          { version: CONFIG_VERSION, [field]: config[field] },
          { checkExisting: true },
        );
      } catch (error) {
        if (options.ignoreUnavailable === true && error?.code === 'LOCAL_CONFIG_ERROR') continue;
        throw error;
      }
      environment[rule.environment] = validated[field];
    }
  }
  return config;
}

function saveLocalConfig(config, options = {}) {
  const environment = options.environment || process.env;
  const filePath = options.filePath || getLocalConfigPath(environment);
  const normalized = normalizeLocalConfig(config, {
    checkExisting: options.checkExisting !== false,
  });
  const serialized = `${JSON.stringify({ version: CONFIG_VERSION, ...normalized }, null, 2)}\n`;
  const temporaryPath = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(temporaryPath, serialized, { encoding: 'utf8', mode: 0o600, flag: 'wx' });
    fs.renameSync(temporaryPath, filePath);
    try { fs.chmodSync(filePath, 0o600); } catch {}
  } catch {
    try { fs.rmSync(temporaryPath, { force: true }); } catch {}
    throw configError('The local configuration could not be saved.');
  }
  return normalized;
}

function normalizeLocalConfig(value, options = {}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw configError('The local configuration must be a JSON object.');
  }
  if (value.version !== undefined && value.version !== CONFIG_VERSION) {
    throw configError(`The local configuration version must be ${CONFIG_VERSION}.`);
  }

  const allowed = new Set(['version', ...Object.keys(FIELD_RULES)]);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw configError(`Unknown local configuration field: ${key}.`);
  }

  const normalized = {};
  for (const [field, rule] of Object.entries(FIELD_RULES)) {
    const raw = value[field];
    if (raw === undefined || raw === null || raw === '') continue;
    if (typeof raw !== 'string') throw configError(`Local configuration field ${field} must be text.`);
    const candidate = stripWrappingQuotes(raw.trim());
    if (!candidate) continue;
    validateField(field, candidate, rule, Boolean(options.checkExisting));
    normalized[field] = candidate;
  }
  return normalized;
}

function validateField(field, candidate, rule, checkExisting) {
  if (rule.type === 'url') {
    let parsed;
    try { parsed = new URL(candidate); } catch {}
    if (!parsed || !['http:', 'https:'].includes(parsed.protocol)) {
      throw configError(`Local configuration field ${field} must be an HTTP(S) URL.`);
    }
    if (parsed.username || parsed.password) {
      throw configError(`Local configuration field ${field} must not contain credentials.`);
    }
    return;
  }

  if (rule.type === 'command' && !path.isAbsolute(candidate)) {
    if (candidate.includes('/') || candidate.includes('\\')) {
      throw configError(`Local configuration field ${field} must be an absolute path or command name.`);
    }
    return;
  }
  if (!path.isAbsolute(candidate)) {
    throw configError(`Local configuration field ${field} must be an absolute path.`);
  }
  if (rule.type === 'gguf' && path.extname(candidate).toLowerCase() !== '.gguf') {
    throw configError(`Local configuration field ${field} must select a GGUF file.`);
  }
  if (rule.type === 'safetensors' && path.extname(candidate).toLowerCase() !== '.safetensors') {
    throw configError(`Local configuration field ${field} must select a safetensors file.`);
  }
  if (!checkExisting) return;

  let stats;
  try { stats = fs.statSync(candidate); } catch {}
  const expectsDirectory = rule.type === 'directory';
  if (!stats || (expectsDirectory ? !stats.isDirectory() : !stats.isFile())) {
    throw configError(`The selected value for ${field} does not exist or has the wrong type.`);
  }
}

function stripWrappingQuotes(value) {
  if (value.length >= 2 && (
    (value.startsWith('"') && value.endsWith('"'))
    || (value.startsWith("'") && value.endsWith("'"))
  )) return value.slice(1, -1).trim();
  return value;
}

function configError(message) {
  return Object.assign(new Error(message), { code: 'LOCAL_CONFIG_ERROR' });
}

function isEnabled(value) {
  return ['1', 'true', 'on', 'yes'].includes(String(value || '').trim().toLowerCase());
}

module.exports = {
  CONFIG_VERSION,
  DEFAULT_CONFIG_PATH,
  FIELD_RULES,
  applyLocalConfig,
  getLocalConfigPath,
  normalizeLocalConfig,
  readLocalConfig,
  saveLocalConfig,
};
