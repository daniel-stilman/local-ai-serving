'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { readLocalConfig } = require('../local-config');

const ROOT = path.resolve(__dirname, '..');
const SELF = 'scripts/check-publish.js';
const PROTECTED_EXTENSIONS = new Set([
  '.safetensors', '.gguf', '.ggml', '.ckpt', '.pt', '.pth', '.onnx', '.bin',
  '.engine', '.plan', '.tflite', '.mlmodel', '.pb', '.h5', '.hdf5', '.weights',
  '.params', '.npy', '.npz', '.pkl', '.pickle', '.joblib',
]);
const GENERIC_MODEL_STEMS = new Set([
  'chat', 'checkpoint', 'decoder', 'diffusion_model', 'encoder', 'model',
  'text_encoder', 'vae', 'weights',
]);
const failures = new Map();
const failureFiles = new Map();
let inspectedFile = '';

main();

function main() {
  const files = publicationFiles();
  const privateNeedles = collectPrivateNeedles();
  for (const relativePath of files) {
    inspectedFile = relativePath;
    inspectFile(relativePath, privateNeedles);
  }
  inspectedFile = '';
  checkIgnoredLocalConfig();
  if (process.argv.includes('--fresh-history')) checkFreshHistory();

  if (failures.size) {
    for (const [category, count] of failures) {
      console.error(`[privacy] ${category}: ${count}`);
      if (process.argv.includes('--diagnostic-paths')) {
        for (const relativePath of [...(failureFiles.get(category) || [])].sort()) {
          console.error(`[privacy]   ${relativePath}`);
        }
      }
    }
    console.error('[privacy] FAIL - publication tree is not sanitized.');
    process.exitCode = 1;
    return;
  }
  console.log(`[privacy] PASS - ${files.length} publication files checked without printing private values.`);
}

function publicationFiles() {
  const git = runGit(['ls-files', '-co', '--exclude-standard', '-z']);
  if (git.status !== 0 || typeof git.stdout !== 'string') {
    fail('Git file enumeration failed');
    return [];
  }
  return [...new Set(git.stdout.split('\0').filter(Boolean))]
    .filter((relativePath) => relativePath !== 'config.local.json')
    .sort((left, right) => left.localeCompare(right));
}

function inspectFile(relativePath, privateNeedles) {
  const absolutePath = path.join(ROOT, relativePath);
  let stats;
  try { stats = fs.lstatSync(absolutePath); } catch {
    fail('Unreadable publication file');
    return;
  }
  if (stats.isSymbolicLink()) {
    fail('Symbolic link in publication tree');
    return;
  }
  if (!stats.isFile()) return;
  if (PROTECTED_EXTENSIONS.has(path.extname(relativePath).toLowerCase())) {
    fail('Tracked or unignored model-like artifact');
  }
  if (stats.size > 8 * 1024 * 1024 || relativePath.startsWith('inference/assets/')) return;

  let bytes;
  try { bytes = fs.readFileSync(absolutePath); } catch {
    fail('Unreadable publication file');
    return;
  }
  if (bytes.includes(0)) return;
  const text = bytes.toString('utf8');
  const lower = text.toLowerCase();
  for (const [needle, sourceKind] of privateNeedles) {
    const matchIndex = lower.indexOf(needle);
    if (matchIndex !== -1) {
      const lineNumber = text.slice(0, matchIndex).split(/\r?\n/).length;
      fail(
        'Private machine or model identifier',
        `${relativePath}:${lineNumber} (${sourceKind}, ${needle.length} chars)`,
      );
      break;
    }
  }
  if (relativePath === SELF) return;

  let staticText = text;
  if (relativePath === 'test/server.test.js') {
    staticText = staticText.replaceAll('10.0.0.1evil', '');
  }
  if (relativePath === 'test/lifecycle-regressions.test.js') {
    staticText = staticText.replaceAll('lifecycle-regression-token', '');
  }
  if (/[A-Za-z]:\\Users\\[^\\\r\n]+|\/(?:Users|home)\/[A-Za-z0-9._-]+/i.test(staticText)) {
    fail('User-home path literal');
  }
  if (/\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})\b/.test(staticText)) {
    fail('Unallowlisted private-network literal');
  }
  const credentialLike = [
    /-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY-----/i,
    /\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[opurs]_[A-Za-z0-9]{20,})\b/,
    /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/,
    /\b(?:sk-(?:ant-api)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35})\b/,
    /\bxox[baprs]-[A-Za-z0-9-]{20,}\b|\bsk_live_[A-Za-z0-9]{16,}\b/,
    /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/,
    /(?:api[_-]?key|client[_-]?secret|password|passwd|access[_-]?token|auth[_-]?token)\s*[:=]\s*["'][^"']{20,}["']/i,
    /(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis):\/\/[^\s:@/]+:[^\s@/]+@/i,
  ].some((pattern) => pattern.test(staticText));
  if (credentialLike) {
    fail('Credential-like literal');
  }
  const nonExampleEmailText = staticText.replace(
    /[A-Za-z0-9._%+-]+@(?:example\.(?:com|org|net)|users\.noreply\.github\.com)/gi,
    '',
  );
  if (/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/.test(nonExampleEmailText)) {
    fail('Email-address literal');
  }
  if (/\.lmstudio|Documents[\\/]+ComfyUI|LM_STUDIO|local-lm-studio/i.test(staticText)) {
    fail('Provider-specific machine-layout assumption');
  }
  if (/version https:\/\/git-lfs\.github\.com\/spec\/v1/.test(staticText)) {
    fail('Git LFS pointer in publication tree');
  }
}

function collectPrivateNeedles() {
  const values = new Map();
  const add = (value, minimumLength = 5, sourceKind = 'machine') => {
    const normalized = String(value || '').trim().toLowerCase();
    if (normalized.length >= minimumLength && !values.has(normalized)) values.set(normalized, sourceKind);
  };
  add(os.userInfo().username, 5, 'account');
  add(os.homedir(), 5, 'home-path');
  add(os.homedir().replaceAll('\\', '/'), 5, 'home-path');
  add(ROOT, 5, 'workspace-path');
  add(ROOT.replaceAll('\\', '/'), 5, 'workspace-path');
  add(os.hostname(), 5, 'host');
  for (const addresses of Object.values(os.networkInterfaces())) {
    for (const address of addresses || []) {
      if (!address.internal) add(address.address, 5, 'network');
    }
  }
  add(os.cpus()[0]?.model, 10, 'hardware');

  const gpu = spawnSync('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], {
    encoding: 'utf8',
    windowsHide: true,
    timeout: 5_000,
  });
  if (gpu.status === 0) {
    for (const line of gpu.stdout.split(/\r?\n/)) add(line, 10, 'hardware');
  }

  let config = {};
  try { config = readLocalConfig(); } catch { fail('Ignored local configuration is invalid'); }
  for (const value of Object.values(config)) {
    if (typeof value !== 'string') continue;
    add(value, 5, 'local-config-path');
    add(value.replaceAll('\\', '/'), 5, 'local-config-path');
  }
  for (const modelPathField of ['textModelPath', 'animaTextEncoderPath', 'animaVaePath']) {
    const value = config[modelPathField];
    if (!value || !path.isAbsolute(value)) continue;
    addModelFilename(value, add);
  }
  for (const rootField of ['textModelsRoot', 'imageModelsRoot']) {
    const directory = config[rootField];
    if (directory) collectModelNames(directory, add);
  }
  return values;
}

function collectModelNames(directory, add) {
  const stack = [directory];
  let visited = 0;
  while (stack.length && visited < 20_000) {
    const current = stack.pop();
    let entries;
    try { entries = fs.readdirSync(current, { withFileTypes: true }); } catch { continue; }
    for (const entry of entries) {
      visited += 1;
      if (entry.isSymbolicLink()) continue;
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(entryPath);
      else if (entry.isFile() && PROTECTED_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
        addModelFilename(entry.name, add);
      }
    }
  }
}

function addModelFilename(value, add) {
  const basename = path.basename(value);
  const stem = path.basename(basename, path.extname(basename));
  const normalized = stem.toLowerCase().replace(/[\s.-]+/g, '_');
  if (normalized.length < 4) return;
  if (GENERIC_MODEL_STEMS.has(normalized)) return;
  if (/^(?:model|pytorch_model|diffusion_pytorch_model)_?\d*_of_\d+$/.test(normalized)) return;
  add(basename, 10, 'local-model-name');
  add(stem, 10, 'local-model-name');
}

function checkIgnoredLocalConfig() {
  if (!fs.existsSync(path.join(ROOT, 'config.local.json'))) return;
  const ignored = runGit(['check-ignore', '-q', 'config.local.json']);
  if (ignored.status !== 0) fail('Local configuration is not ignored');
}

function checkFreshHistory() {
  const history = runGit(['rev-list', '--parents', '--all']);
  const lines = String(history.stdout || '').trim().split(/\r?\n/).filter(Boolean);
  const rootOid = lines.length === 1 && lines[0].trim().split(/\s+/).length === 1
    ? lines[0].trim()
    : '';
  if (history.status !== 0 || lines.length !== 1 || lines[0].trim().split(/\s+/).length !== 1) {
    fail('Repository history is not one parentless commit');
  }
  const reflog = runGit(['rev-list', '--reflog', '--all']);
  const reflogObjects = new Set(String(reflog.stdout || '').trim().split(/\r?\n/).filter(Boolean));
  if (reflog.status !== 0 || reflogObjects.size !== 1) fail('Reflog contains additional history');
  const refs = runGit(['for-each-ref', '--format=%(objectname)']);
  const refObjects = new Set(String(refs.stdout || '').trim().split(/\r?\n/).filter(Boolean));
  if (refs.status !== 0 || refObjects.size !== 1 || !refObjects.has(rootOid)) {
    fail('Repository references do not point to the single root commit');
  }
  const replacements = runGit(['for-each-ref', '--format=%(refname)', 'refs/replace']);
  if (replacements.status !== 0 || String(replacements.stdout || '').trim()) {
    fail('Git replacement reference remains');
  }
  const unreachable = runGit(['fsck', '--full', '--unreachable', '--no-reflogs']);
  if (unreachable.status !== 0) fail('Git object validation failed');
  if (/unreachable (?:commit|tree|blob|tag)/i.test(`${unreachable.stdout}\n${unreachable.stderr}`)) {
    fail('Unreachable old Git objects remain');
  }
  const commonDir = runGit(['rev-parse', '--git-common-dir']);
  const resolvedCommonDir = path.resolve(ROOT, String(commonDir.stdout || '').trim());
  if (commonDir.status !== 0 || resolvedCommonDir !== path.join(ROOT, '.git')) {
    fail('Repository does not use standalone in-project Git metadata');
  }
  for (const relativePath of [
    '.git/shallow', '.git/info/grafts', '.git/objects/info/alternates',
  ]) {
    if (fs.existsSync(path.join(ROOT, relativePath))) fail('Non-standalone Git history mechanism present');
  }
  for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
    if (entry.isDirectory() && /^\.git(?:[-_.].+)?$/i.test(entry.name) && entry.name !== '.git') {
      fail('Old Git metadata backup remains in the project');
    }
  }
  const status = runGit(['status', '--porcelain=v1', '-uall']);
  if (status.status !== 0 || String(status.stdout || '').trim()) fail('Publication tree is not clean');
  const author = runGit(['log', '-1', '--format=%ae']);
  if (!/@users\.noreply\.github\.com\s*$/i.test(String(author.stdout || ''))) {
    fail('Commit author email is not privacy-safe');
  }
  const dates = runGit(['log', '-1', '--format=%aI%n%cI']);
  const dateLines = String(dates.stdout || '').trim().split(/\r?\n/).filter(Boolean);
  if (dates.status !== 0 || dateLines.length !== 2 || dateLines.some((value) => !/(?:Z|\+00:00)$/.test(value))) {
    fail('Commit timestamps do not use UTC');
  }
}

function runGit(args) {
  return spawnSync('git', args, {
    cwd: ROOT,
    encoding: 'utf8',
    windowsHide: true,
    timeout: 60_000,
  });
}

function fail(category, location = inspectedFile) {
  failures.set(category, (failures.get(category) || 0) + 1);
  if (location) {
    if (!failureFiles.has(category)) failureFiles.set(category, new Set());
    failureFiles.get(category).add(location);
  }
}
