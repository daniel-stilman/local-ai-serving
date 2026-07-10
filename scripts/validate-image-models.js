'use strict';

const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { applyLocalConfig } = require('../local-config');

const root = path.resolve(__dirname, '..');
try {
  applyLocalConfig(process.env);
} catch (error) {
  console.error(error.code === 'LOCAL_CONFIG_ERROR' ? error.message : 'Local configuration failed.');
  process.exit(1);
}
if (!process.env.IMAGE_MODELS_ROOT) {
  console.error('Image models are not configured. Run npm run configure.');
  process.exit(1);
}
const python = process.env.IMAGE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const args = [
  path.join(root, 'inference', 'validate_models.py'),
  '--models-root', process.env.IMAGE_MODELS_ROOT,
];

const result = spawnSync(python, args, {
  cwd: root,
  env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
  stdio: 'inherit',
});

if (result.error) {
  console.error(`Could not start the image model validator: ${result.error.message}`);
  process.exitCode = 1;
} else {
  process.exitCode = result.status ?? 1;
}
