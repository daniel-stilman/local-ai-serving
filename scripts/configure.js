'use strict';

const readline = require('node:readline/promises');
const { stdin, stdout } = require('node:process');
const {
  FIELD_RULES,
  normalizeLocalConfig,
  readLocalConfig,
  saveLocalConfig,
} = require('../local-config');

const PROMPTS = [
  ['textServerExecutable', 'Managed text-server executable'],
  ['textModelPath', 'Primary GGUF text model'],
  ['textModelsRoot', 'Folder containing selectable GGUF text models'],
  ['externalTextBaseUrl', 'External OpenAI-compatible private-network base URL'],
  ['imageModelsRoot', 'Image-model library folder'],
  ['imagePythonExecutable', 'CUDA Python executable or command'],
  ['animaTextEncoderPath', 'Anima text-encoder safetensors file'],
  ['animaVaePath', 'Anima VAE safetensors file'],
];

async function main() {
  if (process.argv.includes('--help')) {
    printHelp();
    return;
  }
  if (process.argv.length > 2) throw new Error('Unknown configure option. Use --help for usage.');

  let current = {};
  try { current = readLocalConfig(); } catch (error) {
    if (error.code !== 'LOCAL_CONFIG_ERROR') throw error;
    console.error(error.message);
    process.exitCode = 1;
    return;
  }

  const terminal = readline.createInterface({ input: stdin, output: stdout });
  const next = { ...current };
  try {
    console.log('Local configuration values are stored only in ignored config.local.json.');
    console.log('Press Enter to keep an existing value. Enter - to clear a value.');
    for (const [field, label] of PROMPTS) {
      while (true) {
        const existingHint = current[field] ? ' [currently set]' : ' [optional]';
        const answer = (await terminal.question(`${label}${existingHint}: `)).trim();
        if (!answer) break;
        if (answer === '-') {
          delete next[field];
          break;
        }
        try {
          const validated = normalizeLocalConfig({ version: 1, [field]: answer }, { checkExisting: true });
          next[field] = validated[field];
          break;
        } catch (error) {
          console.error(error.message);
        }
      }
    }
  } finally {
    terminal.close();
  }

  saveLocalConfig(next, { checkExisting: true });
  const managedText = Boolean(next.textServerExecutable && (next.textModelPath || next.textModelsRoot));
  const externalText = Boolean(next.externalTextBaseUrl);
  const image = Boolean(next.imageModelsRoot && next.imagePythonExecutable);
  console.log('Local configuration saved without printing private values.');
  if (!managedText && !externalText) console.log('Text is not configured yet.');
  if (!image) console.log('Image generation is not configured yet.');
}

function printHelp() {
  console.log('Usage: npm run configure');
  console.log('Interactively writes ignored config.local.json. Environment variables override local values.');
  console.log(`Supported fields: ${Object.keys(FIELD_RULES).join(', ')}`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.code === 'LOCAL_CONFIG_ERROR' ? error.message : 'Local configuration failed.');
    process.exitCode = 1;
  });
}

module.exports = { main, PROMPTS };
