'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const ASSET_DIR = path.join(ROOT, 'inference', 'assets');
const ASSETS = [
  {
    filename: 'clip_bpe.txt.gz',
    url: 'https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz',
    sha256: '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a',
  },
  {
    filename: 'qwen_vocab.json',
    url: 'https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/vocab.json?download=true',
    sha256: 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
  },
  {
    filename: 'qwen_merges.txt',
    url: 'https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/merges.txt?download=true',
    sha256: '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
  },
  {
    filename: 't5_tokenizer.json',
    url: 'https://huggingface.co/google-t5/t5-base/resolve/main/tokenizer.json?download=true',
    sha256: 'd2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066',
  },
];

async function main() {
  fs.mkdirSync(ASSET_DIR, { recursive: true });
  for (const asset of ASSETS) {
    const target = path.join(ASSET_DIR, asset.filename);
    if (fs.existsSync(target) && hashFile(target) === asset.sha256) {
      console.log(`Verified ${asset.filename}`);
      continue;
    }
    console.log(`Fetching ${asset.filename}`);
    const response = await fetch(asset.url, { redirect: 'follow' });
    if (!response.ok) throw new Error(`Could not fetch ${asset.filename} (HTTP ${response.status}).`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const actualHash = crypto.createHash('sha256').update(bytes).digest('hex');
    if (actualHash !== asset.sha256) {
      throw new Error(`Checksum mismatch for ${asset.filename}; refusing to store it.`);
    }
    const temporary = `${target}.${process.pid}.tmp`;
    fs.writeFileSync(temporary, bytes, { flag: 'wx' });
    if (fs.existsSync(target)) fs.rmSync(target);
    fs.renameSync(temporary, target);
    console.log(`Verified ${asset.filename}`);
  }
  console.log('Direct image inference tokenizer data is ready.');
}

function hashFile(filename) {
  return crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
