'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const PYTHON = process.env.IMAGE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

const fixture = JSON.parse(execFileSync(PYTHON, ['-c', String.raw`
import json
import sys

sys.path.insert(0, 'inference')
from runtime import ClipTokenizer, QwenTokenizer, T5Tokenizer, _scan_qwen

clip = ClipTokenizer()
qwen = QwenTokenizer()
t5 = T5Tokenizer()
print(json.dumps({
    'clip_numbers': clip.encode('room 2026')[:7],
    'clip_special': clip.encode('<|endoftext|>')[:3],
    'clip_nfc': clip.encode('\u00e9'),
    'clip_decomposed': clip.encode('e\u0301'),
    'qwen_contraction': qwen.encode("can't"),
    'qwen_upper_contraction': qwen.encode("WE'RE"),
    'qwen_newlines': qwen.encode('a\n\nb'),
    'qwen_special': qwen.encode('<|endoftext|>'),
    'qwen_think': qwen.encode('<think>'),
    'qwen_tool_call': qwen.encode('<tool_call>'),
    'qwen_precomposed': qwen.encode('\u00e9'),
    'qwen_decomposed': qwen.encode('e\u0301'),
    'qwen_empty': qwen.encode(''),
    'qwen_whitespace': qwen.encode('   '),
    'qwen_scan_spaces': _scan_qwen('a  b'),
    'qwen_scan_punctuation_newlines': _scan_qwen('hello!!!\n\nworld'),
    't5_empty': t5.encode(''),
    't5_whitespace': t5.encode('  \t\n  '),
}, ensure_ascii=True))
`], {
  cwd: ROOT,
  encoding: 'utf8',
  env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', PYTHONIOENCODING: 'utf-8' },
}));

test('CLIP tokenization matches canonical numeric, special-token, and NFC behavior', () => {
  assert.deepEqual(fixture.clip_numbers, [49406, 1530, 273, 271, 273, 277, 49407]);
  assert.deepEqual(fixture.clip_special, [49406, 49407, 49407]);
  assert.deepEqual(fixture.clip_decomposed, fixture.clip_nfc);
});

test('Qwen tokenization preserves contractions, newline groups, and added tokens', () => {
  assert.deepEqual(fixture.qwen_contraction, [4814, 944]);
  assert.deepEqual(fixture.qwen_upper_contraction, [12457, 94153]);
  assert.deepEqual(fixture.qwen_newlines, [64, 271, 65]);
  assert.deepEqual(fixture.qwen_special, [151643]);
  assert.deepEqual(fixture.qwen_think, [151667]);
  assert.deepEqual(fixture.qwen_tool_call, [151657]);
  assert.deepEqual(fixture.qwen_precomposed, [963]);
  assert.deepEqual(fixture.qwen_decomposed, [68, 53839]);
  assert.deepEqual(fixture.qwen_empty, []);
  assert.deepEqual(fixture.qwen_whitespace, [262]);
  assert.deepEqual(fixture.qwen_scan_spaces, ['a', ' ', ' b']);
  assert.deepEqual(fixture.qwen_scan_punctuation_newlines, ['hello', '!!!\n\n', 'world']);
});

test('T5 blank prompts encode as EOS only', () => {
  assert.deepEqual(fixture.t5_empty, [1]);
  assert.deepEqual(fixture.t5_whitespace, [1]);
});
