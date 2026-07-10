'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const PYTHON = process.env.IMAGE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

const results = JSON.parse(execFileSync(PYTHON, ['-B', '-c', String.raw`
import gc
import json
import struct
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', message='The given buffer is not writable')
sys.path.insert(0, 'inference')
from runtime import InferenceError, SafeTensorFile

def write_file(path, header, data=b''):
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)

def write_raw_header(path, encoded, data=b''):
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)

def rejected(path):
    try:
        reader = SafeTensorFile(path)
    except InferenceError:
        return True
    reader.close()
    return False

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    valid = root / 'valid.safetensors'
    write_file(valid, {
        'weights': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]},
    }, struct.pack('<ff', 1.5, 2.5))
    reader = SafeTensorFile(valid)
    tensor = reader.tensor('weights')
    valid_values = tensor.tolist()
    del tensor
    gc.collect()
    reader.close()

    undersized = root / 'undersized.safetensors'
    write_file(undersized, {
        'weights': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 4]},
    }, b'\0' * 8)

    oversized = root / 'oversized.safetensors'
    write_file(oversized, {
        'weights': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 12]},
    }, b'\0' * 12)

    overlapping = root / 'overlapping.safetensors'
    write_file(overlapping, {
        'left': {'dtype': 'F32', 'shape': [], 'data_offsets': [0, 4]},
        'right': {'dtype': 'F32', 'shape': [], 'data_offsets': [2, 6]},
    }, b'\0' * 6)

    non_object = root / 'non-object.safetensors'
    write_file(non_object, [], b'\0')

    gap = root / 'gap.safetensors'
    write_file(gap, {
        'weights': {'dtype': 'F32', 'shape': [], 'data_offsets': [1, 5]},
    }, b'\0' * 5)

    trailing = root / 'trailing.safetensors'
    write_file(trailing, {
        'weights': {'dtype': 'F32', 'shape': [], 'data_offsets': [0, 4]},
    }, b'\0' * 8)

    invalid_metadata = root / 'invalid-metadata.safetensors'
    write_file(invalid_metadata, {
        '__metadata__': {'format': 123},
        'weights': {'dtype': 'F32', 'shape': [], 'data_offsets': [0, 4]},
    }, b'\0' * 4)

    duplicate = root / 'duplicate.safetensors'
    duplicate_header = (
        b'{"weights":{"dtype":"F32","shape":[],"data_offsets":[0,4]},'
        b'"weights":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}'
    )
    write_raw_header(duplicate, duplicate_header, b'\0' * 4)

    leading_space = root / 'leading-space.safetensors'
    write_raw_header(leading_space, b' {"weights":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}', b'\0' * 4)

    trailing_newline = root / 'trailing-newline.safetensors'
    write_raw_header(trailing_newline, b'{"weights":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}\n', b'\0' * 4)

    current_dtype = root / 'current-dtype.safetensors'
    write_file(current_dtype, {
        'scale': {'dtype': 'F8_E8M0', 'shape': [1], 'data_offsets': [0, 1]},
    }, b'\0')
    current_dtype_reader = SafeTensorFile(current_dtype)
    current_dtype_reader.close()

    print(json.dumps({
        'valid_values': valid_values,
        'undersized': rejected(undersized),
        'oversized': rejected(oversized),
        'overlapping': rejected(overlapping),
        'non_object': rejected(non_object),
        'gap': rejected(gap),
        'trailing': rejected(trailing),
        'invalid_metadata': rejected(invalid_metadata),
        'duplicate': rejected(duplicate),
        'leading_space': rejected(leading_space),
        'trailing_newline': rejected(trailing_newline),
        'current_dtype_header': True,
    }))
`], {
  cwd: ROOT,
  encoding: 'utf8',
  env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', PYTHONIOENCODING: 'utf-8' },
}));

test('SafeTensorFile reads a valid exact tensor span', () => {
  assert.deepEqual(results.valid_values, [1.5, 2.5]);
});

test('SafeTensorFile rejects tensor spans that disagree with dtype and shape', () => {
  assert.equal(results.undersized, true);
  assert.equal(results.oversized, true);
});

test('SafeTensorFile rejects overlapping spans and non-object headers', () => {
  assert.equal(results.overlapping, true);
  assert.equal(results.non_object, true);
});

test('SafeTensorFile rejects gaps, trailing polyglot bytes, duplicate keys, and invalid metadata', () => {
  assert.equal(results.gap, true);
  assert.equal(results.trailing, true);
  assert.equal(results.invalid_metadata, true);
  assert.equal(results.duplicate, true);
  assert.equal(results.leading_space, true);
  assert.equal(results.trailing_newline, true);
});

test('SafeTensorFile recognizes current full-byte safetensors dtypes', () => {
  assert.equal(results.current_dtype_header, true);
});
