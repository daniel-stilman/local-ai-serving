'use strict';

const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const path = require('node:path');
const test = require('node:test');

const { createFolderPicker } = require('../folder-picker');

const WINDOWS_SYSTEM_ROOT = 'C:\\Windows';
const WINDOWS_POWERSHELL = 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe';

test('Windows uses FolderBrowserDialog with hardened spawn options', async () => {
  const calls = [];
  const selectedPath = 'C:\\folder';
  const pickFolder = createFolderPicker({
    platform: 'win32',
    environment: { SystemRoot: WINDOWS_SYSTEM_ROOT },
    isExecutableFile: () => true,
    spawn: fakeSpawn([{ stdout: selectedPath, code: 0 }], calls),
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), selectedPath);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, WINDOWS_POWERSHELL);
  assert.equal(path.win32.isAbsolute(calls[0].command), true);
  assert.equal(path.win32.relative(WINDOWS_SYSTEM_ROOT, calls[0].command).startsWith('..'), false);
  assert.deepEqual(calls[0].options, {
    shell: false,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  assert.ok(calls[0].args.includes('-NoProfile'));
  assert.ok(calls[0].args.includes('-NonInteractive'));
  assert.ok(calls[0].args.includes('-STA'));
  assert.match(calls[0].args.at(-1), /FolderBrowserDialog/);
  assert.match(calls[0].args.at(-1), /TopMost = \$true/);
  assert.match(calls[0].args.at(-1), /ShowDialog\(\$owner\)/);
  assert.equal(calls[0].args.join(' ').includes(selectedPath), false);
});

test('Windows refuses an unvalidated system PowerShell location', async (t) => {
  await t.test('relative SystemRoot', async () => {
    let spawned = false;
    const pickFolder = createFolderPicker({
      platform: 'win32',
      environment: { SystemRoot: 'relative-system-root' },
      isExecutableFile: () => true,
      spawn() {
        spawned = true;
        throw new Error('must not run');
      },
      stat: directoryStat,
    });

    assert.equal(await pickFolder(), null);
    assert.equal(spawned, false);
  });

  await t.test('missing executable', async () => {
    let spawned = false;
    const pickFolder = createFolderPicker({
      platform: 'win32',
      environment: { SystemRoot: WINDOWS_SYSTEM_ROOT },
      isExecutableFile: () => false,
      spawn() {
        spawned = true;
        throw new Error('must not run');
      },
      stat: directoryStat,
    });

    assert.equal(await pickFolder(), null);
    assert.equal(spawned, false);
  });
});

test('macOS uses osascript and preserves an absolute selected directory', async () => {
  const calls = [];
  const selectedPath = '/selected/folder/';
  const pickFolder = createFolderPicker({
    platform: 'darwin',
    spawn: fakeSpawn([{ stdout: `${selectedPath}\n`, code: 0 }], calls),
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), selectedPath);
  assert.equal(calls[0].command, '/usr/bin/osascript');
  assert.deepEqual(calls[0].args.slice(0, 1), ['-e']);
  assert.equal(calls[0].options.shell, false);
});

test('Linux falls back from an unavailable zenity process to kdialog', async () => {
  const calls = [];
  const pickFolder = createFolderPicker({
    platform: 'linux',
    spawn: fakeSpawn([
      { error: makeError('ENOENT') },
      { stdout: '/selected/folder\n', code: 0 },
    ], calls),
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), '/selected/folder');
  assert.deepEqual(calls.map((call) => call.command), ['/usr/bin/zenity', '/usr/bin/kdialog']);
  assert.ok(calls.every((call) => call.options.shell === false && call.options.windowsHide === true));
});

test('cancellation does not open a second Linux picker', async () => {
  const calls = [];
  const pickFolder = createFolderPicker({
    platform: 'linux',
    spawn: fakeSpawn([{ code: 1 }], calls),
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), null);
  assert.equal(calls.length, 1);
});

test('an empty successful response is cancellation', async () => {
  const pickFolder = createFolderPicker({
    platform: 'darwin',
    spawn: fakeSpawn([{ stdout: '', code: 0 }]),
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), null);
});

test('unsupported hosts cancel without spawning anything', async () => {
  let spawned = false;
  const pickFolder = createFolderPicker({
    platform: 'unsupported',
    spawn() {
      spawned = true;
      throw new Error('must not run');
    },
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), null);
  assert.equal(spawned, false);
});

test('only absolute paths to existing directories are returned', async (t) => {
  await t.test('relative paths are rejected before filesystem access', async () => {
    let statCalled = false;
    const pickFolder = createFolderPicker({
      platform: 'linux',
      spawn: fakeSpawn([{ stdout: 'relative/folder\n', code: 0 }]),
      stat: async () => {
        statCalled = true;
        return { isDirectory: () => true };
      },
    });

    assert.equal(await pickFolder(), null);
    assert.equal(statCalled, false);
  });

  await t.test('missing paths are rejected without exposing filesystem diagnostics', async () => {
    const pickFolder = createFolderPicker({
      platform: 'linux',
      spawn: fakeSpawn([{ stdout: '/missing\n', code: 0 }]),
      stat: async () => {
        throw new Error('private filesystem diagnostic');
      },
    });

    assert.equal(await pickFolder(), null);
  });

  await t.test('regular files are rejected', async () => {
    const pickFolder = createFolderPicker({
      platform: 'linux',
      spawn: fakeSpawn([{ stdout: '/selected/file\n', code: 0 }]),
      stat: async () => ({ isDirectory: () => false }),
    });

    assert.equal(await pickFolder(), null);
  });
});

test('invalid UTF-8 and NUL-containing results are rejected', async (t) => {
  await t.test('invalid UTF-8', async () => {
    const pickFolder = createFolderPicker({
      platform: 'linux',
      spawn: fakeSpawn([{ stdout: Buffer.from([0xff]), code: 0 }]),
      stat: directoryStat,
    });
    assert.equal(await pickFolder(), null);
  });

  await t.test('NUL byte', async () => {
    const pickFolder = createFolderPicker({
      platform: 'linux',
      spawn: fakeSpawn([{ stdout: '/selected\0suffix', code: 0 }]),
      stat: directoryStat,
    });
    assert.equal(await pickFolder(), null);
  });
});

test('output is byte-bounded and an overflowing picker is killed', async () => {
  const children = [];
  let statCalled = false;
  const pickFolder = createFolderPicker({
    platform: 'linux',
    spawn: fakeSpawn([{ stdout: Buffer.alloc(9, 65), code: 0 }], undefined, children),
    stat: async () => {
      statCalled = true;
      return { isDirectory: () => true };
    },
  });

  assert.equal(await pickFolder({ maxOutputBytes: 8 }), null);
  assert.equal(children[0].killCalls, 1);
  assert.equal(statCalled, false);
});

test('the hard output ceiling cannot be raised by caller options', async () => {
  const children = [];
  const pickFolder = createFolderPicker({
    platform: 'linux',
    spawn: fakeSpawn([{ stdout: Buffer.alloc((16 * 1024) + 1, 65), code: 0 }], undefined, children),
    stat: directoryStat,
  });

  assert.equal(await pickFolder({ maxOutputBytes: Number.MAX_SAFE_INTEGER }), null);
  assert.equal(children[0].killCalls, 1);
});

test('stdout stream failures are contained and the picker is killed', async () => {
  const children = [];
  const pickFolder = createFolderPicker({
    platform: 'win32',
    environment: { SystemRoot: WINDOWS_SYSTEM_ROOT },
    isExecutableFile: () => true,
    spawn: fakeSpawn([{ stdoutError: new Error('private stream diagnostic') }], undefined, children),
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), null);
  assert.equal(children[0].killCalls, 1);
});

test('timeout is bounded, kills the picker, and resolves without diagnostics', async () => {
  const children = [];
  const timers = [];
  const pickFolder = createFolderPicker({
    platform: 'win32',
    environment: { SystemRoot: WINDOWS_SYSTEM_ROOT },
    isExecutableFile: () => true,
    spawn: fakeSpawn([{ hang: true }], undefined, children),
    stat: directoryStat,
    setTimeout(callback, delay) {
      timers.push({ callback, delay, cleared: false });
      queueMicrotask(callback);
      return timers.at(-1);
    },
    clearTimeout(timer) {
      timer.cleared = true;
    },
  });

  assert.equal(await pickFolder({ timeoutMs: Number.MAX_SAFE_INTEGER }), null);
  assert.equal(timers[0].delay, 300_000);
  assert.equal(timers[0].cleared, true);
  assert.equal(children[0].killCalls, 1);
});

test('a pre-aborted signal resolves cancellation without filesystem or process access', async () => {
  const controller = new AbortController();
  controller.abort();
  let executableChecked = false;
  let spawned = false;
  const pickFolder = createFolderPicker({
    platform: 'win32',
    environment: { SystemRoot: WINDOWS_SYSTEM_ROOT },
    isExecutableFile() {
      executableChecked = true;
      return true;
    },
    spawn() {
      spawned = true;
      throw new Error('must not run');
    },
    stat: directoryStat,
  });

  assert.equal(await pickFolder({ signal: controller.signal }), null);
  assert.equal(executableChecked, false);
  assert.equal(spawned, false);
});

test('aborting an active picker kills it and resolves cancellation promptly', async () => {
  const controller = new AbortController();
  const children = [];
  const clearedTimers = [];
  const pickFolder = createFolderPicker({
    platform: 'win32',
    environment: { SystemRoot: WINDOWS_SYSTEM_ROOT },
    isExecutableFile: () => true,
    spawn: fakeSpawn([{ hang: true }], undefined, children),
    stat: directoryStat,
    setTimeout() {
      return { unref() {} };
    },
    clearTimeout(timer) {
      clearedTimers.push(timer);
    },
  });

  const result = pickFolder({ signal: controller.signal });
  assert.equal(children.length, 1);
  controller.abort();

  assert.equal(await result, null);
  assert.equal(children[0].killCalls, 1);
  assert.equal(clearedTimers.length, 1);
});

test('spawn failures remain private and do not reject the caller', async () => {
  const sensitive = 'sensitive command and path diagnostic';
  const calls = [];
  const pickFolder = createFolderPicker({
    platform: 'linux',
    spawn: fakeSpawn([
      { throw: new Error(sensitive) },
      { error: new Error(sensitive) },
    ], calls),
    stat: directoryStat,
  });

  assert.equal(await pickFolder(), null);
  assert.equal(calls.length, 2);
});

async function directoryStat() {
  return { isDirectory: () => true };
}

function fakeSpawn(scenarios, calls = [], children = []) {
  let index = 0;
  return (command, args, options) => {
    const scenario = scenarios[index++] || { code: 1 };
    calls.push({ command, args, options });
    if (scenario.throw) throw scenario.throw;

    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.killCalls = 0;
    child.kill = () => {
      child.killCalls += 1;
      return true;
    };
    children.push(child);

    queueMicrotask(() => {
      if (scenario.hang) return;
      if (scenario.stdoutError) {
        child.stdout.emit('error', scenario.stdoutError);
        return;
      }
      if (scenario.stdout !== undefined) child.stdout.emit('data', scenario.stdout);
      if (scenario.error) child.emit('error', scenario.error);
      else child.emit('close', scenario.code ?? 0);
    });

    return child;
  };
}

function makeError(code) {
  const error = new Error('unavailable');
  error.code = code;
  return error;
}
