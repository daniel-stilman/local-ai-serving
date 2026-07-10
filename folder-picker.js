'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { TextDecoder } = require('node:util');

const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_TIMEOUT_MS = 300_000;
const MAX_OUTPUT_BYTES = 16 * 1024;

const WINDOWS_SCRIPT = [
  'Add-Type -AssemblyName System.Windows.Forms',
  '$owner = New-Object System.Windows.Forms.Form',
  '$owner.ShowInTaskbar = $false',
  '$owner.TopMost = $true',
  "$owner.Text = 'Local Chat Folder Picker'",
  '$owner.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedToolWindow',
  '$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen',
  '$owner.Size = New-Object System.Drawing.Size(1, 1)',
  '$owner.Opacity = 0.01',
  '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog',
  "$dialog.Description = 'Select a folder'",
  '$dialog.ShowNewFolderButton = $true',
  'try {',
  '  $owner.Show()',
  '  $owner.Activate()',
  '  if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {',
  '    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)',
  '    [Console]::Out.Write($dialog.SelectedPath)',
  '  }',
  '} finally {',
  '  $dialog.Dispose()',
  '  $owner.Close()',
  '  $owner.Dispose()',
  '}',
].join('; ');

const MACOS_SCRIPT = 'POSIX path of (choose folder with prompt "Select a folder")';

/**
 * Build a folder picker whose process and filesystem boundaries can be replaced
 * in tests. The returned function resolves to an absolute directory path or
 * null when the user cancels, no picker is available, or validation fails.
 */
function createFolderPicker(dependencies = {}) {
  const platform = dependencies.platform || process.platform;
  const spawnProcess = dependencies.spawn || spawn;
  const statPath = dependencies.stat || ((selectedPath) => fs.promises.stat(selectedPath));
  const environment = dependencies.environment || process.env;
  const isExecutableFile = dependencies.isExecutableFile || defaultIsExecutableFile;
  const scheduleTimeout = dependencies.setTimeout || setTimeout;
  const cancelTimeout = dependencies.clearTimeout || clearTimeout;
  const pathApi = platform === 'win32' ? path.win32 : path.posix;

  return async function pickFolder(options = {}) {
    const signal = options.signal;
    if (isAborted(signal)) return null;

    const timeoutMs = boundedInteger(options.timeoutMs, DEFAULT_TIMEOUT_MS, 1, MAX_TIMEOUT_MS);
    const outputLimit = boundedInteger(options.maxOutputBytes, MAX_OUTPUT_BYTES, 1, MAX_OUTPUT_BYTES);
    const candidates = candidatesForPlatform(platform, { environment, isExecutableFile });

    for (const candidate of candidates) {
      if (isAborted(signal)) return null;
      const result = await runCandidate(candidate, {
        spawnProcess,
        scheduleTimeout,
        cancelTimeout,
        timeoutMs,
        outputLimit,
        signal,
      });

      if (isAborted(signal)) return null;
      if (result.state === 'unavailable') continue;
      if (result.state !== 'selected') return null;

      const selectedPath = result.value;
      if (!selectedPath || selectedPath.includes('\0') || !pathApi.isAbsolute(selectedPath)) return null;

      try {
        const stats = await statPath(selectedPath);
        return !isAborted(signal)
          && stats && typeof stats.isDirectory === 'function' && stats.isDirectory()
          ? selectedPath
          : null;
      } catch {
        return null;
      }
    }

    return null;
  };
}

function candidatesForPlatform(platform, settings) {
  if (platform === 'win32') {
    const command = resolveWindowsPowerShell(settings.environment, settings.isExecutableFile);
    if (!command) return [];
    return [{
      command,
      args: ['-NoLogo', '-NoProfile', '-NonInteractive', '-STA', '-Command', WINDOWS_SCRIPT],
    }];
  }

  if (platform === 'darwin') {
    return [{
      command: '/usr/bin/osascript',
      args: ['-e', MACOS_SCRIPT],
    }];
  }

  if (platform === 'linux') {
    return [
      {
        command: '/usr/bin/zenity',
        args: ['--file-selection', '--directory', '--title=Select a folder'],
      },
      {
        command: '/usr/bin/kdialog',
        args: ['--getexistingdirectory'],
      },
    ];
  }

  return [];
}

function resolveWindowsPowerShell(environment, isExecutableFile) {
  const rawSystemRoot = String(environment?.SystemRoot || environment?.SYSTEMROOT || '');
  if (!rawSystemRoot || rawSystemRoot.includes('\0') || !path.win32.isAbsolute(rawSystemRoot)) return '';

  const systemRoot = path.win32.normalize(rawSystemRoot);
  if (!/^[A-Za-z]:\\/.test(systemRoot)) return '';
  const command = path.win32.join(
    systemRoot,
    'System32',
    'WindowsPowerShell',
    'v1.0',
    'powershell.exe',
  );
  const relative = path.win32.relative(systemRoot, command);
  if (!relative || relative.startsWith('..') || path.win32.isAbsolute(relative)) return '';

  try {
    return isExecutableFile(command) ? command : '';
  } catch {
    return '';
  }
}

function runCandidate(candidate, settings) {
  return new Promise((resolve) => {
    if (isAborted(settings.signal)) {
      resolve({ state: 'cancelled' });
      return;
    }

    let child;
    try {
      child = settings.spawnProcess(candidate.command, candidate.args.slice(), {
        shell: false,
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
    } catch {
      resolve({ state: 'unavailable' });
      return;
    }

    if (!child || typeof child.once !== 'function' || !child.stdout
        || typeof child.stdout.on !== 'function') {
      safeKill(child);
      resolve({ state: 'unavailable' });
      return;
    }

    let settled = false;
    let timer;
    let abortHandler;
    let byteLength = 0;
    const chunks = [];

    const finish = (result) => {
      if (settled) return;
      settled = true;
      if (timer !== undefined) settings.cancelTimeout(timer);
      if (abortHandler && typeof settings.signal?.removeEventListener === 'function') {
        try { settings.signal.removeEventListener('abort', abortHandler); } catch {}
      }
      resolve(result);
    };

    child.stdout.on('data', (chunk) => {
      if (settled) return;

      let bytes;
      try {
        bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      } catch {
        safeKill(child);
        finish({ state: 'invalid' });
        return;
      }

      byteLength += bytes.length;
      if (byteLength > settings.outputLimit) {
        safeKill(child);
        finish({ state: 'invalid' });
        return;
      }
      chunks.push(bytes);
    });

    child.stdout.on('error', () => {
      safeKill(child);
      finish({ state: 'invalid' });
    });

    child.once('error', () => finish({ state: 'unavailable' }));
    child.once('close', (code) => {
      if (settled) return;
      if (code !== 0) {
        finish({ state: 'cancelled' });
        return;
      }

      let value;
      try {
        value = new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks, byteLength));
      } catch {
        finish({ state: 'invalid' });
        return;
      }

      value = removeProtocolLineBreak(value);
      finish(value ? { state: 'selected', value } : { state: 'cancelled' });
    });

    if (settings.signal && typeof settings.signal.addEventListener === 'function') {
      abortHandler = () => {
        safeKill(child);
        finish({ state: 'cancelled' });
      };
      try {
        settings.signal.addEventListener('abort', abortHandler, { once: true });
      } catch {
        safeKill(child);
        finish({ state: 'cancelled' });
        return;
      }
      if (isAborted(settings.signal)) abortHandler();
      if (settled) return;
    }

    timer = settings.scheduleTimeout(() => {
      safeKill(child);
      finish({ state: 'cancelled' });
    }, settings.timeoutMs);
    if (timer && typeof timer.unref === 'function') timer.unref();
  });
}

function removeProtocolLineBreak(value) {
  if (value.endsWith('\r\n')) return value.slice(0, -2);
  if (value.endsWith('\n') || value.endsWith('\r')) return value.slice(0, -1);
  return value;
}

function safeKill(child) {
  if (!child || typeof child.kill !== 'function') return;
  try {
    child.kill();
  } catch {
    // The picker may already have exited. No diagnostics should escape here.
  }
}

function defaultIsExecutableFile(candidate) {
  try {
    return fs.statSync(candidate).isFile();
  } catch {
    return false;
  }
}

function isAborted(signal) {
  return Boolean(signal && signal.aborted);
}

function boundedInteger(value, fallback, minimum, maximum) {
  return Number.isSafeInteger(value)
    ? Math.min(maximum, Math.max(minimum, value))
    : fallback;
}

const pickFolder = createFolderPicker();

module.exports = {
  createFolderPicker,
  pickFolder,
};
