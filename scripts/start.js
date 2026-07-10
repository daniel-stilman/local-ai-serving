'use strict';

const path = require('node:path');
const { spawn } = require('child_process');

const root = path.resolve(__dirname, '..');
const PORT_ERROR = 'PORT must be an integer from 1 through 65535.';

function parsePort(value) {
  const candidate = value === undefined ? '3000' : String(value);
  if (!/^[0-9]+$/.test(candidate)) throw new Error(PORT_ERROR);
  const port = Number(candidate);
  if (!Number.isSafeInteger(port) || port < 1 || port > 65535) throw new Error(PORT_ERROR);
  return port;
}

function main(environment = process.env) {
  const port = parsePort(environment.PORT);
  const httpsEnabled = !['0', 'false', 'off', 'no'].includes(String(environment.HTTPS ?? '1').toLowerCase());
  const protocol = httpsEnabled ? 'https' : 'http';
  const dashboardUrl = `${protocol}://localhost:${port}/dashboard`;
  const server = spawn(process.execPath, [path.join(root, 'server.js')], {
    cwd: root,
    env: { ...environment, PORT: String(port) },
    stdio: 'inherit',
  });

  let dashboardOpened = false;
  const openTimer = setTimeout(() => {
    dashboardOpened = true;
    openUrl(dashboardUrl);
  }, 900);

  server.on('exit', (code, signal) => {
    clearTimeout(openTimer);
    if (!dashboardOpened && code !== 0) {
      console.error('Server exited before the dashboard could open.');
    }
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code ?? 0);
    }
  });

  process.on('SIGINT', () => {
    server.kill('SIGINT');
  });

  process.on('SIGTERM', () => {
    server.kill('SIGTERM');
  });
}

function openUrl(url, platform = process.platform, spawnProcess = spawn) {
  if (platform === 'win32') {
    spawnProcess('explorer.exe', [url], {
      detached: true,
      shell: false,
      stdio: 'ignore',
      windowsHide: true,
    }).unref();
    return;
  }

  if (platform === 'darwin') {
    spawnProcess('open', [url], { detached: true, shell: false, stdio: 'ignore' }).unref();
    return;
  }

  spawnProcess('xdg-open', [url], { detached: true, shell: false, stdio: 'ignore' }).unref();
}

if (require.main === module) main();

module.exports = { main, openUrl, parsePort };
