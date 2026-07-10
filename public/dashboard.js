'use strict';

const els = {
  qrCode: document.getElementById('qrCode'),
  urlSelect: document.getElementById('urlSelect'),
  accessUrl: document.getElementById('accessUrl'),
  copyButton: document.getElementById('copyButton'),
  refreshButton: document.getElementById('refreshButton'),
  securityMode: document.getElementById('securityMode'),
  tokenInfo: document.getElementById('tokenInfo'),
  certificateInfo: document.getElementById('certificateInfo'),
  modelSetupBadge: document.getElementById('modelSetupBadge'),
  textRuntimeStatus: document.getElementById('textRuntimeStatus'),
  textFolderStatus: document.getElementById('textFolderStatus'),
  textModelCount: document.getElementById('textModelCount'),
  textModelsRootInput: document.getElementById('textModelsRootInput'),
  pickTextModelsRootButton: document.getElementById('pickTextModelsRootButton'),
  saveTextModelsRootButton: document.getElementById('saveTextModelsRootButton'),
  refreshModelSetupButton: document.getElementById('refreshModelSetupButton'),
  modelSetupMessage: document.getElementById('modelSetupMessage'),
  textModelList: document.getElementById('textModelList'),
};

let accessUrls = [];
let textFolderLocked = false;

initialize();

function initialize() {
  els.urlSelect.addEventListener('change', renderSelectedUrl);
  els.refreshButton.addEventListener('click', loadAccessInfo);
  els.copyButton.addEventListener('click', async () => {
    await navigator.clipboard.writeText(els.accessUrl.value);
    els.copyButton.textContent = 'Copied';
    setTimeout(() => {
      els.copyButton.textContent = 'Copy Link';
    }, 900);
  });
  els.pickTextModelsRootButton.addEventListener('click', pickTextModelsRoot);
  els.saveTextModelsRootButton.addEventListener('click', saveTextModelsRoot);
  els.refreshModelSetupButton.addEventListener('click', refreshLocalSetup);
  loadAccessInfo();
  loadLocalSetup();
}

async function loadLocalSetup() {
  setSetupBusy(true);
  try {
    const response = await fetch('/api/local-setup', { cache: 'no-store' });
    const data = await readJson(response);
    if (!response.ok) throw new Error(data.error || 'Local model setup is unavailable.');
    const locked = renderLocalSetup(data);
    if (!locked) setSetupMessage('');
  } catch (error) {
    renderLocalSetup({ text: {} });
    setSetupMessage(error.message || 'Local model setup is unavailable.', true);
  } finally {
    setSetupBusy(false);
  }
}

async function pickTextModelsRoot() {
  setSetupBusy(true);
  setSetupMessage('Opening the folder chooser on this computer...');
  try {
    const response = await fetch('/api/local-setup/pick-folder', {
      method: 'POST',
      headers: { 'X-Local-Setup': '1' },
    });
    const data = await readJson(response);
    if (!response.ok) throw new Error(data.error || 'The folder chooser could not open.');
    if (data.cancelled) {
      setSetupMessage('Folder selection cancelled.');
      return;
    }
    els.textModelsRootInput.value = data.path || '';
    setSetupMessage(data.path ? 'Folder selected. Choose Use This Folder to scan it.' : 'No folder was selected.');
  } catch (error) {
    setSetupMessage(error.message || 'The folder chooser could not open.', true);
  } finally {
    setSetupBusy(false);
  }
}

async function saveTextModelsRoot() {
  const selectedPath = els.textModelsRootInput.value.trim();
  if (!selectedPath) {
    setSetupMessage('Choose or paste a model folder first.', true);
    return;
  }
  setSetupBusy(true);
  setSetupMessage('Validating and scanning the selected folder...');
  try {
    const response = await fetch('/api/local-setup/text-folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Local-Setup': '1' },
      body: JSON.stringify({ path: selectedPath }),
    });
    const data = await readJson(response);
    if (!response.ok) throw new Error(data.error || 'The model folder could not be saved.');
    els.textModelsRootInput.value = '';
    renderLocalSetup(data);
    const text = data.text || {};
    const count = Number(text.modelCount) || 0;
    setSetupMessage(text.managedEnabled
      ? `${count} selectable model${count === 1 ? '' : 's'} ready. Open Chat to choose one.`
      : `${count} compatible model${count === 1 ? '' : 's'} found. Configure the text runtime before opening chat.`,
    );
  } catch (error) {
    setSetupMessage(error.message || 'The model folder could not be saved.', true);
  } finally {
    setSetupBusy(false);
  }
}

async function refreshLocalSetup() {
  setSetupBusy(true);
  setSetupMessage('Rescanning the configured model location...');
  try {
    const response = await fetch('/api/local-setup/refresh-text-models', {
      method: 'POST',
      headers: { 'X-Local-Setup': '1' },
    });
    const data = await readJson(response);
    if (!response.ok) throw new Error(data.error || 'The model location could not be rescanned.');
    const locked = renderLocalSetup(data);
    const text = data.text || {};
    const count = Number(text.modelCount) || 0;
    if (!locked) setSetupMessage(`${count} model${count === 1 ? '' : 's'} found in the latest scan.`);
  } catch (error) {
    setSetupMessage(error.message || 'The model location could not be rescanned.', true);
  } finally {
    setSetupBusy(false);
  }
}

function renderLocalSetup(data) {
  const text = data && data.text || {};
  const models = Array.isArray(text.models) ? text.models : [];
  els.textRuntimeStatus.textContent = text.runtimeConfigured ? 'Configured' : 'Not configured';
  els.textFolderStatus.textContent = text.folderConfigured ? 'Configured locally' : 'Not selected';
  els.textModelCount.textContent = String(Number(text.modelCount) || 0);
  els.modelSetupBadge.textContent = text.managedEnabled && models.length ? 'Ready' : 'Setup';
  els.modelSetupBadge.classList.toggle('ready', Boolean(text.managedEnabled && models.length));
  els.textModelList.replaceChildren();
  if (!models.length) {
    const item = document.createElement('li');
    item.textContent = text.runtimeConfigured
      ? 'No compatible GGUF models found in the selected folder.'
      : 'Configure the text runtime, then choose a GGUF model folder.';
    els.textModelList.append(item);
  } else {
    for (const model of models) {
      const item = document.createElement('li');
      item.textContent = String(model);
      els.textModelList.append(item);
    }
  }
  const locked = Boolean(text.folderLockedByEnvironment);
  textFolderLocked = locked;
  els.textModelsRootInput.disabled = locked;
  els.pickTextModelsRootButton.disabled = locked;
  els.saveTextModelsRootButton.disabled = locked;
  if (locked) setSetupMessage('The model folder is controlled by an environment override and cannot be changed here.', true);
  return locked;
}

function setSetupBusy(busy) {
  els.textModelsRootInput.disabled = busy || textFolderLocked;
  els.pickTextModelsRootButton.disabled = busy || textFolderLocked;
  els.saveTextModelsRootButton.disabled = busy || textFolderLocked;
  els.refreshModelSetupButton.disabled = busy;
}

function setSetupMessage(message, isError = false) {
  els.modelSetupMessage.textContent = message;
  els.modelSetupMessage.classList.toggle('error', isError);
}

async function readJson(response) {
  try { return await response.json(); } catch { return {}; }
}

async function loadAccessInfo() {
  const response = await fetch('/api/access-info', { cache: 'no-store' });
  if (!response.ok) {
    els.qrCode.textContent = 'Dashboard unavailable';
    return;
  }

  const data = await response.json();
  accessUrls = data.accessUrls || [];
  els.securityMode.textContent = data.httpsEnabled ? 'HTTPS' : 'HTTP';
  els.tokenInfo.textContent = `${data.tokenLength} chars, stored only in this server process`;
  els.certificateInfo.textContent = data.httpsEnabled
    ? `${data.certificateFingerprint.slice(0, 23)}...`
    : 'Not enabled';

  els.urlSelect.replaceChildren();
  for (const url of accessUrls) {
    const option = document.createElement('option');
    option.value = url;
    option.textContent = url.replace(/#access=.*/, '#access=...');
    els.urlSelect.append(option);
  }

  renderSelectedUrl();
}

function renderSelectedUrl() {
  const url = els.urlSelect.value || accessUrls[0] || '';
  els.accessUrl.value = url;
  els.qrCode.replaceChildren();

  if (!url) {
    els.qrCode.textContent = 'No LAN address found';
    return;
  }

  els.qrCode.append(renderQr(url));
}

function renderQr(text) {
  const modules = makeQrMatrix(text);
  const quiet = 4;
  const size = modules.length + quiet * 2;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('role', 'img');

  const background = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  background.setAttribute('width', String(size));
  background.setAttribute('height', String(size));
  background.setAttribute('fill', '#ffffff');
  svg.append(background);

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  const commands = [];
  for (let y = 0; y < modules.length; y += 1) {
    for (let x = 0; x < modules.length; x += 1) {
      if (modules[y][x]) {
        commands.push(`M${x + quiet},${y + quiet}h1v1h-1z`);
      }
    }
  }
  path.setAttribute('d', commands.join(''));
  path.setAttribute('fill', '#050512');
  svg.append(path);
  return svg;
}

function makeQrMatrix(text) {
  const version = 5;
  const size = 17 + version * 4;
  const dataCodewords = 108;
  const eccCodewords = 26;
  const modules = Array.from({ length: size }, () => Array(size).fill(null));
  const reserved = Array.from({ length: size }, () => Array(size).fill(false));

  drawFunctionPatterns(modules, reserved, version);
  const data = encodeQrData(text, dataCodewords);
  const ecc = reedSolomon(data, eccCodewords);
  drawCodewords(modules, reserved, data.concat(ecc));
  applyBestMask(modules, reserved);
  return modules;
}

function encodeQrData(text, dataCodewords) {
  if (!/^[\x00-\x7F]*$/.test(text)) {
    throw new Error('QR content must be ASCII');
  }
  const bits = [];
  appendBits(bits, 0b0100, 4);
  appendBits(bits, text.length, 8);
  for (let index = 0; index < text.length; index += 1) {
    appendBits(bits, text.charCodeAt(index), 8);
  }
  appendBits(bits, 0, Math.min(4, dataCodewords * 8 - bits.length));
  while (bits.length % 8) bits.push(0);

  const data = [];
  for (let index = 0; index < bits.length; index += 8) {
    data.push(bitsToByte(bits.slice(index, index + 8)));
  }
  for (let pad = 0; data.length < dataCodewords; pad += 1) {
    data.push(pad % 2 === 0 ? 0xec : 0x11);
  }
  if (data.length > dataCodewords) {
    throw new Error('Access URL is too long for the built-in QR encoder');
  }
  return data;
}

function drawFunctionPatterns(modules, reserved, version) {
  const size = modules.length;
  drawFinder(modules, reserved, 0, 0);
  drawFinder(modules, reserved, size - 7, 0);
  drawFinder(modules, reserved, 0, size - 7);
  drawAlignment(modules, reserved, 30, 30);

  for (let i = 8; i < size - 8; i += 1) {
    setFunctionModule(modules, reserved, i, 6, i % 2 === 0);
    setFunctionModule(modules, reserved, 6, i, i % 2 === 0);
  }

  setFunctionModule(modules, reserved, 8, 4 * version + 9, true);
  reserveFormatAreas(reserved);
}

function drawFinder(modules, reserved, left, top) {
  for (let y = -1; y <= 7; y += 1) {
    for (let x = -1; x <= 7; x += 1) {
      const xx = left + x;
      const yy = top + y;
      if (!isInside(modules, xx, yy)) continue;
      const dark =
        x >= 0 &&
        x <= 6 &&
        y >= 0 &&
        y <= 6 &&
        (x === 0 || x === 6 || y === 0 || y === 6 || (x >= 2 && x <= 4 && y >= 2 && y <= 4));
      setFunctionModule(modules, reserved, xx, yy, dark);
    }
  }
}

function drawAlignment(modules, reserved, centerX, centerY) {
  for (let y = -2; y <= 2; y += 1) {
    for (let x = -2; x <= 2; x += 1) {
      setFunctionModule(modules, reserved, centerX + x, centerY + y, Math.max(Math.abs(x), Math.abs(y)) !== 1);
    }
  }
}

function reserveFormatAreas(reserved) {
  const size = reserved.length;
  for (let i = 0; i <= 8; i += 1) {
    reserved[8][i] = true;
    reserved[i][8] = true;
  }
  for (let i = 0; i < 8; i += 1) {
    reserved[8][size - 1 - i] = true;
    reserved[size - 1 - i][8] = true;
  }
}

function drawCodewords(modules, reserved, codewords) {
  const bits = [];
  for (const codeword of codewords) {
    appendBits(bits, codeword, 8);
  }

  const size = modules.length;
  let bitIndex = 0;
  let upward = true;
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right -= 1;
    for (let vertical = 0; vertical < size; vertical += 1) {
      const y = upward ? size - 1 - vertical : vertical;
      for (let column = 0; column < 2; column += 1) {
        const x = right - column;
        if (reserved[y][x]) continue;
        modules[y][x] = bitIndex < bits.length ? bits[bitIndex] === 1 : false;
        bitIndex += 1;
      }
    }
    upward = !upward;
  }
}

function applyBestMask(modules, reserved) {
  let bestMask = 0;
  let bestPenalty = Infinity;
  let bestModules = null;

  for (let mask = 0; mask < 8; mask += 1) {
    const candidate = modules.map((row) => row.slice());
    applyMask(candidate, reserved, mask);
    drawFormatBits(candidate, reserved, mask);
    const penalty = scoreQr(candidate);
    if (penalty < bestPenalty) {
      bestMask = mask;
      bestPenalty = penalty;
      bestModules = candidate;
    }
  }

  for (let y = 0; y < modules.length; y += 1) {
    for (let x = 0; x < modules.length; x += 1) {
      modules[y][x] = bestModules[y][x];
    }
  }
  drawFormatBits(modules, reserved, bestMask);
}

function applyMask(modules, reserved, mask) {
  for (let y = 0; y < modules.length; y += 1) {
    for (let x = 0; x < modules.length; x += 1) {
      if (reserved[y][x]) continue;
      if (maskCondition(mask, x, y)) {
        modules[y][x] = !modules[y][x];
      }
    }
  }
}

function maskCondition(mask, x, y) {
  if (mask === 0) return (x + y) % 2 === 0;
  if (mask === 1) return y % 2 === 0;
  if (mask === 2) return x % 3 === 0;
  if (mask === 3) return (x + y) % 3 === 0;
  if (mask === 4) return (Math.floor(y / 2) + Math.floor(x / 3)) % 2 === 0;
  if (mask === 5) return ((x * y) % 2) + ((x * y) % 3) === 0;
  if (mask === 6) return (((x * y) % 2) + ((x * y) % 3)) % 2 === 0;
  return (((x + y) % 2) + ((x * y) % 3)) % 2 === 0;
}

function drawFormatBits(modules, reserved, mask) {
  const bits = formatBits(mask);
  const size = modules.length;

  for (let i = 0; i <= 5; i += 1) setFunctionModule(modules, reserved, 8, i, bit(bits, i));
  setFunctionModule(modules, reserved, 8, 7, bit(bits, 6));
  setFunctionModule(modules, reserved, 8, 8, bit(bits, 7));
  setFunctionModule(modules, reserved, 7, 8, bit(bits, 8));
  for (let i = 9; i < 15; i += 1) setFunctionModule(modules, reserved, 14 - i, 8, bit(bits, i));

  for (let i = 0; i < 8; i += 1) setFunctionModule(modules, reserved, size - 1 - i, 8, bit(bits, i));
  for (let i = 8; i < 15; i += 1) setFunctionModule(modules, reserved, 8, size - 15 + i, bit(bits, i));
}

function formatBits(mask) {
  let data = (1 << 3) | mask;
  let value = data << 10;
  const generator = 0x537;
  for (let i = 14; i >= 10; i -= 1) {
    if (((value >> i) & 1) !== 0) {
      value ^= generator << (i - 10);
    }
  }
  return ((data << 10) | value) ^ 0x5412;
}

function scoreQr(modules) {
  const size = modules.length;
  let penalty = 0;

  for (let y = 0; y < size; y += 1) penalty += scoreRuns(modules[y]);
  for (let x = 0; x < size; x += 1) penalty += scoreRuns(modules.map((row) => row[x]));

  for (let y = 0; y < size - 1; y += 1) {
    for (let x = 0; x < size - 1; x += 1) {
      const color = modules[y][x];
      if (modules[y][x + 1] === color && modules[y + 1][x] === color && modules[y + 1][x + 1] === color) {
        penalty += 3;
      }
    }
  }

  const dark = modules.flat().filter(Boolean).length;
  penalty += Math.floor(Math.abs((dark * 20) / (size * size) - 10)) * 10;
  return penalty;
}

function scoreRuns(line) {
  let penalty = 0;
  let runColor = line[0];
  let runLength = 1;
  for (let i = 1; i <= line.length; i += 1) {
    if (line[i] === runColor) {
      runLength += 1;
    } else {
      if (runLength >= 5) penalty += runLength - 2;
      runColor = line[i];
      runLength = 1;
    }
  }
  return penalty;
}

function reedSolomon(data, degree) {
  const generator = rsGenerator(degree);
  const result = Array(degree).fill(0);
  for (const byte of data) {
    const factor = byte ^ result.shift();
    result.push(0);
    for (let i = 0; i < degree; i += 1) {
      result[i] ^= gfMultiply(generator[i], factor);
    }
  }
  return result;
}

function rsGenerator(degree) {
  let result = [1];
  for (let i = 0; i < degree; i += 1) {
    result = rsMultiply(result, [1, gfPow(2, i)]);
  }
  return result.slice(1);
}

function rsMultiply(left, right) {
  const result = Array(left.length + right.length - 1).fill(0);
  for (let i = 0; i < left.length; i += 1) {
    for (let j = 0; j < right.length; j += 1) {
      result[i + j] ^= gfMultiply(left[i], right[j]);
    }
  }
  return result;
}

function gfPow(value, power) {
  let result = 1;
  for (let i = 0; i < power; i += 1) {
    result = gfMultiply(result, value);
  }
  return result;
}

function gfMultiply(left, right) {
  let result = 0;
  let a = left;
  let b = right;
  while (b > 0) {
    if (b & 1) result ^= a;
    a <<= 1;
    if (a & 0x100) a ^= 0x11d;
    b >>= 1;
  }
  return result;
}

function appendBits(bits, value, length) {
  for (let i = length - 1; i >= 0; i -= 1) {
    bits.push((value >> i) & 1);
  }
}

function bitsToByte(bits) {
  return bits.reduce((value, current) => (value << 1) | current, 0);
}

function bit(value, index) {
  return ((value >> index) & 1) !== 0;
}

function setFunctionModule(modules, reserved, x, y, dark) {
  if (!isInside(modules, x, y)) return;
  modules[y][x] = dark;
  reserved[y][x] = true;
}

function isInside(modules, x, y) {
  return y >= 0 && y < modules.length && x >= 0 && x < modules.length;
}
