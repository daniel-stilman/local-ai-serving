'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..');
const packageJson = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const configExample = JSON.parse(fs.readFileSync(path.join(ROOT, 'config.example.json'), 'utf8'));
const gitignore = fs.readFileSync(path.join(ROOT, '.gitignore'), 'utf8');
const localConfigSource = fs.readFileSync(path.join(ROOT, 'local-config.js'), 'utf8');
const readme = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
const html = fs.readFileSync(path.join(ROOT, 'public', 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(ROOT, 'public', 'styles.css'), 'utf8');
const app = fs.readFileSync(path.join(ROOT, 'public', 'app.js'), 'utf8');
const dashboardHtml = fs.readFileSync(path.join(ROOT, 'public', 'dashboard.html'), 'utf8');
const dashboardCss = fs.readFileSync(path.join(ROOT, 'public', 'dashboard.css'), 'utf8');
const dashboardJs = fs.readFileSync(path.join(ROOT, 'public', 'dashboard.js'), 'utf8');
const startJs = fs.readFileSync(path.join(ROOT, 'scripts', 'start.js'), 'utf8');
const trayPs1 = fs.readFileSync(path.join(ROOT, 'scripts', 'tray.ps1'), 'utf8');
const startCmd = fs.readFileSync(path.join(ROOT, 'Start Local LLM Serve.cmd'), 'utf8');
const trayCmd = fs.readFileSync(path.join(ROOT, 'Start Local LLM Serve Tray.cmd'), 'utf8');
const serverJs = fs.readFileSync(path.join(ROOT, 'server.js'), 'utf8');
const imageWorker = fs.readFileSync(path.join(ROOT, 'inference', 'worker.py'), 'utf8');
const imageRuntime = fs.readFileSync(path.join(ROOT, 'inference', 'runtime.py'), 'utf8');
const animaEngine = fs.readFileSync(path.join(ROOT, 'inference', 'anima.py'), 'utf8');
const sdxlEngine = fs.readFileSync(path.join(ROOT, 'inference', 'sdxl.py'), 'utf8');
const loraEngine = fs.readFileSync(path.join(ROOT, 'inference', 'lora.py'), 'utf8');
const modelValidator = fs.readFileSync(path.join(ROOT, 'inference', 'validate_models.py'), 'utf8');
const imageSetup = fs.readFileSync(path.join(ROOT, 'scripts', 'fetch-image-assets.js'), 'utf8');
const configureSource = fs.readFileSync(path.join(ROOT, 'scripts', 'configure.js'), 'utf8');
const smokeSource = fs.readFileSync(path.join(ROOT, 'scripts', 'smoke-hardware.js'), 'utf8');
const browserSmokeSource = fs.readFileSync(path.join(ROOT, 'scripts', 'smoke-browser.js'), 'utf8');
const validatorLauncher = fs.readFileSync(path.join(ROOT, 'scripts', 'validate-image-models.js'), 'utf8');
const textBackendSource = fs.readFileSync(path.join(ROOT, 'text-backend.js'), 'utf8');
const { openUrl, parsePort } = require('../scripts/start');

test('mobile drawer has an overlay and escape-to-close behavior', () => {
  assert.match(html, /id="sidebarOverlay"/);
  assert.match(html, /class="sidebar-overlay mobile-only"/);
  assert.match(app, /sidebarOverlay: document\.getElementById\('sidebarOverlay'\)/);
  assert.match(app, /els\.sidebarOverlay\.addEventListener\('click', closeSidebar\)/);
  assert.match(app, /event\.key === 'Escape'/);
  assert.match(app, /document\.body\.classList\.add\('sidebar-is-open'\)/);
  assert.match(app, /document\.body\.classList\.remove\('sidebar-is-open'\)/);
});

test('default launcher opens the local dashboard while raw server remains available', () => {
  assert.equal(packageJson.scripts.start, 'node scripts/start.js');
  assert.equal(packageJson.scripts.serve, 'node server.js');
  assert.match(startJs, /\/dashboard/);
  assert.match(startJs, /spawn\(process\.execPath, \[path\.join\(root, 'server\.js'\)\]/);
  assert.match(startCmd, /npm start/);
});

test('dashboard launcher validates ports and never invokes a Windows command shell', () => {
  assert.equal(parsePort(undefined), 3000);
  assert.equal(parsePort('1'), 1);
  assert.equal(parsePort('65535'), 65535);
  for (const value of ['', '0', '65536', '-1', '3000.5', ' 3000', '3000 ', '3e3', '3000&calc']) {
    assert.throws(() => parsePort(value), /integer from 1 through 65535/);
  }

  const calls = [];
  const fakeSpawn = (...args) => {
    calls.push(args);
    return { unref() {} };
  };
  const url = 'https://localhost:3000/dashboard';
  openUrl(url, 'win32', fakeSpawn);
  assert.deepEqual(calls, [[
    'explorer.exe',
    [url],
    { detached: true, shell: false, stdio: 'ignore', windowsHide: true },
  ]]);
  calls.length = 0;
  openUrl(url, 'darwin', fakeSpawn);
  openUrl(url, 'linux', fakeSpawn);
  assert.deepEqual(calls, [
    ['open', [url], { detached: true, shell: false, stdio: 'ignore' }],
    ['xdg-open', [url], { detached: true, shell: false, stdio: 'ignore' }],
  ]);
  assert.doesNotMatch(startJs, /(?:cmd(?:\.exe)?|powershell|pwsh)[\s\S]*\/c/i);
});

test('Windows tray shortcut exposes dashboard and server controls', () => {
  assert.match(trayCmd, /-STA -File/);
  assert.match(trayPs1, /NotifyIcon/);
  assert.match(trayPs1, /Open Dashboard/);
  assert.match(trayPs1, /Restart Server/);
  assert.match(trayPs1, /Stop Server/);
});

test('app consumes QR access tokens from URL fragments and sends them as headers', () => {
  assert.match(app, /ACCESS_TOKEN_STORAGE_KEY/);
  assert.match(app, /window\.location\.hash/);
  assert.match(app, /sessionStorage\.setItem\(ACCESS_TOKEN_STORAGE_KEY, token\)/);
  assert.match(app, /window\.history\.replaceState/);
  assert.match(app, /'X-Access-Token': accessToken/);
  assert.match(app, /async function apiFetch/);
  assert.doesNotMatch(app, /fetch\('\/api/);
});

test('app ignores legacy query-string access tokens and scrubs them from the URL', () => {
  const source = extractFunctionSource(app, 'consumeAccessToken');
  assert.match(source, /const token = hashParams\.get\('access'\) \|\| storedToken/);
  assert.doesNotMatch(source, /searchParams\.get\('access'\)/);
  assert.match(source, /searchParams\.delete\('access'\)/);

  const storage = new Map([['test-access-storage', 'stored-value']]);
  const replacements = [];
  const windowState = {
    history: { replaceState(_state, _title, url) { replacements.push(url); } },
    location: {
      hash: '#view=chat',
      pathname: '/chat',
      search: '?access=query-value&theme=dark',
    },
  };
  const consumeAccessToken = vm.runInNewContext(
    `(() => { const ACCESS_TOKEN_STORAGE_KEY = 'test-access-storage'; return (${source}); })()`,
    {
      URLSearchParams,
      sessionStorage: {
        getItem(key) { return storage.get(key) || null; },
        setItem(key, value) { storage.set(key, value); },
      },
      window: windowState,
    },
  );
  assert.equal(consumeAccessToken(), 'stored-value');
  assert.equal(storage.get('test-access-storage'), 'stored-value');
  assert.deepEqual(replacements, ['/chat?theme=dark#view=chat']);

  storage.clear();
  replacements.length = 0;
  assert.equal(consumeAccessToken(), '');
  assert.equal(storage.size, 0);
  assert.deepEqual(replacements, ['/chat?theme=dark#view=chat']);

  windowState.location.hash = '#access=fragment-value&view=chat';
  windowState.location.search = '?theme=dark';
  replacements.length = 0;
  assert.equal(consumeAccessToken(), 'fragment-value');
  assert.equal(storage.get('test-access-storage'), 'fragment-value');
  assert.deepEqual(replacements, ['/chat?theme=dark#view=chat']);

  windowState.location.hash = '#access=mobile-fragment';
  windowState.history.replaceState = () => { throw new Error('history writes blocked'); };
  assert.equal(consumeAccessToken(), 'mobile-fragment');
  assert.equal(storage.get('test-access-storage'), 'mobile-fragment');
});

test('missing and stale phone access never masquerade as empty text or image model lists', async () => {
  assert.match(html, /id="accessRequired"[^>]*role="alert"[^>]*hidden/);
  assert.match(html, /Scan the current QR code from the local dashboard/);
  assert.match(css, /\.access-required\s*{[^}]*position:\s*absolute/s);
  assert.match(css, /\.access-required-window/);
  const initializeSource = extractFunctionSource(app, 'initialize');
  assert.match(initializeSource, /Promise\.all\(\[loadServerConfig\(\), refreshImageConfig\(\)\]\)/);
  assert.doesNotMatch(initializeSource, /if \(!accessToken\)|if \(accessIsRequired\)[\s\S]*return/);
  assert.match(extractFunctionSource(app, 'refreshModels'), /refreshModelsButton\.disabled = accessIsRequired/);
  assert.match(extractFunctionSource(app, 'refreshModels'), /testConnectionButton\.disabled = accessIsRequired/);
  assert.match(extractFunctionSource(app, 'refreshImageConfig'), /refreshImageModelsButton\.disabled = accessIsRequired/);

  const sources = [
    'apiFetch',
    'makeApiHeaders',
    'clearAccessToken',
    'setAccessRequired',
    'renderAccessRequiredModelOptions',
    'replaceSelectWithPlaceholder',
  ].map((name) => {
    const source = extractFunctionSource(app, name);
    return name === 'apiFetch' ? `async ${source}` : source;
  }).join('\n\n');
  const context = {};
  vm.runInNewContext(`
    const ACCESS_TOKEN_STORAGE_KEY = 'synthetic-access-storage';
    const ACCESS_REQUIRED_MESSAGE = 'Access required. Scan the current QR code from the local dashboard.';
    const ACCESS_REQUIRED_OPTION_TEXT = 'Access required - scan the current QR code';
    let accessToken = '';
    let accessIsRequired = false;
    let responseStatus = 200;
    let responseAccessRequired = '';
    let session = new Map();
    let requests = [];
    let bodyClasses = new Set();
    let lastStatus = null;
    let imageConfig = {};
    let els;

    function makeSelect(label) {
      const initial = { value: label ? 'synthetic-id' : '', textContent: label };
      return {
        disabled: false,
        value: initial.value,
        options: label ? [initial] : [],
        replaceChildren(...children) {
          this.options = children;
          this.value = children[0] ? children[0].value : '';
        },
      };
    }

    const document = {
      body: {
        classList: {
          toggle(name, enabled) {
            if (enabled) bodyClasses.add(name);
            else bodyClasses.delete(name);
          },
        },
      },
      createElement() { return { value: '', textContent: '' }; },
    };
    const sessionStorage = {
      getItem(key) { return session.get(key) || null; },
      setItem(key, value) { session.set(key, value); },
      removeItem(key) { session.delete(key); },
    };
    async function fetch(resource, options) {
      requests.push({ resource, headers: { ...(options.headers || {}) } });
      return {
        status: responseStatus,
        ok: responseStatus >= 200 && responseStatus < 300,
        headers: { get(name) { return name.toLowerCase() === 'x-local-access-required' ? responseAccessRequired : null; } },
      };
    }
    function setStatus(kind, text) { lastStatus = { kind, text }; }
    function updateModelSummary() {}
    function setBusy() {}
    function updateImageToolStatus() {}

    ${sources}

    async function runScenario(name, token, status, accessMarker = '') {
      accessToken = token;
      accessIsRequired = false;
      responseStatus = status;
      responseAccessRequired = accessMarker;
      session = new Map(token ? [[ACCESS_TOKEN_STORAGE_KEY, token]] : []);
      requests = [];
      bodyClasses = new Set();
      lastStatus = null;
      els = {
        accessRequired: { hidden: true },
        modelSelect: makeSelect('Synthetic text model'),
        imageModelSelect: makeSelect('Synthetic image model'),
        imageConnectionText: { hidden: true, textContent: '' },
        imageFormError: { textContent: '' },
        imageLoraList: { replaceChildren() {} },
        imageLoraHint: { textContent: '' },
        addImageLoraButton: { disabled: false },
        refreshModelsButton: { disabled: false },
        testConnectionButton: { disabled: false },
        refreshImageModelsButton: { disabled: false },
        generateImageButton: { disabled: false },
      };

      let errorName = '';
      try {
        await apiFetch('/api/models', { method: 'POST' });
      } catch (error) {
        errorName = error.name;
      }
      return {
        name,
        errorName,
        accessToken,
        storedToken: session.get(ACCESS_TOKEN_STORAGE_KEY) || '',
        accessHidden: els.accessRequired.hidden,
        accessClass: bodyClasses.has('access-is-required'),
        requestHeaders: requests[0].headers,
        textOptions: els.modelSelect.options.map((option) => option.textContent),
        imageOptions: els.imageModelSelect.options.map((option) => option.textContent),
        textDisabled: els.modelSelect.disabled,
        imageDisabled: els.imageModelSelect.disabled,
        imageStatus: els.imageConnectionText.textContent,
        refreshDisabled: els.refreshModelsButton.disabled,
        testDisabled: els.testConnectionButton.disabled,
        imageRefreshDisabled: els.refreshImageModelsButton.disabled,
        lastStatus,
      };
    }

    globalThis.resultPromise = (async () => [
      await runScenario('current', 'current-synthetic-token', 200),
      await runScenario('local-bypass', '', 200),
      await runScenario('missing', '', 401, '1'),
      await runScenario('stale', 'stale-synthetic-token', 401, '1'),
      await runScenario('upstream-unauthorized', 'current-synthetic-token', 401),
    ])();
  `, context);

  const [current, localBypass, missing, stale, upstreamUnauthorized] = JSON.parse(JSON.stringify(await context.resultPromise));
  assert.equal(current.errorName, '');
  assert.equal(current.accessHidden, true);
  assert.equal(current.accessClass, false);
  assert.equal(current.requestHeaders['X-Access-Token'], 'current-synthetic-token');
  assert.deepEqual(current.textOptions, ['Synthetic text model']);
  assert.deepEqual(current.imageOptions, ['Synthetic image model']);

  assert.equal(localBypass.errorName, '');
  assert.equal(localBypass.accessHidden, true);
  assert.equal(localBypass.accessClass, false);
  assert.equal(localBypass.requestHeaders['X-Access-Token'], undefined);
  assert.deepEqual(localBypass.textOptions, ['Synthetic text model']);
  assert.deepEqual(localBypass.imageOptions, ['Synthetic image model']);

  for (const denied of [missing, stale]) {
    assert.equal(denied.errorName, 'AccessRequiredError');
    assert.equal(denied.accessToken, '');
    assert.equal(denied.storedToken, '');
    assert.equal(denied.accessHidden, false);
    assert.equal(denied.accessClass, true);
    assert.deepEqual(denied.textOptions, ['Access required - scan the current QR code']);
    assert.deepEqual(denied.imageOptions, ['Access required - scan the current QR code']);
    assert.equal(denied.textDisabled, true);
    assert.equal(denied.imageDisabled, true);
    assert.equal(denied.refreshDisabled, true);
    assert.equal(denied.testDisabled, true);
    assert.equal(denied.imageRefreshDisabled, true);
    assert.match(denied.imageStatus, /Scan the current QR code/);
    assert.deepEqual(denied.lastStatus, { kind: 'error', text: 'Access required' });
  }
  assert.equal(missing.requestHeaders['X-Access-Token'], undefined);
  assert.equal(stale.requestHeaders['X-Access-Token'], 'stale-synthetic-token');

  assert.equal(upstreamUnauthorized.errorName, '');
  assert.equal(upstreamUnauthorized.accessToken, 'current-synthetic-token');
  assert.equal(upstreamUnauthorized.storedToken, 'current-synthetic-token');
  assert.equal(upstreamUnauthorized.accessHidden, true);
  assert.equal(upstreamUnauthorized.accessClass, false);
  assert.deepEqual(upstreamUnauthorized.textOptions, ['Synthetic text model']);
  assert.deepEqual(upstreamUnauthorized.imageOptions, ['Synthetic image model']);
});

test('browser harness contract includes phone authentication, both model systems, and recovered appearance', () => {
  assert.match(browserSmokeSource, /current fragment token/);
  assert.match(browserSmokeSource, /legacy saved backend/);
  assert.match(browserSmokeSource, /current explicit external override/);
  assert.match(browserSmokeSource, /Page\.addScriptToEvaluateOnNewDocument/);
  assert.match(browserSmokeSource, /discovered text models are not selectable/);
  assert.match(browserSmokeSource, /missing token/);
  assert.match(browserSmokeSource, /stale fragment token/);
  assert.match(browserSmokeSource, /X-Local-Access-Required/);
  assert.match(browserSmokeSource, /imageOptionsByKind/);
  assert.match(browserSmokeSource, /bodyBackgroundColor/);
  assert.match(browserSmokeSource, /DESKTOP_VIEWPORT/);
  assert.match(browserSmokeSource, /sidebarPosition/);
  assert.match(browserSmokeSource, /composerBottom/);
  assert.match(browserSmokeSource, /messagesBackgroundImage/);
  assert.match(browserSmokeSource, /topbarBackgroundImage/);
  assert.match(browserSmokeSource, /Emulation\.setDeviceMetricsOverride/);
  assert.match(browserSmokeSource, /\/api\/text\/load/);
  assert.match(browserSmokeSource, /\/api\/text\/status/);
  assert.match(browserSmokeSource, /selectionLoad/);
  assert.match(browserSmokeSource, /selectionReady/);
  assert.match(browserSmokeSource, /indeterminate progress state/);
  assert.match(browserSmokeSource, /class RawWebSocket/);
  assert.doesNotMatch(browserSmokeSource, /safetensors/i);
  assert.doesNotMatch(browserSmokeSource, /['"][^'"\r\n]*\.gguf['"]/i);
});

test('browser harness contract includes actual-server dashboard folder setup', () => {
  assert.match(browserSmokeSource, /runRealServerModelSetupScenario/);
  assert.match(browserSmokeSource, /spawn\(process\.execPath, \['server\.js'\]/);
  assert.match(browserSmokeSource, /LOCAL_CONFIG_FILE: configPath/);
  assert.match(browserSmokeSource, /initial local config contains more than the synthetic runtime path/);
  assert.match(browserSmokeSource, /path\.join\(modelsRoot, `\$\{id\}\.gguf`\)/);
  assert.match(browserSmokeSource, /textModelsRootInput/);
  assert.match(browserSmokeSource, /saveTextModelsRootButton/);
  assert.doesNotMatch(extractFunctionSource(browserSmokeSource, 'configureModelsWithCdp'), /pickTextModelsRootButton/);
  assert.doesNotMatch(browserSmokeSource, /\/api\/local-setup\/pick-folder/);
  assert.match(browserSmokeSource, /refreshModelsThroughDashboard/);
  assert.match(browserSmokeSource, /refreshModelSetupButton/);
  assert.match(browserSmokeSource, /model refresh unexpectedly replaced the server process/);
  assert.match(browserSmokeSource, /real server legacy saved backend migration/);
  assert.match(browserSmokeSource, /managed text models did not populate after legacy state migration/);
  assert.match(browserSmokeSource, /managed text model selector remained disabled/);
  assert.match(browserSmokeSource, /if \(key\.startsWith\('TEXT_'\)\) delete environment\[key\]/);
  assert.match(browserSmokeSource, /removeRealServerFixture/);
  assert.match(browserSmokeSource, /stopBrowserProcess/);
  assert.match(browserSmokeSource, /headless browser profile could not be removed/);
  assert.match(browserSmokeSource, /Direct text engine ready/);
});

test('browser harness contract covers persisted model choice, mixed multi-turn actions, and phone isolation', () => {
  assert.match(browserSmokeSource, /current stale saved model replacement/);
  assert.match(browserSmokeSource, /unavailable saved model was not replaced and persisted/);
  assert.match(browserSmokeSource, /exercisePersistedChatSelection/);
  assert.match(browserSmokeSource, /Page\.reload/);
  assert.match(browserSmokeSource, /chat payload did not use the persisted second model/);
  assert.match(browserSmokeSource, /text\/event-stream/);
  assert.match(browserSmokeSource, /assertTextTurnHistory/);
  assert.match(browserSmokeSource, /assertMixedMediaHistory/);
  assert.match(browserSmokeSource, /assertConversationLifecycle/);
  assert.match(browserSmokeSource, /generateManualImageThroughUi/);
  assert.match(browserSmokeSource, /exerciseConversationIsolation/);
  assert.match(browserSmokeSource, /synthetic-phone\.invalid/);
  assert.match(browserSmokeSource, /host-resolver-rules/);
  assert.match(browserSmokeSource, /phone-style browser exposed host-local setup/);
});

test('managed text migrates legacy backend state while preserving explicit modern overrides', () => {
  assert.match(app, /baseUrlOverride:\s*false/);
  assert.match(app, /managedTextBackendEnabled = Boolean\(config\.managedTextBackend\?\.enabled\)/);
  assert.match(app, /managedTextBackendEnabled && !state\.settings\.baseUrlOverride/);
  assert.match(app, /state\.settings\.baseUrl = serverDefaultBaseUrl/);
  assert.match(app, /state\.settings\.baseUrlOverride = false/);
  assert.match(app, /settings\.baseUrlOverride === true/);
  assert.match(app, /baseUrlEdited/);
  assert.match(browserSmokeSource, /initialState:\s*\{ settings:\s*\{ baseUrl: LEGACY_BASE_URL \} \}/);
  assert.match(browserSmokeSource, /baseUrlOverride: true/);
  assert.match(browserSmokeSource, /JSON\.parse\(modelRequest\.body\)\.baseUrl/);

  const normalizeSettingsSource = extractFunctionSource(app, 'normalizeSettings');
  const normalizeSettings = vm.runInNewContext(
    `(() => {
      const DEFAULT_SETTINGS = { baseUrl: '', baseUrlOverride: false, apiKey: '', modelThinkingModes: {} };
      const normalizeModelThinkingModes = value => value || {};
      return (${normalizeSettingsSource});
    })()`,
  );
  assert.equal(normalizeSettings({ baseUrl: 'http://synthetic-legacy.invalid/v1' }).baseUrlOverride, false);
  assert.equal(normalizeSettings({
    baseUrl: 'http://synthetic-external.invalid/v1',
    baseUrlOverride: true,
  }).baseUrlOverride, true);
});

test('model settings expose host-local folder setup and actionable remote guidance', () => {
  assert.match(html, /id="localModelSetupLink"[^>]*href="\/dashboard#models"[^>]*hidden/);
  assert.match(html, /id="localModelSetupGuidance"[^>]*>[^<]*computer running this server/i);
  assert.match(app, /renderLocalModelSetup\(config\.localSetupAvailable === true\)/);
  assert.match(app, /localModelSetupLink\.hidden = !localSetupAvailable/);
  assert.match(app, /localModelSetupGuidance\.hidden = localSetupAvailable/);
  assert.match(css, /\.local-model-setup\s*\{/);
  assert.match(css, /\.local-model-setup a:focus-visible/);
  assert.match(browserSmokeSource, /localSetupLinkVisible/);
  assert.match(browserSmokeSource, /localSetupGuidanceVisible/);
});

test('optional API keys are not persisted in saved browser state', () => {
  assert.match(app, /settings:\s*{[\s\S]*apiKey:\s*''/);
  assert.match(app, /localStorage\.setItem\(STORAGE_KEY, JSON\.stringify\(savedState\)\)/);
});

test('default generation temperature is tuned for lively local chat', () => {
  assert.match(app, /temperature:\s*0\.95/);
});

test('mobile scroll boundaries do not chain into browser pull-to-refresh', () => {
  assert.match(css, /html,\s*body\s*{[^}]*overscroll-behavior:\s*none/s);
  assert.match(css, /body\s*{[^}]*overflow:\s*hidden/s);
  assert.match(css, /\.messages\s*{[^}]*overscroll-behavior:\s*contain/s);
  assert.match(css, /\.conversation-list\s*{[^}]*overscroll-behavior:\s*contain/s);
  assert.match(dashboardCss, /html,\s*body\s*{[^}]*overscroll-behavior:\s*none/s);
});

test('streaming response updates do not drag the message viewport', () => {
  const updateStreamingMessage = app.match(/function updateStreamingMessage\(message\) \{[\s\S]*?\n\}/)[0];
  assert.doesNotMatch(updateStreamingMessage, /scrollMessagesToBottom/);
  assert.match(updateStreamingMessage, /requestAnimationFrame\(flushStreamingMessages\)/);
  assert.match(app, /node\.replaceChildren\(\.\.\.replacement\.childNodes\)/);
  assert.match(app, /function renderActiveConversation\(options = \{\}\)/);
  assert.match(app, /captureMessagesViewport\(\)/);
  assert.match(app, /restoreMessagesViewport\(viewportSnapshot\)/);
  assert.match(app, /anchorOffset/);
  assert.match(app, /persistAndRender\(\{ messages: \{ preserveScroll: true \} \}\)/);
  assert.match(css, /\.messages\s*{[^}]*overflow-anchor:\s*none/s);
});

test('assistant thinking is separated from the visible answer', () => {
  assert.match(app, /getReasoningDelta\(delta\)/);
  assert.match(app, /reasoning_content/);
  assert.match(app, /reasoningContent/);
  assert.match(app, /const startTag = '<think>'/);
  assert.match(app, /assistantMessage\.reasoning \+= delta\.reasoning/);
  assert.match(app, /assistantMessage\.content \+= delta\.content/);
  assert.match(app, /message\.role === 'assistant' \? getAssistantMessageParts\(message\)\.content : message\.content/);
  assert.match(app, /className = 'thinking-panel'/);
  assert.match(css, /\.thinking-panel/);
  assert.match(css, /\.thinking-content/);
});

test('thinking mode can be configured per selected model', () => {
  assert.match(html, /id="thinkingModeSelect"/);
  assert.match(app, /modelThinkingModes:\s*\{\}/);
  assert.match(app, /thinkingModeSelect: document\.getElementById\('thinkingModeSelect'\)/);
  assert.match(app, /setSelectedModelThinkingMode\(els\.thinkingModeSelect\.value\)/);
  assert.match(app, /thinkingMode:\s*getSelectedModelThinkingMode\(\)/);
  assert.match(app, /function normalizeModelThinkingModes\(value\)/);
  assert.match(html, /id="chatSetupButton"/);
});

test('streaming thinking panels preserve user collapsed state', () => {
  const flushStreamingMessages = extractFunctionSource(app, 'flushStreamingMessages');
  assert.match(app, /const thinkingPanelOpenByMessage = new Map\(\)/);
  assert.match(app, /thinkingPanelOpenByMessage\.get\(message\.id\)/);
  assert.match(app, /panel\.addEventListener\('toggle'/);
  assert.match(flushStreamingMessages, /thinkingPanelOpenByMessage\.set\(message\.id, panel\.open\)/);
});

test('thinking parser handles API reasoning fields and split think tags', () => {
  const parserStart = app.indexOf('function getAssistantMessageParts');
  const parserEnd = app.indexOf('function updateStreamingMessage');
  assert.notEqual(parserStart, -1);
  assert.notEqual(parserEnd, -1);

  const context = {};
  vm.runInNewContext(`${app.slice(parserStart, parserEnd)}
    const parser = createThinkingParser();
    const first = parser.append('Visible <thi');
    const second = parser.append('nk>hidden</thi');
    const third = parser.append('nk> answer');
    const final = parser.flush();
    globalThis.result = {
      first,
      second,
      third,
      final,
      split: splitThinkingFromText('A <think>B</think> C'),
      parts: getAssistantMessageParts({ content: 'A <think>B</think> C', reasoning: 'field' }),
      reasoningField: getReasoningDelta({ reasoning_content: 'api' }),
      reasoningCamel: getReasoningDelta({ reasoningContent: 'camel' })
    };
  `, context);

  const result = JSON.parse(JSON.stringify(context.result));
  assert.deepEqual(result.first, { content: 'Visible ', reasoning: '' });
  assert.deepEqual(result.second, { content: '', reasoning: 'hidden' });
  assert.deepEqual(result.third, { content: ' answer', reasoning: '' });
  assert.deepEqual(result.final, { content: '', reasoning: '' });
  assert.deepEqual(result.split, { content: 'A  C', reasoning: 'B' });
  assert.deepEqual(result.parts, { content: 'A  C', reasoning: 'field\n\nB' });
  assert.equal(result.reasoningField, 'api');
  assert.equal(result.reasoningCamel, 'camel');
});

test('browser chat stream requires valid events and an explicit DONE terminator', async () => {
  const streamStart = app.indexOf('async function readChatStream');
  const streamEnd = app.indexOf('function buildTextBackendMessages');
  assert.notEqual(streamStart, -1);
  assert.notEqual(streamEnd, -1);
  const dependencies = [
    'getReasoningDelta',
    'createThinkingParser',
    'appendThinkingPart',
  ].map((name) => extractFunctionSource(app, name)).join('\n\n');
  const context = { TextDecoder, TextEncoder };
  vm.runInNewContext(`
    ${app.slice(streamStart, streamEnd)}
    ${dependencies}
    function responseFrom(parts) {
      const encoder = new TextEncoder();
      let index = 0;
      return {
        body: {
          getReader() {
            return {
              async read() {
                if (index >= parts.length) return { done: true, value: undefined };
                return { done: false, value: encoder.encode(parts[index++]) };
              }
            };
          }
        }
      };
    }
    globalThis.resultPromise = (async () => {
      const deltas = [];
      await readChatStream(responseFrom([
        'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\\n',
        '\\ndata: {"choices":[{"delta":{"content":"Vis',
        'ible"}}]}\\n\\ndata: [DONE]\\n\\n'
      ]), (delta) => deltas.push(delta));
      let missingDone = '';
      try {
        await readChatStream(responseFrom([
          'data: {"choices":[{"delta":{"content":"partial"}}]}\\n\\n'
        ]), () => {});
      } catch (error) {
        missingDone = error.message;
      }
      let invalidEvent = '';
      try {
        await readChatStream(responseFrom(['data: {invalid}\\n\\ndata: [DONE]\\n\\n']), () => {});
      } catch (error) {
        invalidEvent = error.message;
      }
      return { deltas, missingDone, invalidEvent };
    })();
  `, context);

  const result = JSON.parse(JSON.stringify(await context.resultPromise));
  assert.deepEqual(result.deltas, [
    { content: '', reasoning: 'hidden' },
    { content: 'Visible', reasoning: '' },
  ]);
  assert.match(result.missingDone, /before completion/i);
  assert.match(result.invalidEvent, /invalid stream event/i);
});

test('model context excludes previous assistant thinking without mutating displayed conversation', () => {
  const sources = [
    'buildTextBackendMessages',
    'getAssistantMessageParts',
    'splitThinkingFromText',
    'createThinkingParser',
    'appendThinkingPart',
  ].map((name) => extractFunctionSource(app, name)).join('\n\n');

  const context = {};
  vm.runInNewContext(`
    const state = { settings: { systemPrompt: '  System prompt  ' } };
    let streamingMessageId = 'assistant-streaming';
    ${sources}
    const conversation = {
      messages: [
        { id: 'user-1', role: 'user', content: 'User keeps literal <think>syntax</think>.', reasoning: 'ignored user field' },
        { id: 'assistant-1', role: 'assistant', content: '<think>hidden scratch</think>Visible answer<think>more hidden</think> after', reasoning: 'api hidden' },
        { id: 'assistant-2', role: 'assistant', content: '<THINK>upper hidden</THINK>Mixed case done', reasoning: '' },
        { id: 'assistant-3', role: 'assistant', content: '<think>all hidden</think>', reasoning: 'api-only hidden' },
        { id: 'assistant-4', role: 'assistant', content: 'Visible before <think>unfinished hidden', reasoning: '' },
        { id: 'assistant-5', role: 'assistant', content: 'Partial <thi tag remains visible', reasoning: '' },
        { id: 'assistant-streaming', role: 'assistant', content: 'Streaming answer should be skipped', reasoning: 'streaming hidden' },
        { id: 'user-2', role: 'user', content: 'Continue', reasoning: '' }
      ]
    };
    globalThis.before = JSON.stringify(conversation);
    globalThis.messages = buildTextBackendMessages(conversation);
    globalThis.after = JSON.stringify(conversation);
  `, context);

  assert.equal(context.before, context.after);
  assert.deepEqual(JSON.parse(JSON.stringify(context.messages)), [
    { role: 'system', content: 'System prompt' },
    { role: 'user', content: 'User keeps literal <think>syntax</think>.' },
    { role: 'assistant', content: 'Visible answer after' },
    { role: 'assistant', content: 'Mixed case done' },
    { role: 'assistant', content: 'Visible before ' },
    { role: 'assistant', content: 'Partial <thi tag remains visible' },
    { role: 'user', content: 'Continue' },
  ]);

  const serializedMessages = JSON.stringify(context.messages);
  for (const hiddenText of [
    'hidden scratch',
    'more hidden',
    'api hidden',
    'upper hidden',
    'all hidden',
    'api-only hidden',
    'unfinished hidden',
    'Streaming answer should be skipped',
    'streaming hidden',
  ]) {
    assert.equal(serializedMessages.includes(hiddenText), false, `${hiddenText} leaked into model context`);
  }
});

test('primary controls meet mobile touch target requirements', () => {
  assert.match(css, /--touch:\s*44px;/);
  assert.match(css, /\.primary-button,[\s\S]*?\.menu-action\s*{[^}]*min-height:\s*var\(--touch\)/s);
  assert.match(css, /#promptInput\s*{[^}]*min-height:\s*48px/s);
  assert.match(css, /\.send-button\s*{[^}]*min-height:\s*48px/s);
  assert.match(css, /\.three-actions\s*{[^}]*grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/s);
});

test('desktop, tablet, and phone layouts each have explicit responsive contracts', () => {
  assert.match(css, /@media\s*\(max-width:\s*860px\)/);
  assert.match(css, /@media\s*\(max-width:\s*640px\)/);
  assert.match(css, /@media\s*\(max-width:\s*480px\)/);
  assert.match(css, /\.app-shell\s*{[^}]*grid-template-columns:\s*292px minmax\(0,\s*1fr\)/s);
  assert.match(css, /@media\s*\(max-width:\s*860px\)[\s\S]*?\.app-shell\s*{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/s);
  assert.match(css, /\.chat-layout\s*{[^}]*grid-template-areas:\s*"topbar"\s*"model-loading"\s*"messages"\s*"composer"/s);
  for (const [selector, area] of [
    ['\\.topbar', 'topbar'],
    ['\\.model-loading', 'model-loading'],
    ['\\.messages', 'messages'],
    ['\\.composer', 'composer'],
  ]) {
    assert.match(css, new RegExp(`${selector}\\s*\\{[^}]*grid-area:\\s*${area}`));
  }
});

test('mobile menu and composer placement are comfortable for handheld use', () => {
  assert.match(css, /@media\s*\(max-width:\s*860px\)[\s\S]*?\.sidebar\s*{[^}]*width:\s*min\(340px,\s*88vw\)/s);
  assert.match(css, /body\.sidebar-is-open \.sidebar-overlay\s*{[^}]*display:\s*block/s);
  assert.match(css, /@media\s*\(max-width:\s*860px\)[\s\S]*?\.topbar\s*{[^}]*grid-template-columns:\s*var\(--touch\) minmax\(0,\s*1fr\) minmax\(76px,\s*120px\) var\(--touch\)/s);
  assert.match(css, /@media\s*\(max-width:\s*860px\)[\s\S]*?\.composer-box\s*{[^}]*grid-template-columns:\s*46px minmax\(0,\s*1fr\) 62px/s);
  assert.match(css, /\.composer-meta\s*{[^}]*display:\s*none/s);
  assert.match(css, /@media\s*\(max-width:\s*640px\)[\s\S]*?\.sheet-dialog,[\s\S]*?margin:\s*auto 0 0/s);
});

test('visual system uses a coherent semantic palette', () => {
  for (const token of ['--bg', '--surface', '--text', '--muted', '--accent', '--positive', '--danger']) {
    assert.match(css, new RegExp(`${token}:\\s*#[0-9a-fA-F]{6};`));
  }
  assert.match(html, /Local Chat/);
  assert.match(css, /--radius:\s*0px/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
});

test('retro desktop theme is coherent across app and dashboard', () => {
  assert.match(css, /--bg:\s*#008080;/);
  assert.match(css, /--accent:\s*#000080;/);
  assert.match(css, /--surface:\s*#c0c0c0;/);
  assert.match(css, /--bevel-up:/);
  assert.match(css, /--bevel-field:/);
  assert.match(css, /repeating-conic-gradient/);
  assert.match(css, /--titlebar:/);
  assert.match(css, /"Tahoma", "MS Sans Serif"/);
  assert.match(dashboardCss, /--bg:\s*#008080;/);
  assert.match(dashboardCss, /--accent:\s*#000080;/);
  assert.match(dashboardCss, /repeating-conic-gradient/);
  assert.match(html, /<meta name="theme-color" content="#008080">/);
});

test('assistant image generation is a tool call with user-preset parameters', () => {
  assert.match(app, /const IMAGE_TOOL_NAME = 'generate_image'/);
  assert.match(app, /function makeImageToolDefinition/);
  assert.match(app, /tools: includeTools \? \[makeImageToolDefinition\(\)\] : undefined/);
  assert.match(app, /function getImageGenerationSettings/);
  assert.match(app, /function runImageToolCalls/);
  assert.match(app, /function executeImageToolCall/);
  assert.match(app, /kind: 'tool-result'/);
  assert.match(app, /tool_call_id: message\.toolCallId/);
  assert.match(app, /toolUnsupportedModels/);
  assert.match(html, /id="imageToolToggle"/);
  assert.match(html, /id="imageToolStatus"/);
  assert.match(serverJs, /normalizeChatTools/);
  assert.match(serverJs, /normalizeToolCalls/);
  assert.match(serverJs, /tool_choice: 'auto'/);

  const toolDefinition = app.slice(app.indexOf('function makeImageToolDefinition'), app.indexOf('function canUseImageTool'));
  assert.match(toolDefinition, /required: \['prompt'\]/);
  assert.doesNotMatch(toolDefinition, /steps|cfg|lora|seed|negative/i);
});

test('image generation is model-selectable and keeps image blobs in browser storage', () => {
  assert.match(html, /id="imageDialog"/);
  assert.match(html, /id="imageKindSelect"/);
  assert.match(html, /<option value="anima">Anima<\/option>/);
  assert.match(html, /<option value="sdxl">SDXL<\/option>/);
  assert.match(app, /apiFetch\('\/api\/image\/config'/);
  assert.match(app, /apiFetch\('\/api\/image\/generate'/);
  assert.match(app, /indexedDB\.open\(IMAGE_DATABASE_NAME, 1\)/);
  assert.match(app, /createObjectURL\(blob\)/);
  assert.match(app, /deleteImageBlob\(message\.imageId\)/);
  assert.match(css, /\.generated-image-frame/);
});

test('image model switches preserve user tuning and gallery images are conversation-scoped', () => {
  assert.doesNotMatch(app, /imageModelSelect\.addEventListener\('change',[\s\S]{0,220}applyImageRecommendation\(true\)/);
  assert.match(app, /imageModelSelect\.addEventListener\('change',[\s\S]{0,220}applyImageRecommendation\(false\)/);
  assert.match(html, /id="imageGalleryDialog"/);
  assert.match(html, /id="imageGalleryViewport"/);
  assert.match(app, /function openImageGallery\(messageId\)/);
  assert.match(app, /getActiveConversation\(\)\.messages\.filter/);
  assert.match(app, /frame\.addEventListener\('click', \(\) => openImageGallery\(message\.id\)\)/);
  assert.match(css, /\.gallery-viewport\s*{[^}]*overflow-y:\s*auto/s);
  assert.match(css, /\.gallery-item\s*{[^}]*scroll-snap-align:\s*start/s);
});

test('auto negative interleaves the positive prompt at a persisted selectable interval', () => {
  assert.match(html, /id="autoNegativeEverySelect"/);
  assert.match(html, /id="autoNegativeButton"/);
  assert.match(app, /autoNegativeEvery:\s*4/);
  assert.match(app, /bad, worst, awful, terrible/);
  assert.match(app, /atrocious, miserable, lousy/);
  const sources = [
    extractFunctionSource(app, 'interleaveAutoNegative'),
    extractFunctionSource(app, 'clampRandom'),
    extractFunctionSource(app, 'normalizeAutoNegativeEvery'),
  ].join('\n');
  const makeNegative = vm.runInNewContext(`(() => {
    const DEFAULT_IMAGE_SETTINGS = { autoNegativeEvery: 4 };
    const AUTO_NEGATIVE_TERMS = ['bad', 'boring', 'lousy'];
    ${sources}
    return interleaveAutoNegative;
  })()`);
  assert.equal(
    makeNegative('a cool guy riding a bike', 2, () => 0),
    'bad a cool bad guy riding bad a bike bad',
  );
  assert.equal(makeNegative('single', 10, () => 0.999), 'single lousy');
});

test('image generation preserves the message viewport through rerenders and decoded image replacement', () => {
  const generateImage = extractFunctionSource(app, 'generateImage');
  const hydrateGeneratedImage = extractFunctionSource(app, 'hydrateGeneratedImage');
  assert.match(generateImage, /persistAndRender\(\{ messages: \{ preserveScroll: true \} \}\)/);
  assert.doesNotMatch(generateImage, /scrollToBottom/);
  assert.match(hydrateGeneratedImage, /await loadStoredImageElement/);
  assert.match(hydrateGeneratedImage, /captureMessagesViewport\(\)/);
  assert.match(hydrateGeneratedImage, /restoreMessagesViewport\(viewportSnapshot\)/);
});

test('steps, CFG, and compatible LoRAs are first-class image controls', () => {
  const imageDialog = html.slice(html.indexOf('<dialog class="sheet-dialog image-dialog"'), html.indexOf('<dialog class="action-dialog"'));
  assert.match(html, /id="imageStepsInput"/);
  assert.match(html, /id="imageCfgInput"/);
  assert.ok(imageDialog.indexOf('id="imageStepsInput"') < imageDialog.indexOf('<details class="advanced-panel">'));
  assert.ok(imageDialog.indexOf('id="imageCfgInput"') < imageDialog.indexOf('<details class="advanced-panel">'));
  assert.match(html, /id="imageLoraList"/);
  assert.match(html, /id="addImageLoraButton"/);
  assert.match(app, /lorasByKind/);
  assert.match(app, /LoRA strength must be between -2 and 2/);
  assert.match(serverJs, /discoverImageLoras/);
  assert.match(serverJs, /normalizeLoraSelections/);
  assert.match(loraEngine, /def apply_loras/);
  assert.match(loraEngine, /def openclip_kohya_targets/);
});

test('CFG is unrestricted but finite, and sampler choices are family-specific and persisted', () => {
  const cfgTag = html.match(/<input\b[^>]*id="imageCfgInput"[^>]*>/)?.[0] || '';
  assert.match(cfgTag, /step="any"/);
  assert.doesNotMatch(cfgTag, /\b(?:min|max)=/);
  assert.match(html, /id="imageSamplerSelect"/);
  assert.match(html, /id="imageSamplerHint"/);
  assert.match(app, /samplerByKind:\s*\{\s*anima:\s*'flow_euler',\s*sdxl:\s*'dpmpp_sde_karras'/);
  assert.match(app, /DPM\+\+ SDE Karras - single-step history/);
  assert.match(app, /DPM\+\+ 2M Karras - multistep/);
  assert.match(extractFunctionSource(app, 'generateImage'), /CFG must be a finite number/);
  assert.doesNotMatch(extractFunctionSource(app, 'generateImage'), /cfg\s*[<>]=?\s*20|20\s*[<>]=?\s*cfg/);
  assert.match(serverJs, /const cfg = normalizeFiniteNumber\(payload\.cfg/);
  assert.match(serverJs, /const sampler = normalizeImageSampler\(kind, payload\.sampler\)/);
  assert.match(imageWorker, /cfg = _finite_number\(payload\.get\("cfg"\), "CFG"\)/);
  assert.match(imageWorker, /sampler = _sampler\(payload\.get\("sampler"\), kind\)/);
});

test('single-step-history samplers do not retain prior denoised estimates', () => {
  const sdeStart = sdxlEngine.indexOf('elif sampler == "dpmpp_sde_karras"');
  const multistepStart = sdxlEngine.indexOf('elif sampler == "dpmpp_2m_karras"', sdeStart);
  const samplerEnd = sdxlEngine.indexOf('else:', multistepStart);
  assert.ok(sdeStart > 0 && multistepStart > sdeStart && samplerEnd > multistepStart);
  assert.doesNotMatch(sdxlEngine.slice(sdeStart, multistepStart), /previous_denoised|previous_time/);
  assert.match(sdxlEngine.slice(multistepStart, samplerEnd), /previous_denoised/);
  assert.match(animaEngine, /sampler == "flow_heun"/);
  assert.doesNotMatch(animaEngine, /previous_(?:velocity|denoised)/);
});

test('image generation uses a bespoke warm CUDA worker with no inference framework dependency', () => {
  assert.equal(packageJson.name, 'local-ai-serving');
  assert.equal(packageJson.scripts.configure, 'node scripts/configure.js');
  assert.equal(packageJson.scripts['test:privacy'], 'node scripts/check-publish.js');
  assert.equal(packageJson.scripts['test:privacy:history'], 'node scripts/check-publish.js --fresh-history');
  assert.equal(packageJson.scripts['setup:image'], 'node scripts/fetch-image-assets.js');
  assert.equal(packageJson.scripts['test:models'], 'node scripts/validate-image-models.js');
  assert.equal(packageJson.scripts['test:smoke'], 'node scripts/smoke-hardware.js');
  assert.equal(packageJson.scripts['test:smoke:all'], 'node scripts/smoke-hardware.js --all-text-models');
  assert.equal(packageJson.scripts['test:browser'], 'node scripts/smoke-browser.js');
  assert.equal(packageJson.scripts['test:browser:optional'], 'node scripts/smoke-browser.js --optional');
  assert.equal(packageJson.scripts['test:regression'], 'npm test && npm run test:models');
  assert.equal(
    packageJson.scripts['test:regression:full'],
    'npm run test:regression && npm run test:browser && npm run test:smoke',
  );
  assert.equal(
    packageJson.scripts['test:regression:all-models'],
    'npm run test:regression && npm run test:browser && npm run test:smoke:all',
  );
  assert.match(serverJs, /spawn\(IMAGE_PYTHON, args/);
  assert.match(serverJs, /IMAGE_WORKER_IDLE_MS/);
  assert.match(serverJs, /runPersistentImageWorker/);
  assert.match(serverJs, /PYTHONDONTWRITEBYTECODE: '1'/);
  assert.match(serverJs, /Another image is already generating on this GPU/);
  assert.doesNotMatch(serverJs, /COMFYUI_BASE_URL|fetchComfy|makeAnimaWorkflow|makeSdxlWorkflow/);
  assert.match(imageWorker, /BytePairEncoder\.encode_piece\.cache_clear/);
  assert.match(imageRuntime, /class SafeTensorFile/);
  assert.match(imageRuntime, /def encode_png/);
  assert.match(animaEngine, /class AnimaDenoiser/);
  assert.match(animaEngine, /class AnimaSession/);
  assert.match(animaEngine, /class QwenEncoder/);
  assert.match(animaEngine, /ANIMA_MODEL_PREFIXES = \("net\.", "model\.diffusion_model\."\)/);
  assert.match(serverJs, /ANIMA_MODEL_SIGNATURE/);
  assert.match(serverJs, /isAnimaCheckpoint\(file\.path\)/);
  assert.doesNotMatch(serverJs, /filter\(\(file\) => \/anima\/i/);
  assert.match(sdxlEngine, /class SDXLUNet/);
  assert.match(sdxlEngine, /class SDXLImageDecoder/);
  assert.match(sdxlEngine, /class SDXLSession/);
  assert.match(sdxlEngine, /self\.denoiser\.clear_context\(\)/);
  assert.match(modelValidator, /parameter name and\s+shape/);
  assert.match(modelValidator, /detect_anima_prefix/);
  for (const source of [imageWorker, imageRuntime, animaEngine, sdxlEngine, loraEngine]) {
    assert.doesNotMatch(source, /(?:from|import)\s+(?:comfy|diffusers|transformers|safetensors|PIL|numpy|einops)\b/);
  }
  assert.match(imageSetup, /sha256/);
  assert.match(imageSetup, /Checksum mismatch/);
});

test('machine resources come only from ignored user configuration or explicit environment values', () => {
  assert.match(serverJs, /applyLocalConfig\(process\.env(?:,\s*\{[^}]+\})?\)/);
  assert.match(smokeSource, /applyLocalConfig\(process\.env\)/);
  assert.match(validatorLauncher, /applyLocalConfig\(process\.env\)/);
  assert.match(localConfigSource, /textServerExecutable/);
  assert.match(localConfigSource, /imageModelsRoot/);
  assert.match(configureSource, /Local configuration saved without printing private values/);
  assert.ok(Object.values(configExample).every((value) => value === 1 || value === ''));
  assert.match(gitignore, /^config\.local\.json$/m);
  assert.match(gitignore, /^\*\.gguf$/m);
  assert.match(gitignore, /^\*\.safetensors$/m);
  const providerLayoutPattern = new RegExp(
    ['\\.lm', 'studio|Documents[\\\\/]+Comfy', 'UI|LM_', 'STUDIO|local-lm-', 'studio'].join(''),
    'i',
  );
  for (const source of [serverJs, textBackendSource, smokeSource, validatorLauncher, readme]) {
    assert.doesNotMatch(source, providerLayoutPattern);
  }
  assert.match(serverJs, /ANIMA_TEXT_ENCODER_PATH/);
  assert.match(serverJs, /ANIMA_VAE_PATH/);
});

test('README distinguishes deterministic, structural-image, real-browser, and real-hardware tiers', () => {
  for (const command of [
    'npm test',
    'npm run test:models',
    'npm run test:browser',
    'npm run test:smoke',
    'npm run test:smoke:all',
    'npm run test:privacy',
    'npm run test:privacy:history',
    'npm run test:regression:all-models',
  ]) assert.match(readme, new RegExp(command.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.match(readme, /structurally audits every compatible checkpoint and adapter/);
  assert.match(readme, /cold and warm managed-text completions/i);
  assert.match(readme, /Image inference still runs once per family, not once per checkpoint/i);
});

test('privacy documentation states the application boundary without absolute guarantees', () => {
  assert.doesNotMatch(html, /only in memory|Private to this browser|LOCAL CUDA|Direct CUDA is ready/i);
  assert.doesNotMatch(app, /Working locally|Thinking locally|Generating on your GPU|Generating on this computer/);
  assert.doesNotMatch(html, /COMFYUI/);
  assert.match(readme, /operating-system swap, crash dumps/i);
  assert.match(readme, /Web origin isolation normally prevents/i);
  assert.match(readme, /outside the application boundary/i);
  assert.match(readme, /Default server, smoke, and structural-audit output uses anonymous aliases or ordinals/i);
  assert.doesNotMatch(readme, /RTX[ -]?[0-9]|Core i[3579]-|GeForce RTX/i);
  assert.match(serverJs, /Managed text backend is configured \(loads on demand\)/);
  assert.doesNotMatch(serverJs, /Managed text backend:.*\.alias/);
  assert.match(modelValidator, /MODEL_AUDIT_SHOW_IDENTIFIERS/);
  assert.match(readme, /operating-system clipboard/);
  assert.match(readme, /Export intentionally writes selected conversation data outside browser storage/);
});

test('secondary conversation actions do not permanently consume composer space', () => {
  const composer = html.slice(html.indexOf('<footer class="composer">'), html.indexOf('</footer>') + 9);
  assert.doesNotMatch(composer, /regenerateButton|exportButton|clearButton/);
  assert.match(html, /id="conversationActionsDialog"/);
  assert.match(html, /id="conversationMenuButton"/);
  assert.match(html, /id="renameButton"/);
  assert.match(html, /<h1 id="conversationTitle"/);
  assert.doesNotMatch(html, /<input id="conversationTitle"/);
});

test('coarse pointers use an explicit Send button instead of hijacking the return key', () => {
  assert.match(app, /window\.matchMedia\('\(pointer: fine\)'\)\.matches/);
  assert.match(app, /window\.matchMedia\('\(pointer: coarse\)'\)\.matches/);
  assert.match(html, /id="sendButton"/);
});

test('managed model switches expose semantic loading progress and observable readiness', () => {
  assert.match(html, /<progress id="modelLoadingProgress"/);
  assert.match(html, /id="modelLoading"[^>]*role="status"[^>]*aria-live="polite"/);
  assert.match(css, /\.model-loading\[hidden\]/);
  assert.match(app, /apiFetch\('\/api\/text\/load'/);
  assert.match(app, /apiFetch\('\/api\/text\/status'/);
  assert.match(app, /modelLoadingProgress\.removeAttribute\('value'\)/);
  assert.match(app, /void warmSelectedTextModel\(\)/);
});

test('UI files stay ASCII for broad local-device compatibility', () => {
  for (const [filePath, contents] of [
    ['public/index.html', html],
    ['public/styles.css', css],
    ['public/app.js', app],
    ['public/dashboard.html', dashboardHtml],
    ['public/dashboard.css', dashboardCss],
    ['public/dashboard.js', dashboardJs],
    ['scripts/start.js', startJs],
    ['scripts/tray.ps1', trayPs1],
    ['Start Local LLM Serve.cmd', startCmd],
    ['Start Local LLM Serve Tray.cmd', trayCmd],
  ]) {
    assert.equal(/[\u0080-\uFFFF]/.test(contents), false, `${filePath} contains non-ASCII characters`);
  }
});

test('access dashboard generates tokenized QR links locally', () => {
  assert.match(dashboardHtml, /id="qrCode"/);
  assert.match(dashboardHtml, /id="accessUrl"/);
  assert.match(dashboardJs, /fetch\('\/api\/access-info'/);
  assert.match(dashboardJs, /#access=/);
  assert.match(dashboardJs, /function makeQrMatrix/);
  assert.match(dashboardJs, /function reedSolomon/);
});

test('host dashboard configures local text folders through guarded server APIs', () => {
  assert.match(dashboardHtml, /id="models"/);
  assert.match(dashboardHtml, /id="textModelsRootInput"/);
  assert.match(dashboardHtml, /id="pickTextModelsRootButton"/);
  assert.match(dashboardHtml, /id="saveTextModelsRootButton"/);
  assert.match(dashboardHtml, /id="textModelList"/);
  assert.match(dashboardJs, /fetch\('\/api\/local-setup'/);
  assert.match(dashboardJs, /fetch\('\/api\/local-setup\/pick-folder'/);
  assert.match(dashboardJs, /fetch\('\/api\/local-setup\/text-folder'/);
  assert.match(dashboardJs, /fetch\('\/api\/local-setup\/refresh-text-models'/);
  assert.match(dashboardJs, /refreshModelSetupButton\.addEventListener\('click', refreshLocalSetup\)/);
  assert.match(dashboardJs, /'X-Local-Setup': '1'/);
  assert.match(dashboardJs, /textFolderLocked/);
  assert.match(dashboardJs, /if \(!locked\) setSetupMessage\(''\)/);
  assert.match(dashboardJs, /text\.managedEnabled/);
  assert.match(serverJs, /function assertLocalSetupMutationRequest/);
  assert.match(serverJs, /origin !== expectedOrigin/);
  assert.match(serverJs, /activeFolderPicker/);
});

test('server advertises phone access through the tokenized dashboard, not raw LAN URLs', () => {
  assert.match(serverJs, /LAN access: open the local dashboard and scan its current QR code/);
  assert.doesNotMatch(serverJs, /console\.log\(`LAN:/);
  assert.match(serverJs, /accessUrls: getLanAddresses\(\)\.map\(\(address\) => makeAccessUrl\(address\)\)/);
});

test('README explains certificate warnings and trusted certificate options', () => {
  assert.match(readme, /connection is not private/);
  assert.match(readme, /TLS_CERT_FILE/);
  assert.match(readme, /TLS_KEY_FILE/);
  assert.match(readme, /HTTPS=0/);
  assert.match(readme, /only localhost and loopback identities/);
  assert.match(readme, /Do not port-forward this service/);
  assert.match(readme, /DNS rebinding cannot change the destination/);
  assert.match(readme, /tray shortcuts are optional Windows conveniences/);
});

function extractFunctionSource(source, name) {
  const start = source.indexOf(`function ${name}`);
  assert.notEqual(start, -1, `${name} not found`);
  const signatureEnd = source.indexOf(')', start);
  assert.notEqual(signatureEnd, -1, `${name} has no complete signature`);
  const braceStart = source.indexOf('{', signatureEnd);
  assert.notEqual(braceStart, -1, `${name} has no body`);

  let depth = 0;
  for (let index = braceStart; index < source.length; index += 1) {
    const character = source[index];
    if (character === '{') {
      depth += 1;
    } else if (character === '}') {
      depth -= 1;
      if (depth === 0) {
        return source.slice(start, index + 1);
      }
    }
  }

  throw new Error(`${name} body is incomplete`);
}
