'use strict';

const STORAGE_KEY = 'local-ai-serving-state-v1';
const ACCESS_TOKEN_STORAGE_KEY = 'local-ai-serving-access-token-v1';
const ACCESS_REQUIRED_MESSAGE = 'Access required. Scan the current QR code from the local dashboard.';
const ACCESS_REQUIRED_OPTION_TEXT = 'Access required - scan the current QR code';
const IMAGE_DATABASE_NAME = 'local-chat-private-images-v1';
const IMAGE_STORE_NAME = 'images';
const DEFAULT_SETTINGS = {
  baseUrl: '',
  baseUrlOverride: false,
  apiKey: '',
  model: '',
  modelThinkingModes: {},
  systemPrompt: 'You are a helpful assistant.',
  temperature: 0.95,
  max_tokens: 2048,
};
const DEFAULT_IMAGE_SETTINGS = {
  kind: 'anima',
  modelByKind: { anima: '', sdxl: '' },
  stepsByKind: { anima: '', sdxl: '' },
  cfgByKind: { anima: '', sdxl: '' },
  samplerByKind: { anima: 'flow_euler', sdxl: 'dpmpp_sde_karras' },
  lorasByKind: { anima: [], sdxl: [] },
  size: 'portrait',
  negativePrompt: '',
  autoNegativeEvery: 4,
  toolEnabled: true,
};
const IMAGE_SAMPLERS = Object.freeze({
  anima: Object.freeze([
    Object.freeze({ id: 'flow_euler', label: 'Flow Euler - single-step history', hint: 'Native, fast, and well suited to very low step counts.' }),
    Object.freeze({ id: 'flow_heun', label: 'Flow Heun - single-step history', hint: 'Two evaluations per step for a corrected flow update.' }),
  ]),
  sdxl: Object.freeze([
    Object.freeze({ id: 'dpmpp_sde_karras', label: 'DPM++ SDE Karras - single-step history', hint: 'Stochastic two-stage DPM++ SDE; the default low-step SDXL choice.' }),
    Object.freeze({ id: 'euler_ancestral_karras', label: 'Euler ancestral Karras - single-step history', hint: 'Fast stochastic Euler updates without multistep memory.' }),
    Object.freeze({ id: 'euler_karras', label: 'Euler Karras - single-step history', hint: 'Fast deterministic Euler updates without multistep memory.' }),
    Object.freeze({ id: 'dpmpp_2m_karras', label: 'DPM++ 2M Karras - multistep', hint: 'Uses the previous denoised estimate; retained for comparison.' }),
  ]),
});
const AUTO_NEGATIVE_TERMS = 'bad, worst, awful, terrible, horrible, inferior, undesirable, ineffective, unconvincing, unexceptional, mediocre, disappointing, displeasing, uninspiring, forgettable, lackluster, subpar, inadequate, unenjoyable, shallow, stale, second-rate, disappointing, unimpressive, mundane, inept, low-quality, unenjoyable, pointless, boring, tedious, drab, dull, vapid, trivial, insignificant, negligible, paltry, unremarkable, unfulfilling, unsatisfying, unsolid, uninsightful, unrefreshing, ungratifying, unmasterful, uncompelling, unfavorable, poor, dreadful, abysmal, dismal, pathetic, pitiful, rubbish, shoddy, unsatisfactory, atrocious, miserable, lousy'
  .split(',')
  .map((term) => term.trim());
const THINKING_MODE_VALUES = new Set(['auto', 'on', 'off']);
const IMAGE_KIND_VALUES = new Set(['anima', 'sdxl']);
const IMAGE_SIZE_VALUES = new Set(['portrait', 'square', 'landscape']);
const IMAGE_DIMENSIONS = Object.freeze({
  portrait: Object.freeze({ width: 832, height: 1216 }),
  square: Object.freeze({ width: 1024, height: 1024 }),
  landscape: Object.freeze({ width: 1216, height: 832 }),
});
const IMAGE_TOOL_NAME = 'generate_image';
const VIEWPORT_SCROLL_KEYS = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' ']);
const MAX_TOOL_ROUNDS = 2;
const MODEL_STATUS_POLL_MS = 200;
const MODEL_READY_VISIBLE_MS = 900;

const state = loadState();
let accessToken = consumeAccessToken();
let accessIsRequired = false;
let managedTextBackendEnabled = false;
let serverDefaultBaseUrl = '';
let baseUrlEdited = false;
let activeController = null;
let imageAbortController = null;
let imageGenerationInFlight = false;
let modelLoadInFlight = false;
let modelLoadPromise = null;
let modelLoadSequence = 0;
let messagesScrollIntentVersion = 0;
let pendingMessagesViewportRestore = null;
let streamingMessageId = null;
let streamingFrame = null;
const toolUnsupportedModels = new Set();
const pendingStreamingMessages = new Map();
const thinkingPanelOpenByMessage = new Map();
const sessionImageBlobs = new Map();
const imageObjectUrls = new Map();
let imageConfig = {
  loaded: false,
  connected: false,
  models: { anima: [], sdxl: [] },
  loras: { anima: [], sdxl: [] },
  runtime: {},
  dependencies: { animaTextEncoder: false, animaVae: false },
};

const els = {
  sidebar: document.getElementById('sidebar'),
  sidebarOverlay: document.getElementById('sidebarOverlay'),
  openSidebarButton: document.getElementById('openSidebarButton'),
  closeSidebarButton: document.getElementById('closeSidebarButton'),
  newChatButton: document.getElementById('newChatButton'),
  conversationList: document.getElementById('conversationList'),
  conversationTitle: document.getElementById('conversationTitle'),
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),
  modelLoading: document.getElementById('modelLoading'),
  modelLoadingText: document.getElementById('modelLoadingText'),
  modelLoadingProgress: document.getElementById('modelLoadingProgress'),
  accessRequired: document.getElementById('accessRequired'),
  chatSetupButton: document.getElementById('chatSetupButton'),
  modelSummary: document.getElementById('modelSummary'),
  conversationMenuButton: document.getElementById('conversationMenuButton'),
  renameButton: document.getElementById('renameButton'),
  modelSelect: document.getElementById('modelSelect'),
  localModelSetupLink: document.getElementById('localModelSetupLink'),
  localModelSetupGuidance: document.getElementById('localModelSetupGuidance'),
  thinkingModeSelect: document.getElementById('thinkingModeSelect'),
  refreshModelsButton: document.getElementById('refreshModelsButton'),
  messages: document.getElementById('messages'),
  regenerateButton: document.getElementById('regenerateButton'),
  exportButton: document.getElementById('exportButton'),
  clearButton: document.getElementById('clearButton'),
  imageButton: document.getElementById('imageButton'),
  promptInput: document.getElementById('promptInput'),
  sendButton: document.getElementById('sendButton'),
  stopButton: document.getElementById('stopButton'),
  hintText: document.getElementById('hintText'),
  charCount: document.getElementById('charCount'),
  settingsDialog: document.getElementById('settingsDialog'),
  baseUrlInput: document.getElementById('baseUrlInput'),
  apiKeyInput: document.getElementById('apiKeyInput'),
  systemPromptInput: document.getElementById('systemPromptInput'),
  temperatureInput: document.getElementById('temperatureInput'),
  temperatureOutput: document.getElementById('temperatureOutput'),
  maxTokensInput: document.getElementById('maxTokensInput'),
  testConnectionButton: document.getElementById('testConnectionButton'),
  resetSettingsButton: document.getElementById('resetSettingsButton'),
  saveSettingsButton: document.getElementById('saveSettingsButton'),
  imageDialog: document.getElementById('imageDialog'),
  imageKindSelect: document.getElementById('imageKindSelect'),
  imageModelSelect: document.getElementById('imageModelSelect'),
  imagePromptInput: document.getElementById('imagePromptInput'),
  negativePromptInput: document.getElementById('negativePromptInput'),
  autoNegativeEverySelect: document.getElementById('autoNegativeEverySelect'),
  autoNegativeButton: document.getElementById('autoNegativeButton'),
  imageSizeSelect: document.getElementById('imageSizeSelect'),
  imageSamplerSelect: document.getElementById('imageSamplerSelect'),
  imageSamplerHint: document.getElementById('imageSamplerHint'),
  imageStepsInput: document.getElementById('imageStepsInput'),
  imageCfgInput: document.getElementById('imageCfgInput'),
  imageSeedInput: document.getElementById('imageSeedInput'),
  imageConnectionText: document.getElementById('imageConnectionText'),
  imageToolToggle: document.getElementById('imageToolToggle'),
  imageToolStatus: document.getElementById('imageToolStatus'),
  imageLoraList: document.getElementById('imageLoraList'),
  imageLoraHint: document.getElementById('imageLoraHint'),
  addImageLoraButton: document.getElementById('addImageLoraButton'),
  imageFormError: document.getElementById('imageFormError'),
  refreshImageModelsButton: document.getElementById('refreshImageModelsButton'),
  generateImageButton: document.getElementById('generateImageButton'),
  imageGalleryDialog: document.getElementById('imageGalleryDialog'),
  imageGalleryViewport: document.getElementById('imageGalleryViewport'),
  imageGalleryCount: document.getElementById('imageGalleryCount'),
  closeImageGalleryButton: document.getElementById('closeImageGalleryButton'),
  conversationActionsDialog: document.getElementById('conversationActionsDialog'),
};

initialize();

async function initialize() {
  if (!state.conversations.length) {
    const conversation = createConversation();
    state.conversations.push(conversation);
    state.activeConversationId = conversation.id;
  }

  wireEvents();
  applySettingsToForm();
  applyImageSettingsToForm();
  updateComposerHint();
  render();
  await Promise.all([loadServerConfig(), refreshImageConfig()]);
  if (!accessIsRequired) await refreshModels();
}

function wireEvents() {
  els.openSidebarButton.addEventListener('click', openSidebar);
  els.closeSidebarButton.addEventListener('click', closeSidebar);
  els.sidebarOverlay.addEventListener('click', closeSidebar);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeSidebar();
    const target = event.target;
    const isEditing = target instanceof HTMLInputElement
      || target instanceof HTMLTextAreaElement
      || target instanceof HTMLSelectElement
      || target?.isContentEditable;
    if (!isEditing && VIEWPORT_SCROLL_KEYS.has(event.key)) markMessagesScrollIntent();
  });
  for (const eventName of ['wheel', 'touchstart', 'touchmove', 'pointerdown']) {
    els.messages.addEventListener(eventName, markMessagesScrollIntent, { passive: true });
  }

  els.newChatButton.addEventListener('click', () => {
    if (isBusy()) return;
    const conversation = createConversation();
    state.conversations.unshift(conversation);
    state.activeConversationId = conversation.id;
    persistAndRender();
    closeSidebar();
    els.promptInput.focus();
  });

  els.chatSetupButton.addEventListener('click', openSettings);
  els.conversationMenuButton.addEventListener('click', () => els.conversationActionsDialog.showModal());
  els.renameButton.addEventListener('click', renameConversation);

  els.modelSelect.addEventListener('change', () => {
    state.settings.model = els.modelSelect.value;
    applyModelThinkingModeToForm();
    updateModelSummary();
    saveState();
    void warmSelectedTextModel();
  });
  els.thinkingModeSelect.addEventListener('change', () => {
    setSelectedModelThinkingMode(els.thinkingModeSelect.value);
    saveState();
  });

  els.refreshModelsButton.addEventListener('click', refreshModels);
  els.regenerateButton.addEventListener('click', () => {
    els.conversationActionsDialog.close();
    regenerateLastAssistantMessage();
  });
  els.exportButton.addEventListener('click', () => {
    els.conversationActionsDialog.close();
    exportConversation();
  });
  els.clearButton.addEventListener('click', () => {
    els.conversationActionsDialog.close();
    clearConversation();
  });

  els.imageButton.addEventListener('click', openImageDialog);
  els.sendButton.addEventListener('click', sendPrompt);
  els.stopButton.addEventListener('click', stopStreaming);
  els.promptInput.addEventListener('input', () => {
    autosizePrompt();
    els.charCount.textContent = String(els.promptInput.value.length);
  });
  els.promptInput.addEventListener('keydown', (event) => {
    const keyboardSends = window.matchMedia('(pointer: fine)').matches;
    if (keyboardSends && event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendPrompt();
    }
  });

  els.temperatureInput.addEventListener('input', () => {
    els.temperatureOutput.textContent = Number(els.temperatureInput.value).toFixed(2);
  });
  els.baseUrlInput.addEventListener('input', () => {
    baseUrlEdited = true;
  });
  els.testConnectionButton.addEventListener('click', refreshModels);
  els.resetSettingsButton.addEventListener('click', () => {
    state.settings = normalizeSettings();
    applySettingsToForm();
    saveState();
    refreshModels();
  });
  els.settingsDialog.addEventListener('close', () => {
    if (els.settingsDialog.returnValue !== 'cancel') saveSettingsFromForm();
  });

  els.imageKindSelect.addEventListener('change', () => {
    state.image.kind = normalizeImageKind(els.imageKindSelect.value);
    applyImageKindSettings();
    renderImageSamplerOptions();
    renderImageModelOptions();
    saveState();
  });
  els.imageModelSelect.addEventListener('change', () => {
    state.image.modelByKind[state.image.kind] = els.imageModelSelect.value;
    applyImageRecommendation(false);
    saveState();
  });
  els.imageSizeSelect.addEventListener('change', () => {
    state.image.size = normalizeImageSize(els.imageSizeSelect.value);
    saveState();
  });
  els.imageStepsInput.addEventListener('input', saveImageParametersFromForm);
  els.imageCfgInput.addEventListener('input', saveImageParametersFromForm);
  els.negativePromptInput.addEventListener('input', () => {
    state.image.negativePrompt = els.negativePromptInput.value.slice(0, 3000);
    saveState();
  });
  els.imageSamplerSelect.addEventListener('change', () => {
    state.image.samplerByKind[state.image.kind] = normalizeImageSampler(state.image.kind, els.imageSamplerSelect.value);
    updateImageSamplerHint();
    saveState();
  });
  els.autoNegativeEverySelect.addEventListener('change', () => {
    state.image.autoNegativeEvery = normalizeAutoNegativeEvery(els.autoNegativeEverySelect.value);
    saveState();
  });
  els.autoNegativeButton.addEventListener('click', generateAutoNegativePrompt);
  els.imageToolToggle.addEventListener('change', () => {
    state.image.toolEnabled = els.imageToolToggle.checked;
    saveState();
    updateImageToolStatus();
  });
  els.addImageLoraButton.addEventListener('click', () => addImageLoraRow());
  els.refreshImageModelsButton.addEventListener('click', refreshImageConfig);
  els.generateImageButton.addEventListener('click', generateImage);
  els.closeImageGalleryButton.addEventListener('click', () => els.imageGalleryDialog.close());
}

function updateComposerHint() {
  els.hintText.textContent = window.matchMedia('(pointer: coarse)').matches
    ? 'Tap Send when your message is ready'
    : 'Enter to send, Shift+Enter for a new line';
}

function openSidebar() {
  els.sidebar.classList.add('open');
  document.body.classList.add('sidebar-is-open');
}

function closeSidebar() {
  els.sidebar.classList.remove('open');
  document.body.classList.remove('sidebar-is-open');
}

async function loadServerConfig() {
  try {
    const response = await apiFetch('/api/config', { cache: 'no-store' });
    if (!response.ok) return;
    const config = await response.json();
    managedTextBackendEnabled = Boolean(config.managedTextBackend?.enabled);
    serverDefaultBaseUrl = typeof config.defaultBaseUrl === 'string' ? config.defaultBaseUrl.trim() : '';
    renderLocalModelSetup(config.localSetupAvailable === true);
    const migrateManagedDefault = managedTextBackendEnabled && !state.settings.baseUrlOverride;
    if (serverDefaultBaseUrl && (!safeStorageGet(STORAGE_KEY) || migrateManagedDefault)) {
      state.settings.baseUrl = serverDefaultBaseUrl;
      state.settings.baseUrlOverride = false;
      applySettingsToForm();
      saveState();
    }
  } catch (error) {
    if (isAccessRequiredError(error)) return;
    setStatus('error', 'Could not read server settings');
  }
}

function renderLocalModelSetup(localSetupAvailable) {
  els.localModelSetupLink.hidden = !localSetupAvailable;
  els.localModelSetupGuidance.hidden = localSetupAvailable;
}

async function refreshModels() {
  saveSettingsFromForm();
  setStatus('', 'Checking text backend');
  els.refreshModelsButton.disabled = true;
  els.testConnectionButton.disabled = true;

  try {
    const response = await apiFetch('/api/models', {
      method: 'POST',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        baseUrl: state.settings.baseUrl,
        apiKey: state.settings.apiKey,
      }),
    });
    const data = await parseJsonResponse(response);
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);

    const models = Array.isArray(data.data) ? data.data.map((model) => model.id).filter(Boolean) : [];
    renderModelOptions(models);
    setStatus('connected', models.length ? `${models.length} model${models.length === 1 ? '' : 's'} ready` : 'Text backend connected, no models loaded');
  } catch (error) {
    renderModelOptions([]);
    setStatus('error', isAccessRequiredError(error) ? ACCESS_REQUIRED_MESSAGE : error.message || 'Could not connect to the text backend');
  } finally {
    els.refreshModelsButton.disabled = accessIsRequired;
    els.testConnectionButton.disabled = accessIsRequired;
  }
}

async function sendPrompt() {
  const content = els.promptInput.value.trim();
  if (!content || isBusy()) return;
  if (!state.settings.model) {
    setStatus('error', 'Choose a chat model first');
    openSettings();
    return;
  }

  const conversation = getActiveConversation();
  conversation.messages.push(makeMessage('user', content));
  if (conversation.title === 'New chat') conversation.title = titleFromPrompt(content);
  conversation.updatedAt = Date.now();
  els.promptInput.value = '';
  autosizePrompt();
  els.charCount.textContent = '0';
  persistAndRender({ messages: { scrollToBottom: true } });
  await streamAssistantReply(conversation, { revealNewMessage: true });
}

async function regenerateLastAssistantMessage() {
  if (isBusy()) return;
  const conversation = getActiveConversation();
  const lastAssistantIndex = findLastAssistantIndex(conversation.messages);
  if (lastAssistantIndex === -1) return;
  const removed = conversation.messages.splice(lastAssistantIndex, 1);
  deleteImageBlobsForMessages(removed);
  conversation.updatedAt = Date.now();
  persistAndRender();
  await streamAssistantReply(conversation);
}

async function streamAssistantReply(conversation, options = {}) {
  if (!state.settings.model) {
    setStatus('error', 'Choose a chat model first');
    return;
  }
  if (!await warmSelectedTextModel()) return;

  const toolDepth = options.toolDepth || 0;
  const assistantMessage = makeMessage('assistant', '');
  conversation.messages.push(assistantMessage);
  streamingMessageId = assistantMessage.id;
  activeController = new AbortController();
  setBusy();
  setStatus('connected', `Streaming from ${state.settings.model}`);
  persistAndRender({ messages: { scrollToBottom: Boolean(options.revealNewMessage) } });

  let pendingToolCalls = [];
  let aborted = false;
  let includeTools = toolDepth < MAX_TOOL_ROUNDS
    && canUseImageTool()
    && !toolUnsupportedModels.has(state.settings.model);

  try {
    while (true) {
      try {
        pendingToolCalls = await streamChatCompletion(conversation, assistantMessage, includeTools);
        break;
      } catch (error) {
        const nothingReceived = !assistantMessage.content && !assistantMessage.reasoning;
        if (includeTools && nothingReceived && error.name !== 'AbortError' && isToolsUnsupportedError(error)) {
          toolUnsupportedModels.add(state.settings.model);
          includeTools = false;
          continue;
        }
        throw error;
      }
    }
    assistantMessage.updatedAt = Date.now();
    conversation.updatedAt = Date.now();
    setStatus('connected', `Ready - ${state.settings.model}`);
  } catch (error) {
    if (error.name === 'AbortError') {
      aborted = true;
      setStatus('', 'Generation stopped');
    } else {
      assistantMessage.content = assistantMessage.content || `Error: ${error.message}`;
      setStatus('error', error.message || 'Chat request failed');
    }
  } finally {
    flushStreamingMessages();
    activeController = null;
    streamingMessageId = null;
    setBusy();
    persistAndRender({ messages: { preserveScroll: true } });
  }

  if (!aborted && pendingToolCalls.length) {
    await runImageToolCalls(conversation, pendingToolCalls, toolDepth);
  }
}

async function warmSelectedTextModel() {
  if (!usesManagedTextBackend() || !state.settings.model || accessIsRequired) return true;
  if (modelLoadPromise) return modelLoadPromise;

  const selectedModel = state.settings.model;
  const selectedBaseUrl = state.settings.baseUrl;
  const sequence = ++modelLoadSequence;
  const operation = performManagedTextModelLoad(selectedModel, selectedBaseUrl, sequence);
  modelLoadPromise = operation;
  try {
    return await operation;
  } finally {
    if (modelLoadPromise === operation) modelLoadPromise = null;
  }
}

async function performManagedTextModelLoad(selectedModel, selectedBaseUrl, sequence) {
  const controller = new AbortController();
  modelLoadInFlight = true;
  renderModelLoading({ state: 'loading', phase: 'starting' });
  setStatus('', 'Loading selected text model');
  setBusy();
  const polling = pollTextModelStatus(sequence, controller.signal);

  try {
    const response = await apiFetch('/api/text/load', {
      method: 'POST',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ baseUrl: selectedBaseUrl, model: selectedModel }),
      signal: controller.signal,
    });
    const data = await parseJsonResponse(response);
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (!data.managed || data.state !== 'ready') {
      throw new Error('The selected text model did not report a ready state.');
    }
    renderModelLoading({ state: 'ready', phase: 'ready' });
    setStatus('connected', 'Selected text model is ready');
    setTimeout(() => {
      if (sequence === modelLoadSequence && !modelLoadInFlight) setModelLoadingHidden(true);
    }, MODEL_READY_VISIBLE_MS);
    return true;
  } catch (error) {
    if (error.name === 'AbortError') return false;
    renderModelLoading({ state: 'error', phase: 'error' });
    setStatus('error', error.message || 'The selected text model could not be loaded');
    return false;
  } finally {
    controller.abort();
    await polling;
    modelLoadInFlight = false;
    setBusy();
  }
}

async function pollTextModelStatus(sequence, signal) {
  while (!signal.aborted && sequence === modelLoadSequence) {
    try {
      await new Promise((resolve) => setTimeout(resolve, MODEL_STATUS_POLL_MS));
      if (signal.aborted) return;
      const response = await apiFetch('/api/text/status', { cache: 'no-store', signal });
      if (!response.ok) continue;
      const status = await parseJsonResponse(response);
      if (status.managed && sequence === modelLoadSequence) renderModelLoading(status);
    } catch (error) {
      if (error.name === 'AbortError') return;
    }
  }
}

function renderModelLoading(status) {
  const stateName = status?.state || 'idle';
  const phase = status?.phase || 'idle';
  if (stateName === 'idle' || stateName === 'disabled' || stateName === 'external') {
    setModelLoadingHidden(true);
    return;
  }

  setModelLoadingHidden(false);
  if (stateName === 'ready') {
    els.modelLoadingText.textContent = 'Ready';
    els.modelLoadingProgress.max = 1;
    els.modelLoadingProgress.value = 1;
    return;
  }
  if (stateName === 'error') {
    els.modelLoadingText.textContent = 'Could not load the selected model';
    els.modelLoadingProgress.max = 1;
    els.modelLoadingProgress.value = 0;
    return;
  }

  els.modelLoadingText.textContent = {
    stopping: 'Releasing the current model',
    starting: 'Starting the text engine',
    loading: 'Loading the selected model',
  }[phase] || 'Loading the selected model';
  els.modelLoadingProgress.removeAttribute('value');
}

function usesManagedTextBackend() {
  const selected = String(state.settings.baseUrl || '').replace(/\/+$/, '');
  const managed = String(serverDefaultBaseUrl || '').replace(/\/+$/, '');
  return managedTextBackendEnabled && Boolean(managed) && selected === managed;
}

async function streamChatCompletion(conversation, assistantMessage, includeTools) {
  const response = await apiFetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      baseUrl: state.settings.baseUrl,
      apiKey: state.settings.apiKey,
      model: state.settings.model,
      messages: buildTextBackendMessages(conversation),
      thinkingMode: getSelectedModelThinkingMode(),
      temperature: state.settings.temperature,
      max_tokens: state.settings.max_tokens,
      tools: includeTools ? [makeImageToolDefinition()] : undefined,
    }),
    signal: activeController.signal,
  });
  if (!response.ok || !response.body) {
    const errorData = await parseJsonResponse(response);
    throw new Error(errorData.error || `HTTP ${response.status}`);
  }

  const toolCallsByIndex = new Map();
  await readChatStream(response, (delta) => {
    if (delta.reasoning) assistantMessage.reasoning += delta.reasoning;
    if (delta.content) assistantMessage.content += delta.content;
    if (delta.toolCalls) mergeToolCallDeltas(toolCallsByIndex, delta.toolCalls);
    assistantMessage.updatedAt = Date.now();
    updateStreamingMessage(assistantMessage);
  });

  const toolCalls = finalizeToolCalls(toolCallsByIndex);
  if (toolCalls.length) assistantMessage.toolCalls = toolCalls;
  return toolCalls;
}

function isToolsUnsupportedError(error) {
  return /tool/i.test(String((error && error.message) || ''));
}

async function readChatStream(response, onDelta) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const thinkingParser = createThinkingParser();
  let buffer = '';
  let completed = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || '';
    for (const eventText of events) {
      if (parseChatEvent(eventText, thinkingParser, onDelta)) completed = true;
    }
  }

  if (buffer.trim() && parseChatEvent(buffer, thinkingParser, onDelta)) completed = true;
  if (!completed) throw new Error('The text stream ended before completion.');
  const finalDelta = thinkingParser.flush();
  if (finalDelta.content || finalDelta.reasoning) onDelta(finalDelta);
}

function parseChatEvent(eventText, thinkingParser, onDelta) {
  const lines = eventText.split(/\r?\n/);
  let completed = false;
  for (const line of lines) {
    if (!line.startsWith('data:')) continue;
    const data = line.slice(5).trim();
    if (!data) continue;
    if (data === '[DONE]') {
      completed = true;
      continue;
    }

    let parsed;
    try {
      parsed = JSON.parse(data);
    } catch {
      throw new Error('The text backend returned an invalid stream event.');
    }
    if (parsed.error) throw new Error(typeof parsed.error === 'string' ? parsed.error : parsed.error.message || 'Stream error');
    const choice = parsed.choices && parsed.choices[0];
    const delta = choice && choice.delta ? choice.delta : {};
    const reasoningDelta = getReasoningDelta(delta);
    if (reasoningDelta) onDelta({ content: '', reasoning: reasoningDelta });
    if (typeof delta.content === 'string') {
      const parsedDelta = thinkingParser.append(delta.content);
      if (parsedDelta.content || parsedDelta.reasoning) onDelta(parsedDelta);
    }
    if (Array.isArray(delta.tool_calls) && delta.tool_calls.length) {
      onDelta({ content: '', reasoning: '', toolCalls: delta.tool_calls });
    }
  }
  return completed;
}

function mergeToolCallDeltas(toolCallsByIndex, deltas) {
  for (const delta of deltas) {
    if (!delta || typeof delta !== 'object') continue;
    const index = Number.isInteger(delta.index) ? delta.index : 0;
    let entry = toolCallsByIndex.get(index);
    if (!entry) {
      entry = { id: '', name: '', arguments: '' };
      toolCallsByIndex.set(index, entry);
    }
    if (typeof delta.id === 'string' && delta.id) entry.id = delta.id;
    const fn = delta.function;
    if (fn && typeof fn === 'object') {
      if (typeof fn.name === 'string' && fn.name && entry.name !== fn.name) entry.name += fn.name;
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
    }
  }
}

function finalizeToolCalls(toolCallsByIndex) {
  return [...toolCallsByIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, entry]) => ({
      id: entry.id || `call-${makeId()}`,
      name: entry.name.trim(),
      arguments: entry.arguments,
    }))
    .filter((call) => call.name);
}

function buildTextBackendMessages(conversation) {
  const messages = [];
  const systemPrompt = state.settings.systemPrompt.trim();
  if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
  for (const message of conversation.messages) {
    if (message.id === streamingMessageId || message.kind === 'image' || message.kind === 'image-prompt') continue;
    if (message.kind === 'tool-result') {
      messages.push({ role: 'tool', content: message.content, tool_call_id: message.toolCallId });
      continue;
    }
    const content = message.role === 'assistant' ? getAssistantMessageParts(message).content : message.content;
    if (message.role === 'assistant' && Array.isArray(message.toolCalls) && message.toolCalls.length) {
      messages.push({
        role: 'assistant',
        content,
        tool_calls: message.toolCalls.map((call) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: call.arguments },
        })),
      });
      continue;
    }
    if (content.trim()) messages.push({ role: message.role, content });
  }
  return messages;
}

function stopStreaming() {
  if (activeController) activeController.abort();
  if (imageAbortController) imageAbortController.abort();
}

function clearConversation() {
  if (isBusy()) return;
  const conversation = getActiveConversation();
  if (!conversation.messages.length || !confirm('Clear this conversation?')) return;
  deleteImageBlobsForMessages(conversation.messages);
  conversation.messages = [];
  conversation.updatedAt = Date.now();
  persistAndRender();
}

function exportConversation() {
  const conversation = getActiveConversation();
  const payload = {
    exportedAt: new Date().toISOString(),
    note: 'Generated image files are not included in this JSON export.',
    settings: {
      model: state.settings.model,
      thinkingMode: getSelectedModelThinkingMode(),
      baseUrl: state.settings.baseUrl,
      temperature: state.settings.temperature,
      max_tokens: state.settings.max_tokens,
    },
    conversation,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${slugify(conversation.title)}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function openSettings() {
  applySettingsToForm();
  els.settingsDialog.showModal();
}

function saveSettingsFromForm() {
  const baseUrl = els.baseUrlInput.value.trim() || DEFAULT_SETTINGS.baseUrl;
  if (baseUrlEdited) {
    state.settings.baseUrlOverride = Boolean(baseUrl)
      && (!managedTextBackendEnabled || !serverDefaultBaseUrl || baseUrl !== serverDefaultBaseUrl);
  }
  state.settings.baseUrl = baseUrl;
  baseUrlEdited = false;
  state.settings.apiKey = els.apiKeyInput.value;
  state.settings.systemPrompt = els.systemPromptInput.value;
  state.settings.temperature = Number(els.temperatureInput.value);
  state.settings.max_tokens = Number.parseInt(els.maxTokensInput.value, 10) || DEFAULT_SETTINGS.max_tokens;
  state.settings.modelThinkingModes = normalizeModelThinkingModes(state.settings.modelThinkingModes);
  saveState();
  renderModelOptions(state.modelOptions || []);
}

function applySettingsToForm() {
  els.baseUrlInput.value = state.settings.baseUrl;
  baseUrlEdited = false;
  els.apiKeyInput.value = state.settings.apiKey;
  els.systemPromptInput.value = state.settings.systemPrompt;
  els.temperatureInput.value = state.settings.temperature;
  els.temperatureOutput.textContent = Number(state.settings.temperature).toFixed(2);
  els.maxTokensInput.value = state.settings.max_tokens;
  applyModelThinkingModeToForm();
}

function render(options = {}) {
  renderConversationList();
  renderActiveConversation(options.messages || {});
  renderModelOptions(state.modelOptions || []);
  setBusy();
  updateImageToolStatus();
}

function renderConversationList() {
  const activeId = state.activeConversationId;
  els.conversationList.replaceChildren();

  for (const conversation of state.conversations) {
    const item = document.createElement('div');
    item.className = `conversation-item${conversation.id === activeId ? ' active' : ''}`;

    const open = document.createElement('button');
    open.className = 'conversation-open';
    open.type = 'button';
    open.addEventListener('click', () => {
      if (isBusy()) return;
      state.activeConversationId = conversation.id;
      saveState();
      render();
      closeSidebar();
    });
    const name = document.createElement('span');
    name.className = 'conversation-name';
    name.textContent = conversation.title;
    const meta = document.createElement('span');
    meta.className = 'conversation-meta';
    const count = conversation.messages.filter((message) => message.kind !== 'tool-result').length;
    meta.textContent = `${count} message${count === 1 ? '' : 's'}`;
    open.append(name, meta);

    const remove = document.createElement('button');
    remove.className = 'delete-chat';
    remove.type = 'button';
    remove.textContent = 'X';
    remove.setAttribute('aria-label', `Delete ${conversation.title}`);
    remove.addEventListener('click', () => deleteConversation(conversation.id));

    item.append(open, remove);
    els.conversationList.append(item);
  }
}

function renderActiveConversation(options = {}) {
  const conversation = getActiveConversation();
  const viewportSnapshot = captureMessagesViewport();
  els.conversationTitle.textContent = conversation.title;
  els.messages.replaceChildren();

  if (!conversation.messages.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    const mark = document.createElement('div');
    mark.className = 'empty-mark';
    mark.textContent = 'L';
    const heading = document.createElement('h1');
    heading.textContent = 'What would you like to make?';
    const copy = document.createElement('p');
    copy.textContent = 'Start a conversation, or ask the assistant for a picture and it will draw one with the image tool.';
    const actions = document.createElement('div');
    actions.className = 'empty-actions';
    const chatButton = document.createElement('button');
    chatButton.className = 'secondary-button';
    chatButton.type = 'button';
    chatButton.textContent = 'Choose chat model';
    chatButton.addEventListener('click', openSettings);
    const imageButton = document.createElement('button');
    imageButton.className = 'primary-button';
    imageButton.type = 'button';
    imageButton.textContent = 'Image studio';
    imageButton.addEventListener('click', openImageDialog);
    actions.append(chatButton, imageButton);
    empty.append(mark, heading, copy, actions);
    els.messages.append(empty);
    retryPendingMessagesViewportRestore();
    return;
  }

  for (const message of conversation.messages) {
    if (isHiddenMessage(message)) continue;
    els.messages.append(renderMessage(message));
  }
  if (options.scrollToBottom) scrollMessagesToBottom();
  else if (options.preserveScroll) restoreMessagesViewport(viewportSnapshot);
  retryPendingMessagesViewportRestore();
}

function isHiddenMessage(message) {
  if (message.kind === 'tool-result') return true;
  if (
    message.role === 'assistant' &&
    message.kind !== 'image' &&
    message.id !== streamingMessageId &&
    Array.isArray(message.toolCalls) && message.toolCalls.length
  ) {
    const parts = getAssistantMessageParts(message);
    return !parts.content.trim() && !parts.reasoning.trim();
  }
  return false;
}

function renderMessage(message) {
  const wrapper = document.createElement('article');
  wrapper.className = `message ${message.role}${message.kind ? ` kind-${message.kind}` : ''}`;
  wrapper.dataset.messageId = message.id;
  const parts = message.role === 'assistant'
    ? getAssistantMessageParts(message)
    : { content: message.content || '', reasoning: '' };

  const header = document.createElement('div');
  header.className = 'message-header';
  const role = document.createElement('span');
  role.className = 'message-role';
  role.textContent = getMessageRoleLabel(message);
  header.append(role);
  const detailText = getMessageDetail(message);
  if (detailText) {
    const detail = document.createElement('span');
    detail.className = 'message-detail';
    detail.textContent = detailText;
    header.append(detail);
  }

  const body = document.createElement('div');
  body.className = 'message-body';
  if (message.kind === 'image') {
    renderGeneratedImage(body, message);
  } else {
    if (parts.reasoning) body.append(renderThinkingPanel(message, parts.reasoning));
    const answerText = parts.content || (message.id === streamingMessageId && !parts.reasoning ? '' : '');
    if (answerText) {
      const content = document.createElement('div');
      content.className = 'message-content';
      renderMessageContent(content, answerText);
      body.append(content);
    } else if (message.id === streamingMessageId) {
      const waiting = document.createElement('div');
      waiting.className = 'streaming-indicator';
      waiting.textContent = 'Writing...';
      body.append(waiting);
    }
  }

  if (!isMessageInProgress(message)) body.append(renderMessageTools(message, parts));
  wrapper.append(header, body);
  return wrapper;
}

function getMessageRoleLabel(message) {
  if (message.kind === 'image-prompt') return 'Image prompt';
  if (message.kind === 'image') return message.requestedBy === 'assistant' ? 'Image - via tool' : 'Image';
  return message.role === 'user' ? 'You' : 'Assistant';
}

function getMessageDetail(message) {
  if (message.kind !== 'image') return '';
  if (message.imageStatus === 'generating') return 'Creating...';
  if (message.imageStatus === 'error') return 'Generation failed';
  const model = message.imageModel ? basenameWithoutExtension(message.imageModel) : '';
  const sampler = getImageSamplerLabel(message.imageKind, message.sampler);
  const loraCount = Array.isArray(message.loras) ? message.loras.length : 0;
  return [
    model,
    sampler,
    message.steps ? `${message.steps} steps` : '',
    message.cfg !== undefined ? `CFG ${message.cfg}` : '',
    loraCount ? `${loraCount} LoRA${loraCount === 1 ? '' : 's'}` : '',
    message.seed !== undefined ? `seed ${message.seed}` : '',
  ].filter(Boolean).join(' - ');
}

function renderGeneratedImage(body, message) {
  if (message.imageStatus === 'generating') {
    const frame = document.createElement('div');
    frame.className = 'generated-image-frame generated-image-pending';
    applyGeneratedImageAspectRatio(frame, message);
    const waiting = document.createElement('div');
    waiting.className = 'streaming-indicator';
    waiting.textContent = 'Creating image...';
    frame.append(waiting);
    body.append(frame);
    return;
  }
  if (message.imageStatus === 'error') {
    const frame = document.createElement('div');
    frame.className = 'generated-image-frame generated-image-pending';
    applyGeneratedImageAspectRatio(frame, message);
    const error = document.createElement('div');
    error.className = 'image-error';
    error.textContent = message.error || 'Image generation failed.';
    frame.append(error);
    body.append(frame);
    return;
  }

  const frame = document.createElement('button');
  frame.className = 'generated-image-frame';
  frame.type = 'button';
  frame.setAttribute('aria-label', 'Open this image in the conversation gallery');
  frame.addEventListener('click', () => openImageGallery(message.id));
  applyGeneratedImageAspectRatio(frame, message);
  const placeholder = document.createElement('div');
  placeholder.className = 'image-placeholder';
  placeholder.textContent = 'Loading image...';
  frame.append(placeholder);
  body.append(frame);
  if (message.imageId) hydrateGeneratedImage(frame, message);
}

function applyGeneratedImageAspectRatio(frame, message) {
  const width = Number(message.width);
  const height = Number(message.height);
  frame.style.aspectRatio = Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0
    ? `${width} / ${height}`
    : '1 / 1';
}

function renderMessageTools(message, parts) {
  const tools = document.createElement('div');
  tools.className = 'message-tools';
  const copy = document.createElement('button');
  copy.className = 'mini-button';
  copy.type = 'button';
  copy.textContent = message.kind === 'image' ? 'Copy prompt' : 'Copy';
  copy.addEventListener('click', async () => {
    const text = message.kind === 'image' ? message.prompt || '' : message.role === 'assistant' ? parts.content : message.content;
    await writeToClipboard(text);
    copy.textContent = 'Copied';
    setTimeout(() => { copy.textContent = message.kind === 'image' ? 'Copy prompt' : 'Copy'; }, 900);
  });
  tools.append(copy);

  if (message.role === 'user' && message.kind !== 'image-prompt') {
    const edit = document.createElement('button');
    edit.className = 'mini-button';
    edit.type = 'button';
    edit.textContent = 'Edit';
    edit.addEventListener('click', () => editUserMessage(message.id));
    tools.append(edit);
  }
  return tools;
}

function isMessageInProgress(message) {
  return message.id === streamingMessageId || message.kind === 'image' && message.imageStatus === 'generating';
}

function renderThinkingPanel(message, text) {
  const isStreaming = message.id === streamingMessageId;
  const savedOpenState = thinkingPanelOpenByMessage.get(message.id);
  const panel = document.createElement('details');
  panel.className = 'thinking-panel';
  panel.open = typeof savedOpenState === 'boolean' ? savedOpenState : false;
  panel.addEventListener('toggle', () => thinkingPanelOpenByMessage.set(message.id, panel.open));
  const summary = document.createElement('summary');
  summary.textContent = isStreaming ? 'Thinking...' : 'Thinking';
  const content = document.createElement('div');
  content.className = 'thinking-content';
  renderMessageContent(content, text);
  panel.append(summary, content);
  return panel;
}

function renderMessageContent(container, text) {
  container.replaceChildren();
  const parts = splitCodeFences(text);
  for (const part of parts) {
    if (part.type === 'code') {
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = part.content;
      pre.append(code);
      container.append(pre);
      continue;
    }
    for (const paragraph of part.content.split(/\n{2,}/)) {
      if (!paragraph) continue;
      const node = document.createElement('p');
      node.textContent = paragraph;
      container.append(node);
    }
  }
}

function splitCodeFences(text) {
  const parts = [];
  const regex = /```[^\n]*\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push({ type: 'text', content: text.slice(lastIndex, match.index) });
    parts.push({ type: 'code', content: match[1] });
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) parts.push({ type: 'text', content: text.slice(lastIndex) });
  return parts.length ? parts : [{ type: 'text', content: text }];
}

function getAssistantMessageParts(message) {
  const parsed = splitThinkingFromText(message.content || '');
  const reasoning = [message.reasoning || '', parsed.reasoning].filter(Boolean).join('\n\n');
  return { content: parsed.content, reasoning };
}

function splitThinkingFromText(text) {
  const parser = createThinkingParser();
  const parsed = parser.append(text);
  const finalDelta = parser.flush();
  return {
    content: parsed.content + finalDelta.content,
    reasoning: parsed.reasoning + finalDelta.reasoning,
  };
}

function getReasoningDelta(delta) {
  if (typeof delta.reasoning_content === 'string') return delta.reasoning_content;
  if (typeof delta.reasoning === 'string') return delta.reasoning;
  if (typeof delta.reasoningContent === 'string') return delta.reasoningContent;
  return '';
}

function createThinkingParser() {
  const startTag = '<think>';
  const endTag = '</think>';
  let mode = 'content';
  let tagBuffer = '';
  return {
    append(text) {
      const result = { content: '', reasoning: '' };
      for (const character of text) {
        if (tagBuffer || character === '<') {
          tagBuffer += character;
          const normalized = tagBuffer.toLowerCase();
          if (normalized === startTag) {
            mode = 'reasoning';
            tagBuffer = '';
            continue;
          }
          if (normalized === endTag) {
            mode = 'content';
            tagBuffer = '';
            continue;
          }
          if (startTag.startsWith(normalized) || endTag.startsWith(normalized)) continue;
          appendThinkingPart(result, mode, tagBuffer);
          tagBuffer = '';
          continue;
        }
        appendThinkingPart(result, mode, character);
      }
      return result;
    },
    flush() {
      const result = { content: '', reasoning: '' };
      if (tagBuffer) {
        appendThinkingPart(result, mode, tagBuffer);
        tagBuffer = '';
      }
      return result;
    },
  };
}

function appendThinkingPart(result, mode, text) {
  if (mode === 'reasoning') result.reasoning += text;
  else result.content += text;
}

function updateStreamingMessage(message) {
  pendingStreamingMessages.set(message.id, message);
  if (streamingFrame === null) streamingFrame = requestAnimationFrame(flushStreamingMessages);
}

function flushStreamingMessages() {
  if (streamingFrame !== null) cancelAnimationFrame(streamingFrame);
  streamingFrame = null;
  for (const message of pendingStreamingMessages.values()) {
    const node = els.messages.querySelector(`[data-message-id="${message.id}"]`);
    if (!node) continue;
    const panel = node.querySelector('.thinking-panel');
    if (panel && thinkingPanelOpenByMessage.has(message.id)) thinkingPanelOpenByMessage.set(message.id, panel.open);
    const replacement = renderMessage(message);
    node.replaceChildren(...replacement.childNodes);
  }
  pendingStreamingMessages.clear();
}

function renderModelOptions(models) {
  state.modelOptions = models;
  const previous = state.settings.model;
  els.modelSelect.replaceChildren();
  if (accessIsRequired) {
    replaceSelectWithPlaceholder(els.modelSelect, ACCESS_REQUIRED_OPTION_TEXT);
    applyModelThinkingModeToForm();
    updateModelSummary();
    return;
  }
  if (!models.length) {
    const option = document.createElement('option');
    option.value = previous || '';
    option.textContent = previous ? `Saved: ${previous}` : 'No models loaded';
    els.modelSelect.append(option);
    els.modelSelect.value = previous || '';
    applyModelThinkingModeToForm();
    updateModelSummary();
    return;
  }
  for (const model of models) {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model;
    els.modelSelect.append(option);
  }
  if (models.includes(previous)) els.modelSelect.value = previous;
  else {
    els.modelSelect.value = models[0];
    state.settings.model = models[0];
    saveState();
  }
  applyModelThinkingModeToForm();
  updateModelSummary();
}

function updateModelSummary() {
  if (accessIsRequired) {
    els.modelSummary.textContent = 'Access';
    els.chatSetupButton.title = ACCESS_REQUIRED_MESSAGE;
    return;
  }
  els.modelSummary.textContent = state.settings.model ? basenameWithoutExtension(state.settings.model) : 'Choose';
  els.chatSetupButton.title = state.settings.model || 'Choose a chat model';
}

function getSelectedModelThinkingMode() {
  const model = state.settings.model;
  if (!model) return 'auto';
  const modes = normalizeModelThinkingModes(state.settings.modelThinkingModes);
  return modes[model] || 'auto';
}

function setSelectedModelThinkingMode(mode) {
  const model = state.settings.model;
  state.settings.modelThinkingModes = normalizeModelThinkingModes(state.settings.modelThinkingModes);
  if (!model) return;
  const normalizedMode = normalizeThinkingMode(mode);
  if (normalizedMode === 'auto') delete state.settings.modelThinkingModes[model];
  else state.settings.modelThinkingModes[model] = normalizedMode;
  applyModelThinkingModeToForm();
}

function applyModelThinkingModeToForm() {
  els.thinkingModeSelect.value = getSelectedModelThinkingMode();
  els.thinkingModeSelect.disabled = isBusy() || accessIsRequired || !state.settings.model;
}

function getImageSamplerLabel(kind, samplerId) {
  const normalizedKind = normalizeImageKind(kind);
  return IMAGE_SAMPLERS[normalizedKind].find((sampler) => sampler.id === samplerId)?.label || '';
}

function normalizeModelThinkingModes(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const modes = {};
  for (const [model, mode] of Object.entries(value)) {
    const trimmedModel = String(model).trim();
    const normalizedMode = normalizeThinkingMode(mode);
    if (trimmedModel && normalizedMode !== 'auto') modes[trimmedModel] = normalizedMode;
  }
  return modes;
}

function normalizeThinkingMode(mode) {
  const normalizedMode = String(mode || '').trim().toLowerCase();
  return THINKING_MODE_VALUES.has(normalizedMode) ? normalizedMode : 'auto';
}

async function openImageDialog() {
  if (isBusy()) return;
  applyImageSettingsToForm();
  els.imageFormError.textContent = '';
  els.imageDialog.showModal();
  await refreshImageConfig();
}

async function refreshImageConfig() {
  els.refreshImageModelsButton.disabled = true;
  els.generateImageButton.disabled = true;
  els.imageConnectionText.hidden = false;
  els.imageConnectionText.textContent = 'Loading models...';
  els.imageFormError.textContent = '';
  try {
    const response = await apiFetch('/api/image/config', { cache: 'no-store' });
    const data = await parseJsonResponse(response);
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    imageConfig = {
      loaded: true,
      connected: Boolean(data.connected),
      models: {
        anima: Array.isArray(data.models && data.models.anima) ? data.models.anima : [],
        sdxl: Array.isArray(data.models && data.models.sdxl) ? data.models.sdxl : [],
      },
      loras: {
        anima: Array.isArray(data.loras && data.loras.anima) ? data.loras.anima : [],
        sdxl: Array.isArray(data.loras && data.loras.sdxl) ? data.loras.sdxl : [],
      },
      runtime: data.runtime || {},
      dependencies: data.dependencies || {},
    };
  } catch (error) {
    imageConfig = {
      loaded: false,
      connected: false,
      models: { anima: [], sdxl: [] },
      loras: { anima: [], sdxl: [] },
      runtime: {},
      dependencies: {},
    };
    els.imageFormError.textContent = isAccessRequiredError(error) ? ACCESS_REQUIRED_MESSAGE : error.message || 'Could not load image models.';
  } finally {
    renderImageModelOptions();
    updateImageToolStatus();
    saveState();
    els.refreshImageModelsButton.disabled = accessIsRequired;
  }
}

function applyImageSettingsToForm() {
  els.imageKindSelect.value = state.image.kind;
  els.imageSizeSelect.value = state.image.size;
  els.negativePromptInput.value = state.image.negativePrompt;
  els.autoNegativeEverySelect.value = String(state.image.autoNegativeEvery);
  els.imageSeedInput.value = '';
  els.imageToolToggle.checked = state.image.toolEnabled !== false;
  applyImageKindSettings();
  renderImageSamplerOptions();
  renderImageModelOptions();
}

function applyImageKindSettings() {
  const kind = normalizeImageKind(state.image.kind);
  els.imageStepsInput.value = state.image.stepsByKind[kind] || '';
  els.imageCfgInput.value = state.image.cfgByKind[kind] || '';
  renderImageLoraRows();
}

function saveImageParametersFromForm() {
  const kind = normalizeImageKind(state.image.kind);
  state.image.stepsByKind[kind] = els.imageStepsInput.value;
  state.image.cfgByKind[kind] = els.imageCfgInput.value;
  saveState();
}

function renderImageSamplerOptions() {
  const kind = normalizeImageKind(state.image.kind);
  const samplers = IMAGE_SAMPLERS[kind];
  const selected = normalizeImageSampler(kind, state.image.samplerByKind[kind]);
  els.imageSamplerSelect.replaceChildren();
  for (const sampler of samplers) {
    const option = document.createElement('option');
    option.value = sampler.id;
    option.textContent = sampler.label;
    els.imageSamplerSelect.append(option);
  }
  els.imageSamplerSelect.value = selected;
  state.image.samplerByKind[kind] = selected;
  updateImageSamplerHint();
}

function updateImageSamplerHint() {
  const kind = normalizeImageKind(state.image.kind);
  const sampler = IMAGE_SAMPLERS[kind].find((entry) => entry.id === els.imageSamplerSelect.value);
  els.imageSamplerHint.textContent = sampler?.hint || '';
}

function normalizeImageSampler(kind, value) {
  const normalizedKind = normalizeImageKind(kind);
  const samplers = IMAGE_SAMPLERS[normalizedKind];
  const candidate = String(value || '');
  return samplers.some((sampler) => sampler.id === candidate)
    ? candidate
    : DEFAULT_IMAGE_SETTINGS.samplerByKind[normalizedKind];
}

function generateAutoNegativePrompt() {
  const prompt = els.imagePromptInput.value.trim();
  if (!prompt) {
    els.imageFormError.textContent = 'Describe the positive prompt before creating an automatic negative prompt.';
    els.imagePromptInput.focus();
    return;
  }
  const every = normalizeAutoNegativeEvery(els.autoNegativeEverySelect.value);
  const negativePrompt = interleaveAutoNegative(prompt, every).slice(0, 3000).trim();
  els.negativePromptInput.value = negativePrompt;
  state.image.negativePrompt = negativePrompt;
  state.image.autoNegativeEvery = every;
  els.imageFormError.textContent = '';
  saveState();
}

function interleaveAutoNegative(prompt, every, random = Math.random) {
  const words = String(prompt || '').trim().split(/\s+/).filter(Boolean);
  if (!words.length) return '';
  const interval = normalizeAutoNegativeEvery(every);
  const firstBoundaryRange = Math.min(interval, words.length + 1);
  const firstBoundary = Math.floor(clampRandom(random()) * firstBoundaryRange);
  const output = [];
  let nextBoundary = firstBoundary;
  for (let boundary = 0; boundary <= words.length; boundary += 1) {
    if (boundary === nextBoundary) {
      const termIndex = Math.floor(clampRandom(random()) * AUTO_NEGATIVE_TERMS.length);
      output.push(AUTO_NEGATIVE_TERMS[termIndex]);
      nextBoundary += interval;
    }
    if (boundary < words.length) output.push(words[boundary]);
  }
  return output.join(' ');
}

function clampRandom(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return 0;
  return Math.min(number, 0.999999999999);
}

function normalizeAutoNegativeEvery(value) {
  const number = Number.parseInt(value, 10);
  return Number.isInteger(number) && number >= 1 && number <= 10 ? number : DEFAULT_IMAGE_SETTINGS.autoNegativeEvery;
}

function renderImageModelOptions() {
  if (accessIsRequired) {
    replaceSelectWithPlaceholder(els.imageModelSelect, ACCESS_REQUIRED_OPTION_TEXT);
    els.imageConnectionText.textContent = ACCESS_REQUIRED_MESSAGE;
    els.imageConnectionText.hidden = false;
    els.generateImageButton.disabled = true;
    renderImageLoraRows();
    return;
  }
  const kind = normalizeImageKind(els.imageKindSelect.value || state.image.kind);
  state.image.kind = kind;
  const models = imageConfig.models[kind] || [];
  const previous = state.image.modelByKind[kind] || '';
  els.imageModelSelect.replaceChildren();
  if (!models.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = kind === 'anima' ? 'No Anima models found' : 'No SDXL models found';
    els.imageModelSelect.append(option);
  } else {
    for (const model of models) {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.label || model.id;
      els.imageModelSelect.append(option);
    }
    const selected = models.some((model) => model.id === previous) ? previous : models[0].id;
    els.imageModelSelect.value = selected;
    state.image.modelByKind[kind] = selected;
  }

  const dependenciesReady = areImageDependenciesReady(kind);
  let availabilityMessage = '';
  if (!imageConfig.connected) {
    availabilityMessage = imageConfig.runtime.error || 'Image generation is unavailable.';
  } else if (!dependenciesReady) {
    availabilityMessage = 'Anima requires its text encoder and VAE.';
  } else if (!models.length) {
    availabilityMessage = `No compatible ${kind === 'anima' ? 'Anima' : 'SDXL'} models found.`;
  }
  els.imageConnectionText.textContent = availabilityMessage;
  els.imageConnectionText.hidden = !availabilityMessage;
  els.generateImageButton.disabled = isBusy() || !imageConfig.connected || !models.length || !dependenciesReady;
  applyImageRecommendation(false);
  renderImageLoraRows();
}

function applyImageRecommendation(force) {
  const models = imageConfig.models[state.image.kind] || [];
  const selected = models.find((model) => model.id === els.imageModelSelect.value);
  if (!selected) return;
  if (force || !els.imageStepsInput.value) els.imageStepsInput.value = selected.recommendedSteps;
  if (force || !els.imageCfgInput.value) els.imageCfgInput.value = selected.recommendedCfg;
  saveImageParametersFromForm();
}

function availableImageLoras() {
  return imageConfig.loras[state.image.kind] || [];
}

function renderImageLoraRows() {
  if (!imageConfig.loaded) {
    els.imageLoraList.replaceChildren();
    els.imageLoraHint.textContent = 'Loading LoRAs...';
    els.addImageLoraButton.disabled = true;
    return;
  }
  const available = availableImageLoras();
  const availableIds = new Set(available.map((lora) => lora.id));
  const saved = (state.image.lorasByKind[state.image.kind] || [])
    .filter((lora) => availableIds.has(lora.id))
    .slice(0, 4);
  state.image.lorasByKind[state.image.kind] = saved;
  els.imageLoraList.replaceChildren();
  for (const lora of saved) addImageLoraRow(lora.id, lora.strength, false);
  els.imageLoraHint.textContent = available.length
    ? 'Optional model adjustments'
    : 'No compatible LoRAs found';
  els.addImageLoraButton.disabled = !available.length || saved.length >= 4 || saved.length >= available.length;
  updateLoraOptionAvailability();
}

function addImageLoraRow(selectedId = '', strength = 1, persist = true) {
  const available = availableImageLoras();
  const used = new Set(
    [...els.imageLoraList.querySelectorAll('select')].map((select) => select.value).filter(Boolean),
  );
  const selected = selectedId || available.find((lora) => !used.has(lora.id))?.id || '';
  if (!selected) return;

  const row = document.createElement('div');
  row.className = 'lora-row';
  const picker = document.createElement('label');
  picker.className = 'field';
  const pickerLabel = document.createElement('span');
  pickerLabel.textContent = 'LoRA';
  const select = document.createElement('select');
  select.setAttribute('aria-label', 'LoRA');
  for (const lora of available) {
    const option = document.createElement('option');
    option.value = lora.id;
    option.textContent = lora.label || basenameWithoutExtension(lora.id);
    select.append(option);
  }
  select.value = selected;
  picker.append(pickerLabel, select);

  const strengthField = document.createElement('label');
  strengthField.className = 'field';
  const strengthLabel = document.createElement('span');
  strengthLabel.textContent = 'Strength';
  const strengthInput = document.createElement('input');
  strengthInput.type = 'number';
  strengthInput.min = '-2';
  strengthInput.max = '2';
  strengthInput.step = '0.05';
  strengthInput.value = String(strength);
  strengthInput.setAttribute('aria-label', 'LoRA strength');
  strengthField.append(strengthLabel, strengthInput);

  const remove = document.createElement('button');
  remove.className = 'secondary-button remove-lora-button';
  remove.type = 'button';
  remove.textContent = 'X';
  remove.setAttribute('aria-label', 'Remove LoRA');
  select.addEventListener('change', syncImageLorasFromRows);
  strengthInput.addEventListener('input', syncImageLorasFromRows);
  remove.addEventListener('click', () => {
    row.remove();
    syncImageLorasFromRows();
  });
  row.append(picker, strengthField, remove);
  els.imageLoraList.append(row);
  if (persist) syncImageLorasFromRows();
}

function syncImageLorasFromRows() {
  const selections = [...els.imageLoraList.querySelectorAll('.lora-row')].map((row) => ({
    id: row.querySelector('select').value,
    strength: Number(row.querySelector('input').value),
  }));
  state.image.lorasByKind[state.image.kind] = selections;
  els.addImageLoraButton.disabled = selections.length >= 4 || selections.length >= availableImageLoras().length;
  updateLoraOptionAvailability();
  saveState();
}

function updateLoraOptionAvailability() {
  const selects = [...els.imageLoraList.querySelectorAll('select')];
  const selected = new Set(selects.map((select) => select.value));
  for (const select of selects) {
    for (const option of select.options) option.disabled = option.value !== select.value && selected.has(option.value);
  }
}

function areImageDependenciesReady(kind) {
  return kind !== 'anima' || Boolean(imageConfig.dependencies.animaTextEncoder && imageConfig.dependencies.animaVae);
}

function makeImageToolDefinition() {
  return {
    type: 'function',
    function: {
      name: IMAGE_TOOL_NAME,
      description: 'Generate a picture with the local image engine and show it in the chat. Call this whenever the user asks for an image, drawing, photo, painting, logo, or any other visual.',
      parameters: {
        type: 'object',
        properties: {
          prompt: {
            type: 'string',
            description: 'A detailed visual description of the image: subject, setting, composition, lighting, and style.',
          },
          orientation: {
            type: 'string',
            enum: ['portrait', 'square', 'landscape'],
            description: 'Optional canvas orientation. Omit to use the user preference.',
          },
        },
        required: ['prompt'],
      },
    },
  };
}

function canUseImageTool() {
  return state.image.toolEnabled !== false && Boolean(getImageGenerationSettings());
}

function getImageGenerationSettings() {
  const kind = normalizeImageKind(state.image.kind);
  if (!imageConfig.loaded || !imageConfig.connected || !areImageDependenciesReady(kind)) return null;
  const models = imageConfig.models[kind] || [];
  if (!models.length) return null;
  const selected = models.find((model) => model.id === state.image.modelByKind[kind]) || models[0];
  const savedSteps = Number.parseInt(state.image.stepsByKind[kind], 10);
  const savedCfg = state.image.cfgByKind[kind] === '' ? NaN : Number(state.image.cfgByKind[kind]);
  const availableLoraIds = new Set((imageConfig.loras[kind] || []).map((lora) => lora.id));
  return {
    kind,
    model: selected.id,
    steps: Number.isInteger(savedSteps) && savedSteps >= 1 && savedSteps <= 80 ? savedSteps : selected.recommendedSteps,
    cfg: Number.isFinite(savedCfg) ? savedCfg : selected.recommendedCfg,
    sampler: normalizeImageSampler(kind, state.image.samplerByKind[kind]),
    loras: (state.image.lorasByKind[kind] || []).filter((lora) => availableLoraIds.has(lora.id)).slice(0, 4),
    negativePrompt: state.image.negativePrompt,
    size: state.image.size,
  };
}

function updateImageToolStatus() {
  if (!els.imageToolStatus) return;
  if (accessIsRequired) {
    els.imageToolStatus.textContent = 'Image tool needs access';
    els.imageButton.classList.remove('armed');
    return;
  }
  const armed = state.image.toolEnabled !== false && canUseImageTool();
  let text = 'Image tool off';
  if (state.image.toolEnabled !== false) text = armed ? 'Image tool ready' : 'Image tool waiting for models';
  els.imageToolStatus.textContent = text;
  els.imageButton.classList.toggle('armed', armed);
}

async function runImageToolCalls(conversation, toolCalls, toolDepth) {
  let stopped = false;
  for (const call of toolCalls) {
    const outcome = await executeImageToolCall(conversation, call);
    conversation.messages.push(makeMessage('tool', JSON.stringify(outcome.report), {
      kind: 'tool-result',
      toolCallId: call.id,
      toolName: call.name,
    }));
    conversation.updatedAt = Date.now();
    if (outcome.stopped) {
      stopped = true;
      break;
    }
  }
  persistAndRender({ messages: { preserveScroll: true } });
  if (stopped) return;
  await streamAssistantReply(conversation, { toolDepth: toolDepth + 1 });
}

async function executeImageToolCall(conversation, call) {
  if (call.name !== IMAGE_TOOL_NAME) {
    return { report: { ok: false, error: `Unknown tool "${call.name}". Only ${IMAGE_TOOL_NAME} is available.` } };
  }
  let args = {};
  if (call.arguments) {
    try {
      args = JSON.parse(call.arguments);
    } catch {
      return { report: { ok: false, error: 'The tool arguments were not valid JSON.' } };
    }
  }
  const prompt = typeof args.prompt === 'string' ? args.prompt.trim().slice(0, 5000) : '';
  if (!prompt) {
    return { report: { ok: false, error: 'A prompt describing the image is required.' } };
  }
  let settings = getImageGenerationSettings();
  if (!settings) {
    await refreshImageConfig();
    settings = getImageGenerationSettings();
  }
  if (!settings) {
    return { report: { ok: false, error: 'Image generation is not available right now. Ask the user to open the image settings.' } };
  }
  const orientation = typeof args.orientation === 'string' ? args.orientation.toLowerCase() : '';
  const size = IMAGE_SIZE_VALUES.has(orientation) ? orientation : settings.size;
  const dimensions = IMAGE_DIMENSIONS[size] || IMAGE_DIMENSIONS.square;

  const imageMessage = makeMessage('assistant', '', {
    kind: 'image',
    imageStatus: 'generating',
    imageId: '',
    imageModel: settings.model,
    imageKind: settings.kind,
    steps: settings.steps,
    cfg: settings.cfg,
    sampler: settings.sampler,
    loras: settings.loras,
    width: dimensions.width,
    height: dimensions.height,
    prompt,
    requestedBy: 'assistant',
  });
  conversation.messages.push(imageMessage);
  conversation.updatedAt = Date.now();
  imageGenerationInFlight = true;
  setBusy();
  setStatus('connected', 'Creating image');
  persistAndRender({ messages: { preserveScroll: true } });

  try {
    const data = await requestImageGeneration({
      kind: settings.kind,
      model: settings.model,
      prompt,
      negativePrompt: settings.negativePrompt,
      size,
      steps: settings.steps,
      cfg: settings.cfg,
      sampler: settings.sampler,
      seed: '',
      loras: settings.loras,
    });
    await storeGeneratedImage(imageMessage, data, settings.model, settings.loras);
    conversation.updatedAt = Date.now();
    setStatus('connected', 'Image ready');
    return { report: { ok: true, result: 'The image was generated and the user is looking at it in the chat now.' } };
  } catch (error) {
    const stoppedByUser = error.name === 'AbortError';
    imageMessage.imageStatus = 'error';
    imageMessage.error = stoppedByUser ? 'Image generation was stopped.' : error.message || 'Image generation failed.';
    imageMessage.updatedAt = Date.now();
    if (!stoppedByUser) setStatus('error', imageMessage.error);
    return { report: { ok: false, error: imageMessage.error }, stopped: stoppedByUser };
  } finally {
    imageGenerationInFlight = false;
    setBusy();
    persistAndRender({ messages: { preserveScroll: true } });
  }
}

async function requestImageGeneration(request) {
  imageAbortController = new AbortController();
  setBusy();
  try {
    const response = await apiFetch('/api/image/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: imageAbortController.signal,
    });
    const data = await parseJsonResponse(response);
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (!data.imageBase64 || !data.mimeType) throw new Error('The server returned an empty image.');
    return data;
  } finally {
    imageAbortController = null;
  }
}

async function storeGeneratedImage(imageMessage, data, fallbackModel, fallbackLoras) {
  const imageId = makeId();
  const blob = base64ToBlob(data.imageBase64, data.mimeType);
  let storage = 'browser';
  try {
    await putImageBlob(imageId, blob);
  } catch {
    sessionImageBlobs.set(imageId, blob);
    storage = 'session';
  }
  imageMessage.imageStatus = 'ready';
  imageMessage.imageId = imageId;
  imageMessage.imageModel = data.model || fallbackModel;
  imageMessage.seed = data.seed;
  imageMessage.width = data.width;
  imageMessage.height = data.height;
  imageMessage.steps = data.steps;
  imageMessage.cfg = data.cfg;
  imageMessage.sampler = data.sampler || imageMessage.sampler;
  imageMessage.loras = Array.isArray(data.loras) ? data.loras : fallbackLoras;
  imageMessage.storage = storage;
  imageMessage.updatedAt = Date.now();
}

async function generateImage() {
  if (isBusy()) return;
  const prompt = els.imagePromptInput.value.trim();
  const kind = normalizeImageKind(els.imageKindSelect.value);
  const model = els.imageModelSelect.value;
  const steps = Number.parseInt(els.imageStepsInput.value, 10);
  const cfg = Number(els.imageCfgInput.value);
  const sampler = normalizeImageSampler(kind, els.imageSamplerSelect.value);
  const loras = [...els.imageLoraList.querySelectorAll('.lora-row')].map((row) => ({
    id: row.querySelector('select').value,
    strength: Number(row.querySelector('input').value),
  }));
  if (!prompt) {
    els.imageFormError.textContent = 'Describe the image you want to create.';
    els.imagePromptInput.focus();
    return;
  }
  if (!model || !imageConfig.connected) {
    els.imageFormError.textContent = 'Choose an available model.';
    return;
  }
  if (!Number.isInteger(steps) || steps < 1 || steps > 80) {
    els.imageFormError.textContent = 'Steps must be between 1 and 80.';
    els.imageStepsInput.focus();
    return;
  }
  if (els.imageCfgInput.value.trim() === '' || !Number.isFinite(cfg)) {
    els.imageFormError.textContent = 'CFG must be a finite number.';
    els.imageCfgInput.focus();
    return;
  }
  if (new Set(loras.map((lora) => lora.id)).size !== loras.length) {
    els.imageFormError.textContent = 'Choose each LoRA only once.';
    return;
  }
  if (loras.some((lora) => !Number.isFinite(lora.strength) || lora.strength < -2 || lora.strength > 2)) {
    els.imageFormError.textContent = 'LoRA strength must be between -2 and 2.';
    return;
  }

  state.image.kind = kind;
  state.image.modelByKind[kind] = model;
  state.image.size = normalizeImageSize(els.imageSizeSelect.value);
  state.image.negativePrompt = els.negativePromptInput.value.trim();
  state.image.stepsByKind[kind] = String(steps);
  state.image.cfgByKind[kind] = String(cfg);
  state.image.samplerByKind[kind] = sampler;
  state.image.lorasByKind[kind] = loras;
  saveState();

  const conversation = getActiveConversation();
  const dimensions = IMAGE_DIMENSIONS[state.image.size] || IMAGE_DIMENSIONS.square;
  const promptMessage = makeMessage('user', prompt, { kind: 'image-prompt' });
  const imageMessage = makeMessage('assistant', '', {
    kind: 'image',
    imageStatus: 'generating',
    imageId: '',
    imageModel: model,
    imageKind: kind,
    steps,
    cfg,
    sampler,
    loras,
    width: dimensions.width,
    height: dimensions.height,
    prompt,
  });
  conversation.messages.push(promptMessage, imageMessage);
  if (conversation.title === 'New chat') conversation.title = titleFromPrompt(prompt);
  conversation.updatedAt = Date.now();
  imageGenerationInFlight = true;
  els.imageDialog.close();
  setStatus('connected', 'Creating image');
  persistAndRender({ messages: { preserveScroll: true } });

  try {
    const data = await requestImageGeneration({
      kind,
      model,
      prompt,
      negativePrompt: state.image.negativePrompt,
      size: state.image.size,
      steps,
      cfg,
      sampler,
      seed: els.imageSeedInput.value,
      loras,
    });
    await storeGeneratedImage(imageMessage, data, model, loras);
    conversation.updatedAt = Date.now();
    setStatus('connected', 'Image ready');
  } catch (error) {
    imageMessage.imageStatus = 'error';
    imageMessage.error = error.name === 'AbortError'
      ? 'Image generation was stopped.'
      : error.message || 'Image generation failed.';
    imageMessage.updatedAt = Date.now();
    setStatus('error', imageMessage.error);
  } finally {
    imageGenerationInFlight = false;
    setBusy();
    persistAndRender({ messages: { preserveScroll: true } });
  }
}

async function hydrateGeneratedImage(frame, message) {
  const blob = await getStoredImageBlob(message);
  if (!frame.isConnected) return;
  if (!blob) {
    renderMissingImage(frame);
    return;
  }
  const image = await loadStoredImageElement(message, blob, 'generated-image');
  if (!frame.isConnected) return;
  if (!image) {
    renderMissingImage(frame);
    return;
  }
  const viewportSnapshot = captureMessagesViewport();
  frame.replaceChildren(image);
  restoreMessagesViewport(viewportSnapshot);
  retryPendingMessagesViewportRestore();
}

async function getStoredImageBlob(message) {
  let blob = sessionImageBlobs.get(message.imageId);
  if (!blob) {
    try {
      blob = await getImageBlob(message.imageId);
    } catch {
      blob = null;
    }
  }
  return blob;
}

async function loadStoredImageElement(message, blob, className) {
  let url = imageObjectUrls.get(message.imageId);
  if (!url) {
    url = URL.createObjectURL(blob);
    imageObjectUrls.set(message.imageId, url);
  }
  const image = document.createElement('img');
  image.className = className;
  image.alt = message.prompt ? `Generated image: ${message.prompt}` : 'Generated image';
  image.decoding = 'async';
  image.src = url;
  try {
    await image.decode();
  } catch {
    if (!image.complete || !image.naturalWidth) return null;
  }
  return image.naturalWidth ? image : null;
}

function renderMissingImage(container) {
  const missing = document.createElement('div');
  missing.className = 'image-placeholder';
  missing.textContent = 'This image is no longer available.';
  container.replaceChildren(missing);
}

function openImageGallery(messageId) {
  const images = getActiveConversation().messages.filter((message) => (
    message.kind === 'image' && message.imageStatus === 'ready' && message.imageId
  ));
  if (!images.length) return;
  els.imageGalleryViewport.replaceChildren();
  els.imageGalleryCount.textContent = `${images.length} image${images.length === 1 ? '' : 's'}`;
  for (let index = 0; index < images.length; index += 1) {
    const message = images[index];
    const item = document.createElement('article');
    item.className = 'gallery-item';
    item.dataset.messageId = message.id;
    const shell = document.createElement('div');
    shell.className = 'gallery-image-shell';
    const placeholder = document.createElement('div');
    placeholder.className = 'image-placeholder';
    placeholder.textContent = 'Loading image...';
    shell.append(placeholder);
    const caption = document.createElement('div');
    caption.className = 'gallery-caption';
    const position = document.createElement('strong');
    position.textContent = `Image ${index + 1} of ${images.length}`;
    const prompt = document.createElement('span');
    prompt.textContent = message.prompt || 'Generated image';
    caption.append(position, prompt);
    item.append(shell, caption);
    els.imageGalleryViewport.append(item);
    void hydrateGalleryImage(shell, message);
  }
  els.imageGalleryDialog.showModal();
  requestAnimationFrame(() => {
    const target = Array.from(els.imageGalleryViewport.children)
      .find((item) => item.dataset.messageId === messageId);
    if (target) els.imageGalleryViewport.scrollTop = target.offsetTop - els.imageGalleryViewport.offsetTop;
    els.imageGalleryViewport.focus({ preventScroll: true });
  });
}

async function hydrateGalleryImage(shell, message) {
  const blob = await getStoredImageBlob(message);
  if (!shell.isConnected) return;
  if (!blob) {
    renderMissingImage(shell);
    return;
  }
  const image = await loadStoredImageElement(message, blob, 'gallery-image');
  if (!shell.isConnected) return;
  if (image) shell.replaceChildren(image);
  else renderMissingImage(shell);
}

function base64ToBlob(base64, mimeType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Blob([bytes], { type: mimeType });
}

function openImageDatabase() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is unavailable.'));
      return;
    }
    const request = indexedDB.open(IMAGE_DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(IMAGE_STORE_NAME)) request.result.createObjectStore(IMAGE_STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Could not open image storage.'));
  });
}

async function putImageBlob(imageId, blob) {
  const database = await openImageDatabase();
  try {
    await runImageTransaction(database, 'readwrite', (store) => store.put(blob, imageId));
  } finally {
    database.close();
  }
}

async function getImageBlob(imageId) {
  const database = await openImageDatabase();
  try {
    return await runImageTransaction(database, 'readonly', (store) => store.get(imageId));
  } finally {
    database.close();
  }
}

async function deleteImageBlob(imageId) {
  sessionImageBlobs.delete(imageId);
  const objectUrl = imageObjectUrls.get(imageId);
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  imageObjectUrls.delete(imageId);
  try {
    const database = await openImageDatabase();
    try {
      await runImageTransaction(database, 'readwrite', (store) => store.delete(imageId));
    } finally {
      database.close();
    }
  } catch {
    // Persistent image storage can be unavailable in restrictive browser modes.
  }
}

function runImageTransaction(database, mode, operation) {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(IMAGE_STORE_NAME, mode);
    const request = operation(transaction.objectStore(IMAGE_STORE_NAME));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Image storage failed.'));
    transaction.onabort = () => reject(transaction.error || new Error('Image storage was interrupted.'));
  });
}

function deleteImageBlobsForMessages(messages) {
  for (const message of messages) {
    if (message && message.imageId) deleteImageBlob(message.imageId);
  }
}

function editUserMessage(messageId) {
  if (isBusy()) return;
  const conversation = getActiveConversation();
  const messageIndex = conversation.messages.findIndex((message) => message.id === messageId);
  if (messageIndex === -1) return;
  const message = conversation.messages[messageIndex];
  els.promptInput.value = message.content;
  autosizePrompt();
  els.charCount.textContent = String(message.content.length);
  els.promptInput.focus();
  const removed = conversation.messages.splice(messageIndex);
  deleteImageBlobsForMessages(removed);
  conversation.updatedAt = Date.now();
  persistAndRender();
}

function renameConversation() {
  els.conversationActionsDialog.close();
  if (isBusy()) return;
  const conversation = getActiveConversation();
  const title = window.prompt('Rename conversation', conversation.title);
  if (title === null) return;
  conversation.title = title.trim() || 'Untitled chat';
  conversation.updatedAt = Date.now();
  persistAndRender();
}

function deleteConversation(conversationId) {
  if (isBusy()) return;
  if (state.conversations.length === 1) {
    deleteImageBlobsForMessages(state.conversations[0].messages);
    state.conversations[0] = createConversation();
    state.activeConversationId = state.conversations[0].id;
    persistAndRender();
    return;
  }
  const index = state.conversations.findIndex((conversation) => conversation.id === conversationId);
  if (index === -1) return;
  const removed = state.conversations.splice(index, 1)[0];
  deleteImageBlobsForMessages(removed.messages);
  if (state.activeConversationId === conversationId) state.activeConversationId = state.conversations[Math.max(0, index - 1)].id;
  persistAndRender();
}

function persistAndRender(options = {}) {
  saveState();
  render(options);
}

function persistAndRenderList() {
  saveState();
  renderConversationList();
}

function isBusy() {
  return Boolean(activeController) || imageGenerationInFlight || modelLoadInFlight;
}

function setBusy() {
  const chatBusy = Boolean(activeController);
  const busy = isBusy();
  const stoppable = chatBusy || imageGenerationInFlight;
  els.sendButton.hidden = stoppable;
  els.stopButton.hidden = !stoppable;
  els.sendButton.disabled = busy || accessIsRequired;
  els.promptInput.disabled = busy || accessIsRequired;
  els.imageButton.disabled = busy || accessIsRequired;
  els.modelSelect.disabled = busy || accessIsRequired;
  els.thinkingModeSelect.disabled = busy || accessIsRequired || !state.settings.model;
  els.regenerateButton.disabled = busy || accessIsRequired || findLastAssistantIndex(getActiveConversation().messages) === -1;
  els.clearButton.disabled = busy || accessIsRequired;
  els.generateImageButton.disabled =
    busy ||
    accessIsRequired ||
    !imageConfig.connected ||
    !(imageConfig.models[state.image.kind] || []).length ||
    !areImageDependenciesReady(state.image.kind);
}

function setStatus(kind, text) {
  els.statusDot.classList.toggle('connected', kind === 'connected');
  els.statusDot.classList.toggle('error', kind === 'error');
  els.statusText.textContent = text;
}

function autosizePrompt() {
  els.promptInput.style.height = 'auto';
  els.promptInput.style.height = `${Math.min(180, els.promptInput.scrollHeight)}px`;
}

function scrollMessagesToBottom() {
  markMessagesScrollIntent();
  requestAnimationFrame(() => { els.messages.scrollTop = els.messages.scrollHeight; });
}

function captureMessagesViewport() {
  const containerRect = els.messages.getBoundingClientRect();
  const anchor = Array.from(els.messages.querySelectorAll('.message'))
    .find((message) => message.getBoundingClientRect().bottom > containerRect.top + 1);
  return {
    scrollTop: els.messages.scrollTop,
    anchorId: anchor?.dataset.messageId || '',
    anchorOffset: anchor ? anchor.getBoundingClientRect().top - containerRect.top : 0,
  };
}

function restoreMessagesViewport(snapshot) {
  if (!snapshot) return true;
  const anchor = snapshot.anchorId
    ? Array.from(els.messages.querySelectorAll('.message'))
      .find((message) => message.dataset.messageId === snapshot.anchorId)
    : null;
  if (anchor) {
    const containerTop = els.messages.getBoundingClientRect().top;
    const currentOffset = anchor.getBoundingClientRect().top - containerTop;
    els.messages.scrollTop += currentOffset - snapshot.anchorOffset;
    const restoredOffset = anchor.getBoundingClientRect().top - containerTop;
    return Math.abs(restoredOffset - snapshot.anchorOffset) <= 1;
  }
  const maxScrollTop = Math.max(0, els.messages.scrollHeight - els.messages.clientHeight);
  els.messages.scrollTop = Math.min(snapshot.scrollTop, maxScrollTop);
  return Math.abs(els.messages.scrollTop - snapshot.scrollTop) <= 1;
}

function setModelLoadingHidden(hidden) {
  if (!els.modelLoading || els.modelLoading.hidden === hidden) return;
  const viewportSnapshot = captureMessagesViewport();
  els.modelLoading.hidden = hidden;
  queueMessagesViewportRestore(viewportSnapshot);
}

function queueMessagesViewportRestore(snapshot) {
  if (!pendingMessagesViewportRestore
    || pendingMessagesViewportRestore.intentVersion !== messagesScrollIntentVersion
    || pendingMessagesViewportRestore.conversationId !== state.activeConversationId) {
    pendingMessagesViewportRestore = {
      snapshot,
      intentVersion: messagesScrollIntentVersion,
      conversationId: state.activeConversationId,
    };
  }
  retryPendingMessagesViewportRestore();
  requestAnimationFrame(() => {
    retryPendingMessagesViewportRestore();
    requestAnimationFrame(() => {
      retryPendingMessagesViewportRestore();
    });
  });
}

function retryPendingMessagesViewportRestore() {
  const pending = pendingMessagesViewportRestore;
  if (!pending) return;
  if (pending.intentVersion !== messagesScrollIntentVersion
    || pending.conversationId !== state.activeConversationId) {
    pendingMessagesViewportRestore = null;
    return;
  }
  const anchorExists = !pending.snapshot.anchorId
    || Array.from(els.messages.querySelectorAll('.message'))
      .some((message) => message.dataset.messageId === pending.snapshot.anchorId);
  if (!anchorExists) {
    pendingMessagesViewportRestore = null;
    return;
  }
  if (restoreMessagesViewport(pending.snapshot) && pendingMessagesViewportRestore === pending) {
    pendingMessagesViewportRestore = null;
  }
}

function markMessagesScrollIntent() {
  messagesScrollIntentVersion += 1;
  pendingMessagesViewportRestore = null;
}

function getActiveConversation() {
  let conversation = state.conversations.find((item) => item.id === state.activeConversationId);
  if (!conversation) {
    conversation = state.conversations[0] || createConversation();
    if (!state.conversations.length) state.conversations.push(conversation);
    state.activeConversationId = conversation.id;
  }
  return conversation;
}

function createConversation() {
  const now = Date.now();
  return { id: makeId(), title: 'New chat', createdAt: now, updatedAt: now, messages: [] };
}

function makeMessage(role, content, extras = {}) {
  const now = Date.now();
  return {
    id: makeId(),
    role,
    content,
    reasoning: '',
    createdAt: now,
    updatedAt: now,
    ...extras,
  };
}

function findLastAssistantIndex(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'assistant' && messages[index].kind !== 'image') return index;
  }
  return -1;
}

function titleFromPrompt(prompt) {
  return prompt.replace(/\s+/g, ' ').trim().slice(0, 54) || 'New chat';
}

function basenameWithoutExtension(value) {
  return String(value || '').split('/').pop().replace(/\.safetensors$/i, '');
}

function slugify(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') || 'conversation';
}

function makeId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

async function writeToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.top = '-1000px';
  document.body.append(textarea);
  textarea.select();
  document.execCommand('copy');
  textarea.remove();
}

async function parseJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { error: text };
  }
}

function loadState() {
  const fallback = {
    settings: normalizeSettings(),
    image: normalizeImageSettings(),
    conversations: [],
    activeConversationId: '',
    modelOptions: [],
  };
  try {
    const parsed = JSON.parse(safeStorageGet(STORAGE_KEY) || '{}');
    return {
      ...fallback,
      ...parsed,
      settings: normalizeSettings(parsed.settings || {}),
      image: normalizeImageSettings(parsed.image || {}),
      conversations: Array.isArray(parsed.conversations) ? parsed.conversations : [],
      modelOptions: Array.isArray(parsed.modelOptions) ? parsed.modelOptions : [],
    };
  } catch {
    return fallback;
  }
}

function normalizeSettings(settings = {}) {
  return {
    ...DEFAULT_SETTINGS,
    ...settings,
    baseUrl: typeof settings.baseUrl === 'string' ? settings.baseUrl : DEFAULT_SETTINGS.baseUrl,
    baseUrlOverride: settings.baseUrlOverride === true,
    apiKey: '',
    modelThinkingModes: normalizeModelThinkingModes(settings.modelThinkingModes),
  };
}

function normalizeImageSettings(settings = {}) {
  const modelByKind = settings.modelByKind && typeof settings.modelByKind === 'object' ? settings.modelByKind : {};
  const stepsByKind = settings.stepsByKind && typeof settings.stepsByKind === 'object' ? settings.stepsByKind : {};
  const cfgByKind = settings.cfgByKind && typeof settings.cfgByKind === 'object' ? settings.cfgByKind : {};
  const samplerByKind = settings.samplerByKind && typeof settings.samplerByKind === 'object' ? settings.samplerByKind : {};
  const lorasByKind = settings.lorasByKind && typeof settings.lorasByKind === 'object' ? settings.lorasByKind : {};
  return {
    ...DEFAULT_IMAGE_SETTINGS,
    ...settings,
    kind: normalizeImageKind(settings.kind),
    size: normalizeImageSize(settings.size),
    negativePrompt: typeof settings.negativePrompt === 'string' ? settings.negativePrompt.slice(0, 3000) : '',
    autoNegativeEvery: normalizeAutoNegativeEvery(settings.autoNegativeEvery),
    toolEnabled: settings.toolEnabled !== false,
    modelByKind: {
      anima: typeof modelByKind.anima === 'string' ? modelByKind.anima : '',
      sdxl: typeof modelByKind.sdxl === 'string' ? modelByKind.sdxl : '',
    },
    stepsByKind: {
      anima: normalizeSavedNumber(stepsByKind.anima, 1, 80),
      sdxl: normalizeSavedNumber(stepsByKind.sdxl, 1, 80),
    },
    cfgByKind: {
      anima: normalizeSavedFiniteNumber(cfgByKind.anima),
      sdxl: normalizeSavedFiniteNumber(cfgByKind.sdxl),
    },
    samplerByKind: {
      anima: normalizeImageSampler('anima', samplerByKind.anima),
      sdxl: normalizeImageSampler('sdxl', samplerByKind.sdxl),
    },
    lorasByKind: {
      anima: normalizeSavedLoras(lorasByKind.anima),
      sdxl: normalizeSavedLoras(lorasByKind.sdxl),
    },
  };
}

function normalizeSavedFiniteNumber(value) {
  if (value === '' || value === undefined || value === null) return '';
  const number = Number(value);
  return Number.isFinite(number) ? String(number) : '';
}

function normalizeSavedNumber(value, minimum, maximum) {
  if (value === '' || value === undefined || value === null) return '';
  const number = Number(value);
  return Number.isFinite(number) && number >= minimum && number <= maximum ? String(number) : '';
}

function normalizeSavedLoras(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  const loras = [];
  for (const item of value) {
    const id = item && typeof item.id === 'string' ? item.id : '';
    const strength = Number(item && item.strength);
    if (!id || seen.has(id) || !Number.isFinite(strength) || strength < -2 || strength > 2) continue;
    seen.add(id);
    loras.push({ id, strength });
    if (loras.length === 4) break;
  }
  return loras;
}

function normalizeImageKind(value) {
  const kind = String(value || '').toLowerCase();
  return IMAGE_KIND_VALUES.has(kind) ? kind : 'anima';
}

function normalizeImageSize(value) {
  const size = String(value || '').toLowerCase();
  return IMAGE_SIZE_VALUES.has(size) ? size : 'portrait';
}

function saveState() {
  const savedState = {
    ...state,
    settings: { ...state.settings, apiKey: '' },
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(savedState));
  } catch {
    setStatus('error', 'Browser storage is full or unavailable');
  }
}

function safeStorageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

async function apiFetch(resource, options) {
  const requestOptions = options || {};
  const response = await fetch(resource, {
    ...requestOptions,
    headers: makeApiHeaders(requestOptions.headers),
  });
  const localAccessRequired = response.status === 401
    && response.headers.get('X-Local-Access-Required') === '1';
  if (localAccessRequired) {
    clearAccessToken();
    setAccessRequired(true);
    const error = new Error(ACCESS_REQUIRED_MESSAGE);
    error.name = 'AccessRequiredError';
    throw error;
  }
  if (response.ok) setAccessRequired(false);
  return response;
}

function makeApiHeaders(headers) {
  const requestHeaders = headers || {};
  if (accessToken) return { ...requestHeaders, 'X-Access-Token': accessToken };
  return { ...requestHeaders };
}

function isAccessRequiredError(error) {
  return Boolean(error && error.name === 'AccessRequiredError');
}

function clearAccessToken() {
  accessToken = '';
  try {
    sessionStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    // The in-memory token is still cleared when session storage is blocked.
  }
}

function setAccessRequired(required) {
  accessIsRequired = Boolean(required);
  els.accessRequired.hidden = !accessIsRequired;
  document.body.classList.toggle('access-is-required', accessIsRequired);

  if (accessIsRequired) {
    setModelLoadingHidden(true);
    imageConfig = {
      loaded: false,
      connected: false,
      models: { anima: [], sdxl: [] },
      loras: { anima: [], sdxl: [] },
      runtime: {},
      dependencies: {},
    };
    renderAccessRequiredModelOptions();
    els.imageConnectionText.hidden = false;
    els.imageConnectionText.textContent = ACCESS_REQUIRED_MESSAGE;
    els.imageFormError.textContent = ACCESS_REQUIRED_MESSAGE;
    els.imageLoraList.replaceChildren();
    els.imageLoraHint.textContent = 'Access is required before loading LoRAs.';
    els.addImageLoraButton.disabled = true;
    els.refreshModelsButton.disabled = true;
    els.testConnectionButton.disabled = true;
    els.refreshImageModelsButton.disabled = true;
    els.generateImageButton.disabled = true;
    setStatus('error', 'Access required');
  } else {
    if (els.imageFormError.textContent === ACCESS_REQUIRED_MESSAGE) els.imageFormError.textContent = '';
    if (els.imageConnectionText.textContent === ACCESS_REQUIRED_MESSAGE) els.imageConnectionText.textContent = '';
  }

  updateModelSummary();
  setBusy();
  updateImageToolStatus();
}

function renderAccessRequiredModelOptions() {
  replaceSelectWithPlaceholder(els.modelSelect, ACCESS_REQUIRED_OPTION_TEXT);
  replaceSelectWithPlaceholder(els.imageModelSelect, ACCESS_REQUIRED_OPTION_TEXT);
}

function replaceSelectWithPlaceholder(select, text) {
  const option = document.createElement('option');
  option.value = '';
  option.textContent = text;
  select.replaceChildren(option);
  select.value = '';
  select.disabled = true;
}

function consumeAccessToken() {
  const hashParams = new URLSearchParams(window.location.hash.replace(/^#/, ''));
  const searchParams = new URLSearchParams(window.location.search);
  let storedToken = '';
  try {
    storedToken = sessionStorage.getItem(ACCESS_TOKEN_STORAGE_KEY) || '';
  } catch {
    storedToken = '';
  }
  const token = hashParams.get('access') || storedToken;
  if (token) {
    try { sessionStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token); } catch { /* Session-only access may be blocked. */ }
  }
  // Query strings can leak through browser history, logs, and referrers. Never
  // accept access credentials from them, but remove a legacy value if present.
  if (hashParams.has('access') || searchParams.has('access')) {
    hashParams.delete('access');
    searchParams.delete('access');
    const query = searchParams.toString();
    const hash = hashParams.toString();
    const cleanUrl = `${window.location.pathname}${query ? `?${query}` : ''}${hash ? `#${hash}` : ''}`;
    try {
      window.history.replaceState(null, '', cleanUrl || '/');
    } catch {
      // Some embedded and privacy-focused mobile browsers reject history writes.
      // The token is already held in tab-scoped storage, so boot can continue.
    }
  }
  return token;
}
