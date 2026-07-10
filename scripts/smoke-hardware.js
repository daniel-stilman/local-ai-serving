'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const zlib = require('node:zlib');
const { spawn } = require('node:child_process');
const { applyLocalConfig } = require('../local-config');

const {
  findLlamaServer,
  findTextModel,
  discoverManagedTextCatalog,
  discoverTextModelCandidates,
  isPortInUse,
} = require('../text-backend');

const ROOT = path.resolve(__dirname, '..');
const PNG_SIGNATURE = Buffer.from('89504e470d0a1a0a', 'hex');
const TEXT_MARKER = 'SMOKE_TEXT_OK';
const APP_START_TIMEOUT_MS = 20_000;
const CONFIG_TIMEOUT_MS = 60_000;
const CHAT_TIMEOUT_MS = 3 * 60_000;
const TEXT_LOAD_TIMEOUT_MS = 20 * 60_000;
const IMAGE_TIMEOUT_MS = 20 * 60_000;
const MAX_SMALL_RESPONSE_BYTES = 2 * 1024 * 1024;
const MAX_IMAGE_RESPONSE_BYTES = 60 * 1024 * 1024;
const DEFAULT_SMOKE_BUDGETS = Object.freeze({
  textColdMs: 30_000,
  textWarmMs: 15_000,
  textRestartMs: 30_000,
  imageColdMs: 45_000,
  imageWarmMs: 20_000,
  sdxlDpmppSdeColdMs: 72_000,
  sdxlDpmppSdeWarmMs: 32_000,
});
let activeSmokeChild = null;

async function main() {
  applyLocalConfig(process.env);
  const argumentsSet = new Set(process.argv.slice(2));
  if (argumentsSet.has('--help')) {
    printHelp();
    return;
  }
  for (const argument of argumentsSet) {
    if (!['--all-text-models', '--text-only'].includes(argument)) {
      throw new Error(`Unknown smoke option: ${argument}`);
    }
  }

  const releaseLock = acquireSmokeLock();
  const removeSignalHandlers = installSignalCleanup(releaseLock);
  try {
    const executable = findLlamaServer(process.env.SMOKE_TEXT_SERVER_EXE || process.env.TEXT_SERVER_EXE);
    if (!executable) {
      throw new Error('No CUDA llama-server executable was found. Set SMOKE_TEXT_SERVER_EXE or TEXT_SERVER_EXE.');
    }
    const includeAllTextModels = argumentsSet.has('--all-text-models');
    const textModels = resolveTextModels(includeAllTextModels);
    if (!textModels.length) {
      throw new Error('No GGUF text model was found. Set SMOKE_TEXT_MODEL_PATH or TEXT_MODEL_PATH.');
    }

    const modelsRoot = resolveManagedTextRoot(textModels);
    console.log(`[smoke] Real hardware suite: ${textModels.length} managed text selection(s) in one app process.`);
    const summary = await runAppSmoke({
      executable,
      modelPaths: textModels,
      modelsRoot,
      includeImages: !argumentsSet.has('--text-only'),
    });

    console.log('[smoke] PASS - real text completion and full app handoff checks succeeded.');
    for (let index = 0; index < summary.textModels.length; index += 1) {
      const modelSummary = summary.textModels[index];
      console.log(
        `[smoke] text selection ${index + 1}/${summary.textModels.length}: `
        + `cold ${modelSummary.chatDurationsMs[0]} ms, warm ${modelSummary.chatDurationsMs[1]} ms`
        + (modelSummary.chatDurationsMs.length > 2
          ? `, restart ${modelSummary.chatDurationsMs.slice(2).join('/')} ms`
          : ''),
      );
    }
    if (summary.images.length) {
      console.log(
        `[smoke] image timings: ${summary.images.map((image) => (
          `${image.kind} ${image.durationsMs.join('/')} ms`
        )).join('; ')}; ${summary.readyCount} managed text start(s).`,
      );
    }
    if (process.env.IMAGE_PROFILE === '1') {
      for (const profile of summary.imageProfiles) {
        console.log(`[smoke] IMAGE_PROFILE ${JSON.stringify(profile)}`);
      }
    }
  } finally {
    removeSignalHandlers();
    releaseLock();
  }
}

async function runAppSmoke({ executable, modelPaths, modelsRoot, includeImages }) {
  const [appPort, textPort] = await allocateDistinctPorts(2);
  const token = crypto.randomBytes(24).toString('base64url');
  const appBaseUrl = `http://127.0.0.1:${appPort}`;
  let output = '';
  console.log('[smoke] Starting the managed multi-model catalog.');

  const appEnvironment = {
    ...process.env,
    HOST: '127.0.0.1',
    PORT: String(appPort),
    HTTPS: '0',
    ACCESS_TOKEN: token,
    ALLOW_LOCAL_BYPASS: '0',
    TEXT_BASE_URL: '',
    TEXT_BACKEND: 'managed',
    TEXT_SERVER_EXE: executable,
    TEXT_MODEL_PATH: '',
    TEXT_MODELS_ROOT: modelsRoot,
    TEXT_MODEL_ALIAS: 'smoke-text-default',
    TEXT_PORT: String(textPort),
    TEXT_START_TIMEOUT_MS: process.env.SMOKE_TEXT_START_TIMEOUT_MS || '',
    TEXT_SLEEP_IDLE_SECONDS: '3600',
    TEXT_LOG_VERBOSITY: '4',
    IMAGE_WORKER_PATH: process.env.IMAGE_WORKER_PATH || path.join(ROOT, 'inference', 'worker.py'),
    IMAGE_WORKER_PERSISTENT: '1',
    IMAGE_WORKER_IDLE_MS: '600000',
    PYTHONDONTWRITEBYTECODE: '1',
    PRIVATE_DIAGNOSTICS: '0',
  };
  const expectedCatalog = await discoverManagedTextCatalog(appEnvironment);
  const selectedModels = modelPaths.map((modelPath) => expectedCatalog.find((entry) => (
    sameLocalPath(entry.path, modelPath)
  )));
  expect(expectedCatalog.length > 0, 'The managed text catalog was unexpectedly empty.');
  expect(
    selectedModels.every(Boolean) && new Set(selectedModels.map((entry) => entry.id)).size === selectedModels.length,
    'The smoke selections did not map uniquely into the managed text catalog.',
  );
  const defaultModel = expectedCatalog.find((entry) => entry.default);

  const app = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    env: appEnvironment,
  });
  activeSmokeChild = app;
  const appendOutput = (chunk) => {
    output = (output + chunk.toString('utf8')).slice(-32_000);
  };
  app.stdout.on('data', appendOutput);
  app.stderr.on('data', appendOutput);

  const summary = { textModels: [], readyCount: 0, images: [], imageProfiles: [] };
  let primaryError = null;
  try {
    const config = await waitForApp(app, appBaseUrl, token);
    await verifyHttpSurface(appBaseUrl, token);
    expect(config.managedTextBackend?.enabled === true, 'The app did not enable its managed text backend.');
    expect(config.managedTextBackend.model === defaultModel?.id, 'The app reported the wrong managed text default.');

    const modelList = await requestJson(`${appBaseUrl}/api/models`, {
      method: 'POST',
      headers: apiHeaders(token),
      body: JSON.stringify({ baseUrl: config.defaultBaseUrl }),
    }, CONFIG_TIMEOUT_MS);
    verifyManagedCatalog(modelList, expectedCatalog);
    expect(!(await isPortInUse(textPort)), 'Listing a managed model eagerly started llama.cpp.');

    let expectedReadyCount = 0;
    for (let modelIndex = 0; modelIndex < selectedModels.length; modelIndex += 1) {
      const selected = selectedModels[modelIndex];
      const modelSummary = { chatDurationsMs: [] };
      const coldLoadMs = await runManagedTextPreload(
        appBaseUrl,
        token,
        config.defaultBaseUrl,
        selected.id,
      );
      const coldGenerationMs = await runChatCompletion(
        appBaseUrl,
        token,
        config.defaultBaseUrl,
        selected.id,
        `selection-${modelIndex + 1}-cold`,
      );
      const coldChatMs = coldLoadMs + coldGenerationMs;
      modelSummary.chatDurationsMs.push(coldChatMs);
      enforceMaximumDuration(
        coldChatMs,
        smokeBudget('SMOKE_MAX_TEXT_COLD_MS', DEFAULT_SMOKE_BUDGETS.textColdMs),
        `Text selection ${modelIndex + 1} cold completion`,
      );
      expectedReadyCount += 1;
      await waitForPortState(textPort, true, 10_000, 'managed text startup');
      await waitForReadyCount(() => output, expectedReadyCount);
      assertManagedStartCount(output, expectedReadyCount);
      await verifyDirectTextBackend(config.defaultBaseUrl, selected.id);

      const warmChatMs = await runChatCompletion(
        appBaseUrl,
        token,
        config.defaultBaseUrl,
        selected.id,
        `selection-${modelIndex + 1}-warm`,
      );
      modelSummary.chatDurationsMs.push(warmChatMs);
      enforceMaximumDuration(
        warmChatMs,
        smokeBudget('SMOKE_MAX_TEXT_WARM_MS', DEFAULT_SMOKE_BUDGETS.textWarmMs),
        `Text selection ${modelIndex + 1} warm completion`,
      );
      enforceWarmImprovement(coldChatMs, warmChatMs, `Managed text selection ${modelIndex + 1}`);
      expect(
        countReadyLogs(output) === expectedReadyCount,
        'A warm text completion unexpectedly restarted the managed text engine.',
      );
      await verifyRejectedModelSelection({
        appBaseUrl,
        token,
        textBaseUrl: config.defaultBaseUrl,
        activeModel: selected.id,
        catalogIds: expectedCatalog.map((entry) => entry.id),
        expectedReadyCount,
        readOutput: () => output,
      });

      if (modelIndex === 0 && includeImages) {
        const imageConfig = await requestJson(`${appBaseUrl}/api/image/config`, {
          headers: { 'X-Access-Token': token },
        }, CONFIG_TIMEOUT_MS);
        expect(imageConfig.connected === true && imageConfig.runtime?.ok === true, 'The CUDA image runtime is not ready.');
        expect(Boolean(imageConfig.runtime.gpu), 'The image runtime did not report a CUDA GPU.');
        const kinds = resolveImageKinds(imageConfig);
        for (const kind of kinds) {
          const model = chooseImageModel(
            imageConfig.models[kind],
            process.env[`SMOKE_${kind.toUpperCase()}_MODEL`],
          );
          if (kind === 'anima') {
            expect(
              imageConfig.dependencies?.animaTextEncoder && imageConfig.dependencies?.animaVae,
              'Anima dependencies are incomplete.',
            );
          }
          const imageSummary = await runImageFamilySmoke({
            appBaseUrl,
            token,
            textPort,
            kind,
            model,
          });
          summary.images.push(imageSummary);

          expectedReadyCount += 1;
          const restartLoadMs = await runManagedTextPreload(
            appBaseUrl,
            token,
            config.defaultBaseUrl,
            selected.id,
          );
          const restartGenerationMs = await runChatCompletion(
            appBaseUrl,
            token,
            config.defaultBaseUrl,
            selected.id,
            `selection-1-after-${kind}`,
          );
          const restartChatMs = restartLoadMs + restartGenerationMs;
          modelSummary.chatDurationsMs.push(restartChatMs);
          enforceMaximumDuration(
            restartChatMs,
            smokeBudget('SMOKE_MAX_TEXT_RESTART_MS', DEFAULT_SMOKE_BUDGETS.textRestartMs),
            `Text restart after ${kind}`,
          );
          await waitForPortState(textPort, true, 10_000, `text restart after ${kind}`);
          await waitForReadyCount(() => output, expectedReadyCount);
          assertManagedStartCount(output, expectedReadyCount);
          await verifyDirectTextBackend(config.defaultBaseUrl, selected.id);
        }
      }
      summary.textModels.push(modelSummary);
    }
    summary.readyCount = countReadyLogs(output);
  } catch (error) {
    primaryError = error;
  } finally {
    try {
      await stopChild(app, 12_000);
      await waitForPortState(appPort, false, 5_000, 'app shutdown');
      await waitForPortState(textPort, false, 5_000, 'managed text shutdown');
    } catch (cleanupError) {
      if (primaryError) primaryError.message += `\nCleanup failure: ${cleanupError.message}`;
      else primaryError = cleanupError;
    } finally {
      if (activeSmokeChild === app) activeSmokeChild = null;
    }
  }

  if (primaryError) {
    const sensitiveModelValues = expectedCatalog.flatMap((entry) => [
      entry.id,
      entry.path,
      path.basename(entry.path),
    ]);
    const diagnosticTail = redactLocalDiagnostics(output.trim().slice(-4000), sensitiveModelValues);
    if (diagnosticTail) primaryError.message += `\nApp diagnostics (tail):\n${diagnosticTail}`;
    throw primaryError;
  }
  summary.imageProfiles = extractImageProfiles(output);
  return summary;
}

function extractImageProfiles(output) {
  const profiles = [];
  for (const line of String(output || '').split(/\r?\n/)) {
    const marker = 'IMAGE_PROFILE ';
    const index = line.indexOf(marker);
    if (index === -1) continue;
    try {
      const profile = JSON.parse(line.slice(index + marker.length));
      if (
        profile
        && ['anima', 'anima-warm', 'sdxl', 'sdxl-warm'].includes(profile.pipeline)
        && profile.stagesSeconds
        && typeof profile.stagesSeconds === 'object'
        && Number.isFinite(profile.totalSeconds)
        && Number.isFinite(profile.peakVramMiB)
      ) profiles.push(profile);
    } catch {
      // Ignore unrelated or truncated diagnostics.
    }
  }
  return profiles;
}

async function verifyHttpSurface(appBaseUrl, token) {
  const unauthorized = await requestRaw(`${appBaseUrl}/api/config`, {}, CONFIG_TIMEOUT_MS, MAX_SMALL_RESPONSE_BYTES);
  expect(unauthorized.status === 401, 'An unauthenticated API request was not rejected.');

  const shell = await requestRaw(`${appBaseUrl}/`, {}, CONFIG_TIMEOUT_MS, MAX_SMALL_RESPONSE_BYTES);
  expect(shell.status === 200 && shell.text.includes('id="promptInput"'), 'The main app shell is incomplete.');
  expect(shell.headers.get('content-security-policy')?.includes("default-src 'self'"), 'The app shell lacks its CSP.');

  const dashboard = await requestRaw(`${appBaseUrl}/dashboard`, {}, CONFIG_TIMEOUT_MS, MAX_SMALL_RESPONSE_BYTES);
  expect(dashboard.status === 200 && dashboard.text.includes('id="qrCode"'), 'The local dashboard is incomplete.');

  const accessInfo = await requestJson(`${appBaseUrl}/api/access-info`, {}, CONFIG_TIMEOUT_MS);
  expect(accessInfo.tokenLength === token.length, 'The access dashboard reported the wrong token length.');

  const styles = await requestRaw(`${appBaseUrl}/styles.css`, { method: 'HEAD' }, CONFIG_TIMEOUT_MS, 1);
  expect(styles.status === 200, 'The static HEAD route failed.');
}

async function runChatCompletion(appBaseUrl, token, textBaseUrl, alias, phase) {
  const startedAt = Date.now();
  const response = await fetch(`${appBaseUrl}/api/chat`, {
    method: 'POST',
    headers: apiHeaders(token),
    body: JSON.stringify({
      baseUrl: textBaseUrl,
      model: alias,
      messages: [
        { role: 'system', content: `Reply with exactly ${TEXT_MARKER} and no other text.` },
        { role: 'user', content: `Return the smoke marker for phase ${phase}.` },
      ],
      thinkingMode: 'off',
      temperature: 0,
      max_tokens: normalizeInteger(process.env.SMOKE_TEXT_MAX_TOKENS, 64, 4096, 512),
    }),
    signal: AbortSignal.timeout(normalizeInteger(process.env.SMOKE_CHAT_TIMEOUT_MS, 5_000, 20 * 60_000, CHAT_TIMEOUT_MS)),
    redirect: 'error',
  });
  expect(response.status === 200, `Chat returned HTTP ${response.status}.`);
  expect(response.headers.get('content-type')?.includes('text/event-stream'), 'Chat did not return an SSE stream.');
  const transcript = (await readBodyBounded(response, MAX_SMALL_RESPONSE_BYTES)).toString('utf8');
  const parsed = parseSseTranscript(transcript);
  validateTextCompletion(parsed, normalizeBoolean(process.env.SMOKE_REQUIRE_TEXT_MARKER, true));
  return Date.now() - startedAt;
}

async function runManagedTextPreload(appBaseUrl, token, textBaseUrl, alias) {
  const startedAt = Date.now();
  let settled = false;
  const loadOutcome = requestRaw(`${appBaseUrl}/api/text/load`, {
    method: 'POST',
    headers: apiHeaders(token),
    body: JSON.stringify({ baseUrl: textBaseUrl, model: alias }),
  }, normalizeInteger(
    process.env.SMOKE_TEXT_LOAD_TIMEOUT_MS,
    5_000,
    60 * 60_000,
    TEXT_LOAD_TIMEOUT_MS,
  ), MAX_SMALL_RESPONSE_BYTES).then(
    (value) => {
      settled = true;
      return { value };
    },
    (error) => {
      settled = true;
      return { error };
    },
  );

  let observedLoading = false;
  const observationDeadline = Date.now() + 15_000;
  while (Date.now() < observationDeadline && !settled) {
    const status = await requestJson(`${appBaseUrl}/api/text/status`, {
      headers: { 'X-Access-Token': token },
    }, CONFIG_TIMEOUT_MS);
    if (status.managed === true && status.state === 'loading') {
      observedLoading = true;
      break;
    }
    await delay(50);
  }

  const outcome = await loadOutcome;
  if (outcome.error) throw outcome.error;
  expect(observedLoading, 'A real managed model load never exposed its loading state.');
  expect(outcome.value.status === 200, `Managed model preload failed with HTTP ${outcome.value.status}.`);
  let result;
  try {
    result = JSON.parse(outcome.value.text);
  } catch {
    throw new Error('Managed model preload returned invalid JSON.');
  }
  expect(
    result.managed === true && result.state === 'ready',
    'Managed model preload did not finish in a ready state.',
  );
  const finalStatus = await requestJson(`${appBaseUrl}/api/text/status`, {
    headers: { 'X-Access-Token': token },
  }, CONFIG_TIMEOUT_MS);
  expect(
    finalStatus.managed === true && finalStatus.state === 'ready',
    'Managed model status did not remain ready after preload.',
  );
  return Date.now() - startedAt;
}

function validateTextCompletion(parsed, requireMarker = true) {
  expect(parsed.done, 'The text stream ended without [DONE].');
  expect(parsed.events > 0, 'The text backend returned no stream events.');
  expect(
    parsed.content.trim().length > 0,
    `The text backend generated no visible response text (${parsed.reasoning.length} reasoning characters).`,
  );
  if (requireMarker) {
    expect(
      parsed.content.toUpperCase().includes(TEXT_MARKER),
      'The visible generated text did not contain the smoke marker.',
    );
  }
}

async function verifyDirectTextBackend(baseUrl, alias) {
  const parsed = new URL(baseUrl);
  const health = await requestRaw(`${parsed.origin}/health`, {}, CONFIG_TIMEOUT_MS, MAX_SMALL_RESPONSE_BYTES);
  expect(health.status === 200, 'The managed llama.cpp health endpoint is not ready.');
  const models = await requestJson(`${baseUrl}/models`, {}, CONFIG_TIMEOUT_MS);
  expect(models.data?.some((model) => model?.id === alias), 'The direct llama.cpp model list has the wrong alias.');
}

function verifyManagedCatalog(modelList, expectedCatalog) {
  expect(Array.isArray(modelList?.data), 'The app did not return a managed text model catalog.');
  const returnedIds = modelList.data.map((model) => model?.id);
  const expectedIds = expectedCatalog.map((model) => model.id);
  expect(
    returnedIds.every((id) => typeof id === 'string' && id.length > 0)
      && new Set(returnedIds).size === returnedIds.length,
    'The app returned an invalid managed text model catalog.',
  );
  expect(
    returnedIds.length === expectedIds.length
      && expectedIds.every((id) => returnedIds.includes(id)),
    'The app model list did not match the complete managed text catalog.',
  );
}

async function verifyRejectedModelSelection({
  appBaseUrl,
  token,
  textBaseUrl,
  activeModel,
  catalogIds,
  expectedReadyCount,
  readOutput,
}) {
  let unknownModel = 'smoke-unknown-selection';
  while (catalogIds.includes(unknownModel)) unknownModel += '-unused';
  const response = await requestRaw(`${appBaseUrl}/api/chat`, {
    method: 'POST',
    headers: apiHeaders(token),
    body: JSON.stringify({
      baseUrl: textBaseUrl,
      model: unknownModel,
      messages: [{ role: 'user', content: 'This synthetic selection must be rejected.' }],
      thinkingMode: 'off',
      temperature: 0,
      max_tokens: 1,
    }),
  }, CONFIG_TIMEOUT_MS, MAX_SMALL_RESPONSE_BYTES);
  expect(response.status === 400, 'An unknown managed text selection was not rejected safely.');
  expect(
    countReadyLogs(readOutput()) === expectedReadyCount,
    'Rejecting an unknown managed text selection restarted the active engine.',
  );
  await verifyDirectTextBackend(textBaseUrl, activeModel);
}

async function runImageFamilySmoke({ appBaseUrl, token, textPort, kind, model }) {
  const repeats = normalizeInteger(process.env.SMOKE_IMAGE_REPEATS, 1, 4, 2);
  const steps = normalizeInteger(process.env.SMOKE_IMAGE_STEPS, 1, 80, 2);
  const size = ['portrait', 'square', 'landscape'].includes(process.env.SMOKE_IMAGE_SIZE)
    ? process.env.SMOKE_IMAGE_SIZE
    : 'square';
  const expectedDimensions = {
    portrait: [832, 1216],
    square: [1024, 1024],
    landscape: [1216, 832],
  }[size];
  const hashes = [];
  const durationsMs = [];
  for (let index = 0; index < repeats; index += 1) {
    const seed = 8_675_309 + index;
    const startedAt = Date.now();
    const imageOutcome = requestJson(`${appBaseUrl}/api/image/generate`, {
      method: 'POST',
      headers: apiHeaders(token),
      body: JSON.stringify({
        kind,
        model: model.id,
        prompt: `Hardware regression image ${index + 1}: a ceramic fox under studio lighting.`,
        size,
        seed,
        steps,
        cfg: model.recommendedCfg,
        loras: [],
      }),
    }, normalizeInteger(process.env.SMOKE_IMAGE_TIMEOUT_MS, 30_000, 60 * 60_000, IMAGE_TIMEOUT_MS), MAX_IMAGE_RESPONSE_BYTES)
      .then((value) => ({ value }), (error) => ({ error }));
    if (index === 0) {
      const first = await Promise.race([
        waitForPortState(textPort, false, 15_000, `text stop before ${kind}`)
          .then(() => 'text-stopped'),
        imageOutcome.then(() => 'image-settled'),
      ]);
      if (first !== 'text-stopped') {
        const earlyOutcome = await imageOutcome;
        if (earlyOutcome.error) throw earlyOutcome.error;
        throw new Error(`${kind} completed before the managed text handoff could be observed.`);
      }
    }
    const outcome = await imageOutcome;
    if (outcome.error) throw outcome.error;
    const result = outcome.value;
    const durationMs = Date.now() - startedAt;
    durationsMs.push(durationMs);
    const phase = index === 0 ? 'COLD' : 'WARM';
    enforceMaximumDuration(
      durationMs,
      resolveImageBudget(kind, phase),
      `${kind} ${phase.toLowerCase()} image`,
    );
    expect(result.kind === kind && result.model === model.id, `${kind} returned the wrong model metadata.`);
    expect(result.seed === seed && result.steps === steps, `${kind} returned the wrong generation settings.`);
    expect(
      result.sampler === (kind === 'sdxl' ? 'dpmpp_sde_karras' : 'flow_euler'),
      `${kind} returned the wrong default sampler.`,
    );
    expect(result.mimeType === 'image/png', `${kind} did not return a PNG.`);
    const png = inspectPngBase64(result.imageBase64);
    expect(
      png.width === expectedDimensions[0] && png.height === expectedDimensions[1],
      `${kind} returned ${png.width}x${png.height}; expected ${expectedDimensions.join('x')}.`,
    );
    hashes.push(png.sha256);
  }
  if (hashes.length > 1) {
    expect(new Set(hashes).size === hashes.length, `${kind} ignored different smoke-test seeds.`);
    enforceWarmImprovement(durationsMs[0], durationsMs[1], `${kind} image`);
  }
  return { kind, model: model.id, durationsMs, hashes };
}

function resolveImageKinds(imageConfig) {
  const configured = String(process.env.SMOKE_IMAGE_KINDS || 'anima,sdxl')
    .split(',')
    .map((kind) => kind.trim().toLowerCase())
    .filter(Boolean);
  const allowPartial = normalizeBoolean(process.env.SMOKE_ALLOW_PARTIAL, false);
  const result = [];
  for (const kind of configured) {
    if (!['anima', 'sdxl'].includes(kind)) throw new Error(`Unsupported SMOKE_IMAGE_KINDS entry: ${kind}`);
    if (Array.isArray(imageConfig.models?.[kind]) && imageConfig.models[kind].length) {
      result.push(kind);
    } else if (!allowPartial) {
      throw new Error(`No compatible ${kind} model is available for the hardware smoke.`);
    } else {
      console.log(`[smoke] SKIP - no compatible ${kind} model is available.`);
    }
  }
  if (!result.length) throw new Error('No image model family is available for the full-app smoke.');
  return result;
}

function chooseImageModel(models, configuredId = '') {
  if (!Array.isArray(models) || !models.length) throw new Error('No compatible image model is available.');
  if (configuredId) {
    const configured = models.find((model) => model.id === configuredId);
    if (!configured) throw new Error('The configured smoke image model is unavailable.');
    return configured;
  }
  return [...models].sort((left, right) => (
    Number(left.recommendedSteps) - Number(right.recommendedSteps)
    || String(left.id).localeCompare(String(right.id), undefined, { numeric: true })
  ))[0];
}

function parseSseTranscript(transcript) {
  let done = false;
  let events = 0;
  let content = '';
  let reasoning = '';
  for (const line of String(transcript).split(/\r?\n/)) {
    if (!line.startsWith('data:')) continue;
    const data = line.slice(5).trim();
    if (!data) continue;
    if (data === '[DONE]') {
      done = true;
      continue;
    }
    let payload;
    try {
      payload = JSON.parse(data);
    } catch {
      throw new Error('The text backend returned invalid SSE JSON.');
    }
    if (payload?.error) {
      throw new Error('The text backend reported a stream error.');
    }
    events += 1;
    for (const choice of Array.isArray(payload?.choices) ? payload.choices : []) {
      const delta = choice?.delta || {};
      if (typeof delta.content === 'string') content += delta.content;
      for (const field of ['reasoning_content', 'reasoningContent', 'reasoning']) {
        if (typeof delta[field] === 'string') reasoning += delta[field];
      }
    }
  }
  return { done, events, content, reasoning };
}

function inspectPngBase64(imageBase64) {
  if (typeof imageBase64 !== 'string' || !imageBase64.length || !/^[A-Za-z0-9+/]+={0,2}$/.test(imageBase64)) {
    throw new Error('The image response contained invalid base64.');
  }
  const bytes = Buffer.from(imageBase64, 'base64');
  if (bytes.length < 45 || !bytes.subarray(0, 8).equals(PNG_SIGNATURE)) {
    throw new Error('The image response is not a valid PNG stream.');
  }
  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = -1;
  let compressionMethod = -1;
  let filterMethod = -1;
  let interlaceMethod = -1;
  let sawHeader = false;
  let sawEnd = false;
  const imageDataChunks = [];
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    const chunkEnd = offset + 12 + length;
    if (chunkEnd > bytes.length) throw new Error('The PNG contains a truncated chunk.');
    const expectedCrc = bytes.readUInt32BE(offset + 8 + length);
    const actualCrc = pngCrc32(bytes.subarray(offset + 4, offset + 8 + length));
    if (expectedCrc !== actualCrc) throw new Error('The PNG contains a corrupt chunk.');
    if (!sawHeader && type !== 'IHDR') throw new Error('The PNG does not begin with IHDR.');
    if (type === 'IHDR') {
      if (sawHeader) throw new Error('The PNG contains more than one IHDR chunk.');
      if (length !== 13) throw new Error('The PNG has an invalid IHDR chunk.');
      sawHeader = true;
      width = bytes.readUInt32BE(offset + 8);
      height = bytes.readUInt32BE(offset + 12);
      bitDepth = bytes[offset + 16];
      colorType = bytes[offset + 17];
      compressionMethod = bytes[offset + 18];
      filterMethod = bytes[offset + 19];
      interlaceMethod = bytes[offset + 20];
    }
    if (type === 'IDAT') {
      if (!sawHeader) throw new Error('The PNG contains image data before its header.');
      imageDataChunks.push(bytes.subarray(offset + 8, offset + 8 + length));
    }
    if (type === 'IEND') {
      if (length !== 0) throw new Error('The PNG has an invalid IEND chunk.');
      sawEnd = true;
      offset = chunkEnd;
      break;
    }
    offset = chunkEnd;
  }
  if (!width || !height || !sawEnd) throw new Error('The PNG is missing IHDR or IEND.');
  if (offset !== bytes.length) throw new Error('The PNG contains trailing data after IEND.');
  const content = inspectPngPixels({
    width,
    height,
    bitDepth,
    colorType,
    compressionMethod,
    filterMethod,
    interlaceMethod,
    imageDataChunks,
  });
  return {
    width,
    height,
    bytes: bytes.length,
    sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
    ...content,
  };
}

function inspectPngPixels({
  width,
  height,
  bitDepth,
  colorType,
  compressionMethod,
  filterMethod,
  interlaceMethod,
  imageDataChunks,
}) {
  if (bitDepth !== 8 || ![0, 2, 4, 6].includes(colorType)) {
    throw new Error('The PNG uses an unsupported pixel format for smoke validation.');
  }
  if (compressionMethod !== 0 || filterMethod !== 0 || interlaceMethod !== 0) {
    throw new Error('The PNG uses an unsupported encoding for smoke validation.');
  }
  if (!imageDataChunks.length) throw new Error('The PNG contains no image data.');

  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colorType];
  const rowBytes = width * channels;
  const inflatedBytes = height * (rowBytes + 1);
  if (
    width > 16_384
    || height > 16_384
    || !Number.isSafeInteger(inflatedBytes)
    || inflatedBytes > 128 * 1024 * 1024
  ) {
    throw new Error('The PNG dimensions exceed the smoke decoder safety limit.');
  }

  let filtered;
  try {
    filtered = zlib.inflateSync(Buffer.concat(imageDataChunks), { maxOutputLength: inflatedBytes });
  } catch {
    throw new Error('The PNG contains invalid compressed image data.');
  }
  if (filtered.length !== inflatedBytes) throw new Error('The PNG image data has an invalid length.');

  const pixels = unfilterPngRows(filtered, width, height, channels);
  return analyzePngPixels(pixels, width, height, colorType, channels);
}

function unfilterPngRows(filtered, width, height, bytesPerPixel) {
  const rowBytes = width * bytesPerPixel;
  const pixels = Buffer.allocUnsafe(rowBytes * height);
  for (let y = 0; y < height; y += 1) {
    const filteredOffset = y * (rowBytes + 1);
    const outputOffset = y * rowBytes;
    const filter = filtered[filteredOffset];
    if (filter > 4) throw new Error('The PNG contains an invalid row filter.');
    for (let x = 0; x < rowBytes; x += 1) {
      const encoded = filtered[filteredOffset + 1 + x];
      const left = x >= bytesPerPixel ? pixels[outputOffset + x - bytesPerPixel] : 0;
      const up = y > 0 ? pixels[outputOffset - rowBytes + x] : 0;
      const upperLeft = y > 0 && x >= bytesPerPixel
        ? pixels[outputOffset - rowBytes + x - bytesPerPixel]
        : 0;
      let predictor = 0;
      if (filter === 1) predictor = left;
      else if (filter === 2) predictor = up;
      else if (filter === 3) predictor = Math.floor((left + up) / 2);
      else if (filter === 4) predictor = paethPredictor(left, up, upperLeft);
      pixels[outputOffset + x] = (encoded + predictor) & 0xff;
    }
  }
  return pixels;
}

function paethPredictor(left, up, upperLeft) {
  const estimate = left + up - upperLeft;
  const leftDistance = Math.abs(estimate - left);
  const upDistance = Math.abs(estimate - up);
  const upperLeftDistance = Math.abs(estimate - upperLeft);
  if (leftDistance <= upDistance && leftDistance <= upperLeftDistance) return left;
  return upDistance <= upperLeftDistance ? up : upperLeft;
}

function analyzePngPixels(pixels, width, height, colorType, channels) {
  const pixelCount = width * height;
  const stride = Math.max(1, Math.floor(pixelCount / 65_536));
  const minima = [255, 255, 255];
  const maxima = [0, 0, 0];
  let samples = 0;
  let visibleSamples = 0;
  let alphaTotal = 0;
  let luminanceTotal = 0;
  let luminanceSquaredTotal = 0;
  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += stride) {
    const offset = pixelIndex * channels;
    let red;
    let green;
    let blue;
    let alpha = 255;
    if (colorType === 0 || colorType === 4) {
      red = pixels[offset];
      green = red;
      blue = red;
      if (colorType === 4) alpha = pixels[offset + 1];
    } else {
      red = pixels[offset];
      green = pixels[offset + 1];
      blue = pixels[offset + 2];
      if (colorType === 6) alpha = pixels[offset + 3];
    }
    samples += 1;
    alphaTotal += alpha;
    if (alpha <= 16) continue;
    visibleSamples += 1;
    minima[0] = Math.min(minima[0], red);
    minima[1] = Math.min(minima[1], green);
    minima[2] = Math.min(minima[2], blue);
    maxima[0] = Math.max(maxima[0], red);
    maxima[1] = Math.max(maxima[1], green);
    maxima[2] = Math.max(maxima[2], blue);
    const luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    luminanceTotal += luminance;
    luminanceSquaredTotal += luminance * luminance;
  }

  const visibleFraction = visibleSamples / samples;
  const meanAlpha = alphaTotal / samples;
  if (visibleFraction < 0.1 || meanAlpha < 32) {
    throw new Error('The generated PNG is blank or predominantly transparent.');
  }
  const meanLuminance = luminanceTotal / visibleSamples;
  const luminanceVariance = Math.max(0, luminanceSquaredTotal / visibleSamples - meanLuminance ** 2);
  const channelRange = Math.max(...maxima.map((maximum, index) => maximum - minima[index]));
  if (visibleSamples < 2 || (luminanceVariance < 0.5 && channelRange < 4)) {
    throw new Error('The generated PNG is blank or near-uniform.');
  }
  return {
    sampleCount: samples,
    visibleFraction: Number(visibleFraction.toFixed(4)),
    luminanceVariance: Number(luminanceVariance.toFixed(4)),
    channelRange,
  };
}

function pngCrc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function resolveTextModels(includeAll) {
  const explicitPrimary = process.env.SMOKE_TEXT_MODEL_PATH || process.env.TEXT_MODEL_PATH || '';
  const configuredRoot = process.env.SMOKE_TEXT_MODELS_ROOT
    || process.env.TEXT_MODELS_ROOT
    || (explicitPrimary ? path.dirname(path.resolve(explicitPrimary)) : '');
  const modelsRoot = configuredRoot ? path.resolve(configuredRoot) : '';
  const maximumBytes = normalizeNumber(process.env.TEXT_MODEL_MAX_GIB, 1, 100, 10) * 1024 ** 3;
  const models = [];
  const add = (candidate) => {
    if (!candidate) return;
    const resolved = path.resolve(candidate);
    if (!fs.existsSync(resolved) || path.extname(resolved).toLowerCase() !== '.gguf') {
      throw new Error('A configured smoke text model is missing or is not a GGUF file.');
    }
    if (!models.some((model) => model.toLowerCase() === resolved.toLowerCase())) models.push(resolved);
  };
  add(findTextModel(explicitPrimary, process.env, modelsRoot));
  for (const configured of String(process.env.SMOKE_TEXT_MODELS || '').split(';').filter(Boolean)) add(configured.trim());
  if (modelsRoot && (includeAll || models.length < 2)) {
    const discovered = discoverTextModelCandidates(modelsRoot);
    const skipped = includeAll ? [] : discovered.filter((candidate) => candidate.size > maximumBytes);
    if (skipped.length) {
      console.log(
        `[smoke] SKIP - ${skipped.length} automatic GGUF model(s) exceed the `
        + `${formatGib(maximumBytes)} GiB compatibility cap.`,
      );
    }
    const discoveryLimit = includeAll ? Number.POSITIVE_INFINITY : maximumBytes;
    const smokeCandidates = discoverTextModels(modelsRoot, discoveryLimit);
    if (includeAll) {
      smokeCandidates.sort((left, right) => {
        const leftOverCap = left.size > maximumBytes;
        const rightOverCap = right.size > maximumBytes;
        if (leftOverCap !== rightOverCap) return leftOverCap ? -1 : 1;
        return leftOverCap
          ? left.size - right.size || left.path.localeCompare(right.path, undefined, { numeric: true })
          : right.size - left.size || left.path.localeCompare(right.path, undefined, { numeric: true });
      });
    }
    for (const candidate of smokeCandidates) {
      add(candidate.path);
      if (!includeAll && models.length >= 2) break;
    }
  }
  const defaultMaximum = includeAll ? models.length || 1 : Math.min(2, models.length) || 1;
  const maximumModels = normalizeInteger(process.env.SMOKE_MAX_TEXT_MODELS, 1, 1000, defaultMaximum);
  return models.slice(0, maximumModels);
}

function resolveManagedTextRoot(modelPaths) {
  const rootCandidates = [
    process.env.SMOKE_TEXT_MODELS_ROOT,
    process.env.TEXT_MODELS_ROOT,
    path.dirname(path.resolve(modelPaths[0])),
  ].filter(Boolean).map((candidate) => path.resolve(candidate));
  for (const rootCandidate of rootCandidates) {
    if (modelPaths.every((modelPath) => isPathInside(rootCandidate, path.resolve(modelPath)))) {
      return rootCandidate;
    }
  }
  throw new Error('Configured smoke text models must share one managed model root. Set SMOKE_TEXT_MODELS_ROOT.');
}

function isPathInside(parentPath, candidatePath) {
  const relative = path.relative(parentPath, candidatePath);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function sameLocalPath(left, right) {
  const normalizedLeft = path.resolve(left);
  const normalizedRight = path.resolve(right);
  return process.platform === 'win32'
    ? normalizedLeft.toLowerCase() === normalizedRight.toLowerCase()
    : normalizedLeft === normalizedRight;
}

function discoverTextModels(modelsRoot, maximumBytes) {
  const candidates = discoverTextModelCandidates(modelsRoot)
    .filter((candidate) => candidate.size <= maximumBytes);
  candidates.sort((left, right) => right.size - left.size || left.path.localeCompare(right.path, undefined, { numeric: true }));
  return candidates;
}

function formatGib(bytes) {
  return (bytes / 1024 ** 3).toFixed(1).replace(/\.0$/, '');
}

function redactLocalDiagnostics(value, sensitiveValues = []) {
  let redacted = String(value)
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, '[local-address]')
    .replace(/[A-Za-z]:\\[^\r\n]*/g, '[local-path]')
    .replace(/(^|\s)\/(?:[^/\s]+\/){2,}[^\s]*/gm, '$1[local-path]');
  for (const sensitiveValue of sensitiveValues) {
    if (typeof sensitiveValue !== 'string' || !sensitiveValue) continue;
    redacted = redacted.replace(new RegExp(escapeRegExp(sensitiveValue), 'gi'), '[private-model]');
  }
  return redacted;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function waitForApp(child, baseUrl, token) {
  const deadline = Date.now() + APP_START_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (hasExited(child)) {
      throw new Error('The app exited during startup.');
    }
    try {
      return await requestJson(`${baseUrl}/api/config`, {
        headers: { 'X-Access-Token': token },
      }, 2000);
    } catch {
      await delay(100);
    }
  }
  throw new Error('Timed out waiting for the app server.');
}

async function requestJson(url, options = {}, timeoutMs = CONFIG_TIMEOUT_MS, maximumBytes = MAX_SMALL_RESPONSE_BYTES) {
  const response = await requestRaw(url, options, timeoutMs, maximumBytes);
  let payload;
  try {
    payload = response.text ? JSON.parse(response.text) : {};
  } catch {
    throw new Error(`${new URL(url).pathname} returned invalid JSON.`);
  }
  if (response.status < 200 || response.status >= 300) {
    throw new Error(`${new URL(url).pathname} failed with HTTP ${response.status}.`);
  }
  return payload;
}

async function requestRaw(url, options = {}, timeoutMs = CONFIG_TIMEOUT_MS, maximumBytes = MAX_SMALL_RESPONSE_BYTES) {
  const response = await fetch(url, {
    ...options,
    signal: AbortSignal.timeout(timeoutMs),
    redirect: 'error',
  });
  const bytes = response.body ? await readBodyBounded(response, maximumBytes) : Buffer.alloc(0);
  return { status: response.status, headers: response.headers, text: bytes.toString('utf8') };
}

async function readBodyBounded(response, maximumBytes) {
  const chunks = [];
  let total = 0;
  for await (const chunk of response.body || []) {
    const bytes = Buffer.from(chunk);
    total += bytes.length;
    if (total > maximumBytes) throw new Error(`Response exceeded ${maximumBytes} bytes.`);
    chunks.push(bytes);
  }
  return Buffer.concat(chunks, total);
}

function apiHeaders(token) {
  return { 'Content-Type': 'application/json', 'X-Access-Token': token };
}

async function allocateDistinctPorts(count) {
  const reservations = [];
  try {
    for (let index = 0; index < count; index += 1) {
      const server = net.createServer();
      await new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(0, '127.0.0.1', resolve);
      });
      reservations.push(server);
    }
    return reservations.map((server) => server.address().port);
  } finally {
    await Promise.all(reservations.map((server) => new Promise((resolve) => server.close(resolve))));
  }
}

async function waitForPortState(port, expected, timeoutMs, description) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if ((await isPortInUse(port)) === expected) return;
    await delay(100);
  }
  throw new Error(`Timed out waiting for ${description} (port ${port} ${expected ? 'open' : 'closed'}).`);
}

async function waitForReadyCount(readOutput, expected) {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (countReadyLogs(readOutput()) >= expected) return;
    await delay(50);
  }
  throw new Error(`Managed text backend did not report startup ${expected} time(s).`);
}

function countReadyLogs(output) {
  return (String(output).match(/Direct text engine ready(?:\s|\()/g) || []).length;
}

function assertManagedStartCount(output, expected) {
  expect(countReadyLogs(output) === expected, `Expected exactly ${expected} managed text start(s).`);
  const offloads = [...String(output).matchAll(/GPU offload confirmed:\s*(\d+)\s*\/\s*(\d+)\s+layers/gi)];
  expect(offloads.length === expected, 'A managed text start did not confirm GPU layer offload.');
  for (const match of offloads) {
    const offloaded = Number(match[1]);
    const total = Number(match[2]);
    expect(offloaded > 0 && total >= offloaded, 'The managed text backend reported invalid GPU layer counts.');
    if (normalizeBoolean(process.env.SMOKE_REQUIRE_FULL_TEXT_GPU, true)) {
      expect(offloaded === total, `Only ${offloaded}/${total} text layers were offloaded to the GPU.`);
    }
  }
}

async function stopChild(child, timeoutMs) {
  if (!child || hasExited(child)) return;
  const exited = new Promise((resolve) => child.once('exit', resolve));
  if (child.connected) {
    try { child.send({ type: 'shutdown' }); } catch {}
  } else if (process.platform !== 'win32') {
    child.kill('SIGTERM');
  }
  if (await settlesWithin(exited, timeoutMs) || hasExited(child)) return;
  await forceKillProcessTree(child);
  await settlesWithin(exited, 3000);
  if (!hasExited(child)) throw new Error('The app process could not be stopped.');
}

async function forceKillProcessTree(child) {
  if (!child || hasExited(child)) return;
  if (process.platform === 'win32') {
    const killer = spawn('taskkill', ['/pid', String(child.pid), '/t', '/f'], {
      windowsHide: true,
      stdio: 'ignore',
    });
    await new Promise((resolve) => killer.once('exit', resolve));
    return;
  }
  child.kill('SIGKILL');
}

function installSignalCleanup(releaseLock) {
  let stopping = false;
  const handlers = new Map();
  for (const signal of ['SIGINT', 'SIGTERM']) {
    const handler = () => {
      if (stopping) return;
      stopping = true;
      stopChild(activeSmokeChild, 12_000)
        .catch(() => {})
        .finally(() => {
          releaseLock();
          process.exit(signal === 'SIGINT' ? 130 : 143);
        });
    };
    handlers.set(signal, handler);
    process.once(signal, handler);
  }
  return () => {
    for (const [signal, handler] of handlers) process.removeListener(signal, handler);
  };
}

function settlesWithin(promise, milliseconds) {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(false), milliseconds);
    promise.then(() => {
      clearTimeout(timeout);
      resolve(true);
    }, () => {
      clearTimeout(timeout);
      resolve(true);
    });
  });
}

function acquireSmokeLock() {
  const lockPath = path.join(os.tmpdir(), 'local-ai-serving-hardware-smoke.lock');
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const handle = fs.openSync(lockPath, 'wx');
      fs.writeFileSync(handle, JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString() }));
      return () => {
        try { fs.closeSync(handle); } catch {}
        try { fs.rmSync(lockPath, { force: true }); } catch {}
      };
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      let owner = null;
      try { owner = JSON.parse(fs.readFileSync(lockPath, 'utf8')); } catch {}
      if (owner?.pid && isProcessAlive(owner.pid)) {
        throw new Error(`Another hardware smoke is running (PID ${owner.pid}).`);
      }
      fs.rmSync(lockPath, { force: true });
    }
  }
  throw new Error('Could not acquire the hardware smoke lock.');
}

function isProcessAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function hasExited(child) {
  return child.exitCode !== null || child.signalCode !== null;
}

function normalizeBoolean(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  return !['0', 'false', 'off', 'no'].includes(String(value).trim().toLowerCase());
}

function normalizeNumber(value, minimum, maximum, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
}

function normalizeInteger(value, minimum, maximum, fallback) {
  return Math.round(normalizeNumber(value, minimum, maximum, fallback));
}

function smokeBudget(name, fallback) {
  return normalizeInteger(process.env[name], 1_000, 60 * 60_000, fallback);
}

function resolveImageBudget(kind, phase, environment = process.env) {
  const normalizedPhase = String(phase).toUpperCase();
  const isDefaultSdxlSde = String(kind).toLowerCase() === 'sdxl';
  const fallback = isDefaultSdxlSde
    ? (normalizedPhase === 'COLD'
      ? DEFAULT_SMOKE_BUDGETS.sdxlDpmppSdeColdMs
      : DEFAULT_SMOKE_BUDGETS.sdxlDpmppSdeWarmMs)
    : (normalizedPhase === 'COLD'
      ? DEFAULT_SMOKE_BUDGETS.imageColdMs
      : DEFAULT_SMOKE_BUDGETS.imageWarmMs);
  const familyBudget = environment[`SMOKE_MAX_${String(kind).toUpperCase()}_${normalizedPhase}_MS`];
  const genericBudget = environment[`SMOKE_MAX_IMAGE_${normalizedPhase}_MS`];
  return normalizeInteger(familyBudget || genericBudget, 1_000, 60 * 60_000, fallback);
}

function enforceMaximumDuration(actualMs, maximumMs, label) {
  expect(actualMs <= maximumMs, `${label} took ${actualMs} ms; the regression ceiling is ${maximumMs} ms.`);
}

function enforceWarmImprovement(coldMs, warmMs, label) {
  const maximumRatio = normalizeNumber(process.env.SMOKE_MAX_WARM_RATIO, 0.1, 2, 0.9);
  expect(
    warmMs <= coldMs * maximumRatio,
    `${label} warm time ${warmMs} ms exceeded ${maximumRatio}x its ${coldMs} ms cold time.`,
  );
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function printHelp() {
  console.log('Usage: npm run test:smoke [-- --text-only|--all-text-models]');
  console.log('Runs real managed-text completions and, by default, Anima + SDXL handoff checks.');
  console.log('Use SMOKE_TEXT_MODEL_PATH, SMOKE_TEXT_MODELS, SMOKE_MAX_TEXT_MODELS, SMOKE_ANIMA_MODEL,');
  console.log('SMOKE_SDXL_MODEL, SMOKE_IMAGE_STEPS, SMOKE_IMAGE_REPEATS, and SMOKE_IMAGE_SIZE to override selection.');
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`[smoke] FAIL - ${redactLocalDiagnostics(error.message)}`);
    process.exitCode = 1;
  });
}

module.exports = {
  chooseImageModel,
  DEFAULT_SMOKE_BUDGETS,
  discoverTextModels,
  enforceMaximumDuration,
  enforceWarmImprovement,
  extractImageProfiles,
  inspectPngBase64,
  parseSseTranscript,
  redactLocalDiagnostics,
  assertManagedStartCount,
  resolveImageBudget,
  resolveImageKinds,
  resolveManagedTextRoot,
  resolveTextModels,
  validateTextCompletion,
  verifyManagedCatalog,
};
