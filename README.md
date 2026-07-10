# Local AI Serving

A dependency-light local-network chat and NVIDIA image-generation interface designed for one shared GPU. Text runs through a user-selected OpenAI-compatible backend or a directly managed `llama.cpp` server. Image generation uses the repository's PyTorch implementation and user-selected local safetensors libraries.

The application contains no cloud inference, telemetry, model download, or prompt-upload path. It does not require a separate model-management application or workflow server.

## Requirements

- Node.js 18 or newer.
- For managed chat: a CUDA-capable `llama-server` executable and a local GGUF chat model.
- For images: an NVIDIA CUDA GPU, CUDA-enabled PyTorch, compatible local safetensors files, and the tokenizer assets installed by the setup command.

No personal filesystem locations or checkpoint identifiers are stored in the repository.

## First-time configuration

Run the interactive configuration before starting the app:

```text
npm run configure
npm run setup:image
npm start
```

`npm run configure` asks for the local resources you want to use and writes them to ignored `config.local.json`. Existing values are never printed: Enter keeps a configured value and `-` clears it. The checked-in `config.example.json` documents the supported fields without containing paths.

You can configure either:

- a managed text-server executable plus a primary GGUF file or explicitly selected GGUF library folder; or
- an external OpenAI-compatible HTTP(S) base URL on localhost or a private network.

For a selectable local model library, configure the managed text-server executable, start the app, then use **Local text models** on the host dashboard to choose the GGUF folder. The dashboard can open the computer's native folder chooser, with an absolute-path field as a fallback. It validates and saves the folder in the local configuration without returning that path to phone clients. A configured folder exposes every compatible model through the chat selector. Selecting one begins a managed warm-up and shows path-free loading progress until the engine confirms readiness. The automatic startup allowance scales with aggregate GGUF size, while `TEXT_START_TIMEOUT_MS` remains an optional explicit override. A single-file `TEXT_MODEL_PATH` environment override intentionally remains single-model mode.

Folder changes are host-only: open the dashboard on the computer running the server. The phone UI links to that setup location when it is local and otherwise explains where the change must be made. Environment-controlled model locations and disabled local configuration are read-only in the dashboard.

Image configuration selects the CUDA Python executable, the image library root, and the Anima text-encoder and VAE files. The image library root uses these folders:

```text
diffusion_models/   Anima checkpoints
checkpoints/        SDXL checkpoints
loras/              compatible adapters
```

The local file is strict and versioned. Unknown fields, relative paths, wrong file types, missing configured files, and unsupported versions fail without printing the private value. Explicit environment variables override the corresponding local field. Set `LOCAL_CONFIG_FILE` to use a different ignored configuration file or `LOCAL_CONFIG_DISABLED=1` for a hermetic launch.

## Running

`npm start` launches the server and opens the local access dashboard. To run only the server:

```text
npm run serve
```

The included command and tray shortcuts are optional Windows conveniences. Other platforms can use the npm commands directly.

The managed text backend starts lazily, uses a single generation slot, and is stopped before image inference so both engines never contend for the GPU. It restarts only when text is requested. Image jobs are serialized, and the image worker remains warm briefly to reuse loaded weights.

The browser settings can override the configured text URL for the current site. Server-side checks reject embedded credentials, link-local/metadata targets, public or mixed DNS answers, and redirects. Hostnames are resolved once per request, every answer must be loopback or private-LAN space, and the connection is pinned to a validated address so DNS rebinding cannot change the destination.

## Advanced environment overrides

Environment variables take precedence over `config.local.json`. Machine-resource overrides include:

- `TEXT_SERVER_EXE`, `TEXT_MODEL_PATH`, `TEXT_MODELS_ROOT`, `TEXT_BASE_URL`
- `IMAGE_PYTHON`, `IMAGE_MODELS_ROOT`
- `ANIMA_TEXT_ENCODER_PATH`, `ANIMA_VAE_PATH`

Runtime and tuning overrides include `PORT`, `HOST`, `HTTPS`, `ACCESS_TOKEN`, `TLS_CERT_FILE`, `TLS_KEY_FILE`, `PRIVATE_DIAGNOSTICS`, `TEXT_CONTEXT_SIZE`, `TEXT_KV_DTYPE`, `TEXT_GPU_MARGIN_MIB`, `TEXT_THREADS`, `TEXT_BATCH_THREADS`, `IMAGE_WORKER_IDLE_MS`, and `IMAGE_WORKER_PERSISTENT`.

## Phone access

1. Start the app.
2. Open the local dashboard on the computer.
3. Scan its access QR code from a device on the same trusted network.

The QR link places the access token in the URL fragment. The browser moves it into tab-scoped session storage and removes it from the visible URL. Query-string access tokens are ignored and scrubbed. Remote API calls require the fragment-derived token.

By default, the access token changes whenever the server restarts. Existing tabs and clean bookmarks can therefore lose access; scan the current dashboard QR code again. The app shows an explicit access-required screen in this state instead of presenting text and image model lists as empty. An explicit `ACCESS_TOKEN` override can pin the token when that tradeoff is appropriate.

HTTPS is enabled by default with an in-memory self-signed certificate. To avoid disclosing the computer name or unrelated LAN/VPN interfaces before authentication, that generated certificate contains only localhost and loopback identities. A LAN device will therefore display a "connection is not private" certificate warning. Provide a locally trusted certificate through the TLS environment variables to cover the intended LAN address, or explicitly set `HTTPS=0` with the corresponding loss of transport privacy.

The default bind supports trusted-LAN phone access. Do not port-forward this service or place it behind an Internet-facing reverse proxy. If LAN access is unnecessary, set `HOST=127.0.0.1`.

## Privacy and storage

- Prompts, responses, generated images, and request bodies have no application file-writing path.
- Prompts and pixels still pass through local processes, network buffers, RAM, and GPU memory. Operating-system swap, crash dumps, endpoint software, and device diagnostics remain outside the application boundary.
- Conversations and settings are stored in browser local storage. Finished image blobs are stored in IndexedDB with a session-memory fallback.
- Web origin isolation normally prevents unrelated sites and ordinary applications from reading that browser storage; browser profiles, sync, backups, device management, and privileged access remain outside the application boundary.
- Deleting a conversation deletes its associated image blobs. Clearing site data removes browser-held conversations, settings, and images.
- Access tokens use tab-scoped session storage. Optional external-backend API keys remain in page memory and are omitted from saved browser state.
- Copy places selected text on the operating-system clipboard. Export intentionally writes selected conversation data outside browser storage.
- Default server, smoke, and structural-audit output uses anonymous aliases or ordinals and redacts local paths. Unexpected errors are generic unless `PRIVATE_DIAGNOSTICS=1` is explicitly enabled for local troubleshooting; text-engine identifiers have their own documented diagnostics opt-in.
- Common model-weight formats, local configuration, caches, and generated output directories are ignored by Git.

These boundaries describe repository behavior, not protection against a compromised browser, operating system, network, or physical device.

## Regression tests

```text
npm test
npm run test:models
npm run test:browser
npm run test:smoke
npm run test:privacy
```

The checks are split into four layers:

- `npm test` is deterministic and hardware-independent. It covers authorization, origin defenses, HTTPS, proxy streaming/failures, managed multi-model lifecycle, guarded folder setup, native-picker containment, GPU handoff contracts, persistent image-worker IPC/recovery, safetensors/tokenizers, local configuration, browser behavior, and responsive UI contracts.
- `npm run test:models` structurally audits every compatible checkpoint and adapter under the configured image root without loading weights into GPU memory. Output is anonymous unless `MODEL_AUDIT_SHOW_IDENTIFIERS=1` is explicitly set.
- `npm run test:browser` launches an installed Edge, Chrome, or Chromium build headlessly at phone and desktop viewports. It verifies current, missing, and stale QR credentials; text and image model selectors; a delayed managed-model switch with indeterminate loading and confirmed readiness; saved-backend migration; streaming chat; and a computed-style fingerprint of the recovered interface. A second isolated scenario starts the actual server, saves a synthetic GGUF folder through the real dashboard, and proves its discovered models are selectable in the app without launching inference. Use `npm run test:browser:optional` only where no supported browser is installed.
- `npm run test:smoke` runs the real configured app on isolated loopback ports. It checks authentication, static/dashboard routes, observable model loading, cold and warm managed-text completions, real model switches, stream completion, reported GPU offload, two PNG renders per available image family, text/image handoffs, restart counts, performance ceilings, and process/port cleanup.
- `npm run test:privacy` scans every tracked or publishable untracked file for machine paths, local model identifiers, credentials, model artifacts, private-network literals, and provider-specific directory assumptions. It derives comparison values from ignored local configuration but never prints those values.

Use `npm run test:regression` for deterministic plus structural checks, `npm run test:regression:full` to include the real-browser and hardware tiers, or `npm run test:regression:all-models` for every tier plus every discovered GGUF in the configured text-model folder, including models above the normal automatic compatibility cap. `npm run test:smoke:all` runs only the hardware portion of that text breadth. Image inference still runs once per family, not once per checkpoint.

Hardware-specific benchmark results and private model identifiers are intentionally not committed. Smoke output provides anonymous local timings. Performance ceilings can be overridden with the documented `SMOKE_MAX_*` environment variables when testing another workload or machine.

For a freshly initialized release repository, run `npm run test:privacy:history` after the root commit. It additionally requires one parentless commit, a single clean ref/reflog object, no replacement or unreachable history, standalone in-project Git metadata, a privacy-safe noreply author address, and a clean working tree.
