import { create } from "zustand";

import { APP_STORAGE_PREFIX } from "@/lib/app-config";
import {
  BATCH_JOB_STATUSES,
  RUNNING_BATCH_JOB_STATUSES,
  SETTLED_BATCH_JOB_STATUSES,
  STATUS_CONSTRAINT_FALLBACKS,
  TASK_PHASES,
  normalizeBatchJobStatus,
  normalizeStatusConstraints,
} from "@/lib/status";
import {
  clampPreviewIntervalSteps,
  clampSimpleBatchCount,
  DEFAULT_LIVE_PREVIEW_MODE,
  DEFAULT_PREVIEW_INTERVAL_STEPS,
  buildUiSettingsPatch,
  normalizeLivePreviewMode,
} from "@/lib/workspace";

export const SESSION_TOKEN_HEADER = "X-Krea-Session-Token";
export const SESSION_TOKEN_QUERY = "token";
export const SESSION_TOKEN_STORAGE_KEY = `${APP_STORAGE_PREFIX}:session-token`;
export const INITIAL_STATUS_GLOBAL = "__KREA_2_TURBO_MLX_INITIAL_STATUS__";
export const BUSY_POLL_MS = 400;
export const IDLE_POLL_MS = 1500;
export const MAX_SOURCE_IMAGE_BYTES = 32 * 1024 * 1024;
const DISCONNECTED_EVENT_MESSAGE = "Server disconnected";
const MAX_CLIENT_EVENT_HISTORY = 40;
const TASK_START_PATHS = new Set([
  "/api/generate",
  "/api/generate-batch",
  "/api/load",
]);

export const DEFAULT_STATUS = {
  server: { connected: false, status: "starting" },
  model: {
    loaded: false,
    name: "krea-2-turbo-mlx",
    status: "not loaded",
    path: "",
    precision: "",
  },
  busy: false,
  load_running: false,
  generation_running: false,
  cancel_requested: false,
  batch: null,
  phase: TASK_PHASES.STARTING,
  message: "Connecting",
  progress: 0,
  error: null,
  image: null,
  preview: null,
  recent: [],
  output_dir: { path: "outputs" },
  loras: {
    dir: "loras",
    items: [],
    warnings: [],
    scanned_at_ms: null,
  },
  ui_settings: null,
  task: {
    name: null,
    started_ms: null,
    completed_ms: null,
  },
  constraints: STATUS_CONSTRAINT_FALLBACKS,
  events: [],
};

function resolveWindowLocation(location) {
  return location ?? globalThis?.window?.location ?? null;
}

function resolveSessionStorage(storage) {
  return storage ?? globalThis?.window?.sessionStorage ?? null;
}

function resolveGlobalObject(globalObject) {
  return globalObject ?? globalThis;
}

export function readInitialStatus({ globalObject } = {}) {
  const resolvedGlobal = resolveGlobalObject(globalObject);
  const status =
    resolvedGlobal?.window?.[INITIAL_STATUS_GLOBAL] ??
    resolvedGlobal?.[INITIAL_STATUS_GLOBAL];

  if (!status || typeof status !== "object" || Array.isArray(status)) {
    return null;
  }

  return status;
}

export function readSessionToken({ location, history, storage } = {}) {
  const resolvedLocation = resolveWindowLocation(location);
  const resolvedStorage = resolveSessionStorage(storage);
  const currentUrl = resolvedLocation?.href ?? "http://127.0.0.1/?";
  const url = new URL(currentUrl, "http://127.0.0.1");
  const token = url.searchParams.get(SESSION_TOKEN_QUERY);

  if (token) {
    resolvedStorage?.setItem?.(SESSION_TOKEN_STORAGE_KEY, token);
    url.searchParams.delete(SESSION_TOKEN_QUERY);
    const nextPath = `${url.pathname}${url.search}${url.hash}`;
    const resolvedHistory = history ?? globalThis?.window?.history;
    resolvedHistory?.replaceState?.({}, "", nextPath || "/");
    return token;
  }

  return resolvedStorage?.getItem?.(SESSION_TOKEN_STORAGE_KEY) ?? "";
}

export function appendTokenToUrl(path, token) {
  if (!token) {
    return path;
  }
  const url = new URL(path, "http://127.0.0.1");
  url.searchParams.set(SESSION_TOKEN_QUERY, token);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function pollDelayForStatus(status) {
  return status?.busy || status?.load_running || status?.generation_running
    ? BUSY_POLL_MS
    : IDLE_POLL_MS;
}

function isTaskStartPath(path) {
  return TASK_START_PATHS.has(path);
}

function statusHasActiveTask(status) {
  return Boolean(
    status?.busy || status?.load_running || status?.generation_running,
  );
}

function numericTime(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function reconcileActiveTaskStartedMs(currentStartedMs, status) {
  const started = numericTime(currentStartedMs);

  if (started === null) {
    return null;
  }

  if (statusHasActiveTask(status)) {
    return started;
  }

  const completed = numericTime(status?.task?.completed_ms);
  if (completed !== null) {
    return null;
  }

  return started;
}

function statusAfterConnectionError(status, message, nowMs = Date.now()) {
  const current =
    status && typeof status === "object" ? status : DEFAULT_STATUS;

  return {
    ...current,
    server: { connected: false, status: "error" },
    busy: false,
    load_running: false,
    generation_running: false,
    cancel_requested: false,
    batch: disconnectedBatchSnapshot(current.batch),
    phase: TASK_PHASES.DISCONNECTED,
    message: DISCONNECTED_EVENT_MESSAGE,
    progress: 0,
    error: message,
    preview: null,
    events: eventsAfterConnectionError(current.events, message, nowMs),
    task: taskAfterConnectionError(current.task, nowMs),
  };
}

function taskAfterConnectionError(task, nowMs) {
  if (!task || typeof task !== "object" || Array.isArray(task)) {
    return DEFAULT_STATUS.task;
  }

  const started = numericTime(task.started_ms);
  const completed = numericTime(task.completed_ms);
  if (started === null || completed !== null) {
    return task;
  }

  return {
    ...task,
    completed_ms: Math.max(started, nowMs),
  };
}

function eventsAfterConnectionError(events, message, nowMs) {
  const closedEvents = (Array.isArray(events) ? events : []).map(
    (event, index, all) =>
      index === all.length - 1 ? closeEvent(event, nowMs) : event,
  );
  const latest = closedEvents.at(-1);

  if (isDisconnectedEvent(latest)) {
    return closedEvents.slice(-MAX_CLIENT_EVENT_HISTORY);
  }

  return [
    ...closedEvents,
    {
      id: `client-disconnected-${nowMs}`,
      kind: "error",
      stage: "server",
      message: DISCONNECTED_EVENT_MESSAGE,
      progress: null,
      details: { error: message },
      time: formatEventTime(nowMs),
      time_ms: nowMs,
      completed_ms: nowMs,
    },
  ].slice(-MAX_CLIENT_EVENT_HISTORY);
}

function closeEvent(event, nowMs) {
  if (!event || typeof event !== "object" || Array.isArray(event)) {
    return event;
  }

  if (numericTime(event.completed_ms) !== null) {
    return event;
  }

  const started = numericTime(event.time_ms);
  return {
    ...event,
    completed_ms: started === null ? nowMs : Math.max(started, nowMs),
  };
}

function isDisconnectedEvent(event) {
  return (
    event?.stage === "server" && event?.message === DISCONNECTED_EVENT_MESSAGE
  );
}

function formatEventTime(timeMs) {
  const date = new Date(timeMs);
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");

  return `${hours}:${minutes}:${seconds}`;
}

function disconnectedBatchSnapshot(batch) {
  if (!batch || typeof batch !== "object" || Array.isArray(batch)) {
    return null;
  }

  const activeIndex = numericPositive(batch.index);
  const jobs = Array.isArray(batch.jobs)
    ? batch.jobs.map((job, index) =>
        disconnectedBatchJob(
          job,
          numericPositive(job?.index) ?? index + 1,
          activeIndex,
        ),
      )
    : null;
  const nextBatch = {
    ...batch,
    interrupted: true,
    cancel_current_requested: false,
    clear_queue_requested: true,
    queue_remaining: 0,
    progress: 0,
  };

  if (jobs) {
    nextBatch.jobs = jobs;
  } else {
    nextBatch.status = disconnectedBatchJobStatus(
      batch.status,
      activeIndex,
      activeIndex,
    );
  }

  return nextBatch;
}

function disconnectedBatchJob(job, jobIndex, activeIndex) {
  if (!job || typeof job !== "object" || Array.isArray(job)) {
    return job;
  }

  return {
    ...job,
    status: disconnectedBatchJobStatus(job.status, jobIndex, activeIndex),
  };
}

function disconnectedBatchJobStatus(status, jobIndex, activeIndex) {
  const normalized = normalizeBatchJobStatus(status);
  if (SETTLED_BATCH_JOB_STATUSES.includes(normalized)) {
    return normalized;
  }
  if (RUNNING_BATCH_JOB_STATUSES.includes(normalized)) {
    return BATCH_JOB_STATUSES.CANCELLED;
  }
  if (normalized === BATCH_JOB_STATUSES.QUEUED) {
    return BATCH_JOB_STATUSES.CLEARED;
  }
  if (activeIndex !== null && jobIndex < activeIndex) {
    return BATCH_JOB_STATUSES.DONE;
  }
  if (activeIndex !== null && jobIndex === activeIndex) {
    return BATCH_JOB_STATUSES.CANCELLED;
  }

  return BATCH_JOB_STATUSES.CLEARED;
}

function numericPositive(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

export function buildGeneratePayload({
  prompt,
  width,
  height,
  steps,
  seed,
  loras,
  livePreviewMode = DEFAULT_LIVE_PREVIEW_MODE,
  previewIntervalSteps = DEFAULT_PREVIEW_INTERVAL_STEPS,
  randomizeSeed = false,
}) {
  const payload = {
    prompt: String(prompt ?? "").trim(),
    width: Number(width),
    height: Number(height),
    steps: Number(steps),
    live_preview: normalizeLivePreviewMode(livePreviewMode),
    preview_interval_steps:
      clampPreviewIntervalSteps(previewIntervalSteps).value,
  };
  if (Array.isArray(loras) && loras.length > 0) {
    payload.loras = loras;
  }
  if (randomizeSeed) {
    return payload;
  }

  const seedText = String(seed ?? "").trim();
  if (seedText !== "") {
    payload.seed = Number(seedText);
  }
  return payload;
}

export async function buildSourceImageFilePayload(file) {
  validateSourceImageFile(file);
  return {
    filename: file.name || "Image",
    image_base64: await blobToBase64(file),
  };
}

export function validateSourceImageFile(file) {
  if (!file) {
    throw new Error("Choose an image file.");
  }
  if (file.size > MAX_SOURCE_IMAGE_BYTES) {
    throw new Error(
      `Source image must be ${formatMegabytes(MAX_SOURCE_IMAGE_BYTES)} or smaller.`,
    );
  }

  const type = String(file.type ?? "").toLowerCase();
  const name = String(file.name ?? "").toLowerCase();
  if (type && !type.startsWith("image/")) {
    throw new Error("Choose an image file.");
  }
  if (!type && name && !/\.(png|jpe?g|webp)$/i.test(name)) {
    throw new Error("Choose an image file.");
  }
}

async function blobToBase64(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const chunkSize = 0x8000;
  let binary = "";

  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }

  return globalThis.btoa(binary);
}

export function parseBatchText(text) {
  const payload = JSON.parse(text);
  if (!Array.isArray(payload)) {
    throw new Error("Batch JSON must be an array.");
  }
  return payload;
}

export function buildBatchPayload(
  text,
  {
    constraints = STATUS_CONSTRAINT_FALLBACKS,
    outputDir,
    jobDefaults,
    livePreviewMode = DEFAULT_LIVE_PREVIEW_MODE,
    previewIntervalSteps = DEFAULT_PREVIEW_INTERVAL_STEPS,
  } = {},
) {
  const jobs = parseBatchText(text).map((job) =>
    resolveBatchJob(job, jobDefaults, constraints),
  );
  const outputDirText = String(outputDir ?? "").trim();
  const nextPayload = {
    jobs,
    live_preview: normalizeLivePreviewMode(livePreviewMode),
    preview_interval_steps:
      clampPreviewIntervalSteps(previewIntervalSteps).value,
  };

  if (outputDirText) {
    nextPayload.output_dir = outputDirText;
  }

  return nextPayload;
}

export function buildRepeatedBatchPayload({
  count,
  outputDir,
  prompt,
  width,
  height,
  steps,
  seed,
  generatedSeed,
  randomizeSeed = false,
  loras,
  constraints = STATUS_CONSTRAINT_FALLBACKS,
  maxBatchJobs = DEFAULT_STATUS.constraints.max_batch_jobs,
  livePreviewMode = DEFAULT_LIVE_PREVIEW_MODE,
  previewIntervalSteps = DEFAULT_PREVIEW_INTERVAL_STEPS,
} = {}) {
  const normalizedConstraints = normalizeStatusConstraints(constraints);
  const jobDefaults = {
    width,
    height,
    steps,
    seed,
    generatedSeed,
    randomizeSeed,
    loras,
  };
  const jobCount = clampSimpleBatchCount(
    count,
    finiteNumber(maxBatchJobs, normalizedConstraints.max_batch_jobs),
  ).value;
  const jobs = Array.from({ length: jobCount }, () =>
    resolveBatchJob(
      { prompt: String(prompt ?? "").trim() },
      jobDefaults,
      normalizedConstraints,
    ),
  );
  const outputDirText = String(outputDir ?? "").trim();
  const nextPayload = {
    jobs,
    live_preview: normalizeLivePreviewMode(livePreviewMode),
    preview_interval_steps:
      clampPreviewIntervalSteps(previewIntervalSteps).value,
  };

  if (outputDirText) {
    nextPayload.output_dir = outputDirText;
  }

  return nextPayload;
}

function resolveBatchJob(
  job,
  jobDefaults = {},
  constraints = STATUS_CONSTRAINT_FALLBACKS,
) {
  if (!job || typeof job !== "object" || Array.isArray(job)) {
    return job;
  }

  const normalizedConstraints = normalizeStatusConstraints(constraints);
  const resolved = { ...job };
  const defaults = {
    width: defaultNumber(
      jobDefaults.width,
      normalizedConstraints.default_width,
    ),
    height: defaultNumber(
      jobDefaults.height,
      normalizedConstraints.default_height,
    ),
    steps: defaultNumber(
      jobDefaults.steps,
      normalizedConstraints.default_steps,
    ),
  };
  const seed = defaultSeed(jobDefaults);

  for (const [key, value] of Object.entries(defaults)) {
    if (!hasOwn(job, key)) {
      resolved[key] = value;
    }
  }

  if (!hasOwn(job, "seed") && seed !== undefined) {
    resolved.seed = seed;
  }

  if (
    !hasOwn(job, "loras") &&
    Array.isArray(jobDefaults.loras) &&
    jobDefaults.loras.length > 0
  ) {
    resolved.loras = jobDefaults.loras;
  }

  return resolved;
}

function defaultNumber(value, fallback) {
  const text = String(value ?? "").trim();
  return Number(text === "" ? fallback : text);
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function defaultSeed(jobDefaults = {}) {
  if (jobDefaults.randomizeSeed) {
    return undefined;
  }

  return seedNumber(jobDefaults.seed) ?? seedNumber(jobDefaults.generatedSeed);
}

function seedNumber(value) {
  const text = String(value ?? "").trim();
  return text === "" ? undefined : Number(text);
}

function hasOwn(object, key) {
  return Object.hasOwn(object, key);
}

function formatMegabytes(bytes) {
  return `${Math.round(bytes / (1024 * 1024))} MB`;
}

export async function apiFetch(
  path,
  { token, fetchImpl = globalThis.fetch, method = "GET", body, signal } = {},
) {
  const headers = new Headers();
  if (token) {
    headers.set(SESSION_TOKEN_HEADER, token);
  }
  const init = { method, headers, cache: "no-store", signal };
  if (body !== undefined) {
    headers.set("Content-Type", "application/json");
    init.body = JSON.stringify(body);
  }
  const response = await fetchImpl(path, init);
  const contentType = response.headers?.get?.("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : {};
  if (!response.ok) {
    throw new Error(
      payload.message || `Request failed with ${response.status}`,
    );
  }
  return payload;
}

export function createGuiStore({
  fetchImpl,
  initialStatus,
  location,
  history,
  storage,
  setTimer = globalThis.setTimeout,
  clearTimer = globalThis.clearTimeout,
} = {}) {
  let pollTimer = null;
  let pollingGeneration = 0;
  const startingStatus = normalizeStatusSnapshot(
    initialStatus ?? readInitialStatus() ?? DEFAULT_STATUS,
  );

  function clearPollTimer() {
    if (pollTimer !== null) {
      clearTimer(pollTimer);
      pollTimer = null;
    }
  }

  return create((set, get) => ({
    token: "",
    status: startingStatus,
    lastError: "",
    initialized: false,
    activeTaskStartedMs: null,
    batchValidation: null,
    initializeSession: () => {
      if (get().initialized) {
        return;
      }
      set({
        initialized: true,
        token: readSessionToken({ location, history, storage }),
      });
    },
    imageUrl: (path) => appendTokenToUrl(path, get().token),
    fetchStatus: async () => {
      try {
        const status = normalizeStatusSnapshot(
          await apiFetch("/api/status", {
            token: get().token,
            fetchImpl,
          }),
        );
        set({
          status,
          lastError: "",
          activeTaskStartedMs: reconcileActiveTaskStartedMs(
            get().activeTaskStartedMs,
            status,
          ),
        });
        return status;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set({
          lastError: message,
          activeTaskStartedMs: null,
          status: statusAfterConnectionError(get().status, message),
        });
        return get().status;
      }
    },
    startPolling: () => {
      clearPollTimer();
      pollingGeneration += 1;
      const generation = pollingGeneration;
      const tick = async () => {
        await get().fetchStatus();
        if (generation === pollingGeneration) {
          pollTimer = setTimer(tick, pollDelayForStatus(get().status));
        }
      };
      tick();
    },
    stopPolling: () => {
      pollingGeneration += 1;
      clearPollTimer();
    },
    requestAction: async (path, { method = "POST", body } = {}) => {
      const tracksTask = isTaskStartPath(path);

      if (tracksTask) {
        set({ activeTaskStartedMs: Date.now() });
      }

      try {
        const payload = await apiFetch(path, {
          token: get().token,
          fetchImpl,
          method,
          body,
        });
        await get().fetchStatus();
        return payload;
      } catch (error) {
        if (tracksTask) {
          set({ activeTaskStartedMs: null });
        }
        throw error;
      }
    },
    postAction: (path, body) =>
      get().requestAction(path, { method: "POST", body }),
    persistUiSettings: async (settings) => {
      const payload = await apiFetch("/api/ui-settings", {
        token: get().token,
        fetchImpl,
        method: "POST",
        body: buildUiSettingsPatch(settings),
      });
      const uiSettings = payload.settings ?? get().status.ui_settings;
      set({
        status: {
          ...get().status,
          ui_settings: uiSettings,
        },
      });
      return uiSettings;
    },
    loadModel: () => get().postAction("/api/load"),
    ejectModel: () => get().postAction("/api/eject"),
    cancelCurrentGeneration: () => get().postAction("/api/cancel-current"),
    clearBatchQueue: () => get().postAction("/api/clear-queue"),
    openOutputDir: () => get().postAction("/api/open-output-dir"),
    readBatchClipboard: () =>
      apiFetch("/api/read-batch-clipboard", {
        token: get().token,
        fetchImpl,
        method: "POST",
      }),
    readSourceImageClipboard: () =>
      apiFetch("/api/read-source-image-clipboard", {
        token: get().token,
        fetchImpl,
        method: "POST",
      }),
    selectOutputDir: () => get().postAction("/api/select-output-dir"),
    refreshLoras: async () => {
      const payload = await get().postAction("/api/loras/refresh");
      const loras = payload.loras ?? get().status.loras;
      set({
        status: {
          ...get().status,
          loras,
        },
      });
      return loras;
    },
    generateImage: (form) =>
      get().postAction("/api/generate", buildGeneratePayload(form)),
    deleteImage: (imageId) =>
      get().requestAction(`/api/image/${Number(imageId)}`, {
        method: "DELETE",
      }),
    validateSourceImage: (payload) =>
      apiFetch("/api/validate-source-image", {
        token: get().token,
        fetchImpl,
        method: "POST",
        body: payload,
      }),
    validateSourceImageFile: async (file) =>
      get().validateSourceImage(await buildSourceImageFilePayload(file)),
    validateSourceImagePath: (path) =>
      get().validateSourceImage({ path: String(path ?? "") }),
    validateSourceImageId: (imageId) =>
      get().validateSourceImage({ image_id: Number(imageId) }),
    validateBatch: async (text, options) => {
      const payload = buildBatchPayload(text, options);
      const result = await apiFetch("/api/validate-batch", {
        token: get().token,
        fetchImpl,
        method: "POST",
        body: payload,
      });
      set({ batchValidation: result });
      return result;
    },
    generateBatch: (text, options) =>
      get().postAction("/api/generate-batch", buildBatchPayload(text, options)),
    generateRepeatedBatch: (form) =>
      get().postAction("/api/generate-batch", buildRepeatedBatchPayload(form)),
  }));
}

export const useGuiStore = createGuiStore();

function normalizeStatusSnapshot(status) {
  if (!status || typeof status !== "object" || Array.isArray(status)) {
    return DEFAULT_STATUS;
  }

  return {
    ...status,
    constraints: normalizeStatusConstraints(status.constraints),
  };
}
