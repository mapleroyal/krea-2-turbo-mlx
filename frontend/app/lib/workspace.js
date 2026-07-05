import { APP_STORAGE_PREFIX, THEME_MODES } from "@/lib/app-config";
import { normalizeStatusConstraints } from "@/lib/status";

export const MODEL_NAME = APP_STORAGE_PREFIX;
export const LOCAL_LORA_SCALE_MIN = 0;
export const LOCAL_LORA_SCALE_MAX = 4;
export const DEFAULT_LOCAL_LORA_SCALE = 1;
export const LIVE_PREVIEW_MODES = ["off", "latent", "vae"];
export const DEFAULT_LIVE_PREVIEW_MODE = "off";
export const DEFAULT_PREVIEW_INTERVAL_STEPS = 2;
export const MIN_PREVIEW_INTERVAL_STEPS = 1;
export const MAX_PREVIEW_INTERVAL_STEPS = 100;
export const GENERATION_DIMENSION_ALIGNMENT = 16;
export const GENERATION_DIMENSION_MAX_SIZE = 2048;
export const DEFAULT_GENERATION_WIDTH = 1024;
export const DEFAULT_GENERATION_HEIGHT = 1024;
export const DEFAULT_GENERATION_STEPS = 8;
export const SIMPLE_BATCH_DEFAULT_COUNT = 2;
export const SIMPLE_BATCH_MIN_COUNT = 2;
export const SIMPLE_BATCH_MAX_COUNT = 100;
export const LIVE_PREVIEW_MODE_STORAGE_KEY = `${APP_STORAGE_PREFIX}:live-preview-mode`;
export const PREVIEW_INTERVAL_STORAGE_KEY = `${APP_STORAGE_PREFIX}:preview-interval-steps`;

export const DEFAULT_UI_SETTINGS = {
  theme: "system",
  width: formatGenerationDimension(DEFAULT_GENERATION_WIDTH),
  height: formatGenerationDimension(DEFAULT_GENERATION_HEIGHT),
  steps: formatGenerationSteps(DEFAULT_GENERATION_STEPS),
  randomizationLocked: false,
  livePreviewMode: DEFAULT_LIVE_PREVIEW_MODE,
  previewIntervalSteps: formatPreviewIntervalSteps(
    DEFAULT_PREVIEW_INTERVAL_STEPS,
  ),
  loras: [],
  simpleBatchEnabled: false,
  simpleBatchCount: formatSimpleBatchCount(SIMPLE_BATCH_DEFAULT_COUNT),
};

export const DIMENSION_PRESETS = [
  { width: 64, height: 64 },
  { width: 128, height: 128 },
  { width: 512, height: 512 },
  { width: 768, height: 512 },
  { width: 512, height: 768 },
  { width: 1024, height: 1024 },
  { width: 1216, height: 832 },
  { width: 832, height: 1216 },
  { width: 1536, height: 1024 },
  { width: 1024, height: 1536 },
  { width: 2048, height: 2048 },
].map((preset) => ({
  ...preset,
  value: `${preset.width}x${preset.height}`,
  label: `${preset.width} x ${preset.height}`,
}));

export const CUSTOM_DIMENSION_PRESET = "custom";

export const EXAMPLE_BATCH_PROMPT = `Create the JSON for batch image generation. Output only valid JSON without elaborating. No markdown, comments, explanations, etc.

Schema:

export type BatchJSON = BatchJob[]; // 1 to 100 jobs

export type BatchJob = {
  prompt: string; // non-empty after trimming
  width?: number; // integer, positive, multiple of 16, max 2048
  height?: number; // integer, positive, multiple of 16, max 2048
  steps?: number; // positive integer, usually 8 to 12
  seed?: number; // integer from 0 to 4294967295
  loras?: Array<{
    id: string; // catalog id from the LoRA list
    scale?: number; // 0.0 to 4.0
  }>;
};

Rules:

- Every job must include prompt.
- Omit other fields unless the user specifies how to populate them.
- Only add LoRAs when the user asks for a specific one.

Example:

[
  { "prompt": "a glass library at sunrise", "width": 768, "height": 768, "steps": 8, "seed": 1933305333 },
  { "prompt": "a cedar observatory at night" }
]

My requirements are: [tell the AI what to generate].`;

export function presetForSize(width, height) {
  const normalizedWidth = Number(width);
  const normalizedHeight = Number(height);
  const preset = DIMENSION_PRESETS.find(
    (item) =>
      item.width === normalizedWidth && item.height === normalizedHeight,
  );

  return preset?.value ?? CUSTOM_DIMENSION_PRESET;
}

export function modelVariantLabel(record, fallbackPrecision = "") {
  const explicit = precisionDisplayLabel(
    record?.model_precision ?? record?.precision ?? record?.variant ?? "",
  );
  if (explicit) {
    return explicit;
  }

  const fallback = precisionDisplayLabel(fallbackPrecision);
  if (fallback) {
    return fallback;
  }

  const modelPath = String(record?.model_path ?? "").trim();
  const pathParts = modelPath.split(/[\\/]/).filter(Boolean);
  const basename = pathParts.at(-1) ?? "";
  const withoutExtension = basename.replace(/\.[^.]+$/, "");

  return (
    precisionDisplayLabel(withoutExtension, { allowFreeform: false }) || "model"
  );
}

export function formatMeta({
  derivedFields,
  model = MODEL_NAME,
  variant,
  width,
  height,
  loras,
  seed,
  steps,
}) {
  const modelText = String(model ?? "").trim() || MODEL_NAME;
  const variantText = String(variant ?? "").trim() || "model";
  const derivedFieldSet = new Set(derivedFields ?? []);
  const widthText = derivedMetaValue(width, derivedFieldSet, "width");
  const heightText = derivedMetaValue(height, derivedFieldSet, "height");
  const lines = [
    modelText,
    `${variantText} @ ${widthText}x${heightText}`,
    `Seed: ${derivedMetaValue(seed, derivedFieldSet, "seed")}`,
    `Steps: ${derivedMetaValue(steps, derivedFieldSet, "steps")}`,
    ...formatLoraMetaLines(loras, {
      derived: derivedFieldSet.has("loras"),
    }),
  ];

  return lines.join("\n");
}

export function formatLoraMetaLines(loras, { derived = false } = {}) {
  if (!Array.isArray(loras) || loras.length === 0) {
    return [];
  }

  return loras
    .map((lora) => {
      const name = loraDisplayName(lora);

      if (!name) {
        return "";
      }

      const scale = formatLoraScale(lora?.scale ?? DEFAULT_LOCAL_LORA_SCALE);
      return `${name}: ${scale}${derived ? "*" : ""}`;
    })
    .filter(Boolean);
}

function derivedMetaValue(value, derivedFields, field) {
  return `${value}${derivedFields.has(field) ? "*" : ""}`;
}

export function loraDisplayName(lora) {
  const explicit = String(
    lora?.display_name ?? lora?.displayName ?? lora?.name ?? "",
  ).trim();

  if (explicit) {
    return explicit;
  }

  const id = String(lora?.id ?? "").trim();

  return id;
}

export function randomSeed(max = 4294967295) {
  const normalizedMax = Math.max(
    0,
    Math.min(4294967295, Math.floor(Number(max) || 0)),
  );
  const cryptoApi = globalThis.crypto;

  if (typeof cryptoApi?.getRandomValues === "function") {
    const values = new Uint32Array(1);
    cryptoApi.getRandomValues(values);

    if (normalizedMax === 4294967295) {
      return values[0];
    }

    return values[0] % (normalizedMax + 1);
  }

  return Math.floor(Math.random() * (normalizedMax + 1));
}

export function localLoraScaleLimits(constraints = {}) {
  const normalizedConstraints = normalizeStatusConstraints(constraints);
  return {
    min: normalizedConstraints.local_lora_scale_min,
    max: normalizedConstraints.local_lora_scale_max,
    defaultValue: normalizedConstraints.default_local_lora_scale,
  };
}

export function loraCatalogItems(catalog) {
  const items = Array.isArray(catalog?.items) ? catalog.items : [];
  return items;
}

export function findLoraCatalogItem(catalogItems, id) {
  const loraId = String(id ?? "").trim();
  return (
    (Array.isArray(catalogItems) ? catalogItems : []).find(
      (item) => String(item?.id ?? "") === loraId,
    ) ?? null
  );
}

export function loraScaleLimitsForItem(item, constraints = {}) {
  if (item) {
    return {
      min: finiteNumber(item.scale_min, LOCAL_LORA_SCALE_MIN),
      max: finiteNumber(item.scale_max, LOCAL_LORA_SCALE_MAX),
      defaultValue: finiteNumber(item.default_scale, DEFAULT_LOCAL_LORA_SCALE),
    };
  }
  return localLoraScaleLimits(constraints);
}

export function clampLoraScale(value, limits = localLoraScaleLimits()) {
  const fallback = finiteNumber(limits.defaultValue, DEFAULT_LOCAL_LORA_SCALE);
  const min = finiteNumber(limits.min, LOCAL_LORA_SCALE_MIN);
  const max = finiteNumber(limits.max, LOCAL_LORA_SCALE_MAX);
  const parsed = Number(value);
  const raw = Number.isFinite(parsed) ? parsed : fallback;
  const clamped = Math.max(min, Math.min(max, raw));

  return {
    value: clamped,
    changed: clamped !== raw,
    usedFallback: !Number.isFinite(parsed),
  };
}

export function normalizeLoraSelections(
  value,
  { catalogItems = null, constraints } = {},
) {
  if (!Array.isArray(value)) {
    return [];
  }
  const hasCatalog = Array.isArray(catalogItems);
  const seen = new Set();
  const normalized = [];
  for (const item of value) {
    const id = String(item?.id ?? "").trim();
    if (!id || seen.has(id)) {
      continue;
    }
    const catalogItem = hasCatalog
      ? findLoraCatalogItem(catalogItems, id)
      : null;
    if (hasCatalog && !catalogItem) {
      continue;
    }
    const limits = catalogItem
      ? loraScaleLimitsForItem(catalogItem, constraints)
      : localLoraScaleLimits(constraints);
    const scale = clampLoraScale(
      item?.scale ?? limits.defaultValue,
      limits,
    ).value;
    normalized.push({ id, scale });
    seen.add(id);
  }
  return normalized;
}

export function buildSelectedLoras(loras, options) {
  const normalized = normalizeLoraSelections(loras, options);
  return normalized.length ? normalized : undefined;
}

export function setLoraSelectionEnabled(loras, item, enabled) {
  const id = String(item?.id ?? "").trim();
  if (!id) {
    return Array.isArray(loras) ? loras : [];
  }
  const current = Array.isArray(loras) ? loras : [];
  if (!enabled) {
    return current.filter((lora) => lora.id !== id);
  }
  if (current.some((lora) => lora.id === id)) {
    return current;
  }
  return [
    ...current,
    {
      id,
      scale: finiteNumber(item.default_scale, DEFAULT_LOCAL_LORA_SCALE),
    },
  ];
}

export function setLoraSelectionScale(loras, id, scale) {
  const loraId = String(id ?? "").trim();
  return (Array.isArray(loras) ? loras : []).map((lora) =>
    lora.id === loraId ? { ...lora, scale } : lora,
  );
}

export function formatLoraScale(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return String(DEFAULT_LOCAL_LORA_SCALE.toFixed(1));
  }
  return Number.isInteger(number) ? number.toFixed(1) : String(number);
}

export function normalizeLivePreviewMode(value) {
  const mode = String(value ?? "")
    .trim()
    .toLowerCase();
  return LIVE_PREVIEW_MODES.includes(mode) ? mode : DEFAULT_LIVE_PREVIEW_MODE;
}

export function clampPreviewIntervalSteps(value) {
  const parsed = Number(value);
  const usedFallback = !Number.isFinite(parsed);
  const raw = usedFallback
    ? DEFAULT_PREVIEW_INTERVAL_STEPS
    : Math.trunc(parsed);
  const clamped = Math.max(
    MIN_PREVIEW_INTERVAL_STEPS,
    Math.min(MAX_PREVIEW_INTERVAL_STEPS, raw),
  );

  return {
    value: clamped,
    changed: clamped !== parsed,
    usedFallback,
  };
}

export function formatPreviewIntervalSteps(value) {
  return String(clampPreviewIntervalSteps(value).value);
}

export function generationDimensionLimits(constraints = {}) {
  const normalizedConstraints = normalizeStatusConstraints(constraints);
  return {
    alignment: Math.max(1, Math.trunc(normalizedConstraints.alignment)),
    maxSize: Math.max(1, Math.trunc(normalizedConstraints.max_size)),
    defaultWidth: normalizedConstraints.default_width,
    defaultHeight: normalizedConstraints.default_height,
  };
}

export function clampGenerationDimension(
  value,
  {
    alignment = GENERATION_DIMENSION_ALIGNMENT,
    maxSize = GENERATION_DIMENSION_MAX_SIZE,
    defaultValue = DEFAULT_GENERATION_WIDTH,
  } = {},
) {
  const parsed = Number(value);
  const usedFallback = !Number.isFinite(parsed);
  const raw = usedFallback ? defaultValue : Math.trunc(parsed);
  const normalizedAlignment = Math.max(1, Math.trunc(Number(alignment) || 1));
  const normalizedMax = Math.max(
    normalizedAlignment,
    Math.trunc(Number(maxSize) || GENERATION_DIMENSION_MAX_SIZE),
  );
  const aligned = Math.round(raw / normalizedAlignment) * normalizedAlignment;
  const clamped = Math.max(
    normalizedAlignment,
    Math.min(normalizedMax, aligned),
  );

  return {
    value: clamped,
    changed: clamped !== parsed,
    usedFallback,
  };
}

export function formatGenerationDimension(value, options) {
  return String(clampGenerationDimension(value, options).value);
}

export function clampGenerationSteps(
  value,
  { defaultValue = DEFAULT_GENERATION_STEPS } = {},
) {
  const parsed = Number(value);
  const usedFallback = !Number.isFinite(parsed);
  const raw = usedFallback ? defaultValue : Math.trunc(parsed);
  const clamped = Math.max(1, raw);

  return {
    value: clamped,
    changed: clamped !== parsed,
    usedFallback,
  };
}

export function formatGenerationSteps(value, options) {
  return String(clampGenerationSteps(value, options).value);
}

export function clampSimpleBatchCount(value, max) {
  const parsed = Number(value);
  const usedFallback = !Number.isFinite(parsed);
  const raw = usedFallback ? SIMPLE_BATCH_DEFAULT_COUNT : Math.trunc(parsed);
  const maxValue = Math.max(
    SIMPLE_BATCH_MIN_COUNT,
    Math.trunc(finiteNumber(max, SIMPLE_BATCH_MAX_COUNT)),
  );
  const clamped = Math.max(SIMPLE_BATCH_MIN_COUNT, Math.min(maxValue, raw));

  return {
    value: clamped,
    changed: clamped !== parsed,
    usedFallback,
    min: SIMPLE_BATCH_MIN_COUNT,
    max: maxValue,
  };
}

export function formatSimpleBatchCount(value, max) {
  return String(clampSimpleBatchCount(value, max).value);
}

export function readLivePreviewSettings({
  storage = globalThis?.window?.localStorage,
} = {}) {
  return {
    mode: normalizeLivePreviewMode(
      storage?.getItem?.(LIVE_PREVIEW_MODE_STORAGE_KEY),
    ),
    intervalSteps: formatPreviewIntervalSteps(
      storage?.getItem?.(PREVIEW_INTERVAL_STORAGE_KEY) ??
        DEFAULT_PREVIEW_INTERVAL_STEPS,
    ),
  };
}

export function persistLivePreviewMode(
  mode,
  storage = globalThis?.window?.localStorage,
) {
  storage?.setItem?.(
    LIVE_PREVIEW_MODE_STORAGE_KEY,
    normalizeLivePreviewMode(mode),
  );
}

export function persistPreviewIntervalSteps(
  intervalSteps,
  storage = globalThis?.window?.localStorage,
) {
  storage?.setItem?.(
    PREVIEW_INTERVAL_STORAGE_KEY,
    formatPreviewIntervalSteps(intervalSteps),
  );
}

export function normalizeUiSettings(
  settings,
  { catalogItems, constraints } = {},
) {
  const dimensionLimits = generationDimensionLimits(constraints);
  const generation = settings?.generation;
  const livePreview = settings?.live_preview;
  const simpleBatch = settings?.simple_batch;
  const theme = String(settings?.theme ?? "")
    .trim()
    .toLowerCase();
  const loras = Array.isArray(settings?.loras) ? settings.loras : [];
  const normalizedLoras = normalizeLoraSelections(loras, {
    catalogItems,
    constraints,
  });

  return {
    theme: THEME_MODES.includes(theme) ? theme : DEFAULT_UI_SETTINGS.theme,
    width: formatGenerationDimension(generation?.width, {
      alignment: dimensionLimits.alignment,
      maxSize: dimensionLimits.maxSize,
      defaultValue: dimensionLimits.defaultWidth,
    }),
    height: formatGenerationDimension(generation?.height, {
      alignment: dimensionLimits.alignment,
      maxSize: dimensionLimits.maxSize,
      defaultValue: dimensionLimits.defaultHeight,
    }),
    steps: formatGenerationSteps(generation?.steps, {
      defaultValue: finiteNumber(
        constraints?.default_steps,
        DEFAULT_GENERATION_STEPS,
      ),
    }),
    randomizationLocked:
      typeof generation?.randomization_locked === "boolean"
        ? generation.randomization_locked
        : DEFAULT_UI_SETTINGS.randomizationLocked,
    livePreviewMode: normalizeLivePreviewMode(livePreview?.mode),
    previewIntervalSteps: formatPreviewIntervalSteps(
      livePreview?.interval_steps ?? DEFAULT_PREVIEW_INTERVAL_STEPS,
    ),
    loras: normalizedLoras,
    simpleBatchEnabled:
      typeof simpleBatch?.enabled === "boolean"
        ? simpleBatch.enabled
        : DEFAULT_UI_SETTINGS.simpleBatchEnabled,
    simpleBatchCount: formatSimpleBatchCount(
      simpleBatch?.count ?? SIMPLE_BATCH_DEFAULT_COUNT,
      constraints?.max_batch_jobs,
    ),
  };
}

export function buildUiSettingsPatch(settings = {}) {
  const payload = {};

  if (Object.hasOwn(settings, "theme")) {
    const theme = String(settings.theme ?? "")
      .trim()
      .toLowerCase();
    if (THEME_MODES.includes(theme)) {
      payload.theme = theme;
    }
  }

  if (
    Object.hasOwn(settings, "width") ||
    Object.hasOwn(settings, "height") ||
    Object.hasOwn(settings, "steps") ||
    Object.hasOwn(settings, "randomizationLocked")
  ) {
    payload.generation = {};
    if (Object.hasOwn(settings, "width")) {
      payload.generation.width = clampGenerationDimension(settings.width).value;
    }
    if (Object.hasOwn(settings, "height")) {
      payload.generation.height = clampGenerationDimension(settings.height, {
        defaultValue: DEFAULT_GENERATION_HEIGHT,
      }).value;
    }
    if (Object.hasOwn(settings, "steps")) {
      payload.generation.steps = clampGenerationSteps(settings.steps).value;
    }
    if (Object.hasOwn(settings, "randomizationLocked")) {
      payload.generation.randomization_locked =
        settings.randomizationLocked === true;
    }
  }

  if (
    Object.hasOwn(settings, "livePreviewMode") ||
    Object.hasOwn(settings, "previewIntervalSteps")
  ) {
    payload.live_preview = {};
    if (Object.hasOwn(settings, "livePreviewMode")) {
      payload.live_preview.mode = normalizeLivePreviewMode(
        settings.livePreviewMode,
      );
    }
    if (Object.hasOwn(settings, "previewIntervalSteps")) {
      payload.live_preview.interval_steps = clampPreviewIntervalSteps(
        settings.previewIntervalSteps,
      ).value;
    }
  }

  if (Object.hasOwn(settings, "loras")) {
    payload.loras = normalizeLoraSelections(settings.loras, {
      catalogItems: settings.catalogItems,
      constraints: settings.constraints,
    });
  }

  if (
    Object.hasOwn(settings, "simpleBatchEnabled") ||
    Object.hasOwn(settings, "simpleBatchCount")
  ) {
    payload.simple_batch = {};
    if (Object.hasOwn(settings, "simpleBatchEnabled")) {
      payload.simple_batch.enabled = settings.simpleBatchEnabled === true;
    }
    if (Object.hasOwn(settings, "simpleBatchCount")) {
      payload.simple_batch.count = clampSimpleBatchCount(
        settings.simpleBatchCount,
        settings.simpleBatchMaxCount,
      ).value;
    }
  }

  return payload;
}

function precisionDisplayLabel(value, { allowFreeform = true } = {}) {
  const text = String(value ?? "").trim();
  if (!text) {
    return "";
  }

  const lower = text.toLowerCase();
  if (["full", "full precision", "full-precision"].includes(lower)) {
    return "";
  }

  const tokens = lower.split(/[^a-z0-9]+/).filter(Boolean);
  for (const token of tokens) {
    const normalized = normalizePrecisionToken(token);
    if (normalized) {
      return normalized;
    }
  }

  return allowFreeform ? text : "";
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizePrecisionToken(token) {
  switch (token) {
    case "bf16":
    case "bfloat16":
      return "bf16";
    case "fp16":
    case "f16":
    case "float16":
      return "fp16";
    case "fp32":
    case "f32":
    case "float32":
      return "fp32";
    case "fp64":
    case "f64":
    case "float64":
      return "fp64";
    case "q8":
    case "int8":
    case "i8":
      return token === "q8" ? "q8" : "int8";
    default:
      return "";
  }
}
