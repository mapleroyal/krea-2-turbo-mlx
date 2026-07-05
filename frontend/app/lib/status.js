export const TASK_PHASES = {
  STARTING: "starting",
  LOAD: "load",
  MODEL_LOAD: "model_load",
  GENERATE: "generate",
  COMPLETE: "complete",
  CANCELLED: "cancelled",
  DISCONNECTED: "disconnected",
};

export const BATCH_JOB_STATUSES = {
  QUEUED: "queued",
  RUNNING: "running",
  CANCELLING: "cancelling",
  CANCELLED: "cancelled",
  CLEARED: "cleared",
  DONE: "done",
};

export const BATCH_JOB_STATUS_VALUES = Object.values(BATCH_JOB_STATUSES);
export const RUNNING_BATCH_JOB_STATUSES = [
  BATCH_JOB_STATUSES.RUNNING,
  BATCH_JOB_STATUSES.CANCELLING,
];
export const SETTLED_BATCH_JOB_STATUSES = [
  BATCH_JOB_STATUSES.DONE,
  BATCH_JOB_STATUSES.CANCELLED,
  BATCH_JOB_STATUSES.CLEARED,
];
export const DESTRUCTIVE_BATCH_JOB_STATUSES = [
  BATCH_JOB_STATUSES.CANCELLING,
  BATCH_JOB_STATUSES.CANCELLED,
  BATCH_JOB_STATUSES.CLEARED,
];

export const STATUS_CONSTRAINT_FALLBACKS = {
  alignment: 16,
  max_size: 2048,
  max_seed: 4294967295,
  default_width: 1024,
  default_height: 1024,
  default_steps: 8,
  guidance_scale: 0,
  shift: 1.15,
  max_batch_jobs: 100,
  local_lora_scale_min: 0,
  local_lora_scale_max: 4,
  default_local_lora_scale: 1,
};

export function normalizeTaskPhase(value, fallback = TASK_PHASES.STARTING) {
  const phase = String(value ?? "")
    .trim()
    .toLowerCase();
  return Object.values(TASK_PHASES).includes(phase) ? phase : fallback;
}

export function normalizeBatchJobStatus(value, fallback = "") {
  const status = String(value ?? "")
    .trim()
    .toLowerCase();
  return BATCH_JOB_STATUS_VALUES.includes(status) ? status : fallback;
}

export function normalizeStatusConstraints(constraints) {
  const source =
    constraints &&
    typeof constraints === "object" &&
    !Array.isArray(constraints)
      ? constraints
      : {};

  return Object.fromEntries(
    Object.entries(STATUS_CONSTRAINT_FALLBACKS).map(([key, fallback]) => [
      key,
      finiteNumber(source[key], fallback),
    ]),
  );
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
