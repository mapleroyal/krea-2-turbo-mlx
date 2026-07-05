import { afterEach, describe, expect, it, vi } from "vitest";

import {
  EXAMPLE_BATCH_PROMPT,
  LIVE_PREVIEW_MODE_STORAGE_KEY,
  PREVIEW_INTERVAL_STORAGE_KEY,
  buildSelectedLoras,
  clampGenerationDimension,
  clampGenerationSteps,
  clampPreviewIntervalSteps,
  clampSimpleBatchCount,
  formatMeta,
  formatSimpleBatchCount,
  MODEL_NAME,
  buildUiSettingsPatch,
  modelVariantLabel,
  normalizeLoraSelections,
  normalizeUiSettings,
  persistLivePreviewMode,
  persistPreviewIntervalSteps,
  presetForSize,
  randomSeed,
  readLivePreviewSettings,
} from "./workspace";

describe("workspace helpers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("matches known dimension presets and marks custom sizes", () => {
    expect(presetForSize(1024, 1024)).toBe("1024x1024");
    expect(presetForSize("768", "512")).toBe("768x512");
    expect(presetForSize(1000, 1024)).toBe("custom");
  });

  it("formats image metadata as the shared four-line caption", () => {
    expect(
      formatMeta({
        variant: "bf16",
        width: 1024,
        height: 1024,
        seed: 403005717,
        steps: 5,
      }),
    ).toBe(`${MODEL_NAME}\nbf16 @ 1024x1024\nSeed: 403005717\nSteps: 5`);
  });

  it("adds LoRA display names and scales to image metadata", () => {
    expect(
      formatMeta({
        variant: "bf16",
        width: 1024,
        height: 1024,
        seed: 403005717,
        steps: 5,
        loras: [{ id: "style.safetensors", display_name: "Style", scale: 1.3 }],
      }),
    ).toBe(
      `${MODEL_NAME}\nbf16 @ 1024x1024\nSeed: 403005717\nSteps: 5\nStyle: 1.3`,
    );
  });

  it("derives model variant labels from precision and then paths", () => {
    expect(modelVariantLabel({ model_precision: "BF16" }, "full")).toBe("bf16");
    expect(
      modelVariantLabel(
        { model_path: "/Users/user1/models/krea-2-turbo-bf16.safetensors" },
        "full",
      ),
    ).toBe("bf16");
    expect(
      modelVariantLabel({
        model_path: "/Users/user1/artifacts/krea-2-turbo-mlx",
      }),
    ).toBe("model");
    expect(modelVariantLabel({ model_path: "" }, "bf16")).toBe("bf16");
  });

  it("documents batch JSON as a bare-array contract with a valid JSON example", () => {
    // This prompt is an LLM-facing contract: the schema must describe a bare
    // array and the embedded example must be pasteable JSON (no fences, no
    // trailing-comma array close).
    expect(EXAMPLE_BATCH_PROMPT).toContain(
      "export type BatchJSON = BatchJob[]",
    );
    expect(EXAMPLE_BATCH_PROMPT).not.toContain("```");
    expect(EXAMPLE_BATCH_PROMPT).not.toContain("},\n]");
  });

  it("generates seeds inside the requested range", () => {
    vi.stubGlobal("crypto", {
      getRandomValues: vi.fn((array) => {
        array[0] = 42;
        return array;
      }),
    });

    expect(randomSeed(100)).toBe(42);
  });

  it("normalizes, clamps, and persists live preview settings", () => {
    let values = {
      [LIVE_PREVIEW_MODE_STORAGE_KEY]: "VAE",
      [PREVIEW_INTERVAL_STORAGE_KEY]: "999",
    };
    const storage = {
      getItem: vi.fn((key) => values[key] ?? null),
      setItem: vi.fn((key, value) => {
        values = { ...values, [key]: value };
      }),
    };

    expect(readLivePreviewSettings({ storage })).toEqual({
      mode: "vae",
      intervalSteps: "100",
    });
    expect(clampPreviewIntervalSteps(0).value).toBe(1);

    persistLivePreviewMode("latent", storage);
    persistPreviewIntervalSteps("7", storage);

    expect(storage.setItem).toHaveBeenCalledWith(
      LIVE_PREVIEW_MODE_STORAGE_KEY,
      "latent",
    );
    expect(storage.setItem).toHaveBeenCalledWith(
      PREVIEW_INTERVAL_STORAGE_KEY,
      "7",
    );
  });

  it("normalizes project-local GUI settings and builds partial patches", () => {
    expect(
      normalizeUiSettings({
        theme: "Dark",
        generation: {
          width: 1000,
          height: 3000,
          steps: 0,
          randomization_locked: true,
        },
        live_preview: { mode: "Latent", interval_steps: 999 },
        loras: [
          { id: "style.safetensors", scale: 2.5 },
          { id: "portrait.safetensors", scale: 999 },
        ],
        simple_batch: { enabled: true, count: 999 },
      }),
    ).toEqual({
      theme: "dark",
      width: "1008",
      height: "2048",
      steps: "1",
      randomizationLocked: true,
      livePreviewMode: "latent",
      previewIntervalSteps: "100",
      loras: [
        { id: "style.safetensors", scale: 2.5 },
        { id: "portrait.safetensors", scale: 4 },
      ],
      simpleBatchEnabled: true,
      simpleBatchCount: "100",
    });
    expect(clampGenerationDimension(1000).value).toBe(1008);
    expect(clampGenerationSteps(0).value).toBe(1);

    expect(
      buildUiSettingsPatch({
        theme: "light",
        width: "512",
        height: "768",
        steps: "10",
        randomizationLocked: true,
        livePreviewMode: "vae",
        previewIntervalSteps: "7",
        simpleBatchEnabled: true,
        simpleBatchCount: "4",
      }),
    ).toEqual({
      theme: "light",
      generation: {
        width: 512,
        height: 768,
        steps: 10,
        randomization_locked: true,
      },
      live_preview: {
        mode: "vae",
        interval_steps: 7,
      },
      simple_batch: {
        enabled: true,
        count: 4,
      },
    });

    expect(
      buildUiSettingsPatch({
        loras: [
          { id: "style.safetensors", scale: "5" },
          { id: "portrait.safetensors", scale: "2" },
        ],
      }),
    ).toEqual({
      loras: [
        { id: "style.safetensors", scale: 4 },
        { id: "portrait.safetensors", scale: 2 },
      ],
    });
  });

  it("normalizes selected catalog LoRAs by source-specific limits", () => {
    const catalogItems = [
      {
        id: "style.safetensors",
        default_scale: 1,
        scale_min: 0,
        scale_max: 4,
      },
    ];

    expect(
      normalizeLoraSelections(
        [
          { id: "style.safetensors", scale: 5 },
          { id: "unknown.safetensors", scale: -1 },
        ],
        { catalogItems },
      ),
    ).toEqual([{ id: "style.safetensors", scale: 4 }]);
    expect(buildSelectedLoras([], { catalogItems })).toBeUndefined();
  });

  it("normalizes simple batch counts", () => {
    expect(clampSimpleBatchCount("4", 100)).toEqual({
      value: 4,
      changed: false,
      usedFallback: false,
      min: 2,
      max: 100,
    });
    expect(clampSimpleBatchCount("nope", 100)).toMatchObject({
      value: 2,
      usedFallback: true,
    });
    expect(clampSimpleBatchCount(999, 5)).toMatchObject({
      value: 5,
      changed: true,
      max: 5,
    });
    expect(formatSimpleBatchCount(1, 100)).toBe("2");
  });
});
