import { afterEach, describe, expect, it, vi } from "vitest";

import {
  batchClearQueueFollowupKeyAction,
  batchHasQueuedJobs,
  batchResubmitIntentFromJobs,
  batchPromptStateFromCompletedBatch,
  batchPromptTextFromJobs,
  completedBatchFromJobs,
  completedBatchFromSnapshot,
  formatGallerySpotlightZoom,
  galleryImageByOffset,
  galleryImageClickAction,
  galleryPreviewKeyAction,
  gallerySpotlightInitialImage,
  gallerySpotlightKeyAction,
  generationCancelKeyAction,
  isPromptSubmitShortcut,
  modelStatusPresentation,
  nextGallerySpotlightZoom,
  nextGallerySpotlightUnseenIds,
  nextGalleryImageIdAfterDelete,
  promptSubmitAction,
  shortcutModifierLabel,
  shortcutSheetKeyDownAction,
  shortcutSheetKeyUpAction,
  serverStatusPresentation,
  wrappedGalleryIndex,
} from "./krea-workspace";

describe("KreaWorkspace", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses Command+Enter for prompt submit on macOS", () => {
    expect(
      isPromptSubmitShortcut(
        {
          altKey: false,
          ctrlKey: false,
          isComposing: false,
          key: "Enter",
          metaKey: true,
          shiftKey: false,
        },
        "MacIntel",
      ),
    ).toBe(true);

    expect(
      isPromptSubmitShortcut(
        {
          altKey: false,
          ctrlKey: true,
          isComposing: false,
          key: "Enter",
          metaKey: false,
          shiftKey: false,
        },
        "MacIntel",
      ),
    ).toBe(false);
  });

  it("uses Control+Enter for prompt submit outside macOS", () => {
    expect(
      isPromptSubmitShortcut(
        {
          altKey: false,
          ctrlKey: true,
          isComposing: false,
          key: "Enter",
          metaKey: false,
          shiftKey: false,
        },
        "Win32",
      ),
    ).toBe(true);

    expect(
      isPromptSubmitShortcut(
        {
          altKey: false,
          ctrlKey: false,
          isComposing: false,
          key: "Enter",
          metaKey: true,
          shiftKey: false,
        },
        "Linux x86_64",
      ),
    ).toBe(false);
  });

  it("labels the shortcut modifier by platform", () => {
    expect(shortcutModifierLabel("MacIntel")).toBe("Cmd");
    expect(shortcutModifierLabel("iPad")).toBe("Cmd");
    expect(shortcutModifierLabel("Win32")).toBe("Ctrl");
    expect(shortcutModifierLabel("Linux x86_64")).toBe("Ctrl");
  });

  it("only arms the shortcut sheet from a standalone physical modifier hold", () => {
    expect(
      shortcutSheetKeyDownAction(keyEvent("Meta"), {
        platform: "MacIntel",
      }),
    ).toBe("pressModifier");
    expect(
      shortcutSheetKeyDownAction(keyEvent("Control"), {
        platform: "Win32",
      }),
    ).toBe("pressModifier");
    expect(
      shortcutSheetKeyDownAction(
        keyEvent("4", {
          metaKey: true,
        }),
        { platform: "MacIntel" },
      ),
    ).toBe("suppressChord");
    expect(
      shortcutSheetKeyDownAction(
        keyEvent("Shift", {
          metaKey: true,
        }),
        { modifierHeld: true, platform: "MacIntel" },
      ),
    ).toBe("suppressChord");
    expect(
      shortcutSheetKeyDownAction(
        keyEvent("Meta", {
          repeat: true,
        }),
        { platform: "MacIntel" },
      ),
    ).toBe(null);
  });

  it("releases the shortcut sheet when the modifier is gone or stale", () => {
    expect(
      shortcutSheetKeyUpAction(keyEvent("Meta"), {
        modifierHeld: true,
        platform: "MacIntel",
      }),
    ).toBe("releaseModifier");
    expect(
      shortcutSheetKeyUpAction(
        keyEvent("4", {
          metaKey: false,
        }),
        { modifierHeld: true, platform: "MacIntel" },
      ),
    ).toBe("releaseModifier");
    expect(
      shortcutSheetKeyUpAction(
        keyEvent("4", {
          metaKey: true,
        }),
        { modifierHeld: true, platform: "MacIntel" },
      ),
    ).toBe(null);
  });

  it("routes batch JSON prompts through batch submission", () => {
    expect(promptSubmitAction('[{"prompt":"one"}]')).toBe("batch");
    expect(promptSubmitAction("[]")).toBe("batch");
    expect(promptSubmitAction('[{"prompt":')).toBe("batch");
    expect(promptSubmitAction("[surreal] glass observatory")).toBe("image");
    expect(promptSubmitAction("a glass observatory")).toBe("image");
  });

  it("labels server connection states with matching status tones", () => {
    expect(
      serverStatusPresentation({ connected: true, status: "ready" }),
    ).toEqual({ label: "Server connected", tone: "ready" });
    expect(
      serverStatusPresentation({ connected: false, status: "starting" }),
    ).toEqual({ label: "Server connecting", tone: "loading" });
    expect(
      serverStatusPresentation({ connected: false, status: "error" }),
    ).toEqual({ label: "Server disconnected", tone: "error" });
  });

  it("labels model load states without treating generation errors as model failures", () => {
    expect(
      modelStatusPresentation({
        model: { loaded: true, status: "in memory" },
        phase: "generate",
        error: "image failed",
      }),
    ).toEqual({ label: "Model loaded", tone: "ready" });
    expect(
      modelStatusPresentation({
        model: { loaded: false, status: "not loaded" },
        loadRunning: true,
        phase: "model_load",
      }),
    ).toEqual({ label: "Loading model", tone: "loading" });
    expect(
      modelStatusPresentation({
        model: { loaded: false, status: "not loaded" },
        phase: "model_load",
        error: "artifact missing",
      }),
    ).toEqual({ label: "Model load failed", tone: "error" });
    expect(
      modelStatusPresentation({
        model: { loaded: false, status: "not loaded" },
        phase: "generate",
        error: "LoRA failed",
      }),
    ).toEqual({ label: "Load model", tone: "idle" });
  });

  it("formats resolved batch jobs as resubmittable prompt text", () => {
    const text = batchPromptTextFromJobs([
      {
        index: 1,
        prompt: "one",
        width: 512,
        height: 768,
        steps: 8,
        seed: 123,
        guiDerivedFields: ["seed"],
      },
    ]);

    expect(JSON.parse(text)).toEqual([
      { prompt: "one", width: 512, height: 768, steps: 8, seed: 123 },
    ]);
    expect(text).toContain('\n  {\n    "prompt": "one"');
    expect(text).not.toContain("guiDerivedFields");
    expect(text).not.toContain("index");
  });

  it("keeps GUI-derived batch fields as rerun rules", () => {
    const intent = batchResubmitIntentFromJobs({
      outputDir: "/tmp/krea-batch",
      sourceJobs: [
        { prompt: "one", seed: 7 },
        { prompt: "two", width: 768, loras: [] },
      ],
      submittedJobs: [
        {
          prompt: "one",
          width: 512,
          height: 512,
          steps: 8,
          seed: 7,
          loras: [{ id: "style.safetensors", scale: 1.3 }],
        },
        {
          prompt: "two",
          width: 768,
          height: 512,
          steps: 8,
          seed: 99,
          loras: [],
        },
      ],
    });

    expect(intent).toEqual({
      outputDir: "/tmp/krea-batch",
      derivedFieldsByIndex: [
        ["width", "height", "steps", "loras"],
        ["height", "steps", "seed"],
      ],
      resubmitJobs: [
        { prompt: "one", seed: 7 },
        { prompt: "two", width: 768, loras: [] },
      ],
    });

    expect(
      batchResubmitIntentFromJobs({
        sourceJobs: [{ prompt: "random seed" }],
        submittedJobs: [
          { prompt: "random seed", width: 512, height: 512, steps: 8 },
        ],
      }).derivedFieldsByIndex,
    ).toEqual([["width", "height", "steps", "seed"]]);
  });

  it("marks simple batch values without rewriting the plain prompt", () => {
    const intent = {
      ...batchResubmitIntentFromJobs({
        sourceJobs: [{ prompt: "one" }, { prompt: "one" }],
        submittedJobs: [
          {
            prompt: "one",
            width: 512,
            height: 512,
            steps: 8,
            seed: 11,
            loras: [{ id: "style.safetensors", scale: 1.3 }],
          },
          {
            prompt: "one",
            width: 512,
            height: 512,
            steps: 8,
            seed: 22,
            loras: [{ id: "style.safetensors", scale: 1.3 }],
          },
        ],
      }),
      updatePromptOnCompletion: false,
    };
    const state = batchPromptStateFromCompletedBatch(
      completedBatchFromJobs([
        { prompt: "one", width: 512, height: 512, steps: 8, seed: 101 },
        { prompt: "one", width: 512, height: 512, steps: 8, seed: 202 },
      ]),
      intent,
    );

    expect(state.options).toBeNull();
    expect(state.completedBatch.jobs).toMatchObject([
      {
        guiDerivedFields: ["width", "height", "steps", "seed", "loras"],
        seed: 101,
      },
      {
        guiDerivedFields: ["width", "height", "steps", "seed", "loras"],
        seed: 202,
      },
    ]);
  });

  it("builds a completed batch summary from resolved jobs", () => {
    expect(
      completedBatchFromJobs([
        { prompt: "one", width: 512, height: 512, steps: 8, seed: 11 },
        { prompt: "two", width: 768, height: 512, steps: 9, seed: 22 },
      ]),
    ).toMatchObject({
      index: 2,
      total: 2,
      prompt: "two",
      jobs: [
        { index: 1, prompt: "one" },
        { index: 2, prompt: "two" },
      ],
    });
  });

  it("keeps only final server batch snapshots for completed summaries", () => {
    expect(
      completedBatchFromSnapshot({
        index: 1,
        total: 2,
        jobs: [{ prompt: "one" }, { prompt: "two" }],
      }),
    ).toBe(null);

    expect(
      completedBatchFromSnapshot({
        index: 2,
        total: 2,
        jobs: [{ prompt: "one" }, { prompt: "two" }],
      }),
    ).toMatchObject({
      index: 2,
      total: 2,
      jobs: [
        { index: 1, prompt: "one" },
        { index: 2, prompt: "two" },
      ],
    });

    expect(
      completedBatchFromSnapshot({
        interrupted: true,
        index: 1,
        total: 1,
        jobs: [{ prompt: "one", status: "cancelled" }],
      }),
    ).toBe(null);
  });

  it("ignores modifier Escape, which the OS never delivers to the page", () => {
    // Cmd+Esc is swallowed by macOS and Ctrl+Esc opens the Start menu on
    // Windows, so a modifier+Escape must not be treated as a cancel chord.
    expect(
      generationCancelKeyAction(keyEvent("Escape", { metaKey: true })),
    ).toBe(null);

    expect(
      generationCancelKeyAction(keyEvent("Escape", { ctrlKey: true })),
    ).toBe(null);

    expect(
      generationCancelKeyAction(keyEvent("Escape", { shiftKey: true })),
    ).toBe(null);
  });

  it("maps a deliberate double Escape to cancel generation", () => {
    expect(
      generationCancelKeyAction(keyEvent("Escape", { timeStamp: 100 }), {
        nowMs: 100,
      }),
    ).toBe("primeEscape");

    expect(
      generationCancelKeyAction(keyEvent("Escape", { timeStamp: 500 }), {
        lastEscapeKeyDownMs: 100,
        nowMs: 500,
      }),
    ).toBe("cancel");

    expect(
      generationCancelKeyAction(keyEvent("Escape", { timeStamp: 1200 }), {
        lastEscapeKeyDownMs: 100,
        nowMs: 1200,
      }),
    ).toBe("primeEscape");

    expect(
      generationCancelKeyAction(keyEvent("a"), {
        lastEscapeKeyDownMs: 100,
        nowMs: 500,
      }),
    ).toBe("clearEscape");

    expect(
      generationCancelKeyAction(
        keyEvent("Escape", {
          defaultPrevented: true,
          timeStamp: 500,
        }),
        {
          lastEscapeKeyDownMs: 100,
          nowMs: 500,
        },
      ),
    ).toBe(null);

    expect(
      generationCancelKeyAction(
        keyEvent("Escape", {
          repeat: true,
          timeStamp: 500,
        }),
        {
          lastEscapeKeyDownMs: 100,
          nowMs: 500,
        },
      ),
    ).toBe(null);
  });

  it("arms queue clearing only after a related batch cancel action", () => {
    expect(
      batchClearQueueFollowupKeyAction(
        keyEvent("Escape", {
          metaKey: true,
          timeStamp: 500,
        }),
        {
          armedUntilMs: 1000,
          nowMs: 500,
        },
      ),
    ).toBe("clearQueue");

    expect(
      batchClearQueueFollowupKeyAction(keyEvent("Escape", { timeStamp: 500 }), {
        armedUntilMs: 1000,
        nowMs: 500,
      }),
    ).toBe("clearQueue");

    expect(
      batchClearQueueFollowupKeyAction(
        keyEvent("Escape", { timeStamp: 1200 }),
        {
          armedUntilMs: 1000,
          nowMs: 1200,
        },
      ),
    ).toBe(null);

    expect(
      batchClearQueueFollowupKeyAction(keyEvent("a", { timeStamp: 500 }), {
        armedUntilMs: 1000,
        nowMs: 500,
      }),
    ).toBe(null);
  });

  it("detects queued batch jobs from explicit or inferred status", () => {
    expect(
      batchHasQueuedJobs({
        jobs: [
          { index: 1, status: "running" },
          { index: 2, status: "queued" },
        ],
      }),
    ).toBe(true);

    expect(
      batchHasQueuedJobs({
        jobs: [
          { index: 1, status: "done" },
          { index: 2, status: "cleared" },
        ],
      }),
    ).toBe(false);

    expect(batchHasQueuedJobs({ index: 1, total: 2 })).toBe(true);
    expect(batchHasQueuedJobs({ index: 2, total: 2 })).toBe(false);
  });

  it("maps spotlight keyboard shortcuts while ignoring unsafe key targets", () => {
    expect(gallerySpotlightKeyAction(keyEvent("ArrowDown"))).toBe("next");
    expect(gallerySpotlightKeyAction(keyEvent("ArrowLeft"))).toBe("previous");
    expect(gallerySpotlightKeyAction(keyEvent("="))).toBe("zoomIn");
    expect(gallerySpotlightKeyAction(keyEvent("+"))).toBe("zoomIn");
    expect(gallerySpotlightKeyAction(keyEvent("-"))).toBe("zoomOut");
    expect(gallerySpotlightKeyAction(keyEvent("0"))).toBe("zoomReset");
    expect(gallerySpotlightKeyAction(keyEvent("Enter"))).toBe("openImage");
    expect(
      gallerySpotlightKeyAction(keyEvent("Enter", { code: "NumpadEnter" })),
    ).toBe("openImage");
    expect(gallerySpotlightKeyAction(keyEvent("Delete"))).toBe("delete");
    expect(gallerySpotlightKeyAction(keyEvent("Escape"))).toBe("close");
    expect(gallerySpotlightKeyAction(keyEvent("g"))).toBe("close");
    expect(gallerySpotlightKeyAction(keyEvent(" "))).toBe(null);

    expect(
      gallerySpotlightKeyAction(
        keyEvent("Backspace", {
          target: { closest: () => true },
        }),
      ),
    ).toBe(null);
    expect(gallerySpotlightKeyAction(keyEvent("=", { ctrlKey: true }))).toBe(
      null,
    );
    expect(gallerySpotlightKeyAction(keyEvent("+", { shiftKey: true }))).toBe(
      null,
    );
  });

  it("maps preview keyboard shortcuts for image cycling and spotlight opening", () => {
    expect(galleryPreviewKeyAction(keyEvent("ArrowDown"))).toBe("next");
    expect(galleryPreviewKeyAction(keyEvent("ArrowRight"))).toBe("next");
    expect(galleryPreviewKeyAction(keyEvent("ArrowUp"))).toBe("previous");
    expect(galleryPreviewKeyAction(keyEvent("ArrowLeft"))).toBe("previous");
    expect(galleryPreviewKeyAction(keyEvent("g"))).toBe("openSpotlight");
    expect(galleryPreviewKeyAction(keyEvent(" "))).toBe("openSpotlight");
    expect(galleryPreviewKeyAction(keyEvent("G", { shiftKey: true }))).toBe(
      null,
    );
    expect(
      galleryPreviewKeyAction(
        keyEvent("g", {
          target: { closest: () => true },
        }),
      ),
    ).toBe(null);
  });

  it("wraps spotlight gallery navigation and deletion fallback", () => {
    const items = [{ id: "first" }, { id: "second" }, { id: "third" }];

    expect(wrappedGalleryIndex(0, -1, items.length)).toBe(2);
    expect(galleryImageByOffset(items, "third", 1)).toBe(items[0]);
    expect(galleryImageByOffset(items, "missing", 1)).toBe(items[1]);
    expect(nextGalleryImageIdAfterDelete(items, "third")).toBe("first");
    expect(nextGalleryImageIdAfterDelete([items[0]], "first")).toBe(null);
  });

  it("snaps spotlight zoom through fit width and 100% without treating fit as a cap", () => {
    expect(nextGallerySpotlightZoom(1.1, -1, Number.POSITIVE_INFINITY)).toBe(1);
    expect(nextGallerySpotlightZoom(0.9, 1, Number.POSITIVE_INFINITY)).toBe(1);
    expect(nextGallerySpotlightZoom(0.8, 1, 4, 0.92)).toBe(0.92);
    expect(nextGallerySpotlightZoom(0.92, 1, 4, 0.92)).toBe(1);
    expect(nextGallerySpotlightZoom(1, 1, 4, 0.92)).toBe(1.25);
    expect(nextGallerySpotlightZoom(1.2, -1, 4, 0.92)).toBe(1);
    expect(nextGallerySpotlightZoom(1, -1, 4, 0.92)).toBe(0.92);
    expect(nextGallerySpotlightZoom(3.9, 1, 4, 0.92)).toBe(4);
    expect(formatGallerySpotlightZoom(1.1)).toBe("110%");
  });

  it("starts spotlight from the top image during batch generation", () => {
    const recent = [{ id: "top" }, { id: "selected" }];

    expect(
      gallerySpotlightInitialImage({
        activeImage: recent[1],
        batch: { running: true },
        recent,
      }),
    ).toBe(recent[0]);

    expect(
      gallerySpotlightInitialImage({
        activeImage: recent[1],
        batch: null,
        recent,
      }),
    ).toBe(recent[1]);
  });

  it("opens spotlight from gallery clicks while generation owns the preview", () => {
    expect(
      galleryImageClickAction({
        desktopViewport: true,
        spotlightActive: false,
        stageGenerating: true,
      }),
    ).toBe("openSpotlight");

    expect(
      galleryImageClickAction({
        desktopViewport: true,
        spotlightActive: true,
        stageGenerating: true,
      }),
    ).toBe("focusSpotlight");

    expect(
      galleryImageClickAction({
        desktopViewport: false,
        spotlightActive: false,
        stageGenerating: true,
      }),
    ).toBe("selectPreview");

    expect(
      galleryImageClickAction({
        desktopViewport: true,
        spotlightActive: false,
        stageGenerating: false,
      }),
    ).toBe("selectPreview");
  });

  it("tracks generated images unseen by the active spotlight focus", () => {
    const previousRecentIds = new Set(["old"]);
    const recent = [{ id: "newest" }, { id: "old" }];

    expect(
      nextGallerySpotlightUnseenIds({
        currentIds: [],
        focusedId: "old",
        previousRecentIds,
        recent,
        spotlightActive: true,
      }),
    ).toEqual(["newest"]);

    expect(
      nextGallerySpotlightUnseenIds({
        currentIds: ["newest"],
        focusedId: "newest",
        previousRecentIds: new Set(["newest", "old"]),
        recent,
        spotlightActive: true,
      }),
    ).toEqual([]);

    expect(
      nextGallerySpotlightUnseenIds({
        currentIds: ["newest"],
        focusedId: "old",
        previousRecentIds,
        recent,
        spotlightActive: false,
      }),
    ).toEqual([]);
  });
});

function keyEvent(key, overrides = {}) {
  return {
    altKey: false,
    ctrlKey: false,
    defaultPrevented: false,
    isComposing: false,
    key,
    metaKey: false,
    repeat: false,
    shiftKey: false,
    target: null,
    timeStamp: 0,
    ...overrides,
  };
}
