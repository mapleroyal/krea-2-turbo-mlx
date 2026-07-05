// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { BatchStatus } from "./batch-status";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

describe("BatchStatus", () => {
  let container;
  let root;
  let scrollIntoView;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    scrollIntoView = vi.fn();
    globalThis.Element.prototype.scrollIntoView = scrollIntoView;
    vi.spyOn(
      globalThis.Element.prototype,
      "getBoundingClientRect",
    ).mockImplementation(function getBoundingClientRect() {
      if (this.getAttribute("data-slot") === "scroll-area-viewport") {
        return domRect(10);
      }

      if (this.getAttribute("aria-current") === "step") {
        return domRect(70);
      }

      return domRect(0);
    });
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("expands from the current job into the full batch queue", () => {
    renderBatch(batchSnapshot(2));

    expect(container.textContent).toContain("a cedar observatory");
    expect(container.textContent).not.toContain("a glass library");
    expect(container.textContent).not.toContain("a moonlit greenhouse");
    expect(batchPromptElements()).toHaveLength(1);

    act(() => {
      container.querySelector('[aria-label="Expand batch jobs"]').click();
    });

    expect(container.textContent).toContain("a glass library");
    expect(container.textContent).toContain("a cedar observatory");
    expect(container.textContent).toContain("a moonlit greenhouse");
    expect(scrollIntoView).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-slot="scroll-area-viewport"]').scrollTop,
    ).toBe(60);
    expect(batchPromptElements()).toHaveLength(3);

    renderBatch(batchSnapshot(3));

    expect(
      container.querySelector('[aria-current="step"]').textContent,
    ).toContain("Job 3 of 3");
    expect(scrollIntoView).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-slot="scroll-area-viewport"]').scrollTop,
    ).toBe(60);

    act(() => {
      container.querySelector('[aria-label="Collapse batch jobs"]').click();
    });

    expect(container.textContent).toContain("a moonlit greenhouse");
    expect(container.textContent).not.toContain("a glass library");
    expect(container.textContent).not.toContain("a cedar observatory");
  });

  it("shows completed jobs while collapsed with clear and resubmit actions", () => {
    const onClear = vi.fn();
    const onExpandedChange = vi.fn();
    const onResubmit = vi.fn();

    renderBatch(batchSnapshot(3), {
      completed: true,
      expanded: false,
      onClear,
      onExpandedChange,
      onResubmit,
    });

    expect(container.textContent).toContain("a glass library");
    expect(container.textContent).toContain("a cedar observatory");
    expect(container.textContent).toContain("a moonlit greenhouse");
    expect(batchPromptElements()).toHaveLength(3);
    expect(container.textContent).toContain("Done");
    expect(
      container.querySelector('[aria-label="Cancel current job"]'),
    ).toBeNull();
    expect(container.querySelector('[aria-label="Clear queue"]')).toBeNull();

    act(() => {
      container.querySelector('[aria-label="Rerun batch"]').click();
    });
    act(() => {
      container.querySelector('[aria-label="Clear batch report"]').click();
    });
    act(() => {
      container.querySelector('[aria-label="Expand batch jobs"]').click();
    });

    expect(onResubmit).toHaveBeenCalledTimes(1);
    expect(onClear).toHaveBeenCalledTimes(1);
    expect(onExpandedChange).toHaveBeenCalledWith(true);
  });

  it("offers separate active-job cancel and queue clear actions", () => {
    const onCancelCurrent = vi.fn();
    const onClearQueue = vi.fn();

    renderBatch(batchSnapshot(2), {
      onCancelCurrent,
      onClearQueue,
    });

    act(() => {
      container.querySelector('[aria-label="Cancel current job"]').click();
    });
    act(() => {
      container.querySelector('[aria-label="Clear queue"]').click();
    });

    expect(onCancelCurrent).toHaveBeenCalledTimes(1);
    expect(onClearQueue).toHaveBeenCalledTimes(1);
  });

  it("marks cancelled and cleared batch rows from snapshot status", () => {
    renderBatch(
      {
        ...batchSnapshot(2),
        cancel_current_requested: true,
        clear_queue_requested: true,
        jobs: batchSnapshot(2).jobs.map((job, index) =>
          index === 1
            ? { ...job, status: "cancelling" }
            : index === 2
              ? { ...job, status: "cleared" }
              : { ...job, status: "done" },
        ),
      },
      {
        expanded: true,
      },
    );

    expect(container.textContent).toContain("Cancelling");
    expect(container.textContent).toContain("Cleared");
    expect(
      container.querySelector('[aria-label="Cancelling current job"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[aria-label="Queue cleared"]'),
    ).not.toBeNull();
  });

  it("marks GUI-derived completed batch metadata", () => {
    renderBatch(
      {
        ...batchSnapshot(3),
        jobs: batchSnapshot(3).jobs.map((job, index) =>
          index === 0 ? { ...job, guiDerivedFields: ["width", "seed"] } : job,
        ),
      },
      {
        completed: true,
      },
    );

    expect(container.textContent).toContain(
      "* Reruns use the current value from the GUI control.",
    );
    expect(container.textContent).toContain("bf16 @ 1024*x1024");
    expect(container.textContent).toContain("Seed: 11*");
  });

  function renderBatch(batch, props = {}) {
    act(() => {
      root.render(
        <TooltipProvider>
          <BatchStatus
            batch={batch}
            fallbackPrecision="bf16"
            onCancel={() => {}}
            {...props}
          />
        </TooltipProvider>,
      );
    });
  }

  function batchPromptElements() {
    return Array.from(
      container.querySelectorAll('[data-slot="batch-job-prompt"]'),
    );
  }
});

function batchSnapshot(index) {
  const jobs = [
    {
      index: 1,
      prompt: "a glass library",
      width: 1024,
      height: 1024,
      seed: 11,
      steps: 8,
    },
    {
      index: 2,
      prompt: "a cedar observatory",
      width: 1024,
      height: 1024,
      seed: 22,
      steps: 8,
    },
    {
      index: 3,
      prompt: "a moonlit greenhouse",
      width: 1024,
      height: 1024,
      seed: 33,
      steps: 8,
    },
  ];
  const current = jobs[index - 1];

  return {
    ...current,
    index,
    total: jobs.length,
    jobs,
    progress: 0.4,
    overall_progress: (index - 1 + 0.4) / jobs.length,
  };
}

function domRect(top) {
  return {
    bottom: top + 20,
    height: 20,
    left: 0,
    right: 100,
    top,
    width: 100,
    x: 0,
    y: top,
    toJSON() {
      return this;
    },
  };
}
