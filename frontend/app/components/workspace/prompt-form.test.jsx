// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { PromptForm } from "./prompt-form";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let root = null;
let container = null;

afterEach(() => {
  if (root) {
    act(() => root.unmount());
  }
  root = null;
  container?.remove();
  container = null;
});

describe("PromptForm", () => {
  it("exposes an inline clear action for non-empty prompts", () => {
    const onClearPrompt = vi.fn();

    renderPromptForm({
      onClearPrompt,
      prompt: '[{"prompt":"one"}]',
    });

    click('[aria-label="Clear prompt"]');

    expect(onClearPrompt).toHaveBeenCalledTimes(1);
  });

  it("omits the clear action when the prompt is empty", () => {
    renderPromptForm({ prompt: "" });

    expect(container.querySelector('[aria-label="Clear prompt"]')).toBeNull();
  });
});

function renderPromptForm(overrides = {}) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);

  act(() => {
    root.render(
      <TooltipProvider>
        <PromptForm
          canSubmit
          cancelRequested={false}
          generationRunning={false}
          onCancel={vi.fn()}
          onClearPrompt={vi.fn()}
          onPromptChange={vi.fn()}
          onSubmit={(event) => event.preventDefault()}
          prompt="a glass observatory"
          {...overrides}
        />
      </TooltipProvider>,
    );
  });
}

function click(selector) {
  const element = container.querySelector(selector);
  expect(element).not.toBeNull();
  act(() => {
    element.click();
  });
}
