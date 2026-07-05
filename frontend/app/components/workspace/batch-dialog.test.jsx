// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { BatchDialog } from "./batch-dialog";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const INVALID_BATCH_TEXT = `[
  {
    "prompt": "one",
  }
]`;
const JSON_ERROR =
  "Expected double-quoted property name in JSON at position 31";

describe("BatchDialog", () => {
  let container;
  let root;
  let readBatchClipboard;

  beforeEach(() => {
    vi.useFakeTimers();
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    readBatchClipboard = vi.fn(async () => ({
      source: "Clipboard",
      text: INVALID_BATCH_TEXT,
    }));
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.replaceChildren();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("does not restart invalid JSON validation when the validator callback identity changes", async () => {
    const firstValidateBatch = vi.fn(async () => {
      throw new Error(JSON_ERROR);
    });
    const secondValidateBatch = vi.fn(async () => {
      throw new Error("Unexpected revalidation");
    });

    renderDialog({ validateBatch: firstValidateBatch });
    await clickButton("Clipboard");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(350);
    });

    expect(firstValidateBatch).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain(JSON_ERROR);
    expect(document.body.textContent).not.toContain("Validating...");

    renderDialog({ validateBatch: secondValidateBatch });

    expect(document.body.textContent).toContain(JSON_ERROR);
    expect(document.body.textContent).not.toContain("Validating...");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });

    expect(secondValidateBatch).not.toHaveBeenCalled();
  });

  function renderDialog({ validateBatch }) {
    act(() => {
      root.render(
        <TooltipProvider>
          <BatchDialog
            defaultOutputDir="outputs"
            disabled={false}
            generateBatch={vi.fn()}
            onOpenChange={vi.fn()}
            open
            readBatchClipboard={readBatchClipboard}
            selectOutputDir={vi.fn()}
            validateBatch={validateBatch}
          />
        </TooltipProvider>,
      );
    });
  }
});

async function clickButton(label) {
  const button = Array.from(document.body.querySelectorAll("button")).find(
    (element) => element.textContent.includes(label),
  );
  expect(button).toBeTruthy();

  await act(async () => {
    button.click();
    await Promise.resolve();
  });
}
