// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";
import { DEFAULT_STATUS } from "@/stores/use-gui-store";

import { ControlRail } from "./control-rail";

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

describe("ControlRail", () => {
  it("commits numeric fields on blur without leaking DOM events", () => {
    const onDimensionsBlur = vi.fn();
    const onStepsBlur = vi.fn();
    const onSimpleBatchCountBlur = vi.fn();

    renderControlRail({
      onDimensionsBlur,
      onSimpleBatchCountBlur,
      onStepsBlur,
    });

    blur("#width");
    blur("#height");
    blur("#steps");
    blur("#simple-batch-count");

    expect(onDimensionsBlur).toHaveBeenCalledTimes(2);
    expect(onDimensionsBlur).toHaveBeenNthCalledWith(1);
    expect(onDimensionsBlur).toHaveBeenNthCalledWith(2);
    expect(onStepsBlur).toHaveBeenCalledWith();
    expect(onSimpleBatchCountBlur).toHaveBeenCalledWith();
  });

  it("renders catalog LoRAs and routes refresh and scale events", () => {
    const onLoraRefresh = vi.fn();
    const onLoraScaleBlur = vi.fn();
    const onLoraScaleChange = vi.fn();
    renderControlRail({
      loraItems: [
        ...DEFAULT_STATUS.loras.items,
        {
          id: "styles/glass.safetensors",
          display_name: "Glass",
          default_scale: 1,
          scale_min: 0,
          scale_max: 4,
          warnings: [],
        },
      ],
      selectedLoras: [{ id: "styles/glass.safetensors", scale: 2 }],
      onLoraRefresh,
      onLoraScaleBlur,
      onLoraScaleChange,
    });

    click('[aria-label="Refresh LoRAs"]');
    input("#lora-styles-glass-safetensors-scale", "2.5");
    blur("#lora-styles-glass-safetensors-scale");

    expect(container.textContent).toContain("Glass");
    expect(onLoraRefresh).toHaveBeenCalledWith();
    expect(onLoraScaleChange).toHaveBeenCalledWith(
      "styles/glass.safetensors",
      "2.5",
    );
    expect(onLoraScaleBlur).toHaveBeenCalledWith("styles/glass.safetensors");
  });
});

function renderControlRail(overrides = {}) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);

  act(() => {
    root.render(
      <TooltipProvider>
        <ControlRail
          busy={false}
          constraints={DEFAULT_STATUS.constraints}
          generatedSeed="123"
          height="1024"
          loraCatalog={DEFAULT_STATUS.loras}
          loraItems={DEFAULT_STATUS.loras.items}
          loraWarning=""
          onBatchOpen={vi.fn()}
          onDimensionPresetChange={vi.fn()}
          onDimensionsBlur={vi.fn()}
          onLoraEnabledChange={vi.fn()}
          onLoraRefresh={vi.fn()}
          onLoraScaleBlur={vi.fn()}
          onLoraScaleChange={vi.fn()}
          onLoadParamsOpen={vi.fn()}
          onRandomSeed={vi.fn()}
          onRandomizationLockChange={vi.fn()}
          onSimpleBatchCountBlur={vi.fn()}
          onSimpleBatchCountChange={vi.fn()}
          onSimpleBatchEnabledChange={vi.fn()}
          onStepsBlur={vi.fn()}
          randomizationLocked={false}
          seed=""
          selectedLoras={[]}
          setHeight={vi.fn()}
          setSeed={vi.fn()}
          setSteps={vi.fn()}
          setWidth={vi.fn()}
          simpleBatchCount="4"
          simpleBatchEnabled
          simpleBatchMaxCount={DEFAULT_STATUS.constraints.max_batch_jobs}
          simpleBatchWarning=""
          steps="12"
          width="1024"
          {...overrides}
        />
      </TooltipProvider>,
    );
  });
}

function blur(selector) {
  const element = container.querySelector(selector);
  expect(element).not.toBeNull();
  act(() => {
    element.dispatchEvent(new window.FocusEvent("focusout", { bubbles: true }));
  });
}

function click(selector) {
  const element = container.querySelector(selector);
  expect(element).not.toBeNull();
  act(() => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

function input(selector, value) {
  const element = container.querySelector(selector);
  expect(element).not.toBeNull();
  act(() => {
    setNativeValue(element, value);
    element.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}

function setNativeValue(element, value) {
  const descriptor = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(element),
    "value",
  );
  descriptor?.set?.call(element, value);
}
