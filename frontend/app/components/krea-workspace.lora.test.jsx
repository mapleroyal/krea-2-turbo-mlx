// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_STATUS, useGuiStore } from "@/stores/use-gui-store";

import { KreaWorkspace } from "./krea-workspace";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const baseStoreState = useGuiStore.getState();

let root = null;
let container = null;

afterEach(() => {
  if (root) {
    act(() => root.unmount());
  }
  root = null;
  container?.remove();
  container = null;
  useGuiStore.setState(baseStoreState, true);
  vi.unstubAllGlobals();
});

describe("KreaWorkspace LoRA controls", () => {
  it("clears a completed batch report without clearing the prompt", async () => {
    useGuiStore.setState({
      status: statusWithCatalogLora([]),
      initialized: true,
      initializeSession: vi.fn(),
      startPolling: vi.fn(),
      stopPolling: vi.fn(),
    });

    await renderWorkspace();
    input('textarea[aria-label="Prompt"]', "keep this prompt");

    await act(async () => {
      useGuiStore.setState((state) => ({
        status: {
          ...state.status,
          batch: completedBatchSnapshot(),
        },
      }));
    });
    await act(async () => {
      useGuiStore.setState((state) => ({
        status: {
          ...state.status,
          batch: null,
        },
      }));
    });

    expect(container.textContent).toContain("Done");
    click('[aria-label="Clear batch report"]');

    expect(inputValue('textarea[aria-label="Prompt"]')).toBe(
      "keep this prompt",
    );
    expect(
      container.querySelector('[aria-label="Clear batch report"]'),
    ).toBeNull();
  });

  it("enables and disables multiple LoRAs through workspace controls", async () => {
    const persistUiSettings = vi.fn(async (settings) => settings);
    useGuiStore.setState({
      status: statusWithCatalogLora([
        { id: "portraits/soft.safetensors", scale: 1.5 },
      ]),
      initialized: true,
      initializeSession: vi.fn(),
      startPolling: vi.fn(),
      stopPolling: vi.fn(),
      persistUiSettings,
    });

    await renderWorkspace();

    click("#lora-styles-glass-safetensors");
    expect(persistUiSettings).toHaveBeenLastCalledWith(
      expect.objectContaining({
        loras: [
          { id: "portraits/soft.safetensors", scale: 1.5 },
          { id: "styles/glass.safetensors", scale: 1 },
        ],
      }),
    );

    click("#lora-portraits-soft-safetensors");
    expect(persistUiSettings).toHaveBeenLastCalledWith(
      expect.objectContaining({
        loras: [{ id: "styles/glass.safetensors", scale: 1 }],
      }),
    );
  });

  it("clamps edited LoRA scales before persisting workspace settings", async () => {
    const persistUiSettings = vi.fn(async (settings) => settings);
    const status = statusWithCatalogLora([
      { id: "styles/glass.safetensors", scale: 2 },
    ]);
    useGuiStore.setState({
      status,
      initialized: true,
      initializeSession: vi.fn(),
      startPolling: vi.fn(),
      stopPolling: vi.fn(),
      persistUiSettings,
    });

    await renderWorkspace();
    expect(inputValue("#lora-styles-glass-safetensors-scale")).toBe("2");

    input("#lora-styles-glass-safetensors-scale", "9");
    blur("#lora-styles-glass-safetensors-scale");

    expect(inputValue("#lora-styles-glass-safetensors-scale")).toBe("4");
    expect(persistUiSettings).toHaveBeenLastCalledWith(
      expect.objectContaining({
        loras: [{ id: "styles/glass.safetensors", scale: 4 }],
      }),
    );
  });
});

async function renderWorkspace() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);

  await act(async () => {
    root.render(<KreaWorkspace />);
  });
}

function statusWithCatalogLora(loras) {
  const status = JSON.parse(JSON.stringify(DEFAULT_STATUS));
  status.server = { connected: true, status: "ready" };
  status.phase = "idle";
  status.message = "Ready";
  status.ui_settings = {
    loras,
  };
  status.loras.items = [
    {
      id: "portraits/soft.safetensors",
      display_name: "Soft Portraits",
      source_type: "catalog",
      adapter_type: "standard",
      default_scale: 1,
      scale_min: 0,
      scale_max: 4,
      target_count: 1,
      skipped_count: 0,
      warnings: [],
    },
    {
      id: "styles/glass.safetensors",
      display_name: "Glass",
      source_type: "catalog",
      adapter_type: "standard",
      default_scale: 1,
      scale_min: 0,
      scale_max: 4,
      target_count: 1,
      skipped_count: 0,
      warnings: [],
    },
  ];
  return status;
}

function completedBatchSnapshot() {
  const job = {
    index: 1,
    prompt: "finished batch prompt",
    width: 512,
    height: 512,
    steps: 8,
    seed: 7,
    loras: [],
  };

  return {
    ...job,
    total: 1,
    jobs: [job],
  };
}

function inputValue(selector) {
  const element = query(selector);
  return element.value;
}

function blur(selector) {
  const element = query(selector);
  act(() => {
    element.dispatchEvent(new window.FocusEvent("focusout", { bubbles: true }));
  });
}

function click(selector) {
  const element = query(selector);
  act(() => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

function input(selector, value) {
  const element = query(selector);
  act(() => {
    setNativeValue(element, value);
    element.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}

function query(selector) {
  const element = container.querySelector(selector);
  expect(element).not.toBeNull();
  return element;
}

function setNativeValue(element, value) {
  const descriptor = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(element),
    "value",
  );
  descriptor?.set?.call(element, value);
}
