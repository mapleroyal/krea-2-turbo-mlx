// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { GallerySpotlight } from "./gallery-spotlight";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let root = null;
let container = null;

const IMAGE = {
  filename: "image.png",
  height: 512,
  id: "image",
  prompt: "spotlight prompt",
  seed: 7,
  steps: 8,
  url: "/api/image/7",
  width: 512,
};

afterEach(() => {
  if (root) {
    act(() => root.unmount());
  }
  root = null;
  container?.remove();
  container = null;
});

describe("GallerySpotlight", () => {
  it("renders nothing without an image", () => {
    renderSpotlight({ image: null });

    expect(container.textContent).toBe("");
    expect(container.querySelector("img")).toBeNull();
  });

  it("links the full-resolution image through the tokenized url", () => {
    renderSpotlight();

    const anchor = container.querySelector("a");
    const img = container.querySelector("img");
    expect(anchor.getAttribute("href")).toBe("/api/image/7?token=secret");
    expect(anchor.getAttribute("target")).toBe("_blank");
    expect(anchor.getAttribute("rel")).toContain("noopener");
    expect(img.getAttribute("src")).toBe("/api/image/7?token=secret");
  });

  it("scales the displayed frame width by the zoom factor", () => {
    renderSpotlight({ zoomScale: 1.25 });
    expect(frameWidth()).toBe("640px");

    renderSpotlight({ zoomScale: 2 });
    expect(frameWidth()).toBe("1024px");
  });

  it("dismisses on a backdrop click but not on an image click", () => {
    const onDismiss = vi.fn();
    renderSpotlight({ onDismiss });

    // Clicking the image itself must not close the spotlight.
    click(container.querySelector("a").parentElement);
    expect(onDismiss).not.toHaveBeenCalled();

    // Clicking the surrounding viewport (the backdrop) closes it.
    click(viewport());
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("suppresses the dismiss click that ends a pan drag", () => {
    const onDismiss = vi.fn();
    renderSpotlight({ onDismiss });

    // A pointer drag past the movement threshold, then the click it produces,
    // must not be treated as a backdrop dismiss.
    pointer("pointerdown", { clientX: 100, clientY: 100 });
    pointer("pointermove", { clientX: 160, clientY: 130 });
    pointer("pointerup", { clientX: 160, clientY: 130 });
    click(viewport());

    expect(onDismiss).not.toHaveBeenCalled();
  });

  it("reports the measured viewport width on mount", () => {
    const onViewportWidthChange = vi.fn();
    renderSpotlight({ onViewportWidthChange });

    expect(onViewportWidthChange).toHaveBeenCalledWith(expect.any(Number));
  });
});

function renderSpotlight(overrides = {}) {
  if (!container) {
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
  }

  act(() => {
    root.render(
      <TooltipProvider>
        <GallerySpotlight
          fallbackPrecision="bf16"
          image={IMAGE}
          imageUrl={(url) => `${url}?token=secret`}
          onDismiss={() => {}}
          onViewportWidthChange={() => {}}
          zoomScale={1.25}
          {...overrides}
        />
      </TooltipProvider>,
    );
  });
}

function viewport() {
  return container.firstElementChild;
}

function frameWidth() {
  return container.querySelector("a").parentElement.style.width;
}

function click(element) {
  act(() => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

function pointer(type, { clientX, clientY }) {
  act(() => {
    viewport().dispatchEvent(
      new window.MouseEvent(type, {
        bubbles: true,
        button: 0,
        clientX,
        clientY,
      }),
    );
  });
}
