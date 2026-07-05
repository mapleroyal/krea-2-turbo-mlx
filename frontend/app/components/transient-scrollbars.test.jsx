// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TransientScrollbars } from "./transient-scrollbars";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

describe("TransientScrollbars", () => {
  let container;
  let root;

  beforeEach(() => {
    vi.useFakeTimers();
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);

    act(() => {
      root.render(<TransientScrollbars />);
    });
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    vi.useRealTimers();
  });

  it("marks a native scroll container only while scrolling is active", () => {
    const scroller = document.createElement("div");
    document.body.append(scroller);

    act(() => {
      scroller.dispatchEvent(new globalThis.Event("scroll"));
    });

    expect(scroller.getAttribute("data-scrollbar-scrolling")).toBe("true");

    act(() => {
      vi.advanceTimersByTime(699);
    });

    expect(scroller.getAttribute("data-scrollbar-scrolling")).toBe("true");

    act(() => {
      vi.advanceTimersByTime(1);
    });

    expect(scroller.hasAttribute("data-scrollbar-scrolling")).toBe(false);

    scroller.remove();
  });
});
