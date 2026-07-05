import { useEffect } from "react";

const SCROLLING_ATTRIBUTE = "data-scrollbar-scrolling";
const SCROLLBAR_HIDE_DELAY_MS = 700;

function scrollElementFromTarget(target) {
  if (target === document) {
    return document.scrollingElement ?? document.documentElement;
  }

  return target instanceof globalThis.HTMLElement ? target : null;
}

export function TransientScrollbars() {
  useEffect(() => {
    const hideTimers = new Map();

    const markScrolling = (element) => {
      if (!(element instanceof globalThis.HTMLElement)) {
        return;
      }

      element.setAttribute(SCROLLING_ATTRIBUTE, "true");

      const existingTimer = hideTimers.get(element);

      if (existingTimer) {
        window.clearTimeout(existingTimer);
      }

      const hideTimer = window.setTimeout(() => {
        element.removeAttribute(SCROLLING_ATTRIBUTE);
        hideTimers.delete(element);
      }, SCROLLBAR_HIDE_DELAY_MS);

      hideTimers.set(element, hideTimer);
    };

    const handleElementScroll = (event) => {
      markScrolling(scrollElementFromTarget(event.target));
    };

    const handleWindowScroll = () => {
      markScrolling(document.scrollingElement ?? document.documentElement);
    };

    document.addEventListener("scroll", handleElementScroll, {
      capture: true,
      passive: true,
    });
    window.addEventListener("scroll", handleWindowScroll, { passive: true });

    return () => {
      document.removeEventListener("scroll", handleElementScroll, true);
      window.removeEventListener("scroll", handleWindowScroll);

      for (const [element, hideTimer] of hideTimers) {
        window.clearTimeout(hideTimer);
        element.removeAttribute(SCROLLING_ATTRIBUTE);
      }

      hideTimers.clear();
    };
  }, []);

  return null;
}
