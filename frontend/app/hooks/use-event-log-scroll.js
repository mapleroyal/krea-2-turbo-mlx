import { useEffect, useLayoutEffect, useRef } from "react";

function resolveViewport(container) {
  return (
    container?.querySelector?.("[data-slot='scroll-area-viewport']") ??
    container ??
    null
  );
}

export function useEventLogScroll(containerRef, events) {
  const stateRef = useRef({
    atTop: true,
    scrollHeight: 0,
    scrollTop: 0,
  });
  const eventSignature = (events ?? [])
    .map((event) =>
      [
        event.id,
        event.time,
        event.kind,
        event.stage,
        event.message,
        event.time_ms,
        event.completed_ms,
        event.progress,
        event.step_index,
        event.step_count,
      ].join("\u001f"),
    )
    .join("\u001e");

  useLayoutEffect(() => {
    const viewport = resolveViewport(containerRef.current);

    if (!viewport) {
      return;
    }

    const previous = stateRef.current;
    const nextScrollHeight = viewport.scrollHeight;

    if (previous.scrollHeight > 0) {
      if (previous.atTop) {
        viewport.scrollTop = 0;
      } else {
        viewport.scrollTop =
          previous.scrollTop + nextScrollHeight - previous.scrollHeight;
      }
    }

    stateRef.current = {
      atTop: viewport.scrollTop <= 2,
      scrollHeight: viewport.scrollHeight,
      scrollTop: viewport.scrollTop,
    };
  }, [containerRef, eventSignature]);

  useEffect(() => {
    const viewport = resolveViewport(containerRef.current);

    if (!viewport) {
      return undefined;
    }

    const recordScrollState = () => {
      stateRef.current = {
        atTop: viewport.scrollTop <= 2,
        scrollHeight: viewport.scrollHeight,
        scrollTop: viewport.scrollTop,
      };
    };

    recordScrollState();
    viewport.addEventListener("scroll", recordScrollState, { passive: true });

    return () => {
      viewport.removeEventListener("scroll", recordScrollState);
    };
  }, [containerRef]);
}
