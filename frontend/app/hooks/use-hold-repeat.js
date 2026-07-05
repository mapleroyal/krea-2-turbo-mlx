import { useCallback, useEffect, useRef } from "react";

export function useHoldRepeat({
  onStep,
  initialDelay = 300,
  minInterval = 55,
  accel = 0.82,
}) {
  const onStepRef = useRef(onStep);
  const timerRef = useRef(null);
  const keyActiveRef = useRef(false);
  const pointerActiveRef = useRef(false);

  useEffect(() => {
    onStepRef.current = onStep;
  }, [onStep]);

  const stop = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    pointerActiveRef.current = false;
  }, []);

  const repeat = useCallback(
    (step, interval) => {
      onStepRef.current(step);
      const nextInterval = Math.max(minInterval, Math.round(interval * accel));
      timerRef.current = setTimeout(
        () => repeat(step, nextInterval),
        nextInterval,
      );
    },
    [accel, minInterval],
  );

  const start = useCallback(
    (step) => {
      stop();
      onStepRef.current(step);
      timerRef.current = setTimeout(
        () => repeat(step, initialDelay),
        initialDelay,
      );
    },
    [initialDelay, repeat, stop],
  );

  useEffect(() => stop, [stop]);

  useEffect(() => {
    window.addEventListener("pointercancel", stop);
    window.addEventListener("pointerup", stop);

    return () => {
      window.removeEventListener("pointercancel", stop);
      window.removeEventListener("pointerup", stop);
    };
  }, [stop]);

  const getRepeatProps = useCallback(
    (step) => ({
      onBlur: () => {
        keyActiveRef.current = false;
        stop();
      },
      onKeyDown: (event) => {
        if (
          (event.key === "Enter" || event.key === " ") &&
          !keyActiveRef.current
        ) {
          event.preventDefault();
          keyActiveRef.current = true;
          start(step);
        }
      },
      onKeyUp: (event) => {
        if (event.key === "Enter" || event.key === " ") {
          keyActiveRef.current = false;
          stop();
        }
      },
      onPointerCancel: stop,
      onPointerDown: (event) => {
        if (event.button !== 0) {
          return;
        }

        pointerActiveRef.current = true;
        event.preventDefault();
        event.currentTarget.setPointerCapture?.(event.pointerId);
        start(step);
      },
      onLostPointerCapture: () => {
        if (pointerActiveRef.current) {
          stop();
        }
      },
      onPointerUp: stop,
    }),
    [start, stop],
  );

  return { getRepeatProps, stop };
}
