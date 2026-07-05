import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import AiContentGenerator01Icon from "@hugeicons/core-free-icons/AiContentGenerator01Icon";
import CheckmarkCircle02Icon from "@hugeicons/core-free-icons/CheckmarkCircle02Icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { createPortal } from "react-dom";

import { cn } from "@/lib/utils";

import { IconActionButton } from "./icon-action-button";

const VIEWPORT_MARGIN = 12;
const COPIED_FADE_IN_MS = 300;
const COPIED_HOLD_MS = 1500;
const COPIED_FADE_OUT_MS = 750;

const useBrowserLayoutEffect =
  typeof window === "undefined" ? useEffect : useLayoutEffect;

export function PromptInfoButton({ anchorRef, className, prompt }) {
  const popupId = useId();
  const triggerRef = useRef(null);
  const [open, setOpen] = useState(false);

  if (!prompt) {
    return null;
  }

  const handleClick = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setOpen((currentOpen) => !currentOpen);
  };

  return (
    <>
      <IconActionButton
        ariaLabel="Show prompt"
        buttonRef={triggerRef}
        className={className}
        expanded={open}
        icon={AiContentGenerator01Icon}
        onClick={handleClick}
        tooltip="Show prompt"
      />
      <PromptPopup
        anchorRef={anchorRef}
        id={popupId}
        onOpenChange={setOpen}
        open={open}
        prompt={prompt}
        triggerRef={triggerRef}
      />
    </>
  );
}

function PromptPopup({
  anchorRef,
  id,
  onOpenChange,
  open,
  prompt,
  triggerRef,
}) {
  const popupRef = useRef(null);
  const timersRef = useRef([]);
  const [phase, setPhase] = useState("idle");
  const [position, setPosition] = useState(null);

  useEffect(() => {
    if (open) {
      setPhase("idle");
      return;
    }

    clearTimers(timersRef);
    setPhase("idle");
    setPosition(null);
  }, [open]);

  useEffect(() => {
    return () => clearTimers(timersRef);
  }, []);

  useBrowserLayoutEffect(() => {
    if (!open) {
      return;
    }

    const updatePosition = () => {
      const anchor = anchorRef.current;
      const popup = popupRef.current;

      if (!anchor || !popup) {
        return;
      }

      const nextPosition = promptPopupPositionForImage(
        anchor.getBoundingClientRect(),
        popup.getBoundingClientRect(),
        {
          viewportHeight: window.innerHeight,
          viewportWidth: window.innerWidth,
        },
      );
      setPosition((currentPosition) =>
        samePosition(currentPosition, nextPosition)
          ? currentPosition
          : nextPosition,
      );
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);

    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [anchorRef, open]);

  useBrowserLayoutEffect(() => {
    if (!open || phase !== "idle") {
      return;
    }

    const anchor = anchorRef.current;
    const popup = popupRef.current;

    if (!anchor || !popup) {
      return;
    }

    const nextPosition = promptPopupPositionForImage(
      anchor.getBoundingClientRect(),
      popup.getBoundingClientRect(),
      {
        viewportHeight: window.innerHeight,
        viewportWidth: window.innerWidth,
      },
    );
    setPosition((currentPosition) =>
      samePosition(currentPosition, nextPosition)
        ? currentPosition
        : nextPosition,
    );
  }, [anchorRef, open, phase, prompt]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const handlePointerDown = (event) => {
      const target = event.target;

      if (
        popupRef.current?.contains(target) ||
        triggerRef.current?.contains(target)
      ) {
        return;
      }

      onOpenChange(false);
    };
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        onOpenChange(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onOpenChange, open, triggerRef]);

  useEffect(() => {
    if (open && position) {
      popupRef.current?.focus();
    }
  }, [open, position]);

  if (!open) {
    return null;
  }

  const handleCopy = async () => {
    if (phase !== "idle") {
      return;
    }

    await copyTextToClipboard(prompt);
    setPhase("copied");
    clearTimers(timersRef);
    timersRef.current = [
      window.setTimeout(() => {
        setPhase("closing");
      }, COPIED_FADE_IN_MS + COPIED_HOLD_MS),
      window.setTimeout(
        () => {
          onOpenChange(false);
        },
        COPIED_FADE_IN_MS + COPIED_HOLD_MS + COPIED_FADE_OUT_MS,
      ),
    ];
  };
  const handleKeyDown = (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }

    event.preventDefault();
    handleCopy();
  };
  const closing = phase === "closing";

  return createPortal(
    <div
      ref={popupRef}
      id={id}
      aria-label={phase === "idle" ? "Copy prompt" : "Copied prompt"}
      role="button"
      tabIndex={0}
      className={cn(
        "fixed z-50 max-h-[calc(100vh-1.5rem)] max-w-sm cursor-copy overflow-y-auto whitespace-pre-wrap rounded-lg bg-zinc-900 px-3 py-2 text-body-sm leading-relaxed text-zinc-50 shadow-2xl outline-none transition-opacity dark:bg-zinc-800",
        position ? "visible" : "invisible",
        closing ? "opacity-0 duration-[750ms]" : "opacity-100 duration-300",
      )}
      onClick={handleCopy}
      onKeyDown={handleKeyDown}
      style={{
        left: `${position?.x ?? 0}px`,
        top: `${position?.y ?? 0}px`,
      }}
    >
      <div
        className={cn(
          "transition-opacity duration-300",
          phase === "idle" ? "opacity-100" : "opacity-0",
        )}
      >
        {prompt}
      </div>
      <div
        aria-live="polite"
        className={cn(
          "absolute inset-0 flex items-center justify-center gap-3 text-title-lg font-semibold transition-opacity",
          phase === "copied"
            ? "opacity-100 duration-300"
            : closing
              ? "opacity-0 duration-[750ms]"
              : "opacity-0 duration-300",
        )}
      >
        <HugeiconsIcon
          icon={CheckmarkCircle02Icon}
          strokeWidth={2}
          className="size-8 shrink-0"
        />
        <span>Copied</span>
      </div>
    </div>,
    document.body,
  );
}

export function promptPopupPositionForImage(
  anchorRect,
  popupRect,
  { margin = VIEWPORT_MARGIN, viewportHeight, viewportWidth },
) {
  if (
    !anchorRect ||
    !popupRect ||
    anchorRect.width <= 0 ||
    anchorRect.height <= 0 ||
    popupRect.width <= 0 ||
    popupRect.height <= 0
  ) {
    return null;
  }

  const centerX = anchorRect.left + anchorRect.width / 2;
  const centerY = anchorRect.top + anchorRect.height / 2;

  return {
    x: clamp(
      centerX - popupRect.width / 2,
      margin,
      viewportWidth - popupRect.width - margin,
    ),
    y: clamp(
      centerY - popupRect.height / 2,
      margin,
      viewportHeight - popupRect.height - margin,
    ),
  };
}

function clearTimers(timersRef) {
  for (const timer of timersRef.current) {
    window.clearTimeout(timer);
  }

  timersRef.current = [];
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function clamp(value, min, max) {
  if (max < min) {
    return min;
  }

  return Math.min(Math.max(value, min), max);
}

function samePosition(currentPosition, nextPosition) {
  return (
    currentPosition?.x === nextPosition?.x &&
    currentPosition?.y === nextPosition?.y
  );
}
