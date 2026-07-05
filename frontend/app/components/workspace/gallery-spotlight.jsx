import { useEffect, useRef } from "react";
import { formatMeta, modelVariantLabel } from "@/lib/workspace";

export function GallerySpotlight({
  className = "",
  fallbackPrecision,
  image,
  imageUrl,
  onDismiss,
  onViewportWidthChange,
  zoomScale,
}) {
  const viewportRef = useRef(null);
  const frameRef = useRef(null);
  const dragStateRef = useRef(null);
  const draggedRef = useRef(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const viewport = viewportRef.current;
    if (!viewport) {
      return undefined;
    }

    const updateWidth = () => {
      const frame = frameRef.current ?? viewport;
      const style = window.getComputedStyle(frame);
      const horizontalPadding =
        Number.parseFloat(style.paddingLeft || "0") +
        Number.parseFloat(style.paddingRight || "0");
      onViewportWidthChange?.(
        Math.max(0, viewport.clientWidth - horizontalPadding),
      );
    };

    updateWidth();

    if (!window.ResizeObserver) {
      window.addEventListener("resize", updateWidth);
      return () => window.removeEventListener("resize", updateWidth);
    }

    const observer = new window.ResizeObserver(updateWidth);
    observer.observe(viewport);

    return () => observer.disconnect();
  }, [image?.id, onViewportWidthChange]);

  if (!image) {
    return null;
  }

  const handleBackdropClick = (event) => {
    if (event.target === event.currentTarget) {
      onDismiss?.();
    }
  };

  const handlePointerDown = (event) => {
    if (event.button !== 0 || !viewportRef.current) {
      return;
    }

    dragStateRef.current = {
      captured: false,
      pointerId: event.pointerId,
      scrollLeft: viewportRef.current.scrollLeft,
      scrollTop: viewportRef.current.scrollTop,
      x: event.clientX,
      y: event.clientY,
    };
    draggedRef.current = false;
  };

  const handlePointerMove = (event) => {
    const dragState = dragStateRef.current;
    const viewport = viewportRef.current;
    if (!dragState || !viewport || dragState.pointerId !== event.pointerId) {
      return;
    }

    const deltaX = event.clientX - dragState.x;
    const deltaY = event.clientY - dragState.y;
    if (!draggedRef.current && Math.abs(deltaX) <= 3 && Math.abs(deltaY) <= 3) {
      return;
    }

    if (!draggedRef.current) {
      draggedRef.current = true;
      dragState.captured = true;
      event.currentTarget.setPointerCapture?.(event.pointerId);
    }

    viewport.scrollLeft = dragState.scrollLeft - deltaX;
    viewport.scrollTop = dragState.scrollTop - deltaY;
    event.preventDefault();
  };

  const handlePointerEnd = (event) => {
    const dragState = dragStateRef.current;
    if (dragState?.pointerId === event.pointerId) {
      dragStateRef.current = null;
      if (
        dragState.captured &&
        event.currentTarget.hasPointerCapture?.(event.pointerId)
      ) {
        event.currentTarget.releasePointerCapture?.(event.pointerId);
      }
    }
  };

  const handleClickCapture = (event) => {
    if (!draggedRef.current) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    draggedRef.current = false;
  };

  const handleImageClick = (event) => {
    event.stopPropagation();
  };

  const baseWidth = positiveNumber(image.width);
  const displayWidth = baseWidth ? baseWidth * zoomScale : undefined;
  const meta = formatMeta({
    variant: modelVariantLabel(image, fallbackPrecision),
    width: image.width ?? "-",
    height: image.height ?? "-",
    loras: image.loras,
    seed: image.seed ?? "random",
    steps: image.steps ?? "-",
  });

  return (
    <div
      ref={viewportRef}
      className={`relative hidden min-h-0 cursor-grab overflow-auto overscroll-contain active:cursor-grabbing lg:block ${className}`}
      onClickCapture={handleClickCapture}
      onClick={handleBackdropClick}
      onPointerCancel={handlePointerEnd}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerEnd}
    >
      <figure
        ref={frameRef}
        className="grid min-h-full w-max min-w-full place-items-center p-1 pr-6"
        onClick={handleBackdropClick}
      >
        <div
          className="shrink-0"
          style={{ width: displayWidth ? `${displayWidth}px` : undefined }}
        >
          <a
            href={imageUrl(image.url)}
            target="_blank"
            rel="noopener noreferrer"
            draggable={false}
            onClick={handleImageClick}
            className="block w-full"
          >
            <img
              src={imageUrl(image.url)}
              alt={image.prompt || image.filename || "Generated image"}
              width={image.width || undefined}
              height={image.height || undefined}
              draggable={false}
              className="h-auto w-full rounded-lg object-contain shadow-2xl shadow-black/25"
            />
          </a>
          <figcaption className="sr-only">{meta}</figcaption>
        </div>
      </figure>
    </div>
  );
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}
