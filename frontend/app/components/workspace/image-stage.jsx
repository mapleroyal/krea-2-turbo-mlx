import { useEffect, useRef, useState } from "react";
import AiImageIcon from "@hugeicons/core-free-icons/AiImageIcon";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { formatMeta, modelVariantLabel } from "@/lib/workspace";

import { ImageMetadataActions } from "./image-metadata-actions";

const PREVIEW_PLACEHOLDER_SIZE = 512;
const PLACEHOLDER_DOTS = [".", "..", "..."];

export function ImageStage({
  fallbackPrecision,
  generating = false,
  image,
  imageUrl,
  onDelete,
  onLoadSettings,
  pendingImage,
  preview,
}) {
  const imageRef = useRef(null);

  if (preview) {
    const previewMode = preview.mode === "vae" ? "VAE" : preview.mode;
    const meta = imageMeta(pendingImage, fallbackPrecision, {
      height: preview.height ?? "-",
      seed: "-",
      steps: preview.step_count ?? "-",
      width: preview.width ?? "-",
    });

    return (
      <figure className="flex min-h-[360px] flex-col items-center gap-3 py-3">
        <div
          className="w-full max-w-full"
          style={{ width: preview.width ? `${preview.width}px` : undefined }}
        >
          <a
            href={imageUrl(preview.url)}
            target="_blank"
            rel="noopener noreferrer"
            className="block w-full"
          >
            <img
              ref={imageRef}
              key={preview.revision}
              src={imageUrl(preview.url)}
              alt="Live preview"
              width={preview.width || undefined}
              height={preview.height || undefined}
              className="h-auto w-full rounded-lg object-contain"
            />
          </a>
          <figcaption className="relative mt-3 w-full pr-8 text-left text-body-sm text-muted-foreground">
            <span className="block text-primary motion-safe:animate-pulse">
              {`Live ${previewMode} preview · Step ${preview.step}/${preview.step_count}`}
            </span>
            <span className="block whitespace-pre-line">{meta}</span>
            <ImageMetadataActions
              anchorRef={imageRef}
              className="absolute top-0 right-0"
              prompt={pendingImage?.prompt}
            />
          </figcaption>
        </div>
      </figure>
    );
  }

  if (generating) {
    return (
      <GeneratingPlaceholder
        fallbackPrecision={fallbackPrecision}
        image={pendingImage}
      />
    );
  }

  if (!image) {
    return (
      <Empty className="min-h-[360px] rounded-none border-0 p-10">
        <EmptyHeader>
          <EmptyMedia size="lg">
            <HugeiconsIcon icon={AiImageIcon} strokeWidth={1.5} />
          </EmptyMedia>
          <EmptyTitle className="text-title-lg text-muted-foreground">
            Generate an image or select one from the gallery.
          </EmptyTitle>
        </EmptyHeader>
      </Empty>
    );
  }

  const meta = imageMeta(image, fallbackPrecision);
  const handleDelete = () => {
    onDelete?.(image);
  };
  const handleLoadSettings = () => {
    onLoadSettings?.(image);
  };

  return (
    <figure className="flex min-h-[360px] flex-col items-center gap-3 py-3">
      <div
        className="w-full max-w-full"
        style={{ width: image.width ? `${image.width}px` : undefined }}
      >
        <a
          href={imageUrl(image.url)}
          target="_blank"
          rel="noopener noreferrer"
          className="block w-full"
        >
          <img
            ref={imageRef}
            src={imageUrl(image.url)}
            alt={image.prompt || image.filename || "Generated image"}
            width={image.width || undefined}
            height={image.height || undefined}
            className="h-auto w-full rounded-lg object-contain"
          />
        </a>
        <figcaption className="relative mt-3 w-full whitespace-pre-line pr-24 text-left text-body-sm text-muted-foreground">
          <span>{meta}</span>
          <ImageMetadataActions
            anchorRef={imageRef}
            className="absolute top-0 right-0"
            onDelete={onDelete ? handleDelete : undefined}
            onLoadSettings={onLoadSettings ? handleLoadSettings : undefined}
            prompt={image.prompt}
          />
        </figcaption>
      </div>
    </figure>
  );
}

function GeneratingPlaceholder({ fallbackPrecision, image }) {
  const placeholderRef = useRef(null);
  const [dotIndex, setDotIndex] = useState(0);
  const meta = image ? imageMeta(image, fallbackPrecision) : "";

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setDotIndex((current) => (current + 1) % PLACEHOLDER_DOTS.length);
    }, 450);

    return () => window.clearInterval(timer);
  }, []);

  return (
    <figure className="flex min-h-[360px] flex-col items-center gap-3 py-3">
      <div
        className="w-full max-w-full"
        style={{ width: `${PREVIEW_PLACEHOLDER_SIZE}px` }}
      >
        <div
          ref={placeholderRef}
          className="flex aspect-square w-full flex-col items-center justify-center gap-6 rounded-lg border border-border bg-muted/20 text-muted-foreground"
          role="status"
          aria-live="polite"
          aria-label="Preparing preview"
        >
          <HugeiconsIcon
            icon={AiImageIcon}
            strokeWidth={1.35}
            className="size-64 opacity-50"
          />
          <p
            className="relative inline-block text-display-sm text-muted-foreground/75"
            aria-hidden="true"
          >
            <span>Preparing preview</span>
            <span className="absolute top-0 left-full inline-block w-[3ch] text-left">
              {PLACEHOLDER_DOTS[dotIndex]}
            </span>
          </p>
        </div>
        {image && (
          <figcaption className="relative mt-3 w-full whitespace-pre-line pr-8 text-left text-body-sm text-muted-foreground">
            <span>{meta}</span>
            <ImageMetadataActions
              anchorRef={placeholderRef}
              className="absolute top-0 right-0"
              prompt={image.prompt}
            />
          </figcaption>
        )}
      </div>
    </figure>
  );
}

function imageMeta(image, fallbackPrecision, defaults = {}) {
  return formatMeta({
    variant: modelVariantLabel(image, fallbackPrecision),
    width: image?.width ?? defaults.width ?? "-",
    height: image?.height ?? defaults.height ?? "-",
    loras: image?.loras ?? defaults.loras,
    seed: image?.seed ?? defaults.seed ?? "random",
    steps: image?.steps ?? defaults.steps ?? "-",
  });
}
