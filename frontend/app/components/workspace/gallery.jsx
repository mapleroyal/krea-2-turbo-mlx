import { useEffect, useRef } from "react";
import FolderOpenIcon from "@hugeicons/core-free-icons/FolderOpenIcon";
import GridViewIcon from "@hugeicons/core-free-icons/GridViewIcon";
import ScanImageIcon from "@hugeicons/core-free-icons/ScanImageIcon";

import {
  VerticalColumn03Icon,
  VerticalColumn03NotFoundIcon,
} from "@/lib/icons";
import { cn } from "@/lib/utils";
import { formatMeta, modelVariantLabel } from "@/lib/workspace";

import { IconActionButton } from "./icon-action-button";
import { ImageMetadataActions } from "./image-metadata-actions";

export function Gallery({
  className,
  expanded,
  fallbackPrecision,
  galleryVisible = true,
  imageUrl,
  items,
  onDelete,
  onExpandedChange,
  onGalleryVisibleToggle,
  onLoadSettings,
  onOpenOutputDir,
  onSelect,
  onSpotlightToggle,
  selectedId,
  spotlightActive = false,
  spotlightAvailable = false,
  spotlightUnseenCount = 0,
}) {
  const selectedCardRef = useRef(null);

  const handleExpandedChange = () => {
    onExpandedChange(!expanded);
  };

  useEffect(() => {
    selectedCardRef.current?.scrollIntoView({
      block: "nearest",
      inline: "nearest",
    });
  }, [selectedId, spotlightActive]);

  return (
    <section
      className={cn(
        "flex min-h-[360px] min-w-0 flex-col overflow-hidden bg-background px-1 pt-1 lg:min-h-0 lg:will-change-transform",
        className,
      )}
    >
      <div className="flex min-h-0 w-full flex-1 flex-col">
        <div className="mb-3 flex shrink-0 items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <h2 className="text-title-sm">Gallery</h2>
            {spotlightUnseenCount > 0 && (
              <span
                aria-label={`${spotlightUnseenCount} unseen generated images`}
                className="inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-full bg-primary px-1.5 text-label-sm font-semibold text-primary-foreground"
              >
                {spotlightUnseenCount}
              </span>
            )}
          </div>
          <div className="flex min-w-0 items-center justify-end gap-2">
            <IconActionButton
              ariaLabel={
                spotlightActive
                  ? "Exit spotlight gallery"
                  : "Enter spotlight gallery"
              }
              className="max-lg:hidden"
              disabled={!spotlightAvailable}
              expanded={spotlightActive}
              icon={ScanImageIcon}
              onClick={onSpotlightToggle}
              tooltip={
                spotlightActive
                  ? "Exit spotlight gallery"
                  : "Enter spotlight gallery"
              }
            />
            <IconActionButton
              ariaLabel={expanded ? "Collapse gallery" : "Expand gallery"}
              expanded={expanded}
              icon={expanded ? VerticalColumn03Icon : GridViewIcon}
              onClick={handleExpandedChange}
              tooltip={expanded ? "Collapse gallery" : "Expand gallery"}
            />
            <IconActionButton
              ariaLabel="Hide gallery"
              expanded={galleryVisible}
              icon={VerticalColumn03NotFoundIcon}
              onClick={onGalleryVisibleToggle}
              tooltip="Hide gallery"
            />
            <IconActionButton
              ariaLabel="Open output folder"
              icon={FolderOpenIcon}
              onClick={onOpenOutputDir}
              tooltip="Open output folder"
            />
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-1">
          {items.length ? (
            <div className="grid grid-cols-1 gap-2.5 lg:justify-center lg:grid-cols-[repeat(auto-fit,minmax(min(100%,var(--gallery-card-width)),var(--gallery-card-width)))]">
              {items.map((item) => (
                <GalleryCard
                  fallbackPrecision={fallbackPrecision}
                  imageUrl={imageUrl}
                  item={item}
                  key={item.id}
                  onDelete={onDelete}
                  onLoadSettings={onLoadSettings}
                  onSelect={onSelect}
                  cardRef={selectedId === item.id ? selectedCardRef : null}
                  selected={selectedId === item.id}
                  spotlightMetaVisible={
                    spotlightActive && selectedId === item.id
                  }
                />
              ))}
            </div>
          ) : (
            <div className="flex min-h-full items-center justify-center rounded-lg border border-dashed border-border p-8 text-center text-body-sm text-muted-foreground">
              No images yet — generate one to get started.
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function GalleryCard({
  cardRef,
  fallbackPrecision,
  imageUrl,
  item,
  onDelete,
  onLoadSettings,
  onSelect,
  selected,
  spotlightMetaVisible = false,
}) {
  const imageFrameRef = useRef(null);
  const meta = formatMeta({
    variant: modelVariantLabel(item, fallbackPrecision),
    width: item.width,
    height: item.height,
    loras: item.loras,
    seed: item.seed,
    steps: item.steps,
  });

  const handleClick = (event) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return;
    }

    event.preventDefault();
    onSelect(item);
  };

  const handleDelete = (event) => {
    event.preventDefault();
    event.stopPropagation();
    onDelete(item);
  };

  const handleLoadSettings = (event) => {
    event.preventDefault();
    event.stopPropagation();
    onLoadSettings(item);
  };

  return (
    <div
      ref={cardRef}
      className={cn(
        "group relative aspect-square overflow-hidden rounded-lg border bg-card text-left transition-colors",
        selected ? "border-0 ring-[3px] ring-primary/50" : "border-border",
      )}
    >
      <a
        href={imageUrl(item.url)}
        target="_blank"
        rel="noopener noreferrer"
        onClick={handleClick}
        className="block size-full"
      >
        <div ref={imageFrameRef} className="size-full bg-background">
          <img
            src={imageUrl(item.url)}
            alt={item.prompt || item.filename || "Generated image"}
            loading="lazy"
            className="size-full object-contain"
          />
        </div>
      </a>
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 bottom-0 flex max-h-[78%] items-end gap-2 overflow-hidden bg-card/95 p-2 text-body-sm opacity-0 shadow-[0_-10px_20px_-16px_rgba(0,0,0,0.6)] [transform:translateY(0.5rem)] backdrop-blur-sm transition-[opacity,transform] duration-200 group-hover:opacity-100 group-hover:[transform:translateY(0)] group-focus-within:opacity-100 group-focus-within:[transform:translateY(0)]",
          spotlightMetaVisible && "opacity-100 [transform:translateY(0)]",
        )}
      >
        <p className="min-w-0 flex-1 whitespace-pre-line text-muted-foreground">
          {meta}
        </p>
        <ImageMetadataActions
          anchorRef={imageFrameRef}
          className="pointer-events-auto shrink-0"
          onDelete={handleDelete}
          onLoadSettings={onLoadSettings ? handleLoadSettings : undefined}
          prompt={item.prompt}
        />
      </div>
    </div>
  );
}
