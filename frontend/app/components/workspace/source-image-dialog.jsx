import { useEffect, useRef, useState } from "react";
import AlertCircleIcon from "@hugeicons/core-free-icons/AlertCircleIcon";
import ClipboardPasteIcon from "@hugeicons/core-free-icons/ClipboardPasteIcon";
import File01Icon from "@hugeicons/core-free-icons/File01Icon";
import FileValidationIcon from "@hugeicons/core-free-icons/FileValidationIcon";
import InformationCircleIcon from "@hugeicons/core-free-icons/InformationCircleIcon";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";

import { fileDragStateFromTransfer } from "./dropzone-utils";

const VALIDATION_IDLE = { state: "idle", message: "" };

export function SourceImageDialog({
  disabled,
  initialImage,
  onApply,
  onOpenChange,
  open,
  readSourceImageClipboard,
  validateSourceImageFile,
  validateSourceImageId,
  validateSourceImagePath,
}) {
  const fileInputRef = useRef(null);
  const dragDepthRef = useRef(0);
  const validationRequestRef = useRef(0);
  const [metadata, setMetadata] = useState(null);
  const [validation, setValidation] = useState(VALIDATION_IDLE);
  const [dragState, setDragState] = useState("idle");

  useEffect(() => {
    if (!open) {
      validationRequestRef.current += 1;
      setMetadata(null);
      setValidation(VALIDATION_IDLE);
      setDragState("idle");
      dragDepthRef.current = 0;
    }
  }, [open]);

  useEffect(() => {
    if (!open || !initialImage?.id) {
      return;
    }

    validateCandidate(
      () => validateSourceImageId(initialImage.id),
      initialImage.filename || "Gallery image",
    );
  }, [initialImage?.filename, initialImage?.id, open, validateSourceImageId]);

  const validateCandidate = async (loader, label = "Image") => {
    const requestId = validationRequestRef.current + 1;
    validationRequestRef.current = requestId;
    setMetadata(null);
    setValidation({ state: "validating", message: `Validating ${label}...` });

    try {
      const result = await loader();
      if (validationRequestRef.current !== requestId) {
        return;
      }
      setMetadata(result);
      setValidation({ state: "valid", message: "" });
    } catch (error) {
      if (validationRequestRef.current !== requestId) {
        return;
      }
      setValidation({
        state: "invalid",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const loadFile = (file) => {
    if (!file) {
      return;
    }
    validateCandidate(
      () => validateSourceImageFile(file),
      file.name || "Image",
    );
  };

  const loadPath = (path) => {
    const trimmed = String(path ?? "").trim();
    if (!trimmed) {
      setMetadata(null);
      setValidation({ state: "invalid", message: "Enter an image path." });
      return;
    }
    validateCandidate(() => validateSourceImagePath(trimmed), "Path");
  };

  const handlePasteButton = async () => {
    let browserError;

    try {
      const result = await readBrowserClipboardSource();
      if (result?.file) {
        loadFile(result.file);
        return;
      }
      if (result?.path) {
        loadPath(result.path);
        return;
      }
    } catch (error) {
      browserError = error;
    }

    validateCandidate(async () => {
      try {
        return await readSourceImageClipboard();
      } catch (error) {
        if (browserError instanceof Error) {
          throw browserError;
        }
        throw error;
      }
    }, "Clipboard");
  };

  const handlePaste = (event) => {
    const file = singleImageFile(event.clipboardData?.files);
    if (file) {
      event.preventDefault();
      loadFile(file);
      return;
    }

    const text = event.clipboardData?.getData("text/plain")?.trim();
    if (text) {
      event.preventDefault();
      loadPath(text);
    }
  };

  const applyDragFeedback = (dataTransfer) => {
    const nextDragState = fileDragStateFromTransfer(
      dataTransfer,
      isImageMimeType,
    );
    setDragState(nextDragState);
    dataTransfer.dropEffect = nextDragState === "reject" ? "none" : "copy";
  };

  const handleDragEnter = (event) => {
    event.preventDefault();
    dragDepthRef.current += 1;
    applyDragFeedback(event.dataTransfer);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    applyDragFeedback(event.dataTransfer);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);

    if (dragDepthRef.current === 0) {
      setDragState("idle");
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    dragDepthRef.current = 0;
    setDragState("idle");

    const file = singleImageFile(event.dataTransfer.files);
    if (file) {
      loadFile(file);
      return;
    }

    const text = event.dataTransfer.getData("text/plain")?.trim();
    if (text) {
      loadPath(text);
      return;
    }

    setMetadata(null);
    setValidation({ state: "invalid", message: "Drop one image file." });
  };

  const handleApply = () => {
    if (!metadata) {
      return;
    }
    onApply(metadata);
    onOpenChange(false);
  };

  const applyDisabled =
    disabled || validation.state !== "valid" || metadata === null;
  const sourceLabel = metadata?.source || initialImage?.filename || "Image";
  const showSourcePicker = !initialImage?.id;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="flex max-h-[min(92dvh,760px)] max-w-[min(calc(100vw-2rem),780px)] flex-col gap-4 overflow-hidden rounded-lg border-border bg-card p-4 text-card-foreground sm:max-w-[780px]"
        onPaste={showSourcePicker ? handlePaste : undefined}
      >
        <DialogHeader>
          <DialogTitle className="text-title-lg">
            Load settings from image
          </DialogTitle>
          <DialogDescription className="sr-only">
            {showSourcePicker
              ? "Choose, paste, or drop a source image."
              : "Review the source image metadata."}
          </DialogDescription>
        </DialogHeader>

        {showSourcePicker && (
          <div
            className={cn(
              "flex min-h-32 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border p-6 text-center transition-[background-color,border-color,box-shadow] duration-150",
              dragState === "accept" && "border-primary bg-primary/5",
              dragState === "reject" && "border-destructive bg-destructive/5",
            )}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <div className="flex flex-wrap items-center justify-center gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
              >
                <HugeiconsIcon icon={File01Icon} strokeWidth={2} />
                File
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={handlePasteButton}
              >
                <HugeiconsIcon icon={ClipboardPasteIcon} strokeWidth={2} />
                Clipboard
              </Button>
            </div>
            <p className="text-body-md text-muted-foreground">
              Or drop an image here.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="sr-only"
              onChange={(event) => {
                loadFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </div>
        )}

        <ValidationDetails validation={validation} />

        {metadata && validation.state === "valid" && (
          <div className="min-h-0 overflow-hidden rounded-lg border border-border bg-background">
            <div className="flex items-center justify-between gap-3 border-b border-border px-3 py-2 text-body-sm text-muted-foreground">
              <span className="flex min-w-0 items-center gap-2">
                <HugeiconsIcon
                  icon={FileValidationIcon}
                  strokeWidth={2}
                  className="size-4 shrink-0"
                />
                <span className="truncate" title={sourceLabel}>
                  {sourceLabel}
                </span>
              </span>
              <span className="shrink-0">
                {metadata.image_size?.width} x {metadata.image_size?.height}
              </span>
            </div>
            <div className="max-h-[min(36dvh,22rem)] overflow-auto p-3">
              <MetadataSection
                entries={metadata.supported}
                title="Supported"
                supported
              />
              <MetadataSection entries={metadata.other} title="Other" />
            </div>
          </div>
        )}

        <DialogFooter className="mt-1 flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <DialogClose asChild>
            <Button type="button" variant="outline">
              Cancel
            </Button>
          </DialogClose>
          <Button type="button" disabled={applyDisabled} onClick={handleApply}>
            Load
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function MetadataSection({ entries = [], supported = false, title }) {
  if (!entries.length) {
    return null;
  }

  return (
    <section className="mb-4 last:mb-0">
      <div className="mb-2 flex items-center gap-2">
        <h3 className="text-label-md text-muted-foreground">{title}</h3>
      </div>
      <div className="grid gap-2">
        {entries.map((entry) => (
          <div
            key={entry.key}
            className={cn(
              "grid gap-1 rounded-lg border border-border p-2 text-body-sm",
              supported && "border-primary/40 bg-primary/5",
            )}
          >
            <span className="text-label-md text-muted-foreground">
              {entry.label || entry.key}
            </span>
            <MetadataValue value={entry.value} />
          </div>
        ))}
      </div>
    </section>
  );
}

function MetadataValue({ value }) {
  if (Array.isArray(value) && value.length === 0) {
    return <span className="text-foreground">None</span>;
  }
  if (value && typeof value === "object") {
    return (
      <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-body-sm text-foreground">
        {JSON.stringify(value, null, 2)}
      </pre>
    );
  }

  return (
    <span className="whitespace-pre-wrap break-words text-foreground">
      {String(value ?? "")}
    </span>
  );
}

function ValidationDetails({ validation }) {
  if (validation.state === "idle" || validation.state === "valid") {
    return null;
  }

  if (validation.state === "invalid") {
    return (
      <DetailRow icon={AlertCircleIcon} className="text-destructive">
        {validation.message}
      </DetailRow>
    );
  }

  return (
    <DetailRow icon={InformationCircleIcon} className="text-muted-foreground">
      <Spinner />
      <span>{validation.message}</span>
    </DetailRow>
  );
}

function DetailRow({ children, className, icon }) {
  return (
    <div className={cn("flex items-start gap-2 text-body-sm", className)}>
      <HugeiconsIcon icon={icon} strokeWidth={2} className="size-4 shrink-0" />
      <span className="min-w-0">{children}</span>
    </div>
  );
}

async function readBrowserClipboardSource() {
  const clipboard = globalThis.navigator?.clipboard;
  if (!clipboard) {
    throw new Error("Clipboard access is not available.");
  }

  if (typeof clipboard.read === "function") {
    const items = await clipboard.read();
    for (const item of items) {
      const imageType = item.types.find((type) => isImageMimeType(type));
      if (imageType) {
        return { file: await item.getType(imageType) };
      }
    }

    for (const item of items) {
      const textType = item.types.find((type) => type === "text/plain");
      if (textType) {
        return { path: await (await item.getType(textType)).text() };
      }
    }
  }

  if (typeof clipboard.readText === "function") {
    return { path: await clipboard.readText() };
  }

  throw new Error("Clipboard access is not available.");
}

function singleImageFile(files) {
  const candidates = Array.from(files ?? []);
  if (candidates.length !== 1) {
    return null;
  }
  return candidates[0];
}

function isImageMimeType(type) {
  return String(type ?? "")
    .toLowerCase()
    .startsWith("image/");
}
