import { useEffect, useMemo, useRef, useState } from "react";
import AlertCircleIcon from "@hugeicons/core-free-icons/AlertCircleIcon";
import ClipboardPasteIcon from "@hugeicons/core-free-icons/ClipboardPasteIcon";
import Copy01Icon from "@hugeicons/core-free-icons/Copy01Icon";
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
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { FieldLabel } from "@/components/ui/field";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { EXAMPLE_BATCH_PROMPT } from "@/lib/workspace";
import { cn } from "@/lib/utils";

import { fileDragStateFromTransfer } from "./dropzone-utils";

const VALIDATION_IDLE = { state: "idle", message: "", count: 0 };
const MAX_BATCH_JSON_BYTES = 256_000;
const JSON_FILE_EXTENSION = ".json";
const JSON_MIME_TYPES = new Set(["application/json", "text/json"]);

export function BatchDialog({
  defaultOutputDir,
  disabled,
  generateBatch,
  onOpenChange,
  open,
  readBatchClipboard,
  selectOutputDir,
  validateBatch,
}) {
  const fileInputRef = useRef(null);
  const dragDepthRef = useRef(0);
  const [batchText, setBatchText] = useState("");
  const [sourceLabel, setSourceLabel] = useState("");
  const [inputError, setInputError] = useState("");
  const [outputDir, setOutputDir] = useState(defaultOutputDir ?? "");
  const [validation, setValidation] = useState(VALIDATION_IDLE);
  const [copied, setCopied] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [selectingOutputDir, setSelectingOutputDir] = useState(false);
  const [dragState, setDragState] = useState("idle");
  const validateBatchRef = useRef(validateBatch);

  useEffect(() => {
    validateBatchRef.current = validateBatch;
  }, [validateBatch]);

  useEffect(() => {
    if (!open) {
      setBatchText("");
      setSourceLabel("");
      setInputError("");
      setOutputDir(defaultOutputDir ?? "");
      setValidation(VALIDATION_IDLE);
      setCopied(false);
      setGenerating(false);
      setSelectingOutputDir(false);
      setDragState("idle");
      dragDepthRef.current = 0;
    }
  }, [defaultOutputDir, open]);

  useEffect(() => {
    if (open) {
      setOutputDir(defaultOutputDir ?? "");
    }
  }, [defaultOutputDir, open]);

  useEffect(() => {
    if (!open || !batchText.trim()) {
      setValidation(VALIDATION_IDLE);
      return undefined;
    }

    let cancelled = false;
    setValidation({ state: "validating", message: "Validating...", count: 0 });

    const timer = setTimeout(async () => {
      try {
        const result = await validateBatchRef.current(batchText);

        if (!cancelled) {
          setValidation({
            state: "valid",
            message: "",
            count: result.count,
          });
        }
      } catch (error) {
        if (!cancelled) {
          setValidation({
            state: "invalid",
            message: error instanceof Error ? error.message : String(error),
            count: 0,
          });
        }
      }
    }, 350);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [batchText, open]);

  const effectiveValidation = inputError
    ? { state: "invalid", message: inputError, count: 0 }
    : validation;
  const resolvedOutputDir = outputDir || defaultOutputDir || "outputs";
  const displayOutputDir = formatDirectoryPath(resolvedOutputDir);
  const outputDirWidth = `${Math.max(
    8,
    Math.min(displayOutputDir.length + 1, 42),
  )}ch`;
  const jobCountLabel =
    effectiveValidation.state === "valid"
      ? effectiveValidation.count === 1
        ? "1 job"
        : `${effectiveValidation.count} jobs`
      : "";
  const jsonPreview = useMemo(() => {
    if (effectiveValidation.state !== "valid") {
      return "";
    }

    try {
      return JSON.stringify(JSON.parse(batchText), null, 2);
    } catch {
      return batchText;
    }
  }, [batchText, effectiveValidation.state]);

  const loadFile = async (file) => {
    if (!file) {
      return;
    }

    try {
      validateJsonFile(file);
      const text = await file.text();
      acceptBatchText(text, file.name || "File");
    } catch (error) {
      rejectBatchInput(error);
    }
  };

  const acceptBatchText = (text, source) => {
    validateTextCandidate(text);
    setBatchText(text);
    setSourceLabel(source);
    setInputError("");
    setCopied(false);
  };

  const rejectBatchInput = (error) => {
    setBatchText("");
    setSourceLabel("");
    setInputError(error instanceof Error ? error.message : String(error));
  };

  const handlePaste = async () => {
    let browserError;

    try {
      const text = await readClipboardBatchText();
      acceptBatchText(text, "Clipboard");
      return;
    } catch (error) {
      browserError = error;
    }

    try {
      const result = await readBatchClipboard();
      acceptBatchText(result.text, result.source || "Clipboard");
    } catch (error) {
      rejectBatchInput(isBatchInputError(browserError) ? browserError : error);
    }
  };

  const handleCopyPrompt = async () => {
    await navigator.clipboard.writeText(EXAMPLE_BATCH_PROMPT);
    setCopied(true);
  };

  const handleGenerate = async () => {
    setGenerating(true);

    try {
      await generateBatch(batchText, { outputDir: resolvedOutputDir });
      onOpenChange(false);
    } catch (error) {
      setValidation({
        state: "invalid",
        message: error instanceof Error ? error.message : String(error),
        count: 0,
      });
    } finally {
      setGenerating(false);
    }
  };

  const handleSelectOutputDir = async () => {
    setSelectingOutputDir(true);

    try {
      const result = await selectOutputDir();
      if (result?.path) {
        setOutputDir(result.path);
      }
    } catch (error) {
      setInputError(error instanceof Error ? error.message : String(error));
    } finally {
      setSelectingOutputDir(false);
    }
  };

  const applyDragFeedback = (dataTransfer) => {
    const nextDragState = fileDragStateFromTransfer(
      dataTransfer,
      isJsonMimeType,
    );
    setDragState(nextDragState);
    dataTransfer.dropEffect = nextDragState === "accept" ? "copy" : "none";
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

    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length !== 1) {
      rejectBatchInput(new Error("Drop one JSON file."));
      return;
    }
    loadFile(files[0]);
  };

  const generateDisabled =
    disabled || generating || effectiveValidation.state !== "valid";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[min(92dvh,760px)] max-w-[min(calc(100vw-2rem),780px)] flex-col gap-4 overflow-hidden rounded-lg border-border bg-card p-4 text-card-foreground sm:max-w-[780px]">
        <DialogHeader>
          <DialogTitle className="text-title-lg">
            Run batch from JSON
          </DialogTitle>
          <DialogDescription className="sr-only">
            Choose, paste, or drop a batch JSON array.
          </DialogDescription>
        </DialogHeader>

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
            <Button type="button" variant="outline" onClick={handlePaste}>
              <HugeiconsIcon icon={ClipboardPasteIcon} strokeWidth={2} />
              Clipboard
            </Button>
          </div>
          <p className="text-body-md text-muted-foreground">
            Or drop a JSON file here.
          </p>
          <p className="flex items-center justify-center gap-1.5 text-body-sm text-muted-foreground">
            <HugeiconsIcon
              icon={InformationCircleIcon}
              strokeWidth={2}
              className="size-3.5 shrink-0"
            />
            <span>Omitted fields use the current settings.</span>
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            className="sr-only"
            onChange={(event) => {
              loadFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </div>

        {effectiveValidation.state === "valid" && (
          <div className="min-h-0 overflow-hidden rounded-lg border border-border bg-background">
            <div className="flex items-center justify-between gap-3 border-b border-border px-3 py-2 text-body-sm text-muted-foreground">
              <span className="flex min-w-0 items-center gap-2">
                <HugeiconsIcon
                  icon={FileValidationIcon}
                  strokeWidth={2}
                  className="size-4 shrink-0"
                />
                <span className="truncate">{sourceLabel || "JSON"}</span>
              </span>
              <span className="shrink-0">{jobCountLabel}</span>
            </div>
            <pre className="max-h-[min(30dvh,18rem)] overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-body-sm text-foreground">
              {jsonPreview}
            </pre>
          </div>
        )}

        <BatchDetails validation={effectiveValidation} />

        <div className="flex flex-wrap items-center gap-1 text-body-sm text-muted-foreground">
          <span>Copy</span>
          <Button
            type="button"
            variant="link"
            className="h-auto px-0 text-body-sm"
            onClick={handleCopyPrompt}
          >
            this example prompt
          </Button>
          <span>to have your agent generate a batch file.</span>
          {copied && (
            <span className="inline-flex items-center gap-1 text-primary">
              <HugeiconsIcon
                icon={Copy01Icon}
                strokeWidth={2}
                className="size-3"
              />
              Copied.
            </span>
          )}
        </div>

        <DialogFooter className="mt-1 flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex min-w-0 flex-col gap-2">
            <FieldLabel
              htmlFor="batch-output-directory"
              className="text-label-md"
            >
              Output location
            </FieldLabel>
            <InputGroup className="h-8 w-fit max-w-full">
              <InputGroupAddon
                align="inline-start"
                className="py-1 pr-1 pl-2 has-[>button]:ml-0"
              >
                <InputGroupButton
                  type="button"
                  variant="outline"
                  size="xs"
                  className="px-3 text-foreground"
                  aria-label="Select output directory"
                  onClick={handleSelectOutputDir}
                  disabled={disabled || selectingOutputDir || generating}
                >
                  {selectingOutputDir ? <Spinner /> : null}
                  Browse...
                </InputGroupButton>
              </InputGroupAddon>
              <Tooltip>
                <TooltipTrigger asChild>
                  <InputGroupInput
                    id="batch-output-directory"
                    aria-label="Output directory"
                    className="h-8 flex-none cursor-default truncate px-2 text-left text-body-sm text-muted-foreground [direction:rtl]"
                    readOnly
                    value={displayOutputDir}
                    style={{ width: outputDirWidth, maxWidth: "19rem" }}
                  />
                </TooltipTrigger>
                <TooltipContent
                  sideOffset={6}
                  className="max-w-[min(calc(100vw-2rem),34rem)] break-all text-left"
                >
                  <span dir="ltr">{displayOutputDir}</span>
                </TooltipContent>
              </Tooltip>
            </InputGroup>
          </div>
          <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
            <DialogClose asChild>
              <Button type="button" variant="outline">
                Cancel
              </Button>
            </DialogClose>
            <Button
              type="button"
              disabled={generateDisabled}
              onClick={handleGenerate}
            >
              {generating ? <Spinner /> : null}
              Generate
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function BatchDetails({ validation }) {
  if (validation.state === "idle") {
    return null;
  }

  if (validation.state === "valid") {
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
    <DetailRow icon={null} className="text-muted-foreground">
      <Spinner />
      <span>{validation.message}</span>
    </DetailRow>
  );
}

function DetailRow({ children, className, icon }) {
  return (
    <div className={cn("flex items-start gap-2 text-body-sm", className)}>
      {icon ? (
        <HugeiconsIcon
          icon={icon}
          strokeWidth={2}
          className="size-4 shrink-0"
        />
      ) : null}
      <span className="min-w-0">{children}</span>
    </div>
  );
}

function validateJsonFile(file) {
  if (!isJsonFile(file)) {
    throw new Error("Choose a .json file.");
  }
  if (file.size > MAX_BATCH_JSON_BYTES) {
    throw new Error(
      `Batch JSON must be ${formatBytes(MAX_BATCH_JSON_BYTES)} or smaller.`,
    );
  }
}

function isJsonFile(file) {
  const type = String(file.type ?? "").toLowerCase();
  const name = String(file.name ?? "").toLowerCase();

  return (
    name.endsWith(JSON_FILE_EXTENSION) ||
    JSON_MIME_TYPES.has(type) ||
    type.endsWith("+json")
  );
}

function validateTextCandidate(text) {
  if (byteLength(text) > MAX_BATCH_JSON_BYTES) {
    throw new Error(
      `Batch JSON must be ${formatBytes(MAX_BATCH_JSON_BYTES)} or smaller.`,
    );
  }

  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("Batch JSON is empty.");
  }
  if (!trimmed.startsWith("[")) {
    throw new Error("Batch JSON must be a JSON array (starting with [).");
  }
}

async function readClipboardBatchText() {
  const clipboard = globalThis.navigator?.clipboard;

  if (!clipboard) {
    throw new Error("Clipboard access is not available.");
  }

  if (typeof clipboard.read === "function") {
    try {
      const itemText = await readClipboardItems(await clipboard.read());
      if (itemText !== null) {
        return itemText;
      }
    } catch (error) {
      if (isBatchInputError(error)) {
        throw error;
      }
      if (typeof clipboard.readText !== "function") {
        throw new Error("Clipboard access was denied.", { cause: error });
      }
    }
  }

  if (typeof clipboard.readText !== "function") {
    throw new Error("Clipboard text is not available.");
  }

  return clipboard.readText();
}

async function readClipboardItems(items) {
  let plainTextBlob = null;

  for (const item of items) {
    const jsonType = item.types.find((type) => isJsonMimeType(type));
    if (jsonType) {
      return textFromBlob(await item.getType(jsonType));
    }

    const textType = item.types.find((type) => type === "text/plain");
    if (textType && plainTextBlob === null) {
      plainTextBlob = await item.getType(textType);
    }
  }

  if (plainTextBlob !== null) {
    return textFromBlob(plainTextBlob);
  }

  return null;
}

async function textFromBlob(blob) {
  if (blob.size > MAX_BATCH_JSON_BYTES) {
    throw new Error(
      `Batch JSON must be ${formatBytes(MAX_BATCH_JSON_BYTES)} or smaller.`,
    );
  }

  return blob.text();
}

function isJsonMimeType(type) {
  const normalized = String(type ?? "").toLowerCase();
  return JSON_MIME_TYPES.has(normalized) || normalized.endsWith("+json");
}

function byteLength(text) {
  return new globalThis.TextEncoder().encode(text).length;
}

function formatBytes(bytes) {
  return `${Math.round(bytes / 1024)} KB`;
}

function formatDirectoryPath(path) {
  const text = String(path ?? "").trim() || "outputs";
  return /[\\/]$/.test(text) ? text : `${text}/`;
}

function isBatchInputError(error) {
  return (
    error instanceof Error &&
    (error.message.startsWith("Batch JSON") ||
      error.message.startsWith("Choose a .json") ||
      error.message.startsWith("Drop one JSON"))
  );
}
