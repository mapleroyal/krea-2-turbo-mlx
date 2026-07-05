import SidebarRightIcon from "@hugeicons/core-free-icons/SidebarRightIcon";
import { HugeiconsIcon } from "@hugeicons/react";

import { ThemeModeSwitcher } from "@/components/theme/theme-mode-switcher";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import {
  MAX_PREVIEW_INTERVAL_STEPS,
  MIN_PREVIEW_INTERVAL_STEPS,
} from "@/lib/workspace";

export function SettingsSheet({
  busy,
  livePreviewMode,
  onLivePreviewModeChange,
  onPreviewIntervalBlur,
  onThemeChange,
  previewIntervalSteps,
  previewIntervalWarning,
  setPreviewIntervalSteps,
}) {
  return (
    <Sheet>
      <Tooltip>
        <TooltipTrigger asChild>
          <SheetTrigger asChild>
            <button
              type="button"
              aria-label="Open settings"
              className="group/settings-trigger inline-flex size-6 items-center justify-center rounded-sm outline-none transition-opacity duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/50"
            >
              <HugeiconsIcon
                icon={SidebarRightIcon}
                strokeWidth={2}
                className="size-6 text-muted-foreground transition-colors duration-200 group-hover/settings-trigger:text-foreground group-focus-visible/settings-trigger:text-foreground"
              />
            </button>
          </SheetTrigger>
        </TooltipTrigger>
        <TooltipContent sideOffset={6}>Settings</TooltipContent>
      </Tooltip>
      <SheetContent className="overflow-hidden">
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription className="sr-only">
            Adjust workspace settings.
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-4 pb-4">
          <div className="flex min-h-full flex-col gap-5 py-1">
            <LivePreviewField
              disabled={busy}
              intervalSteps={previewIntervalSteps}
              mode={livePreviewMode}
              onIntervalBlur={onPreviewIntervalBlur}
              onModeChange={onLivePreviewModeChange}
              setIntervalSteps={setPreviewIntervalSteps}
              warning={previewIntervalWarning}
            />
          </div>
        </div>

        <SheetFooter className="gap-2 border-t px-4 py-2.5">
          <ThemeModeSwitcher
            className="ml-auto w-28 shrink-0"
            onThemeChange={onThemeChange}
          />
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function LivePreviewField({
  disabled,
  intervalSteps,
  mode,
  onIntervalBlur,
  onModeChange,
  setIntervalSteps,
  warning,
}) {
  const enabled = mode !== "off";

  return (
    <Field>
      <FieldLabel className="text-label-md">Live preview</FieldLabel>
      <ToggleGroup
        type="single"
        value={mode}
        disabled={disabled}
        onValueChange={(value) => {
          if (value) {
            onModeChange(value);
          }
        }}
        variant="outline"
        className="grid w-full grid-cols-3"
      >
        <ToggleGroupItem value="off" className="w-full text-label-md">
          Off
        </ToggleGroupItem>
        <ToggleGroupItem value="latent" className="w-full text-label-md">
          Latent
        </ToggleGroupItem>
        <ToggleGroupItem value="vae" className="w-full text-label-md">
          VAE
        </ToggleGroupItem>
      </ToggleGroup>
      {enabled && (
        <div className="space-y-1.5">
          <label
            className="block text-label-md text-muted-foreground"
            htmlFor="settings-preview-interval-steps"
          >
            Interval
          </label>
          <Input
            id="settings-preview-interval-steps"
            type="number"
            min={MIN_PREVIEW_INTERVAL_STEPS}
            max={MAX_PREVIEW_INTERVAL_STEPS}
            step={1}
            disabled={disabled}
            value={intervalSteps}
            onBlur={onIntervalBlur}
            onChange={(event) => setIntervalSteps(event.target.value)}
            className="text-body-md"
          />
        </div>
      )}
      <FieldDescription
        className={cn("text-body-sm", warning && "text-destructive")}
      >
        {warning || livePreviewDescription(mode)}
      </FieldDescription>
    </Field>
  );
}

function livePreviewDescription(mode) {
  if (mode === "latent") {
    return "Rough latent RGB preview. Fast.";
  }
  if (mode === "vae") {
    return "Decoded preview. Slower, but can be more accurate.";
  }
  return "No intermediate previews.";
}
