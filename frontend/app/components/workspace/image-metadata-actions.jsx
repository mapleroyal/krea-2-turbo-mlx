import ImageDelete01Icon from "@hugeicons/core-free-icons/ImageDelete01Icon";
import PreferenceVerticalIcon from "@hugeicons/core-free-icons/PreferenceVerticalIcon";

import { cn } from "@/lib/utils";

import { IconActionButton } from "./icon-action-button";
import { PromptInfoButton } from "./prompt-preview";

export function ImageMetadataActions({
  anchorRef,
  className,
  onDelete,
  onLoadSettings,
  prompt,
}) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <PromptInfoButton anchorRef={anchorRef} prompt={prompt} />
      {onLoadSettings && (
        <IconActionButton
          ariaLabel="Load image settings"
          icon={PreferenceVerticalIcon}
          onClick={onLoadSettings}
          tooltip="Load image settings"
        />
      )}
      {onDelete && (
        <IconActionButton
          ariaLabel="Delete image"
          icon={ImageDelete01Icon}
          onClick={onDelete}
          tone="destructive"
          tooltip="Delete image"
        />
      )}
    </div>
  );
}
