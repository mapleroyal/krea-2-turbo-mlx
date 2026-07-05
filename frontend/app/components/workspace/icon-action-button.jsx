import { HugeiconsIcon } from "@hugeicons/react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export function IconActionButton({
  ariaLabel,
  buttonRef,
  className,
  disabled = false,
  expanded = undefined,
  icon,
  iconClassName,
  iconStrokeWidth = 2,
  onClick,
  tone = "default",
  tooltip,
}) {
  const iconToneClass =
    tone === "destructive"
      ? "group-hover/icon-action:text-destructive group-focus-visible/icon-action:text-destructive group-data-[state=delayed-open]/icon-action:text-destructive group-data-[state=instant-open]/icon-action:text-destructive dark:group-hover/icon-action:text-destructive dark:group-focus-visible/icon-action:text-destructive dark:group-data-[state=delayed-open]/icon-action:text-destructive dark:group-data-[state=instant-open]/icon-action:text-destructive"
      : "group-hover/icon-action:text-foreground group-focus-visible/icon-action:text-foreground group-data-[state=delayed-open]/icon-action:text-foreground group-data-[state=instant-open]/icon-action:text-foreground";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          ref={buttonRef}
          aria-label={ariaLabel}
          aria-expanded={expanded}
          className={cn(
            "group/icon-action inline-flex size-6 items-center justify-center rounded-sm outline-none transition-opacity duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-45",
            className,
          )}
          disabled={disabled}
          type="button"
          onClick={onClick}
        >
          <HugeiconsIcon
            icon={icon}
            strokeWidth={iconStrokeWidth}
            className={cn(
              "size-6 text-muted-foreground transition-colors duration-200 group-disabled/icon-action:text-muted-foreground",
              iconToneClass,
              iconClassName,
            )}
          />
        </button>
      </TooltipTrigger>
      <TooltipContent sideOffset={6}>{tooltip}</TooltipContent>
    </Tooltip>
  );
}
