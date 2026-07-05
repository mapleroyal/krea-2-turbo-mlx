import { useState } from "react";
import ArrowExpand01Icon from "@hugeicons/core-free-icons/ArrowExpand01Icon";
import ArrowShrink01Icon from "@hugeicons/core-free-icons/ArrowShrink01Icon";
import ArrowUp02Icon from "@hugeicons/core-free-icons/ArrowUp02Icon";
import Cancel01Icon from "@hugeicons/core-free-icons/Cancel01Icon";
import TextClearIcon from "@hugeicons/core-free-icons/TextClearIcon";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  InputGroup,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

import { ATTACHED_CONTROL_SHADOW } from "./attached-control-stack";
import { IconActionButton } from "./icon-action-button";

const PROMPT_PLACEHOLDER =
  "An abandoned, Victorian-style glass conservatory being reclaimed by sentient...";

export function PromptForm({
  canSubmit,
  cancelRequested,
  generationRunning,
  onCancel,
  onClearPrompt,
  onPromptChange,
  onSubmit,
  prompt,
}) {
  const [promptCollapsed, setPromptCollapsed] = useState(false);

  return (
    <form className="relative z-10" onSubmit={onSubmit}>
      <InputGroup
        className={cn(
          "min-h-[112px] items-stretch border-border bg-card",
          ATTACHED_CONTROL_SHADOW,
        )}
      >
        <InputGroupTextarea
          aria-label="Prompt"
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          placeholder={PROMPT_PLACEHOLDER}
          className={cn(
            "min-h-[110px] overflow-y-auto px-3 pt-3 pr-24 pb-12 text-body-md",
            promptCollapsed ? "max-h-40" : "max-h-[50dvh]",
          )}
        />
        <div className="absolute top-3 right-3 flex items-center">
          <IconActionButton
            ariaLabel={promptCollapsed ? "Expand prompt" : "Collapse prompt"}
            expanded={!promptCollapsed}
            icon={promptCollapsed ? ArrowExpand01Icon : ArrowShrink01Icon}
            onClick={() => setPromptCollapsed((current) => !current)}
            tooltip={promptCollapsed ? "Expand prompt" : "Collapse prompt"}
          />
        </div>
        <div className="absolute right-3 bottom-3 flex items-center gap-2">
          {prompt ? (
            <IconActionButton
              ariaLabel="Clear prompt"
              icon={TextClearIcon}
              onClick={onClearPrompt}
              tooltip="Clear prompt"
            />
          ) : null}
          {cancelRequested ? (
            <InputGroupButton
              type="button"
              size="icon-sm"
              variant="destructive"
              disabled
              aria-label="Cancelling generation"
              className="size-9 shadow-sm"
            >
              <Spinner />
            </InputGroupButton>
          ) : generationRunning ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <InputGroupButton
                  type="button"
                  size="icon-sm"
                  variant="destructive"
                  aria-label="Cancel generation"
                  className="size-9 shadow-sm"
                  onClick={onCancel}
                >
                  <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} />
                </InputGroupButton>
              </TooltipTrigger>
              <TooltipContent>Cancel generation</TooltipContent>
            </Tooltip>
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <InputGroupButton
                  type="submit"
                  size="icon-sm"
                  variant="default"
                  disabled={!canSubmit}
                  aria-label="Generate"
                  className="size-9 shadow-sm"
                >
                  <HugeiconsIcon icon={ArrowUp02Icon} strokeWidth={2.5} />
                </InputGroupButton>
              </TooltipTrigger>
              <TooltipContent>Generate</TooltipContent>
            </Tooltip>
          )}
        </div>
      </InputGroup>
    </form>
  );
}
