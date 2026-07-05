import CircleLock01Icon from "@hugeicons/core-free-icons/CircleLock01Icon";
import CircleUnlock01Icon from "@hugeicons/core-free-icons/CircleUnlock01Icon";
import Infinity01Icon from "@hugeicons/core-free-icons/Infinity01Icon";
import PreferenceVerticalIcon from "@hugeicons/core-free-icons/PreferenceVerticalIcon";
import Queue01Icon from "@hugeicons/core-free-icons/Queue01Icon";
import RefreshIcon from "@hugeicons/core-free-icons/RefreshIcon";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Input } from "@/components/ui/input";
import {
  NativeSelect,
  NativeSelectOption,
} from "@/components/ui/native-select";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  CUSTOM_DIMENSION_PRESET,
  DIMENSION_PRESETS,
  SIMPLE_BATCH_MIN_COUNT,
  formatLoraScale,
  loraDisplayName,
  loraScaleLimitsForItem,
  presetForSize,
} from "@/lib/workspace";

import {
  AttachedControlStack,
  ATTACHED_CONTROL_SHADOW,
} from "./attached-control-stack";

export function ControlRail({
  busy,
  constraints,
  generatedSeed,
  height,
  loraCatalog,
  loraItems,
  loraWarning,
  onBatchOpen,
  onDimensionPresetChange,
  onDimensionsBlur,
  onLoraEnabledChange,
  onLoraRefresh,
  onLoraScaleBlur,
  onLoraScaleChange,
  onLoadParamsOpen,
  onRandomSeed,
  onRandomizationLockChange,
  onSimpleBatchCountBlur,
  onSimpleBatchCountChange,
  onSimpleBatchEnabledChange,
  onStepsBlur,
  randomizationLocked,
  seed,
  selectedLoras,
  setHeight,
  setSeed,
  setSteps,
  setWidth,
  simpleBatchCount,
  simpleBatchEnabled,
  simpleBatchMaxCount,
  simpleBatchWarning,
  steps,
  width,
}) {
  return (
    <aside className="flex flex-col gap-5">
      <FieldGroup className="gap-5">
        <DimensionsField
          constraints={constraints}
          height={height}
          onBlur={onDimensionsBlur}
          onPresetChange={onDimensionPresetChange}
          setHeight={setHeight}
          setWidth={setWidth}
          width={width}
        />
        <SeedField
          generatedSeed={generatedSeed}
          maxSeed={constraints.max_seed}
          onRandomSeed={onRandomSeed}
          onRandomizationLockChange={onRandomizationLockChange}
          randomizationLocked={randomizationLocked}
          seed={seed}
          setSeed={setSeed}
        />
        <StepsField
          defaultSteps={constraints.default_steps}
          onBlur={onStepsBlur}
          setSteps={setSteps}
          steps={steps}
        />
        <LorasField
          catalog={loraCatalog}
          constraints={constraints}
          items={loraItems}
          onEnabledChange={onLoraEnabledChange}
          onRefresh={onLoraRefresh}
          onScaleBlur={onLoraScaleBlur}
          onScaleChange={onLoraScaleChange}
          selected={selectedLoras}
          warning={loraWarning}
        />
        <LoadParamsField disabled={busy} onOpen={onLoadParamsOpen} />
      </FieldGroup>

      <BatchField
        count={simpleBatchCount}
        disabled={busy}
        enabled={simpleBatchEnabled}
        maxCount={simpleBatchMaxCount}
        onAdvancedOpen={onBatchOpen}
        onCountBlur={onSimpleBatchCountBlur}
        onCountChange={onSimpleBatchCountChange}
        onEnabledChange={onSimpleBatchEnabledChange}
        warning={simpleBatchWarning}
      />
    </aside>
  );
}

function DimensionsField({
  constraints,
  height,
  onBlur,
  onPresetChange,
  setHeight,
  setWidth,
  width,
}) {
  const selectedPreset = presetForSize(width, height);

  const handlePresetChange = (event) => {
    const nextValue = event.target.value;

    if (nextValue === CUSTOM_DIMENSION_PRESET) {
      return;
    }

    const preset = DIMENSION_PRESETS.find((item) => item.value === nextValue);

    if (preset) {
      const next = {
        width: String(preset.width),
        height: String(preset.height),
      };
      setWidth(next.width);
      setHeight(next.height);
      onPresetChange(next);
    }
  };

  return (
    <Field>
      <FieldLabel htmlFor="dimension-preset" className="text-label-md">
        Dimensions
      </FieldLabel>
      <NativeSelect
        id="dimension-preset"
        className="w-full"
        value={selectedPreset}
        onChange={handlePresetChange}
      >
        {DIMENSION_PRESETS.map((preset) => (
          <NativeSelectOption key={preset.value} value={preset.value}>
            {preset.label}
          </NativeSelectOption>
        ))}
        <NativeSelectOption value={CUSTOM_DIMENSION_PRESET}>
          Custom
        </NativeSelectOption>
      </NativeSelect>
      <div className="grid grid-cols-2 gap-2">
        <NumberInput
          id="width"
          label="Width"
          max={constraints.max_size}
          min={constraints.alignment}
          onBlur={() => onBlur()}
          onChange={setWidth}
          step={constraints.alignment}
          value={width}
        />
        <NumberInput
          id="height"
          label="Height"
          max={constraints.max_size}
          min={constraints.alignment}
          onBlur={() => onBlur()}
          onChange={setHeight}
          step={constraints.alignment}
          value={height}
        />
      </div>
      <FieldDescription className="text-body-sm">
        Multiples of {constraints.alignment}. Max {constraints.max_size}.
      </FieldDescription>
    </Field>
  );
}

function NumberInput({ id, label, max, min, onBlur, onChange, step, value }) {
  return (
    <div className="space-y-1.5">
      <label className="block text-label-md text-muted-foreground" htmlFor={id}>
        {label}
      </label>
      <Input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onBlur={onBlur}
        onChange={(event) => onChange(event.target.value)}
        className="text-body-md"
      />
    </div>
  );
}

function SeedField({
  generatedSeed,
  maxSeed,
  onRandomSeed,
  onRandomizationLockChange,
  randomizationLocked,
  seed,
  setSeed,
}) {
  return (
    <Field>
      <FieldLabel htmlFor="seed" className="text-label-md">
        Seed
      </FieldLabel>
      <AttachedControlStack className="relative">
        <InputGroup
          className={cn(
            "relative z-10 border-border bg-card",
            ATTACHED_CONTROL_SHADOW,
          )}
        >
          <InputGroupInput
            id="seed"
            inputMode="numeric"
            value={randomizationLocked ? "" : seed}
            placeholder={randomizationLocked ? "" : generatedSeed}
            readOnly={randomizationLocked}
            onChange={(event) => setSeed(event.target.value)}
            className="text-body-md"
          />
          {randomizationLocked && (
            <HugeiconsIcon
              icon={Infinity01Icon}
              strokeWidth={2}
              aria-hidden="true"
              className="pointer-events-none absolute top-1/2 left-1/2 size-9 -translate-x-1/2 -translate-y-1/2 text-muted-foreground"
            />
          )}
          {!randomizationLocked && (
            <InputGroupAddon align="inline-end">
              <Tooltip>
                <TooltipTrigger asChild>
                  <InputGroupButton
                    type="button"
                    size="icon-xs"
                    aria-label="Randomize seed"
                    onClick={onRandomSeed}
                  >
                    <HugeiconsIcon icon={RefreshIcon} strokeWidth={2} />
                  </InputGroupButton>
                </TooltipTrigger>
                <TooltipContent>Randomize seed</TooltipContent>
              </Tooltip>
            </InputGroupAddon>
          )}
        </InputGroup>
        <div className="[--seed-drawer-attachment-depth:var(--radius-xl)] -mt-[var(--seed-drawer-attachment-depth)]">
          <section className="relative overflow-hidden rounded-b-lg border border-t-0 border-border bg-input/40 pt-[var(--seed-drawer-attachment-depth)]">
            <button
              type="button"
              aria-pressed={randomizationLocked}
              className={cn(
                "relative flex h-9 w-full items-center gap-2 bg-transparent px-3 text-left text-label-md text-muted-foreground outline-none transition-colors duration-200 hover:text-primary focus-visible:text-primary focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring/50",
                randomizationLocked && "text-primary",
              )}
              onClick={() => onRandomizationLockChange(!randomizationLocked)}
            >
              <HugeiconsIcon
                icon={
                  randomizationLocked ? CircleLock01Icon : CircleUnlock01Icon
                }
                strokeWidth={2}
                className="size-4 transition-colors duration-200"
              />
              <span>Randomization lock</span>
            </button>
          </section>
        </div>
      </AttachedControlStack>
      <FieldDescription className="text-body-sm">
        {randomizationLocked
          ? "Every single-image generation receives a fresh seed."
          : `Integer from 0 to ${maxSeed}. Empty uses the shown generated seed.`}
      </FieldDescription>
    </Field>
  );
}

function StepsField({ defaultSteps, onBlur, setSteps, steps }) {
  return (
    <Field>
      <FieldLabel htmlFor="steps" className="text-label-md">
        Steps
      </FieldLabel>
      <Input
        id="steps"
        type="number"
        min={1}
        step={1}
        value={steps}
        onBlur={() => onBlur()}
        onChange={(event) => setSteps(event.target.value)}
        className="text-body-md"
      />
      <FieldDescription className="text-body-sm">
        Positive integer. Turbo default is {defaultSteps}.
      </FieldDescription>
    </Field>
  );
}

function LorasField({
  catalog,
  constraints,
  items,
  onEnabledChange,
  onRefresh,
  onScaleBlur,
  onScaleChange,
  selected,
  warning,
}) {
  const selectedById = new Map(
    (Array.isArray(selected) ? selected : []).map((item) => [item.id, item]),
  );
  const warnings = [
    ...(Array.isArray(catalog?.warnings) ? catalog.warnings : []),
    ...(Array.isArray(items)
      ? items.flatMap((item) =>
          Array.isArray(item.warnings) ? item.warnings : [],
        )
      : []),
  ];
  return (
    <Field>
      <div className="flex items-center justify-between gap-2">
        <FieldLabel className="text-label-md">LoRAs</FieldLabel>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label="Refresh LoRAs"
              onClick={() => onRefresh()}
            >
              <HugeiconsIcon icon={RefreshIcon} strokeWidth={2} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Refresh LoRAs</TooltipContent>
        </Tooltip>
      </div>
      <div className="flex flex-col gap-2">
        {(Array.isArray(items) ? items : []).map((item) => {
          const selectedItem = selectedById.get(item.id);
          const enabled = Boolean(selectedItem);
          const limits = loraScaleLimitsForItem(item, constraints);
          const scale =
            selectedItem?.scale ?? formatLoraScale(item.default_scale);
          const inputId = `lora-${cssSafeId(item.id)}-scale`;
          const checkboxId = `lora-${cssSafeId(item.id)}`;
          return (
            <div
              key={item.id}
              className="flex min-w-0 items-center gap-2 rounded-lg border border-border bg-card px-3 py-2"
            >
              <Checkbox
                id={checkboxId}
                checked={enabled}
                onCheckedChange={(checked) => onEnabledChange(item, checked)}
              />
              <Tooltip>
                <TooltipTrigger asChild>
                  <label
                    htmlFor={checkboxId}
                    className="min-w-0 flex-1 cursor-pointer truncate text-label-md text-foreground outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                  >
                    {loraDisplayName(item)}
                  </label>
                </TooltipTrigger>
                <TooltipContent>{item.id}</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="shrink-0">
                    <Input
                      id={inputId}
                      type="number"
                      min={limits.min}
                      max={limits.max}
                      step={0.1}
                      disabled={!enabled}
                      value={String(scale)}
                      onBlur={() => onScaleBlur(item.id)}
                      onChange={(event) =>
                        onScaleChange(item.id, event.target.value)
                      }
                      aria-label={`${loraDisplayName(item)} scale`}
                      className="h-7 w-20 px-2 text-center text-body-md"
                    />
                  </span>
                </TooltipTrigger>
                <TooltipContent>Scale</TooltipContent>
              </Tooltip>
            </div>
          );
        })}
      </div>
      {warning && (
        <FieldDescription className="text-body-sm text-destructive">
          {warning}
        </FieldDescription>
      )}
      {!warning && warnings.length > 0 && (
        <FieldDescription className="text-body-sm text-muted-foreground">
          {warnings.length} warning{warnings.length === 1 ? "" : "s"}
        </FieldDescription>
      )}
    </Field>
  );
}

function cssSafeId(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function LoadParamsField({ disabled, onOpen }) {
  return (
    <Field>
      <FieldLabel className="text-label-md">Load params</FieldLabel>
      <Button
        type="button"
        variant="outline"
        disabled={disabled}
        className="w-full justify-center text-label-md"
        onClick={onOpen}
      >
        <HugeiconsIcon
          icon={PreferenceVerticalIcon}
          data-icon="inline-start"
          strokeWidth={2}
        />
        Choose source
      </Button>
    </Field>
  );
}

function BatchField({
  count,
  disabled,
  enabled,
  maxCount,
  onAdvancedOpen,
  onCountBlur,
  onCountChange,
  onEnabledChange,
  warning,
}) {
  const countDisabled = disabled || !enabled;

  return (
    <Field>
      <FieldLabel htmlFor="simple-batch-enabled" className="text-label-md">
        Batch
      </FieldLabel>
      <div className="flex min-w-0 items-center gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-border bg-card px-3 py-2">
          <Switch
            id="simple-batch-enabled"
            checked={enabled}
            disabled={disabled}
            onCheckedChange={onEnabledChange}
            aria-label="Enable repeated batch"
          />
          <span className="min-w-0 flex-1" aria-hidden="true" />
          <Input
            id="simple-batch-count"
            type="number"
            min={SIMPLE_BATCH_MIN_COUNT}
            max={maxCount}
            step={1}
            disabled={countDisabled}
            value={count}
            onBlur={() => onCountBlur()}
            onChange={(event) => onCountChange(event.target.value)}
            aria-label="Batch count"
            className="h-7 w-14 shrink-0 px-2 text-center text-body-md"
          />
        </div>
        <Button
          type="button"
          variant="outline"
          disabled={disabled}
          className="shrink-0 justify-center px-3 text-label-md"
          onClick={onAdvancedOpen}
        >
          <HugeiconsIcon
            icon={Queue01Icon}
            data-icon="inline-start"
            strokeWidth={2}
          />
          Advanced
        </Button>
      </div>
      {warning && (
        <FieldDescription className="text-body-sm text-destructive">
          {warning}
        </FieldDescription>
      )}
    </Field>
  );
}
