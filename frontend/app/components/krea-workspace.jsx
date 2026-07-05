import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AlertCircleIcon from "@hugeicons/core-free-icons/AlertCircleIcon";
import Download01Icon from "@hugeicons/core-free-icons/Download01Icon";
import ImageActualSizeIcon from "@hugeicons/core-free-icons/ImageActualSizeIcon";
import ZoomInAreaIcon from "@hugeicons/core-free-icons/ZoomInAreaIcon";
import ZoomOutAreaIcon from "@hugeicons/core-free-icons/ZoomOutAreaIcon";
import { HugeiconsIcon } from "@hugeicons/react";

import { TooltipProvider } from "@/components/ui/tooltip";
import { BatchDialog } from "@/components/workspace/batch-dialog";
import { BatchStatus } from "@/components/workspace/batch-status";
import { AttachedControlStack } from "@/components/workspace/attached-control-stack";
import { ControlRail } from "@/components/workspace/control-rail";
import { EventLog } from "@/components/workspace/event-log";
import { Gallery } from "@/components/workspace/gallery";
import { GallerySpotlight } from "@/components/workspace/gallery-spotlight";
import { IconActionButton } from "@/components/workspace/icon-action-button";
import { ImageStage } from "@/components/workspace/image-stage";
import { KeyboardShortcutsSheet } from "@/components/workspace/keyboard-shortcuts-sheet";
import { PromptForm } from "@/components/workspace/prompt-form";
import { SettingsSheet } from "@/components/workspace/settings-sheet";
import { SourceImageDialog } from "@/components/workspace/source-image-dialog";
import { StatusDot } from "@/components/workspace/status-dot";
import {
  EjectIcon,
  VerticalColumn03Icon,
  VerticalColumn03NotFoundIcon,
} from "@/lib/icons";
import {
  BATCH_JOB_STATUSES,
  TASK_PHASES,
  normalizeBatchJobStatus,
  normalizeStatusConstraints,
} from "@/lib/status";
import {
  DOUBLE_ESCAPE_CANCEL_MS,
  generationCancelKeyAction,
  hasPlatformShortcutModifier,
  isPromptSubmitShortcut,
  isShortcutModifierKey,
  SHORTCUT_CHEAT_SHEET_HOLD_MS,
  shortcutModifierLabel as platformShortcutModifierLabel,
} from "@/lib/keyboard-shortcuts";
import { cn } from "@/lib/utils";
import {
  buildSelectedLoras,
  clampGenerationDimension,
  clampGenerationSteps,
  clampLoraScale,
  clampPreviewIntervalSteps,
  clampSimpleBatchCount,
  formatLoraScale,
  formatPreviewIntervalSteps,
  formatSimpleBatchCount,
  findLoraCatalogItem,
  loraCatalogItems,
  loraScaleLimitsForItem,
  MAX_PREVIEW_INTERVAL_STEPS,
  MIN_PREVIEW_INTERVAL_STEPS,
  normalizeLoraSelections,
  normalizeUiSettings,
  persistLivePreviewMode,
  persistPreviewIntervalSteps,
  randomSeed,
  readLivePreviewSettings,
  setLoraSelectionEnabled,
  setLoraSelectionScale,
  SIMPLE_BATCH_MIN_COUNT,
  normalizeLivePreviewMode,
} from "@/lib/workspace";
import { useAppStore } from "@/stores/use-app-store";
import {
  DEFAULT_STATUS,
  buildBatchPayload,
  buildRepeatedBatchPayload,
  parseBatchText,
  useGuiStore,
} from "@/stores/use-gui-store";

const WORKSPACE_COLUMN_PADDING = "px-1 pt-1";
const GALLERY_SPOTLIGHT_DESKTOP_QUERY = "(min-width: 1024px)";
const GALLERY_SPOTLIGHT_ZOOM_STEP = 0.25;
const GALLERY_SPOTLIGHT_RESET_ZOOM = 1;
const GALLERY_SPOTLIGHT_MIN_ZOOM = 0.25;
const GALLERY_SPOTLIGHT_MAX_ZOOM = 4;
const GALLERY_SPOTLIGHT_ZOOM_EPSILON = 0.001;
const BATCH_PROMPT_JSON_ARRAY_START = /^\[\s*(?:\{|\])/;
const BATCH_JOB_PROMPT_KEYS = [
  "prompt",
  "width",
  "height",
  "steps",
  "seed",
  "loras",
];
const BATCH_JOB_GUI_DERIVED_KEYS = [
  "width",
  "height",
  "steps",
  "seed",
  "loras",
];

export function KreaWorkspace() {
  const status = useGuiStore((state) => state.status);
  const lastError = useGuiStore((state) => state.lastError);
  const initializeSession = useGuiStore((state) => state.initializeSession);
  const startPolling = useGuiStore((state) => state.startPolling);
  const stopPolling = useGuiStore((state) => state.stopPolling);
  const loadModel = useGuiStore((state) => state.loadModel);
  const ejectModel = useGuiStore((state) => state.ejectModel);
  const cancelCurrentGeneration = useGuiStore(
    (state) => state.cancelCurrentGeneration,
  );
  const clearBatchQueue = useGuiStore((state) => state.clearBatchQueue);
  const openOutputDir = useGuiStore((state) => state.openOutputDir);
  const readBatchClipboard = useGuiStore((state) => state.readBatchClipboard);
  const readSourceImageClipboard = useGuiStore(
    (state) => state.readSourceImageClipboard,
  );
  const selectOutputDir = useGuiStore((state) => state.selectOutputDir);
  const generateImage = useGuiStore((state) => state.generateImage);
  const validateSourceImageFile = useGuiStore(
    (state) => state.validateSourceImageFile,
  );
  const validateSourceImageId = useGuiStore(
    (state) => state.validateSourceImageId,
  );
  const validateSourceImagePath = useGuiStore(
    (state) => state.validateSourceImagePath,
  );
  const validateBatch = useGuiStore((state) => state.validateBatch);
  const generateBatch = useGuiStore((state) => state.generateBatch);
  const generateRepeatedBatch = useGuiStore(
    (state) => state.generateRepeatedBatch,
  );
  const persistUiSettings = useGuiStore((state) => state.persistUiSettings);
  const refreshLoras = useGuiStore((state) => state.refreshLoras);
  const deleteImage = useGuiStore((state) => state.deleteImage);
  const imageUrl = useGuiStore((state) => state.imageUrl);
  const activeTaskStartedMs = useGuiStore((state) => state.activeTaskStartedMs);
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);

  const constraints = useMemo(
    () => normalizeStatusConstraints(status.constraints),
    [status.constraints],
  );
  const loraCatalog = status.loras ?? DEFAULT_STATUS.loras;
  const catalogItems = useMemo(
    () => loraCatalogItems(loraCatalog),
    [loraCatalog],
  );
  const busy = Boolean(
    status.busy ||
    status.load_running ||
    status.generation_running ||
    activeTaskStartedMs,
  );
  const recent = status.recent ?? [];
  const events = status.events ?? [];
  const fallbackPrecision =
    status.model?.precision ?? DEFAULT_STATUS.model.precision;
  const serverError = status.error || lastError;
  const initialUiSettings = useMemo(
    () =>
      normalizeUiSettings(status.ui_settings, { catalogItems, constraints }),
    [catalogItems, constraints, status.ui_settings],
  );

  const [prompt, setPrompt] = useState("");
  const [promptBatchResubmitOptions, setPromptBatchResubmitOptions] =
    useState(null);
  const [completedBatch, setCompletedBatch] = useState(null);
  const [batchStatusExpanded, setBatchStatusExpanded] = useState(false);
  const [width, setWidth] = useState(() => initialUiSettings.width);
  const [height, setHeight] = useState(() => initialUiSettings.height);
  const [steps, setSteps] = useState(() => initialUiSettings.steps);
  const [seed, setSeed] = useState("");
  const [randomizationLocked, setRandomizationLocked] = useState(
    () => initialUiSettings.randomizationLocked,
  );
  const [generatedSeed, setGeneratedSeed] = useState(() =>
    String(randomSeed(constraints.max_seed)),
  );
  const [formError, setFormError] = useState("");
  const [selectedImageId, setSelectedImageId] = useState(
    status.image?.id ?? null,
  );
  const [galleryExpanded, setGalleryExpanded] = useState(false);
  const [galleryVisible, setGalleryVisible] = useState(true);
  const [gallerySpotlight, setGallerySpotlight] = useState(false);
  const [gallerySpotlightImageId, setGallerySpotlightImageId] = useState(null);
  const [gallerySpotlightZoom, setGallerySpotlightZoom] = useState(
    GALLERY_SPOTLIGHT_RESET_ZOOM,
  );
  const [gallerySpotlightViewportWidth, setGallerySpotlightViewportWidth] =
    useState(0);
  const [gallerySpotlightZoomFeedback, setGallerySpotlightZoomFeedback] =
    useState(() => ({
      label: formatGallerySpotlightZoom(GALLERY_SPOTLIGHT_RESET_ZOOM),
      visible: false,
    }));
  const [gallerySpotlightUnseenIds, setGallerySpotlightUnseenIds] = useState(
    [],
  );
  const gallerySpotlightZoomFeedbackTimerRef = useRef(null);
  const [batchOpen, setBatchOpen] = useState(false);
  const [sourceImageOpen, setSourceImageOpen] = useState(false);
  const [sourceImageInitialImage, setSourceImageInitialImage] = useState(null);
  const [simpleBatchEnabled, setSimpleBatchEnabled] = useState(
    () => initialUiSettings.simpleBatchEnabled,
  );
  const [simpleBatchCount, setSimpleBatchCount] = useState(
    () => initialUiSettings.simpleBatchCount,
  );
  const [simpleBatchWarning, setSimpleBatchWarning] = useState("");
  const [pendingGenerationImage, setPendingGenerationImage] = useState(null);
  const [selectedLoras, setSelectedLoras] = useState(() =>
    status.ui_settings ? initialUiSettings.loras : [],
  );
  const [loraWarning, setLoraWarning] = useState("");
  const [livePreviewMode, setLivePreviewMode] = useState(() =>
    status.ui_settings
      ? initialUiSettings.livePreviewMode
      : readLivePreviewSettings().mode,
  );
  const [previewIntervalSteps, setPreviewIntervalSteps] = useState(() =>
    status.ui_settings
      ? initialUiSettings.previewIntervalSteps
      : readLivePreviewSettings().intervalSteps,
  );
  const [previewIntervalWarning, setPreviewIntervalWarning] = useState("");
  const [uiSettingsHydrated, setUiSettingsHydrated] = useState(false);
  const [shortcutsVisible, setShortcutsVisible] = useState(false);
  const [shortcutModifierKeyLabel, setShortcutModifierKeyLabel] =
    useState("Ctrl");
  const lastEscapeKeyDownMsRef = useRef(null);
  const batchClearQueueArmedUntilMsRef = useRef(null);
  const lastRunningBatchRef = useRef(null);
  const lastSubmittedBatchIntentRef = useRef(null);
  const galleryRecentIdsRef = useRef(new Set(recent.map((item) => item.id)));
  const shortcutSheetHoldTimerRef = useRef(null);
  const shortcutSheetModifierHeldRef = useRef(false);
  const shortcutSheetChordSuppressedRef = useRef(false);

  useEffect(() => {
    initializeSession();
    startPolling();
    return () => stopPolling();
  }, [initializeSession, startPolling, stopPolling]);

  useEffect(() => {
    setShortcutModifierKeyLabel(platformShortcutModifierLabel());
  }, []);

  useEffect(
    () => () => {
      if (gallerySpotlightZoomFeedbackTimerRef.current !== null) {
        clearTimeout(gallerySpotlightZoomFeedbackTimerRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    if (status.batch) {
      lastRunningBatchRef.current = status.batch;
      return;
    }

    const completed = completedBatchFromSnapshot(lastRunningBatchRef.current);
    if (completed) {
      const batchPrompt = batchPromptStateFromCompletedBatch(
        completed,
        lastSubmittedBatchIntentRef.current,
      );
      setCompletedBatch(batchPrompt.completedBatch);
      if (batchPrompt.options) {
        setPrompt(batchPrompt.options.displayText);
        setPromptBatchResubmitOptions(batchPrompt.options);
      }
    }
    lastRunningBatchRef.current = null;
  }, [status.batch]);

  useEffect(() => {
    if (uiSettingsHydrated || !status.ui_settings) {
      return;
    }

    const settings = normalizeUiSettings(status.ui_settings, {
      catalogItems,
      constraints,
    });
    setWidth(settings.width);
    setHeight(settings.height);
    setSteps(settings.steps);
    setRandomizationLocked(settings.randomizationLocked);
    setLivePreviewMode(settings.livePreviewMode);
    setPreviewIntervalSteps(settings.previewIntervalSteps);
    setSelectedLoras(settings.loras);
    setSimpleBatchEnabled(settings.simpleBatchEnabled);
    setSimpleBatchCount(settings.simpleBatchCount);
    persistLivePreviewMode(settings.livePreviewMode);
    persistPreviewIntervalSteps(settings.previewIntervalSteps);
    if (theme !== settings.theme) {
      setTheme(settings.theme);
    }
    setUiSettingsHydrated(true);
  }, [
    catalogItems,
    constraints,
    status.ui_settings,
    setTheme,
    theme,
    uiSettingsHydrated,
  ]);

  useEffect(() => {
    if (!gallerySpotlight && status.image) {
      setSelectedImageId(status.image.id);
    }
  }, [status.image?.id]);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) {
      return undefined;
    }

    const mediaQuery = window.matchMedia(GALLERY_SPOTLIGHT_DESKTOP_QUERY);
    const exitSpotlightBelowDesktop = () => {
      if (!mediaQuery.matches) {
        setGallerySpotlight(false);
      }
    };

    exitSpotlightBelowDesktop();
    mediaQuery.addEventListener("change", exitSpotlightBelowDesktop);

    return () => {
      mediaQuery.removeEventListener("change", exitSpotlightBelowDesktop);
    };
  }, []);

  useEffect(() => {
    if (!status.generation_running && !activeTaskStartedMs) {
      setPendingGenerationImage(null);
    }
  }, [activeTaskStartedMs, status.generation_running]);

  useEffect(() => {
    if (gallerySpotlight) {
      if (!recent.length) {
        setGallerySpotlight(false);
        setGallerySpotlightImageId(null);
        return;
      }

      const spotlightGalleryImageExists =
        gallerySpotlightImageId !== null &&
        recent.some((item) => item.id === gallerySpotlightImageId);

      if (!spotlightGalleryImageExists) {
        setGallerySpotlightImageId(recent[0].id);
      }

      return;
    }

    if (gallerySpotlightImageId !== null) {
      setGallerySpotlightImageId(null);
    }

    if (selectedImageId === null) {
      return;
    }

    const selectedStillExists =
      recent.some((item) => item.id === selectedImageId) ||
      status.image?.id === selectedImageId;

    if (!selectedStillExists) {
      setSelectedImageId(status.image?.id ?? null);
    }
  }, [
    gallerySpotlight,
    gallerySpotlightImageId,
    recent,
    selectedImageId,
    status.image?.id,
  ]);

  const activeImage = useMemo(() => {
    if (selectedImageId !== null) {
      const selected =
        recent.find((item) => item.id === selectedImageId) ??
        (status.image?.id === selectedImageId ? status.image : null);

      if (selected) {
        return selected;
      }
    }

    return status.image ?? recent[0] ?? null;
  }, [recent, selectedImageId, status.image]);

  const gallerySpotlightImage = useMemo(() => {
    if (!gallerySpotlight || !recent.length) {
      return null;
    }

    if (gallerySpotlightImageId !== null) {
      const selected = recent.find(
        (item) => item.id === gallerySpotlightImageId,
      );
      if (selected) {
        return selected;
      }
    }

    return recent[0];
  }, [gallerySpotlight, gallerySpotlightImageId, recent]);
  const gallerySpotlightAvailable = recent.length > 0;
  const gallerySpotlightActive =
    galleryVisible && gallerySpotlight && gallerySpotlightAvailable;

  useEffect(() => {
    const previousRecentIds = galleryRecentIdsRef.current;
    const nextUnseenIds = nextGallerySpotlightUnseenIds({
      currentIds: gallerySpotlightUnseenIds,
      focusedId: gallerySpotlightImage?.id,
      previousRecentIds,
      recent,
      spotlightActive: gallerySpotlightActive,
    });
    galleryRecentIdsRef.current = new Set(recent.map((item) => item.id));
    if (!sameStringArray(nextUnseenIds, gallerySpotlightUnseenIds)) {
      setGallerySpotlightUnseenIds(nextUnseenIds);
    }
  }, [
    gallerySpotlightActive,
    gallerySpotlightImage?.id,
    gallerySpotlightUnseenIds,
    recent,
  ]);

  const gallerySpotlightFitZoom = gallerySpotlightZoomForWidth(
    gallerySpotlightImage,
    gallerySpotlightViewportWidth,
  );
  const gallerySpotlightMaxZoom = gallerySpotlightZoomMax(
    gallerySpotlightFitZoom,
  );
  const gallerySpotlightEffectiveZoom = clampGallerySpotlightZoom(
    gallerySpotlightZoom,
    gallerySpotlightMaxZoom,
    gallerySpotlightFitZoom,
  );
  const gallerySpotlightResetZoom = clampGallerySpotlightZoom(
    GALLERY_SPOTLIGHT_RESET_ZOOM,
    gallerySpotlightMaxZoom,
    gallerySpotlightFitZoom,
  );
  const gallerySpotlightZoomLabel = formatGallerySpotlightZoom(
    gallerySpotlightEffectiveZoom,
  );
  const gallerySpotlightZoomIncreaseDisabled =
    !gallerySpotlightImage ||
    gallerySpotlightEffectiveZoom >=
      gallerySpotlightMaxZoom - GALLERY_SPOTLIGHT_ZOOM_EPSILON;
  const gallerySpotlightZoomDecreaseDisabled =
    !gallerySpotlightImage ||
    gallerySpotlightEffectiveZoom <=
      gallerySpotlightLowerZoomBound(gallerySpotlightMaxZoom) +
        GALLERY_SPOTLIGHT_ZOOM_EPSILON;
  const gallerySpotlightZoomResetDisabled =
    !gallerySpotlightImage ||
    Math.abs(gallerySpotlightEffectiveZoom - gallerySpotlightResetZoom) <=
      GALLERY_SPOTLIGHT_ZOOM_EPSILON;

  const currentError = useMemo(
    () => formError || serverError,
    [formError, serverError],
  );
  const canSubmit = Boolean(prompt.trim()) && !busy;
  const activePreview = status.generation_running ? status.preview : null;
  const batchGenerationImage = imageDetailsFromGeneration(status.batch);
  const generationImage = batchGenerationImage ?? pendingGenerationImage;
  const stageGenerating = Boolean(
    status.generation_running ||
    (activeTaskStartedMs && pendingGenerationImage),
  );
  const galleryOverlayActive = galleryVisible && galleryExpanded;

  const runAction = useCallback(async (action) => {
    setFormError("");

    try {
      await action();
      return true;
    } catch (error) {
      setFormError(error instanceof Error ? error.message : String(error));
      return false;
    }
  }, []);

  const requestCancelCurrentGeneration = useCallback(
    () => runAction(cancelCurrentGeneration),
    [cancelCurrentGeneration, runAction],
  );

  const requestClearBatchQueue = useCallback(
    () => runAction(clearBatchQueue),
    [clearBatchQueue, runAction],
  );

  const saveUiSettingsPatch = useCallback(
    (settings) => {
      persistUiSettings(settings).catch((error) => {
        setFormError(error instanceof Error ? error.message : String(error));
      });
    },
    [persistUiSettings],
  );

  const commitGenerationDimensions = useCallback(
    (next = {}) => {
      const widthValue = clampGenerationDimension(next.width ?? width, {
        alignment: constraints.alignment,
        maxSize: constraints.max_size,
        defaultValue: constraints.default_width,
      }).value;
      const heightValue = clampGenerationDimension(next.height ?? height, {
        alignment: constraints.alignment,
        maxSize: constraints.max_size,
        defaultValue: constraints.default_height,
      }).value;
      setWidth(String(widthValue));
      setHeight(String(heightValue));
      saveUiSettingsPatch({ width: widthValue, height: heightValue });

      return { width: widthValue, height: heightValue };
    },
    [
      constraints.alignment,
      constraints.default_height,
      constraints.default_width,
      constraints.max_size,
      height,
      saveUiSettingsPatch,
      width,
    ],
  );

  const commitGenerationSteps = useCallback(
    (nextSteps = steps) => {
      const stepsValue = clampGenerationSteps(nextSteps, {
        defaultValue: constraints.default_steps,
      }).value;
      setSteps(String(stepsValue));
      saveUiSettingsPatch({ steps: stepsValue });
      return stepsValue;
    },
    [constraints.default_steps, saveUiSettingsPatch, steps],
  );

  const handleDimensionPresetChange = useCallback(
    (next) => {
      commitGenerationDimensions(next);
    },
    [commitGenerationDimensions],
  );

  const normalizeSelectedLoras = useCallback(
    (nextLoras = selectedLoras) =>
      normalizeLoraSelections(nextLoras, {
        catalogItems,
        constraints,
      }),
    [catalogItems, constraints, selectedLoras],
  );

  const commitSelectedLoras = useCallback(
    (nextLoras = selectedLoras) => {
      const normalized = normalizeSelectedLoras(nextLoras);
      setSelectedLoras(normalized);
      saveUiSettingsPatch({ loras: normalized, catalogItems, constraints });
      const warning = loraClampWarning(nextLoras, normalized, catalogItems);
      setLoraWarning(warning);
      return normalized;
    },
    [
      catalogItems,
      constraints,
      normalizeSelectedLoras,
      saveUiSettingsPatch,
      selectedLoras,
    ],
  );

  const currentLorasForSubmit = useCallback(() => {
    const normalized = commitSelectedLoras();
    return buildSelectedLoras(normalized, { catalogItems, constraints });
  }, [catalogItems, commitSelectedLoras, constraints]);

  const currentLorasForValidation = useCallback(
    () => buildSelectedLoras(selectedLoras, { catalogItems, constraints }),
    [catalogItems, constraints, selectedLoras],
  );

  const commitPreviewIntervalSteps = useCallback(() => {
    const clamped = clampPreviewIntervalSteps(previewIntervalSteps);
    const formatted = formatPreviewIntervalSteps(clamped.value);
    setPreviewIntervalSteps(formatted);
    persistPreviewIntervalSteps(clamped.value);
    saveUiSettingsPatch({ previewIntervalSteps: clamped.value });
    if (clamped.changed || clamped.usedFallback) {
      setPreviewIntervalWarning(
        `Interval must be ${MIN_PREVIEW_INTERVAL_STEPS} to ${MAX_PREVIEW_INTERVAL_STEPS}. Clamped to ${formatted}.`,
      );
    } else {
      setPreviewIntervalWarning("");
    }
    return clamped.value;
  }, [previewIntervalSteps, saveUiSettingsPatch]);

  const commitSimpleBatchCount = useCallback(() => {
    const clamped = clampSimpleBatchCount(
      simpleBatchCount,
      constraints.max_batch_jobs,
    );
    const formatted = formatSimpleBatchCount(
      clamped.value,
      constraints.max_batch_jobs,
    );
    setSimpleBatchCount(formatted);
    saveUiSettingsPatch({
      simpleBatchCount: clamped.value,
      simpleBatchMaxCount: constraints.max_batch_jobs,
    });
    if (clamped.changed || clamped.usedFallback) {
      setSimpleBatchWarning(
        `Count must be ${SIMPLE_BATCH_MIN_COUNT} to ${clamped.max}. Clamped to ${formatted}.`,
      );
    } else {
      setSimpleBatchWarning("");
    }
    return clamped.value;
  }, [constraints.max_batch_jobs, saveUiSettingsPatch, simpleBatchCount]);

  const currentPreviewSettingsForSubmit = useCallback(
    () => ({
      livePreviewMode,
      previewIntervalSteps: commitPreviewIntervalSteps(),
    }),
    [commitPreviewIntervalSteps, livePreviewMode],
  );
  const currentPreviewSettingsForValidation = useCallback(
    () => ({
      livePreviewMode,
      previewIntervalSteps:
        clampPreviewIntervalSteps(previewIntervalSteps).value,
    }),
    [livePreviewMode, previewIntervalSteps],
  );

  const handleLoraEnabledChange = useCallback(
    (item, checked) => {
      const next = setLoraSelectionEnabled(
        selectedLoras,
        item,
        checked === true,
      );
      const normalized = normalizeSelectedLoras(next);
      setSelectedLoras(normalized);
      setLoraWarning("");
      saveUiSettingsPatch({ loras: normalized, catalogItems, constraints });
    },
    [
      catalogItems,
      constraints,
      normalizeSelectedLoras,
      saveUiSettingsPatch,
      selectedLoras,
    ],
  );

  const handleLoraScaleChange = useCallback((id, value) => {
    setSelectedLoras((current) => setLoraSelectionScale(current, id, value));
  }, []);

  const handleLoraScaleBlur = useCallback(
    (id) => {
      const item = findLoraCatalogItem(catalogItems, id);
      const current = selectedLoras.find((lora) => lora.id === id);
      if (!item || !current) {
        return;
      }
      const limits = loraScaleLimitsForItem(item, constraints);
      const clamped = clampLoraScale(current.scale, limits);
      const next = setLoraSelectionScale(
        selectedLoras,
        id,
        formatLoraScale(clamped.value),
      );
      commitSelectedLoras(next);
    },
    [catalogItems, commitSelectedLoras, constraints, selectedLoras],
  );

  const handleLoraRefresh = useCallback(
    () => runAction(refreshLoras),
    [refreshLoras, runAction],
  );

  const handleLivePreviewModeChange = useCallback(
    (value) => {
      const mode = normalizeLivePreviewMode(value);
      setLivePreviewMode(mode);
      persistLivePreviewMode(mode);
      saveUiSettingsPatch({ livePreviewMode: mode });
      if (mode === "off") {
        setPreviewIntervalWarning("");
      }
    },
    [saveUiSettingsPatch],
  );

  const handleThemeChange = useCallback(
    (nextTheme) => {
      saveUiSettingsPatch({ theme: nextTheme });
    },
    [saveUiSettingsPatch],
  );

  const handleSimpleBatchEnabledChange = useCallback(
    (checked) => {
      const enabled = checked === true;
      setSimpleBatchEnabled(enabled);
      saveUiSettingsPatch({ simpleBatchEnabled: enabled });
      if (!enabled) {
        setSimpleBatchWarning("");
      }
    },
    [saveUiSettingsPatch],
  );

  const updateSimpleBatchCountBy = useCallback(
    (offset) => {
      const current = clampSimpleBatchCount(
        simpleBatchCount,
        constraints.max_batch_jobs,
      );
      const next = clampSimpleBatchCount(
        current.value + offset,
        constraints.max_batch_jobs,
      );
      const formatted = formatSimpleBatchCount(
        next.value,
        constraints.max_batch_jobs,
      );

      setSimpleBatchEnabled(true);
      setSimpleBatchCount(formatted);
      setSimpleBatchWarning("");
      saveUiSettingsPatch({
        simpleBatchEnabled: true,
        simpleBatchCount: next.value,
        simpleBatchMaxCount: constraints.max_batch_jobs,
      });
    },
    [constraints.max_batch_jobs, saveUiSettingsPatch, simpleBatchCount],
  );

  const handleRandomizationLockChange = useCallback(
    (checked) => {
      const locked = checked === true;
      setRandomizationLocked(locked);
      saveUiSettingsPatch({ randomizationLocked: locked });
    },
    [saveUiSettingsPatch],
  );

  const validateBatchWithDefaults = useCallback(
    (text) =>
      validateBatch(text, {
        constraints,
        jobDefaults: {
          width,
          height,
          steps,
          seed,
          generatedSeed,
          randomizeSeed: randomizationLocked,
          loras: currentLorasForValidation(),
        },
        ...currentPreviewSettingsForValidation(),
      }),
    [
      constraints,
      currentLorasForValidation,
      currentPreviewSettingsForValidation,
      generatedSeed,
      height,
      randomizationLocked,
      seed,
      steps,
      validateBatch,
      width,
    ],
  );

  const generateBatchWithDefaults = useCallback(
    async (text, options) => {
      const batchOptions = {
        ...options,
        constraints,
        jobDefaults: {
          width,
          height,
          steps,
          seed,
          generatedSeed,
          randomizeSeed: randomizationLocked,
          loras: currentLorasForSubmit(),
        },
        ...currentPreviewSettingsForSubmit(),
      };
      const sourceJobs = parseBatchText(text);
      const payload = buildBatchPayload(text, batchOptions);
      const outputDir = payload.output_dir ?? "";
      const batchPrompt = batchPromptStateFromSubmittedJobs({
        outputDir,
        sourceJobs,
        submittedJobs: payload.jobs,
      });
      const firstJob = batchPrompt.displayJobs[0];

      setPendingGenerationImage(imageDetailsFromGeneration(firstJob));
      await generateBatch(text, { ...batchOptions, outputDir });
      lastSubmittedBatchIntentRef.current = batchPrompt.intent;
      setPrompt(batchPrompt.options.displayText);
      setPromptBatchResubmitOptions(batchPrompt.options);
      setCompletedBatch(completedBatchFromJobs(batchPrompt.displayJobs));
    },
    [
      constraints,
      currentLorasForSubmit,
      currentPreviewSettingsForSubmit,
      generateBatch,
      generatedSeed,
      height,
      randomizationLocked,
      seed,
      steps,
      width,
    ],
  );

  const submitPrompt = useCallback(() => {
    if (!canSubmit) {
      return;
    }

    const promptText = prompt.trim();
    if (promptSubmitAction(promptText) === "batch") {
      const resubmitOptions =
        promptBatchResubmitOptions?.displayText.trim() === promptText
          ? promptBatchResubmitOptions
          : null;
      const outputDir = resubmitOptions?.outputDir;
      const batchText = resubmitOptions?.resubmitText ?? promptText;

      runAction(() => generateBatchWithDefaults(batchText, { outputDir }));
      return;
    }

    setCompletedBatch(null);
    const resolvedSeed = seed.trim() || generatedSeed;
    const pendingImageDetails = {
      prompt: promptText,
      width,
      height,
      steps,
      seed: randomizationLocked ? "random" : resolvedSeed,
    };

    if (simpleBatchEnabled) {
      const count = commitSimpleBatchCount();
      const loras = currentLorasForSubmit();
      const previewSettings = currentPreviewSettingsForSubmit();
      const payload = buildRepeatedBatchPayload({
        count,
        prompt: promptText,
        width,
        height,
        steps,
        seed: resolvedSeed,
        generatedSeed,
        randomizeSeed: randomizationLocked,
        loras,
        constraints,
        maxBatchJobs: constraints.max_batch_jobs,
        ...previewSettings,
      });
      const intent = batchResubmitIntentFromJobs({
        sourceJobs: payload.jobs.map((job) => ({ prompt: job.prompt })),
        submittedJobs: payload.jobs,
      });

      setPendingGenerationImage(
        imageDetailsFromGeneration({ ...pendingImageDetails, loras }),
      );
      runAction(async () => {
        await generateRepeatedBatch({
          count,
          prompt: promptText,
          width,
          height,
          steps,
          seed: resolvedSeed,
          generatedSeed,
          randomizeSeed: randomizationLocked,
          loras,
          constraints,
          maxBatchJobs: constraints.max_batch_jobs,
          ...previewSettings,
        });
        lastSubmittedBatchIntentRef.current = {
          ...intent,
          updatePromptOnCompletion: false,
        };
      });
      return;
    }

    lastSubmittedBatchIntentRef.current = null;
    const loras = currentLorasForSubmit();
    const previewSettings = currentPreviewSettingsForSubmit();
    setPendingGenerationImage(
      imageDetailsFromGeneration({ ...pendingImageDetails, loras }),
    );

    runAction(() =>
      generateImage({
        prompt: promptText,
        width,
        height,
        steps,
        seed: resolvedSeed,
        randomizeSeed: randomizationLocked,
        loras,
        ...previewSettings,
      }),
    );
  }, [
    canSubmit,
    commitSimpleBatchCount,
    constraints.max_batch_jobs,
    currentLorasForSubmit,
    currentPreviewSettingsForSubmit,
    generateImage,
    generateBatchWithDefaults,
    generateRepeatedBatch,
    generatedSeed,
    height,
    prompt,
    promptBatchResubmitOptions,
    randomizationLocked,
    runAction,
    seed,
    simpleBatchEnabled,
    steps,
    width,
  ]);

  useEffect(() => {
    const handlePromptSubmitShortcut = (event) => {
      if (
        event.defaultPrevented ||
        !isPromptSubmitShortcut(event) ||
        batchOpen ||
        hasOpenModal()
      ) {
        return;
      }

      if (!canSubmit) {
        return;
      }

      event.preventDefault();
      submitPrompt();
    };

    window.addEventListener("keydown", handlePromptSubmitShortcut);

    return () => {
      window.removeEventListener("keydown", handlePromptSubmitShortcut);
    };
  }, [batchOpen, canSubmit, submitPrompt]);

  useEffect(() => {
    const currentGenerationCanCancel = Boolean(
      status.generation_running && !status.cancel_requested,
    );
    const batchQueueCanClear = batchHasQueuedJobs(status.batch);

    if (!currentGenerationCanCancel && !batchQueueCanClear) {
      lastEscapeKeyDownMsRef.current = null;
      batchClearQueueArmedUntilMsRef.current = null;
      return undefined;
    }

    const handleGenerationCancelShortcut = (event) => {
      if (batchOpen || hasOpenModal()) {
        return;
      }

      const action = generationCancelKeyAction(event, {
        lastEscapeKeyDownMs: lastEscapeKeyDownMsRef.current,
      });
      const clearQueueFollowup = batchClearQueueFollowupKeyAction(event, {
        armedUntilMs: batchClearQueueArmedUntilMsRef.current,
      });

      if (!action) {
        return;
      }

      if (action === "clearEscape") {
        lastEscapeKeyDownMsRef.current = null;
        batchClearQueueArmedUntilMsRef.current = null;
        return;
      }

      const escapeKeyDownMs = event.timeStamp;
      runAfterKeyDownHandlers(() => {
        if (event.defaultPrevented || batchOpen || hasOpenModal()) {
          lastEscapeKeyDownMsRef.current = null;
          batchClearQueueArmedUntilMsRef.current = null;
          return;
        }

        if (clearQueueFollowup && batchQueueCanClear) {
          event.preventDefault();
          lastEscapeKeyDownMsRef.current = null;
          batchClearQueueArmedUntilMsRef.current = null;
          requestClearBatchQueue();
          return;
        }

        if (action === "primeEscape") {
          lastEscapeKeyDownMsRef.current = escapeKeyDownMs;
          return;
        }

        if (!currentGenerationCanCancel) {
          return;
        }

        event.preventDefault();
        lastEscapeKeyDownMsRef.current = null;
        requestCancelCurrentGeneration();
        batchClearQueueArmedUntilMsRef.current =
          status.batch && batchQueueCanClear
            ? escapeKeyDownMs + DOUBLE_ESCAPE_CANCEL_MS
            : null;
      });
    };

    window.addEventListener("keydown", handleGenerationCancelShortcut);

    return () => {
      window.removeEventListener("keydown", handleGenerationCancelShortcut);
    };
  }, [
    batchOpen,
    requestCancelCurrentGeneration,
    requestClearBatchQueue,
    status.cancel_requested,
    status.batch,
    status.generation_running,
  ]);

  useEffect(() => {
    const clearShortcutSheetTimer = () => {
      if (shortcutSheetHoldTimerRef.current !== null) {
        clearTimeout(shortcutSheetHoldTimerRef.current);
        shortcutSheetHoldTimerRef.current = null;
      }
    };

    const cancelShortcutSheet = () => {
      clearShortcutSheetTimer();
      setShortcutsVisible(false);
    };

    const releaseShortcutSheetModifier = () => {
      shortcutSheetModifierHeldRef.current = false;
      shortcutSheetChordSuppressedRef.current = false;
      cancelShortcutSheet();
    };

    const suppressShortcutSheetChord = () => {
      shortcutSheetChordSuppressedRef.current = true;
      cancelShortcutSheet();
    };

    const handleShortcutSheetKeyDown = (event) => {
      if (hasOpenModal()) {
        releaseShortcutSheetModifier();
        return;
      }

      const action = shortcutSheetKeyDownAction(event, {
        modifierHeld: shortcutSheetModifierHeldRef.current,
      });

      if (action === "suppressChord") {
        suppressShortcutSheetChord();
        return;
      }

      if (action !== "pressModifier") {
        return;
      }

      shortcutSheetModifierHeldRef.current = true;
      shortcutSheetChordSuppressedRef.current = false;
      if (shortcutSheetHoldTimerRef.current !== null) {
        return;
      }

      shortcutSheetHoldTimerRef.current = setTimeout(() => {
        shortcutSheetHoldTimerRef.current = null;
        if (
          shortcutSheetModifierHeldRef.current &&
          !shortcutSheetChordSuppressedRef.current &&
          globalThis.document?.hasFocus?.() !== false &&
          !hasOpenModal()
        ) {
          setShortcutsVisible(true);
        }
      }, SHORTCUT_CHEAT_SHEET_HOLD_MS);
    };

    const handleShortcutSheetKeyUp = (event) => {
      if (
        shortcutSheetKeyUpAction(event, {
          modifierHeld: shortcutSheetModifierHeldRef.current,
        }) === "releaseModifier"
      ) {
        releaseShortcutSheetModifier();
      }
    };

    window.addEventListener("keydown", handleShortcutSheetKeyDown);
    window.addEventListener("keyup", handleShortcutSheetKeyUp);
    window.addEventListener("blur", releaseShortcutSheetModifier);
    window.addEventListener("pointerdown", releaseShortcutSheetModifier);
    window.addEventListener("mousedown", releaseShortcutSheetModifier);
    document.addEventListener("visibilitychange", releaseShortcutSheetModifier);

    return () => {
      shortcutSheetModifierHeldRef.current = false;
      shortcutSheetChordSuppressedRef.current = false;
      clearShortcutSheetTimer();
      window.removeEventListener("keydown", handleShortcutSheetKeyDown);
      window.removeEventListener("keyup", handleShortcutSheetKeyUp);
      window.removeEventListener("blur", releaseShortcutSheetModifier);
      window.removeEventListener("pointerdown", releaseShortcutSheetModifier);
      window.removeEventListener("mousedown", releaseShortcutSheetModifier);
      document.removeEventListener(
        "visibilitychange",
        releaseShortcutSheetModifier,
      );
    };
  }, []);

  useEffect(() => {
    if (gallerySpotlightActive) {
      return undefined;
    }

    const handleBatchCountShortcut = (event) => {
      if (busy || batchOpen || hasOpenModal()) {
        return;
      }

      const action = sizeAdjustmentKeyAction(event);
      if (!action) {
        return;
      }

      event.preventDefault();
      updateSimpleBatchCountBy(action === "increase" ? 1 : -1);
    };

    window.addEventListener("keydown", handleBatchCountShortcut);

    return () => {
      window.removeEventListener("keydown", handleBatchCountShortcut);
    };
  }, [batchOpen, busy, gallerySpotlightActive, updateSimpleBatchCountBy]);

  useEffect(() => {
    if (gallerySpotlightActive) {
      return undefined;
    }

    const handleGalleryPreviewKeyDown = (event) => {
      if (batchOpen || hasOpenModal()) {
        return;
      }

      const action = galleryPreviewKeyAction(event);
      if (!action) {
        return;
      }

      if (action === "openSpotlight") {
        if (!isGallerySpotlightDesktopViewport()) {
          return;
        }

        const initialImage = gallerySpotlightInitialImage({
          activeImage,
          batch: status.batch,
          recent,
        });
        if (!initialImage) {
          return;
        }

        event.preventDefault();
        openGallerySpotlightForImage(initialImage);
        return;
      }

      if (stageGenerating) {
        return;
      }

      const nextImage = galleryImageByOffset(
        recent,
        activeImage?.id,
        action === "previous" ? -1 : 1,
      );
      if (!nextImage) {
        return;
      }

      event.preventDefault();
      setSelectedImageId(nextImage.id);
    };

    window.addEventListener("keydown", handleGalleryPreviewKeyDown);

    return () => {
      window.removeEventListener("keydown", handleGalleryPreviewKeyDown);
    };
  }, [
    activeImage,
    batchOpen,
    gallerySpotlightActive,
    recent,
    stageGenerating,
    status.batch,
  ]);

  useEffect(() => {
    if (!gallerySpotlightActive) {
      return undefined;
    }

    const handleGallerySpotlightKeyDown = (event) => {
      if (batchOpen || hasOpenModal() || !isGallerySpotlightDesktopViewport()) {
        return;
      }

      const action = gallerySpotlightKeyAction(event);
      if (!action) {
        return;
      }

      event.preventDefault();

      if (action === "close") {
        setGallerySpotlight(false);
        return;
      }

      if (action === "previous") {
        selectGalleryImageByOffset(-1);
        return;
      }

      if (action === "next") {
        selectGalleryImageByOffset(1);
        return;
      }

      if (action === "zoomIn") {
        increaseGallerySpotlightZoom();
        return;
      }

      if (action === "zoomOut") {
        decreaseGallerySpotlightZoom();
        return;
      }

      if (action === "zoomReset") {
        resetGallerySpotlightZoom();
        return;
      }

      if (action === "openImage") {
        openGallerySpotlightImageInNewTab();
        return;
      }

      if (action === "delete" && gallerySpotlightImage) {
        handleDeleteImage(gallerySpotlightImage);
      }
    };

    window.addEventListener("keydown", handleGallerySpotlightKeyDown);

    return () => {
      window.removeEventListener("keydown", handleGallerySpotlightKeyDown);
    };
  }, [
    batchOpen,
    gallerySpotlightActive,
    gallerySpotlightEffectiveZoom,
    gallerySpotlightImageId,
    gallerySpotlightImage,
    gallerySpotlightMaxZoom,
    recent,
  ]);

  function handleRandomSeed() {
    setGeneratedSeed(String(randomSeed(constraints.max_seed)));
    setSeed("");
  }

  function handleGenerate(event) {
    event.preventDefault();
    submitPrompt();
  }

  function handlePromptChange(nextPrompt) {
    setPrompt(nextPrompt);
    setCompletedBatch(null);
    if (promptBatchResubmitOptions?.displayText !== nextPrompt) {
      lastSubmittedBatchIntentRef.current = null;
    }
    setPromptBatchResubmitOptions((current) =>
      current?.displayText === nextPrompt ? current : null,
    );
  }

  function handleClearPrompt() {
    setPrompt("");
    setPromptBatchResubmitOptions(null);
    lastSubmittedBatchIntentRef.current = null;
    setCompletedBatch(null);
    setFormError("");
  }

  function handleClearCompletedBatch() {
    setCompletedBatch(null);
    setBatchStatusExpanded(false);
    setFormError("");
  }

  function handleSelectImage(image) {
    const action = galleryImageClickAction({
      desktopViewport: isGallerySpotlightDesktopViewport(),
      spotlightActive: gallerySpotlightActive,
      stageGenerating,
    });

    if (action === "focusSpotlight") {
      setGallerySpotlightImageId(image.id);
      return;
    }

    if (action === "openSpotlight") {
      openGallerySpotlightForImage(image);
      return;
    }

    setSelectedImageId(image.id);
  }

  async function handleDeleteImage(image) {
    const deletingSpotlightImage =
      gallerySpotlightActive && gallerySpotlightImage?.id === image.id;
    const nextImageId = nextGalleryImageIdAfterDelete(recent, image.id);
    const deleted = await runAction(() => deleteImage(image.id));
    if (!deleted) {
      return;
    }

    if (selectedImageId === image.id) {
      setSelectedImageId(nextImageId);
    }

    if (deletingSpotlightImage || gallerySpotlightImageId === image.id) {
      setGallerySpotlightImageId(nextImageId);
      if (nextImageId === null) {
        setGallerySpotlight(false);
      }
    }
  }

  function handleLoadImageSettings(image) {
    setSourceImageInitialImage(image);
    setSourceImageOpen(true);
  }

  function handleGalleryVisibleToggle() {
    setGalleryVisible((visible) => {
      if (visible) {
        setGalleryExpanded(false);
        setGallerySpotlight(false);
      }

      return !visible;
    });
  }

  function handleGalleryExpandedChange(expanded) {
    setGalleryExpanded(expanded);
    if (expanded) {
      setGallerySpotlight(false);
    }
  }

  function handleGallerySpotlightToggle() {
    if (gallerySpotlight) {
      setGallerySpotlight(false);
      return;
    }

    openGallerySpotlightFromCurrentContext();
  }

  function openGallerySpotlightFromCurrentContext() {
    const initialImage = gallerySpotlightInitialImage({
      activeImage,
      batch: status.batch,
      recent,
    });
    if (!initialImage) {
      return;
    }

    openGallerySpotlightForImage(initialImage);
  }

  function openGallerySpotlightForImage(image) {
    setGalleryVisible(true);
    setGalleryExpanded(false);
    setGallerySpotlightImageId(image.id);
    setGallerySpotlight(true);
  }

  function selectGalleryImageByOffset(offset) {
    const nextImage = galleryImageByOffset(
      recent,
      gallerySpotlightImageId,
      offset,
    );
    if (nextImage) {
      setGallerySpotlightImageId(nextImage.id);
    }
  }

  function showGallerySpotlightZoomFeedback(zoom) {
    if (gallerySpotlightZoomFeedbackTimerRef.current !== null) {
      clearTimeout(gallerySpotlightZoomFeedbackTimerRef.current);
    }

    setGallerySpotlightZoomFeedback({
      label: formatGallerySpotlightZoom(zoom),
      visible: true,
    });
    gallerySpotlightZoomFeedbackTimerRef.current = setTimeout(() => {
      setGallerySpotlightZoomFeedback((current) => ({
        ...current,
        visible: false,
      }));
      gallerySpotlightZoomFeedbackTimerRef.current = null;
    }, 1300);
  }

  function updateGallerySpotlightZoom(nextZoom) {
    const clampedZoom = clampGallerySpotlightZoom(
      nextZoom,
      gallerySpotlightMaxZoom,
      gallerySpotlightFitZoom,
    );

    setGallerySpotlightZoom(clampedZoom);
    showGallerySpotlightZoomFeedback(clampedZoom);
  }

  function decreaseGallerySpotlightZoom() {
    updateGallerySpotlightZoom(
      nextGallerySpotlightZoom(
        gallerySpotlightEffectiveZoom,
        -1,
        gallerySpotlightMaxZoom,
        gallerySpotlightFitZoom,
      ),
    );
  }

  function increaseGallerySpotlightZoom() {
    updateGallerySpotlightZoom(
      nextGallerySpotlightZoom(
        gallerySpotlightEffectiveZoom,
        1,
        gallerySpotlightMaxZoom,
        gallerySpotlightFitZoom,
      ),
    );
  }

  function resetGallerySpotlightZoom() {
    updateGallerySpotlightZoom(GALLERY_SPOTLIGHT_RESET_ZOOM);
  }

  function openGallerySpotlightImageInNewTab() {
    if (!gallerySpotlightImage) {
      return;
    }

    globalThis.window?.open?.(
      imageUrl(gallerySpotlightImage.url),
      "_blank",
      "noopener,noreferrer",
    );
  }

  function handleSourceImageOpenChange(open) {
    setSourceImageOpen(open);
    if (!open) {
      setSourceImageInitialImage(null);
    }
  }

  function handleApplySourceImageMetadata(result) {
    const settings = result?.settings ?? {};
    setFormError("");

    if (typeof settings.prompt === "string") {
      setPrompt(settings.prompt);
      setCompletedBatch(null);
      setPromptBatchResubmitOptions(null);
    }
    if (settings.width !== undefined || settings.height !== undefined) {
      commitGenerationDimensions({
        width: settings.width ?? width,
        height: settings.height ?? height,
      });
    }
    if (settings.steps !== undefined) {
      commitGenerationSteps(settings.steps);
    }
    if (settings.seed !== undefined) {
      setSeed(String(settings.seed));
      setRandomizationLocked(false);
      saveUiSettingsPatch({ randomizationLocked: false });
    }
    if (Array.isArray(settings.loras)) {
      applyImportedLoras(settings.loras);
    }
  }

  function applyImportedLoras(loras) {
    const normalized = normalizeLoraSelections(loras, {
      catalogItems,
      constraints,
    });
    setSelectedLoras(normalized);
    saveUiSettingsPatch({ loras: normalized, catalogItems, constraints });
    setLoraWarning(loraClampWarning(loras, normalized, catalogItems));
  }

  return (
    <TooltipProvider delayDuration={350}>
      <main className="min-h-dvh bg-background text-foreground lg:h-dvh lg:overflow-hidden">
        <div className="mx-auto flex min-h-dvh w-full max-w-[1680px] flex-col gap-3 px-4 py-4 sm:px-6 lg:h-full lg:min-h-0 lg:pb-0">
          <WorkspaceTopBar
            busy={busy}
            galleryVisible={galleryVisible}
            loadRunning={Boolean(status.load_running)}
            livePreviewMode={livePreviewMode}
            model={status.model}
            modelError={status.error}
            onEjectModel={() => runAction(ejectModel)}
            onGalleryVisibleToggle={handleGalleryVisibleToggle}
            onLoadModel={() => runAction(loadModel)}
            onLivePreviewModeChange={handleLivePreviewModeChange}
            onPreviewIntervalBlur={commitPreviewIntervalSteps}
            onThemeChange={handleThemeChange}
            phase={status.phase}
            previewIntervalSteps={previewIntervalSteps}
            previewIntervalWarning={previewIntervalWarning}
            server={status.server}
            setPreviewIntervalSteps={setPreviewIntervalSteps}
            spotlightControls={
              gallerySpotlightActive ? (
                <GallerySpotlightTopBarControls
                  onZoomDecrease={decreaseGallerySpotlightZoom}
                  onZoomIncrease={increaseGallerySpotlightZoom}
                  onZoomReset={resetGallerySpotlightZoom}
                  zoomDecreaseDisabled={gallerySpotlightZoomDecreaseDisabled}
                  zoomFeedbackLabel={
                    gallerySpotlightZoomFeedback.visible
                      ? gallerySpotlightZoomFeedback.label
                      : gallerySpotlightZoomLabel
                  }
                  zoomFeedbackVisible={gallerySpotlightZoomFeedback.visible}
                  zoomIncreaseDisabled={gallerySpotlightZoomIncreaseDisabled}
                  zoomResetDisabled={gallerySpotlightZoomResetDisabled}
                />
              ) : null
            }
          />

          <div
            className={cn(
              "relative grid min-w-0 flex-1 gap-6 lg:min-h-0 lg:items-stretch lg:overflow-hidden lg:transition-[grid-template-columns] lg:duration-300 lg:ease-in-out lg:[--gallery-card-width:calc(var(--gallery-sidebar-width)-1rem)] lg:[--gallery-sidebar-width:280px] xl:[--gallery-sidebar-width:320px]",
              galleryVisible
                ? "lg:grid-cols-[270px_minmax(0,1fr)_280px] xl:grid-cols-[270px_minmax(0,1fr)_320px]"
                : "lg:grid-cols-[270px_minmax(0,1fr)_0px] xl:grid-cols-[270px_minmax(0,1fr)_0px]",
            )}
          >
            <div
              className={cn(
                "min-h-0 min-w-0 overflow-hidden transition-opacity duration-200 ease-out lg:overflow-y-auto lg:overscroll-contain",
                WORKSPACE_COLUMN_PADDING,
                galleryOverlayActive && "lg:pointer-events-none lg:opacity-0",
                gallerySpotlightActive &&
                  "lg:pointer-events-none lg:opacity-20",
              )}
            >
              <ControlRail
                busy={busy}
                constraints={constraints}
                height={height}
                loraCatalog={loraCatalog}
                loraItems={catalogItems}
                loraWarning={loraWarning}
                onBatchOpen={() => setBatchOpen(true)}
                onDimensionPresetChange={handleDimensionPresetChange}
                onDimensionsBlur={commitGenerationDimensions}
                onLoraEnabledChange={handleLoraEnabledChange}
                onLoraRefresh={handleLoraRefresh}
                onLoraScaleBlur={handleLoraScaleBlur}
                onLoraScaleChange={handleLoraScaleChange}
                onLoadParamsOpen={() => handleLoadImageSettings(null)}
                onRandomSeed={handleRandomSeed}
                onRandomizationLockChange={handleRandomizationLockChange}
                onSimpleBatchCountBlur={commitSimpleBatchCount}
                onSimpleBatchCountChange={setSimpleBatchCount}
                onSimpleBatchEnabledChange={handleSimpleBatchEnabledChange}
                onStepsBlur={commitGenerationSteps}
                generatedSeed={generatedSeed}
                randomizationLocked={randomizationLocked}
                seed={seed}
                setHeight={setHeight}
                selectedLoras={selectedLoras}
                setSeed={setSeed}
                setSteps={setSteps}
                setWidth={setWidth}
                simpleBatchCount={simpleBatchCount}
                simpleBatchEnabled={simpleBatchEnabled}
                simpleBatchMaxCount={constraints.max_batch_jobs}
                simpleBatchWarning={simpleBatchWarning}
                steps={steps}
                width={width}
              />
            </div>

            <section
              className={cn(
                "flex min-h-0 min-w-0 flex-col gap-3 overflow-hidden transition-opacity duration-200 ease-out lg:overflow-y-auto lg:overscroll-contain",
                WORKSPACE_COLUMN_PADDING,
                galleryOverlayActive && "lg:pointer-events-none lg:opacity-0",
                gallerySpotlightActive &&
                  "lg:pointer-events-none lg:opacity-20",
              )}
            >
              <AttachedControlStack className="flex min-w-0 flex-col">
                {status.batch ? (
                  <BatchStatus
                    batch={status.batch}
                    cancelRequested={Boolean(status.cancel_requested)}
                    expanded={batchStatusExpanded}
                    fallbackPrecision={fallbackPrecision}
                    onCancelCurrent={requestCancelCurrentGeneration}
                    onClearQueue={requestClearBatchQueue}
                    onExpandedChange={setBatchStatusExpanded}
                  />
                ) : completedBatch ? (
                  <BatchStatus
                    batch={completedBatch}
                    canResubmit={canSubmit}
                    completed
                    expanded={batchStatusExpanded}
                    fallbackPrecision={fallbackPrecision}
                    onClear={handleClearCompletedBatch}
                    onExpandedChange={setBatchStatusExpanded}
                    onResubmit={submitPrompt}
                  />
                ) : (
                  <PromptForm
                    canSubmit={canSubmit}
                    cancelRequested={Boolean(status.cancel_requested)}
                    generationRunning={Boolean(status.generation_running)}
                    onCancel={requestCancelCurrentGeneration}
                    onClearPrompt={handleClearPrompt}
                    onPromptChange={handlePromptChange}
                    onSubmit={handleGenerate}
                    prompt={prompt}
                  />
                )}
                <EventLog
                  activeStartedMs={activeTaskStartedMs}
                  attached
                  batch={status.batch}
                  busy={busy}
                  events={events}
                  fallbackMessage={status.message}
                  progress={status.progress}
                  task={status.task}
                />
              </AttachedControlStack>

              {currentError && (
                <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-body-sm text-destructive">
                  <HugeiconsIcon
                    icon={AlertCircleIcon}
                    strokeWidth={2}
                    className="size-4 shrink-0"
                  />
                  <span>{currentError}</span>
                </div>
              )}

              <ImageStage
                fallbackPrecision={fallbackPrecision}
                generating={stageGenerating}
                image={activeImage}
                imageUrl={imageUrl}
                onDelete={handleDeleteImage}
                onLoadSettings={handleLoadImageSettings}
                pendingImage={generationImage}
                preview={activePreview}
              />
            </section>

            {gallerySpotlightActive && (
              <GallerySpotlight
                className="lg:absolute lg:inset-y-0 lg:left-0 lg:z-20 lg:w-[calc(100%-var(--gallery-sidebar-width))]"
                fallbackPrecision={fallbackPrecision}
                image={gallerySpotlightImage}
                imageUrl={imageUrl}
                onDismiss={() => setGallerySpotlight(false)}
                onViewportWidthChange={setGallerySpotlightViewportWidth}
                zoomScale={gallerySpotlightEffectiveZoom}
              />
            )}

            <Gallery
              className={cn(
                "lg:absolute lg:inset-y-0 lg:right-0 lg:z-10 lg:h-full lg:w-[var(--gallery-sidebar-width)] lg:transition-all lg:duration-300 lg:ease-in-out",
                galleryOverlayActive && "lg:w-full",
                gallerySpotlightActive && "lg:z-30",
                !galleryVisible &&
                  "max-lg:hidden lg:pointer-events-none lg:translate-x-[calc(100%+1rem)] lg:opacity-0",
              )}
              expanded={galleryExpanded}
              fallbackPrecision={fallbackPrecision}
              imageUrl={imageUrl}
              items={recent}
              onDelete={handleDeleteImage}
              onExpandedChange={handleGalleryExpandedChange}
              onLoadSettings={handleLoadImageSettings}
              onOpenOutputDir={() => runAction(openOutputDir)}
              onSelect={handleSelectImage}
              onSpotlightToggle={handleGallerySpotlightToggle}
              selectedId={
                gallerySpotlightActive
                  ? gallerySpotlightImage?.id
                  : activeImage?.id
              }
              spotlightActive={gallerySpotlightActive}
              spotlightAvailable={gallerySpotlightAvailable}
              spotlightUnseenCount={gallerySpotlightUnseenIds.length}
            />
          </div>
        </div>

        <BatchDialog
          defaultOutputDir={
            status.output_dir?.path ?? DEFAULT_STATUS.output_dir.path
          }
          disabled={busy}
          generateBatch={generateBatchWithDefaults}
          onOpenChange={setBatchOpen}
          open={batchOpen}
          readBatchClipboard={readBatchClipboard}
          selectOutputDir={selectOutputDir}
          validateBatch={validateBatchWithDefaults}
        />
        <SourceImageDialog
          disabled={busy}
          initialImage={sourceImageInitialImage}
          onApply={handleApplySourceImageMetadata}
          onOpenChange={handleSourceImageOpenChange}
          open={sourceImageOpen}
          readSourceImageClipboard={readSourceImageClipboard}
          validateSourceImageFile={validateSourceImageFile}
          validateSourceImageId={validateSourceImageId}
          validateSourceImagePath={validateSourceImagePath}
        />
        <KeyboardShortcutsSheet
          modifierLabel={shortcutModifierKeyLabel}
          visible={shortcutsVisible}
        />
      </main>
    </TooltipProvider>
  );
}

export default KreaWorkspace;

function WorkspaceTopBar({
  busy,
  galleryVisible,
  loadRunning,
  livePreviewMode,
  model,
  modelError,
  onEjectModel,
  onGalleryVisibleToggle,
  onLoadModel,
  onLivePreviewModeChange,
  onPreviewIntervalBlur,
  onThemeChange,
  phase,
  previewIntervalSteps,
  previewIntervalWarning,
  server,
  setPreviewIntervalSteps,
  spotlightControls,
}) {
  const modelLoaded = Boolean(model?.loaded);

  return (
    <div className="grid min-h-8 w-full shrink-0 grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 bg-background px-1 py-1">
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <ServerStatus server={server} />
        <div className="flex min-w-0 items-center gap-1">
          <ModelStatus
            busy={busy}
            error={modelError}
            loadRunning={loadRunning}
            model={model}
            phase={phase}
          />
          <ModelActionButton
            busy={busy}
            loaded={modelLoaded}
            onEjectModel={onEjectModel}
            onLoadModel={onLoadModel}
          />
        </div>
      </div>

      <div className="flex min-w-0 items-center justify-center">
        {spotlightControls}
      </div>

      <div className="flex min-w-0 shrink-0 items-center justify-end gap-2">
        <IconActionButton
          ariaLabel={galleryVisible ? "Hide gallery" : "Show gallery"}
          expanded={galleryVisible}
          icon={
            galleryVisible ? VerticalColumn03NotFoundIcon : VerticalColumn03Icon
          }
          onClick={onGalleryVisibleToggle}
          tooltip={galleryVisible ? "Hide gallery" : "Show gallery"}
        />
        <SettingsSheet
          busy={busy}
          livePreviewMode={livePreviewMode}
          onLivePreviewModeChange={onLivePreviewModeChange}
          onPreviewIntervalBlur={onPreviewIntervalBlur}
          onThemeChange={onThemeChange}
          previewIntervalSteps={previewIntervalSteps}
          previewIntervalWarning={previewIntervalWarning}
          setPreviewIntervalSteps={setPreviewIntervalSteps}
        />
      </div>
    </div>
  );
}

function GallerySpotlightTopBarControls({
  onZoomDecrease,
  onZoomIncrease,
  onZoomReset,
  zoomDecreaseDisabled,
  zoomFeedbackLabel,
  zoomFeedbackVisible,
  zoomIncreaseDisabled,
  zoomResetDisabled,
}) {
  return (
    <div className="hidden items-center gap-2 lg:flex">
      <IconActionButton
        ariaLabel="Decrease spotlight image size"
        disabled={zoomDecreaseDisabled}
        icon={ZoomOutAreaIcon}
        onClick={onZoomDecrease}
        tooltip="Decrease image size"
      />
      <IconActionButton
        ariaLabel="Reset spotlight image size"
        disabled={zoomResetDisabled}
        icon={ImageActualSizeIcon}
        onClick={onZoomReset}
        tooltip="Reset image size"
      />
      <IconActionButton
        ariaLabel="Increase spotlight image size"
        disabled={zoomIncreaseDisabled}
        icon={ZoomInAreaIcon}
        onClick={onZoomIncrease}
        tooltip="Increase image size"
      />
      <span
        aria-hidden="true"
        className={cn(
          "w-16 text-center text-title-sm font-semibold text-foreground transition-opacity duration-300 ease-out",
          zoomFeedbackVisible ? "opacity-100" : "opacity-0",
        )}
      >
        {zoomFeedbackLabel}
      </span>
    </div>
  );
}

function ServerStatus({ server }) {
  const { label, tone } = serverStatusPresentation(server);

  return <StatusIndicator tone={tone}>{label}</StatusIndicator>;
}

export function serverStatusPresentation(server) {
  const connected = Boolean(server?.connected);
  const status = String(server?.status ?? "")
    .trim()
    .toLowerCase();

  if (connected) {
    return { label: "Server connected", tone: "ready" };
  }

  if (["error", "disconnected", "offline"].includes(status)) {
    return { label: "Server disconnected", tone: "error" };
  }

  return { label: "Server connecting", tone: "loading" };
}

function StatusIndicator({ children, tone }) {
  return (
    <div className="flex h-7 min-w-0 items-center gap-2 px-1 text-label-md font-medium text-foreground">
      <StatusDot tone={tone} />
      <span className="truncate">{children}</span>
    </div>
  );
}

function ModelStatus({ busy, error, loadRunning, model, phase }) {
  const { label, tone } = modelStatusPresentation({
    busy,
    error,
    loadRunning,
    model,
    phase,
  });

  return <StatusIndicator tone={tone}>{label}</StatusIndicator>;
}

export function modelStatusPresentation({
  busy,
  error,
  loadRunning,
  model,
  phase,
} = {}) {
  const loaded = Boolean(model?.loaded);
  const modelStatus = String(model?.status ?? "")
    .trim()
    .toLowerCase();
  const hasError = Boolean(error);
  const loadPhase =
    phase === TASK_PHASES.LOAD || phase === TASK_PHASES.MODEL_LOAD;
  const loading =
    !loaded &&
    (loadRunning ||
      phase === TASK_PHASES.LOAD ||
      (busy && phase === TASK_PHASES.MODEL_LOAD));
  const failed =
    !loaded &&
    (["error", "failed", "unavailable"].includes(modelStatus) ||
      (hasError && loadPhase));

  if (loaded) {
    return { label: "Model loaded", tone: "ready" };
  }

  if (loading) {
    return { label: "Loading model", tone: "loading" };
  }

  if (failed) {
    return { label: "Model load failed", tone: "error" };
  }

  return { label: "Load model", tone: "idle" };
}

function ModelActionButton({ busy, loaded, onEjectModel, onLoadModel }) {
  const icon = loaded ? EjectIcon : Download01Icon;
  const label = loaded ? "Eject" : "Load";

  return (
    <IconActionButton
      ariaLabel={`${label} model`}
      disabled={busy}
      icon={icon}
      iconClassName={loaded ? "size-[18px]" : undefined}
      iconStrokeWidth={loaded ? 3 : undefined}
      onClick={loaded ? onEjectModel : onLoadModel}
      tooltip={`${label} model`}
    />
  );
}

function hasOpenModal() {
  return Boolean(
    globalThis.document?.querySelector?.(
      "[data-slot='dialog-content'], [data-slot='sheet-content']",
    ),
  );
}

function runAfterKeyDownHandlers(callback) {
  const defer = globalThis.queueMicrotask ?? ((next) => setTimeout(next, 0));
  defer(callback);
}

export function batchHasQueuedJobs(batch) {
  if (!batch) {
    return false;
  }

  const jobs = Array.isArray(batch.jobs) ? batch.jobs : [];
  if (
    jobs.some(
      (job) =>
        normalizeBatchJobStatus(job?.status) === BATCH_JOB_STATUSES.QUEUED,
    )
  ) {
    return true;
  }

  if (jobs.some((job) => typeof job?.status === "string")) {
    return false;
  }

  const queueRemaining = Number(batch.queue_remaining);
  if (Number.isFinite(queueRemaining)) {
    return queueRemaining > 0;
  }

  const index = Number(batch.index);
  const total = Number(batch.total);
  return Number.isFinite(index) && Number.isFinite(total) && index < total;
}

export function batchClearQueueFollowupKeyAction(
  event,
  {
    armedUntilMs = null,
    nowMs = typeof event.timeStamp === "number" ? event.timeStamp : Date.now(),
  } = {},
) {
  if (
    armedUntilMs === null ||
    nowMs > armedUntilMs ||
    event.defaultPrevented ||
    event.isComposing ||
    event.altKey ||
    event.shiftKey ||
    event.repeat ||
    event.key !== "Escape"
  ) {
    return null;
  }

  return "clearQueue";
}

function isGallerySpotlightDesktopViewport() {
  const matchMedia = globalThis.window?.matchMedia;
  if (!matchMedia) {
    return true;
  }

  return matchMedia(GALLERY_SPOTLIGHT_DESKTOP_QUERY).matches;
}

export function gallerySpotlightKeyAction(event) {
  if (!isPlainGalleryKeyEvent(event)) {
    return null;
  }

  if (event.key === "Escape" || event.key === "g") {
    return "close";
  }

  if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
    return "previous";
  }

  if (event.key === "ArrowDown" || event.key === "ArrowRight") {
    return "next";
  }

  const sizeAction = plainSizeAdjustmentKeyAction(event);
  if (sizeAction === "increase") {
    return "zoomIn";
  }

  if (sizeAction === "decrease") {
    return "zoomOut";
  }

  if (event.key === "0") {
    return "zoomReset";
  }

  if (event.key === "Enter") {
    return "openImage";
  }

  if (event.key === "Delete" || event.key === "Backspace") {
    return "delete";
  }

  return null;
}

export function sizeAdjustmentKeyAction(event) {
  if (!isPlainGalleryKeyEvent(event)) {
    return null;
  }

  return plainSizeAdjustmentKeyAction(event);
}

function plainSizeAdjustmentKeyAction(event) {
  const key = String(event.key ?? "");
  const code = String(event.code ?? "");

  if (key === "=" || key === "+" || code === "Equal" || code === "NumpadAdd") {
    return "increase";
  }

  if (key === "-" || code === "Minus" || code === "NumpadSubtract") {
    return "decrease";
  }

  return null;
}

export {
  generationCancelKeyAction,
  isPromptSubmitShortcut,
  shortcutModifierLabel,
} from "@/lib/keyboard-shortcuts";

function isPlainGalleryKeyEvent(event) {
  return !(
    event.defaultPrevented ||
    event.isComposing ||
    event.altKey ||
    event.ctrlKey ||
    event.metaKey ||
    event.shiftKey ||
    isEditableKeyTarget(event.target)
  );
}

function isEditableKeyTarget(target) {
  if (!target || typeof target.closest !== "function") {
    return false;
  }

  return Boolean(
    target.closest(
      "input, textarea, select, [contenteditable='true'], [contenteditable=''], [role='textbox']",
    ),
  );
}

export function galleryPreviewKeyAction(event) {
  if (!isPlainGalleryKeyEvent(event)) {
    return null;
  }

  if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
    return "previous";
  }

  if (event.key === "ArrowDown" || event.key === "ArrowRight") {
    return "next";
  }

  if (event.key.toLowerCase() === "g" || event.key === " ") {
    return "openSpotlight";
  }

  return null;
}

export function shortcutSheetKeyDownAction(
  event,
  { modifierHeld = false, platform } = {},
) {
  if (isShortcutModifierKey(event, platform)) {
    if (
      event.repeat ||
      event.defaultPrevented ||
      event.isComposing ||
      event.altKey ||
      event.shiftKey
    ) {
      return null;
    }

    return "pressModifier";
  }

  if (modifierHeld || hasPlatformShortcutModifier(event, platform)) {
    return "suppressChord";
  }

  return null;
}

export function shortcutSheetKeyUpAction(
  event,
  { modifierHeld = false, platform } = {},
) {
  if (isShortcutModifierKey(event, platform)) {
    return "releaseModifier";
  }

  if (modifierHeld && !hasPlatformShortcutModifier(event, platform)) {
    return "releaseModifier";
  }

  return null;
}

export function galleryImageByOffset(items, selectedId, offset) {
  if (!items.length) {
    return null;
  }

  const selectedIndex = items.findIndex((item) => item.id === selectedId);
  const startIndex = selectedIndex >= 0 ? selectedIndex : 0;
  const nextIndex = wrappedGalleryIndex(startIndex, offset, items.length);

  return items[nextIndex];
}

export function wrappedGalleryIndex(index, offset, length) {
  if (length <= 0) {
    return -1;
  }

  return (index + offset + length) % length;
}

export function nextGalleryImageIdAfterDelete(items, imageId) {
  if (items.length <= 1) {
    return null;
  }

  const deletedIndex = items.findIndex((item) => item.id === imageId);
  if (deletedIndex < 0) {
    return items[0]?.id ?? null;
  }

  const nextIndex = deletedIndex >= items.length - 1 ? 0 : deletedIndex + 1;
  return items[nextIndex]?.id ?? null;
}

export function gallerySpotlightInitialImage({ activeImage, batch, recent }) {
  if (!recent.length) {
    return null;
  }

  if (batch) {
    return recent[0];
  }

  if (activeImage?.id !== undefined) {
    return recent.find((item) => item.id === activeImage.id) ?? recent[0];
  }

  return recent[0];
}

export function galleryImageClickAction({
  desktopViewport,
  spotlightActive,
  stageGenerating,
}) {
  if (spotlightActive) {
    return "focusSpotlight";
  }

  if (stageGenerating && desktopViewport) {
    return "openSpotlight";
  }

  return "selectPreview";
}

export function nextGallerySpotlightUnseenIds({
  currentIds,
  focusedId,
  previousRecentIds,
  recent,
  spotlightActive,
}) {
  const recentIds = (Array.isArray(recent) ? recent : [])
    .map((item) => item?.id)
    .filter((id) => id !== undefined && id !== null);
  const recentIdSet = new Set(recentIds);

  if (!spotlightActive) {
    return [];
  }

  const previousIdSet =
    previousRecentIds instanceof Set
      ? previousRecentIds
      : new Set(previousRecentIds ?? []);
  const nextIds = (Array.isArray(currentIds) ? currentIds : []).filter(
    (id) => recentIdSet.has(id) && id !== focusedId,
  );

  for (const id of recentIds) {
    if (previousIdSet.has(id) || id === focusedId || nextIds.includes(id)) {
      continue;
    }

    nextIds.push(id);
  }

  return nextIds;
}

function sameStringArray(left, right) {
  if (left === right) {
    return true;
  }

  if (!Array.isArray(left) || !Array.isArray(right)) {
    return false;
  }

  return (
    left.length === right.length &&
    left.every((item, index) => item === right[index])
  );
}

export function promptSubmitAction(text) {
  const trimmed = String(text ?? "").trim();
  return BATCH_PROMPT_JSON_ARRAY_START.test(trimmed) ? "batch" : "image";
}

export function batchPromptTextFromJobs(jobs) {
  return JSON.stringify(
    Array.isArray(jobs) ? jobs.map(batchPromptJobFromDisplayJob) : [],
    null,
    2,
  );
}

function batchPromptStateFromSubmittedJobs({
  outputDir = "",
  sourceJobs,
  submittedJobs,
}) {
  const intent = batchResubmitIntentFromJobs({
    outputDir,
    sourceJobs,
    submittedJobs,
  });
  const displayJobs = batchJobsWithGuiDerivedFields(
    submittedJobs,
    intent.derivedFieldsByIndex,
  );
  const displayText = batchPromptTextFromJobs(displayJobs);
  const resubmitText = batchPromptTextFromJobs(intent.resubmitJobs);

  return {
    displayJobs,
    intent,
    options: {
      displayText,
      outputDir,
      resubmitText,
    },
  };
}

export function batchPromptStateFromCompletedBatch(completedBatch, intent) {
  if (!intent) {
    return {
      completedBatch,
      options: null,
    };
  }

  const jobs = Array.isArray(completedBatch?.jobs) ? completedBatch.jobs : [];
  const displayJobs = batchJobsWithGuiDerivedFields(
    jobs,
    intent.derivedFieldsByIndex,
  );
  const nextCompletedBatch = completedBatchFromJobs(displayJobs);
  const displayText = batchPromptTextFromJobs(displayJobs);
  const updatePromptOnCompletion = intent.updatePromptOnCompletion !== false;

  return {
    completedBatch: nextCompletedBatch ?? completedBatch,
    options: updatePromptOnCompletion
      ? {
          displayText,
          outputDir: intent.outputDir,
          resubmitText: batchPromptTextFromJobs(intent.resubmitJobs),
        }
      : null,
  };
}

export function batchResubmitIntentFromJobs({
  outputDir = "",
  sourceJobs,
  submittedJobs,
}) {
  const normalizedSourceJobs = Array.isArray(sourceJobs) ? sourceJobs : [];
  const normalizedSubmittedJobs = Array.isArray(submittedJobs)
    ? submittedJobs
    : [];
  const derivedFieldsByIndex = normalizedSubmittedJobs.map((job, index) =>
    batchGuiDerivedFields(normalizedSourceJobs[index], job),
  );

  return {
    derivedFieldsByIndex,
    outputDir,
    resubmitJobs: normalizedSubmittedJobs.map((job, index) =>
      batchResubmitJobFromDisplayJob(job, derivedFieldsByIndex[index]),
    ),
  };
}

function batchGuiDerivedFields(sourceJob, displayJob) {
  if (!sourceJob || typeof sourceJob !== "object" || Array.isArray(sourceJob)) {
    return [];
  }

  return BATCH_JOB_GUI_DERIVED_KEYS.filter(
    (key) =>
      !Object.hasOwn(sourceJob, key) &&
      (key === "seed" || displayJob?.[key] !== undefined),
  );
}

function batchJobsWithGuiDerivedFields(jobs, derivedFieldsByIndex) {
  return (Array.isArray(jobs) ? jobs : []).map((job, index) => {
    const derivedFields = derivedFieldsByIndex[index] ?? [];
    if (!derivedFields.length) {
      return { ...job };
    }

    return {
      ...job,
      guiDerivedFields: derivedFields,
    };
  });
}

function batchResubmitJobFromDisplayJob(job, derivedFields = []) {
  const derivedFieldSet = new Set(derivedFields);
  const promptJob = batchPromptJobFromDisplayJob(job);

  for (const field of derivedFieldSet) {
    delete promptJob[field];
  }

  return promptJob;
}

function batchPromptJobFromDisplayJob(job) {
  if (!job || typeof job !== "object" || Array.isArray(job)) {
    return job;
  }

  const promptJob = {};
  for (const key of BATCH_JOB_PROMPT_KEYS) {
    if (Object.hasOwn(job, key)) {
      promptJob[key] = job[key];
    }
  }

  return promptJob;
}

export function completedBatchFromJobs(jobs) {
  if (!Array.isArray(jobs) || jobs.length === 0) {
    return null;
  }

  const normalizedJobs = jobs.map((job, index) => {
    const jobIndex = Number(job?.index);
    return {
      ...job,
      index: Number.isFinite(jobIndex) && jobIndex > 0 ? jobIndex : index + 1,
    };
  });
  const total = normalizedJobs.length;

  return {
    ...normalizedJobs[total - 1],
    index: total,
    total,
    jobs: normalizedJobs,
  };
}

export function completedBatchFromSnapshot(batch) {
  if (!isFinalBatchSnapshot(batch)) {
    return null;
  }

  if (Array.isArray(batch.jobs) && batch.jobs.length > 0) {
    return completedBatchFromJobs(batch.jobs);
  }

  return completedBatchFromJobs([batch]);
}

function isFinalBatchSnapshot(batch) {
  if (batch?.interrupted) {
    return false;
  }

  const index = Number(batch?.index);
  const total = Number(batch?.total);

  return (
    Number.isFinite(index) &&
    Number.isFinite(total) &&
    total > 0 &&
    index >= total
  );
}

function gallerySpotlightZoomForWidth(image, viewportWidth) {
  const imageWidth = positiveNumber(image?.width);
  const availableWidth = positiveNumber(viewportWidth);

  if (!imageWidth || !availableWidth) {
    return null;
  }

  return availableWidth / imageWidth;
}

function gallerySpotlightZoomMax(fitZoom) {
  return Math.max(
    GALLERY_SPOTLIGHT_MAX_ZOOM,
    GALLERY_SPOTLIGHT_RESET_ZOOM,
    positiveNumber(fitZoom) ?? 0,
  );
}

export function nextGallerySpotlightZoom(
  currentZoom,
  direction,
  maxZoom,
  fitZoom,
) {
  const current = clampGallerySpotlightZoom(currentZoom, maxZoom, fitZoom);
  const offset = Math.sign(direction) * GALLERY_SPOTLIGHT_ZOOM_STEP;

  if (offset === 0) {
    return current;
  }

  const next = current + offset;
  const snapZoom = crossedGallerySpotlightSnapZoom(current, next, direction, [
    fitZoom,
    GALLERY_SPOTLIGHT_RESET_ZOOM,
  ]);

  if (snapZoom !== null) {
    return clampGallerySpotlightZoom(snapZoom, maxZoom, fitZoom);
  }

  return clampGallerySpotlightZoom(next, maxZoom, fitZoom);
}

function crossedGallerySpotlightSnapZoom(current, next, direction, snapZooms) {
  const orderedSnapZooms = snapZooms
    .map((snapZoom) => positiveNumber(snapZoom))
    .filter((snapZoom) => snapZoom !== null)
    .filter(
      (snapZoom, index, values) =>
        values.findIndex(
          (value) =>
            Math.abs(value - snapZoom) <= GALLERY_SPOTLIGHT_ZOOM_EPSILON,
        ) === index,
    )
    .sort((left, right) => (direction > 0 ? left - right : right - left));

  return (
    orderedSnapZooms.find((snapZoom) =>
      direction > 0
        ? current < snapZoom && next > snapZoom
        : current > snapZoom && next < snapZoom,
    ) ?? null
  );
}

export function formatGallerySpotlightZoom(zoom) {
  const value = positiveNumber(zoom) ?? GALLERY_SPOTLIGHT_RESET_ZOOM;
  return `${Math.round(value * 100)}%`;
}

function clampGallerySpotlightZoom(zoom, maxZoom, fitZoom) {
  const upperBound = positiveNumber(maxZoom)
    ? maxZoom
    : Number.POSITIVE_INFINITY;
  const lowerBound = gallerySpotlightLowerZoomBound(upperBound, fitZoom);
  const value = positiveNumber(zoom) ?? GALLERY_SPOTLIGHT_RESET_ZOOM;

  return Math.min(Math.max(value, lowerBound), upperBound);
}

function gallerySpotlightLowerZoomBound(maxZoom, fitZoom) {
  const lowerBound = Math.min(
    GALLERY_SPOTLIGHT_MIN_ZOOM,
    positiveNumber(fitZoom) ?? GALLERY_SPOTLIGHT_MIN_ZOOM,
  );

  if (!Number.isFinite(maxZoom)) {
    return lowerBound;
  }

  return Math.min(lowerBound, maxZoom);
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function imageDetailsFromGeneration(value) {
  if (!value?.prompt) {
    return null;
  }

  return {
    prompt: value.prompt,
    width: numericDisplayValue(value.width),
    height: numericDisplayValue(value.height),
    loras: Array.isArray(value.loras) ? value.loras : [],
    steps: numericDisplayValue(value.steps),
    seed: value.seed ?? "random",
  };
}

function loraClampWarning(sourceLoras, normalizedLoras, catalogItems) {
  const source = Array.isArray(sourceLoras) ? sourceLoras : [];
  for (const normalized of normalizedLoras) {
    const raw = source.find((item) => item?.id === normalized.id);
    if (!raw) {
      continue;
    }
    const rawScale = Number(raw.scale);
    if (!Number.isFinite(rawScale) || rawScale === Number(normalized.scale)) {
      continue;
    }
    const item = findLoraCatalogItem(catalogItems, normalized.id);
    const limits = loraScaleLimitsForItem(item);
    return `Scale must be ${limits.min} to ${limits.max}. Clamped to ${formatLoraScale(
      normalized.scale,
    )}.`;
  }
  return "";
}

function numericDisplayValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : "-";
}
