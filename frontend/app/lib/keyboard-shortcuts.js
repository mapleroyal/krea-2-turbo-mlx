export const DOUBLE_ESCAPE_CANCEL_MS = 900;
export const SHORTCUT_CHEAT_SHEET_HOLD_MS = 1000;

export function currentPlatform() {
  const navigator = globalThis.navigator;
  return navigator?.userAgentData?.platform ?? navigator?.platform ?? "";
}

export function isApplePlatform(platform) {
  return /^(Mac|iPhone|iPad|iPod)/i.test(String(platform ?? ""));
}

export function shortcutModifierLabel(platform = currentPlatform()) {
  return isApplePlatform(platform) ? "Cmd" : "Ctrl";
}

export function isShortcutModifierKey(event, platform = currentPlatform()) {
  const key = String(event.key ?? "");
  return isApplePlatform(platform)
    ? key === "Meta" || key === "OS"
    : key === "Control";
}

export function hasPlatformShortcutModifier(
  event,
  platform = currentPlatform(),
) {
  if (isApplePlatform(platform)) {
    return Boolean(event.metaKey) && !event.ctrlKey;
  }

  return Boolean(event.ctrlKey) && !event.metaKey;
}

export function isPromptSubmitShortcut(event, platform = currentPlatform()) {
  if (
    event.key !== "Enter" ||
    event.altKey ||
    event.shiftKey ||
    event.isComposing
  ) {
    return false;
  }

  return hasPlatformShortcutModifier(event, platform);
}

export function generationCancelKeyAction(
  event,
  {
    doubleEscapeMs = DOUBLE_ESCAPE_CANCEL_MS,
    lastEscapeKeyDownMs = null,
    nowMs = eventTimestamp(event),
  } = {},
) {
  if (
    event.defaultPrevented ||
    event.isComposing ||
    event.altKey ||
    event.shiftKey
  ) {
    return null;
  }

  if (event.key !== "Escape" || event.ctrlKey || event.metaKey) {
    return lastEscapeKeyDownMs === null ? null : "clearEscape";
  }

  if (event.repeat) {
    return null;
  }

  if (
    lastEscapeKeyDownMs !== null &&
    nowMs - lastEscapeKeyDownMs <= doubleEscapeMs
  ) {
    return "cancel";
  }

  return "primeEscape";
}

function eventTimestamp(event) {
  return typeof event.timeStamp === "number" ? event.timeStamp : Date.now();
}
