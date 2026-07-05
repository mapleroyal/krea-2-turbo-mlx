import LaptopPhoneSyncIcon from "@hugeicons/core-free-icons/LaptopPhoneSyncIcon";
import Moon02Icon from "@hugeicons/core-free-icons/Moon02Icon";
import Sun01Icon from "@hugeicons/core-free-icons/Sun01Icon";
import { HugeiconsIcon } from "@hugeicons/react";

import { cn } from "@/lib/utils";
import { isTheme, useAppStore } from "@/stores/use-app-store";

const THEME_OPTIONS = [
  {
    value: "light",
    label: "Light",
    icon: Sun01Icon,
  },
  {
    value: "system",
    label: "System",
    icon: LaptopPhoneSyncIcon,
  },
  {
    value: "dark",
    label: "Dark",
    icon: Moon02Icon,
  },
];

const themeIndexes = {
  light: 0,
  system: 1,
  dark: 2,
};

export function ThemeModeSwitcher({ className, onThemeChange }) {
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);

  return (
    <div
      role="group"
      aria-label="Theme mode"
      className={cn(
        "relative grid h-9 w-full grid-cols-3 items-center rounded-4xl bg-muted p-0.5 shadow-inner transition-colors duration-300",
        className,
      )}
      style={{ "--theme-index": themeIndexes[theme] }}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute top-0.5 left-0.5 h-8 w-[calc((100%-0.25rem)/3)] rounded-4xl border border-border bg-background shadow-sm transition-transform duration-300 ease-[cubic-bezier(0.25,1,0.5,1)]"
        style={{
          transform: "translateX(calc(var(--theme-index) * 100%))",
        }}
      />

      {THEME_OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => {
            if (isTheme(option.value)) {
              setTheme(option.value);
              onThemeChange?.(option.value);
            }
          }}
          aria-label={option.label}
          aria-pressed={theme === option.value}
          className={cn(
            "relative z-10 flex h-full w-full items-center justify-center rounded-full bg-transparent text-muted-foreground transition-colors duration-200 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50",
            theme === option.value && "text-foreground hover:text-foreground",
          )}
        >
          <HugeiconsIcon
            icon={option.icon}
            strokeWidth={2.5}
            className="size-4"
          />
        </button>
      ))}
    </div>
  );
}
