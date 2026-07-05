import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { cn } from "@/lib/utils";

const SHORTCUT_SECTIONS = [
  {
    title: "Prompt",
    shortcuts: [
      { keys: ["modifier", "Enter"], label: "Generate" },
      { keys: ["Esc", "Esc"], joiner: "then", label: "Cancel current" },
      { keys: ["Esc"], label: "Clear queue after cancel" },
      { keys: ["+", "-"], joiner: "/", label: "Batch count" },
    ],
  },
  {
    title: "Gallery",
    shortcuts: [
      { keys: ["Arrows"], label: "Select image" },
      { keys: ["G"], label: "Open spotlight" },
      { keys: ["Space"], label: "Open spotlight" },
    ],
  },
  {
    title: "Spotlight",
    shortcuts: [
      { keys: ["Esc"], label: "Close" },
      { keys: ["G"], label: "Close" },
      { keys: ["Arrows"], label: "Previous / next" },
      { keys: ["+", "-"], joiner: "/", label: "Resize image" },
      { keys: ["0"], label: "Reset size" },
      { keys: ["Enter"], label: "Open image" },
      { keys: ["Delete", "Backspace"], joiner: "/", label: "Delete image" },
    ],
  },
];

export function KeyboardShortcutsSheet({ modifierLabel, visible }) {
  return (
    <div
      aria-hidden={!visible}
      className={cn(
        "pointer-events-none fixed right-4 bottom-4 z-50 w-[min(calc(100vw-2rem),42rem)] transition-all duration-150 ease-out",
        visible
          ? "translate-y-0 opacity-100"
          : "translate-y-2 opacity-0 motion-safe:scale-[0.98]",
      )}
    >
      <div className="rounded-lg border border-border bg-popover/95 p-4 text-popover-foreground shadow-2xl ring-1 ring-foreground/5 backdrop-blur">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-title-md">Keyboard shortcuts</h2>
          <Kbd>{modifierLabel}</Kbd>
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          {SHORTCUT_SECTIONS.map((section) => (
            <section key={section.title} className="min-w-0">
              <h3 className="mb-2 text-label-md text-muted-foreground">
                {section.title}
              </h3>
              <dl className="grid gap-2">
                {section.shortcuts.map((shortcut) => (
                  <div
                    key={`${section.title}-${shortcut.label}-${shortcut.keys.join("-")}`}
                    className="grid min-h-6 grid-cols-[minmax(0,1fr)_auto] items-center gap-3"
                  >
                    <dt className="truncate text-body-sm text-popover-foreground">
                      {shortcut.label}
                    </dt>
                    <dd>
                      <ShortcutKeys
                        joiner={shortcut.joiner}
                        keys={shortcut.keys}
                        modifierLabel={modifierLabel}
                      />
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

function ShortcutKeys({ joiner = "+", keys, modifierLabel }) {
  return (
    <KbdGroup className="justify-end">
      {keys.map((key, index) => (
        <span
          key={`${key}-${index}`}
          className="inline-flex items-center gap-1"
        >
          {index > 0 && (
            <span className="text-label-sm text-muted-foreground">
              {joiner}
            </span>
          )}
          <Kbd>{key === "modifier" ? modifierLabel : key}</Kbd>
        </span>
      ))}
    </KbdGroup>
  );
}
