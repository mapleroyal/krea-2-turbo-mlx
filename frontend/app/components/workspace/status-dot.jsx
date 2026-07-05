import { cn } from "@/lib/utils";

const TONE_CLASSES = {
  ready: "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.55)]",
  loading: "bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.5)]",
  error: "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]",
  idle: "bg-zinc-500",
};

export function StatusDot({ tone = "idle", className }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-block size-2 shrink-0 rounded-full",
        TONE_CLASSES[tone] ?? TONE_CLASSES.idle,
        className,
      )}
    />
  );
}
