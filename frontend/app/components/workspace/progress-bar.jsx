import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export function ProgressBar({
  ariaLabel = "Generation progress",
  busy,
  className,
  progress,
}) {
  const normalized = Math.max(0, Math.min(1, Number(progress) || 0));
  const value = Math.round(normalized * 100);

  return (
    <Progress
      value={value}
      indeterminate={busy && value <= 0}
      className={cn("h-1.5 rounded-[3px] bg-muted", className)}
      aria-label={ariaLabel}
    />
  );
}
