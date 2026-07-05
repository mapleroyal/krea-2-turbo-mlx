import { cn } from "@/lib/utils";

export const ATTACHED_CONTROL_SHADOW =
  "shadow-[0_6px_8px_-1px_rgba(0,0,0,0.06)]";

export function AttachedControlStack({ className, ...props }) {
  return <div className={cn(className)} {...props} />;
}
