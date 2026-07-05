import { clsx } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

const SEMANTIC_TEXT_CLASSES = [
  "text-display-lg",
  "text-display-md",
  "text-display-sm",
  "text-headline-lg",
  "text-headline-md",
  "text-headline-sm",
  "text-title-lg",
  "text-title-md",
  "text-title-sm",
  "text-body-lg",
  "text-body-md",
  "text-body-sm",
  "text-label-lg",
  "text-label-md",
  "text-label-sm",
];

const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      "font-size": SEMANTIC_TEXT_CLASSES,
    },
  },
});

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
