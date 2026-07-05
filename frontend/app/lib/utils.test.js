import { describe, expect, it } from "vitest";

import { cn } from "./utils";

describe("cn", () => {
  it("merges conflicting Tailwind classes and keeps non-conflicts", () => {
    expect(cn("px-2 py-1", "px-4", "text-sm")).toBe("py-1 px-4 text-sm");
  });

  it("keeps semantic typography classes when text colors are merged", () => {
    expect(cn("text-body-md", "text-foreground")).toBe(
      "text-body-md text-foreground",
    );
    expect(cn("text-muted-foreground", "text-label-md")).toBe(
      "text-muted-foreground text-label-md",
    );
  });
});
