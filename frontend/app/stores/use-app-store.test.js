import { describe, expect, it, vi } from "vitest";

import { THEME_STORAGE_KEY } from "@/lib/app-config";
import {
  createAppStore,
  getInitialTheme,
  getResolvedTheme,
} from "./use-app-store";

function createStorage(initialTheme = null) {
  let value = initialTheme;

  return {
    getItem: vi.fn(() => value),
    setItem: vi.fn((_, nextTheme) => {
      value = nextTheme;
    }),
  };
}

describe("useAppStore", () => {
  it("getInitialTheme prefers a persisted user theme", () => {
    const storage = createStorage("dark");
    const matchMedia = vi.fn(() => ({ matches: false }));

    expect(getInitialTheme({ matchMedia, storage })).toBe("dark");
    expect(matchMedia).not.toHaveBeenCalled();
  });

  it("getInitialTheme falls back to system preference when no persisted theme exists", () => {
    const storage = createStorage();
    const matchMedia = vi.fn(() => ({ matches: true }));

    expect(getInitialTheme({ matchMedia, storage })).toBe("system");
    expect(matchMedia).not.toHaveBeenCalled();
  });

  it("getInitialTheme falls back to system when storage is unavailable", () => {
    expect(getInitialTheme()).toBe("system");
  });

  it("getResolvedTheme uses the OS preference when the user selects system", () => {
    const matchMedia = vi.fn(() => ({ matches: true }));

    expect(getResolvedTheme("system", matchMedia)).toBe("dark");
    expect(matchMedia).toHaveBeenCalledWith("(prefers-color-scheme: dark)");
  });

  it("setTheme persists valid themes and ignores invalid ones", () => {
    const storage = createStorage();
    const store = createAppStore({ storage });

    store.getState().setTheme("dark");
    expect(store.getState().theme).toBe("dark");
    expect(storage.setItem).toHaveBeenLastCalledWith(THEME_STORAGE_KEY, "dark");

    // Values that fail the isTheme guard are a no-op: no state change, no persist.
    store.getState().setTheme("neon");
    expect(store.getState().theme).toBe("dark");
    expect(storage.setItem).toHaveBeenCalledTimes(1);
  });

  it("toggleTheme flips the resolved theme and persists an explicit choice", () => {
    const storage = createStorage("system");
    const matchMedia = vi.fn(() => ({ matches: true }));
    const store = createAppStore({ storage, matchMedia });

    store.getState().toggleTheme();
    expect(store.getState().theme).toBe("light");
    expect(storage.setItem).toHaveBeenLastCalledWith(
      THEME_STORAGE_KEY,
      "light",
    );

    store.getState().toggleTheme();
    expect(store.getState().theme).toBe("dark");
    expect(storage.setItem).toHaveBeenLastCalledWith(THEME_STORAGE_KEY, "dark");
  });
});
