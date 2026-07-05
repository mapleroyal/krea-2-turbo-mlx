import { describe, expect, it } from "vitest";

import { promptPopupPositionForImage } from "./prompt-preview";

describe("promptPopupPositionForImage", () => {
  it("centers the prompt popup on a gallery image", () => {
    const position = promptPopupPositionForImage(
      { height: 300, left: 20, top: 30, width: 300 },
      { height: 80, width: 240 },
      {
        viewportHeight: 900,
        viewportWidth: 1200,
      },
    );

    expect(position).toEqual({
      x: 50,
      y: 140,
    });
  });

  it("keeps centered prompt popups inside the viewport margin", () => {
    const position = promptPopupPositionForImage(
      { height: 300, left: 0, top: 0, width: 300 },
      { height: 120, width: 380 },
      {
        viewportHeight: 360,
        viewportWidth: 400,
      },
    );

    expect(position).toEqual({
      x: 12,
      y: 90,
    });
  });
});
