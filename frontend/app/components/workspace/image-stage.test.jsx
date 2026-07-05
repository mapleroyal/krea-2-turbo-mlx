import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { ImageStage } from "./image-stage";

describe("ImageStage", () => {
  it("renders the active live preview before the saved image", () => {
    const html = renderToString(
      <TooltipProvider>
        <ImageStage
          fallbackPrecision="bf16"
          image={{
            filename: "final.png",
            height: 512,
            prompt: "final prompt",
            seed: 7,
            steps: 8,
            url: "/api/image/7",
            width: 512,
          }}
          imageUrl={(url) => url}
          pendingImage={{
            height: 1024,
            prompt: "new prompt",
            seed: 4127718317,
            steps: 8,
            width: 1024,
          }}
          preview={{
            height: 512,
            mode: "latent",
            revision: 3,
            step: 2,
            step_count: 8,
            url: "/api/preview/current?rev=3",
            width: 512,
          }}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("/api/preview/current?rev=3");
    expect(html).toContain("Live latent preview · Step 2/8");
    expect(html).toContain("krea-2-turbo-mlx");
    expect(html).toContain("bf16 @ 1024x1024");
    expect(html).toContain("Seed: 4127718317");
    expect(html).toContain("Steps: 8");
    expect(html).not.toContain("/api/image/7");
    expect(html).not.toContain("final prompt");
  });

  it("renders the generation placeholder before the saved image", () => {
    const html = renderToString(
      <TooltipProvider>
        <ImageStage
          fallbackPrecision="bf16"
          generating
          image={{
            filename: "old.png",
            height: 512,
            prompt: "old prompt",
            seed: 7,
            steps: 8,
            url: "/api/image/7",
            width: 512,
          }}
          imageUrl={(url) => url}
          pendingImage={{
            height: 128,
            prompt: "new prompt",
            seed: 11,
            steps: 3,
            width: 64,
          }}
          preview={null}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("Preparing preview");
    expect(html).toContain("64x128");
    expect(html).toContain("Seed: 11");
    expect(html).toContain("Steps: 3");
    expect(html).not.toContain("/api/image/7");
    expect(html).not.toContain("old prompt");
  });

  it("renders live preview before the generation placeholder", () => {
    const html = renderToString(
      <TooltipProvider>
        <ImageStage
          fallbackPrecision="bf16"
          generating
          image={null}
          imageUrl={(url) => url}
          pendingImage={{
            height: 128,
            prompt: "new prompt",
            seed: 11,
            steps: 3,
            width: 64,
          }}
          preview={{
            height: 512,
            mode: "vae",
            revision: 4,
            step: 1,
            step_count: 3,
            url: "/api/preview/current?rev=4",
            width: 512,
          }}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("/api/preview/current?rev=4");
    expect(html).toContain("Live VAE preview");
    expect(html).not.toContain("Preparing preview");
    expect(html).toContain("Seed: 11");
    expect(html).toContain("Steps: 3");
  });
});
