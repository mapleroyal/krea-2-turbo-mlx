import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { Gallery } from "./gallery";

describe("Gallery", () => {
  it("renders all supplied images without limit controls", () => {
    const items = Array.from({ length: 5 }, (_, index) => ({
      filename: `image-${index}.png`,
      height: 512,
      id: `image-${index}`,
      prompt: `gallery prompt ${index}`,
      seed: index,
      steps: 8,
      url: `/images/image-${index}.png`,
      width: 512,
    }));
    items[0].loras = [
      { id: "style.safetensors", display_name: "Style", scale: 1.3 },
    ];

    const html = renderToString(
      <TooltipProvider>
        <Gallery
          expanded={false}
          fallbackPrecision="bf16"
          imageUrl={(url) => url}
          items={items}
          onDelete={() => {}}
          onExpandedChange={() => {}}
          onOpenOutputDir={() => {}}
          onSelect={() => {}}
          selectedId={null}
        />
      </TooltipProvider>,
    );

    for (const item of items) {
      expect(html).toContain(item.prompt);
    }

    expect(html).toContain("Style: 1.3");
    expect(html).not.toContain("Show fewer gallery images");
    expect(html).not.toContain("Show more gallery images");
    expect(html).not.toContain("more in the gallery are not shown");
  });

  it("renders load settings actions when supplied", () => {
    const html = renderToString(
      <TooltipProvider>
        <Gallery
          expanded={false}
          fallbackPrecision="bf16"
          imageUrl={(url) => url}
          items={[
            {
              filename: "image.png",
              height: 512,
              id: "image",
              prompt: "gallery prompt",
              seed: 7,
              steps: 8,
              url: "/images/image.png",
              width: 512,
            },
          ]}
          onDelete={() => {}}
          onExpandedChange={() => {}}
          onLoadSettings={() => {}}
          onOpenOutputDir={() => {}}
          onSelect={() => {}}
          selectedId={null}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("Load image settings");
    expect(html).toContain("Delete image");
    expect(html).toContain("Show prompt");
  });

  it("renders a spotlight unseen image count beside the title", () => {
    const html = renderToString(
      <TooltipProvider>
        <Gallery
          expanded={false}
          fallbackPrecision="bf16"
          imageUrl={(url) => url}
          items={[]}
          onDelete={() => {}}
          onExpandedChange={() => {}}
          onOpenOutputDir={() => {}}
          onSelect={() => {}}
          selectedId={null}
          spotlightUnseenCount={3}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("Gallery");
    expect(html).toContain("3 unseen generated images");
  });

  it("places the hide control between gallery layout and output folder actions", () => {
    const html = renderToString(
      <TooltipProvider>
        <Gallery
          expanded={false}
          fallbackPrecision="bf16"
          imageUrl={(url) => url}
          items={[]}
          onDelete={() => {}}
          onExpandedChange={() => {}}
          onGalleryVisibleToggle={() => {}}
          onOpenOutputDir={() => {}}
          onSelect={() => {}}
          selectedId={null}
        />
      </TooltipProvider>,
    );

    const expandIndex = html.indexOf('aria-label="Expand gallery"');
    const hideIndex = html.indexOf('aria-label="Hide gallery"');
    const outputIndex = html.indexOf('aria-label="Open output folder"');

    expect(expandIndex).toBeGreaterThan(-1);
    expect(hideIndex).toBeGreaterThan(expandIndex);
    expect(outputIndex).toBeGreaterThan(hideIndex);
  });
});
