import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { EventLog } from "./event-log";

describe("EventLog", () => {
  it("shows active elapsed time from the local task start", () => {
    const startedAt = Date.now() - 2500;
    const html = renderToString(
      <TooltipProvider>
        <EventLog
          activeStartedMs={startedAt}
          busy
          events={[]}
          fallbackMessage="Generation queued"
          progress={0}
          task={{ started_ms: null, completed_ms: null }}
        />
      </TooltipProvider>,
    );

    expect(html).toMatch(/2\.5\d?s/);
  });

  it("shows separate job and batch progress meters for batch generation", () => {
    const now = Date.now();
    const html = renderToString(
      <TooltipProvider>
        <EventLog
          batch={{
            index: 3,
            total: 10,
            current_started_ms: now - 5000,
            progress: 0.4,
          }}
          busy
          events={[]}
          fallbackMessage="Job 3 of 10"
          progress={0.4}
          task={{ started_ms: now - 120000, completed_ms: null }}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("Job 3");
    expect(html).toContain("Batch (10)");
    expect(html).toContain("Current job progress");
    expect(html).toContain("Batch progress");
    expect(html).toContain("2m 0s");
  });

  it("collapses the idle progress meter when progress resets to zero", () => {
    const html = renderToString(
      <TooltipProvider>
        <EventLog
          events={[
            {
              id: 7,
              completed_ms: null,
              details: {},
              kind: "system",
              message: "Model ejected from memory",
              progress: 0,
              stage: "model_load",
              time: "12:00:00",
              time_ms: 1_779_840_000_000,
            },
          ]}
          fallbackMessage="Model ejected"
          progress={0}
          task={{
            started_ms: 1_779_839_992_000,
            completed_ms: 1_779_840_000_000,
          }}
        />
      </TooltipProvider>,
    );

    expect(html).toContain('aria-hidden="true"');
    expect(html).toContain("0.000s");
    expect(html).not.toContain("8.00s");
  });

  it("shows completed task summaries while collapsing the progress meter", () => {
    const html = renderToString(
      <TooltipProvider>
        <EventLog
          events={[
            {
              id: 11,
              completed_ms: 1_779_840_008_000,
              details: { elapsed_seconds: 8 },
              kind: "task",
              message: "Generate image",
              progress: 1,
              stage: "task",
              time: "12:00:08",
              time_ms: 1_779_840_008_000,
            },
          ]}
          fallbackMessage="Saved image"
          progress={1}
          task={{
            name: "Generate image",
            started_ms: 1_779_840_000_000,
            completed_ms: 1_779_840_008_000,
          }}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("Generate image");
    expect(html).toContain("8.00s");
    expect(html).toContain('aria-hidden="true"');
  });

  it("keeps completed delete event durations fixed while another timer is active", () => {
    const now = Date.now();
    const html = renderToString(
      <TooltipProvider>
        <EventLog
          activeStartedMs={now - 5000}
          busy
          events={[
            {
              id: 12,
              completed_ms: now - 44_980,
              details: { elapsed_seconds: 0.02 },
              kind: "system",
              message: "Deleted image.png",
              progress: null,
              stage: "output",
              time: "12:00:09",
              time_ms: now - 45_000,
            },
          ]}
          fallbackMessage="Deleted image"
          progress={0.5}
          task={{ started_ms: now - 5000, completed_ms: null }}
        />
      </TooltipProvider>,
    );

    expect(html).toContain("Deleted image.png");
    expect(html).toContain("0.020s");
    expect(html).not.toContain("45.00s");
  });
});
