import { useEffect, useMemo, useRef, useState } from "react";
import ArrowExpand01Icon from "@hugeicons/core-free-icons/ArrowExpand01Icon";
import ArrowShrink01Icon from "@hugeicons/core-free-icons/ArrowShrink01Icon";

import { ScrollArea } from "@/components/ui/scroll-area";
import { useEventLogScroll } from "@/hooks/use-event-log-scroll";
import { cn } from "@/lib/utils";

import { IconActionButton } from "./icon-action-button";
import { ProgressBar } from "./progress-bar";

const TICK_MS = 250;

export function EventLog({
  activeStartedMs,
  attached = false,
  batch,
  busy = false,
  className,
  events,
  fallbackMessage,
  progress,
  task,
}) {
  const [expanded, setExpanded] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const scrollContainerRef = useRef(null);
  const progressValue = normalizedProgress(progress);
  const timerRunning = isTimerRunning({
    activeStartedMs,
    busy,
    progressValue,
    task,
  });
  const annotatedEvents = useMemo(
    () => annotateEvents(events ?? [], now, timerRunning),
    [events, now, timerRunning],
  );
  const rows = annotatedEvents.length
    ? annotatedEvents.slice().reverse()
    : [fallbackEvent(fallbackMessage)];
  const currentEventKey = eventKey(rows[0], 0);
  const totalDuration = taskDuration(task, now, activeStartedMs);
  const taskName = taskLabel(task);
  const jobProgress = normalizedProgress(batch?.progress ?? progress);
  const batchProgress = batch
    ? normalizedProgress(
        batch.overall_progress ?? batchProgressValue(batch, jobProgress),
      )
    : progressValue;
  const jobDuration = batch
    ? durationFrom(batch.current_started_ms, null, now)
    : totalDuration;
  const progressFooterVisible =
    !hasCompletedTask(task) &&
    !isIdleAtZero({ busy, progressValue, timerRunning }) &&
    (hasTaskTiming(task, activeStartedMs) ||
      timerRunning ||
      busy ||
      progressValue > 0);

  useEventLogScroll(scrollContainerRef, rows);

  useEffect(() => {
    if (!timerRunning) {
      return undefined;
    }

    setNow(Date.now());
    const interval = window.setInterval(() => setNow(Date.now()), TICK_MS);

    return () => window.clearInterval(interval);
  }, [timerRunning]);

  return (
    <div
      className={cn(
        "[--prompt-log-attachment-depth:0px]",
        attached &&
          "[--prompt-log-attachment-depth:var(--radius-xl)] -mt-[var(--prompt-log-attachment-depth)]",
        className,
      )}
    >
      <section
        className={cn(
          "overflow-hidden border border-border bg-card",
          attached ? "relative rounded-b-lg border-t-0" : "relative rounded-lg",
        )}
      >
        <div ref={scrollContainerRef}>
          <ScrollArea
            className={
              expanded
                ? "h-[calc(480px+var(--prompt-log-attachment-depth))]"
                : "h-[calc(120px+var(--prompt-log-attachment-depth))]"
            }
          >
            <div>
              {rows.map((event, index) => {
                const key = eventKey(event, index);

                return (
                  <EventRow
                    event={event}
                    extendsBehindPrompt={attached && index === 0}
                    key={key}
                    current={key === currentEventKey}
                  />
                );
              })}
            </div>
          </ScrollArea>
        </div>
        <div className="absolute top-[calc(var(--prompt-log-attachment-depth)+0.5rem)] right-3 z-10">
          <IconActionButton
            ariaLabel={expanded ? "Collapse events" : "Expand events"}
            expanded={expanded}
            icon={expanded ? ArrowShrink01Icon : ArrowExpand01Icon}
            onClick={() => setExpanded((current) => !current)}
            tooltip={expanded ? "Collapse log" : "Expand log"}
          />
        </div>
      </section>
      {batch ? (
        <div
          className={cn(
            "mt-1 grid gap-1 text-label-md text-muted-foreground",
            !attached && "px-1",
          )}
        >
          <ProgressMeter
            ariaLabel="Current job progress"
            busy={timerRunning || busy}
            duration={jobDuration}
            label={batch.index ? `Job ${batch.index}` : "Job"}
            progress={jobProgress}
          />
          <ProgressMeter
            ariaLabel="Batch progress"
            busy={timerRunning || busy}
            duration={totalDuration}
            label={`Batch (${batch.total || 0})`}
            progress={batchProgress}
          />
        </div>
      ) : (
        <ProgressFooter attached={attached} visible={progressFooterVisible}>
          <ProgressBar
            ariaLabel={`${taskName} progress`}
            busy={timerRunning || busy}
            progress={progress}
            className="h-1.5 flex-1 rounded-[3px]"
          />
          <span className="shrink-0 type-numeric">
            {progressFooterVisible ? totalDuration : formatDuration(0)}
          </span>
        </ProgressFooter>
      )}
    </div>
  );
}

function ProgressFooter({ attached, children, visible }) {
  return (
    <div
      aria-hidden={!visible}
      className={cn(
        "overflow-hidden transition-[max-height,opacity,margin-top] duration-200 ease-out",
        visible
          ? "mt-1 max-h-8 opacity-100 delay-0"
          : "mt-0 max-h-0 opacity-0 delay-300",
      )}
    >
      <div
        className={cn(
          "flex items-center gap-3 text-label-md text-muted-foreground",
          !attached && "px-1",
        )}
      >
        {children}
      </div>
    </div>
  );
}

function ProgressMeter({ ariaLabel, busy, duration, label, progress }) {
  return (
    <div className="grid grid-cols-[72px_minmax(0,1fr)_auto] items-center gap-3">
      <span className="truncate">{label}</span>
      <ProgressBar
        ariaLabel={ariaLabel}
        busy={busy}
        progress={progress}
        className="h-1.5 rounded-[3px]"
      />
      <span className="shrink-0 type-numeric">{duration}</span>
    </div>
  );
}

function EventRow({ current, event, extendsBehindPrompt }) {
  const isError = event.kind === "error";
  const duration =
    event.duration_ms === null ? "" : formatDuration(event.duration_ms);

  return (
    <div
      className={cn(
        "grid h-10 grid-cols-[72px_minmax(0,1fr)_auto] items-center gap-3 border-t border-border px-3 first:border-t-0",
        current && "bg-muted/50",
        extendsBehindPrompt &&
          "h-[calc(2.5rem+var(--prompt-log-attachment-depth))] pt-[var(--prompt-log-attachment-depth)]",
      )}
    >
      <time className="font-mono text-label-md text-muted-foreground type-numeric [font-feature-settings:'liga'_0,'calt'_0] [font-variant-ligatures:none]">
        {event.time}
      </time>
      <div className="flex min-w-0 items-center gap-3">
        <p
          className={cn(
            "truncate text-body-md",
            isError ? "text-destructive" : "text-foreground",
          )}
        >
          {event.message}
        </p>
        {duration && (
          <span className="shrink-0 text-label-md text-muted-foreground type-numeric">
            {duration}
          </span>
        )}
      </div>
      <span aria-hidden="true" className="size-6" />
    </div>
  );
}

function annotateEvents(events, now, busy) {
  return events.map((event, index) => {
    const started = numericTime(event.time_ms);
    const completed = numericTime(event.completed_ms);
    const explicitDuration = numericSeconds(event.details?.elapsed_seconds);
    let duration = null;

    if (explicitDuration !== null) {
      duration = explicitDuration * 1000;
    } else if (started !== null && completed !== null) {
      duration = completed - started;
    } else if (started !== null && busy && index === events.length - 1) {
      duration = now - started;
    } else if (started !== null) {
      duration = 0;
    }

    return {
      ...event,
      duration_ms: duration === null ? null : Math.max(0, duration),
    };
  });
}

function fallbackEvent(fallbackMessage) {
  return {
    duration_ms: 0,
    id: "fallback",
    kind: "info",
    message: fallbackMessage || "Waiting for events",
    stage: "status",
    time: "--:--:--",
  };
}

function isTimerRunning({ activeStartedMs, busy, progressValue, task }) {
  const activeStarted = numericTime(activeStartedMs);
  const started = numericTime(task?.started_ms);
  const completed = numericTime(task?.completed_ms);

  return (
    activeStarted !== null ||
    busy ||
    (started !== null && completed === null) ||
    (progressValue > 0 && progressValue < 1)
  );
}

function hasTaskTiming(task, activeStartedMs) {
  return (
    numericTime(activeStartedMs) !== null ||
    numericTime(task?.started_ms) !== null ||
    numericTime(task?.completed_ms) !== null
  );
}

function hasCompletedTask(task) {
  return (
    numericTime(task?.started_ms) !== null &&
    numericTime(task?.completed_ms) !== null
  );
}

function taskLabel(task) {
  const name = String(task?.name ?? "").trim();
  return name || "Task";
}

function isIdleAtZero({ busy, progressValue, timerRunning }) {
  return !timerRunning && !busy && progressValue <= 0;
}

function taskDuration(task, now, activeStartedMs) {
  const completed = numericTime(task?.completed_ms);
  const activeStarted = numericTime(activeStartedMs);
  const started = numericTime(task?.started_ms);

  if (completed === null && activeStarted !== null) {
    return formatDuration(Math.max(0, now - activeStarted));
  }

  if (started !== null) {
    return formatDuration(Math.max(0, (completed ?? now) - started));
  }

  if (completed !== null && activeStarted !== null) {
    return formatDuration(Math.max(0, completed - activeStarted));
  }

  return formatDuration(0);
}

function durationFrom(startedMs, completedMs, now) {
  const started = numericTime(startedMs);
  if (started === null) {
    return formatDuration(0);
  }

  const completed = numericTime(completedMs);
  return formatDuration(Math.max(0, (completed ?? now) - started));
}

function numericTime(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function numericSeconds(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function normalizedProgress(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
}

function batchProgressValue(batch, jobProgress) {
  const index = Number(batch?.index) || 0;
  const total = Number(batch?.total) || 0;
  if (total <= 0 || index <= 0) {
    return 0;
  }

  const completedJobs = Math.max(0, Math.min(total, index - 1));
  return normalizedProgress((completedJobs + jobProgress) / total);
}

function eventKey(event, index) {
  const id = event?.id;
  if (id !== null && id !== undefined && id !== "") {
    return `event-${id}`;
  }

  return [
    "event",
    event?.time_ms ?? event?.time ?? "unknown",
    event?.stage ?? "status",
    event?.kind ?? "info",
    index,
  ].join("-");
}

function formatDuration(durationMs) {
  const safeMs = Math.max(0, Number(durationMs) || 0);
  const totalSeconds = safeMs / 1000;

  if (totalSeconds < 1) {
    return `${totalSeconds.toFixed(3)}s`;
  }

  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(2)}s`;
  }

  const roundedSeconds = Math.floor(totalSeconds);
  const hours = Math.floor(roundedSeconds / 3600);
  const minutes = Math.floor((roundedSeconds % 3600) / 60);
  const seconds = roundedSeconds % 60;
  const parts = [];

  if (hours) {
    parts.push(`${hours}h`);
  }

  if (minutes || hours) {
    parts.push(`${minutes}m`);
  }

  parts.push(`${seconds}s`);
  return parts.join(" ");
}
