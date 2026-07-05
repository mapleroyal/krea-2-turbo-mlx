import { useEffect, useRef, useState } from "react";
import ArrowExpand01Icon from "@hugeicons/core-free-icons/ArrowExpand01Icon";
import ArrowShrink01Icon from "@hugeicons/core-free-icons/ArrowShrink01Icon";
import ArrowUp02Icon from "@hugeicons/core-free-icons/ArrowUp02Icon";
import Cancel01Icon from "@hugeicons/core-free-icons/Cancel01Icon";
import FileRemoveIcon from "@hugeicons/core-free-icons/FileRemoveIcon";
import PlayListRemoveIcon from "@hugeicons/core-free-icons/PlayListRemoveIcon";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  BATCH_JOB_STATUSES,
  DESTRUCTIVE_BATCH_JOB_STATUSES,
  RUNNING_BATCH_JOB_STATUSES,
  SETTLED_BATCH_JOB_STATUSES,
  normalizeBatchJobStatus,
} from "@/lib/status";
import { cn } from "@/lib/utils";
import { formatMeta, modelVariantLabel } from "@/lib/workspace";

import { ATTACHED_CONTROL_SHADOW } from "./attached-control-stack";
import { IconActionButton } from "./icon-action-button";

export function BatchStatus({
  batch,
  cancelRequested = false,
  canResubmit = true,
  className,
  completed = false,
  expanded: expandedProp,
  fallbackPrecision,
  onCancel,
  onCancelCurrent,
  onClear,
  onClearQueue,
  onExpandedChange,
  onResubmit,
}) {
  const [internalExpanded, setInternalExpanded] = useState(false);
  const expanded = expandedProp ?? internalExpanded;
  const activeJobRef = useRef(null);
  const queueScrollRootRef = useRef(null);
  const wasExpandedRef = useRef(false);

  useEffect(() => {
    if (expanded && !wasExpandedRef.current) {
      scrollQueueToActiveJob(queueScrollRootRef.current, activeJobRef.current);
    }

    wasExpandedRef.current = expanded;
  }, [expanded]);

  if (!batch) {
    return null;
  }

  const jobs = batchJobs(batch);
  const activeIndex = completed ? null : numericJobIndex(batch.index);
  const showJobList = expanded || completed;
  const sectionLabel = completed ? "Completed batch" : "Batch generation";
  const actionPadding = completed ? "pr-24" : "pr-14";
  const showDerivedFieldsHelp = completed && jobs.some(hasGuiDerivedFields);
  const cancelCurrentRequested = Boolean(
    cancelRequested || batch.cancel_current_requested,
  );
  const clearQueueRequested = Boolean(batch.clear_queue_requested);
  const hasQueuedJobs = jobs.some(
    (job, index) =>
      batchJobStatus(
        job,
        numericJobIndex(job.index) ?? index + 1,
        activeIndex,
        completed,
      ) === BATCH_JOB_STATUSES.QUEUED,
  );
  const hasRunningJob = jobs.some((job, index) =>
    RUNNING_BATCH_JOB_STATUSES.includes(
      batchJobStatus(
        job,
        numericJobIndex(job.index) ?? index + 1,
        activeIndex,
        completed,
      ),
    ),
  );
  const handleExpandedChange = () => {
    const nextExpanded = !expanded;
    setInternalExpanded(nextExpanded);
    onExpandedChange?.(nextExpanded);
  };

  return (
    <section
      aria-label={sectionLabel}
      className={cn(
        "relative z-10 overflow-hidden rounded-lg border border-border bg-card",
        expanded ? "h-[50dvh] min-h-[220px]" : "min-h-[112px]",
        ATTACHED_CONTROL_SHADOW,
        className,
      )}
    >
      <div className="absolute top-3 right-3 z-10">
        <IconActionButton
          ariaLabel={expanded ? "Collapse batch jobs" : "Expand batch jobs"}
          expanded={expanded}
          icon={expanded ? ArrowShrink01Icon : ArrowExpand01Icon}
          onClick={handleExpandedChange}
          tooltip={expanded ? "Collapse batch jobs" : "Expand batch jobs"}
        />
      </div>

      {showJobList ? (
        <div
          ref={queueScrollRootRef}
          className={cn(
            expanded
              ? "flex h-full flex-col"
              : "max-h-[min(34dvh,22rem)] overflow-y-auto",
          )}
        >
          {showDerivedFieldsHelp && (
            <p className="border-b border-border px-4 py-2 pr-24 text-label-sm text-muted-foreground">
              * Reruns use the current value from the GUI control.
            </p>
          )}
          {expanded ? (
            <ScrollArea className="min-h-0 flex-1">
              <div role="list">
                {jobs.map((job, index) => {
                  const jobIndex = numericJobIndex(job.index) ?? index + 1;
                  const active = !completed && activeIndex === jobIndex;
                  const status = batchJobStatus(
                    job,
                    jobIndex,
                    activeIndex,
                    completed,
                  );

                  return (
                    <BatchJobRow
                      active={active}
                      activeIndex={activeIndex}
                      completed={completed}
                      fallbackPrecision={fallbackPrecision}
                      job={job}
                      jobIndex={jobIndex}
                      key={batchJobKey(job, jobIndex)}
                      paddingClassName={actionPadding}
                      rowRef={active ? activeJobRef : undefined}
                      status={status}
                      total={batch.total}
                    />
                  );
                })}
              </div>
            </ScrollArea>
          ) : (
            <div role="list">
              {jobs.map((job, index) => {
                const jobIndex = numericJobIndex(job.index) ?? index + 1;
                const status = batchJobStatus(
                  job,
                  jobIndex,
                  activeIndex,
                  completed,
                );

                return (
                  <BatchJobRow
                    active={false}
                    activeIndex={activeIndex}
                    completed={completed}
                    fallbackPrecision={fallbackPrecision}
                    job={job}
                    jobIndex={jobIndex}
                    key={batchJobKey(job, jobIndex)}
                    paddingClassName={actionPadding}
                    status={status}
                    total={batch.total}
                  />
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <div className={cn("px-4 py-3.5", actionPadding)}>
          <BatchJobDetails
            fallbackPrecision={fallbackPrecision}
            job={batch}
            jobIndex={batch.index}
            status={batchJobStatus(batch, batch.index, activeIndex, completed)}
            total={batch.total}
          />
        </div>
      )}

      {completed ? (
        <CompletedBatchActions
          canResubmit={canResubmit}
          onClear={onClear}
          onResubmit={onResubmit}
        />
      ) : (
        <div className="absolute right-3 bottom-3 z-10">
          <ActiveBatchActions
            cancelCurrentRequested={cancelCurrentRequested}
            clearQueueRequested={clearQueueRequested}
            hasQueuedJobs={hasQueuedJobs}
            hasRunningJob={hasRunningJob}
            onCancelCurrent={onCancelCurrent ?? onCancel}
            onClearQueue={onClearQueue}
          />
        </div>
      )}
    </section>
  );
}

function BatchJobRow({
  active,
  activeIndex,
  completed,
  fallbackPrecision,
  job,
  jobIndex,
  paddingClassName,
  rowRef,
  status,
  total,
}) {
  return (
    <article
      ref={rowRef}
      aria-current={active ? "step" : undefined}
      role="listitem"
      className={cn(
        "relative border-t border-border px-4 py-3.5 first:border-t-0",
        paddingClassName,
        active && "bg-muted/50",
        isSettledJob(jobIndex, activeIndex, completed, status) &&
          !active &&
          "opacity-70",
      )}
    >
      <BatchJobDetails
        active={active}
        activeIndex={activeIndex}
        completed={completed}
        fallbackPrecision={fallbackPrecision}
        job={job}
        jobIndex={jobIndex}
        showState
        status={status}
        total={total}
      />
    </article>
  );
}

function BatchJobDetails({
  active,
  activeIndex,
  completed,
  fallbackPrecision,
  job,
  jobIndex,
  showState = false,
  status,
  total,
}) {
  const meta = batchJobMeta(job, fallbackPrecision);
  const stateLabel = batchJobState(
    jobIndex,
    activeIndex,
    active,
    completed,
    status,
  );
  const stateDestructive = DESTRUCTIVE_BATCH_JOB_STATUSES.includes(status);

  return (
    <div className="min-w-0">
      <div className="flex min-w-0 items-center gap-2">
        {showState && active && (
          <span
            aria-hidden="true"
            className="size-2 shrink-0 rounded-full bg-primary shadow-[0_0_0_3px_color-mix(in_oklab,var(--primary)_18%,transparent)] motion-safe:animate-pulse"
          />
        )}
        <h2 className="truncate text-title-md">
          Job {jobIndex || "?"} of {total || "?"}
        </h2>
        {showState && (
          <span
            className={cn(
              "shrink-0 text-label-sm",
              stateDestructive
                ? "text-destructive"
                : active && !completed
                  ? "text-primary"
                  : "text-muted-foreground",
            )}
          >
            {stateLabel}
          </span>
        )}
      </div>
      <p
        data-slot="batch-job-prompt"
        className="mt-1 whitespace-pre-wrap break-words text-body-md text-foreground/80"
      >
        {job.prompt || "Preparing next job"}
      </p>
      <p className="mt-1 whitespace-pre-line text-body-sm text-muted-foreground">
        {meta}
      </p>
    </div>
  );
}

function ActiveBatchActions({
  cancelCurrentRequested,
  clearQueueRequested,
  hasQueuedJobs,
  hasRunningJob,
  onCancelCurrent,
  onClearQueue,
}) {
  const cancelDisabled = cancelCurrentRequested || !hasRunningJob;
  const clearDisabled = clearQueueRequested || !hasQueuedJobs;

  return (
    <ButtonGroup orientation="vertical" className="shadow-sm">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            type="button"
            variant="destructive"
            size="icon-sm"
            aria-label={
              cancelCurrentRequested
                ? "Cancelling current job"
                : "Cancel current job"
            }
            disabled={cancelDisabled}
            onClick={cancelDisabled ? undefined : onCancelCurrent}
          >
            {cancelCurrentRequested ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {cancelCurrentRequested
            ? "Cancelling current job"
            : "Cancel current job"}
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            type="button"
            variant="destructive"
            size="icon-sm"
            aria-label={clearQueueRequested ? "Queue cleared" : "Clear queue"}
            disabled={clearDisabled}
            onClick={clearDisabled ? undefined : onClearQueue}
          >
            <HugeiconsIcon icon={PlayListRemoveIcon} strokeWidth={2} />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {clearQueueRequested ? "Queue cleared" : "Clear queue"}
        </TooltipContent>
      </Tooltip>
    </ButtonGroup>
  );
}

function CompletedBatchActions({ canResubmit, onClear, onResubmit }) {
  return (
    <div className="absolute right-3 bottom-3 z-10 flex items-center gap-2">
      <IconActionButton
        ariaLabel="Clear batch report"
        icon={FileRemoveIcon}
        onClick={onClear}
        tooltip="Clear batch report"
      />
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            type="button"
            variant="default"
            size="icon-sm"
            aria-label="Rerun batch"
            className="shadow-sm"
            disabled={!canResubmit}
            onClick={onResubmit}
          >
            <HugeiconsIcon icon={ArrowUp02Icon} strokeWidth={2.5} />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Rerun batch</TooltipContent>
      </Tooltip>
    </div>
  );
}

function batchJobMeta(job, fallbackPrecision) {
  return formatMeta({
    derivedFields: job.guiDerivedFields,
    variant: modelVariantLabel(job, fallbackPrecision),
    width: job.width ?? "-",
    height: job.height ?? "-",
    loras: job.loras,
    seed: job.seed ?? "-",
    steps: job.steps ?? "-",
  });
}

function hasGuiDerivedFields(job) {
  return (
    Array.isArray(job?.guiDerivedFields) && job.guiDerivedFields.length > 0
  );
}

function scrollQueueToActiveJob(root, activeJob) {
  const viewport = root?.querySelector?.('[data-slot="scroll-area-viewport"]');
  if (!viewport || !activeJob) {
    return;
  }

  const viewportRect = viewport.getBoundingClientRect();
  const activeRect = activeJob.getBoundingClientRect();
  viewport.scrollTop += activeRect.top - viewportRect.top;
}

function batchJobs(batch) {
  if (Array.isArray(batch.jobs) && batch.jobs.length > 0) {
    return batch.jobs.map((job, index) => ({
      ...job,
      index: numericJobIndex(job?.index) ?? index + 1,
    }));
  }

  return [batch];
}

function numericJobIndex(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function isSettledJob(jobIndex, activeIndex, completed, status) {
  return (
    completed ||
    SETTLED_BATCH_JOB_STATUSES.includes(status) ||
    (activeIndex !== null && jobIndex < activeIndex)
  );
}

function batchJobState(jobIndex, activeIndex, active, completed, status) {
  if (status === BATCH_JOB_STATUSES.CANCELLING) {
    return "Cancelling";
  }
  if (status === BATCH_JOB_STATUSES.CANCELLED) {
    return "Cancelled";
  }
  if (status === BATCH_JOB_STATUSES.CLEARED) {
    return "Cleared";
  }
  if (status === BATCH_JOB_STATUSES.DONE) {
    return "Done";
  }

  if (completed) {
    return "Done";
  }

  if (active || activeIndex === jobIndex) {
    return "Running";
  }

  if (activeIndex !== null && jobIndex < activeIndex) {
    return "Done";
  }

  return "Queued";
}

function batchJobStatus(job, jobIndex, activeIndex, completed) {
  const status = normalizeBatchJobStatus(job?.status);
  if (status) {
    return status;
  }
  if (completed) {
    return BATCH_JOB_STATUSES.DONE;
  }
  if (activeIndex !== null && jobIndex < activeIndex) {
    return BATCH_JOB_STATUSES.DONE;
  }
  if (activeIndex === jobIndex) {
    return BATCH_JOB_STATUSES.RUNNING;
  }
  return BATCH_JOB_STATUSES.QUEUED;
}

function batchJobKey(job, jobIndex) {
  return [
    "batch-job",
    jobIndex,
    job?.seed ?? "seed",
    job?.width ?? "width",
    job?.height ?? "height",
    job?.steps ?? "steps",
    job?.prompt ?? "prompt",
  ].join("-");
}
