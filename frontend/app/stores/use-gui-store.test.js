import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_STATUS,
  buildBatchPayload,
  createGuiStore,
} from "./use-gui-store";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, reject, resolve };
}

function jsonResponse(payload) {
  return {
    ok: true,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => payload,
  };
}

describe("createGuiStore polling", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("does not reschedule an in-flight poll after stopPolling", async () => {
    vi.useFakeTimers();

    const statusRequest = deferred();
    const fetchImpl = vi.fn(() => statusRequest.promise);
    const store = createGuiStore({
      fetchImpl,
      initialStatus: DEFAULT_STATUS,
    });

    store.getState().startPolling();

    expect(fetchImpl).toHaveBeenCalledTimes(1);

    store.getState().stopPolling();
    statusRequest.resolve(jsonResponse(DEFAULT_STATUS));
    await statusRequest.promise;
    await vi.runAllTimersAsync();

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("batch payload constraints", () => {
  it("uses runtime status constraints when filling omitted batch defaults", () => {
    const payload = buildBatchPayload('[{"prompt":"runtime defaults"}]', {
      constraints: {
        default_width: 640,
        default_height: 832,
        default_steps: 12,
      },
    });

    expect(payload.jobs[0]).toMatchObject({
      prompt: "runtime defaults",
      width: 640,
      height: 832,
      steps: 12,
    });
  });
});
