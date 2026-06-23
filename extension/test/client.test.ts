import { afterEach, describe, expect, it, vi } from "vitest";
import { getHealth } from "../src/client";

describe("getHealth", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requires /health.ready=true", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({ status: "ok", ready: false }),
    })));
    await expect(getHealth("http://127.0.0.1:8000")).resolves.toBe(false);
  });

  it("returns true for a ready backend", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({ status: "ok", ready: true }),
    })));
    await expect(getHealth("http://127.0.0.1:8000")).resolves.toBe(true);
  });
});
