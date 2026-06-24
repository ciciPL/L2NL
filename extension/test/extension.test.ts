import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function extensionSource(): string {
  return readFileSync(new URL("../src/extension.ts", import.meta.url), "utf8");
}

describe("extension offline runtime lifecycle", () => {
  it("sets NO_PROXY before extension health checks", () => {
    const src = extensionSource();
    expect(src).toContain("function ensureLocalNoProxy()");
    expect(src).toContain('process.env.NO_PROXY = mergeNoProxy(process.env.NO_PROXY)');
    expect(src).toContain('process.env.no_proxy = mergeNoProxy(process.env.no_proxy)');
    expect(src).toContain("ensureLocalNoProxy();");
  });

  it("restarts the configured local GGUF runtime before offline summarize", () => {
    const src = extensionSource();
    expect(src).toContain("async function ensureOfflineRuntime");
    expect(src).toContain('cfg.get("offline.modelPath"');
    expect(src).toContain("testConnection({ base_url: baseUrl");
    expect(src).toContain("prepareLocalGguf(ctx, modelPath");
    expect(src).toContain("if (!(await ensureOfflineRuntime(ctx))) return;");
  });
});
