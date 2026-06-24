import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { buildRequest, RawSettings } from "../src/config";

const base: RawSettings = {
  mode: "online",
  online: { baseUrl: "https://api.openai.com/v1", apiKey: "K", model: "gpt-4o-mini" },
  offline: { baseUrl: "http://localhost:8080/v1", model: "local-model" },
  params: { k: 3, temperatures: [0, 0.4, 0.8], lambda: 0.5, threshold: 0.5, maxRepairIters: 3 },
};

describe("buildRequest", () => {
  it("uses the online model block when mode=online", () => {
    const req = buildRequest("def foo; end", "ruby", base);
    expect(req.model.base_url).toBe("https://api.openai.com/v1");
    expect(req.model.api_key).toBe("K");
    expect(req.model.model).toBe("gpt-4o-mini");
    expect(req.params.k).toBe(3);
    expect(req.trace).toBe(true);
  });

  it("uses the offline model block when mode=offline", () => {
    const req = buildRequest("x", "ruby", { ...base, mode: "offline" });
    expect(req.model.base_url).toBe("http://localhost:8080/v1");
    expect(req.model.api_key).toBe("sk-no-key");
    expect(req.model.model).toBe("local-model");
  });

  it("defaults extension interaction to one translation candidate for latency", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(pkg.contributes.configuration.properties["codeSummary.params.temperatures"].default).toEqual([0]);
  });

  it("defaults retrieval to the paper Top-3 setting", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(pkg.contributes.configuration.properties["codeSummary.params.k"].default).toBe(3);

    const extensionSource = readFileSync(new URL("../src/extension.ts", import.meta.url), "utf8");
    expect(extensionSource).toContain('k: c.get("params.k", 3)');
  });

  it("contributes customer-visible commands for online and offline configuration", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    const titles = Object.fromEntries(
      pkg.contributes.commands.map((c: { command: string; title: string }) => [c.command, c.title]),
    );

    expect(titles["codeSummary.configureModel"]).toBe("Code Summary: Configure Model");
    expect(titles["codeSummary.configureOnline"]).toBe("Code Summary: Configure Online API");
    expect(titles["codeSummary.configureOffline"]).toBe("Code Summary: Configure Offline Local Model");

    const extensionSource = readFileSync(new URL("../src/extension.ts", import.meta.url), "utf8");
    expect(extensionSource).toContain('statusItem.command = "codeSummary.configureModel"');
    expect(extensionSource).toContain('WizardPanel.open(ctx, { initialMode: "offline", initialStep: "model" })');
  });
});
