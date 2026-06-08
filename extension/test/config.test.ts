import { describe, it, expect } from "vitest";
import { buildRequest, RawSettings } from "../src/config";

const base: RawSettings = {
  mode: "online",
  online: { baseUrl: "https://api.openai.com/v1", apiKey: "K", model: "gpt-4o-mini" },
  offline: { baseUrl: "http://localhost:8080/v1", model: "local-model" },
  params: { k: 5, temperatures: [0, 0.4, 0.8], lambda: 0.5, threshold: 0.5, maxRepairIters: 3 },
};

describe("buildRequest", () => {
  it("uses the online model block when mode=online", () => {
    const req = buildRequest("def foo; end", "ruby", base);
    expect(req.model.base_url).toBe("https://api.openai.com/v1");
    expect(req.model.api_key).toBe("K");
    expect(req.model.model).toBe("gpt-4o-mini");
    expect(req.params.k).toBe(5);
    expect(req.trace).toBe(true);
  });

  it("uses the offline model block when mode=offline", () => {
    const req = buildRequest("x", "ruby", { ...base, mode: "offline" });
    expect(req.model.base_url).toBe("http://localhost:8080/v1");
    expect(req.model.api_key).toBe("sk-no-key");
    expect(req.model.model).toBe("local-model");
  });
});
