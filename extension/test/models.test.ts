import { describe, it, expect } from "vitest";
import { CLOUD_VENDORS, LOCAL_PRESETS, buildTestRequest } from "../src/models";

describe("presets", () => {
  it("offers DeepSeek and OpenAI cloud vendors with base URLs", () => {
    const ids = CLOUD_VENDORS.map((v) => v.id);
    expect(ids).toContain("deepseek");
    expect(ids).toContain("openai");
    expect(CLOUD_VENDORS.find((v) => v.id === "deepseek")!.baseUrl)
      .toBe("https://api.deepseek.com/v1");
  });
  it("offers local runtimes that do not need a key", () => {
    expect(LOCAL_PRESETS.find((v) => v.id === "ollama")!.needsKey).toBe(false);
  });
});

describe("buildTestRequest", () => {
  it("builds an OpenAI-compatible chat/completions probe with auth when keyed", () => {
    const r = buildTestRequest({ base_url: "https://api.deepseek.com/v1/", api_key: "K", model: "deepseek-chat" });
    expect(r.url).toBe("https://api.deepseek.com/v1/chat/completions");
    expect(r.headers.Authorization).toBe("Bearer K");
    expect(JSON.parse(r.body).model).toBe("deepseek-chat");
    expect(JSON.parse(r.body).max_tokens).toBe(1);
  });
  it("omits Authorization when there is no key (local runtimes)", () => {
    const r = buildTestRequest({ base_url: "http://localhost:11434/v1", api_key: "", model: "x" });
    expect(r.headers.Authorization).toBeUndefined();
  });
});
