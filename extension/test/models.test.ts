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

import { mergeModelConfig } from "../src/models";

describe("mergeModelConfig", () => {
  const existing = { mode: "online", online: { baseUrl: "u", apiKey: "OLDKEY", model: "m" }, offline: { baseUrl: "lu", model: "lm" } };

  it("keeps the existing online key when the form key is blank", () => {
    const out = mergeModelConfig(existing, { mode: "online", base_url: "u2", api_key: "", model: "m2" });
    expect(out.online.apiKey).toBe("OLDKEY");
    expect(out.online.baseUrl).toBe("u2");
    expect(out.online.model).toBe("m2");
  });
  it("overwrites the key when the form provides one", () => {
    const out = mergeModelConfig(existing, { mode: "online", base_url: "u", api_key: "NEW", model: "m" });
    expect(out.online.apiKey).toBe("NEW");
  });
  it("writes offline fields and switches mode without touching the key", () => {
    const out = mergeModelConfig(existing, { mode: "offline", base_url: "http://local/v1", api_key: "", model: "lc" });
    expect(out.mode).toBe("offline");
    expect(out.offline.baseUrl).toBe("http://local/v1");
    expect(out.offline.model).toBe("lc");
    expect(out.online.apiKey).toBe("OLDKEY");
  });
});
