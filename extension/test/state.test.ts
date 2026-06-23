import { describe, it, expect } from "vitest";
import { stepsNeeded, ProvState, ProvInputs } from "../src/state";

const inputs: ProvInputs = {
  extVersion: "0.2.0", device: "cpu", reqHash: "abc",
  assets: {
    "extractor/pytorch_model.bin": "h1",
    "corpus/corpus_30k.jsonl": "h2",
    "codebert-base": "h3",
  },
};

describe("stepsNeeded", () => {
  it("needs every step when there is no prior state", () => {
    expect([...stepsNeeded(null, inputs)].sort())
      .toEqual(["assets", "codebert", "deps", "launch", "torch", "venv"]);
  });

  it("needs nothing but launch when state matches", () => {
    const prev: ProvState = { schemaVersion: 1, ...inputs, codebertReady: true };
    expect([...stepsNeeded(prev, inputs)]).toEqual(["launch"]);
  });

  it("re-runs torch (and deps) when device changes", () => {
    const prev: ProvState = { schemaVersion: 1, ...inputs, device: "cuda", codebertReady: true };
    const s = stepsNeeded(prev, inputs);
    expect(s.has("torch")).toBe(true);
    expect(s.has("deps")).toBe(true);
    expect(s.has("assets")).toBe(false);
  });

  it("re-downloads only the asset whose hash changed", () => {
    const prev: ProvState = {
      schemaVersion: 1, ...inputs, codebertReady: true,
      assets: { "extractor/pytorch_model.bin": "h1", "corpus/corpus_30k.jsonl": "OLD" },
    };
    const s = stepsNeeded(prev, inputs);
    expect(s.has("assets")).toBe(true);
    expect(s.has("torch")).toBe(false);
  });

  it("re-runs codebert when prior run did not finish it", () => {
    const prev: ProvState = { schemaVersion: 1, ...inputs, codebertReady: false };
    expect(stepsNeeded(prev, inputs).has("codebert")).toBe(true);
  });

  it("re-runs codebert when the CodeBERT archive hash changes", () => {
    const prev: ProvState = {
      schemaVersion: 1, ...inputs, codebertReady: true,
      assets: { ...inputs.assets, "codebert-base": "OLD" },
    };
    const s = stepsNeeded(prev, inputs);
    expect(s.has("assets")).toBe(true);
    expect(s.has("codebert")).toBe(true);
  });
});
