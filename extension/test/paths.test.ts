import { describe, it, expect } from "vitest";
import { provPaths } from "../src/paths";

describe("provPaths", () => {
  it("lays out posix paths under the storage root", () => {
    const p = provPaths("/gs", "darwin");
    expect(p.venvPython).toBe("/gs/venv/bin/python");
    expect(p.extractorWeights).toBe("/gs/assets/extractor/pytorch_model.bin");
    expect(p.corpus).toBe("/gs/assets/corpus/corpus_30k.jsonl");
    expect(p.hfCache).toBe("/gs/hf_cache");
    expect(p.state).toBe("/gs/state.json");
  });

  it("uses Scripts/python.exe on win32", () => {
    const p = provPaths("C:\\gs", "win32");
    expect(p.venvPython).toBe("C:\\gs\\venv\\Scripts\\python.exe");
    expect(p.extractorWeights).toBe("C:\\gs\\assets\\extractor\\pytorch_model.bin");
  });
});
