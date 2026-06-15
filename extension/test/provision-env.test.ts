import { describe, it, expect } from "vitest";
import { buildBackendEnv } from "../src/provision";
import { provPaths } from "../src/paths";

describe("buildBackendEnv", () => {
  it("mirrors star_start.sh CS_* contract, localized to globalStorage", () => {
    const p = provPaths("/gs", "darwin");
    const env = buildBackendEnv(p, "cpu");
    expect(env.CS_PYTHON).toBe("/gs/venv/bin/python");
    expect(env.CS_EXTRACTOR_WEIGHTS).toBe("/gs/assets/extractor/pytorch_model.bin");
    expect(env.CS_CORPUS_PATH).toBe("/gs/assets/corpus/corpus_30k.jsonl");
    expect(env.CS_CORPUS_LIMIT).toBe("30000");
    expect(env.CS_DEVICE).toBe("cpu");
    expect(env.HF_HOME).toBe("/gs/hf_cache");
    // CodeBERT path is left unset so transformers resolves the HF id via HF_HOME.
    expect(env.CS_CODEBERT_PATH).toBeUndefined();
  });
});
