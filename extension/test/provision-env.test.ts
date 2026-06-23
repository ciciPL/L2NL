import { describe, it, expect } from "vitest";
import { buildBackendEnv } from "../src/provision";
import { provPaths } from "../src/paths";

describe("buildBackendEnv", () => {
  it("mirrors star_start.sh CS_* contract, localized to globalStorage", () => {
    const p = provPaths("/gs", "darwin");
    const env = buildBackendEnv(p, { device: "cpu", allowStubs: false });
    expect(env.CS_PYTHON).toBe("/gs/venv/bin/python");
    expect(env.CS_EXTRACTOR_WEIGHTS).toBe("/gs/assets/extractor/pytorch_model.bin");
    expect(env.CS_CORPUS_PATH).toBe("/gs/assets/corpus/corpus_30k.jsonl");
    expect(env.CS_CORPUS_LIMIT).toBe("30000");
    expect(env.CS_DEVICE).toBe("cpu");
    expect(env.HF_HOME).toBe("/gs/hf_cache");
    expect(env.CS_CODEBERT_PATH).toBe("/gs/assets/codebert-base");
    expect(env.CS_ASSETS_MANIFEST).toBe("/gs/assets/assets-manifest.v2.json");
    expect(env.CS_ALLOW_STUBS).toBe("0");
    expect(env.NO_PROXY).toContain("127.0.0.1");
    expect(env.no_proxy).toContain("localhost");
  });
});
