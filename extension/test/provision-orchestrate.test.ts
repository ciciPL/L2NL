import { describe, it, expect } from "vitest";
import { provision, ProvDeps, ProvOptions } from "../src/provision";
import { provPaths } from "../src/paths";

function fakeDeps(record: string[]): ProvDeps {
  return {
    run: async (cmd, args = []) => { record.push(`run ${cmd} ${args.join(" ")}`); return { code: 0, stdout: "ok", stderr: "" }; },
    spawnBackend: (_python, _args, opts) => { record.push(`spawn NO_PROXY=${opts.env.NO_PROXY} CODEBERT=${opts.env.CS_CODEBERT_PATH}`); return { pid: 123 }; },
    download: async (_url, dest) => { record.push(`download ${dest}`); },
    verify: async () => true,            // pretend assets are already correct
    health: async () => true,            // backend healthy immediately
    mkdirp: (d) => record.push(`mkdir ${d}`),
    writeManifest: (p) => record.push(`writeManifest ${p}`),
    readState: () => null,               // fresh install
    writeState: () => record.push("writeState"),
    onStep: (s, st) => record.push(`step ${s}:${st}`),
  };
}

const opts: ProvOptions = {
  device: "cpu", extVersion: "0.2.0", reqHash: "abc",
  hostPython: ["python3.11"],
  pythonIndexUrl: "https://pypi.tuna.tsinghua.edu.cn/simple",
  requirementsPath: "/ext/backend/requirements.txt", backendCwd: "/ext/backend",
    baseUrlOverride: undefined,
    localAssetDir: undefined,
    includeGlobalMirrors: false,
    allowStubs: false,
    manifest: {
    version: 2,
    release: "v0.2-assets-cn",
    sources: [{ id: "gitee", baseUrl: "https://gitee.example/releases/v0.2-assets-cn", enabledByDefault: true }],
    assets: [
      { name: "extractor/pytorch_model.bin", file: "pytorch_model.bin", sha256: "h1", size: 1 },
      { name: "corpus/corpus_30k.jsonl", file: "corpus_30k.jsonl", sha256: "h2", size: 1 },
      {
        name: "codebert-base",
        target: "codebert-base.tar.gz",
        sha256: "h3",
        size: 1,
        extractTo: "codebert-base",
        parts: [{ file: "codebert-base.tar.gz.part001", sha256: "p3", size: 1 }],
      },
    ],
  },
};

describe("provision (fresh install)", () => {
  it("runs venv, torch/domestic deps, codebert, then launches and reports a url", async () => {
    const rec: string[] = [];
    const res = await provision(provPaths("/gs", "darwin"), opts, fakeDeps(rec));
    expect(res.backendUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    const joined = rec.join("\n");
    expect(joined).toContain("-m venv");
    expect(joined).toContain("--retries 5 --timeout 60 -i https://pypi.tuna.tsinghua.edu.cn/simple");
    expect(joined).toContain("-m pip install --retries 5 --timeout 60 -i https://pypi.tuna.tsinghua.edu.cn/simple torch");
    expect(joined).toContain("-r /ext/backend/requirements.txt");
    expect(joined).toContain("tar -xf /gs/assets/codebert-base.tar.gz -C /gs/assets");
    expect(joined).toContain("spawn NO_PROXY=127.0.0.1,localhost,::1 CODEBERT=/gs/assets/codebert-base");
    expect(rec).toContain("writeState");
    expect(joined).not.toContain("download ");  // assets already correct -> no download
    expect(joined).not.toContain("download.pytorch.org");
    expect(joined).not.toContain("AutoModel.from_pretrained");
  });

  it("uses a caller-provided pip mirror instead of PyTorch wheel indexes", async () => {
    const rec: string[] = [];
    await provision(provPaths("/gs", "darwin"), {
      ...opts,
      device: "cuda",
      pythonIndexUrl: "https://mirrors.aliyun.com/pypi/simple",
    }, fakeDeps(rec));
    const joined = rec.join("\n");
    expect(joined).toContain("-i https://mirrors.aliyun.com/pypi/simple torch");
    expect(joined).not.toContain("download.pytorch.org");
  });

  it("skips install/download steps when prior state matches (idempotent)", async () => {
    const rec: string[] = [];
    const deps: ProvDeps = {
      ...fakeDeps(rec),
      readState: () => ({
        schemaVersion: 1, extVersion: "0.2.0", device: "cpu", reqHash: "abc",
        assets: { "extractor/pytorch_model.bin": "h1", "corpus/corpus_30k.jsonl": "h2", "codebert-base": "h3" },
        codebertReady: true,
      }),
    };
    await provision(provPaths("/gs", "darwin"), opts, deps);
    const joined = rec.join("\n");
    expect(joined).not.toContain("-m venv");
    expect(joined).not.toContain("pip install");
    expect(joined).toContain("spawn");        // launch always runs
    expect(rec).toContain("writeState");
  });

  it("throws and reports the failing step when a command exits non-zero", async () => {
    const rec: string[] = [];
    const deps: ProvDeps = {
      ...fakeDeps(rec),
      run: async (cmd, args = []) => {
        if (`${cmd} ${args.join(" ")}`.includes("-m venv")) {
          return { code: 1, stdout: "", stderr: "venv boom" };
        }
        return { code: 0, stdout: "ok", stderr: "" };
      },
    };
    await expect(provision(provPaths("/gs", "darwin"), opts, deps)).rejects.toThrow(/exited 1/);
    expect(rec.join("\n")).toContain("step venv:error");
  });
});
