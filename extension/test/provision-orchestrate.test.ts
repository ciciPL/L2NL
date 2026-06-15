import { describe, it, expect } from "vitest";
import { provision, ProvDeps, ProvOptions } from "../src/provision";
import { provPaths } from "../src/paths";

function fakeDeps(record: string[]): ProvDeps {
  return {
    run: async (cmd, args = []) => { record.push(`run ${cmd} ${args.join(" ")}`); return { code: 0, stdout: "ok", stderr: "" }; },
    spawnBackend: (_python, _args, _opts) => { record.push("spawn"); return { pid: 123 }; },
    download: async (_url, dest) => { record.push(`download ${dest}`); },
    verify: async () => true,            // pretend assets are already correct
    health: async () => true,            // backend healthy immediately
    mkdirp: (d) => record.push(`mkdir ${d}`),
    readState: () => null,               // fresh install
    writeState: () => record.push("writeState"),
    onStep: (s, st) => record.push(`step ${s}:${st}`),
  };
}

const opts: ProvOptions = {
  device: "cpu", extVersion: "0.2.0", reqHash: "abc",
  hostPython: ["python3.11"],
  requirementsPath: "/ext/backend/requirements.txt", backendCwd: "/ext/backend",
  baseUrlOverride: undefined,
  manifest: {
    release: "v0.1-assets", baseUrl: "https://x/",
    assets: [
      { name: "extractor/pytorch_model.bin", file: "pytorch_model.bin", sha256: "h1", size: 1 },
      { name: "corpus/corpus_30k.jsonl", file: "corpus_30k.jsonl", sha256: "h2", size: 1 },
    ],
  },
};

describe("provision (fresh install)", () => {
  it("runs venv, torch(cpu index), deps, codebert, then launches and reports a url", async () => {
    const rec: string[] = [];
    const res = await provision(provPaths("/gs", "darwin"), opts, fakeDeps(rec));
    expect(res.backendUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    const joined = rec.join("\n");
    expect(joined).toContain("-m venv");
    expect(joined).toContain("download.pytorch.org/whl/cpu");
    expect(joined).toContain("-r /ext/backend/requirements.txt");
    expect(joined).toContain("spawn");
    expect(rec).toContain("writeState");
    expect(joined).not.toContain("download ");  // assets already correct -> no download
  });

  it("uses the cuda wheel index when device=cuda", async () => {
    const rec: string[] = [];
    await provision(provPaths("/gs", "darwin"), { ...opts, device: "cuda" }, fakeDeps(rec));
    expect(rec.join("\n")).toContain("download.pytorch.org/whl/cu121");
  });

  it("skips install/download steps when prior state matches (idempotent)", async () => {
    const rec: string[] = [];
    const deps: ProvDeps = {
      ...fakeDeps(rec),
      readState: () => ({
        schemaVersion: 1, extVersion: "0.2.0", device: "cpu", reqHash: "abc",
        assets: { "extractor/pytorch_model.bin": "h1", "corpus/corpus_30k.jsonl": "h2" },
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
