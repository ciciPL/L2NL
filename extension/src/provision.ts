import * as net from "node:net";
import { ProvPaths } from "./paths";
import { Manifest, AssetEntry, assetUrl } from "./assets";
import { ProvState, ProvInputs, Step, stepsNeeded } from "./state";
import { Runner } from "./env";

export function buildBackendEnv(p: ProvPaths, device: string): Record<string, string> {
  return {
    CS_PYTHON: p.venvPython,
    CS_EXTRACTOR_WEIGHTS: p.extractorWeights,
    CS_CORPUS_PATH: p.corpus,
    CS_CORPUS_LIMIT: "30000",
    CS_DEVICE: device,
    HF_HOME: p.hfCache,
  };
}

export function findFreePort(start = 8000, tries = 20): Promise<number> {
  function check(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const srv = net.createServer();
      srv.once("error", () => resolve(false));
      srv.once("listening", () => srv.close(() => resolve(true)));
      srv.listen(port, "127.0.0.1");
    });
  }
  return (async () => {
    for (let i = 0; i < tries; i++) {
      if (await check(start + i)) return start + i;
    }
    throw new Error(`no free port in ${start}..${start + tries}`);
  })();
}

export type StepStatus = "running" | "done" | "skipped" | "error";

export interface ProvDeps {
  run: Runner;
  spawnBackend: (python: string, args: string[], opts: { cwd: string; env: Record<string, string>; logPath: string }) => { pid?: number };
  download: (url: string, dest: string, onBytes?: (d: number, t: number) => void) => Promise<void>;
  verify: (path: string, a: AssetEntry) => Promise<boolean>;
  health: (url: string) => Promise<boolean>;
  mkdirp: (dir: string) => void;
  readState: () => ProvState | null;
  writeState: (inputs: ProvInputs, codebertReady: boolean) => void;
  onStep: (step: Step, status: StepStatus, detail?: string) => void;
}

export interface ProvOptions {
  device: string;
  extVersion: string;
  reqHash: string;
  hostPython: string;
  requirementsPath: string;
  backendCwd: string;
  baseUrlOverride?: string;
  manifest: Manifest;
}

const TORCH_INDEX: Record<string, string> = {
  cpu: "https://download.pytorch.org/whl/cpu",
  cuda: "https://download.pytorch.org/whl/cu121",
};

function assetHashes(m: Manifest): Record<string, string> {
  const out: Record<string, string> = {};
  for (const a of m.assets) out[a.name] = a.sha256;
  return out;
}

export async function provision(
  p: ProvPaths, opts: ProvOptions, deps: ProvDeps,
): Promise<{ backendUrl: string }> {
  const inputs: ProvInputs = {
    extVersion: opts.extVersion, device: opts.device, reqHash: opts.reqHash,
    assets: assetHashes(opts.manifest),
  };
  const need = stepsNeeded(deps.readState(), inputs);
  const did = (s: Step) => deps.onStep(s, need.has(s) ? "done" : "skipped");
  let codebertReady = !need.has("codebert");
  let current: Step = "venv";

  try {
    if (need.has("venv")) {
      current = "venv";
      deps.onStep("venv", "running");
      deps.mkdirp(p.root);
      const [hp, ...hpArgs] = opts.hostPython.split(" ");
      await deps.run(hp, [...hpArgs, "-m", "venv", p.venv]);
      await deps.run(p.venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
    }
    did("venv");

    if (need.has("torch")) {
      current = "torch";
      deps.onStep("torch", "running");
      const index = TORCH_INDEX[opts.device] ?? TORCH_INDEX.cpu;
      await deps.run(p.venvPython, ["-m", "pip", "install", "torch", "--index-url", index]);
    }
    did("torch");

    if (need.has("deps")) {
      current = "deps";
      deps.onStep("deps", "running");
      await deps.run(p.venvPython, ["-m", "pip", "install", "-r", opts.requirementsPath]);
    }
    did("deps");

    if (need.has("assets")) {
      current = "assets";
      deps.onStep("assets", "running");
      for (const a of opts.manifest.assets) {
        const dest = a.name.includes("extractor") ? p.extractorWeights : p.corpus;
        if (await deps.verify(dest, a)) continue;
        const url = assetUrl(opts.manifest, a, opts.baseUrlOverride);
        await deps.download(url, dest, (d, t) => deps.onStep("assets", "running", `${a.file} ${d}/${t}`));
        if (!(await deps.verify(dest, a))) throw new Error(`checksum mismatch: ${a.file}`);
      }
    }
    did("assets");

    if (need.has("codebert")) {
      current = "codebert";
      deps.onStep("codebert", "running");
      await deps.run(
        p.venvPython,
        ["-c", "from transformers import AutoModel,AutoTokenizer;AutoModel.from_pretrained('microsoft/codebert-base');AutoTokenizer.from_pretrained('microsoft/codebert-base')"],
        { HF_HOME: p.hfCache },
      );
      codebertReady = true;
    }
    did("codebert");

    current = "launch";
    deps.onStep("launch", "running");
    const port = await findFreePort(8000);
    const env = buildBackendEnv(p, opts.device);
    deps.spawnBackend(p.venvPython, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(port)],
      { cwd: opts.backendCwd, env, logPath: p.backendLog });
    const url = `http://127.0.0.1:${port}`;
    let healthy = false;
    for (let i = 0; i < 60; i++) {
      if (await deps.health(url)) { healthy = true; break; }
      await new Promise((r) => setTimeout(r, 1000));
    }
    if (!healthy) throw new Error("backend did not become healthy in time");
    deps.onStep("launch", "done");

    deps.writeState(inputs, codebertReady);
    return { backendUrl: url };
  } catch (e) {
    deps.onStep(current, "error", (e as Error).message);
    throw e;
  }
}
