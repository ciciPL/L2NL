import * as net from "node:net";
import * as fs from "node:fs";
import { ProvPaths } from "./paths";
import { Manifest, AssetEntry, installAsset, assetTarget } from "./assets";
import { ProvState, ProvInputs, Step, stepsNeeded } from "./state";
import { Runner } from "./env";

const LOCAL_NO_PROXY = "127.0.0.1,localhost,::1";

export function buildBackendEnv(
  p: ProvPaths,
  opts: { device: string; allowStubs: boolean },
): Record<string, string> {
  return {
    CS_PYTHON: p.venvPython,
    CS_EXTRACTOR_WEIGHTS: p.extractorWeights,
    CS_CORPUS_PATH: p.corpus,
    CS_CORPUS_LIMIT: "30000",
    CS_DEVICE: opts.device,
    CS_CODEBERT_PATH: p.codebertDir,
    CS_ASSETS_MANIFEST: p.assetsManifest,
    CS_ALLOW_STUBS: opts.allowStubs ? "1" : "0",
    HF_HOME: p.hfCache,
    NO_PROXY: LOCAL_NO_PROXY,
    no_proxy: LOCAL_NO_PROXY,
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
  writeManifest: (path: string, text: string) => void;
  readState: () => ProvState | null;
  writeState: (inputs: ProvInputs, codebertReady: boolean) => void;
  onStep: (step: Step, status: StepStatus, detail?: string) => void;
}

export interface ProvOptions {
  device: string;
  extVersion: string;
  reqHash: string;
  hostPython: string[];
  pythonIndexUrl: string;
  requirementsPath: string;
  backendCwd: string;
  baseUrlOverride?: string;
  localAssetDir?: string;
  includeGlobalMirrors: boolean;
  allowStubs: boolean;
  manifest: Manifest;
}

function pipInstallArgs(indexUrl: string, extra: string[]): string[] {
  return ["-m", "pip", "install", "--retries", "5", "--timeout", "60", "-i", indexUrl, ...extra];
}

function assetHashes(m: Manifest): Record<string, string> {
  const out: Record<string, string> = {};
  for (const a of m.assets) out[a.name] = a.sha256;
  return out;
}

function assetDest(p: ProvPaths, a: AssetEntry): string {
  if (a.name.includes("extractor")) return p.extractorWeights;
  if (a.name.includes("corpus")) return p.corpus;
  if (a.extractTo || a.name.toLowerCase().includes("codebert")) return p.codebertArchive;
  return `${p.assets}/${assetTarget(a)}`;
}

function codebertAsset(m: Manifest): AssetEntry | undefined {
  return m.assets.find((a) => a.extractTo === "codebert-base" || a.name.toLowerCase().includes("codebert"));
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
  const mustRun = async (cmd: string, args: string[], env?: Record<string, string>) => {
    const r = await deps.run(cmd, args, env);
    if (r.code !== 0) throw new Error(`${cmd} exited ${r.code}: ${r.stderr.slice(0, 300)}`);
  };
  let codebertReady = !need.has("codebert");
  let current: Step = "venv";

  try {
    if (need.has("venv")) {
      current = "venv";
      deps.onStep("venv", "running");
      deps.mkdirp(p.root);
      const [hp, ...hpArgs] = opts.hostPython;
      await mustRun(hp, [...hpArgs, "-m", "venv", p.venv]);
      await mustRun(p.venvPython, pipInstallArgs(opts.pythonIndexUrl, ["--upgrade", "pip"]));
    }
    did("venv");

    if (need.has("torch")) {
      current = "torch";
      deps.onStep("torch", "running");
      await mustRun(p.venvPython, pipInstallArgs(opts.pythonIndexUrl, ["torch"]));
    }
    did("torch");

    if (need.has("deps")) {
      current = "deps";
      deps.onStep("deps", "running");
      await mustRun(p.venvPython, pipInstallArgs(opts.pythonIndexUrl, ["-r", opts.requirementsPath]));
    }
    did("deps");

    if (need.has("assets")) {
      current = "assets";
      deps.onStep("assets", "running");
      for (const a of opts.manifest.assets) {
        const dest = assetDest(p, a);
        await installAsset(opts.manifest, a, dest, {
          localAssetDir: opts.localAssetDir,
          baseUrlOverride: opts.baseUrlOverride,
          includeGlobalMirrors: opts.includeGlobalMirrors,
          download: deps.download,
          verify: deps.verify,
          onBytes: (d, t) => deps.onStep("assets", "running", `${assetTarget(a)} ${d}/${t}`),
        });
      }
      deps.mkdirp(p.assets);
      deps.writeManifest(p.assetsManifest, JSON.stringify(opts.manifest, null, 2));
    }
    did("assets");

    if (need.has("codebert")) {
      current = "codebert";
      deps.onStep("codebert", "running");
      const cb = codebertAsset(opts.manifest);
      if (!cb) throw new Error("manifest is missing the CodeBERT archive asset");
      fs.rmSync(p.codebertDir, { recursive: true, force: true });
      await mustRun("tar", ["-xf", assetDest(p, cb), "-C", p.assets]);
      codebertReady = true;
    }
    did("codebert");

    current = "launch";
    deps.onStep("launch", "running");
    const port = await findFreePort(8000);
    const env = buildBackendEnv(p, { device: opts.device, allowStubs: opts.allowStubs });
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
