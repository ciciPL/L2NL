import * as vscode from "vscode";
import * as fs from "node:fs";
import * as path from "node:path";
import { ChildProcess, spawn, execFile } from "node:child_process";
import { getHealth } from "./client";
import { provPaths } from "./paths";
import { detectPython, detectGpu } from "./env";
import { parseManifest, downloadAsset, verifyAsset, sha256File } from "./assets";
import { readState, writeState, Step } from "./state";
import { provision, ProvDeps, StepStatus } from "./provision";
import { buildTestRequest, ModelConfig } from "./models";

let backendProc: ChildProcess | undefined;

export type OnStep = (step: Step, status: StepStatus, detail?: string) => void;

// Run a command to completion, capturing output. maxBuffer is large because
// pip install produces a lot of output (torch wheels etc.).
export function run(cmd: string, args: string[] = [], extraEnv?: Record<string, string>) {
  return new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
    execFile(cmd, args,
      { env: { ...process.env, ...extraEnv }, maxBuffer: 64 * 1024 * 1024 },
      (err, stdout, stderr) =>
        resolve({ code: err ? ((err as any).code ?? 1) : 0, stdout, stderr }));
  });
}

export interface EnvInfo {
  python?: { cmd: string[]; version: [number, number] };
  gpu: boolean;
}

export async function detectEnv(): Promise<EnvInfo> {
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  const py = await detectPython(
    process.platform, cfg.get("backend.pythonPath", "") || undefined, run);
  const gpu = await detectGpu(run);
  return { python: py ? { cmd: py.cmd, version: py.version } : undefined, gpu };
}

export async function runProvision(
  ctx: vscode.ExtensionContext, onStep: OnStep,
): Promise<string | undefined> {
  const root = ctx.globalStorageUri.fsPath;
  fs.mkdirSync(root, { recursive: true });
  const paths = provPaths(root, process.platform);
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  const backendCwd = path.join(ctx.extensionPath, "backend");
  const requirementsPath = path.join(backendCwd, "requirements.txt");
  const manifest = parseManifest(
    fs.readFileSync(path.join(backendCwd, "assets", "manifest.json"), "utf8"));

  const py = await detectPython(
    process.platform, cfg.get("backend.pythonPath", "") || undefined, run);
  if (!py) {
    vscode.window.showErrorMessage(
      "Code Summary needs Python ≥3.10. Install it from https://www.python.org/downloads/ and retry.");
    return undefined;
  }
  const device = cfg.get("backend.device", "cpu");
  const reqHash = await sha256File(requirementsPath);

  const deps: ProvDeps = {
    run,
    spawnBackend: (python, args, o) => {
      const log = fs.createWriteStream(o.logPath, { flags: "a" });
      backendProc = spawn(python, args, { cwd: o.cwd, env: { ...process.env, ...o.env } });
      backendProc.stdout?.pipe(log);
      backendProc.stderr?.pipe(log);
      backendProc.on("exit", () => (backendProc = undefined));
      return { pid: backendProc.pid };
    },
    download: (url, dest, onBytes) => downloadAsset(url, dest, onBytes),
    verify: (p2, a) => verifyAsset(p2, a),
    health: (url) => getHealth(url),
    mkdirp: (d) => fs.mkdirSync(d, { recursive: true }),
    readState: () => readState(paths.state),
    writeState: (inputs, cb) => writeState(paths.state, inputs, cb),
    onStep,
  };

  const { backendUrl } = await provision(paths, {
    device,
    hostPython: py.cmd,
    extVersion: ctx.extension.packageJSON.version,
    reqHash,
    requirementsPath,
    backendCwd,
    baseUrlOverride: cfg.get("assets.baseUrl", "") || undefined,
    manifest,
  }, deps);
  await cfg.update("backend.url", backendUrl, vscode.ConfigurationTarget.Global);
  await cfg.update("backend.managed", true, vscode.ConfigurationTarget.Global);
  return backendUrl;
}

export function stopBackend() {
  backendProc?.kill();
  backendProc = undefined;
}

export async function testConnection(m: ModelConfig): Promise<{ ok: boolean; message: string }> {
  try {
    const r = buildTestRequest(m);
    const res = await fetch(r.url, { method: "POST", headers: r.headers, body: r.body });
    if (res.ok) return { ok: true, message: "OK" };
    return { ok: false, message: `HTTP ${res.status}: ${(await res.text()).slice(0, 200)}` };
  } catch (e: any) {
    return { ok: false, message: e.message };
  }
}
