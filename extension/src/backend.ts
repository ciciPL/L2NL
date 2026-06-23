import * as vscode from "vscode";
import * as fs from "node:fs";
import * as path from "node:path";
import { ChildProcess, spawn, execFile } from "node:child_process";
import { getHealth } from "./client";
import { provPaths } from "./paths";
import { detectPython, detectGpu } from "./env";
import { AssetEntry, Manifest, installAsset, parseManifest, downloadAsset, verifyAsset, sha256File } from "./assets";
import { readState, writeState, Step } from "./state";
import { findFreePort, provision, ProvDeps, StepStatus } from "./provision";
import { buildTestRequest, ModelConfig } from "./models";
import {
  DEFAULT_RUNTIME_MANIFEST,
  MODEL_SCOPE_PRESETS,
  ModelScopeFile,
  ModelScopePreset,
  RuntimeManifest,
  RuntimeAsset,
  buildLlamaServerArgs,
  downloadModelScopeFile,
  fetchModelScopeGgufFiles,
  findRuntimeAsset,
  modelLocalPath,
  platformKey,
  runtimeExecutablePath,
} from "./localRuntime";

let backendProc: ChildProcess | undefined;
let llamaProc: ChildProcess | undefined;

export type OnStep = (step: Step, status: StepStatus, detail?: string) => void;
export type RuntimeStep = "model" | "runtime" | "launch" | "test";
export type OnRuntimeStep = (step: RuntimeStep, status: StepStatus, detail?: string) => void;

const LOCAL_NO_PROXY = "127.0.0.1,localhost,::1";

export interface OfflineModelSelection {
  modelId: string;
  filePath: string;
  size?: number;
  sha256?: string;
}

export interface OfflineModelSearchResult {
  presets: ModelScopePreset[];
  files: ModelScopeFile[];
  searchUrl?: string;
}

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
    writeManifest: (p2, text) => fs.writeFileSync(p2, text),
    readState: () => readState(paths.state),
    writeState: (inputs, cb) => writeState(paths.state, inputs, cb),
    onStep,
  };

  const { backendUrl } = await provision(paths, {
    device,
    hostPython: py.cmd,
    pythonIndexUrl: cfg.get("python.indexUrl", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    extVersion: ctx.extension.packageJSON.version,
    reqHash,
    requirementsPath,
    backendCwd,
    baseUrlOverride: cfg.get("assets.baseUrl", "") || undefined,
    localAssetDir: cfg.get("assets.localDir", "") || undefined,
    includeGlobalMirrors: cfg.get("assets.tryGlobalMirrors", false),
    allowStubs: cfg.get("backend.allowStubs", false),
    manifest,
  }, deps);
  await cfg.update("backend.url", backendUrl, vscode.ConfigurationTarget.Global);
  await cfg.update("backend.managed", true, vscode.ConfigurationTarget.Global);
  return backendUrl;
}

function executableExists(p: string): boolean {
  try {
    fs.accessSync(p, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findOnPath(cmd: string): Promise<string | undefined> {
  const tool = process.platform === "win32" ? "where" : "which";
  const res = await run(tool, [cmd]);
  if (res.code !== 0) return undefined;
  return res.stdout.split(/\r?\n/).map((s) => s.trim()).find(Boolean);
}

function runtimeAssetEntry(runtime: RuntimeAsset): AssetEntry {
  return {
    name: runtime.target,
    target: runtime.target,
    sourcePath: runtime.sourcePath,
    sha256: runtime.sha256,
    size: runtime.size,
    parts: runtime.parts,
  };
}

function runtimeManifestFor(asset: RuntimeAsset): Manifest {
  return {
    version: DEFAULT_RUNTIME_MANIFEST.version,
    release: DEFAULT_RUNTIME_MANIFEST.release,
    sources: DEFAULT_RUNTIME_MANIFEST.sources,
    assets: [runtimeAssetEntry(asset)],
  };
}

function loadRuntimeManifest(ctx: vscode.ExtensionContext): RuntimeManifest {
  const manifestPath = path.join(ctx.extensionPath, "runtime-manifest.json");
  try {
    const parsed = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as RuntimeManifest;
    if (Array.isArray(parsed.sources) && Array.isArray(parsed.runtimes)) return parsed;
  } catch {
    // Fall back to the embedded domestic source; missing runtimes produce a clear
    // error later with the platform name.
  }
  return DEFAULT_RUNTIME_MANIFEST;
}

async function ensureManagedLlamaServer(
  ctx: vscode.ExtensionContext,
  onStep: OnRuntimeStep,
): Promise<string> {
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  const configured = cfg.get("offline.llamaServerPath", "");
  if (configured) {
    if (executableExists(configured)) return configured;
    throw new Error(`Configured llama-server not found or not executable: ${configured}`);
  }

  const binName = process.platform === "win32" ? "llama-server.exe" : "llama-server";
  const pathHit = await findOnPath(binName);
  if (pathHit) return pathHit;

  const key = platformKey();
  const runtimeManifest = loadRuntimeManifest(ctx);
  const asset = findRuntimeAsset(runtimeManifest, key);
  if (!key) {
    throw new Error("Offline automatic setup currently supports Windows x64 and macOS only.");
  }
  if (!asset) {
    throw new Error(
      `llama.cpp runtime package is not published for ${key}. ` +
      "Upload v0.2-runtime-llamacpp to Gitee, or set codeSummary.offline.llamaServerPath to an existing llama-server.",
    );
  }

  const root = path.join(ctx.globalStorageUri.fsPath, "runtime", "llamacpp", key);
  const archive = path.join(root, asset.target);
  const exe = runtimeExecutablePath(root, asset);
  if (!executableExists(exe)) {
    onStep("runtime", "running", `downloading ${asset.target}`);
    const manifest = {
      ...runtimeManifestFor(asset),
      version: runtimeManifest.version,
      release: runtimeManifest.release,
      sources: runtimeManifest.sources,
    };
    await installAsset(manifest, manifest.assets[0], archive, {
      includeGlobalMirrors: false,
      download: (url, dest, onBytes) => downloadAsset(url, dest, onBytes),
      verify: (p, a) => verifyAsset(p, a),
      onBytes: (d, t) => onStep("runtime", "running", `${asset.target} ${d}/${t}`),
    });
    onStep("runtime", "running", `extracting ${asset.target}`);
    const extracted = await run("tar", ["-xf", archive, "-C", root]);
    if (extracted.code !== 0) {
      throw new Error(`failed to extract ${asset.target}: ${extracted.stderr.slice(0, 300)}`);
    }
    if (process.platform !== "win32") fs.chmodSync(exe, 0o755);
  }
  if (!executableExists(exe)) throw new Error(`llama-server missing after runtime install: ${exe}`);
  return exe;
}

function modelNameFromFile(filePath: string): string {
  return path.basename(filePath).replace(/\.gguf$/i, "");
}

export async function searchModelScopeGguf(query: string): Promise<OfflineModelSearchResult> {
  const q = query.trim();
  if (!q) return { presets: MODEL_SCOPE_PRESETS, files: [] };
  if (q.includes("/")) {
    const files = await fetchModelScopeGgufFiles(q);
    return { presets: [], files };
  }
  const lowered = q.toLowerCase();
  const presets = MODEL_SCOPE_PRESETS.filter((p) =>
    p.label.toLowerCase().includes(lowered) ||
    p.modelId.toLowerCase().includes(lowered) ||
    p.filePath.toLowerCase().includes(lowered));
  return {
    presets,
    files: [],
    searchUrl: `https://modelscope.cn/models?name=${encodeURIComponent(q)}`,
  };
}

export async function prepareOfflineLlama(
  ctx: vscode.ExtensionContext,
  selection: OfflineModelSelection,
  onStep: OnRuntimeStep,
): Promise<{ baseUrl: string; model: string; modelPath: string; runtimePath: string }> {
  const root = ctx.globalStorageUri.fsPath;
  fs.mkdirSync(root, { recursive: true });
  const modelsRoot = path.join(root, "models");
  const modelPath = modelLocalPath(modelsRoot, selection.modelId, selection.filePath);

  onStep("model", "running", `downloading ${selection.filePath}`);
  await downloadModelScopeFile(
    selection.modelId,
    selection.filePath,
    modelPath,
    { sha256: selection.sha256, size: selection.size },
    (d, t) => onStep("model", "running", `${path.basename(selection.filePath)} ${d}/${t}`),
  );
  onStep("model", "done", modelPath);

  onStep("runtime", "running", "resolving llama-server");
  const runtimePath = await ensureManagedLlamaServer(ctx, onStep);
  onStep("runtime", "done", runtimePath);

  const port = await findFreePort(8080, 50);
  const baseUrl = `http://127.0.0.1:${port}/v1`;
  const model = modelNameFromFile(selection.filePath);
  stopLlamaRuntime();

  onStep("launch", "running", `starting ${baseUrl}`);
  const logPath = path.join(root, "llama-server.log");
  const log = fs.createWriteStream(logPath, { flags: "a" });
  llamaProc = spawn(runtimePath, buildLlamaServerArgs({ modelPath, port }), {
    env: { ...process.env, NO_PROXY: LOCAL_NO_PROXY, no_proxy: LOCAL_NO_PROXY },
  });
  llamaProc.stdout?.pipe(log);
  llamaProc.stderr?.pipe(log);
  llamaProc.on("exit", () => (llamaProc = undefined));
  llamaProc.on("error", (e) => log.write(`\n${e.message}\n`));

  onStep("test", "running", "probing /chat/completions");
  let last = "";
  for (let i = 0; i < 60; i++) {
    const probe = await testConnection({ base_url: baseUrl, api_key: "", model });
    if (probe.ok) {
      const cfg = vscode.workspace.getConfiguration("codeSummary");
      await cfg.update("mode", "offline", vscode.ConfigurationTarget.Global);
      await cfg.update("offline.baseUrl", baseUrl, vscode.ConfigurationTarget.Global);
      await cfg.update("offline.model", model, vscode.ConfigurationTarget.Global);
      await cfg.update("offline.modelPath", modelPath, vscode.ConfigurationTarget.Global);
      onStep("launch", "done", baseUrl);
      onStep("test", "done", "OK");
      return { baseUrl, model, modelPath, runtimePath };
    }
    last = probe.message;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  stopLlamaRuntime();
  throw new Error(`llama-server did not become ready: ${last || "timeout"}. See ${logPath}`);
}

export function stopLlamaRuntime() {
  llamaProc?.kill();
  llamaProc = undefined;
}

export function stopBackend() {
  backendProc?.kill();
  backendProc = undefined;
  stopLlamaRuntime();
}

export async function testConnection(
  m: ModelConfig,
  timeoutMs = 15000,
): Promise<{ ok: boolean; message: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = buildTestRequest(m);
    const res = await fetch(r.url, {
      method: "POST",
      headers: r.headers,
      body: r.body,
      signal: controller.signal,
    });
    if (res.ok) return { ok: true, message: "OK" };
    return { ok: false, message: `HTTP ${res.status}: ${(await res.text()).slice(0, 200)}` };
  } catch (e: any) {
    if (e?.name === "AbortError") return { ok: false, message: `Timed out after ${timeoutMs / 1000}s` };
    return { ok: false, message: e.message };
  } finally {
    clearTimeout(timer);
  }
}
