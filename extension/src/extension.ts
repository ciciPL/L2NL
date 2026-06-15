import * as vscode from "vscode";
import * as fs from "node:fs";
import * as path from "node:path";
import { ChildProcess, spawn, execFile } from "node:child_process";
import { RawSettings, buildRequest } from "./config";
import { getHealth, postSummarize } from "./client";
import { ResultPanel } from "./panel";
import { provPaths } from "./paths";
import { detectPython } from "./env";
import { parseManifest, downloadAsset, verifyAsset, sha256File } from "./assets";
import { readState, writeState } from "./state";
import { provision, ProvDeps } from "./provision";

let backendProc: ChildProcess | undefined;
let statusItem: vscode.StatusBarItem;

function readSettings(): RawSettings {
  const c = vscode.workspace.getConfiguration("codeSummary");
  return {
    mode: c.get("mode", "online") as "online" | "offline",
    online: {
      baseUrl: c.get("online.baseUrl", "https://api.openai.com/v1"),
      apiKey: c.get("online.apiKey", ""),
      model: c.get("online.model", "gpt-4o-mini"),
    },
    offline: {
      baseUrl: c.get("offline.baseUrl", "http://localhost:8080/v1"),
      model: c.get("offline.model", "local-model"),
    },
    params: {
      k: c.get("params.k", 5),
      temperatures: c.get("params.temperatures", [0, 0.4, 0.8]),
      lambda: c.get("params.lambda", 0.5),
      threshold: c.get("params.threshold", 0.5),
      maxRepairIters: c.get("params.maxRepairIters", 3),
    },
  };
}

function backendUrl(): string {
  return vscode.workspace.getConfiguration("codeSummary").get(
    "backend.url", "http://localhost:8000");
}

function updateStatus() {
  const s = readSettings();
  const icon = s.mode === "offline" ? "$(vm)" : "$(cloud)";
  const model = s.mode === "offline" ? s.offline.model : s.online.model;
  statusItem.text = `${icon} Summary: ${s.mode} · ${model}`;
  statusItem.show();
}

// Run a command to completion, capturing output. maxBuffer is large because
// pip install produces a lot of output (torch wheels etc.).
function run(cmd: string, args: string[] = [], extraEnv?: Record<string, string>) {
  return new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
    execFile(cmd, args,
      { env: { ...process.env, ...extraEnv }, maxBuffer: 64 * 1024 * 1024 },
      (err, stdout, stderr) =>
        resolve({ code: err ? ((err as any).code ?? 1) : 0, stdout, stderr }));
  });
}

async function provisionAndStart(ctx: vscode.ExtensionContext): Promise<string | undefined> {
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
    onStep: (s, st, detail) =>
      console.log(`[provision] ${s}: ${st}${detail ? " " + detail : ""}`),
  };

  return await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Code Summary: setting up backend",
      cancellable: false,
    },
    async () => {
      const { backendUrl: newUrl } = await provision(paths, {
        device,
        hostPython: py.cmd.join(" "),
        extVersion: ctx.extension.packageJSON.version,
        reqHash,
        requirementsPath,
        backendCwd,
        baseUrlOverride: cfg.get("assets.baseUrl", "") || undefined,
        manifest,
      }, deps);
      await cfg.update("backend.url", newUrl, vscode.ConfigurationTarget.Global);
      await cfg.update("backend.managed", true, vscode.ConfigurationTarget.Global);
      return newUrl;
    });
}

async function ensureBackend(ctx: vscode.ExtensionContext): Promise<boolean> {
  const url = backendUrl();
  if (await getHealth(url)) return true;
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  if (!cfg.get("backend.autoStart", true) && !cfg.get("backend.managed", false)) {
    vscode.window.showErrorMessage(
      `Code Summary backend not reachable at ${url}. Start it manually.`);
    return false;
  }
  try {
    const newUrl = await provisionAndStart(ctx);
    if (!newUrl) return false;
    for (let i = 0; i < 10; i++) {
      if (await getHealth(newUrl)) return true;
      await new Promise((r) => setTimeout(r, 500));
    }
    return await getHealth(newUrl);
  } catch (e: any) {
    vscode.window.showErrorMessage(
      `Backend setup failed: ${e.message}. See the developer console and ${backendUrl()}.`);
    return false;
  }
}

function stopBackend() {
  backendProc?.kill();
  backendProc = undefined;
}

async function summarizeSelection(ctx: vscode.ExtensionContext) {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Select some code first.");
    return;
  }
  const code = editor.document.getText(editor.selection);
  const language = editor.document.languageId;
  if (!(await ensureBackend(ctx))) return;

  ResultPanel.loading();
  try {
    const body = buildRequest(code, language, readSettings());
    const resp = await postSummarize(backendUrl(), body);
    ResultPanel.show(resp);
  } catch (e: any) {
    vscode.window.showErrorMessage(`Summarize failed: ${e.message}`);
  }
}

export function activate(ctx: vscode.ExtensionContext) {
  statusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right, 100);
  statusItem.command = "codeSummary.toggleMode";
  updateStatus();

  ctx.subscriptions.push(
    statusItem,
    vscode.commands.registerCommand("codeSummary.summarizeSelection", () => summarizeSelection(ctx)),
    vscode.commands.registerCommand("codeSummary.startBackend", () => provisionAndStart(ctx)),
    vscode.commands.registerCommand("codeSummary.stopBackend", stopBackend),
    vscode.commands.registerCommand("codeSummary.setup", () => provisionAndStart(ctx)),
    vscode.commands.registerCommand("codeSummary.toggleMode", async () => {
      const cfg = vscode.workspace.getConfiguration("codeSummary");
      const next = cfg.get("mode", "online") === "online" ? "offline" : "online";
      await cfg.update("mode", next, vscode.ConfigurationTarget.Global);
      updateStatus();
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("codeSummary")) updateStatus();
    }),
  );
}

export function deactivate() {
  stopBackend();
}
