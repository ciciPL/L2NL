import * as vscode from "vscode";
import { RawSettings, buildRequest } from "./config";
import { getHealth, postSummarize } from "./client";
import { ResultPanel } from "./panel";
import { provPaths } from "./paths";
import { runProvision, stopBackend } from "./backend";

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

async function provisionAndStart(ctx: vscode.ExtensionContext): Promise<string | undefined> {
  return await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Code Summary: setting up backend",
      cancellable: false,
    },
    () => runProvision(ctx, (s, st, d) =>
      console.log(`[provision] ${s}: ${st}${d ? " " + d : ""}`)));
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
    return true; // provision() already polled /health and threw on timeout
  } catch (e: any) {
    const log = provPaths(ctx.globalStorageUri.fsPath, process.platform).backendLog;
    vscode.window.showErrorMessage(`Backend setup failed: ${e.message}. See ${log}.`);
    return false;
  }
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
