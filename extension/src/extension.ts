import * as vscode from "vscode";
import { RawSettings, buildRequest } from "./config";
import { getHealth, postSummarize } from "./client";
import { ResultPanel } from "./panel";
import { provPaths } from "./paths";
import { runProvision, stopBackend } from "./backend";
import { WizardPanel } from "./wizard";
import { loadOnlineApiKey } from "./secrets";

let statusItem: vscode.StatusBarItem;

async function readSettings(ctx: vscode.ExtensionContext): Promise<RawSettings> {
  const c = vscode.workspace.getConfiguration("codeSummary");
  return {
    mode: c.get("mode", "online") as "online" | "offline",
    online: {
      baseUrl: c.get("online.baseUrl", "https://api.openai.com/v1"),
      apiKey: await loadOnlineApiKey(ctx.secrets, c.get("online.apiKey", "")),
      model: c.get("online.model", "gpt-4o-mini"),
    },
    offline: {
      baseUrl: c.get("offline.baseUrl", "http://localhost:8080/v1"),
      model: c.get("offline.model", "local-model"),
    },
    params: {
      k: c.get("params.k", 3),
      temperatures: c.get("params.temperatures", [0]),
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
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  if (!cfg.get("backend.managed", false)) {
    statusItem.text = "$(rocket) Summary: setup";
    statusItem.tooltip = "Run Code Summary setup";
    statusItem.command = "codeSummary.setup";
    statusItem.show();
    return;
  }
  const mode = cfg.get<"online" | "offline">("mode", "online");
  const icon = mode === "offline" ? "$(vm)" : "$(cloud)";
  const model = mode === "offline" ? cfg.get("offline.model", "local-model") : cfg.get("online.model", "gpt-4o-mini");
  statusItem.text = `${icon} Summary: ${mode} · ${model}`;
  statusItem.tooltip = "Configure Code Summary model (online/offline)";
  statusItem.command = "codeSummary.configureModel";
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
    const body = buildRequest(code, language, await readSettings(ctx));
    const resp = await postSummarize(backendUrl(), body);
    ResultPanel.show(resp, code);
  } catch (e: any) {
    vscode.window.showErrorMessage(`Summarize failed: ${e.message}`);
  }
}

async function configureModel(ctx: vscode.ExtensionContext) {
  const pick = await vscode.window.showQuickPick([
    {
      label: "Configure Offline Local Model",
      description: "Use an existing GGUF or download one from ModelScope",
      command: "codeSummary.configureOffline",
    },
    {
      label: "Configure Online API",
      description: "DeepSeek, OpenAI, or an OpenAI-compatible endpoint",
      command: "codeSummary.configureOnline",
    },
    {
      label: "Run Full Setup",
      description: "Backend assets, Python environment, and model configuration",
      command: "codeSummary.setup",
    },
    {
      label: "Toggle Active Mode",
      description: "Switch between already configured online/offline modes",
      command: "codeSummary.toggleMode",
    },
  ], { placeHolder: "Choose how to configure Code Summary" });
  if (pick) await vscode.commands.executeCommand(pick.command);
}

export function activate(ctx: vscode.ExtensionContext) {
  statusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right, 100);
  updateStatus();

  ctx.subscriptions.push(
    statusItem,
    vscode.commands.registerCommand("codeSummary.summarizeSelection", () => summarizeSelection(ctx)),
    vscode.commands.registerCommand("codeSummary.startBackend", () => provisionAndStart(ctx)),
    vscode.commands.registerCommand("codeSummary.stopBackend", stopBackend),
    vscode.commands.registerCommand("codeSummary.setup", () => WizardPanel.open(ctx)),
    vscode.commands.registerCommand("codeSummary.configureModel", () => configureModel(ctx)),
    vscode.commands.registerCommand("codeSummary.configureOnline", () =>
      WizardPanel.open(ctx, { initialMode: "online", initialStep: "model" })),
    vscode.commands.registerCommand("codeSummary.configureOffline", () =>
      WizardPanel.open(ctx, { initialMode: "offline", initialStep: "model" })),
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

  const cfg = vscode.workspace.getConfiguration("codeSummary");
  if (cfg.get("backend.managed", false)) {
    // Already provisioned: bring the managed backend up on launch so it's ready
    // without a manual trigger (ensureBackend is a no-op if already healthy).
    void ensureBackend(ctx);
  } else {
    WizardPanel.open(ctx);
  }
}

export function deactivate() {
  stopBackend();
}
