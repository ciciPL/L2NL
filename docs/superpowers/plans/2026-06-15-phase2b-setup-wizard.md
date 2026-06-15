# Phase 2B — Setup Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A visual multi-step webview Setup wizard (welcome → environment check → backend install with live progress → cloud/local model configuration → done) that drives the Phase 2A provisioner engine, so users never edit settings.json by hand.

**Architecture:** Extract the vscode-aware backend orchestration (`run`, `detectEnv`, `runProvision(ctx, onStep)`, `stopBackend`) out of `extension.ts` into a shared `backend.ts`, parameterized by an `onStep` callback so BOTH the lazy headless path (logs to console) and the wizard (streams to the webview) drive the same engine. Pure model-config helpers (vendor presets, connection-test request builder) live in a unit-tested `models.ts`. `wizard.ts` is an interactive webview (scripts enabled) that bridges its UI to `backend.ts` via `postMessage`. `extension.ts` opens the wizard on first run and via the `codeSummary.setup` command.

**Tech Stack:** TypeScript, VS Code webview API (`enableScripts`, `postMessage`, `acquireVsCodeApi`), vitest, esbuild.

**Builds on:** Phase 2A (merged to `main`). Engine exports already in place: `provision()`, `ProvDeps`, `Step`, `StepStatus` (`extension/src/provision.ts`); `detectPython`, `detectGpu`, `PyInfo` (`env.ts`).

**Spec:** `docs/superpowers/specs/2026-06-15-phase2-provisioner-wizard-design.md` §5.

**Repo convention:** work on a feature branch off `main`; commit per task.

---

## File Structure

| File | Responsibility | vscode? |
|---|---|---|
| `extension/src/backend.ts` | NEW. Owns `backendProc`; `run()`, `detectEnv()`, `runProvision(ctx, onStep)`, `stopBackend()`, `testConnection()`. Moved out of extension.ts. | yes |
| `extension/src/models.ts` | NEW. Pure: `CLOUD_VENDORS`/`LOCAL_PRESETS` presets + `buildTestRequest()`. Unit-tested. | no |
| `extension/src/wizard.ts` | NEW. `WizardPanel` interactive webview + message bridge. | yes |
| `extension/src/extension.ts` | MODIFY. Use `backend.ts`; open wizard on first run + `codeSummary.setup`. | yes |
| `extension/package.json` | MODIFY. `codeSummary.setup` title (already exists); nothing else required. | — |
| Tests | `extension/test/models.test.ts` (pure helpers). Wizard verified by build + manual smoke. | — |

---

## Task 1: Extract `backend.ts` from `extension.ts`

Pure refactor — move the orchestration glue into a reusable module parameterized by `onStep`. Behavior identical; unit suite + build stay green.

**Files:**
- Create: `extension/src/backend.ts`
- Modify: `extension/src/extension.ts`

- [ ] **Step 1: Create `extension/src/backend.ts`:**

```ts
import * as vscode from "vscode";
import * as fs from "node:fs";
import * as path from "node:path";
import { ChildProcess, spawn, execFile } from "node:child_process";
import { getHealth } from "./client";
import { provPaths } from "./paths";
import { detectPython, detectGpu } from "./env";
import { parseManifest, downloadAsset, verifyAsset, sha256File } from "./assets";
import { readState, writeState } from "./state";
import { provision, ProvDeps, Step, StepStatus } from "./provision";
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
```

- [ ] **Step 2: Rewrite `extension/src/extension.ts`** to consume `backend.ts`. Replace the top imports + the `run`/`provisionAndStart`/`stopBackend` definitions. The new extension.ts is:

```ts
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
```

> Note: `codeSummary.setup` still points at `provisionAndStart` here; Task 5 repoints it to the wizard. Keeping this task a pure behavior-preserving extraction makes it easy to verify.

- [ ] **Step 3: Verify** from `extension/`: `npm run build` (clean), `npx tsc --noEmit` (clean), `npx vitest run` (still 28 passing — `models.ts` test arrives in Task 2, so `testConnection`'s import of `./models` must resolve: create a STUB `models.ts` now if needed OR do Task 2 first). To avoid a broken import, **do Task 2 before building Task 1**, or create `models.ts` as part of this task. Simplest: implement Task 2's `models.ts` first, then this task. (The executor should reorder: Task 2 → Task 1.)

- [ ] **Step 4: Commit:**
```bash
git add extension/src/backend.ts extension/src/extension.ts
git commit -m "refactor(ext): extract backend.ts (run/detectEnv/runProvision/testConnection)"
```

---

## Task 2: `models.ts` — vendor presets + connection-test builder (pure)

**Do this BEFORE Task 1** (Task 1 imports it).

**Files:**
- Create: `extension/src/models.ts`
- Test: `extension/test/models.test.ts`

- [ ] **Step 1: Write the failing test** — `extension/test/models.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { CLOUD_VENDORS, LOCAL_PRESETS, buildTestRequest } from "../src/models";

describe("presets", () => {
  it("offers DeepSeek and OpenAI cloud vendors with base URLs", () => {
    const ids = CLOUD_VENDORS.map((v) => v.id);
    expect(ids).toContain("deepseek");
    expect(ids).toContain("openai");
    expect(CLOUD_VENDORS.find((v) => v.id === "deepseek")!.baseUrl)
      .toBe("https://api.deepseek.com/v1");
  });
  it("offers local runtimes that do not need a key", () => {
    expect(LOCAL_PRESETS.find((v) => v.id === "ollama")!.needsKey).toBe(false);
  });
});

describe("buildTestRequest", () => {
  it("builds an OpenAI-compatible chat/completions probe with auth when keyed", () => {
    const r = buildTestRequest({ base_url: "https://api.deepseek.com/v1/", api_key: "K", model: "deepseek-chat" });
    expect(r.url).toBe("https://api.deepseek.com/v1/chat/completions");
    expect(r.headers.Authorization).toBe("Bearer K");
    expect(JSON.parse(r.body).model).toBe("deepseek-chat");
    expect(JSON.parse(r.body).max_tokens).toBe(1);
  });
  it("omits Authorization when there is no key (local runtimes)", () => {
    const r = buildTestRequest({ base_url: "http://localhost:11434/v1", api_key: "", model: "x" });
    expect(r.headers.Authorization).toBeUndefined();
  });
});
```

- [ ] **Step 2:** Run `npx vitest run test/models.test.ts` — expect FAIL (module missing).

- [ ] **Step 3: Write `extension/src/models.ts`:**

```ts
export interface ModelConfig { base_url: string; api_key: string; model: string; }

export interface VendorPreset {
  id: string;
  label: string;
  baseUrl: string;      // "" for custom
  defaultModel: string;
  needsKey: boolean;
}

export const CLOUD_VENDORS: VendorPreset[] = [
  { id: "deepseek", label: "DeepSeek", baseUrl: "https://api.deepseek.com/v1", defaultModel: "deepseek-chat", needsKey: true },
  { id: "openai", label: "OpenAI", baseUrl: "https://api.openai.com/v1", defaultModel: "gpt-4o-mini", needsKey: true },
  { id: "custom", label: "Custom (OpenAI-compatible)", baseUrl: "", defaultModel: "", needsKey: true },
];

export const LOCAL_PRESETS: VendorPreset[] = [
  { id: "ollama", label: "Ollama", baseUrl: "http://localhost:11434/v1", defaultModel: "qwen2.5-coder", needsKey: false },
  { id: "llamacpp", label: "llama.cpp server", baseUrl: "http://localhost:8080/v1", defaultModel: "local-model", needsKey: false },
  { id: "vllm", label: "vLLM / SGLang", baseUrl: "http://localhost:8000/v1", defaultModel: "local-model", needsKey: false },
  { id: "custom", label: "Custom", baseUrl: "", defaultModel: "", needsKey: false },
];

export function buildTestRequest(
  m: ModelConfig,
): { url: string; headers: Record<string, string>; body: string } {
  const base = m.base_url.replace(/\/+$/, "");
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (m.api_key) headers.Authorization = `Bearer ${m.api_key}`;
  return {
    url: `${base}/chat/completions`,
    headers,
    body: JSON.stringify({ model: m.model, messages: [{ role: "user", content: "ping" }], max_tokens: 1 }),
  };
}
```

- [ ] **Step 4:** Run `npx vitest run test/models.test.ts` — expect PASS (4 tests).

- [ ] **Step 5: Commit:**
```bash
git add extension/src/models.ts extension/test/models.test.ts
git commit -m "feat(ext): model vendor presets + connection-test request builder"
```

---

## Task 3: `WizardPanel` webview (`wizard.ts`)

The interactive wizard. One cohesive file: a panel class + inlined HTML/CSS/JS + a host-side message handler that bridges to `backend.ts`. Verified by build + manual smoke (webview UI is not unit-tested, matching `panel.ts`).

**Files:**
- Create: `extension/src/wizard.ts`

- [ ] **Step 1: Write `extension/src/wizard.ts`:**

```ts
import * as vscode from "vscode";
import { detectEnv, runProvision, testConnection } from "./backend";
import { CLOUD_VENDORS, LOCAL_PRESETS } from "./models";

export class WizardPanel {
  private static current: WizardPanel | undefined;
  private readonly panel: vscode.WebviewPanel;

  static open(ctx: vscode.ExtensionContext) {
    if (WizardPanel.current) { WizardPanel.current.panel.reveal(); return; }
    const panel = vscode.window.createWebviewPanel(
      "codeSummarySetup", "Code Summary — Setup",
      vscode.ViewColumn.Active, { enableScripts: true, retainContextWhenHidden: true });
    WizardPanel.current = new WizardPanel(panel, ctx);
  }

  private constructor(panel: vscode.WebviewPanel, ctx: vscode.ExtensionContext) {
    this.panel = panel;
    panel.webview.html = this.html();
    panel.onDidDispose(() => (WizardPanel.current = undefined));
    panel.webview.onDidReceiveMessage((msg) => this.handle(msg, ctx));
  }

  private post(m: any) { this.panel.webview.postMessage(m); }

  private async handle(msg: any, ctx: vscode.ExtensionContext) {
    const cfg = vscode.workspace.getConfiguration("codeSummary");
    switch (msg.type) {
      case "detectEnv": {
        const env = await detectEnv();
        this.post({ type: "envResult", env });
        break;
      }
      case "selectDevice":
        await cfg.update("backend.device", msg.device, vscode.ConfigurationTarget.Global);
        break;
      case "startInstall": {
        try {
          const url = await runProvision(ctx, (step, status, detail) =>
            this.post({ type: "stepState", step, status, detail }));
          if (url) this.post({ type: "installDone", url });
          else this.post({ type: "installError", message: "Python ≥3.10 not found." });
        } catch (e: any) {
          this.post({ type: "installError", message: e.message });
        }
        break;
      }
      case "saveModel": {
        const p = msg.payload;
        await cfg.update("mode", p.mode, vscode.ConfigurationTarget.Global);
        if (p.mode === "online") {
          await cfg.update("online.baseUrl", p.base_url, vscode.ConfigurationTarget.Global);
          await cfg.update("online.apiKey", p.api_key, vscode.ConfigurationTarget.Global);
          await cfg.update("online.model", p.model, vscode.ConfigurationTarget.Global);
        } else {
          await cfg.update("offline.baseUrl", p.base_url, vscode.ConfigurationTarget.Global);
          await cfg.update("offline.model", p.model, vscode.ConfigurationTarget.Global);
        }
        this.post({ type: "modelSaved" });
        break;
      }
      case "testConnection": {
        const r = await testConnection(msg.payload);
        this.post({ type: "testResult", ok: r.ok, message: r.message });
        break;
      }
    }
  }

  private html(): string {
    const presets = JSON.stringify({ cloud: CLOUD_VENDORS, local: LOCAL_PRESETS });
    return /* html */ `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
      body { font-family: var(--vscode-font-family); padding: 16px; color: var(--vscode-foreground); }
      h1 { font-size: 1.3em; } h2 { font-size: 1.05em; margin-top: 0; }
      .step { display: none; } .step.active { display: block; }
      .nav { margin-top: 18px; display: flex; gap: 8px; }
      button { background: var(--vscode-button-background); color: var(--vscode-button-foreground);
        border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
      button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
      button:disabled { opacity: .5; cursor: default; }
      input, select { width: 100%; padding: 6px; margin: 4px 0 10px; box-sizing: border-box;
        background: var(--vscode-input-background); color: var(--vscode-input-foreground);
        border: 1px solid var(--vscode-input-border, transparent); border-radius: 4px; }
      .row { display: flex; gap: 6px; align-items: center; }
      .ok { color: var(--vscode-testing-iconPassed, #3fb950); } .bad { color: var(--vscode-errorForeground); }
      .steps { font-family: var(--vscode-editor-font-family, monospace); }
      .steps div { padding: 2px 0; } .muted { color: var(--vscode-descriptionForeground); }
      .tabs { display: flex; gap: 8px; margin-bottom: 8px; }
      .tab { padding: 4px 10px; border-radius: 4px; cursor: pointer; background: var(--vscode-button-secondaryBackground); }
      .tab.sel { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
      pre.log { max-height: 160px; overflow: auto; background: var(--vscode-textCodeBlock-background); padding: 8px; border-radius: 4px; }
    </style></head><body>
    <div id="s-welcome" class="step active">
      <h1>Code Summary — Setup</h1>
      <p>This wizard sets up a private local backend (a Python venv + paper pipeline assets, ~400&nbsp;MB) and your summarization model. Nothing is installed system-wide; removing the extension removes it all.</p>
      <div class="nav"><button onclick="go('env')">Get started</button></div>
    </div>

    <div id="s-env" class="step">
      <h2>1 · Environment check</h2>
      <div id="envBody" class="muted">Checking…</div>
      <div id="deviceWrap" style="display:none">
        <label>Inference device</label>
        <select id="device"><option value="cpu">CPU (works everywhere)</option></select>
      </div>
      <div class="nav"><button class="secondary" onclick="go('welcome')">Back</button>
        <button id="envNext" disabled onclick="go('install')">Next</button></div>
    </div>

    <div id="s-install" class="step">
      <h2>2 · Install backend</h2>
      <div id="steps" class="steps"></div>
      <pre id="installLog" class="log muted">Idle.</pre>
      <div class="nav"><button id="installBtn" onclick="startInstall()">Install</button>
        <button id="installNext" disabled onclick="go('model')">Next</button></div>
    </div>

    <div id="s-model" class="step">
      <h2>3 · Model</h2>
      <div class="tabs"><div id="tab-online" class="tab sel" onclick="setMode('online')">Cloud API</div>
        <div id="tab-offline" class="tab" onclick="setMode('offline')">Local runtime</div></div>
      <label>Provider</label><select id="vendor" onchange="applyVendor()"></select>
      <label>Base URL</label><input id="baseUrl" placeholder="https://…/v1">
      <div id="keyWrap"><label>API key</label><input id="apiKey" type="password" placeholder="sk-…"></div>
      <label>Model</label><input id="model" placeholder="model name">
      <div class="row"><button class="secondary" onclick="doTest()">Test connection</button>
        <span id="testMsg" class="muted"></span></div>
      <div class="nav"><button class="secondary" onclick="go('install')">Back</button>
        <button onclick="saveModel()">Save & finish</button></div>
    </div>

    <div id="s-done" class="step">
      <h2>✓ All set</h2>
      <p>Select some code and run <b>Code Summary: Summarize Selection</b> (right-click or ⌘⇧P) to try it.</p>
      <div class="nav"><button onclick="close()">Close</button></div>
    </div>

    <script>
      const vscode = acquireVsCodeApi();
      const PRESETS = ${presets};
      let mode = "online";
      const STEPS = ["venv","torch","deps","assets","codebert","launch"];

      function go(id){ document.querySelectorAll(".step").forEach(e=>e.classList.remove("active"));
        document.getElementById("s-"+id).classList.add("active");
        if(id==="env"){ vscode.postMessage({type:"detectEnv"}); }
        if(id==="model"){ renderVendors(); } }
      function close(){ vscode.postMessage({type:"close"}); }

      function renderSteps(map){ const el=document.getElementById("steps");
        el.innerHTML = STEPS.map(s=>{ const st=map[s]||"pending";
          const icon = st==="done"?"✓":st==="error"?"✗":st==="running"?"…":st==="skipped"?"·":"○";
          const cls = st==="done"||st==="skipped"?"ok":st==="error"?"bad":"muted";
          return '<div class="'+cls+'">'+icon+' '+s+'</div>'; }).join(""); }

      const stepMap={};
      function startInstall(){ document.getElementById("installBtn").disabled=true;
        STEPS.forEach(s=>stepMap[s]="pending"); renderSteps(stepMap);
        document.getElementById("installLog").textContent="Installing… (first run downloads dependencies; this can take a few minutes)";
        vscode.postMessage({type:"startInstall"}); }

      function setMode(m){ mode=m;
        document.getElementById("tab-online").classList.toggle("sel",m==="online");
        document.getElementById("tab-offline").classList.toggle("sel",m==="offline");
        document.getElementById("keyWrap").style.display = m==="online"?"block":"none";
        renderVendors(); }
      function renderVendors(){ const list = mode==="online"?PRESETS.cloud:PRESETS.local;
        const sel=document.getElementById("vendor");
        sel.innerHTML=list.map(v=>'<option value="'+v.id+'">'+v.label+'</option>').join("");
        applyVendor(); }
      function applyVendor(){ const list = mode==="online"?PRESETS.cloud:PRESETS.local;
        const v=list.find(x=>x.id===document.getElementById("vendor").value)||list[0];
        document.getElementById("baseUrl").value=v.baseUrl;
        document.getElementById("model").value=v.defaultModel; }
      function payload(){ return { mode, base_url:document.getElementById("baseUrl").value,
        api_key: mode==="online"?document.getElementById("apiKey").value:"",
        model:document.getElementById("model").value }; }
      function doTest(){ document.getElementById("testMsg").textContent="Testing…";
        vscode.postMessage({type:"testConnection",payload:payload()}); }
      function saveModel(){ vscode.postMessage({type:"saveModel",payload:payload()}); }

      window.addEventListener("message", (ev)=>{ const m=ev.data;
        if(m.type==="envResult"){ const e=m.env; const py=e.python;
          document.getElementById("envBody").innerHTML =
            (py?'<div class="ok">✓ Python '+py.version.join(".")+'</div>'
               :'<div class="bad">✗ Python ≥3.10 not found — install from python.org and reopen.</div>')
            + (e.gpu?'<div class="ok">✓ NVIDIA GPU detected</div>':'<div class="muted">· No NVIDIA GPU — using CPU</div>');
          const dw=document.getElementById("deviceWrap"); const dev=document.getElementById("device");
          dw.style.display="block";
          if(e.gpu && !dev.querySelector('option[value="cuda"]')){ const o=document.createElement("option");
            o.value="cuda"; o.text="CUDA (NVIDIA GPU)"; dev.add(o); }
          dev.onchange=()=>vscode.postMessage({type:"selectDevice",device:dev.value});
          document.getElementById("envNext").disabled = !py; }
        if(m.type==="stepState"){ stepMap[m.step]=m.status; renderSteps(stepMap);
          if(m.detail) document.getElementById("installLog").textContent=m.step+": "+m.detail; }
        if(m.type==="installDone"){ document.getElementById("installLog").textContent="Backend ready at "+m.url;
          document.getElementById("installNext").disabled=false; }
        if(m.type==="installError"){ document.getElementById("installLog").textContent="Failed: "+m.message;
          document.getElementById("installBtn").disabled=false; }
        if(m.type==="testResult"){ const el=document.getElementById("testMsg");
          el.textContent=m.message; el.className=m.ok?"ok":"bad"; }
        if(m.type==="modelSaved"){ go("done"); }
      });
    </script></body></html>`;
  }
}
```

- [ ] **Step 2: Handle the `close` message** — add to the `switch` in `handle()`:
```ts
      case "close": this.panel.dispose(); break;
```

- [ ] **Step 3: Verify build/typecheck** from `extension/`: `npm run build` (clean), `npx tsc --noEmit` (clean), `npx vitest run` (still all green — wizard has no unit tests).

- [ ] **Step 4: Commit:**
```bash
git add extension/src/wizard.ts
git commit -m "feat(ext): interactive setup wizard webview (env/install/model steps)"
```

---

## Task 4: Open the wizard from the extension (command + first run)

**Files:**
- Modify: `extension/src/extension.ts`

- [ ] **Step 1:** Add `import { WizardPanel } from "./wizard";` to extension.ts imports.

- [ ] **Step 2:** Repoint the `codeSummary.setup` command from `provisionAndStart` to the wizard:
```ts
    vscode.commands.registerCommand("codeSummary.setup", () => WizardPanel.open(ctx)),
```

- [ ] **Step 3:** Add first-run auto-open at the end of `activate()` (after the `subscriptions.push(...)`), so a fresh install greets the user with the wizard instead of failing silently on first summarize:
```ts
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  if (!cfg.get("backend.managed", false)) {
    WizardPanel.open(ctx);
  }
```

- [ ] **Step 4: Verify** from `extension/`: `npm run build`, `npx tsc --noEmit`, `npx vitest run` — all clean/green.

- [ ] **Step 5: Commit:**
```bash
git add extension/src/extension.ts
git commit -m "feat(ext): open setup wizard on first run and via Code Summary: Run Setup"
```

---

## Task 5: Package + manual smoke (verification)

**Files:** none.

- [ ] **Step 1:** From `extension/`: `npm run package` — `.vsix` builds with `wizard.ts` bundled (esbuild bundles all imports into `dist/extension.js`; no new file to whitelist).

- [ ] **Step 2 (manual, by the user):** F5 / install the `.vsix`. On first activation the wizard auto-opens. Walk: Get started → Environment check (shows Python 3.11 ✓, device selector) → Install (live per-step ✓: venv/torch/deps/assets/codebert/launch) → Model (pick DeepSeek, paste key, Test connection → OK) → Save & finish → Done. Then summarize a Ruby `fib`. Re-running **Code Summary: Run Setup** reopens the wizard; the install step is instant (idempotent — all steps "skipped" but launch).

- [ ] **Step 3 (optional):** capture a sentence in `extension/README.md` documenting the wizard entry point. Commit if changed.

---

## Self-Review

**Spec §5 coverage:**
- §5.1 steps welcome → env check → install (streamed) → model config (cloud/local) → done → Task 3 (all five `.step` sections) + Task 4 (auto-open). ✓
- §5.1 device selector (CPU default, CUDA only if detected) → Task 3 `envResult` handler adds the CUDA option only when `env.gpu`. ✓
- §5.2 postMessage protocol (detectEnv/selectDevice/startInstall/saveModel/testConnection ↔ envResult/stepState/installDone/installError/testResult/modelSaved) → Task 3 `handle()` + the webview `message` listener. ✓
- §5.1 model config writes `codeSummary.online.*/offline.*/mode` → Task 3 `saveModel`. ✓
- "reopen via Code Summary: Run Setup" + first-run auto-open → Task 4. ✓
- Engine reuse (no duplicate provisioning logic) → Task 1 extracts `runProvision(ctx, onStep)`, used by BOTH `provisionAndStart` (console) and the wizard (postMessage). ✓

**Placeholder scan:** No TBD/TODO. Every step has complete code or exact commands. The webview HTML/JS is provided in full.

**Type consistency:** `OnStep` in backend.ts matches `ProvDeps.onStep` shape `(step: Step, status: StepStatus, detail?: string)`. `ModelConfig` shared by models.ts/backend.ts. `detectEnv()` returns `EnvInfo` consumed by the wizard `envResult` handler. `runProvision(ctx, onStep)` signature identical in both call sites.

**Ordering note (important):** Implement **Task 2 (models.ts) before Task 1 (backend.ts)**, because backend.ts imports `./models`. The task numbering reflects narrative order, not execution order — the executor must do 2 → 1 → 3 → 4 → 5.

**Scope:** Focused on the wizard layer over the existing engine. No engine changes. SSE streaming, multi-IDE, marketplace polish remain out of scope (spec §10).
