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
      case "close":
        this.panel.dispose();
        break;
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
      <div class="nav"><button onclick="closeWizard()">Close</button></div>
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
      function closeWizard(){ vscode.postMessage({type:"close"}); }

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
