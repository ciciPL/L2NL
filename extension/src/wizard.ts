import * as vscode from "vscode";
import { detectEnv, prepareOfflineLlama, runProvision, searchModelScopeGguf, testConnection } from "./backend";
import { CLOUD_VENDORS, LOCAL_PRESETS, mergeModelConfig, SavedModel } from "./models";
import { BASE_CSS } from "./ui";
import { loadOnlineApiKey, saveOnlineApiKey } from "./secrets";
import { MODEL_SCOPE_PRESETS } from "./localRuntime";

async function currentModel(ctx: vscode.ExtensionContext): Promise<SavedModel> {
  const c = vscode.workspace.getConfiguration("codeSummary");
  return {
    mode: c.get("mode", "online"),
    online: {
      baseUrl: c.get("online.baseUrl", ""),
      apiKey: await loadOnlineApiKey(ctx.secrets, c.get("online.apiKey", "")),
      model: c.get("online.model", ""),
    },
    offline: { baseUrl: c.get("offline.baseUrl", ""), model: c.get("offline.model", "") },
  };
}

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
      case "requestInit": {
        const model = await currentModel(ctx);
        this.post({
          type: "initModel",
          model: { ...model, online: { ...model.online, apiKey: "" } },
          assets: {
            localDir: cfg.get("assets.localDir", ""),
            tryGlobalMirrors: cfg.get("assets.tryGlobalMirrors", false),
          },
          modelScopePresets: MODEL_SCOPE_PRESETS,
        });
        break;
      }
      case "selectSetupMode":
        await cfg.update("mode", msg.mode === "offline" ? "offline" : "online", vscode.ConfigurationTarget.Global);
        this.post({ type: "setupMode", mode: msg.mode === "offline" ? "offline" : "online" });
        break;
      case "detectEnv":
        this.post({ type: "envResult", env: await detectEnv() });
        break;
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
      case "pickLocalAssetDir": {
        const dirs = await vscode.window.showOpenDialog({
          canSelectFiles: false,
          canSelectFolders: true,
          canSelectMany: false,
          title: "Select Code Summary asset folder",
        });
        const selected = dirs?.[0]?.fsPath ?? "";
        if (selected) {
          await cfg.update("assets.localDir", selected, vscode.ConfigurationTarget.Global);
          this.post({ type: "assetSource", localDir: selected });
        }
        break;
      }
      case "setAssetOptions":
        await cfg.update("assets.tryGlobalMirrors", !!msg.tryGlobalMirrors, vscode.ConfigurationTarget.Global);
        break;
      case "useGiteeAssets":
        await cfg.update("assets.localDir", "", vscode.ConfigurationTarget.Global);
        this.post({ type: "assetSource", localDir: "" });
        break;
      case "saveModel": {
        const merged = mergeModelConfig(await currentModel(ctx), msg.payload);
        await cfg.update("mode", merged.mode, vscode.ConfigurationTarget.Global);
        await cfg.update("online.baseUrl", merged.online.baseUrl, vscode.ConfigurationTarget.Global);
        await saveOnlineApiKey(ctx.secrets, merged.online.apiKey);
        await cfg.update("online.apiKey", "", vscode.ConfigurationTarget.Global);
        await cfg.update("online.model", merged.online.model, vscode.ConfigurationTarget.Global);
        await cfg.update("offline.baseUrl", merged.offline.baseUrl, vscode.ConfigurationTarget.Global);
        await cfg.update("offline.model", merged.offline.model, vscode.ConfigurationTarget.Global);
        this.post({ type: "modelSaved" });
        break;
      }
      case "testConnection": {
        const payload = { ...msg.payload };
        if (payload.mode === "online" && !payload.api_key) {
          payload.api_key = (await currentModel(ctx)).online.apiKey;
        }
        this.post({ type: "testResult", ...(await testConnection(payload)) });
        break;
      }
      case "searchModelScope":
        try {
          this.post({ type: "modelScopeResult", ...(await searchModelScopeGguf(msg.query ?? "")) });
        } catch (e: any) {
          this.post({ type: "modelScopeError", message: e.message });
        }
        break;
      case "prepareOfflineLlama":
        try {
          const result = await prepareOfflineLlama(ctx, msg.selection, (step, status, detail) =>
            this.post({ type: "offlineStep", step, status, detail }));
          this.post({ type: "offlineReady", result });
        } catch (e: any) {
          this.post({ type: "offlineError", message: e.message });
        }
        break;
      case "close":
        this.panel.dispose();
        break;
    }
  }

  private html(): string {
    const presets = JSON.stringify({ cloud: CLOUD_VENDORS, local: LOCAL_PRESETS });
    return /* html */ `<!DOCTYPE html><html><head><meta charset="utf-8"><style>${BASE_CSS}
      .step { display: none; } .step.active { display: block; }
      .steps div { padding: 2px 0; }
      pre.log { max-height: 160px; overflow: auto; }
      .choice { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
      .choice button { width:100%; text-align:left; min-height:82px; }
      .choice strong { display:block; margin-bottom:4px; }
      .subtle { font-size:12px; color:var(--vscode-descriptionForeground); }
      .offline-only { display:none; }
      .online-only { display:block; }
    </style></head><body>
    <div id="wstepper" class="cs-stepper"></div>

    <div id="s-welcome" class="step active">
      <h1>Code Summary — Setup</h1>
      <p>This wizard sets up a private local backend (a Python venv + paper pipeline assets, ~400&nbsp;MB) and your summarization model. Nothing is installed system-wide; removing the extension removes it all.</p>
      <div class="nav"><button onclick="go('mode')">Get started</button></div>
    </div>

    <div id="s-mode" class="step">
      <h2>Choose model mode</h2>
      <div class="choice">
        <button class="secondary" onclick="chooseSetupMode('online')">
          <strong>Online API</strong>
          <span class="subtle">DeepSeek, OpenAI, or any OpenAI-compatible endpoint. API key is stored in VS Code SecretStorage.</span>
        </button>
        <button class="secondary" onclick="chooseSetupMode('offline')">
          <strong>Offline llama.cpp</strong>
          <span class="subtle">Use ModelScope GGUF models and a local llama-server on 127.0.0.1.</span>
        </button>
      </div>
      <div class="nav"><button class="secondary" onclick="go('welcome')">Back</button>
        <button onclick="go('env')">Next</button></div>
    </div>

    <div id="s-env" class="step">
      <h2>Environment check</h2>
      <div id="envBody" class="cs-card muted">Checking…</div>
      <div id="deviceWrap" style="display:none">
        <div class="cs-label">inference device</div>
        <select id="device"><option value="cpu">CPU (works everywhere)</option></select>
      </div>
      <div class="nav"><button class="secondary" onclick="go('welcome')">Back</button>
        <button id="envNext" disabled onclick="go('install')">Next</button></div>
    </div>

    <div id="s-install" class="step">
      <h2>Install backend</h2>
      <div class="cs-card">
        <div class="cs-label">asset source</div>
        <div class="nav"><button class="secondary" onclick="useGitee()">Online via Gitee</button>
          <button class="secondary" onclick="pickLocalAssets()">Use local asset folder</button>
          <button class="secondary" onclick="retryParts()">Retry failed parts</button></div>
        <div id="assetSource" class="muted">Gitee Release multipart assets</div>
        <label><input id="globalMirrors" type="checkbox" onchange="setAssetOptions()"> try global mirrors</label>
      </div>
      <div id="steps" class="cs-card steps"></div>
      <pre id="installLog" class="cs-code log muted">Idle.</pre>
      <div class="nav"><button id="installBtn" onclick="startInstall()">Install</button>
        <button id="installNext" disabled onclick="go('model')">Next</button></div>
    </div>

    <div id="s-model" class="step">
      <h2>Model</h2>
      <div class="cs-seg"><div id="seg-online" class="cs-seg__opt cs-seg__opt--sel" onclick="setMode('online')">Online API</div>
        <div id="seg-offline" class="cs-seg__opt" onclick="setMode('offline')">Offline llama.cpp</div></div>

      <div id="onlineBox" class="online-only">
        <div class="cs-label">provider</div><select id="vendor" onchange="applyVendor()"></select>
        <div class="cs-label">base URL</div><input id="baseUrl" placeholder="https://…/v1">
        <div id="keyWrap"><div class="cs-label">API key</div><input id="apiKey" type="password" placeholder="leave blank to keep existing"></div>
        <div class="cs-label">model</div><input id="model" placeholder="model name">
        <div style="display:flex;gap:8px;align-items:center"><button class="secondary" onclick="doTest()">Test connection</button>
          <span id="testMsg" class="muted"></span></div>
      </div>

      <div id="offlineBox" class="offline-only">
        <div class="cs-label">ModelScope model</div>
        <select id="offlinePreset" onchange="selectOfflinePreset()"></select>
        <div id="customModelWrap" style="display:none">
          <div class="cs-label">ModelScope repo ID or keyword</div>
          <input id="modelScopeQuery" placeholder="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF">
          <button class="secondary" onclick="searchModelScope()">Search GGUF files</button>
        </div>
        <div class="cs-label">GGUF file</div>
        <select id="ggufFile"></select>
        <pre id="offlineLog" class="cs-code log muted">Choose a preset or search ModelScope.</pre>
        <div class="nav"><button id="offlineStartBtn" onclick="prepareOffline()">Download model & start llama.cpp</button></div>
      </div>

      <div class="nav"><button class="secondary" onclick="go('install')">Back</button>
        <button id="saveModelBtn" onclick="saveModel()">Save & finish</button></div>
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
      let saved = null;
      let assetState = { localDir:"", tryGlobalMirrors:false };
      let modelScopePresets = [];
      let modelScopeFiles = [];
      const STEPS = ["venv","torch","deps","assets","codebert","launch"];
      const ORDER = ["welcome","mode","env","install","model","done"];
      const WLABELS = ["Welcome","Mode","Environment","Install","Model","Done"];
      const MARK = { done:"✓", active:"●", pending:"○", error:"✕" };

      function renderStepper(id){ const ai = ORDER.indexOf(id);
        document.getElementById("wstepper").innerHTML = WLABELS.map((label,i)=>{
          const st = i<ai?"done":i===ai?"active":"pending";
          return (i?'<span class="cs-step__line"></span>':'')+
            '<span class="cs-step cs-step--'+st+'"><span class="cs-step__dot cs-step__dot--'+st+'">'+MARK[st]+'</span><span class="cs-step__label">'+label+'</span></span>';
        }).join(""); }

      function go(id){ document.querySelectorAll(".step").forEach(e=>e.classList.remove("active"));
        document.getElementById("s-"+id).classList.add("active"); renderStepper(id);
        if(id==="env"){ vscode.postMessage({type:"detectEnv"}); }
        if(id==="model"){ prefill(); } }
      function chooseSetupMode(m){ setMode(m); vscode.postMessage({type:"selectSetupMode",mode:m}); }
      function closeWizard(){ vscode.postMessage({type:"close"}); }
      function useGitee(){ assetState.localDir=""; document.getElementById("assetSource").textContent="Gitee Release multipart assets"; vscode.postMessage({type:"useGiteeAssets"}); }
      function pickLocalAssets(){ vscode.postMessage({type:"pickLocalAssetDir"}); }
      function retryParts(){ startInstall(); }
      function setAssetOptions(){ const v=document.getElementById("globalMirrors").checked;
        assetState.tryGlobalMirrors=v; vscode.postMessage({type:"setAssetOptions",tryGlobalMirrors:v}); }

      function renderSteps(map){ document.getElementById("steps").innerHTML = STEPS.map(s=>{
          const st=map[s]||"pending";
          const cls = st==="done"||st==="skipped"?"ok":st==="error"?"cs-err":"muted";
          return '<div class="'+cls+'">'+(MARK[st]||(st==="skipped"?"·":"○"))+' '+s+'</div>'; }).join(""); }
      const stepMap={};
      function startInstall(){ document.getElementById("installBtn").disabled=true;
        STEPS.forEach(s=>stepMap[s]="pending"); renderSteps(stepMap);
        document.getElementById("installLog").textContent="Installing… (first run downloads dependencies; this can take a few minutes)";
        vscode.postMessage({type:"startInstall"}); }

      function setMode(m){ mode=m;
        document.getElementById("seg-online").classList.toggle("cs-seg__opt--sel",m==="online");
        document.getElementById("seg-offline").classList.toggle("cs-seg__opt--sel",m==="offline");
        document.getElementById("onlineBox").style.display = m==="online"?"block":"none";
        document.getElementById("offlineBox").style.display = m==="offline"?"block":"none";
        document.getElementById("saveModelBtn").style.display = m==="online"?"inline-block":"none";
        renderVendors(); }
      function renderVendors(){ const list = mode==="online"?PRESETS.cloud:PRESETS.local;
        const sel=document.getElementById("vendor");
        sel.innerHTML=list.map(v=>'<option value="'+v.id+'">'+v.label+'</option>').join(""); applyVendor(); }
      function applyVendor(){ const list = mode==="online"?PRESETS.cloud:PRESETS.local;
        const v=list.find(x=>x.id===document.getElementById("vendor").value)||list[0];
        document.getElementById("baseUrl").value=v.baseUrl;
        document.getElementById("model").value=v.defaultModel; }
      function prefill(){ mode = (saved && saved.mode) || "online"; setMode(mode);
        if(!saved) return;
        const blk = mode==="online"?saved.online:saved.offline;
        if(blk && blk.baseUrl) document.getElementById("baseUrl").value=blk.baseUrl;
        if(blk && blk.model) document.getElementById("model").value=blk.model; }
      function payload(){ return { mode, base_url:document.getElementById("baseUrl").value,
        api_key: mode==="online"?document.getElementById("apiKey").value:"",
        model:document.getElementById("model").value }; }
      function doTest(){ document.getElementById("testMsg").textContent="Testing…";
        const p = payload();
        if(mode==="online" && !p.api_key && saved && saved.online) p.api_key = saved.online.apiKey;
        vscode.postMessage({type:"testConnection",payload:p}); }
      function saveModel(){ vscode.postMessage({type:"saveModel",payload:payload()}); }
      function renderOfflinePresets(){
        const sel=document.getElementById("offlinePreset");
        sel.innerHTML = modelScopePresets.map((p,i)=>'<option value="'+i+'">'+p.label+'</option>').join("")
          + '<option value="custom">Custom ModelScope repo</option>';
        selectOfflinePreset();
      }
      function selectOfflinePreset(){
        const v=document.getElementById("offlinePreset").value;
        document.getElementById("customModelWrap").style.display = v==="custom"?"block":"none";
        if(v==="custom"){ modelScopeFiles=[]; renderGgufFiles([]); return; }
        const p=modelScopePresets[Number(v)];
        modelScopeFiles = p ? [{ path:p.filePath, size:p.size, sha256:p.sha256, modelId:p.modelId }] : [];
        renderGgufFiles(modelScopeFiles);
      }
      function renderGgufFiles(files){
        const sel=document.getElementById("ggufFile");
        sel.innerHTML = files.map((f,i)=>'<option value="'+i+'">'+f.path+' ('+formatBytes(f.size)+')</option>').join("");
      }
      function formatBytes(n){ if(!n) return "unknown size"; const u=["B","KiB","MiB","GiB"]; let i=0; while(n>=1024&&i<u.length-1){n/=1024;i++;} return n.toFixed(i?1:0)+" "+u[i]; }
      function searchModelScope(){
        const q=document.getElementById("modelScopeQuery").value.trim();
        document.getElementById("offlineLog").textContent="Searching ModelScope…";
        vscode.postMessage({type:"searchModelScope",query:q});
      }
      function selectedOfflineModel(){
        const pidx=document.getElementById("offlinePreset").value;
        const fidx=Number(document.getElementById("ggufFile").value||"0");
        if(pidx!=="custom"){
          const p=modelScopePresets[Number(pidx)];
          return p && { modelId:p.modelId, filePath:p.filePath, size:p.size, sha256:p.sha256 };
        }
        const modelId=document.getElementById("modelScopeQuery").value.trim();
        const f=modelScopeFiles[fidx];
        return modelId && f && { modelId, filePath:f.path, size:f.size, sha256:f.sha256 };
      }
      function prepareOffline(){
        const selection=selectedOfflineModel();
        if(!selection){ document.getElementById("offlineLog").textContent="Choose a ModelScope GGUF file first."; return; }
        document.getElementById("offlineStartBtn").disabled=true;
        document.getElementById("offlineLog").textContent="Preparing local runtime…";
        vscode.postMessage({type:"prepareOfflineLlama",selection});
      }

      window.addEventListener("message",(ev)=>{ const m=ev.data;
        if(m.type==="initModel"){ saved=m.model; assetState=m.assets||assetState; modelScopePresets=m.modelScopePresets||[];
          renderOfflinePresets();
          if(assetState.localDir) document.getElementById("assetSource").textContent=assetState.localDir;
          document.getElementById("globalMirrors").checked=!!assetState.tryGlobalMirrors; }
        if(m.type==="setupMode"){ mode=m.mode; setMode(mode); }
        if(m.type==="envResult"){ const e=m.env; const py=e.python;
          document.getElementById("envBody").innerHTML =
            (py?'<div class="ok">✓ Python '+py.version.join(".")+'</div>'
               :'<div class="cs-err">✕ Python ≥3.10 not found — install from python.org and reopen.</div>')
            + (e.gpu?'<div class="ok">✓ NVIDIA GPU detected</div>':'<div class="muted">· No NVIDIA GPU — using CPU</div>');
          const dw=document.getElementById("deviceWrap"); const dev=document.getElementById("device");
          dw.style.display="block";
          if(e.gpu && !dev.querySelector('option[value="cuda"]')){ const o=document.createElement("option"); o.value="cuda"; o.text="CUDA (NVIDIA GPU)"; dev.add(o); }
          dev.onchange=()=>vscode.postMessage({type:"selectDevice",device:dev.value});
          document.getElementById("envNext").disabled = !py; }
        if(m.type==="stepState"){ stepMap[m.step]=m.status; renderSteps(stepMap);
          if(m.detail) document.getElementById("installLog").textContent=m.step+": "+m.detail; }
        if(m.type==="installDone"){ document.getElementById("installLog").textContent="Backend ready at "+m.url;
          document.getElementById("installNext").disabled=false; }
        if(m.type==="installError"){ document.getElementById("installLog").textContent="Failed: "+m.message;
          document.getElementById("installBtn").disabled=false; }
        if(m.type==="assetSource"){ assetState.localDir=m.localDir; document.getElementById("assetSource").textContent=m.localDir||"Gitee Release multipart assets"; }
        if(m.type==="testResult"){ const el=document.getElementById("testMsg"); el.textContent=m.message; el.className=m.ok?"ok":"cs-err"; }
        if(m.type==="modelScopeResult"){
          if(m.files && m.files.length){ modelScopeFiles=m.files; renderGgufFiles(modelScopeFiles); document.getElementById("offlineLog").textContent="Found "+m.files.length+" GGUF file(s)."; }
          else if(m.presets && m.presets.length){ modelScopePresets=m.presets; renderOfflinePresets(); document.getElementById("offlineLog").textContent="Found "+m.presets.length+" preset match(es)."; }
          else document.getElementById("offlineLog").textContent=m.searchUrl ? "No direct GGUF result. Open ModelScope search and paste a repo ID: "+m.searchUrl : "No GGUF files found.";
        }
        if(m.type==="modelScopeError"){ document.getElementById("offlineLog").textContent="ModelScope search failed: "+m.message; }
        if(m.type==="offlineStep"){ document.getElementById("offlineLog").textContent=m.step+": "+m.status+(m.detail?" — "+m.detail:""); }
        if(m.type==="offlineReady"){ document.getElementById("offlineLog").textContent="Local runtime ready at "+m.result.baseUrl+" using "+m.result.model; document.getElementById("offlineStartBtn").disabled=false; go("done"); }
        if(m.type==="offlineError"){ document.getElementById("offlineLog").textContent="Offline setup failed: "+m.message; document.getElementById("offlineStartBtn").disabled=false; }
        if(m.type==="modelSaved"){ go("done"); }
      });

      renderStepper("welcome");
      vscode.postMessage({type:"requestInit"});
    </script></body></html>`;
  }
}
