# UI Redesign (Pipeline Showpiece) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the result panel, setup wizard, and status bar into one "pipeline showpiece" visual language (a shared `ui.ts`), and fix two wizard UX bugs (prefill saved model config; never overwrite a non-empty key with empty).

**Architecture:** A new vscode-free `ui.ts` holds the visual language — `BASE_CSS` (theme-variable styles) plus pure HTML render helpers, all unit-tested. `panel.ts` (static, no scripts) and `wizard.ts` (interactive) both consume `BASE_CSS`; `panel.ts` uses the render helpers directly, `wizard.ts` mirrors the same `.cs-*` classes in its client JS. Pure logic (`mergeModelConfig`) lives in `models.ts` with tests.

**Tech Stack:** TypeScript, VS Code webviews (`--vscode-*` theme variables, no external fonts/icons — inline glyphs only), vitest, esbuild.

**Builds on:** main (Phase 2A+2B merged). Spec: `docs/superpowers/specs/2026-06-15-ui-redesign-pipeline-showpiece-design.md`.

**Repo convention:** feature branch off `main`; commit per task. All colors via `--vscode-*` vars (auto-adapts to the user's Light Modern + dark).

---

## File Structure

| File | Responsibility |
|---|---|
| `extension/src/ui.ts` | NEW. `escapeHtml`, `BASE_CSS`, pure helpers: `stepper`, `scoreBadge`, `warnBadges`, `scoreBar`, `highlightCoreBlocks`. vscode-free. |
| `extension/src/panel.ts` | Rewrite `renderBody`; `show(resp, source)`; consume `BASE_CSS` + helpers. `enableScripts:false`. |
| `extension/src/wizard.ts` | Apply `BASE_CSS`; JS stepper; segmented cloud/local; `requestInit`→`initModel` prefill; `mergeModelConfig` on save. |
| `extension/src/models.ts` | Add `mergeModelConfig(existing, form)` (pure). |
| `extension/src/extension.ts` | Pass `source` to `ResultPanel.show`; status-bar `setup` state. |
| Tests | `extension/test/ui.test.ts` (helpers); `extension/test/models.test.ts` (add `mergeModelConfig`). |

---

## Task 1: `ui.ts` — shared visual language + pure helpers

**Files:**
- Create: `extension/src/ui.ts`
- Test: `extension/test/ui.test.ts`

- [ ] **Step 1: Write the failing test** — `extension/test/ui.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { escapeHtml, stepper, scoreBadge, warnBadges, scoreBar, highlightCoreBlocks } from "../src/ui";

describe("escapeHtml", () => {
  it("escapes angle brackets and ampersands", () => {
    expect(escapeHtml("<a> & </a>")).toBe("&lt;a&gt; &amp; &lt;/a&gt;");
  });
});

describe("stepper", () => {
  it("marks done/active/error per step and joins with connectors", () => {
    const html = stepper([
      { label: "Translate", status: "done" },
      { label: "Retrieve", status: "active" },
      { label: "Extract", status: "error" },
    ]);
    expect(html).toContain("cs-step--done");
    expect(html).toContain("cs-step--active");
    expect(html).toContain("cs-step--error");
    expect((html.match(/cs-step__line/g) || []).length).toBe(2); // connectors between 3 steps
    expect(html).toContain("Translate");
  });
});

describe("scoreBadge / warnBadges", () => {
  it("formats the back-translation score to 2 decimals", () => {
    expect(scoreBadge(1)).toContain("1.00");
    expect(scoreBadge(0.8333)).toContain("0.83");
  });
  it("shows only the flags that are set", () => {
    expect(warnBadges(true, false)).toContain("repaired");
    expect(warnBadges(true, false)).not.toContain("fell back");
    expect(warnBadges(false, false)).toBe("");
  });
});

describe("scoreBar", () => {
  it("fills proportionally to max and prints the rounded score", () => {
    const html = scoreBar(35, 70);
    expect(html).toContain("width:50%");
    expect(html).toContain("35.0");
  });
  it("clamps and survives max=0", () => {
    expect(scoreBar(5, 0)).toContain("width:0%");
  });
});

describe("highlightCoreBlocks", () => {
  it("wraps matching lines with a highlight + prob chip, leaves others", () => {
    const code = "a = 1\nfor x in xs:\n    y += x";
    const html = highlightCoreBlocks(code, [{ text: "for x in xs:", prob: 0.62 }]);
    expect(html).toContain('cs-hl');
    expect(html).toContain("0.62");
    expect(html).toContain("a = 1");          // untouched line still present
    expect(html).not.toContain("a = 1</span><span class=\"cs-chip"); // not highlighted
  });
  it("escapes code content", () => {
    expect(highlightCoreBlocks("x = a<b", [])).toContain("a&lt;b");
  });
});
```

- [ ] **Step 2: Run** `cd extension && npx vitest run test/ui.test.ts` — expect FAIL (module missing).

- [ ] **Step 3: Write `extension/src/ui.ts`:**

```ts
// Shared visual language for the Code Summary webviews (pipeline showpiece).
// Pure / vscode-free: returns HTML/CSS strings. All colors are --vscode-* theme vars.

export type StepStatus = "done" | "active" | "pending" | "error";

export function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const MARK: Record<StepStatus, string> = { done: "✓", active: "●", pending: "○", error: "✕" };

export function stepper(steps: { label: string; status: StepStatus }[]): string {
  const items = steps.map((s, i) =>
    (i ? `<span class="cs-step__line"></span>` : "") +
    `<span class="cs-step cs-step--${s.status}">` +
    `<span class="cs-step__dot cs-step__dot--${s.status}">${MARK[s.status]}</span>` +
    `<span class="cs-step__label">${escapeHtml(s.label)}</span></span>`
  ).join("");
  return `<div class="cs-stepper">${items}</div>`;
}

export function scoreBadge(score: number): string {
  return `<span class="cs-badge cs-badge--score" title="back-translation consistency">↺ ${score.toFixed(2)}</span>`;
}

export function warnBadges(repaired: boolean, fellBack: boolean): string {
  const b = (on: boolean, label: string) =>
    on ? `<span class="cs-badge cs-badge--warn">${label}</span>` : "";
  return b(repaired, "repaired") + b(fellBack, "fell back");
}

export function scoreBar(score: number, max: number): string {
  const pct = max > 0 ? Math.max(0, Math.min(100, Math.round((score / max) * 100))) : 0;
  return `<span class="cs-scorebar"><span class="cs-scorebar__fill" style="width:${pct}%"></span></span>` +
    `<span class="cs-scorebar__num">${score.toFixed(1)}</span>`;
}

export function highlightCoreBlocks(
  pivotCode: string, blocks: { text: string; prob: number }[],
): string {
  const wanted = new Map<string, number>();
  for (const b of blocks) wanted.set(b.text.trim(), b.prob);
  return pivotCode.split("\n").map((line) => {
    const key = line.trim();
    if (key && wanted.has(key)) {
      return `<span class="cs-hl">${escapeHtml(line)}` +
        `<span class="cs-chip--prob">${wanted.get(key)!.toFixed(2)}</span></span>`;
    }
    return escapeHtml(line);
  }).join("\n");
}

export const BASE_CSS = `
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 14px 16px; line-height: 1.5; }
  h1 { font-size: 1.05em; font-weight: 600; margin: 0 0 8px; }
  .cs-summary { font-size: 1.25em; font-weight: 500; margin: 4px 0 16px; }
  .cs-label { font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--vscode-descriptionForeground); margin-bottom: 4px; }
  .cs-stepper { display: flex; align-items: center; gap: 4px; margin: 0 0 16px; flex-wrap: wrap; }
  .cs-step { display: inline-flex; align-items: center; gap: 6px; }
  .cs-step__dot { width: 18px; height: 18px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); }
  .cs-step__dot--done { background: var(--vscode-testing-iconPassed, #3fb950); color: var(--vscode-editor-background); }
  .cs-step__dot--active { background: var(--vscode-progressBar-background, var(--vscode-textLink-foreground)); color: var(--vscode-editor-background); }
  .cs-step__dot--error { background: var(--vscode-errorForeground); color: var(--vscode-editor-background); }
  .cs-step__dot--pending { background: transparent; border: 1px solid var(--vscode-descriptionForeground); color: var(--vscode-descriptionForeground); }
  .cs-step__label { font-size: 12px; color: var(--vscode-descriptionForeground); }
  .cs-step--done .cs-step__label, .cs-step--active .cs-step__label { color: var(--vscode-foreground); }
  .cs-step__line { flex: 1; min-width: 12px; height: 1px; background: var(--vscode-widget-border, var(--vscode-editorWidget-border, rgba(128,128,128,.35))); }
  .cs-badge { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 6px; margin-left: 6px; background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); }
  .cs-badge--score { background: color-mix(in srgb, var(--vscode-testing-iconPassed, #3fb950) 22%, transparent); color: var(--vscode-testing-iconPassed, #3fb950); }
  .cs-badge--warn { background: color-mix(in srgb, var(--vscode-editorWarning-foreground, #cca700) 22%, transparent); color: var(--vscode-editorWarning-foreground, #cca700); }
  .cs-card { background: var(--vscode-editorWidget-background); border: 1px solid var(--vscode-widget-border, transparent); border-radius: 8px; padding: 10px 12px; margin: 8px 0; }
  .cs-code { background: var(--vscode-textCodeBlock-background); padding: 8px 10px; border-radius: 6px; white-space: pre-wrap; font-family: var(--vscode-editor-font-family, monospace); font-size: .92em; overflow-x: auto; margin: 0; }
  .cs-split { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
  .cs-scorebar { display: inline-block; width: 80px; height: 6px; border-radius: 3px; background: var(--vscode-editorWidget-background); vertical-align: middle; overflow: hidden; }
  .cs-scorebar__fill { display: block; height: 100%; background: var(--vscode-progressBar-background, var(--vscode-textLink-foreground)); }
  .cs-scorebar__num { font-size: 11px; color: var(--vscode-descriptionForeground); margin-left: 6px; }
  .cs-hl { background: var(--vscode-editor-findMatchHighlightBackground, rgba(234,92,0,.22)); border-radius: 3px; }
  .cs-chip--prob { font-size: 10px; color: var(--vscode-descriptionForeground); margin-left: 6px; font-family: var(--vscode-editor-font-family, monospace); }
  .cs-blocklist { margin: 8px 0 0; padding-left: 18px; } .cs-blocklist li { margin: 2px 0; }
  .cs-err { color: var(--vscode-errorForeground); }
  details { margin-top: 10px; } summary { cursor: pointer; font-weight: 500; }
  input, select { background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border, transparent); border-radius: 4px; padding: 6px; box-sizing: border-box; width: 100%; margin: 4px 0 10px; }
  button { background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
  button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
  button:disabled { opacity: .5; cursor: default; }
  .cs-seg { display: inline-flex; border: 1px solid var(--vscode-widget-border, var(--vscode-input-border, rgba(128,128,128,.4))); border-radius: 6px; overflow: hidden; margin: 4px 0 12px; }
  .cs-seg__opt { padding: 5px 14px; cursor: pointer; font-size: 13px; background: var(--vscode-input-background); }
  .cs-seg__opt--sel { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
  .nav { margin-top: 18px; display: flex; gap: 8px; }
  .ok { color: var(--vscode-testing-iconPassed, #3fb950); } .muted { color: var(--vscode-descriptionForeground); }
`;
```

- [ ] **Step 4: Run** `cd extension && npx vitest run test/ui.test.ts` — expect PASS (8 assertions across 5 describes).

- [ ] **Step 5: Commit:**
```bash
git add extension/src/ui.ts extension/test/ui.test.ts
git commit -m "feat(ext): shared pipeline-showpiece visual language (ui.ts) + helpers"
```

---

## Task 2: `mergeModelConfig` in `models.ts`

**Files:**
- Modify: `extension/src/models.ts`
- Test: `extension/test/models.test.ts` (append)

- [ ] **Step 1: Append the failing test** to `extension/test/models.test.ts`:

```ts
import { mergeModelConfig } from "../src/models";

describe("mergeModelConfig", () => {
  const existing = { mode: "online", online: { baseUrl: "u", apiKey: "OLDKEY", model: "m" }, offline: { baseUrl: "lu", model: "lm" } };

  it("keeps the existing online key when the form key is blank", () => {
    const out = mergeModelConfig(existing, { mode: "online", base_url: "u2", api_key: "", model: "m2" });
    expect(out.online.apiKey).toBe("OLDKEY");
    expect(out.online.baseUrl).toBe("u2");
    expect(out.online.model).toBe("m2");
  });
  it("overwrites the key when the form provides one", () => {
    const out = mergeModelConfig(existing, { mode: "online", base_url: "u", api_key: "NEW", model: "m" });
    expect(out.online.apiKey).toBe("NEW");
  });
  it("writes offline fields and switches mode without touching the key", () => {
    const out = mergeModelConfig(existing, { mode: "offline", base_url: "http://local/v1", api_key: "", model: "lc" });
    expect(out.mode).toBe("offline");
    expect(out.offline.baseUrl).toBe("http://local/v1");
    expect(out.offline.model).toBe("lc");
    expect(out.online.apiKey).toBe("OLDKEY");
  });
});
```

- [ ] **Step 2: Run** `cd extension && npx vitest run test/models.test.ts` — expect FAIL (`mergeModelConfig` not exported).

- [ ] **Step 3: Append to `extension/src/models.ts`:**

```ts
export interface SavedModel {
  mode: string;
  online: { baseUrl: string; apiKey: string; model: string };
  offline: { baseUrl: string; model: string };
}
export interface ModelFormInput { mode: string; base_url: string; api_key: string; model: string; }

// Merge a wizard form submission into the saved config. Never blanks a key:
// if the online key field is empty, the existing key is preserved.
export function mergeModelConfig(existing: SavedModel, form: ModelFormInput): SavedModel {
  const next: SavedModel = {
    mode: form.mode,
    online: { ...existing.online },
    offline: { ...existing.offline },
  };
  if (form.mode === "online") {
    next.online.baseUrl = form.base_url;
    next.online.model = form.model;
    if (form.api_key) next.online.apiKey = form.api_key;
  } else {
    next.offline.baseUrl = form.base_url;
    next.offline.model = form.model;
  }
  return next;
}
```

- [ ] **Step 4: Run** `cd extension && npx vitest run test/models.test.ts` — expect PASS (all prior + 3 new).

- [ ] **Step 5: Commit:**
```bash
git add extension/src/models.ts extension/test/models.test.ts
git commit -m "feat(ext): mergeModelConfig — never overwrite a non-empty key with blank"
```

---

## Task 3: Rewrite `panel.ts` (pipeline showpiece) + thread source

**Files:**
- Modify: `extension/src/panel.ts` (full rewrite)
- Modify: `extension/src/extension.ts` (pass source to `ResultPanel.show`)

- [ ] **Step 1: Overwrite `extension/src/panel.ts` with:**

```ts
import * as vscode from "vscode";
import { SummarizeResponse } from "./client";
import {
  escapeHtml, BASE_CSS, stepper, scoreBadge, warnBadges, scoreBar, highlightCoreBlocks, StepStatus,
} from "./ui";

const PIPE = ["Translate", "Retrieve", "Extract", "Generate"];
const STAGE_INDEX: Record<string, number> = { translate: 0, retrieve: 1, extract: 2, generate: 3 };

function pipeSteps(resp: SummarizeResponse): { label: string; status: StepStatus }[] {
  const failAt = resp.error && resp.failed_stage ? (STAGE_INDEX[resp.failed_stage] ?? -1) : -1;
  return PIPE.map((label, i) => ({
    label,
    status: (failAt < 0 ? "done" : i < failAt ? "done" : i === failAt ? "error" : "pending") as StepStatus,
  }));
}

function renderBody(resp: SummarizeResponse, source: string): string {
  const head = stepper(pipeSteps(resp));
  if (resp.error) {
    return `${head}<div class="cs-card"><h1 class="cs-err">Failed at: ${escapeHtml(resp.failed_stage ?? "unknown")}</h1>
      <pre class="cs-code cs-err">${escapeHtml(resp.error)}</pre></div>`;
  }
  let html = `${head}<p class="cs-summary">${escapeHtml(resp.summary)}</p>`;
  const t = resp.trace;
  if (!t) return html;

  const cands = t.translation.candidates.length > 1
    ? `<details><summary>Candidates (${t.translation.candidates.length})</summary>${
        t.translation.candidates.map((c) => `<pre class="cs-code">${escapeHtml(c)}</pre>`).join("")}</details>`
    : "";
  html += `<details open><summary>① Pivot translation ${scoreBadge(t.translation.selected_score)} ${warnBadges(t.translation.repaired, t.translation.fell_back)}</summary>
    <div class="cs-split">
      <div><div class="cs-label">source</div><pre class="cs-code">${escapeHtml(source)}</pre></div>
      <div><div class="cs-label">python pivot</div><pre class="cs-code">${escapeHtml(t.translation.pivot_code)}</pre></div>
    </div>${cands}</details>`;

  const maxScore = Math.max(1, ...t.retrieved.map((e) => e.score));
  const exs = t.retrieved.map((e) =>
    `<div class="cs-card">${scoreBar(e.score, maxScore)}
      <div style="margin:6px 0 4px">${escapeHtml(e.summary)}</div>
      <pre class="cs-code">${escapeHtml(e.code)}</pre></div>`).join("");
  html += `<details><summary>② Retrieved examples (${t.retrieved.length})</summary>${exs}</details>`;

  const hl = highlightCoreBlocks(t.translation.pivot_code, t.core_blocks);
  const list = t.core_blocks.map((b) =>
    `<li><span class="cs-chip--prob">${b.prob.toFixed(2)}</span> ${escapeHtml(b.text)}</li>`).join("");
  html += `<details><summary>③ Core statement blocks (${t.core_blocks.length})</summary>
    <pre class="cs-code">${hl}</pre><ul class="cs-blocklist">${list}</ul></details>`;

  html += `<details><summary>④ Final prompt</summary><pre class="cs-code">${escapeHtml(t.prompt)}</pre></details>`;
  return html;
}

export class ResultPanel {
  private static current: vscode.WebviewPanel | undefined;

  private static ensure(): vscode.WebviewPanel {
    if (!this.current) {
      this.current = vscode.window.createWebviewPanel(
        "codeSummaryResult", "Code Summary", vscode.ViewColumn.Beside, { enableScripts: false });
      this.current.onDidDispose(() => (this.current = undefined));
    }
    return this.current;
  }

  static show(resp: SummarizeResponse, source = "") {
    const p = this.ensure();
    p.webview.html = `<!DOCTYPE html><html><head><style>${BASE_CSS}</style></head><body>${renderBody(resp, source)}</body></html>`;
    p.reveal(vscode.ViewColumn.Beside);
  }

  static loading() {
    const p = this.ensure();
    const head = stepper(PIPE.map((label) => ({ label, status: "pending" as StepStatus })));
    p.webview.html = `<!DOCTYPE html><html><head><style>${BASE_CSS}</style></head><body>${head}<p class="muted">Summarizing…</p></body></html>`;
  }
}
```

- [ ] **Step 2: Update the call site in `extension/src/extension.ts`.** In `summarizeSelection`, change:
```ts
    ResultPanel.show(resp);
```
to:
```ts
    ResultPanel.show(resp, code);
```
(`code` is already in scope — the selected text.)

- [ ] **Step 3: Verify** from `extension/`: `npm run build` (clean), `npx tsc --noEmit` (clean), `npx vitest run` (all green — panel has no unit tests; ui/models tests cover the logic).

- [ ] **Step 4: Commit:**
```bash
git add extension/src/panel.ts extension/src/extension.ts
git commit -m "feat(ext): pipeline-showpiece result panel (stepper, side-by-side pivot, highlighted core blocks)"
```

---

## Task 4: Rework `wizard.ts` (shared style + stepper + segmented + prefill + safe save)

**Files:**
- Modify: `extension/src/wizard.ts` (full replacement below)

- [ ] **Step 1: Overwrite `extension/src/wizard.ts` with:**

```ts
import * as vscode from "vscode";
import { detectEnv, runProvision, testConnection } from "./backend";
import { CLOUD_VENDORS, LOCAL_PRESETS, mergeModelConfig, SavedModel } from "./models";
import { BASE_CSS } from "./ui";

function currentModel(): SavedModel {
  const c = vscode.workspace.getConfiguration("codeSummary");
  return {
    mode: c.get("mode", "online"),
    online: { baseUrl: c.get("online.baseUrl", ""), apiKey: c.get("online.apiKey", ""), model: c.get("online.model", "") },
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
      case "requestInit":
        this.post({ type: "initModel", model: currentModel() });
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
      case "saveModel": {
        const merged = mergeModelConfig(currentModel(), msg.payload);
        await cfg.update("mode", merged.mode, vscode.ConfigurationTarget.Global);
        await cfg.update("online.baseUrl", merged.online.baseUrl, vscode.ConfigurationTarget.Global);
        await cfg.update("online.apiKey", merged.online.apiKey, vscode.ConfigurationTarget.Global);
        await cfg.update("online.model", merged.online.model, vscode.ConfigurationTarget.Global);
        await cfg.update("offline.baseUrl", merged.offline.baseUrl, vscode.ConfigurationTarget.Global);
        await cfg.update("offline.model", merged.offline.model, vscode.ConfigurationTarget.Global);
        this.post({ type: "modelSaved" });
        break;
      }
      case "testConnection":
        this.post({ type: "testResult", ...(await testConnection(msg.payload)) });
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
    </style></head><body>
    <div id="wstepper" class="cs-stepper"></div>

    <div id="s-welcome" class="step active">
      <h1>Code Summary — Setup</h1>
      <p>This wizard sets up a private local backend (a Python venv + paper pipeline assets, ~400&nbsp;MB) and your summarization model. Nothing is installed system-wide; removing the extension removes it all.</p>
      <div class="nav"><button onclick="go('env')">Get started</button></div>
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
      <div id="steps" class="cs-card steps"></div>
      <pre id="installLog" class="cs-code log muted">Idle.</pre>
      <div class="nav"><button id="installBtn" onclick="startInstall()">Install</button>
        <button id="installNext" disabled onclick="go('model')">Next</button></div>
    </div>

    <div id="s-model" class="step">
      <h2>Model</h2>
      <div class="cs-seg"><div id="seg-online" class="cs-seg__opt cs-seg__opt--sel" onclick="setMode('online')">Cloud API</div>
        <div id="seg-offline" class="cs-seg__opt" onclick="setMode('offline')">Local runtime</div></div>
      <div class="cs-label">provider</div><select id="vendor" onchange="applyVendor()"></select>
      <div class="cs-label">base URL</div><input id="baseUrl" placeholder="https://…/v1">
      <div id="keyWrap"><div class="cs-label">API key</div><input id="apiKey" type="password" placeholder="leave blank to keep existing"></div>
      <div class="cs-label">model</div><input id="model" placeholder="model name">
      <div style="display:flex;gap:8px;align-items:center"><button class="secondary" onclick="doTest()">Test connection</button>
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
      let saved = null;
      const STEPS = ["venv","torch","deps","assets","codebert","launch"];
      const ORDER = ["welcome","env","install","model","done"];
      const WLABELS = ["Welcome","Environment","Install","Model","Done"];
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
        if(id==="model"){ renderVendors(); } }
      function closeWizard(){ vscode.postMessage({type:"close"}); }

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
        document.getElementById("keyWrap").style.display = m==="online"?"block":"none";
        renderVendors(); }
      function renderVendors(){ const list = mode==="online"?PRESETS.cloud:PRESETS.local;
        const sel=document.getElementById("vendor");
        sel.innerHTML=list.map(v=>'<option value="'+v.id+'">'+v.label+'</option>').join(""); applyVendor(); }
      function applyVendor(){ const list = mode==="online"?PRESETS.cloud:PRESETS.local;
        const v=list.find(x=>x.id===document.getElementById("vendor").value)||list[0];
        document.getElementById("baseUrl").value=v.baseUrl;
        document.getElementById("model").value=v.defaultModel; }
      function prefill(){ if(!saved) return; mode = saved.mode || "online"; setMode(mode);
        const blk = mode==="online"?saved.online:saved.offline;
        if(blk.baseUrl) document.getElementById("baseUrl").value=blk.baseUrl;
        if(blk.model) document.getElementById("model").value=blk.model; }
      function payload(){ return { mode, base_url:document.getElementById("baseUrl").value,
        api_key: mode==="online"?document.getElementById("apiKey").value:"",
        model:document.getElementById("model").value }; }
      function doTest(){ document.getElementById("testMsg").textContent="Testing…";
        const p = payload();
        if(mode==="online" && !p.api_key && saved && saved.online) p.api_key = saved.online.apiKey;
        vscode.postMessage({type:"testConnection",payload:p}); }
      function saveModel(){ vscode.postMessage({type:"saveModel",payload:payload()}); }

      window.addEventListener("message",(ev)=>{ const m=ev.data;
        if(m.type==="initModel"){ saved=m.model; }
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
        if(m.type==="testResult"){ const el=document.getElementById("testMsg"); el.textContent=m.message; el.className=m.ok?"ok":"cs-err"; }
        if(m.type==="modelSaved"){ go("done"); }
      });

      renderStepper("welcome");
      vscode.postMessage({type:"requestInit"});
      const _origGoModel = (id)=>{ if(id==="model") prefill(); };
      const _go = go; go = function(id){ _go(id); _origGoModel(id); };
    </script></body></html>`;
  }
}
```

> Note the small `go` wrapper at the end calls `prefill()` when entering the model step (after `renderVendors()` ran), so saved base_url/model override the preset defaults. The key field stays blank by design (placeholder "leave blank to keep existing"); `mergeModelConfig` + the `doTest` fallback preserve the saved key.

- [ ] **Step 2: Verify** from `extension/`: `npm run build` (clean), `npx tsc --noEmit` (clean), `npx vitest run` (all green).

- [ ] **Step 3: Commit:**
```bash
git add extension/src/wizard.ts
git commit -m "feat(ext): wizard restyle (stepper, segmented, prefill saved config, safe key save)"
```

---

## Task 5: Status-bar `setup` state

**Files:**
- Modify: `extension/src/extension.ts`

- [ ] **Step 1:** Replace the `updateStatus` function with:
```ts
function updateStatus() {
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  if (!cfg.get("backend.managed", false)) {
    statusItem.text = "$(rocket) Summary: setup";
    statusItem.tooltip = "Run Code Summary setup";
    statusItem.command = "codeSummary.setup";
    statusItem.show();
    return;
  }
  const s = readSettings();
  const icon = s.mode === "offline" ? "$(vm)" : "$(cloud)";
  const model = s.mode === "offline" ? s.offline.model : s.online.model;
  statusItem.text = `${icon} Summary: ${s.mode} · ${model}`;
  statusItem.tooltip = "Toggle online/offline";
  statusItem.command = "codeSummary.toggleMode";
  statusItem.show();
}
```

- [ ] **Step 2:** In `activate`, remove the line `statusItem.command = "codeSummary.toggleMode";` (the command is now set inside `updateStatus`). Leave the rest.

- [ ] **Step 3: Verify** from `extension/`: `npm run build`, `npx tsc --noEmit`, `npx vitest run` — all clean/green.

- [ ] **Step 4: Commit:**
```bash
git add extension/src/extension.ts
git commit -m "feat(ext): status bar shows a setup affordance until the backend is provisioned"
```

---

## Task 6: Package + manual smoke (verification)

- [ ] **Step 1:** From `extension/`: `npm run package` — `.vsix` builds (esbuild bundles `ui.ts` into `dist/extension.js`; no new packaged file).

- [ ] **Step 2 (manual, user):** install/F5. (a) Re-run a summary on the `merge_intervals` example → panel shows the pipeline stepper, summary hero, source↔pivot side-by-side with the `↺` score badge, retrieved cards with score bars, core blocks highlighted on the pivot code with prob chips. (b) `Code Summary: Run Setup` → wizard shows the progress stepper, segmented cloud/local, and the model step is **prefilled** (base_url/model from settings; key blank). Saving with a blank key **keeps** the existing key. (c) Before provisioning (fresh install), the status bar reads `Summary: setup` and clicking it opens the wizard.

---

## Self-Review

**Spec coverage:**
- §1 ui.ts (escapeHtml, BASE_CSS, stepper, scoreBadge, warnBadges, scoreBar, highlightCoreBlocks) → Task 1, all unit-tested. ✓
- §2 result panel (stepper, summary hero, ① side-by-side + score badges, ② cards + score bars, ③ highlighted core blocks, ④ prompt, loading/error states, no-scripts) → Task 3. ✓ (source threaded via `show(resp, source)`).
- §3 wizard (shared CSS, stepper, segmented, prefill via requestInit/initModel, safe save via mergeModelConfig) → Tasks 2 + 4. ✓
- §4 status bar setup state → Task 5. ✓
- §6 tests (pure helpers + mergeModelConfig via vitest; webviews manual) → Tasks 1, 2 + Task 6. ✓

**Placeholder scan:** No TBD/placeholder. All steps have complete code/commands. The `go` reassignment wrapper in Task 4 is real code with a rationale note, not a placeholder.

**Type consistency:** `StepStatus` exported from ui.ts, reused in panel.ts. `SavedModel`/`ModelFormInput` defined in models.ts (Task 2) and consumed by wizard.ts (Task 4). `mergeModelConfig(existing, form)` signature identical across test (Task 2) and wizard call (Task 4). `ResultPanel.show(resp, source)` matches the updated call site in extension.ts (Task 3). Wizard message types (requestInit/initModel, detectEnv/envResult, startInstall/stepState/installDone/installError, saveModel/modelSaved, testConnection/testResult, selectDevice, close) are paired between `handle()` and the webview listener.

**Execution-order note:** Task 2 (models.mergeModelConfig) before Task 4 (wizard imports it). Task 1 (ui.ts) before Tasks 3 & 4. Suggested order: 1 → 2 → 3 → 4 → 5 → 6.
