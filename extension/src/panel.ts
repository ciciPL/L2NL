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
