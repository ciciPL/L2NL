import * as vscode from "vscode";
import { SummarizeResponse } from "./client";

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderBody(resp: SummarizeResponse): string {
  if (resp.error) {
    return `<h2>Failed at stage: ${esc(resp.failed_stage ?? "unknown")}</h2>
            <pre class="err">${esc(resp.error)}</pre>`;
  }
  let html = `<h1>Summary</h1><p class="summary">${esc(resp.summary)}</p>`;
  const t = resp.trace;
  if (!t) return html;

  const badge = (b: boolean, label: string) =>
    b ? `<span class="badge">${label}</span>` : "";
  html += `<details open><summary>① Pivot translation
    ${badge(t.translation.repaired, "repaired")}
    ${badge(t.translation.fell_back, "fell back")}
    <span class="score">score ${t.translation.selected_score.toFixed(3)}</span>
    </summary><pre>${esc(t.translation.pivot_code)}</pre></details>`;

  const exs = t.retrieved.map(
    (e) => `<li><b>[${e.score.toFixed(2)}]</b> ${esc(e.summary)}
            <pre>${esc(e.code)}</pre></li>`).join("");
  html += `<details><summary>② Retrieved examples (${t.retrieved.length})</summary>
           <ul>${exs}</ul></details>`;

  const blocks = t.core_blocks.map(
    (b) => `<li>[${b.block_type} ${b.prob.toFixed(2)}] ${esc(b.text)}</li>`).join("");
  html += `<details><summary>③ Core statement blocks (${t.core_blocks.length})</summary>
           <ul class="blocks">${blocks}</ul></details>`;

  html += `<details><summary>④ Final prompt</summary>
           <pre>${esc(t.prompt)}</pre></details>`;
  return html;
}

const STYLE = `
  body { font-family: var(--vscode-font-family); padding: 12px; }
  .summary { font-size: 1.1em; font-weight: 600; }
  pre { background: var(--vscode-textCodeBlock-background); padding: 8px;
        white-space: pre-wrap; border-radius: 4px; }
  .badge { background: var(--vscode-badge-background);
           color: var(--vscode-badge-foreground); border-radius: 4px;
           padding: 0 6px; margin-left: 6px; font-size: 0.8em; }
  .score { color: var(--vscode-descriptionForeground); margin-left: 6px; }
  .err { color: var(--vscode-errorForeground); }
  details { margin-top: 10px; } summary { cursor: pointer; font-weight: 600; }
`;

export class ResultPanel {
  private static current: vscode.WebviewPanel | undefined;

  static show(resp: SummarizeResponse) {
    const col = vscode.ViewColumn.Beside;
    if (!this.current) {
      this.current = vscode.window.createWebviewPanel(
        "codeSummaryResult", "Code Summary", col, { enableScripts: false });
      this.current.onDidDispose(() => (this.current = undefined));
    }
    this.current.webview.html =
      `<!DOCTYPE html><html><head><style>${STYLE}</style></head>
       <body>${renderBody(resp)}</body></html>`;
    this.current.reveal(col);
  }

  static loading() {
    if (!this.current) {
      this.current = vscode.window.createWebviewPanel(
        "codeSummaryResult", "Code Summary", vscode.ViewColumn.Beside,
        { enableScripts: false });
      this.current.onDidDispose(() => (this.current = undefined));
    }
    this.current.webview.html =
      `<!DOCTYPE html><html><head><style>${STYLE}</style></head>
       <body><p>Summarizing…</p></body></html>`;
  }
}
