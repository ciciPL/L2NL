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
  .cs-step--error .cs-step__label { color: var(--vscode-errorForeground); }
  .cs-step__line { flex: 1; min-width: 12px; height: 1px; background: var(--vscode-widget-border, var(--vscode-editorWidget-border, rgba(128,128,128,.35))); }
  .cs-badge { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 6px; margin-left: 6px; background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); }
  .cs-badge--score { background: color-mix(in srgb, var(--vscode-testing-iconPassed, #3fb950) 22%, transparent); color: var(--vscode-testing-iconPassed, #3fb950); }
  .cs-badge--warn { background: color-mix(in srgb, var(--vscode-editorWarning-foreground, #cca700) 22%, transparent); color: var(--vscode-editorWarning-foreground, #cca700); }
  .cs-card { background: var(--vscode-editorWidget-background); border: 1px solid var(--vscode-widget-border, transparent); border-radius: 8px; padding: 10px 12px; margin: 8px 0; }
  .cs-code { background: var(--vscode-textCodeBlock-background); padding: 8px 10px; border-radius: 6px; white-space: pre-wrap; font-family: var(--vscode-editor-font-family, monospace); font-size: .92em; overflow-x: auto; margin: 0; }
  .cs-split { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
  .cs-scorebar { display: inline-block; width: 80px; height: 6px; border-radius: 3px; background: var(--vscode-editorWidget-background); vertical-align: middle; overflow: hidden; }
  .cs-scorebar__fill { display: block; height: 100%; background: var(--vscode-progressBar-background, var(--vscode-textLink-foreground, #0078d4)); }
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
