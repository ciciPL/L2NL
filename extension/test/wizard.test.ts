import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it } from "vitest";

function wizardSource(): string {
  return readFileSync(new URL("../src/wizard.ts", import.meta.url), "utf8");
}

function inlineWizardScript(): string {
  const match = wizardSource().match(/<script>([\s\S]*?)<\/script>/);
  expect(match).not.toBeNull();
  return match![1].replace("const PRESETS = ${presets};", "const PRESETS = { cloud: [], local: [] };");
}

describe("setup wizard webview", () => {
  it("ships a syntactically valid inline script", () => {
    expect(() => new vm.Script(inlineWizardScript())).not.toThrow();
  });

  it("has real selectable setup and asset-source controls", () => {
    const src = wizardSource();
    expect(src).toContain('id="modeCardOnline"');
    expect(src).toContain('id="modeCardOffline"');
    expect(src).toContain('id="modeStatus"');
    expect(src).toContain('id="assetGiteeBtn"');
    expect(src).toContain('id="assetLocalBtn"');
    expect(src).toContain("function renderModeChoice()");
    expect(src).toContain("function renderAssetChoice()");
  });

  it("supports opening directly to online or offline model configuration", () => {
    const src = wizardSource();
    expect(src).toContain("WizardOpenOptions");
    expect(src).toContain("initialMode");
    expect(src).toContain("initialStep");
    expect(src).toContain('type==="openTarget"');
  });
});
