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
    expect((html.match(/cs-step__line/g) || []).length).toBe(2);
    expect(html).toContain("Translate");
  });
});

describe("scoreBadge / warnBadges", () => {
  it("formats the back-translation score to 2 decimals", () => {
    expect(scoreBadge(1)).toContain("1.00");
    expect(scoreBadge(0.8333)).toContain("0.83");
  });
  it("marks unavailable selection scores without pretending a real score exists", () => {
    expect(scoreBadge(null)).toContain("n/a");
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
    expect(html).toContain("a = 1");
    expect(html).not.toContain("a = 1</span><span class=\"cs-chip");
  });
  it("escapes code content", () => {
    expect(highlightCoreBlocks("x = a<b", [])).toContain("a&lt;b");
  });
});
