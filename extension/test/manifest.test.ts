import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import { parseManifest } from "../src/assets";

describe("shipped manifest", () => {
  it("parses the Gitee-first v2 manifest and keeps existing asset hashes", () => {
    const raw = fs.readFileSync(
      path.join(__dirname, "../../backend/assets/manifest.json"), "utf8");
    expect(raw).not.toContain("REPLACE_WITH");
    const m = parseManifest(raw);
    expect(m.version).toBe(2);
    expect(m.sources?.[0].id).toBe("gitee");
    const names = m.assets.map((a) => a.name).sort();
    expect(names).toEqual(["codebert-base", "corpus/corpus_30k.jsonl", "extractor/pytorch_model.bin"]);
    const ckpt = m.assets.find((a) => a.name === "extractor/pytorch_model.bin")!;
    expect(ckpt.size).toBe(316071890);
    expect(ckpt.sha256).toBe(
      "638efb2ab2862843d987e98117ad679065c4037f7246793f1e9ec4c3f8477de8");
    expect(ckpt.parts?.length).toBeGreaterThan(1);
    expect(m.assets.find((a) => a.name === "codebert-base")?.parts?.length).toBeGreaterThan(1);
    const allParts = m.assets.flatMap((a) => a.parts ?? []);
    expect(allParts.length).toBeLessThanOrEqual(20);
    expect(allParts.every((p) => p.size < 100 * 1024 * 1024)).toBe(true);
  });
});
