import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import { parseManifest } from "../src/assets";

describe("shipped manifest", () => {
  it("parses and lists both provisioner assets with real hashes", () => {
    const raw = fs.readFileSync(
      path.join(__dirname, "../../backend/assets/manifest.json"), "utf8");
    const m = parseManifest(raw);
    const names = m.assets.map((a) => a.name).sort();
    expect(names).toEqual(["corpus/corpus_30k.jsonl", "extractor/pytorch_model.bin"]);
    const ckpt = m.assets.find((a) => a.name === "extractor/pytorch_model.bin")!;
    expect(ckpt.size).toBe(316071890);
    expect(ckpt.sha256).toBe(
      "638efb2ab2862843d987e98117ad679065c4037f7246793f1e9ec4c3f8477de8");
  });
});
