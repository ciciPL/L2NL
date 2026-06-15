import { describe, it, expect } from "vitest";
import * as os from "node:os";
import * as fs from "node:fs";
import * as path from "node:path";
import { parseManifest, assetUrl, sha256File } from "../src/assets";

const RAW = JSON.stringify({
  release: "v0.1-assets",
  baseUrl: "https://github.com/ciciPL/L2NL/releases/download/v0.1-assets/",
  assets: [
    { name: "extractor/pytorch_model.bin", file: "pytorch_model.bin", sha256: "x", size: 1 },
  ],
});

describe("parseManifest", () => {
  it("parses a valid manifest", () => {
    const m = parseManifest(RAW);
    expect(m.assets[0].file).toBe("pytorch_model.bin");
  });
  it("throws on missing fields", () => {
    expect(() => parseManifest('{"assets":[]}')).toThrow();
  });
});

describe("assetUrl", () => {
  const m = parseManifest(RAW);
  it("joins baseUrl + file", () => {
    expect(assetUrl(m, m.assets[0])).toBe(
      "https://github.com/ciciPL/L2NL/releases/download/v0.1-assets/pytorch_model.bin");
  });
  it("honors a baseUrl override with a trailing slash added", () => {
    expect(assetUrl(m, m.assets[0], "/tmp/cs_assets")).toBe("/tmp/cs_assets/pytorch_model.bin");
  });
});

describe("sha256File", () => {
  it("hashes file bytes", async () => {
    const f = path.join(os.tmpdir(), `csa-${Date.now()}.txt`);
    fs.writeFileSync(f, "abc");
    // sha256("abc")
    expect(await sha256File(f)).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    fs.unlinkSync(f);
  });
});
