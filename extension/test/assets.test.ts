import { describe, it, expect } from "vitest";
import * as os from "node:os";
import * as fs from "node:fs";
import * as path from "node:path";
import {
  parseManifest, assetUrl, sha256File, assetParts, sourcePartUrls, installAsset,
} from "../src/assets";

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
  it("parses manifest v2 with sources and multipart assets", () => {
    const m = parseManifest(JSON.stringify({
      version: 2,
      release: "v0.2-assets-cn",
      sources: [
        { id: "gitee", baseUrl: "https://gitee.com/acme/l2nl/releases/download/v0.2-assets-cn/", enabledByDefault: true },
        { id: "github", baseUrl: "https://github.com/acme/l2nl/releases/download/v0.2-assets-cn/", global: true },
      ],
      assets: [
        {
          name: "extractor/pytorch_model.bin",
          target: "extractor/pytorch_model.bin",
          sha256: "whole",
          size: 6,
          parts: [
            { file: "pytorch_model.bin.part001", sha256: "p1", size: 3 },
            { file: "pytorch_model.bin.part002", sha256: "p2", size: 3 },
          ],
        },
      ],
    }));
    expect(m.version).toBe(2);
    expect(m.sources![0].id).toBe("gitee");
    expect(assetParts(m.assets[0])).toHaveLength(2);
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
  it("builds Gitee part URLs before optional global mirrors", () => {
    const m = parseManifest(JSON.stringify({
      version: 2,
      release: "v0.2-assets-cn",
      sources: [
        { id: "gitee", baseUrl: "https://gitee.com/acme/l2nl/releases/download/v0.2-assets-cn", enabledByDefault: true },
        { id: "github", baseUrl: "https://github.com/acme/l2nl/releases/download/v0.2-assets-cn", global: true },
      ],
      assets: [{
        name: "corpus/corpus_30k.jsonl",
        target: "corpus/corpus_30k.jsonl",
        sha256: "whole",
        size: 6,
        parts: [{ file: "corpus_30k.jsonl.part001", sha256: "p1", size: 6 }],
      }],
    }));
    const urls = sourcePartUrls(m, m.assets[0], { includeGlobalMirrors: true });
    expect(urls.map((u) => u.sourceId)).toEqual(["gitee", "github"]);
    expect(urls[0].parts[0]).toBe("https://gitee.com/acme/l2nl/releases/download/v0.2-assets-cn/corpus_30k.jsonl.part001");
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

describe("installAsset", () => {
  it("uses a local asset directory before downloading remote parts", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "csa-local-"));
    const local = path.join(root, "local");
    const dest = path.join(root, "out", "asset.bin");
    fs.mkdirSync(local, { recursive: true });
    fs.writeFileSync(path.join(local, "asset.bin.part001"), "abc");
    fs.writeFileSync(path.join(local, "asset.bin.part002"), "def");
    const whole = await sha256FileFromBytes("abcdef");
    const p1 = await sha256FileFromBytes("abc");
    const p2 = await sha256FileFromBytes("def");
    const m = parseManifest(JSON.stringify({
      version: 2,
      release: "v0.2-assets-cn",
      sources: [{ id: "gitee", baseUrl: "https://gitee.example/release", enabledByDefault: true }],
      assets: [{
        name: "asset.bin",
        target: "asset.bin",
        sha256: whole,
        size: 6,
        parts: [
          { file: "asset.bin.part001", sha256: p1, size: 3 },
          { file: "asset.bin.part002", sha256: p2, size: 3 },
        ],
      }],
    }));
    const downloads: string[] = [];
    await installAsset(m, m.assets[0], dest, {
      localAssetDir: local,
      download: async (url) => { downloads.push(url); },
    });
    expect(fs.readFileSync(dest, "utf8")).toBe("abcdef");
    expect(downloads).toEqual([]);
    expect(fs.existsSync(`${dest}.parts`)).toBe(false);
  });

  it("deletes temp parts when a local asset part checksum fails", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "csa-local-bad-"));
    const local = path.join(root, "local");
    const dest = path.join(root, "out", "asset.bin");
    fs.mkdirSync(local, { recursive: true });
    fs.writeFileSync(path.join(local, "asset.bin.part001"), "abc");
    fs.writeFileSync(path.join(local, "asset.bin.part002"), "WRONG");
    const whole = await sha256FileFromBytes("abcdef");
    const p1 = await sha256FileFromBytes("abc");
    const p2 = await sha256FileFromBytes("def");
    const m = parseManifest(JSON.stringify({
      version: 2,
      release: "v0.2-assets-cn",
      sources: [{ id: "gitee", baseUrl: "https://gitee.example/release", enabledByDefault: true }],
      assets: [{
        name: "asset.bin",
        target: "asset.bin",
        sha256: whole,
        size: 6,
        parts: [
          { file: "asset.bin.part001", sha256: p1, size: 3 },
          { file: "asset.bin.part002", sha256: p2, size: 3 },
        ],
      }],
    }));

    await expect(installAsset(m, m.assets[0], dest, {
      localAssetDir: local,
      download: async () => { throw new Error("remote should not run"); },
    })).rejects.toThrow(/checksum/i);

    expect(fs.existsSync(dest)).toBe(false);
    expect(fs.existsSync(`${dest}.tmp`)).toBe(false);
    expect(fs.existsSync(`${dest}.parts`)).toBe(false);
  });

  it("deletes partial output when a downloaded part checksum fails", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "csa-bad-"));
    const dest = path.join(root, "out", "asset.bin");
    const goodWhole = await sha256FileFromBytes("abcdef");
    const p1 = await sha256FileFromBytes("abc");
    const m = parseManifest(JSON.stringify({
      version: 2,
      release: "v0.2-assets-cn",
      sources: [{ id: "gitee", baseUrl: "https://gitee.example/release", enabledByDefault: true }],
      assets: [{
        name: "asset.bin",
        target: "asset.bin",
        sha256: goodWhole,
        size: 6,
        parts: [
          { file: "asset.bin.part001", sha256: p1, size: 3 },
          { file: "asset.bin.part002", sha256: "badsha", size: 3 },
        ],
      }],
    }));
    await expect(installAsset(m, m.assets[0], dest, {
      download: async (_url, out) => fs.writeFileSync(out, path.basename(out).includes("part001") ? "abc" : "def"),
    })).rejects.toThrow(/checksum/i);
    expect(fs.existsSync(dest)).toBe(false);
    expect(fs.existsSync(`${dest}.tmp`)).toBe(false);
  });
});

async function sha256FileFromBytes(text: string): Promise<string> {
  const f = path.join(os.tmpdir(), `csa-hash-${Date.now()}-${Math.random()}`);
  fs.writeFileSync(f, text);
  const out = await sha256File(f);
  fs.unlinkSync(f);
  return out;
}
