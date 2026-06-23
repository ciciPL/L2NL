import * as fs from "node:fs";
import { describe, expect, it } from "vitest";
import {
  MODEL_SCOPE_PRESETS,
  buildLlamaServerArgs,
  filterGgufFiles,
  modelScopeDownloadUrl,
  modelScopeRepoFilesUrl,
  normalizeModelScopeFiles,
  platformKey,
  runtimePartUrls,
  RuntimeManifest,
} from "../src/localRuntime";

describe("local llama.cpp runtime helpers", () => {
  it("ships runtime manifest entries for Windows and macOS production setup", () => {
    const manifest = JSON.parse(fs.readFileSync("runtime-manifest.json", "utf8")) as RuntimeManifest;
    expect(manifest.release).toBe("v0.2-runtime-llamacpp");
    expect(manifest.runtimes.map((r) => r.platform).sort()).toEqual([
      "darwin-arm64",
      "darwin-x64",
      "win32-x64",
    ]);
    for (const runtime of manifest.runtimes) {
      expect(runtime.sourcePath).toBe("v0.2-runtime-llamacpp");
      expect(runtime.sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(runtime.size).toBeGreaterThan(1_000_000);
      expect(runtime.executable).toMatch(/llama-server(\.exe)?$/);
    }
  });

  it("maps supported desktop platforms to runtime package ids", () => {
    expect(platformKey("darwin", "arm64")).toBe("darwin-arm64");
    expect(platformKey("darwin", "x64")).toBe("darwin-x64");
    expect(platformKey("win32", "x64")).toBe("win32-x64");
    expect(platformKey("linux", "x64")).toBeUndefined();
  });

  it("builds Gitee per-runtime multipart URLs with sourcePath", () => {
    const manifest: RuntimeManifest = {
      version: 1,
      release: "v0.2-runtime-llamacpp",
      sources: [{
        id: "gitee",
        baseUrl: "https://gitee.com/ch2n2000/L2NL/releases/download",
        enabledByDefault: true,
      }],
      runtimes: [{
        id: "llamacpp",
        platform: "darwin-arm64",
        target: "llama.cpp-darwin-arm64.tar.gz",
        sha256: "archive",
        size: 2,
        sourcePath: "v0.2-runtime-llamacpp",
        executable: "bin/llama-server",
        parts: [
          { file: "llama.cpp-darwin-arm64.tar.gz.part001", sha256: "p1", size: 1 },
          { file: "llama.cpp-darwin-arm64.tar.gz.part002", sha256: "p2", size: 1 },
        ],
      }],
    };
    expect(runtimePartUrls(manifest, manifest.runtimes[0])[0]).toEqual({
      sourceId: "gitee",
      parts: [
        "https://gitee.com/ch2n2000/L2NL/releases/download/v0.2-runtime-llamacpp/llama.cpp-darwin-arm64.tar.gz.part001",
        "https://gitee.com/ch2n2000/L2NL/releases/download/v0.2-runtime-llamacpp/llama.cpp-darwin-arm64.tar.gz.part002",
      ],
    });
  });

  it("uses ModelScope repo APIs and keeps only GGUF files", () => {
    expect(modelScopeRepoFilesUrl("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"))
      .toBe("https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/repo/files?Revision=master");
    expect(modelScopeDownloadUrl("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "dir/model.gguf"))
      .toBe("https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/repo?Revision=master&FilePath=dir%2Fmodel.gguf");

    const files = normalizeModelScopeFiles({
      Data: {
        Files: [
          { Path: "README.md", Size: 12, Type: "file" },
          { Path: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", Size: 1117320768, Type: "file", Sha256: "abc" },
          { Path: "nested/qwen-q8_0.GGUF", Size: "22", Type: "file" },
        ],
      },
    });
    expect(filterGgufFiles(files)).toEqual([
      { path: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", size: 1117320768, sha256: "abc" },
      { path: "nested/qwen-q8_0.GGUF", size: 22, sha256: undefined },
    ]);
  });

  it("ships ModelScope Qwen2.5-Coder presets with known q4_k_m files", () => {
    const fast = MODEL_SCOPE_PRESETS.find((p) => p.id === "qwen25-coder-1_5b-q4_k_m");
    const quality = MODEL_SCOPE_PRESETS.find((p) => p.id === "qwen25-coder-7b-q4_k_m");
    expect(fast?.modelId).toBe("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF");
    expect(fast?.filePath).toBe("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
    expect(fast?.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(quality?.modelId).toBe("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
    expect(quality?.filePath).toBe("qwen2.5-coder-7b-instruct-q4_k_m.gguf");
    expect(quality?.sha256).toMatch(/^[a-f0-9]{64}$/);
  });

  it("generates the fixed llama-server command line", () => {
    expect(buildLlamaServerArgs({ modelPath: "/models/qwen.gguf", port: 8123 })).toEqual([
      "-m", "/models/qwen.gguf",
      "--host", "127.0.0.1",
      "--port", "8123",
      "-c", "8192",
      "--parallel", "1",
    ]);
  });
});
