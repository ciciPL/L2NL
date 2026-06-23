import * as fs from "node:fs";
import * as path from "node:path";
import { AssetPart, AssetSource, SourceParts, downloadAsset, sha256File } from "./assets";

export type RuntimePlatform = "darwin-arm64" | "darwin-x64" | "win32-x64";

export interface RuntimeAsset {
  id: "llamacpp";
  platform: RuntimePlatform;
  target: string;
  sha256: string;
  size: number;
  executable: string;
  sourcePath?: string;
  parts?: AssetPart[];
}

export interface RuntimeManifest {
  version: number;
  release: string;
  sources: AssetSource[];
  runtimes: RuntimeAsset[];
}

export interface ModelScopeFile {
  path: string;
  size: number;
  sha256?: string;
}

export interface ModelScopePreset extends ModelScopeFile {
  id: string;
  label: string;
  modelId: string;
  filePath: string;
  quantization: string;
}

export interface LlamaStartOptions {
  modelPath: string;
  port: number;
  metal?: boolean;
}

export const MODEL_SCOPE_BASE = "https://modelscope.cn";

export const DEFAULT_RUNTIME_MANIFEST: RuntimeManifest = {
  version: 1,
  release: "v0.2-runtime-llamacpp",
  sources: [
    {
      id: "gitee",
      baseUrl: "https://gitee.com/ch2n2000/L2NL/releases/download",
      enabledByDefault: true,
    },
  ],
  // Filled by release engineering once llama.cpp runtime archives are published
  // to v0.2-runtime-llamacpp. The code also supports an explicit path or PATH.
  runtimes: [],
};

export const MODEL_SCOPE_PRESETS: ModelScopePreset[] = [
  {
    id: "qwen25-coder-1_5b-q4_k_m",
    label: "Fast: Qwen2.5-Coder 1.5B Q4_K_M",
    modelId: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    filePath: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    path: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    quantization: "q4_k_m",
    size: 1117320768,
    sha256: "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046",
  },
  {
    id: "qwen25-coder-7b-q4_k_m",
    label: "Quality: Qwen2.5-Coder 7B Q4_K_M",
    modelId: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    filePath: "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    path: "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    quantization: "q4_k_m",
    size: 4683073536,
    sha256: "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c",
  },
];

export function platformKey(
  platform: NodeJS.Platform = process.platform,
  arch: NodeJS.Architecture = process.arch,
): RuntimePlatform | undefined {
  if (platform === "darwin" && arch === "arm64") return "darwin-arm64";
  if (platform === "darwin" && arch === "x64") return "darwin-x64";
  if (platform === "win32" && arch === "x64") return "win32-x64";
  return undefined;
}

export function runtimeParts(a: RuntimeAsset): AssetPart[] {
  if (a.parts?.length) return a.parts;
  return [{ file: a.target, sha256: a.sha256, size: a.size }];
}

export function runtimePartUrls(
  manifest: RuntimeManifest,
  runtime: RuntimeAsset,
  opts: { includeGlobalMirrors?: boolean; baseUrlOverride?: string } = {},
): SourceParts[] {
  const parts = runtimeParts(runtime);
  if (opts.baseUrlOverride) {
    const base = opts.baseUrlOverride.replace(/\/+$/, "");
    return [{ sourceId: "override", parts: parts.map((p) => `${base}/${p.file}`) }];
  }
  return manifest.sources
    .filter((s) => s.enabledByDefault || (opts.includeGlobalMirrors && s.global))
    .map((s) => {
      const base = s.baseUrl.replace(/\/+$/, "");
      const prefix = runtime.sourcePath ? `${runtime.sourcePath.replace(/^\/+|\/+$/g, "")}/` : "";
      return { sourceId: s.id, parts: parts.map((p) => `${base}/${prefix}${p.file}`) };
    });
}

export function findRuntimeAsset(
  manifest: RuntimeManifest,
  platform: RuntimePlatform | undefined,
): RuntimeAsset | undefined {
  return platform ? manifest.runtimes.find((r) => r.id === "llamacpp" && r.platform === platform) : undefined;
}

export function runtimeExecutablePath(runtimeRoot: string, runtime: RuntimeAsset): string {
  return path.join(runtimeRoot, runtime.executable);
}

export function buildLlamaServerArgs(opts: LlamaStartOptions): string[] {
  const args = [
    "-m", opts.modelPath,
    "--host", "127.0.0.1",
    "--port", String(opts.port),
    "-c", "8192",
    "--parallel", "1",
  ];
  if (opts.metal) args.push("-ngl", "99");
  return args;
}

export function encodeModelId(modelId: string): string {
  return modelId.split("/").map(encodeURIComponent).join("/");
}

export function modelScopeRepoFilesUrl(modelId: string, revision = "master"): string {
  return `${MODEL_SCOPE_BASE}/api/v1/models/${encodeModelId(modelId)}/repo/files?Revision=${encodeURIComponent(revision)}`;
}

export function modelScopeDownloadUrl(modelId: string, filePath: string, revision = "master"): string {
  return `${MODEL_SCOPE_BASE}/api/v1/models/${encodeModelId(modelId)}/repo?Revision=${encodeURIComponent(revision)}&FilePath=${encodeURIComponent(filePath)}`;
}

export function normalizeModelScopeFiles(payload: unknown): ModelScopeFile[] {
  const data = (payload as any)?.Data ?? payload;
  const files = Array.isArray(data?.Files) ? data.Files : Array.isArray(data) ? data : [];
  const out: ModelScopeFile[] = [];
  for (const f of files) {
    const filePath = f?.Path ?? f?.FilePath ?? f?.Name ?? f?.path ?? f?.name;
    const size = Number(f?.Size ?? f?.size ?? 0);
    if (!filePath || !Number.isFinite(size) || size <= 0) continue;
    if (f?.Type && String(f.Type).toLowerCase() !== "file") continue;
    out.push({
      path: String(filePath),
      size,
      sha256: f?.Sha256 ?? f?.SHA256 ?? f?.sha256,
    });
  }
  return out;
}

export function filterGgufFiles(files: ModelScopeFile[]): ModelScopeFile[] {
  return files.filter((f) => f.path.toLowerCase().endsWith(".gguf"));
}

export function safeModelDir(modelId: string): string {
  return modelId.replace(/[^a-zA-Z0-9._-]+/g, "__");
}

export function modelLocalPath(modelsRoot: string, modelId: string, filePath: string): string {
  return path.join(modelsRoot, safeModelDir(modelId), path.basename(filePath));
}

export async function fetchModelScopeGgufFiles(
  modelId: string,
  fetcher: typeof fetch = fetch,
): Promise<ModelScopeFile[]> {
  const res = await fetcher(modelScopeRepoFilesUrl(modelId));
  if (!res.ok) throw new Error(`ModelScope ${modelId} -> HTTP ${res.status}`);
  return filterGgufFiles(normalizeModelScopeFiles(await res.json()));
}

export async function downloadModelScopeFile(
  modelId: string,
  filePath: string,
  dest: string,
  expected?: { sha256?: string; size?: number },
  onBytes?: (done: number, total: number) => void,
): Promise<void> {
  if (expected?.size && fs.existsSync(dest) && fs.statSync(dest).size === expected.size) {
    if (!expected.sha256 || (await sha256File(dest)) === expected.sha256) return;
  }
  const tmp = `${dest}.tmp`;
  fs.rmSync(tmp, { force: true });
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  try {
    await downloadAsset(modelScopeDownloadUrl(modelId, filePath), tmp, onBytes);
    if (expected?.size && fs.statSync(tmp).size !== expected.size) {
      throw new Error(`size mismatch: ${path.basename(filePath)}`);
    }
    if (expected?.sha256 && (await sha256File(tmp)) !== expected.sha256) {
      throw new Error(`checksum mismatch: ${path.basename(filePath)}`);
    }
    fs.renameSync(tmp, dest);
  } catch (e) {
    fs.rmSync(tmp, { force: true });
    throw e;
  }
}
