import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { Readable } from "node:stream";

export interface AssetPart { file: string; sha256: string; size: number; }
export interface AssetSource {
  id: string;
  baseUrl: string;
  enabledByDefault?: boolean;
  global?: boolean;
}
export interface AssetEntry {
  name: string;
  file?: string;
  target?: string;
  sha256: string;
  size: number;
  parts?: AssetPart[];
  extractTo?: string;
}
export interface Manifest {
  version?: number;
  release: string;
  baseUrl?: string;
  sources?: AssetSource[];
  assets: AssetEntry[];
}
export interface SourceParts { sourceId: string; parts: string[]; }

export function parseManifest(json: string): Manifest {
  const m = JSON.parse(json) as Manifest;
  if (!m.release || !Array.isArray(m.assets) || m.assets.length === 0) {
    throw new Error("invalid manifest: missing release/baseUrl/assets");
  }
  if (!m.baseUrl && (!Array.isArray(m.sources) || m.sources.length === 0)) {
    throw new Error("invalid manifest: missing baseUrl/sources");
  }
  if (m.sources) {
    for (const s of m.sources) {
      if (!s.id || !s.baseUrl) throw new Error(`invalid manifest source: ${JSON.stringify(s)}`);
    }
  }
  for (const a of m.assets) {
    if (!a.name || !a.sha256 || typeof a.size !== "number") {
      throw new Error(`invalid manifest asset: ${JSON.stringify(a)}`);
    }
    if (!a.file && !a.target && !a.parts?.length) {
      throw new Error(`invalid manifest asset: ${JSON.stringify(a)}`);
    }
    for (const p of assetParts(a)) {
      if (!p.file || !p.sha256 || typeof p.size !== "number") {
        throw new Error(`invalid manifest asset part: ${JSON.stringify(p)}`);
      }
    }
  }
  return m;
}

export function assetUrl(m: Manifest, a: AssetEntry, baseOverride?: string): string {
  const base = (baseOverride ?? m.baseUrl ?? m.sources?.[0]?.baseUrl ?? "").replace(/\/+$/, "");
  return `${base}/${assetParts(a)[0].file}`;
}

export function assetParts(a: AssetEntry): AssetPart[] {
  if (a.parts?.length) return a.parts;
  return [{ file: a.file ?? a.target ?? a.name, sha256: a.sha256, size: a.size }];
}

export function assetTarget(a: AssetEntry): string {
  return a.target ?? a.file ?? a.name;
}

export function sourcePartUrls(
  m: Manifest,
  a: AssetEntry,
  opts: { baseUrlOverride?: string; includeGlobalMirrors?: boolean } = {},
): SourceParts[] {
  const parts = assetParts(a);
  if (opts.baseUrlOverride) {
    const base = opts.baseUrlOverride.replace(/\/+$/, "");
    return [{ sourceId: "override", parts: parts.map((p) => `${base}/${p.file}`) }];
  }
  const sources = m.sources?.length
    ? m.sources
    : [{ id: "default", baseUrl: m.baseUrl ?? "", enabledByDefault: true }];
  return sources
    .filter((s) => s.enabledByDefault || (opts.includeGlobalMirrors && s.global))
    .map((s) => {
      const base = s.baseUrl.replace(/\/+$/, "");
      return { sourceId: s.id, parts: parts.map((p) => `${base}/${p.file}`) };
    });
}

export function sha256File(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const h = crypto.createHash("sha256");
    fs.createReadStream(path)
      .on("data", (d) => h.update(d))
      .on("error", reject)
      .on("end", () => resolve(h.digest("hex")));
  });
}

export async function verifyAsset(path: string, a: AssetEntry): Promise<boolean> {
  return verifyFileAgainst(path, { sha256: a.sha256, size: a.size });
}

async function verifyFileAgainst(path: string, a: { sha256: string; size: number }): Promise<boolean> {
  try {
    if (fs.statSync(path).size !== a.size) return false;
    return (await sha256File(path)) === a.sha256;
  } catch {
    return false;
  }
}

export async function installAsset(
  m: Manifest,
  a: AssetEntry,
  dest: string,
  opts: {
    localAssetDir?: string;
    baseUrlOverride?: string;
    includeGlobalMirrors?: boolean;
    download?: typeof downloadAsset;
    verify?: (path: string, a: AssetEntry) => Promise<boolean>;
    onBytes?: (done: number, total: number) => void;
  } = {},
): Promise<void> {
  const verify = opts.verify ?? verifyAsset;
  if (await verify(dest, a)) return;

  const tmp = `${dest}.tmp`;
  const partDir = `${dest}.parts`;
  cleanup(dest, tmp, partDir);
  fs.mkdirSync(path.dirname(dest), { recursive: true });

  if (opts.localAssetDir) {
    try {
      const local = await tryInstallFromLocal(a, opts.localAssetDir, tmp, partDir);
      if (local) {
        await finishInstall(tmp, dest, a);
        cleanupTemp(tmp, partDir);
        return;
      }
    } catch (e) {
      cleanup(dest, tmp, partDir);
      throw e;
    }
  }

  const download = opts.download ?? downloadAsset;
  let lastError: unknown;
  for (const source of sourcePartUrls(m, a, opts)) {
    cleanup(dest, tmp, partDir);
    try {
      await downloadParts(a, source.parts, partDir, download, opts.onBytes);
      await concatParts(a, partDir, tmp);
      await finishInstall(tmp, dest, a);
      cleanupTemp(tmp, partDir);
      return;
    } catch (e) {
      lastError = e;
      cleanup(dest, tmp, partDir);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(`failed to install ${a.name}`);
}

function cleanup(dest: string, tmp: string, partDir: string) {
  fs.rmSync(tmp, { force: true });
  fs.rmSync(partDir, { recursive: true, force: true });
  fs.rmSync(dest, { force: true });
}

function cleanupTemp(tmp: string, partDir: string) {
  fs.rmSync(tmp, { force: true });
  fs.rmSync(partDir, { recursive: true, force: true });
}

async function tryInstallFromLocal(a: AssetEntry, localDir: string, tmp: string, partDir: string): Promise<boolean> {
  const parts = assetParts(a);
  const partPaths = parts.map((p) => path.join(localDir, p.file));
  if (!partPaths.every((p) => fs.existsSync(p))) return false;
  fs.mkdirSync(partDir, { recursive: true });
  for (let i = 0; i < parts.length; i++) {
    const out = path.join(partDir, parts[i].file);
    fs.copyFileSync(partPaths[i], out);
    if (!(await verifyFileAgainst(out, parts[i]))) {
      throw new Error(`checksum mismatch: ${parts[i].file}`);
    }
  }
  await concatParts(a, partDir, tmp);
  return true;
}

async function downloadParts(
  a: AssetEntry,
  urls: string[],
  partDir: string,
  download: typeof downloadAsset,
  onBytes?: (done: number, total: number) => void,
) {
  fs.mkdirSync(partDir, { recursive: true });
  const parts = assetParts(a);
  for (let i = 0; i < parts.length; i++) {
    const out = path.join(partDir, parts[i].file);
    await download(urls[i], out, onBytes);
    if (!(await verifyFileAgainst(out, parts[i]))) {
      throw new Error(`checksum mismatch: ${parts[i].file}`);
    }
  }
}

async function concatParts(a: AssetEntry, partDir: string, tmp: string) {
  fs.mkdirSync(path.dirname(tmp), { recursive: true });
  const out = fs.createWriteStream(tmp);
  try {
    for (const p of assetParts(a)) {
      const input = fs.createReadStream(path.join(partDir, p.file));
      await new Promise<void>((resolve, reject) => {
        input.on("error", reject);
        input.on("end", resolve);
        input.pipe(out, { end: false });
      });
    }
  } finally {
    await new Promise<void>((resolve) => out.end(resolve));
  }
}

async function finishInstall(tmp: string, dest: string, a: AssetEntry) {
  if (!(await verifyFileAgainst(tmp, a))) {
    fs.rmSync(tmp, { force: true });
    throw new Error(`checksum mismatch: ${assetTarget(a)}`);
  }
  fs.renameSync(tmp, dest);
}

/** Stream a URL (http(s) or local file path) to dest, reporting bytes. */
export async function downloadAsset(
  url: string, dest: string, onBytes?: (done: number, total: number) => void,
): Promise<void> {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (!/^https?:\/\//.test(url)) {
    // Local-path / file source (used by codeSummary.assets.baseUrl override).
    const src = url.replace(/^file:\/\//, "");
    await fs.promises.copyFile(src, dest);
    const sz = fs.statSync(dest).size;
    onBytes?.(sz, sz);
    return;
  }
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`download ${url} -> HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  let done = 0;
  const out = fs.createWriteStream(dest);
  const nodeStream = Readable.fromWeb(res.body as any);
  nodeStream.on("data", (c: Buffer) => { done += c.length; onBytes?.(done, total); });
  await new Promise<void>((resolve, reject) => {
    out.on("error", reject);
    nodeStream.on("error", reject);
    nodeStream.pipe(out).on("finish", () => resolve());
  });
}
