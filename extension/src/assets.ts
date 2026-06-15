import * as fs from "node:fs";
import * as crypto from "node:crypto";
import { Readable } from "node:stream";

export interface AssetEntry { name: string; file: string; sha256: string; size: number; }
export interface Manifest { release: string; baseUrl: string; assets: AssetEntry[]; }

export function parseManifest(json: string): Manifest {
  const m = JSON.parse(json) as Manifest;
  if (!m.release || !m.baseUrl || !Array.isArray(m.assets) || m.assets.length === 0) {
    throw new Error("invalid manifest: missing release/baseUrl/assets");
  }
  for (const a of m.assets) {
    if (!a.name || !a.file || !a.sha256 || typeof a.size !== "number") {
      throw new Error(`invalid manifest asset: ${JSON.stringify(a)}`);
    }
  }
  return m;
}

export function assetUrl(m: Manifest, a: AssetEntry, baseOverride?: string): string {
  const base = (baseOverride ?? m.baseUrl).replace(/\/+$/, "");
  return `${base}/${a.file}`;
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
  try {
    if (fs.statSync(path).size !== a.size) return false;
    return (await sha256File(path)) === a.sha256;
  } catch {
    return false;
  }
}

/** Stream a URL (http(s) or local file path) to dest, reporting bytes. */
export async function downloadAsset(
  url: string, dest: string, onBytes?: (done: number, total: number) => void,
): Promise<void> {
  fs.mkdirSync(require("node:path").dirname(dest), { recursive: true });
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
    nodeStream.pipe(out).on("finish", () => resolve()).on("error", reject);
    nodeStream.on("error", reject);
  });
}
