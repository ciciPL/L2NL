# Phase 2A — Provisioner Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless TypeScript provisioner that detects Python/GPU, creates a private venv, installs torch+deps, downloads + verifies model/corpus assets, pre-warms CodeBERT, and launches the bundled FastAPI backend with the correct `CS_*` env — all inside the extension's `globalStorage`.

**Architecture:** Pure, vscode-free modules (`paths`, `env`, `assets`, `state`, and the env-builder/port helpers in `provision`) hold all testable logic; side effects (exec, fs, fetch, spawn) are injected through a `Deps` object so the orchestrator can be driven by fakes in tests and by real Node APIs at runtime. `extension.ts` calls the orchestrator from `ensureBackend()` when the backend isn't yet provisioned. The webview wizard (Phase 2B) is a thin layer over this engine and is **out of scope** for this plan.

**Tech Stack:** TypeScript, VS Code extension host (Node 18+ — global `fetch`, `node:crypto`, `node:child_process`, `node:fs`, `node:net`), vitest, esbuild, `@vscode/vsce`.

**Repo convention:** This artifact repo works on `master`/`main` directly (all prior commits do). Work on the current branch; commit per task.

**Spec:** `docs/superpowers/specs/2026-06-15-phase2-provisioner-wizard-design.md`

---

## File Structure

All new files live in `extension/src/` unless noted. Each is single-responsibility and (except the orchestrator's side-effecting parts) unit-tested.

| File | Responsibility | vscode? |
|---|---|---|
| `extension/src/paths.ts` | Compute the globalStorage layout (venv python, assets, hf_cache, state path) — platform-aware, **pure** | no |
| `extension/src/env.ts` | Detect Python ≥3.10 and GPU; pure parsers + injected `Runner` | no |
| `extension/src/state.ts` | `state.json` schema + `stepsNeeded()` diff logic + read/write | no |
| `extension/src/assets.ts` | Manifest parse, asset URL, sha256 verify, streaming download | no |
| `extension/src/provision.ts` | Orchestrator + pure `buildBackendEnv()` + `findFreePort()` | no |
| `extension/src/extension.ts` | Wire provisioner into `ensureBackend()` (**modify**) | yes |
| `backend/assets/manifest.json` | Asset manifest (data) | — |
| `extension/scripts/copy-backend.js` | Copy `../backend` source into `extension/backend/` for packaging | no |
| `extension/package.json` | scripts + settings + activation (**modify**) | — |
| `extension/.vscodeignore` | Include bundled `backend/`, exclude its junk (**modify**) | — |

Test files mirror under `extension/test/`.

---

## Task 1: Paths module

**Files:**
- Create: `extension/src/paths.ts`
- Test: `extension/test/paths.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// extension/test/paths.test.ts
import { describe, it, expect } from "vitest";
import { provPaths } from "../src/paths";

describe("provPaths", () => {
  it("lays out posix paths under the storage root", () => {
    const p = provPaths("/gs", "darwin");
    expect(p.venvPython).toBe("/gs/venv/bin/python");
    expect(p.extractorWeights).toBe("/gs/assets/extractor/pytorch_model.bin");
    expect(p.corpus).toBe("/gs/assets/corpus/corpus_30k.jsonl");
    expect(p.hfCache).toBe("/gs/hf_cache");
    expect(p.state).toBe("/gs/state.json");
  });

  it("uses Scripts/python.exe on win32", () => {
    const p = provPaths("C:\\gs", "win32");
    expect(p.venvPython).toBe("C:\\gs\\venv\\Scripts\\python.exe");
    expect(p.extractorWeights).toBe("C:\\gs\\assets\\extractor\\pytorch_model.bin");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension && npx vitest run test/paths.test.ts`
Expected: FAIL — "Cannot find module '../src/paths'".

- [ ] **Step 3: Write minimal implementation**

```ts
// extension/src/paths.ts
export interface ProvPaths {
  root: string;
  venv: string;
  venvPython: string;
  assets: string;
  extractorWeights: string;
  corpus: string;
  hfCache: string;
  backendLog: string;
  state: string;
}

// Join with the separator implied by `platform` (we cannot use node:path here
// because tests cross-compile both layouts on one host).
function join(platform: NodeJS.Platform, ...parts: string[]): string {
  const sep = platform === "win32" ? "\\" : "/";
  return parts.join(sep);
}

export function provPaths(root: string, platform: NodeJS.Platform): ProvPaths {
  const j = (...p: string[]) => join(platform, root, ...p);
  const venv = j("venv");
  const venvPython = platform === "win32"
    ? join(platform, venv, "Scripts", "python.exe")
    : join(platform, venv, "bin", "python");
  return {
    root,
    venv,
    venvPython,
    assets: j("assets"),
    extractorWeights: j("assets", "extractor", "pytorch_model.bin"),
    corpus: j("assets", "corpus", "corpus_30k.jsonl"),
    hfCache: j("hf_cache"),
    backendLog: j("backend.log"),
    state: j("state.json"),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension && npx vitest run test/paths.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add extension/src/paths.ts extension/test/paths.test.ts
git commit -m "feat(ext): provisioner globalStorage path layout (platform-aware)"
```

---

## Task 2: Python + GPU detection

**Files:**
- Create: `extension/src/env.ts`
- Test: `extension/test/env.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// extension/test/env.test.ts
import { describe, it, expect } from "vitest";
import { parsePyVersion, pythonCandidates, detectPython, detectGpu, Runner } from "../src/env";

describe("parsePyVersion", () => {
  it("reads a 'major minor' line", () => {
    expect(parsePyVersion("3 11\n")).toEqual([3, 11]);
  });
  it("returns null on garbage", () => {
    expect(parsePyVersion("not a version")).toBeNull();
  });
});

describe("pythonCandidates", () => {
  it("puts the configured interpreter first on posix", () => {
    const c = pythonCandidates("darwin", "/usr/bin/python3.12");
    expect(c[0]).toEqual(["/usr/bin/python3.12"]);
    expect(c).toContainEqual(["python3.11"]);
  });
  it("uses the py launcher on win32", () => {
    const c = pythonCandidates("win32");
    expect(c).toContainEqual(["py", "-3.11"]);
  });
});

describe("detectPython", () => {
  const ok: Runner = async (cmd) =>
    cmd.includes("3.11") || cmd === "python3.11"
      ? { code: 0, stdout: "3 11", stderr: "" }
      : { code: 1, stdout: "", stderr: "no" };

  it("returns the first interpreter that is >=3.10", async () => {
    const info = await detectPython("darwin", undefined, ok);
    expect(info?.ok).toBe(true);
    expect(info?.version).toEqual([3, 11]);
  });

  it("rejects when everything is too old", async () => {
    const old: Runner = async () => ({ code: 0, stdout: "3 9", stderr: "" });
    const info = await detectPython("darwin", undefined, old);
    expect(info).toBeNull();
  });
});

describe("detectGpu", () => {
  it("is true when nvidia-smi exits 0", async () => {
    const run: Runner = async (c) => c === "nvidia-smi"
      ? { code: 0, stdout: "GPU 0", stderr: "" } : { code: 127, stdout: "", stderr: "" };
    expect(await detectGpu(run)).toBe(true);
  });
  it("is false when nvidia-smi is missing", async () => {
    const run: Runner = async () => ({ code: 127, stdout: "", stderr: "not found" });
    expect(await detectGpu(run)).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension && npx vitest run test/env.test.ts`
Expected: FAIL — "Cannot find module '../src/env'".

- [ ] **Step 3: Write minimal implementation**

```ts
// extension/src/env.ts
export type Runner = (
  cmd: string, args?: string[], env?: Record<string, string>,
) => Promise<{ code: number; stdout: string; stderr: string }>;

export interface PyInfo { cmd: string[]; version: [number, number]; ok: boolean; }

const VERSION_PROBE = "import sys;print(sys.version_info[0], sys.version_info[1])";

export function parsePyVersion(stdout: string): [number, number] | null {
  const m = stdout.trim().match(/^(\d+)\s+(\d+)/);
  if (!m) return null;
  return [parseInt(m[1], 10), parseInt(m[2], 10)];
}

export function pythonCandidates(
  platform: NodeJS.Platform, configured?: string,
): string[][] {
  const list: string[][] = [];
  if (configured && configured.trim()) list.push([configured.trim()]);
  if (platform === "win32") {
    list.push(["py", "-3.11"], ["py", "-3.10"], ["py", "-3"], ["python"]);
  } else {
    list.push(["python3.11"], ["python3.10"], ["python3"], ["python"]);
  }
  return list;
}

export async function detectPython(
  platform: NodeJS.Platform, configured: string | undefined, run: Runner,
): Promise<PyInfo | null> {
  for (const cand of pythonCandidates(platform, configured)) {
    const [cmd, ...args] = cand;
    let res;
    try {
      res = await run(cmd, [...args, "-c", VERSION_PROBE]);
    } catch {
      continue;
    }
    if (res.code !== 0) continue;
    const v = parsePyVersion(res.stdout);
    if (!v) continue;
    const okVer = v[0] > 3 || (v[0] === 3 && v[1] >= 10);
    if (okVer) return { cmd: cand, version: v, ok: true };
  }
  return null;
}

export async function detectGpu(run: Runner): Promise<boolean> {
  try {
    const res = await run("nvidia-smi", ["-L"]);
    return res.code === 0;
  } catch {
    return false;
  }
}
```

> Note: the test's `ok` runner ignores args, so `detectPython` matching on the
> `cmd` value (`python3.11`) is what the first-candidate assertion checks.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension && npx vitest run test/env.test.ts`
Expected: PASS (8 assertions across 4 describes).

- [ ] **Step 5: Commit**

```bash
git add extension/src/env.ts extension/test/env.test.ts
git commit -m "feat(ext): Python>=3.10 + GPU detection with injectable runner"
```

---

## Task 3: Provisioning state + step diff

**Files:**
- Create: `extension/src/state.ts`
- Test: `extension/test/state.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// extension/test/state.test.ts
import { describe, it, expect } from "vitest";
import { stepsNeeded, ProvState, ProvInputs } from "../src/state";

const inputs: ProvInputs = {
  extVersion: "0.2.0", device: "cpu", reqHash: "abc",
  assets: { "extractor/pytorch_model.bin": "h1", "corpus/corpus_30k.jsonl": "h2" },
};

describe("stepsNeeded", () => {
  it("needs every step when there is no prior state", () => {
    expect([...stepsNeeded(null, inputs)].sort())
      .toEqual(["assets", "codebert", "deps", "launch", "torch", "venv"]);
  });

  it("needs nothing but launch when state matches", () => {
    const prev: ProvState = { schemaVersion: 1, ...inputs, codebertReady: true };
    expect([...stepsNeeded(prev, inputs)]).toEqual(["launch"]);
  });

  it("re-runs torch (and deps) when device changes", () => {
    const prev: ProvState = { schemaVersion: 1, ...inputs, device: "cuda", codebertReady: true };
    const s = stepsNeeded(prev, inputs);
    expect(s.has("torch")).toBe(true);
    expect(s.has("deps")).toBe(true);
    expect(s.has("assets")).toBe(false);
  });

  it("re-downloads only the asset whose hash changed", () => {
    const prev: ProvState = {
      schemaVersion: 1, ...inputs, codebertReady: true,
      assets: { "extractor/pytorch_model.bin": "h1", "corpus/corpus_30k.jsonl": "OLD" },
    };
    const s = stepsNeeded(prev, inputs);
    expect(s.has("assets")).toBe(true);
    expect(s.has("torch")).toBe(false);
  });

  it("re-runs codebert when prior run did not finish it", () => {
    const prev: ProvState = { schemaVersion: 1, ...inputs, codebertReady: false };
    expect(stepsNeeded(prev, inputs).has("codebert")).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension && npx vitest run test/state.test.ts`
Expected: FAIL — "Cannot find module '../src/state'".

- [ ] **Step 3: Write minimal implementation**

```ts
// extension/src/state.ts
import * as fs from "node:fs";

export type Step = "venv" | "torch" | "deps" | "assets" | "codebert" | "launch";

export interface ProvInputs {
  extVersion: string;
  device: string;
  reqHash: string;
  assets: Record<string, string>; // name -> sha256
}

export interface ProvState extends ProvInputs {
  schemaVersion: number;
  codebertReady: boolean;
}

const SCHEMA = 1;

export function stepsNeeded(old: ProvState | null, next: ProvInputs): Set<Step> {
  const steps = new Set<Step>();
  // launch always runs (process is not persisted across sessions).
  steps.add("launch");
  if (!old || old.schemaVersion !== SCHEMA) {
    (["venv", "torch", "deps", "assets", "codebert"] as Step[]).forEach((s) => steps.add(s));
    return steps;
  }
  const envChanged = old.reqHash !== next.reqHash;
  const deviceChanged = old.device !== next.device;
  if (deviceChanged || envChanged) { steps.add("torch"); steps.add("deps"); }
  const assetsChanged = Object.keys(next.assets).some(
    (k) => old.assets[k] !== next.assets[k],
  );
  if (assetsChanged) steps.add("assets");
  if (!old.codebertReady) steps.add("codebert");
  // venv only needs (re)building if the schema/ext changed — covered above; an
  // existing matching state implies the venv exists.
  return steps;
}

export function readState(path: string): ProvState | null {
  try {
    return JSON.parse(fs.readFileSync(path, "utf8")) as ProvState;
  } catch {
    return null;
  }
}

export function writeState(path: string, inputs: ProvInputs, codebertReady: boolean): void {
  const s: ProvState = { schemaVersion: SCHEMA, ...inputs, codebertReady };
  fs.writeFileSync(path, JSON.stringify(s, null, 2));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension && npx vitest run test/state.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add extension/src/state.ts extension/test/state.test.ts
git commit -m "feat(ext): provisioning state.json + idempotent step diff"
```

---

## Task 4: Asset manifest + sha256 verification

**Files:**
- Create: `extension/src/assets.ts`
- Test: `extension/test/assets.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// extension/test/assets.test.ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension && npx vitest run test/assets.test.ts`
Expected: FAIL — "Cannot find module '../src/assets'".

- [ ] **Step 3: Write minimal implementation**

```ts
// extension/src/assets.ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension && npx vitest run test/assets.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add extension/src/assets.ts extension/test/assets.test.ts
git commit -m "feat(ext): asset manifest parse + sha256 verify + streaming download"
```

---

## Task 5: Backend asset manifest file

**Files:**
- Create: `backend/assets/manifest.json`
- Test: `extension/test/manifest.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// extension/test/manifest.test.ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension && npx vitest run test/manifest.test.ts`
Expected: FAIL — ENOENT reading `backend/assets/manifest.json`.

- [ ] **Step 3: Write the manifest**

```json
{
  "release": "v0.1-assets",
  "baseUrl": "https://github.com/ciciPL/L2NL/releases/download/v0.1-assets/",
  "assets": [
    {
      "name": "extractor/pytorch_model.bin",
      "file": "pytorch_model.bin",
      "sha256": "638efb2ab2862843d987e98117ad679065c4037f7246793f1e9ec4c3f8477de8",
      "size": 316071890
    },
    {
      "name": "corpus/corpus_30k.jsonl",
      "file": "corpus_30k.jsonl",
      "sha256": "08b64305f2a846cad3283ab3340a95ae2c403120be5da42fe942055467860258",
      "size": 71406879
    }
  ]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension && npx vitest run test/manifest.test.ts`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add backend/assets/manifest.json extension/test/manifest.test.ts
git commit -m "feat: ship asset manifest (L2NL v0.1-assets release URLs + sha256)"
```

---

## Task 6: Backend env builder + free-port helper

**Files:**
- Create: `extension/src/provision.ts`
- Test: `extension/test/provision-env.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// extension/test/provision-env.test.ts
import { describe, it, expect } from "vitest";
import { buildBackendEnv } from "../src/provision";
import { provPaths } from "../src/paths";

describe("buildBackendEnv", () => {
  it("mirrors star_start.sh CS_* contract, localized to globalStorage", () => {
    const p = provPaths("/gs", "darwin");
    const env = buildBackendEnv(p, "cpu");
    expect(env.CS_PYTHON).toBe("/gs/venv/bin/python");
    expect(env.CS_EXTRACTOR_WEIGHTS).toBe("/gs/assets/extractor/pytorch_model.bin");
    expect(env.CS_CORPUS_PATH).toBe("/gs/assets/corpus/corpus_30k.jsonl");
    expect(env.CS_CORPUS_LIMIT).toBe("30000");
    expect(env.CS_DEVICE).toBe("cpu");
    expect(env.HF_HOME).toBe("/gs/hf_cache");
    // CodeBERT path is left unset so transformers resolves the HF id via HF_HOME.
    expect(env.CS_CODEBERT_PATH).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension && npx vitest run test/provision-env.test.ts`
Expected: FAIL — "Cannot find module '../src/provision'" (or `buildBackendEnv` undefined).

- [ ] **Step 3: Write minimal implementation (start `provision.ts`)**

```ts
// extension/src/provision.ts
import * as net from "node:net";
import { ProvPaths } from "./paths";

export function buildBackendEnv(p: ProvPaths, device: string): Record<string, string> {
  return {
    CS_PYTHON: p.venvPython,
    CS_EXTRACTOR_WEIGHTS: p.extractorWeights,
    CS_CORPUS_PATH: p.corpus,
    CS_CORPUS_LIMIT: "30000",
    CS_DEVICE: device,
    HF_HOME: p.hfCache,
  };
}

export function findFreePort(start = 8000, tries = 20): Promise<number> {
  function check(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const srv = net.createServer();
      srv.once("error", () => resolve(false));
      srv.once("listening", () => srv.close(() => resolve(true)));
      srv.listen(port, "127.0.0.1");
    });
  }
  return (async () => {
    for (let i = 0; i < tries; i++) {
      if (await check(start + i)) return start + i;
    }
    throw new Error(`no free port in ${start}..${start + tries}`);
  })();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension && npx vitest run test/provision-env.test.ts`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add extension/src/provision.ts extension/test/provision-env.test.ts
git commit -m "feat(ext): backend CS_* env builder + free-port helper"
```

---

## Task 7: Provision orchestrator

**Files:**
- Modify: `extension/src/provision.ts`
- Test: `extension/test/provision-orchestrate.test.ts`

This wires the steps together with injected side-effect dependencies so the
ordering/skip logic is testable without touching the real OS.

- [ ] **Step 1: Write the failing test**

```ts
// extension/test/provision-orchestrate.test.ts
import { describe, it, expect } from "vitest";
import { provision, ProvDeps, ProvOptions } from "../src/provision";
import { provPaths } from "../src/paths";

function fakeDeps(record: string[]): ProvDeps {
  return {
    run: async (cmd, args = []) => { record.push(`run ${cmd} ${args.join(" ")}`); return { code: 0, stdout: "ok", stderr: "" }; },
    spawnBackend: (_python, _args, _opts) => { record.push("spawn"); return { pid: 123 } as any; },
    download: async (url, dest) => { record.push(`download ${dest}`); },
    verify: async () => true,            // pretend already-correct assets
    health: async () => true,            // backend healthy immediately
    mkdirp: (d) => record.push(`mkdir ${d}`),
    readState: () => null,               // fresh install
    writeState: () => record.push("writeState"),
    onStep: (s, st) => record.push(`step ${s}:${st}`),
  };
}

const opts: ProvOptions = {
  platform: "darwin", device: "cpu", extVersion: "0.2.0", reqHash: "abc",
  requirementsPath: "/ext/backend/requirements.txt", backendCwd: "/ext/backend",
  baseUrlOverride: undefined,
  manifest: {
    release: "v0.1-assets", baseUrl: "https://x/",
    assets: [
      { name: "extractor/pytorch_model.bin", file: "pytorch_model.bin", sha256: "h1", size: 1 },
      { name: "corpus/corpus_30k.jsonl", file: "corpus_30k.jsonl", sha256: "h2", size: 1 },
    ],
  },
};

describe("provision (fresh install)", () => {
  it("runs venv, torch(cpu index), deps, codebert, then launches and reports a url", async () => {
    const rec: string[] = [];
    const res = await provision(provPaths("/gs", "darwin"), opts, fakeDeps(rec));
    expect(res.backendUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    const joined = rec.join("\n");
    expect(joined).toContain("-m venv");
    expect(joined).toContain("download.pytorch.org/whl/cpu");
    expect(joined).toContain("-r /ext/backend/requirements.txt");
    expect(joined).toContain("spawn");
    expect(rec).toContain("writeState");
    // assets already correct (verify=true) so no download happened
    expect(joined).not.toContain("download ");
  });

  it("uses the cuda wheel index when device=cuda", async () => {
    const rec: string[] = [];
    await provision(provPaths("/gs", "darwin"), { ...opts, device: "cuda" }, fakeDeps(rec));
    expect(rec.join("\n")).toContain("download.pytorch.org/whl/cu121");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension && npx vitest run test/provision-orchestrate.test.ts`
Expected: FAIL — `provision` / `ProvDeps` / `ProvOptions` not exported.

- [ ] **Step 3: Extend `provision.ts`**

Append to `extension/src/provision.ts`:

```ts
import { ProvPaths } from "./paths";       // (already imported above — keep one import)
import { Manifest, AssetEntry, assetUrl } from "./assets";
import { Step, ProvInputs, stepsNeeded } from "./state";
import { Runner } from "./env";

export type StepStatus = "running" | "done" | "skipped" | "error";

export interface ProvDeps {
  run: Runner;
  spawnBackend: (python: string, args: string[], opts: { cwd: string; env: Record<string, string>; logPath: string }) => { pid?: number };
  download: (url: string, dest: string, onBytes?: (d: number, t: number) => void) => Promise<void>;
  verify: (path: string, a: AssetEntry) => Promise<boolean>;
  health: (url: string) => Promise<boolean>;
  mkdirp: (dir: string) => void;
  readState: () => import("./state").ProvState | null;
  writeState: (inputs: ProvInputs, codebertReady: boolean) => void;
  onStep: (step: Step, status: StepStatus, detail?: string) => void;
}

export interface ProvOptions {
  platform: NodeJS.Platform;
  device: string;
  extVersion: string;
  reqHash: string;
  requirementsPath: string;
  backendCwd: string;
  baseUrlOverride?: string;
  manifest: Manifest;
}

const TORCH_INDEX: Record<string, string> = {
  cpu: "https://download.pytorch.org/whl/cpu",
  cuda: "https://download.pytorch.org/whl/cu121",
};

function assetHashes(m: Manifest): Record<string, string> {
  const out: Record<string, string> = {};
  for (const a of m.assets) out[a.name] = a.sha256;
  return out;
}

export async function provision(
  p: ProvPaths, opts: ProvOptions, deps: ProvDeps,
): Promise<{ backendUrl: string }> {
  const inputs: ProvInputs = {
    extVersion: opts.extVersion, device: opts.device, reqHash: opts.reqHash,
    assets: assetHashes(opts.manifest),
  };
  const need = stepsNeeded(deps.readState(), inputs);
  const did = (s: Step) => deps.onStep(s, need.has(s) ? "done" : "skipped");
  let codebertReady = !need.has("codebert");

  if (need.has("venv")) {
    deps.onStep("venv", "running");
    deps.mkdirp(p.root);
    await deps.run(opts.platformPython(), ["-m", "venv", p.venv]);
    await deps.run(p.venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
  }
  did("venv");

  if (need.has("torch")) {
    deps.onStep("torch", "running");
    const index = TORCH_INDEX[opts.device] ?? TORCH_INDEX.cpu;
    await deps.run(p.venvPython, ["-m", "pip", "install", "torch", "--index-url", index]);
  }
  did("torch");

  if (need.has("deps")) {
    deps.onStep("deps", "running");
    await deps.run(p.venvPython, ["-m", "pip", "install", "-r", opts.requirementsPath]);
  }
  did("deps");

  if (need.has("assets")) {
    deps.onStep("assets", "running");
    for (const a of opts.manifest.assets) {
      const dest = a.name.includes("extractor") ? p.extractorWeights : p.corpus;
      if (await deps.verify(dest, a)) continue;
      const url = assetUrl(opts.manifest, a, opts.baseUrlOverride);
      await deps.download(url, dest, (d, t) => deps.onStep("assets", "running", `${a.file} ${d}/${t}`));
      if (!(await deps.verify(dest, a))) throw new Error(`checksum mismatch: ${a.file}`);
    }
  }
  did("assets");

  if (need.has("codebert")) {
    deps.onStep("codebert", "running");
    await deps.run(
      p.venvPython,
      ["-c", "from transformers import AutoModel,AutoTokenizer;AutoModel.from_pretrained('microsoft/codebert-base');AutoTokenizer.from_pretrained('microsoft/codebert-base')"],
      { HF_HOME: p.hfCache },
    );
    codebertReady = true;
  }
  did("codebert");

  // launch
  deps.onStep("launch", "running");
  const port = await findFreePort(8000);
  const env = { ...buildBackendEnv(p, opts.device) };
  deps.spawnBackend(p.venvPython, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(port)],
    { cwd: opts.backendCwd, env, logPath: p.backendLog });
  const url = `http://127.0.0.1:${port}`;
  let healthy = false;
  for (let i = 0; i < 60; i++) {
    if (await deps.health(url)) { healthy = true; break; }
    await new Promise((r) => setTimeout(r, 1000));
  }
  if (!healthy) { deps.onStep("launch", "error", "backend did not become healthy"); throw new Error("backend health timeout"); }
  deps.onStep("launch", "done");

  deps.writeState(inputs, codebertReady);
  return { backendUrl: url };
}
```

> The orchestrator references `opts.platformPython()` for the **host** Python
> (the interpreter used to *create* the venv, from Task 2's detection). Add it
> to `ProvOptions` as a field instead of a method to keep it plain data:

Replace the `opts.platformPython()` call with `opts.hostPython` and add
`hostPython: string;` to `ProvOptions`. Update the test's `opts` to include
`hostPython: "python3.11"` and `platform: "darwin"`. Re-run.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension && npx vitest run test/provision-orchestrate.test.ts`
Expected: PASS (2 tests). The fresh-install test asserts venv/torch-cpu/deps/spawn/writeState ran and that no real download occurred (verify returned true).

- [ ] **Step 5: Commit**

```bash
git add extension/src/provision.ts extension/test/provision-orchestrate.test.ts
git commit -m "feat(ext): provision orchestrator (venv->torch->deps->assets->codebert->launch)"
```

---

## Task 8: Wire the engine into `extension.ts`

**Files:**
- Modify: `extension/src/extension.ts`

Replace the naive `startBackend()` (which assumed a ready interpreter) with a
real provisioning path that builds the `ProvDeps` from Node APIs + vscode.

- [ ] **Step 1: Add a `provisionAndStart` helper and call it from `ensureBackend`**

Add these imports at the top of `extension/src/extension.ts`:

```ts
import * as fs from "node:fs";
import * as path from "node:path";
import { execFile } from "node:child_process";
import { provPaths } from "./paths";
import { detectPython, detectGpu } from "./env";
import { parseManifest, downloadAsset, verifyAsset, sha256File } from "./assets";
import { readState, writeState } from "./state";
import { provision, ProvDeps } from "./provision";
import { getHealth } from "./client";
```

Add this runner + orchestration glue (above `activate`):

```ts
function run(cmd: string, args: string[] = [], extraEnv?: Record<string, string>) {
  return new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
    execFile(cmd, args, { env: { ...process.env, ...extraEnv }, maxBuffer: 1 << 20 },
      (err, stdout, stderr) => resolve({ code: err ? ((err as any).code ?? 1) : 0, stdout, stderr }));
  });
}

async function provisionAndStart(ctx: vscode.ExtensionContext): Promise<string | undefined> {
  const root = ctx.globalStorageUri.fsPath;
  fs.mkdirSync(root, { recursive: true });
  const paths = provPaths(root, process.platform);
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  const backendCwd = path.join(ctx.extensionPath, "backend");
  const requirementsPath = path.join(backendCwd, "requirements.txt");
  const manifest = parseManifest(fs.readFileSync(path.join(backendCwd, "assets", "manifest.json"), "utf8"));

  const py = await detectPython(process.platform, cfg.get("backend.pythonPath", "") || undefined, run);
  if (!py) {
    vscode.window.showErrorMessage("Code Summary needs Python ≥3.10. Install it from https://www.python.org/downloads/ and retry.");
    return undefined;
  }
  const device = cfg.get("backend.device", "cpu");
  const reqHash = await sha256File(requirementsPath);

  const deps: ProvDeps = {
    run,
    spawnBackend: (python, args, o) => {
      const log = fs.createWriteStream(o.logPath, { flags: "a" });
      backendProc = spawn(python, args, { cwd: o.cwd, env: { ...process.env, ...o.env } });
      backendProc.stdout?.pipe(log); backendProc.stderr?.pipe(log);
      backendProc.on("exit", () => (backendProc = undefined));
      return { pid: backendProc.pid };
    },
    download: (url, dest, onBytes) => downloadAsset(url, dest, onBytes),
    verify: (p2, a) => verifyAsset(p2, a),
    health: (url) => getHealth(url),
    mkdirp: (d) => fs.mkdirSync(d, { recursive: true }),
    readState: () => readState(paths.state),
    writeState: (inputs, cb) => writeState(paths.state, inputs, cb),
    onStep: (s, st, detail) => console.log(`[provision] ${s}: ${st}${detail ? " " + detail : ""}`),
  };

  return await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Code Summary: setting up backend", cancellable: false },
    async () => {
      const { backendUrl } = await provision(paths, {
        platform: process.platform, hostPython: py.cmd.join(" "), device,
        extVersion: ctx.extension.packageJSON.version, reqHash,
        requirementsPath, backendCwd, baseUrlOverride: cfg.get("assets.baseUrl", "") || undefined,
        manifest,
      }, deps);
      await cfg.update("backend.url", backendUrl, vscode.ConfigurationTarget.Global);
      await cfg.update("backend.managed", true, vscode.ConfigurationTarget.Global);
      return backendUrl;
    });
}
```

> `hostPython` is a single string ("python3.11" or "py -3.11"); split it on
> space before passing to `run` inside the orchestrator. Update the orchestrator's
> venv step to `const [hp, ...hpArgs] = opts.hostPython.split(" ");` and call
> `await deps.run(hp, [...hpArgs, "-m", "venv", p.venv]);`. Adjust the Task 7
> test's `hostPython` accordingly (already `"python3.11"`).

- [ ] **Step 2: Replace `ensureBackend()` body**

```ts
async function ensureBackend(ctx: vscode.ExtensionContext): Promise<boolean> {
  const url = backendUrl();
  if (await getHealth(url)) return true;
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  // Already provisioned but not running? just relaunch via provision (launch step only).
  if (!cfg.get("backend.autoStart", true) && !cfg.get("backend.managed", false)) {
    vscode.window.showErrorMessage(`Code Summary backend not reachable at ${url}. Start it manually.`);
    return false;
  }
  try {
    const newUrl = await provisionAndStart(ctx);
    if (!newUrl) return false;
    for (let i = 0; i < 10; i++) { if (await getHealth(newUrl)) return true; await new Promise((r) => setTimeout(r, 500)); }
    return await getHealth(newUrl);
  } catch (e: any) {
    vscode.window.showErrorMessage(`Backend setup failed: ${e.message}. See the developer console / ${backendUrl()}.`);
    return false;
  }
}
```

> Thread `ctx` through: change `summarizeSelection()` to accept `ctx` and call
> `ensureBackend(ctx)`; register the command with a closure `() => summarizeSelection(ctx)`.
> Remove the old `startBackend()` body's interpreter assumptions but keep the
> `codeSummary.startBackend` command calling `provisionAndStart(ctx)`.

- [ ] **Step 3: Build to verify it compiles**

Run: `cd extension && npm run build`
Expected: esbuild writes `dist/extension.js` with no type/bundle errors.

- [ ] **Step 4: Run the whole unit suite**

Run: `cd extension && npx vitest run`
Expected: PASS — all prior tests still green (this task adds no new test; it is wiring verified by the build + the later end-to-end task).

- [ ] **Step 5: Commit**

```bash
git add extension/src/extension.ts
git commit -m "feat(ext): provision-managed backend lifecycle in ensureBackend"
```

---

## Task 9: Bundle the backend into the .vsix

**Files:**
- Create: `extension/scripts/copy-backend.js`
- Modify: `extension/package.json` (scripts + settings + activation)
- Modify: `extension/.vscodeignore`

- [ ] **Step 1: Write the copy script**

```js
// extension/scripts/copy-backend.js
// Copy the backend Python source into extension/backend/ so vsce can package it.
const fs = require("node:fs");
const path = require("node:path");

const SRC = path.join(__dirname, "..", "..", "backend");
const DST = path.join(__dirname, "..", "backend");
const SKIP = new Set([".venv", "__pycache__", ".pytest_cache", "tests",
  "code_summary_backend.egg-info", ".egg-info"]);

function copyDir(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const ent of fs.readdirSync(src, { withFileTypes: true })) {
    if (SKIP.has(ent.name) || ent.name.endsWith(".pyc")) continue;
    const s = path.join(src, ent.name), d = path.join(dst, ent.name);
    if (ent.isDirectory()) copyDir(s, d);
    else fs.copyFileSync(s, d);
  }
}

fs.rmSync(DST, { recursive: true, force: true });
copyDir(SRC, DST);
console.log(`copied backend -> ${DST}`);
```

- [ ] **Step 2: Update `package.json` scripts, settings, activation**

In `extension/package.json`:
- Set `"activationEvents": ["onStartupFinished"]`.
- Add to `scripts`: `"copy-backend": "node scripts/copy-backend.js"`, and `"package": "npm run copy-backend && npm run build && vsce package"`.
- Add to `contributes.configuration.properties`:

```json
"codeSummary.assets.baseUrl": { "type": "string", "default": "", "description": "Override the asset download base URL (e.g. a local folder for testing). Empty = use the shipped manifest." },
"codeSummary.backend.device": { "type": "string", "enum": ["cpu", "cuda"], "default": "cpu", "description": "Inference device for the backend (CUDA needs an NVIDIA GPU)." },
"codeSummary.backend.managed": { "type": "boolean", "default": false, "description": "Set automatically when the extension provisions and manages the backend." }
```

- Add to `contributes.commands`: `{ "command": "codeSummary.setup", "title": "Code Summary: Run Setup" }`.

- [ ] **Step 3: Update `.vscodeignore`**

Set `extension/.vscodeignore` to:

```
src/**
test/**
scripts/**
node_modules/**
.gitignore
tsconfig.json
esbuild.js
vitest.config.ts
**/*.map
.vscode/**
backend/**/__pycache__/**
backend/**/*.pyc
backend/.venv/**
backend/.pytest_cache/**
backend/tests/**
backend/**/*.egg-info/**
backend/assets/extractor/**
backend/assets/corpus/**
```

> The last two lines ensure the big runtime assets are **never** packed into the
> .vsix — they are downloaded at provision time. The manifest under
> `backend/assets/manifest.json` is NOT ignored (only the binary subfolders are).

- [ ] **Step 4: Verify packaging includes backend source, excludes assets/junk**

Run:
```bash
cd extension && npm run copy-backend && npx vsce ls | grep -E "backend/(app|vendor|requirements|assets/manifest)" | head
cd extension && npx vsce ls | grep -E "pytorch_model|corpus_30k|__pycache__|\.venv" || echo "no heavy/junk files packed ✅"
```
Expected: first command lists `backend/app/main.py`, `backend/vendor/easc/model.py`, `backend/requirements.txt`, `backend/assets/manifest.json`; second prints the "no heavy/junk files packed ✅" line.

- [ ] **Step 5: Commit**

```bash
git add extension/scripts/copy-backend.js extension/package.json extension/.vscodeignore
echo "backend/" >> extension/.gitignore   # the copied tree is a build artifact
git add extension/.gitignore
git commit -m "build(ext): bundle backend source into .vsix; exclude runtime assets"
```

---

## Task 10: Real end-to-end provision on this Mac

**Files:** none (verification task). Uses the local asset source to avoid a 390MB download.

- [ ] **Step 1: Point the asset source at the local copy**

In VS Code settings (`~/Library/Application Support/Code/User/settings.json`), set:
```json
"codeSummary.assets.baseUrl": "/Users/somnusyi/cs_assets",
"codeSummary.backend.device": "cpu"
```
(`~/cs_assets/pytorch_model.bin` and `corpus_30k.jsonl` already exist with the manifest's exact sizes/hashes.)

- [ ] **Step 2: Launch the extension in the Extension Development Host**

Run: open `extension/` in VS Code, press F5 (or `cd extension && npm run build` then load the dev host).
Expected: on startup the provisioner runs; a progress notification "Code Summary: setting up backend" appears; the developer console logs `[provision] venv: done`, `torch: done`, `deps: done`, `assets: done` (copied from local), `codebert: done`, `launch: done`.

- [ ] **Step 3: Confirm the managed backend is healthy and wired**

Run (after provisioning finishes):
```bash
PORT=$(grep -o '"codeSummary.backend.url":[^,]*' ~/Library/Application\ Support/Code/User/settings.json)
echo "$PORT"
curl -s "$(echo "$PORT" | grep -oE 'http://127.0.0.1:[0-9]+')/health"
```
Expected: `backend.url` is `http://127.0.0.1:<port>`; `/health` returns `{"status":"ok",...}`.

- [ ] **Step 4: Run the fib example end-to-end and check core blocks**

In the dev host: open a Ruby snippet, select a `fib` function, run **Code Summary: Summarize Selection** (online mode + a working LLM key in settings).
Expected: the result panel shows a summary and a trace whose core blocks include the loop (~0.519) and return (~0.511) — matching Phase 1's verified output. Record the actual probabilities.

- [ ] **Step 5: Commit a short verification note**

```bash
# capture the observed health + core-block result in the integration log
git add backend/INTEGRATION.md   # append a "Phase 2A end-to-end verified" line
git commit -m "docs: record Phase 2A local provision end-to-end verification"
```

---

## Self-Review

**Spec coverage:**
- §3.1/§3.2 state machine → Tasks 6–8 (orchestrator runs detect→venv→torch→deps→assets→codebert→launch→health). ✅
- §3.3 idempotent `state.json` → Task 3. ✅
- §3.4 `CS_*` env contract → Task 6 `buildBackendEnv`. ✅
- §4 manifest + L2NL release + `assets.baseUrl` override → Tasks 4, 5, plus override threaded in Tasks 7–8. ✅
- §2 globalStorage layout + bundled backend → Tasks 1, 9. ✅
- §7 cross-platform (python discovery, venv python path, nvidia-smi) → Tasks 1, 2. ✅
- §8 error handling (missing Python, checksum mismatch, health timeout) → Tasks 7, 8. ✅
- §9 unit tests for pure logic → Tasks 1–7. ✅
- **Out of scope (Phase 2B):** webview wizard UI, model-config cards, `codeSummary.setup` opening a UI (the command is registered in Task 9 but its UI is 2B). The `onStartupFinished` auto-trigger here provisions silently via progress notification; 2B replaces/augments it with the wizard.

**Placeholder scan:** No "TBD"/"add error handling here" — every step has concrete code or exact commands. The two prose "replace X with Y" notes in Tasks 7–8 (`platformPython()`→`hostPython`, `hostPython.split`) are explicit refactor instructions with the exact replacement, not placeholders.

**Type consistency:** `Runner` (env.ts) reused by `ProvDeps.run`. `ProvPaths` fields (`venvPython`, `extractorWeights`, `corpus`, `hfCache`, `state`) consistent across paths.ts/provision.ts/extension.ts. `ProvInputs`/`ProvState` consistent across state.ts and provision.ts. `Manifest`/`AssetEntry` consistent across assets.ts and provision.ts. `provision()` returns `{ backendUrl }` consumed in extension.ts.
