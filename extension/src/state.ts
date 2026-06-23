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
  // extVersion is recorded for diagnostics but is not itself a re-provision
  // trigger: dependency changes are captured by reqHash, and the bundled backend
  // source is read live from extensionPath on every launch (not copied in).
  const envChanged = old.reqHash !== next.reqHash;
  const deviceChanged = old.device !== next.device;
  if (deviceChanged || envChanged) { steps.add("torch"); steps.add("deps"); }
  const assetsChanged = Object.keys(next.assets).some(
    (k) => old.assets[k] !== next.assets[k],
  );
  if (assetsChanged) steps.add("assets");
  const codebertChanged = Object.keys(next.assets).some(
    (k) => k.toLowerCase().includes("codebert") && old.assets[k] !== next.assets[k],
  );
  if (!old.codebertReady || codebertChanged) steps.add("codebert");
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
