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
