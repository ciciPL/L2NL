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
