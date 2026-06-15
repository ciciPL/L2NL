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
