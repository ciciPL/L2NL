export interface ProvPaths {
  root: string;
  venv: string;
  venvPython: string;
  assets: string;
  extractorWeights: string;
  corpus: string;
  codebertArchive: string;
  codebertDir: string;
  assetsManifest: string;
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
    codebertArchive: j("assets", "codebert-base.tar.gz"),
    codebertDir: j("assets", "codebert-base"),
    assetsManifest: j("assets", "assets-manifest.v2.json"),
    hfCache: j("hf_cache"),
    backendLog: j("backend.log"),
    state: j("state.json"),
  };
}
