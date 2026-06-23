# Backend Production Assets

The VS Code extension starts this backend from the packaged `backend/` folder.
Production mode is asset-gated: extractor checkpoint, BM25 corpus, and local
CodeBERT files must all be installed before `/summarize` runs.

## Runtime Inputs

The extension writes these environment variables when it launches the backend:

| Variable | Purpose |
|---|---|
| `CS_EXTRACTOR_WEIGHTS` | Local `extractor/pytorch_model.bin` checkpoint. |
| `CS_CORPUS_PATH` | Local `corpus/corpus_30k.jsonl` retrieval corpus. |
| `CS_CODEBERT_PATH` | Extracted local `assets/codebert-base` directory. |
| `CS_ASSETS_MANIFEST` | Installed `assets-manifest.v2.json` for health checks. |
| `CS_ALLOW_STUBS` | `0` by default. Set `1` only for demo/local development. |
| `NO_PROXY`, `no_proxy` | Includes `127.0.0.1,localhost,::1` for local LLM endpoints. |

## Asset Rules

- The installer reads a manifest v2 with Gitee as the default source.
- Each asset may be split into `.partNNN` files. Every part is checked with
  sha256 before merge, and the merged file is checked again before install.
- CodeBERT is installed from a local archive and extracted to
  `assets/codebert-base`; runtime must not fetch from Hugging Face.
- `/health` reports `ready`, per-asset `present`, and per-asset `checksum`
  (`true`, `false`, or `null` if no matching manifest entry exists).
- `/summarize` fails clearly in production when any required asset is missing or
  checksum verification fails.

## Release Asset Build

Prepare this source layout outside Git:

```text
assets-src/
  extractor/pytorch_model.bin
  corpus/corpus_30k.jsonl
  codebert/codebert-base.tar.gz
```

Generate the free domestic release attachment set:

```bash
python backend/deploy/build_assets_release.py \
  --src assets-src \
  --out release-assets/v0.2-assets-cn \
  --gitee-base-url https://gitee.com/ch2n2000/L2NL/releases/download/v0.2-assets-cn \
  --global-base-url https://github.com/ciciPL/L2NL/releases/download/v0.2-assets-cn
```

Upload `assets-manifest.v2.json`, `SHA256SUMS.txt`, and all `.partNNN` files to
the Gitee Release. Do not commit model or corpus files to Git.
