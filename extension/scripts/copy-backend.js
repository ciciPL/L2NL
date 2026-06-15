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
