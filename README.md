# Low-Resource Code Summary

论文配套 VS Code 插件 + FastAPI 后端。插件会在首次 setup 时创建私有
Python venv、下载经过校验的流水线资产，并连接 OpenAI-compatible 模型服务。

## 用户安装

1. 从 Gitee Release 下载 `code-summary-0.2.0.vsix`。
2. VS Code 中执行 `Extensions: Install from VSIX...`。
3. 首次启动会打开 `Code Summary - Setup`：
   - 先选择 `Online API` 或 `Offline llama.cpp`。
   - 国内默认走 Gitee Release 分片资产。
   - 如果下载慢或公司网络拦截，先手动下载全部 `.partNNN` 文件到同一目录，再选择 `Use local asset folder`。
   - `try global mirrors` 只在国内源失败且用户明确勾选时使用。
4. 配置模型：
   - 在线：DeepSeek / OpenAI / 兼容服务，API key 存入 VS Code SecretStorage，不写 settings 明文。
   - 离线：默认 llama.cpp，从 ModelScope 选择或搜索 GGUF；插件下载模型、启动本地 `llama-server`，并写入 `http://127.0.0.1:<port>/v1`。
5. 选中代码后右键 `Code Summary: Summarize Selection`。

## 生产资产

生产模式默认不允许 demo stub。缺少任一资产时 `/health.ready=false`，并且
`/summarize` 返回明确错误。仅开发调试可设置：

```jsonc
{
  "codeSummary.backend.allowStubs": true
}
```

必须发布的资产：

- `extractor/pytorch_model.bin`
- `corpus/corpus_30k.jsonl`
- `codebert/codebert-base.tar.gz`

生成 Gitee 友好的分片、manifest 和校验文件：

```bash
cd backend
python deploy/build_assets_release.py \
  --src /path/to/prepared-assets \
  --out /path/to/release-assets \
  --release v0.2-assets-cn \
  --gitee-base-url https://gitee.com/ch2n2000/L2NL/releases/download \
  --global-base-url https://github.com/ciciPL/L2NL/releases/download \
  --split-size-mib 30 \
  --per-asset-releases
```

Gitee 主 Release `v0.2-assets-cn` 上传 `code-summary-0.2.0.vsix`、
`assets-manifest.v2.json`、`SHA256SUMS.txt`、`INSTALL_CN.md`。资产分片分别上传到：

- `v0.2-assets-cn-extractor`
- `v0.2-assets-cn-corpus`
- `v0.2-assets-cn-codebert`

离线 llama.cpp 还需要发布 runtime Release：

- Release tag：`v0.2-runtime-llamacpp`
- 平台：`win32-x64`、`darwin-arm64`、`darwin-x64`
- 每个平台上传 runtime archive 分片，并把 sha256、size、`executable` 写入 `extension/runtime-manifest.json`
- 用户已经安装 llama.cpp 时，也可以直接设置 `codeSummary.offline.llamaServerPath`

发布 VSIX 前，用生成出的 `assets-manifest.v2.json` 覆盖
`backend/assets/manifest.json`，确认 `extension/runtime-manifest.json` 已包含 runtime 包，再运行：

Gitee 单个附件限制为 100M，页面实测还会限制一次 Release 的附件数量；因此
默认使用 30MiB 分片并按资产拆分到多个 Release，降低网页登录上传失败率。

```bash
cd extension
npm run package
```

## 开发验证

```bash
cd extension && npm test && npx tsc --noEmit && npm run build
cd ../backend && .venv/bin/python -m pytest -q
```

本地模型如果开启系统代理，后端会自动设置 `NO_PROXY/no_proxy` 以绕过
`127.0.0.1,localhost,::1`，避免 Ollama/vLLM/llama.cpp 被代理劫持。

Python 依赖默认使用清华 PyPI 镜像：

```jsonc
{
  "codeSummary.python.indexUrl": "https://pypi.tuna.tsinghua.edu.cn/simple"
}
```

如果镜像慢，可切换到阿里云或中科大；安装器所有 pip 命令都会带
`--retries 5 --timeout 60 -i <indexUrl>`。

## 评测诚信边界

- 生产模式默认禁用 stub；真实论文/报告结果不得开启 `codeSummary.backend.allowStubs`。
- `corpus_30k.jsonl` 只能作为检索示例库使用，正式评测时必须确认它不包含目标
  test 集样本或参考摘要，避免 retrieval 泄漏答案。
- 当前 back-translation/Jina 选择分数未接入时，界面显示 `n/a`，不会用固定
  `1.0` 冒充真实一致性分数。
