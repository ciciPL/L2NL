# Low-Resource Code Summary

低资源代码摘要 VS Code 插件。首次启动会打开 setup wizard，安装私有后端和论文流水线资产。

## 安装

1. 下载 `code-summary-0.2.0.vsix`。
2. VS Code 执行 `Extensions: Install from VSIX...`。
3. 按 setup wizard 完成：
   - 先选择 `Online API` 或 `Offline llama.cpp`。
   - `Online via Gitee`：国内默认资产源。
   - `Use local asset folder`：把 Gitee Release 的全部 `.partNNN` 分片放到一个目录后选择它。
   - `Retry failed parts`：网络中断后重试，已校验成功的资产会跳过。
4. 配置模型：
   - 云 API 的 key 存在 VS Code SecretStorage，不写入 settings。
   - 离线 llama.cpp 可从 ModelScope 选择/搜索 GGUF，插件会下载模型并启动本地 `llama-server`。
   - 已有本地服务时，也可以填 OpenAI-compatible 地址。

## 常见问题

- Gitee 下载慢：手动下载分片，使用 `Use local asset folder`。
- sha256 不一致：删除对应分片后重下；安装器不会使用校验失败的文件。
- pip 下载慢：修改 `codeSummary.python.indexUrl`，默认是清华 PyPI 镜像。
- 离线 runtime 缺失：发布 `v0.2-runtime-llamacpp`，或设置 `codeSummary.offline.llamaServerPath` 指向已有 `llama-server`。
- 本地模型 502：插件启动的后端和 llama.cpp 会绕过 `127.0.0.1,localhost,::1` 代理；仍失败时检查模型服务是否提供 `/v1/chat/completions`。
- `/health.ready=false`：生产资产缺失。仅开发调试可开启 `codeSummary.backend.allowStubs`。
