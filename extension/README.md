# Low-Resource Code Summary

低资源代码摘要 VS Code 插件。首次启动会打开 setup wizard，安装私有后端和论文流水线资产。

## 安装

1. 下载 `code-summary-0.2.0.vsix`。
2. VS Code 执行 `Extensions: Install from VSIX...`。
3. 按 setup wizard 完成：
   - `Online via Gitee`：国内默认资产源。
   - `Use local asset folder`：把 Gitee Release 的全部 `.partNNN` 分片放到一个目录后选择它。
   - `Retry failed parts`：网络中断后重试，已校验成功的资产会跳过。
4. 配置模型：
   - 云 API 的 key 存在 VS Code SecretStorage，不写入 settings。
   - 本地 Ollama / llama.cpp / vLLM 使用 OpenAI-compatible 地址。

## 常见问题

- Gitee 下载慢：手动下载分片，使用 `Use local asset folder`。
- sha256 不一致：删除对应分片后重下；安装器不会使用校验失败的文件。
- 本地模型 502：插件启动的后端会绕过 `127.0.0.1,localhost,::1` 代理；仍失败时检查模型服务是否提供 `/v1/chat/completions`。
- `/health.ready=false`：生产资产缺失。仅开发调试可开启 `codeSummary.backend.allowStubs`。
