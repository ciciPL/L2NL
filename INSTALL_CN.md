# Code Summary 插件安装说明

## 需要下载的文件

从 Gitee Release `v0.2-assets-cn` 下载：

- `code-summary-0.2.0.vsix`
- `assets-manifest.v2.json`
- `SHA256SUMS.txt`
- 所有 `.partNNN` 资产分片

## 推荐安装方式

1. VS Code 打开命令面板，执行 `Extensions: Install from VSIX...`。
2. 选择 `code-summary-0.2.0.vsix`。
3. 插件首次启动后，setup wizard 会自动打开。
4. 如果机器能访问 Gitee，直接点 `Online via Gitee` 后安装。
5. 如果下载不稳定，把所有 `.partNNN` 放到同一个本地目录，点 `Use local asset folder` 选择该目录。
6. 配置模型服务：
   - DeepSeek / OpenAI / 兼容 API：填写 base URL、model、API key。
   - Ollama / llama.cpp / vLLM：选择 Local runtime 并填写本地 `/v1` 地址。

## 排错

- 分片缺失：重新下载缺失的 `.partNNN`，再点 `Retry failed parts`。
- sha256 不一致：删除报错分片并重新下载。
- 本地模型连接失败：确认服务暴露 OpenAI-compatible `/v1/chat/completions`。
- `/health.ready=false`：说明生产资产没有装完整。不要开启 stub 给真实用户使用。
