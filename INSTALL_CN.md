# Code Summary 插件安装说明

## 需要下载的文件

从 Gitee Release `v0.2-assets-cn` 下载：

- `code-summary-0.2.0.vsix`
- `assets-manifest.v2.json`
- `SHA256SUMS.txt`
- 论文资产 `.partNNN` 分片

如果选择离线 llama.cpp，并且机器没有现成 `llama-server`，还需要
Gitee Release `v0.2-runtime-llamacpp` 里的对应平台 runtime 分片。

## 推荐安装方式

1. VS Code 打开命令面板，执行 `Extensions: Install from VSIX...`。
2. 选择 `code-summary-0.2.0.vsix`。
3. 插件首次启动后，setup wizard 会自动打开。
4. 先选择模型模式：
   - `Online API`：DeepSeek / OpenAI / 兼容 API。
   - `Offline llama.cpp`：ModelScope GGUF + 本地 `llama-server`。
5. 安装后端依赖和论文资产：
   - 机器能访问 Gitee：点 `Online via Gitee` 后安装。
   - 下载不稳定：把所有论文资产 `.partNNN` 放到同一个目录，点 `Use local asset folder` 选择该目录。
6. 在线模式：
   - 选择 DeepSeek/OpenAI/Custom。
   - 填写 base URL、model、API key。
   - API key 存在 VS Code SecretStorage，不写入 settings 明文。
   - 点 `Test connection`，通过后保存。
7. 离线模式：
   - 已经有本地模型：点 `Use local GGUF file`，选择已有 `.gguf` 文件，不会再下载模型。
   - 需要下载模型：默认选择 `Fast: Qwen2.5-Coder 1.5B Q4_K_M`。
   - 也可以输入 ModelScope repo ID，例如 `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`，搜索 `.gguf` 文件。
   - 点 `Download model & start llama.cpp`。
   - 插件会启动 `llama-server -m <gguf> --host 127.0.0.1 --port <freePort> -c 8192 --parallel 1`，并自动写入本地 `/v1` 地址。

Python 依赖默认走清华 PyPI 镜像。可在 settings 里修改：

```jsonc
{
  "codeSummary.python.indexUrl": "https://pypi.tuna.tsinghua.edu.cn/simple"
}
```

## 排错

- 分片缺失：重新下载缺失的 `.partNNN`，再点 `Retry failed parts`。
- sha256 不一致：删除报错分片并重新下载。
- pip 下载慢或失败：把 `codeSummary.python.indexUrl` 改成阿里云或中科大镜像后重试。
- 离线 runtime 缺失：确认 `v0.2-runtime-llamacpp` 已发布对应平台包；或手动安装 llama.cpp 后设置 `codeSummary.offline.llamaServerPath`。
- 本地模型连接失败：确认服务暴露 OpenAI-compatible `/v1/chat/completions`，并检查 `llama-server.log`。
- 系统代理导致 localhost 失败：插件启动后端和 llama.cpp 时会写入 `NO_PROXY/no_proxy=127.0.0.1,localhost,::1`。
- `/health.ready=false`：说明生产资产没有装完整。不要开启 stub 给真实用户使用。
