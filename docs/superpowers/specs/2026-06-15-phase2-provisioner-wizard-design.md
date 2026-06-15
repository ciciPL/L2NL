# 设计文档：Phase 2 —— Provisioner 引擎 + Setup 向导

- 日期：2026-06-15
- 目标定位：把 Phase 1 验证过的"纯本地流水线"包装成**成熟插件** —— 用户装上 .vsix，由插件**首次运行自动装后端**（分发方案 A），并提供**可视化 setup 向导**（环境检测 + 模型配置），无需手改 settings.json。
- 前置：Phase 1 已让 4 组件全真实现的流水线在 Mac 纯本地 CPU 跑通（与 star CUDA 结果一致）；`backend/` 是论文可复现实现，`backend/requirements.txt` 已就绪。
- 关联设计：`docs/superpowers/specs/2026-06-08-code-summary-vscode-artifact-design.md`（壳子总设计）。

---

## 1. 目标与范围

**本期做（一个 spec，实现分两阶段）：**

- **引擎层 `provision`（headless TS，无 UI 依赖）**：环境检测 → 建 venv → 装 torch+依赖 → 下资产 → 预热 codebert → 起 uvicorn → 等 `/health`。可被向导驱动，也可被现有 `ensureBackend()` 静默调用。
- **向导层 `wizard`（webview）**：把引擎每一步状态/进度可视化，外加模型配置（云 / 本地）UI。

**本期不做（YAGNI）：** 自动安装 Python（缺则给引导链接）、自动起 LLM（用户自配 OpenAI 兼容端点）、SSE 流式分阶段、Marketplace 发布打磨、多 IDE、遥测。

**两平台：** Mac（CPU，Phase 1 已验证）+ Windows（myci 实测）。设备默认 CPU，CUDA 可选。

---

## 2. 架构与磁盘布局

延续方案 A：VS Code 扩展（薄客户端）↔ 本地 Python 后端（FastAPI）。新增的是"后端由插件自己装、自己起、自己配"。

```
<globalStorage>/                       ← context.globalStorageUri，卸载即清，与工作区无关
├── venv/                              ← python -m venv，后端专用环境
├── assets/
│   ├── extractor/pytorch_model.bin    ← 从 L2NL Release 下（SelectorNet checkpoint）
│   └── corpus/corpus_30k.jsonl        ← 从 L2NL Release 下（BM25 语料，30k 采样）
├── hf_cache/                          ← HF_HOME，codebert 落这里（自包含，可随卸载清）
├── backend.log                        ← uvicorn 输出
└── state.json                         ← provisioning 标记（见 §3.3）

<extensionPath>/backend/               ← 后端 Python 源码，随 .vsix 分发（打包时拷入）
```

- 后端**源码随 .vsix 分发**（打包步骤把 `backend/` 拷进 bundle，`.vscodeignore` 放行）。
- venv 用 globalStorage 那个；uvicorn 以 `<extensionPath>/backend` 为 cwd 跑 `app.main:app`。
- 一切重资源（venv、权重、语料、HF 缓存）都落在 globalStorage，**卸载插件即可彻底清理**，不污染用户系统。

---

## 3. Provisioner 引擎

### 3.1 状态机（每步可独立重试）

```
detect → venv → torch → deps → assets → codebert → launch → health → ready
  │        │      │       │       │         │          │        │
  └─ 任一步失败 → 停在该步，向导显示错误 + 重试 + 查看日志 + 手动命令兜底
```

### 3.2 步骤定义

| 步 | 动作 | 要点 |
|---|---|---|
| **detect** | 找 Python ≥3.10；探 GPU | 见 §7 跨平台。GPU 探测决定是否放开 CUDA 选项 |
| **venv** | `<py> -m venv <gs>/venv` | venv/pip 是 stdlib，detect 过即有；建完升级 pip |
| **torch** | 按 device 选 index-url 单独装 torch | CPU=`https://download.pytorch.org/whl/cpu`；CUDA=`.../whl/cu121`。**先单独装 torch** 再装其余，保证跨平台确定性（避免 Linux 默认拉巨大 CUDA 轮 / device 不一致） |
| **deps** | `pip install -r <extensionPath>/backend/requirements.txt` | torch 已满足，pip 不会重装；其余为 fastapi/uvicorn/rank-bm25/transformers/sentence-transformers 等 slim 依赖 |
| **assets** | 按 manifest 下 checkpoint+语料到 `assets/`，逐个校验 sha256 | 大文件（316MB+71MB）走带进度、可断点续传的流式下载 |
| **codebert** | venv 内执行 `AutoModel.from_pretrained('microsoft/codebert-base')`（`HF_HOME=<gs>/hf_cache`）预热 | 把首个 summarize 的卡顿挪进向导；离线/HF 不可达时此步失败可重试 |
| **launch** | venv python 起 uvicorn，注入 `CS_*` env（§3.4）；探测空闲端口；写回 `codeSummary.backend.url` | 进程句柄交给现有生命周期管理；`backend.managed=true` |
| **health** | 轮询 `/health` 直到 `{status:ok}` 或超时 | 超时 → 报错 + 指向 backend.log |

### 3.3 幂等标记 `state.json`

```jsonc
{
  "schemaVersion": 1,
  "extVersion": "0.2.0",          // 扩展版本变 → 重新校核
  "device": "cpu",               // 切换 cpu/cuda → 重跑 torch 步
  "reqHash": "<sha256 of requirements.txt>",  // 依赖变 → 重跑 torch/deps
  "assets": { "extractor/pytorch_model.bin": "<sha256>", "corpus/corpus_30k.jsonl": "<sha256>" },
  "codebertReady": true
}
```
每步开始前比对标记：已完成且输入未变就跳过。升级扩展 / 改 requirements / 切 device → 只重跑受影响的步骤。

### 3.4 注入的 env（复刻 `deploy/star_start.sh`，本地化）

后端 env 契约（见 `backend/app/config.py`）：

```
CS_PYTHON            = <gs>/venv/bin/python        (win: <gs>/venv/Scripts/python.exe)
CS_EXTRACTOR_WEIGHTS = <gs>/assets/extractor/pytorch_model.bin
CS_CODEBERT_PATH     = (不设 → 默认 "microsoft/codebert-base"，走 HF_HOME 缓存)
CS_CORPUS_PATH       = <gs>/assets/corpus/corpus_30k.jsonl
CS_CORPUS_LIMIT      = 30000
CS_DEVICE            = cpu | cuda
HF_HOME              = <gs>/hf_cache
```

> Mac 必须 `CS_DEVICE=cpu`（MPS 会因 `_init_hidden_state` 只把隐藏态移 cuda 而 device 不一致，Phase 1 已验证）。

---

## 4. 资产 manifest + L2NL Release

`backend/assets/manifest.json`（随 .vsix）：

```jsonc
{
  "release": "v0.1-assets",
  "baseUrl": "https://github.com/ciciPL/L2NL/releases/download/v0.1-assets/",
  "assets": [
    { "name": "extractor/pytorch_model.bin",
      "file": "pytorch_model.bin",
      "sha256": "638efb2ab2862843d987e98117ad679065c4037f7246793f1e9ec4c3f8477de8",
      "size": 316071890 },
    { "name": "corpus/corpus_30k.jsonl",
      "file": "corpus_30k.jsonl",
      "sha256": "08b64305f2a846cad3283ab3340a95ae2c403120be5da42fe942055467860258",
      "size": 71406879 }
  ]
}
```

- 下载 URL = `baseUrl + file`，默认指向 ciciPL/L2NL 的 Release。
- `codeSummary.assets.baseUrl` 设置可覆盖 base（本机测试指 `~/cs_assets/` 或 `file://`，不卡在 Release 上线）。
- codebert **不进** manifest —— 由 transformers 从 HF 自动拉到 `HF_HOME`（codebert 步）。

---

## 5. Setup 向导（webview）

形态选定：**专用多步 webview 向导**（相比纯原生 QuickPick / Walkthrough，唯一能满足"可视化环境检测面板 + 云/本地配置卡片 + 流式安装进度"）。引擎是 headless，向导通过 postMessage 驱动它。

### 5.1 步骤

1. **欢迎**：插件做什么；setup 将做什么（建本地 venv、下约 ~390MB 资产 + codebert）；「开始」。
2. **环境检测**：展示 Python 版本（✓/✗ ≥3.10）、pip、GPU/CUDA。Python 缺/旧 → 阻断 + 安装链接。设备选择器（CPU 默认 / 检测到 CUDA 才放开）。
3. **安装后端**：venv → torch → deps → assets → codebert → launch → health，逐步显示状态 + 流式日志 + 下载进度条。失败 → 该步重试 / 查看日志 / 手动命令。
4. **模型配置**（两卡）：
   - **云**：厂商预设下拉（DeepSeek / OpenAI / 自定义 → 填 base_url）+ API key + model。
   - **本地**：base_url（Ollama / llama.cpp / vllm / sglang，OpenAI 兼容）+ model + 「测试连接」。
   - 写入 `codeSummary.online.* / offline.* / mode`（复用 `config.ts`，`buildRequest` 不变）。
5. **完成**：摘要可用，给一段示例代码「试一下」。

何时弹：首次激活（`onStartupFinished`）检测到未 provisioned → 自动开；命令 `Code Summary: Setup` 随时重开。

### 5.2 postMessage 协议

```
webview → host : start | retry(step) | selectDevice(cpu|cuda) | saveModel(payload) | testConnection(payload) | openLog
host → webview : envResult(python,pip,gpu) | stepState(step,status) | stepProgress(step,pct,log)
                 | stepError(step,message) | ready(backendUrl) | testResult(ok,message)
```

host 侧用引擎执行各步，把进度/日志推给 webview；webview 只渲染、收集配置回写设置。

---

## 6. 清单 / 打包 / 重构改动

- **package.json**
  - 新命令：`codeSummary.setup`。
  - 新设置：`assets.baseUrl`(string)、`backend.device`(enum cpu/cuda，默认 cpu)、`backend.managed`(bool，标记由 provisioner 托管)。
  - `activationEvents: ["onStartupFinished"]`（首次检测是否需要 setup）。
- **`.vscodeignore` / `esbuild.js`**：打包时把 `backend/`（含 `app/`、`vendor/`、`requirements.txt`、`assets/manifest.json`）拷入 bundle，排除 `backend/.venv`、`__pycache__`、`*.pyc`、`tests`、`.pytest_cache`。
- **`extension.ts` 重构**：`startBackend()` 改为走引擎托管的 venv + `CS_*` env（不再用裸 `backend.pythonPath`）；`ensureBackend()` 先查 `state.json`，未 provisioned 则开向导而非直接 spawn。保留 `Start/Stop Backend` 命令。
- **新文件**：`provision.ts`（引擎编排）、`env.ts`（Python/GPU 检测）、`assets.ts`（manifest 解析 + 下载 + sha256 校验）、`state.ts`（state.json 读写比对）、`wizard.ts`（webview 面板 + 协议）。各文件单一职责、接口清晰、可独立单测。

---

## 7. 跨平台细节

| 关注点 | macOS / Linux | Windows |
|---|---|---|
| Python 发现 | 依次试 `backend.pythonPath` → `python3.11` → `python3.10` → `python3` → `python`，各跑 `-c "import sys;print(sys.version_info[:2])"` 取首个 ≥3.10 | `py -3.11` → `py -3.10` → `py -3` → `python` |
| venv python 路径 | `<gs>/venv/bin/python` | `<gs>/venv/Scripts/python.exe` |
| GPU 探测 | `nvidia-smi` 成功 → 放开 CUDA（Mac 无 CUDA，永远 CPU；MPS 不用） | `nvidia-smi` 成功 → 放开 CUDA |
| 默认设备 | CPU | CPU（有 N 卡用户可选 CUDA） |

---

## 8. 错误处理与回退

- 每步独立可重试；失败给**可操作文案** + 「查看日志」（output channel / backend.log）+ **手动命令兜底**（贴等效 shell，让用户能自己救）。
- Python 缺 / 版本低 → 阻断并给官方安装链接（不自动装）。
- 资产下载断 → 断点续传重试；sha256 不匹配 → 删除重下。
- CUDA 装 torch 失败 → 一键回退 CPU 重装 torch。
- LLM 连接错误（模型配置「测试连接」）→ 明确提示（云：检查 key/base_url；本地：检查 llama-server/Ollama 是否启动）。

---

## 9. 测试策略

- **引擎纯函数走 vitest 单测**：Python 发现解析、manifest 解析、sha256 校验、`CS_*` env 拼装、`state.json` 比对逻辑。
- 有副作用的步骤（venv/pip/下载/spawn）通过**接口注入 + mock** 测编排（成功/失败/重试路径），不真跑。
- 向导：手动冒烟。
- **整链路真跑一次**：本机 `codeSummary.assets.baseUrl` 指向 `~/cs_assets/`，跑完整 provision → fib 例端到端，核心块 `loop 0.519 / return 0.511` 与 Phase 1 一致即通过。

---

## 10. 范围边界（YAGNI）

- **不做**：自动装 Python / LLM、SSE 流式、多 IDE、Marketplace 打磨、账户/遥测、Windows 上 CUDA 版本自动匹配（给用户选 cu121，不猜）。
- **做**：env 检测 + venv + torch(CPU/CUDA) + deps + 资产下载校验 + codebert 预热 + uvicorn 托管启动 + 健康等待 + 可视化向导 + 云/本地模型配置。

---

## 11. 实现阶段划分

1. **引擎**（headless，可单测）：`env.ts` → `assets.ts` → `state.ts` → `provision.ts` 编排 → 接入 `extension.ts` 的 `ensureBackend()`；用 `~/cs_assets/` 作 baseUrl 本机真跑验证。
2. **向导**（webview）：`wizard.ts` + 步骤 UI + postMessage 接引擎 + 模型配置两卡 → 首次激活自动弹 + `Code Summary: Setup` 命令。
3. **打包**：`.vscodeignore`/esbuild 带上 `backend/` + `manifest.json`；重打 .vsix；Mac 装测一遍。

> 资产侧（独立于代码）：把 `pytorch_model.bin` + `corpus_30k.jsonl` 挂到 ciciPL/L2NL 的 `v0.1-assets` Release；L2NL 仓库代码用整理好的 `L2NL_release.zip` 替换。
