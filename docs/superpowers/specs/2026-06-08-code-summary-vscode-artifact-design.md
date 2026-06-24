# 设计文档：低资源代码摘要 VS Code 插件（论文配套 artifact）

- 日期：2026-06-08
- 目标定位：论文配套 artifact / demo —— 忠实复现论文 4 段流水线、可复现优先，可接受本地后端与下载模型/语料
- 论文：*Bridging the Data Gap: Leveraging High-Resource Pivots for Low-Resource Code Summarization without Parallel Corpora*（ESWA 投稿）

---

## 1. 背景与目标

论文方法是一条 4 段流水线：把低资源语言代码（如 Ruby）"借道"高资源 pivot 语言（如 Python），再生成自然语言摘要：

1. **跨语言翻译**：多温度采样生成 pivot 候选 → AST 可解析性校验 + 基于 parser 反馈的迭代修复 → 基于回译一致性 `S(x,x̂)=λ·BLEU+(1-λ)·SBERT_sim` 选优。
2. **检索增强**：以 pivot 代码为 query，BM25 从高资源平行语料检索 top-k 代码-摘要示例。
3. **核心语句块抽取**：AST 按语法边界切语义块（loop/conditional/assignment/signature/other），**训练好的二分类器**判定哪些是核心块（过阈值保留）。
4. **结构化摘要生成**：把 pivot 代码 + 其核心块 + k 个示例（每个含代码/核心块/摘要）拼成**带 XML 标签边界**的结构化 prompt，引导 LLM 输出摘要，按标签抽取摘要正文。

**插件目标**：给审稿人/读者一个能装、能跑、能看到每一段中间产物的 demo，同时后端本体就是论文的可复现实现。

**现有资产**（用户确认都在）：整理好的流水线代码（本地）、训练好的抽取器权重与模型（远程）、高资源 BM25 语料。本设计先搭"接口干净的壳子 + 可端到端跑通的 stub"，真实现之后 file-by-file 填入。

---

## 2. 架构总览

方案 A：**VS Code 扩展（薄客户端） ↔ 本地 Python 后端（FastAPI，HTTP）**。

- 后端承载完整流水线，常驻进程（启动时加载 SBERT / 抽取器权重 / BM25 索引，避免每请求冷启动）。
- 后端带 `cli.py`，**脱离插件也能跑通整条流水线** —— 即论文的可复现入口。
- 扩展只负责：取选中代码 → 拼请求 → 调后端 → 渲染摘要与 trace、管理后端进程生命周期、模型在线/离线切换。
- **在线/离线不分叉代码**：模型层是单一 OpenAI 兼容客户端，仅 `base_url` 不同（在线指向云端 API；离线指向本地 `llama-server` 的 `/v1`，llama.cpp server 原生 OpenAI 兼容）。

### 仓库布局

```
code-summary-artifact/
├── backend/
│   ├── pyproject.toml
│   ├── cli.py                  # 独立可复现入口
│   ├── assets/                 # 抽取器权重、BM25 语料（后续填入）
│   └── app/
│       ├── main.py             # FastAPI: POST /summarize, GET /health
│       ├── pipeline.py         # 编排 4 段
│       ├── schemas.py          # pydantic 请求/响应 + Trace
│       ├── config.py
│       ├── models/
│       │   ├── llm.py          # OpenAI 兼容客户端（base_url 切在线/离线）
│       │   └── embedder.py     # SBERT 封装
│       └── components/
│           ├── translator.py   # ① 翻译
│           ├── retriever.py    # ② BM25 检索
│           ├── extractor.py    # ③ 核心块抽取
│           └── generator.py    # ④ 结构化 prompt 生成
└── extension/
    ├── package.json
    └── src/
        ├── extension.ts        # 激活、命令、后端生命周期
        ├── client.ts           # 后端 HTTP 客户端
        ├── panel.ts            # webview：摘要 + trace
        └── config.ts           # 设置 → 请求
```

---

## 3. 后端组件接口

所有跨组件数据结构用 pydantic（便于序列化进 trace）。

### 数据结构（`schemas.py`）

```python
class CoreBlock(BaseModel):
    text: str
    block_type: str          # loop / conditional / assignment / signature / other
    prob: float

class Example(BaseModel):
    code: str
    core_blocks: list[CoreBlock]
    summary: str
    score: float             # BM25 分

class TranslationResult(BaseModel):
    pivot_code: str
    candidates: list[str]    # 各温度候选（trace 展示）
    selected_score: float    # 回译一致性 S(x,x̂)
    repaired: bool
    fell_back: bool          # 是否回退到 τ=0 候选

class Trace(BaseModel):
    translation: TranslationResult
    retrieved: list[Example]
    core_blocks: list[CoreBlock]   # pivot 代码的核心块
    prompt: str                    # 喂给 LLM 的完整 prompt

class SummarizeResponse(BaseModel):
    summary: str
    trace: Trace | None
    error: str | None = None       # 失败时填
    failed_stage: str | None = None
```

### 组件类（`components/*.py`）

```python
class Translator:
    def __init__(self, llm: LLMClient, embedder: Embedder, params: TranslateParams): ...
    def translate(self, code: str, src_lang: str) -> TranslationResult: ...
    # 多温度采样 → AST 校验/迭代修复 → 回译选优(BLEU+SBERT)

class Retriever:
    def __init__(self, corpus_path: str): ...   # 启动时建 BM25 索引
    def retrieve(self, pivot_code: str, k: int) -> list[Example]: ...

class Extractor:
    def __init__(self, weights_path: str): ...   # 加载训练好的分类器
    def extract(self, code: str, threshold: float) -> list[CoreBlock]: ...
    # AST 切语义块 → 分类器打分 → 过阈值

class Generator:
    def __init__(self, llm: LLMClient): ...
    def generate(self, pivot_code: str, core_blocks: list[CoreBlock],
                 examples: list[Example]) -> tuple[str, str]: ...
    # 拼结构化带标签 prompt → LLM → 抽标签内摘要；返回 (summary, full_prompt)
```

### 模型层（`models/`）

```python
class LLMClient:                                  # 在线/离线唯一区别在 base_url
    def __init__(self, base_url: str, api_key: str, model: str): ...
    def chat(self, messages: list[dict], temperature: float) -> str: ...

class FakeLLMClient(LLMClient):                   # 测试/stub 用，确定性返回
    ...

class Embedder:                                   # SBERT，本地常驻
    def encode(self, texts: list[str]) -> "np.ndarray": ...
```

### 编排（`pipeline.py`）

`translate → retrieve(pivot) → extract(pivot) + extract(每个 example) → generate`，一路把中间产物收进 `Trace`。

设计要点：
- `Retriever`/`Extractor`/`Embedder` 在**进程启动时**加载常驻资源，不在请求路径里重建。
- `Extractor.extract` 对 pivot 与每个检索示例都跑（论文里示例也带 core blocks）。
- `Generator` 回传完整 prompt 供 trace 展示。
- 组件构造时注入依赖（llm / embedder / 路径）。注：若用户现有代码偏脚本/函数式，组件内部可直接调用现有函数，类只作为薄适配层 —— 接口签名不变。

---

## 4. API 契约与数据流

### 端点

```
GET  /health  -> {"status": "ok", "model_loaded": bool}
POST /summarize
```

请求：
```json
{
  "code": "def foo...",
  "language": "ruby",
  "model":  { "base_url": "...", "api_key": "...", "model": "..." },
  "params": { "k": 5, "temperatures": [0, 0.4, 0.8],
              "lambda": 0.5, "threshold": 0.5, "max_repair_iters": 3 },
  "trace": true
}
```

响应：`SummarizeResponse`（见上）。`trace=false` 时只返回 `summary`。

数据流：扩展取选中代码与当前 `mode` 对应的模型块 → POST /summarize → 后端跑流水线 → 返回摘要 + trace → 面板渲染。

MVP 为单次请求/响应（含完整 trace）。**后续增强**：SSE 流式分阶段推送（翻译完→检索完→抽取完→摘要），面板逐段点亮。

---

## 5. VS Code 扩展

### 命令与入口
- `Code Summary: Summarize Selection` —— 编辑器右键菜单 + 命令面板。
- `Code Summary: Start Backend` / `Stop Backend`。
- 状态栏项：显示 `☁️ online · <model>` 或 `💻 offline · <model>`，点击切换 mode。

### 结果面板（webview，侧边）
- 顶部：最终摘要（醒目）。
- 可折叠 trace：① Pivot 翻译（候选列表 / 回译得分 / `repaired`·`fell_back` 徽章）② 检索示例（k 个，带 BM25 分）③ 核心语句块（高亮，带概率）④ 完整 prompt。
- 运行时进度提示（流水线多步，可能慢）。

### 配置（`contributes.configuration`）
```
codeSummary.mode                : "online" | "offline"   (default "online")
codeSummary.online.baseUrl      : string                 (default "https://api.openai.com/v1")
codeSummary.online.apiKey       : string
codeSummary.online.model        : string
codeSummary.offline.baseUrl     : string                 (default "http://localhost:8080/v1")
codeSummary.offline.model       : string
codeSummary.params.k            : number  (default 3)
codeSummary.params.temperatures : number[] (default [0,0.4,0.8])
codeSummary.params.lambda       : number  (default 0.5)
codeSummary.params.threshold    : number  (default 0.5)
codeSummary.params.maxRepairIters: number (default 3)
codeSummary.backend.url         : string  (default "http://localhost:8000")
codeSummary.backend.autoStart   : boolean (default true)
codeSummary.backend.pythonPath  : string  (default "python")
```
`config.ts` 按 `mode` 选 online/offline 块拼进请求。

### 后端生命周期
激活时 ping `/health`；挂了且 `autoStart` 为真，用 `pythonPath` 拉起后端（`python -m app.main` / uvicorn），失败给出可操作提示。

---

## 6. 错误处理

- 边界用 pydantic 校验请求。
- 每段流水线包一层；失败返回 `{error, failed_stage}`，面板显示"卡在哪一段"（demo 调试友好）。
- 翻译修复用尽 → 回退 τ=0 候选（论文行为），置 `fell_back=true`，**不算错误**，继续。
- LLM/网络错误明确提示；离线连接被拒 → 提示"llama-server 是否已启动"。

---

## 7. 测试策略

- **桩先行**：每个组件先发返回合理假数据的 stub，配 `FakeLLMClient`，使后端 + 插件**立刻端到端跑通**；真实现 file-by-file 替换。
- 后端组件级单测用 `FakeLLMClient`（确定性）测编排，不依赖真实模型/权重。
- 可选：一个打到本地 `llama-server` 的集成测试。
- 扩展：最小冒烟（手动）+ `config.ts` 的设置→请求映射单测。

---

## 8. 技术选型

- 后端：FastAPI + uvicorn、rank-bm25、sentence-transformers、tree-sitter（多语言：ruby/python 等）、openai Python SDK（在线/离线通吃）。
- 扩展：纯 TypeScript + vanilla webview（不引重框架）。
- 仓库：独立 git 仓库，位于 `…/eswa_submission_workspace/code-summary-artifact/`，不污染论文目录。

---

## 9. 范围边界（YAGNI）

- **本期不做**：抽取器训练流程（用现成权重）、SSE 流式、多 IDE（仅 VS Code）、Marketplace 发布打磨、用户账户/遥测。
- **本期做**：端到端可跑的壳子（stub）+ 4 段接口 + 在线/离线切换 + trace 面板 + cli 复现入口。
