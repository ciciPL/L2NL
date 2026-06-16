# 设计文档：插件 UI 重设计 —— Pipeline Showpiece（方向 B）

- 日期：2026-06-15
- 目标：把插件三个界面（结果面板 / 设置向导 / 状态栏）统一到「pipeline showpiece」视觉语言——把论文方法「翻译→检索→抽取→生成」做成可视化、可读、像配套 artifact 的演示界面；顺带修两个向导 UX 坑。
- 用户已选方向 B 并批准本设计。
- 范围：**纯前端视觉 + 两个向导 UX 修复**。不动论文流水线、后端、HTTP 协议、provisioner。

> ⚠️ 这些 webview 渲染在 VS Code 内，配色一律用 `--vscode-*` 主题变量（自动适配 Light Modern / 深色）。不要硬编码颜色。不要依赖 Tabler/codicon 字体——图标用内联 SVG 或排版字符。

---

## 1. 共享视觉语言（新 `extension/src/ui.ts`）

把 B 的设计语言抽成一个共享、**无 vscode 依赖**的模块，`panel.ts` 与 `wizard.ts` 都引它（DRY）。导出：

- `escapeHtml(s)`：HTML 转义（panel.ts 现有 `esc` 迁移过来，统一）。
- `BASE_CSS: string`：共享样式表字符串（tokens + 组件类）。组件类：
  - `.cs-stepper` / `.cs-step`：4 步管线条（圆点/编号 + 标签 + 连接线；末步/当前步用 `--vscode-textLink-foreground` 强调）。
  - `.cs-badge`（中性，`--vscode-badge-background/foreground`）、`.cs-badge--score`（回译分数，用 `--vscode-testing-iconPassed` 绿；`↺` 前缀字符）、`.cs-badge--warn`（repaired / fell-back，用 `--vscode-charts-yellow` 或 `editorWarning`）。
  - `.cs-card`：卡片（`--vscode-editorWidget-background` + 1px `--vscode-widget-border`，圆角）。
  - `.cs-scorebar`：BM25 分数条（背景轨 + 填充宽度按归一化分数；填充色 `--vscode-progressBar-background`）。
  - `.cs-code`：等宽代码块（`--vscode-textCodeBlock-background`，`white-space:pre-wrap`，可手选）。
  - `.cs-hl`：核心块高亮 span（`--vscode-editor-findMatchHighlightBackground` 类的半透明底）。
  - `.cs-chip--prob`：概率芯片（小、等宽数字）。
- 纯渲染 helper（可单测）：
  - `stepper(steps: {label: string, status: "done"|"active"|"pending"|"error"}[]): string` —— 通用 N 步条（面板用 4 个管线步、向导用 5 个进度步，状态由调用方给）。
  - `scoreBadge(score: number): string`、`warnBadges(repaired, fellBack): string`。
  - `scoreBar(score: number, max: number): string` —— 归一化到 0–100% 宽度。
  - `highlightCoreBlocks(pivotCode: string, blocks: {text:string,prob:number}[]): string` —— 在 pivot 代码里把命中的语句包成 `.cs-hl` + 末尾概率芯片；未命中原样。**纯字符串处理**，按 block.text 在代码里定位（trim 后逐行匹配；匹配不到则在 ③ 区另列），可单测。

`stepper`/`scoreBadge`/`scoreBar`/`highlightCoreBlocks` 走 vitest。

---

## 2. 结果面板 `panel.ts`（主秀，保持 `enableScripts:false`）

`renderBody(resp)` 重写为，自上而下：

1. **管线 stepper**：翻译→检索→抽取→生成，四步全亮（成功态），末步「生成」强调色——表示这次摘要走完了全链路。失败响应时，把失败的那一步标红（用 `resp.failed_stage` 映射到步序）。
2. **摘要 hero**：`resp.summary`，大字（~1.25em，weight 500），醒目置顶。
3. **① Pivot 翻译**（`<details open>`）：标题行带 `scoreBadge(selected_score)` + `warnBadges(repaired, fell_back)`；内容是 **源代码 ↔ pivot_code 并排**（窄屏自动堆叠，用 CSS grid `repeat(auto-fit,minmax(240px,1fr))`），各自 `.cs-code`。`candidates` 多温度候选折进一个次级 `<details>`。
4. **② 检索示例**（`<details>`）：每个示例一张 `.cs-card`：摘要 + `scoreBar(score, maxScore)` + 代码（`.cs-code`）。`maxScore` = 本次检索最大分，用于归一化。
5. **③ 核心语句块**（`<details>`）：用 `highlightCoreBlocks(pivot_code, core_blocks)` 在 pivot 代码上**原位高亮**选中句 + 概率芯片。底下附一行小字：阈值/选中数。
6. **④ 完整 prompt**（`<details>`）：`.cs-code` 等宽全文。

`STYLE` 常量替换为 `BASE_CSS`（来自 ui.ts）。`ResultPanel.loading()` 也套同壳（顶部 stepper 灰 + 「Summarizing…」）。错误态：失败步在 stepper 标红 + `.cs-card` 里显示 `failed_stage` + `error`。

---

## 3. 设置向导 `wizard.ts`（同一视觉语言 + 两个 UX 修复）

- **视觉**：顶部 5 步进度 stepper（welcome/env/install/model/done，当前步强调）；环境检测结果、安装步骤、模型配置全部卡片化；安装步用统一状态图标（✓/…/✗/·）；云/本地用 **segmented 切换**（替代现在的 tab div）；引 `BASE_CSS`。webview 仍 `enableScripts:true`（向导本就要交互）。
- **修坑①（预填）**：webview 加载后，host 把当前设置发给 webview（新消息 `type:"initModel"`，载 `mode/online.*/offline.*`），表单据此预填——key、base_url、model 不再一片空白。
- **修坑②（不覆盖空 key）**：`saveModel` 的纯逻辑抽成 `mergeModelConfig(existing, formInput)`（在 `models.ts` 或 `ui.ts`）：**当 online.apiKey 表单为空时，保留 existing.apiKey**（不写空覆盖）。其余字段照表单。该函数走 vitest。host 的 saveModel 调它再写设置。

---

## 4. 状态栏（`extension.ts`，微调）

保持 `☁️/💻 Summary: mode · model`。微调：用 codicon `$(cloud)/$(vm)` 已在用，保留；当 `!backend.managed` 时文案显示 `$(rocket) Summary: setup`（点击触发 `codeSummary.setup` 开向导而非 toggle）。已 managed 时维持现状（点击 toggle）。状态栏能动的有限，止于此。

---

## 5. 文件结构改动

| 文件 | 改动 |
|---|---|
| `extension/src/ui.ts` | 新建：`escapeHtml` + `BASE_CSS` + 纯渲染 helper（stepper/scoreBadge/scoreBar/warnBadges/highlightCoreBlocks） |
| `extension/src/panel.ts` | 重写 `renderBody`/样式，引 ui.ts；保持 `enableScripts:false` |
| `extension/src/wizard.ts` | 套 BASE_CSS + stepper + 卡片 + segmented；加 `initModel` 预填 + 用 `mergeModelConfig` |
| `extension/src/models.ts` | 加 `mergeModelConfig(existing, form)`（纯，+ 单测） |
| `extension/src/extension.ts` | 状态栏微调（setup 态） |
| 测试 | `ui.test.ts`（helper）、`models.test.ts` 增 `mergeModelConfig` 用例 |

---

## 6. 测试策略

- **纯函数走 vitest**：`escapeHtml`、`stepper`、`scoreBadge`、`scoreBar`、`highlightCoreBlocks`（命中/未命中/多块）、`mergeModelConfig`（空 key 不覆盖、其余覆盖）。
- **webview 手动冒烟**（VS Code 老规矩，无 webview 单测）：用之前 merge_intervals 例子看面板新样式；重开向导看预填 + segmented + 进度条。
- `npm run build` + `npx tsc --noEmit` 干净；`.vsix` 能打包。

---

## 7. 范围边界（YAGNI）

- **不做**：结果面板里加脚本（一键复制 / 流式逐段点亮——后续增强，本轮放弃，代码可手选）；改后端/协议/流水线；多 IDE；Marketplace 打磨。
- **做**：三界面统一到 B 视觉语言 + ui.ts 共享层 + 向导两个 UX 修复 + 状态栏 setup 态微调。
