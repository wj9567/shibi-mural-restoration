# CHANGELOG

## 2026-11-16 — 阶段 5：DeepSeek 修复评估与人工复核建议

- 新增“AI 修复评估（DeepSeek）”：发送原图、本地修复图、已确认区域（编号/类型/修复方法/掩膜占比）和必要的修复前后裁剪图，可选附用户参考背景；结果按 `region_id` 返回观察、修复评价、建议、依据、风险、不确定性与人工复核原因。
- 严格区域绑定：未知、缺失、重复的 `region_id`，字段缺失或无效 JSON 一律丢弃并提示，不生成不存在的检测结果。
- 评估为只读建议：新增 `assessment` 记录与 `job.ai_review`，不修改掩膜、像素、候选选择或审核状态；接口失败时本地修复与人工审核照常可用。
- 本地确定性复核规则与模型判断并行：面部重叠（OpenCV 级联）、大面积（>12%）、推测性补全、模型高风险/未知、依据不足、用户背景涉及人物/纹样/文字/宗教内容时强制要求人工复核。
- UI：结果页新增“AI 修复评估”按钮与背景输入框；逐区域展示本地检测、修复候选、DeepSeek 建议、风险与复核原因。
- 修复真实 API 兼容问题：`response_format: json_object` 与 `safety_check` 同时发送会返回 HTTP 400（该组合也存在于原 DeepSeek 缺损识别请求中）；现改为不发送 `response_format`，解析器接受围栏或夹杂文本中的 JSON。
- 真实验收：使用本机 `.env` 密钥成功请求 1 次（7.16 秒，`deepseek-flash`，区域 ID 对齐，密钥未泄漏），报告 `artifacts/repair-assessment-verification.json`。
- 测试 **174 项通过**（新增 Mock HTTP、结构校验、区域关联、本地规则、只读性、失败降级与 UI 面板回归）。


## 2026-11-16 — 阶段 4：本地自动修复与区域候选

- 新增 `region_repair`：已确认且掩膜有效的缺损项可生成修复候选；裂隙、颜料缺失、地仗缺失、起甲/剥落和手动区域优先本地 LaMa，缺权重时明确回退 Telea；褪色/污染保持色调匹配等分析路径，不强制补洞。
- 支持单区域与多区域：每个区域独立任务，整图 `result.png`、区域 `region.png`、整图 `difference.png`、区域 `region_difference.png` 分开保存。
- 任务与候选新增 `checks`：输出尺寸一致、掩膜外像素不变、原图 SHA-256 未变；候选记录工具参数、设备、权重 SHA-256 与耗时；失败任务不产生候选、不改写原图，并记录失败原因。
- 界面：缺损清单新增“生成修复候选（本地补全）”和逐项“生成修复候选”；结果页新增区域修复/区域差异预览与“重新生成此区域”。
- 修复 `addManualDefect` 返回值缺少 `mask_file_url` 的问题，手动区域可直接生成候选。
- 真实本地推理：下载并校验 `big-lama.pt`（196 MB，SHA-256 `344c77…`），在 900×626 真实壁画上以人工确认掩膜生成候选，8.46 秒、掩膜外差异 0、原图未变；报告 `artifacts/region-repair-verification.json`。
- 测试 **160 项通过**（新增区域策略、输出与检查、失败安全、跳过规则与 WebEngine 区域预览回归）。


## 2026-11-16 — 阶段 3：本地 Qwen3-VL + SAM 2.1 病害识别与掩膜候选

- 分析来源默认改为**本地 Qwen3-VL**；DeepSeek 保留为可选来源，不再静默替代本地识别。
- 新增 `vision_analysis.detect_local`：Qwen 候选框映射到原图像素；裂隙、起甲、颜料缺失再调用 SAM 2.1 分割；只有通过尺寸、非空、置信分、越框面积、越框范围的掩膜才标 `sam`，否则保留候选框并标 `needs_review`，要求人工修正。
- 褪色、污染归为表面观察，不默认补洞；修复建议改走局部色调匹配。
- 新增 `local_model_setup` 就绪检查：缺权重时提示所需文件与体积（Qwen 4.27 GB / SAM 156 MB），保留手动画选，不自动改走云端。
- 画布新增候选框 + 半透明掩膜 + 区域编号 + 病害类型 + 来源模型叠加；缺损清单支持确认、删除、修改掩膜和手动画选。
- 缺损项持久化为 proposal，新增 `editLocalDefect` / `listLocalDefects`；未确认或缺掩膜的条目拒绝启动修复任务（`Store.delete` 支持删除）。
- 修复 Qwen 2B 实际按百分比输出坐标导致的定位错误：提示词改为百分比，解析层自动识别 0–100 并换算到 0–1000。
- 真实推理跑通：RTX 4060 上 Qwen3-VL-2B + SAM 2.1 Hiera Tiny，6 个候选、4 个掩膜通过校验（越框比例 0.0–0.0161），整轮 187.83 秒；报告见 `artifacts/local-detection-verification.json`。
- 测试 **154 项通过**（新增本地检测、坐标换算、掩膜越框、重复框/整图框、缺模型降级、缺损编辑与持久化回归）。


## 2026-11-09 — 云端全图重绘（Grok Imagine）

- 移植可插拔云端图片编辑客户端 `mural_agent/imagine.py`（OpenAI 兼容 Images，默认 xAI `grok-imagine-image-2.0`）。
- 新增“完整原图 → 全图重绘”路径：不把原图涂空，直接把完整原图发给云端重绘；结果作为 `imagine` 工具候选，标记 `speculative`。
- 三分屏右栏优先使用云端全图重绘；未配置 Key 时回退本地 SD。设置弹窗新增云端服务地址、模型和 Key。
- 配置写入本机 `.env`，密钥不进入仓库；需要代理时可用 `HTTPS_PROXY`。
- 真实联调：xAI API 200，任务 `review`、候选 `tool=imagine`、掩膜外 0 差异；新增云端客户端/重绘测试。
- 当前测试 **76 项通过**；云端结果仍为假设性重构，不是历史复原。

## 2026-11-09 — 三分屏对照与全图 AI 生成

- 新增“三分屏对照”：原图 / AI 修补大面积 / 全 AI 生成三栏弹窗，可分别生成并反复查看。
- 新增 `generateFullImage`：自动创建全图掩膜，用原图边缘做 ControlNet 引导执行全图扩散生成；结果标记为 speculative。
- 新增 `generateLargeRepair`：把当前缺损候选掩膜取并集后走实验扩散，生成“大面积修补”栏。
- 全图/大面积生成均需 SD 权重，结果只能按假设归档；全图生成为全新画面，不是历史复原。
- 用真实 SD 权重跑通全图生成：任务 `review`、候选 `speculative`、掩膜外差异 0；三栏 UI 回归通过，测试 70 项通过。
- 三栏实测后修复：无 SD 时生成按钮禁用并提示安装，不再等点击后才报错；全图/大面积生成改用英文提示词，减少模型生成文字状伪影。
- 真实壁画实测后，全图生成的 ControlNet 条件由“边缘图”改为“原图 RGB”，保留颜色与大致明暗；输出仍偏软，只能作示意。

## 2026-11-09 — AI 尝试性补全（实验）

- 高级设置新增「允许 AI 尝试性补全大面积缺损」：自动切到 speculative + diffusion，允许大面积及 uncertain/unrestorable 掩膜；结果必须按假设归档，不能当定稿。
- 缺损清单新增「AI 生成式补全（实验）」：单项/全部可直接用扩散生成，无掩膜项自动补矩形掩膜；同样按假设归档。
- 增加 `allow_generative_attempt` 服务端校验：显式开关可越过风险分类，但保留掩膜核对、风险提示和假设归档要求；任务记录原风险与开关。
- 用真实 SD 权重端到端跑通一次缺陷生成：任务 `review`、候选 `speculative`、掩膜外差异 0；测试 70 项通过。
- 文档补充实验路径、限制与操作方式。

## 2026-11-09 — 缺损清单弹窗、手动添加、取消缺陷修复风险限制

- 缺损清单改为弹窗：AI 分析后自动弹出，图像上方与右侧均有“缺损清单”按钮可反复打开。
- 点击清单条目会在主图上用红框标出对应区域；没有可用掩膜的条目运行时自动按框生成矩形掩膜。
- 支持“手动添加缺损区域”：在图像上拖框即可加入清单，并立即参与修复。
- 缺损清单修复路径**取消风险分类与确认限制**：`uncertain` / `unrestorable` 也会尝试修复，原风险记录保存在任务 `defect_risk_class` 中；LaMa 未安装时自动回退 Telea。
- 保留手动画布/专业分步的门禁与导出校验，缺损清单是明确的“全部可试修”入口。
- 新增 `addManualDefect` 桥接与回归测试；测试 69 项通过。
- 实际点击验收中修复：弹窗宽度被通用 `.modal-card` 样式覆盖成 440px；任务结束后清单按钮禁用状态不刷新；批量运行中个别无掩膜项失败会中止整批（现在逐项跳过）；弹窗字号过小/“修补方案”标签换行。

## 2026-11-09 — 取消比赛观察模式

- 移除比赛观察包入口与观察预览区、三色观察清单、门禁横幅、演示样例按钮和导出观察标注/复核记录的快捷操作。
- 删除 `human_gate.py`、`scripts/run_demo_competition_risk.py` 以及比赛 demo 的 bridge 方法（`loadCompetitionDemo` / `applyGateReview` / `loadGateSample` / `exportObserveAnnotation` / `exportObserveReview`）和 `/api/competition-demo` 端点。
- 移除 `contest_demo` 字段与比赛导出专用门禁；保留专业分步的风险门禁、复核决定与导出校验。
- 复核决定（采纳 / 修改后采纳 / 驳回）保留在候选对照页，专业分步流程不受影响。
- 全量测试 67 项通过（较取消前少一项比赛专用测试）；文档同步更新。

## 2026-11-09 — 门禁、导出与验证基线修复（P0）

### 风险与门禁

- **统一风险枚举**：`fillable_lacuna / uncertain / unrestorable` 成为唯一规范词表，旧名 `repairable / uncertainty` 自动归一；`risk_regions`、`professional`、UI 与文档共用同一套常量。
- **服务端推导风险标志**：`derive_risk_flags(mask_record)` 从冻结掩膜记录推导 `has_uncertain / has_unrestorable`，创建工作区任务时不再信任前端传入的布尔值。
- **修复两条门禁绕过**（已用脚本复现并验证修复）：
  - professional_step 中 `region_class='uncertainty'` 且未传 `has_uncertain` 时曾可直接建任务；现在会被拒绝并要求专家强制放行 + 理由。
  - `workflow='standard'` 曾可整体绕过 enhanced/speculative/risk 门禁；现在两种工作流都在建任务时和 `engine.run` 执行前双重校验，`enhanced` 必须显式勾选掩膜核对 / 允许加强 / 风险知悉，`speculative` 必须显式确认风险。
  - `unrestorable` 区域任何情况下禁止像素修复。
- **比赛观察导出曾加门禁**（该模式已于同日取消，见上方条目）。

### 导出与保存

- `validate_selected_export()` 统一校验复核决定：驳回的候选不能导出 ZIP；假设性结果只能导出档案（`manifest.selected_as = hypothesis`），不能保存为普通修复 PNG。
- `engine.export` 与桌面端 `saveResult` 都使用该校验；manifest 新增 `selected_as / review` 字段。

### 合成与候选

- **修复羽化削弱薄掩膜**：`composite()` 对 <4 px 的薄掩膜直接写回完整预测；较厚掩膜只在 1–3 px 边界环带过渡、内部保持完整预测；掩膜外像素仍保证 0 差异。
- **候选数量不再静默减少**：优先使用不同算法，可用方法不足时用不同半径补足用户请求的 1–3 个候选；移除 engine 中“重复方法即提前结束”的分支。

### 测试与验证

- 更新 15 个过期测试，新增 H1/H2 门禁绕过回归、导出决定校验、薄/厚掩膜合成测试；当时 `pytest -q`：**68 passed**（取消比赛模式后为 67）。
- 新增 GitHub Actions：非 Web 测试 + `node --check` + `compileall`。
- 发货 WebEngine 外壳验收重跑通过：LaMa + CUDA、掩膜外差异 0、退出码 0。

### 文档与交付

- 重写 `VALIDATION.md` 为当前实测基线；修正使用说明的重复章节编号、过期测试数量、风险/加强表述与内部引用。
- 新增 MIT `LICENSE`（代码；模型权重与示例图各自许可）；新增本 CHANGELOG。
- 明确本机 `dist/Shibi-1.0` 为旧构建。

## 2026-10-31 — 早期迭代

历史记录保留在 git 提交中（`fe47956` baseline → `9bda39c` 稳定版 → `44d05c3` 专业流程 → `7fae2cf` 移除旧 UI）。
