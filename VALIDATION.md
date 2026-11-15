# 验证状态

更新时间：2026-11-16（阶段 3：本地 Qwen3-VL + SAM 2.1 识别与掩膜）

## 自动化测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

最近一次完整运行：**154 passed, 1 warning**（约 55 秒，exit 0；130 项非 WebEngine + 24 项 WebEngine）。覆盖：

- 业务/API/规划器/存储/备份；
- 风险分类与专业门禁（含两条门禁绕过回归：uncertain 掩膜、standard 工作流 enhanced）；
- 复核决定约束导出（驳回 / 假设性结果）；
- 缺损清单不按风险分类拦截（uncertain/unrestorable 仍可建任务）；
- 缺损清单弹窗、手动添加缺损、点击条目主图定位；
- AI 尝试性补全门禁绕过（unrestorable 掩膜 + speculative + 显式实验开关可建任务）；
- 真实扩散模型端到端一次尝试性缺陷修复：任务 `review`、候选标记 `speculative`、掩膜外 0 差异；
- 三分屏对照 UI（原图/大面积/全图三栏）与全图 AI 生成通道：真实 SD 权重任务 `review`、候选 `speculative`、掩膜外 0 差异；
- 云端全图重绘：完整原图（非空白掩膜）请求 Grok Imagine，任务 `review`、候选 `tool=imagine`、`speculative=true`、掩膜外 0 差异；
- 云端配置、密钥脱敏、异常与 Key 不落盘/不进入返回值的单元测试；
- 薄掩膜与厚掩膜合成（掩膜外 0 差异、薄掩膜完整预测）；
- 发货 Qt WebEngine 外壳（页面加载、QWebChannel、掩膜往返、建任务→复核→导出、AI 分析按钮、滑动对齐）；
- 本地病害识别（阶段 3）：Qwen 候选框从 0–1000 映射到原图像素（含百分比坐标自动换算）、只有通过尺寸/非空/置信分/越框面积/越框范围检查的 SAM 掩膜才标 `sam`、其余保留候选框并提示人工修正、褪色/污染不默认补洞、空结果、重复框、整图框、模型缺失；
- 缺本地权重时给出安装提示且保留手动流程、不静默调用 DeepSeek；
- 画布叠加候选框/半透明掩膜/编号/类型/来源，缺损清单确认、删除、修改掩膜、手动画选，`editLocalDefect` / `listLocalDefects` 的保存与恢复；
- 未确认或缺掩膜的缺损项拒绝启动修复任务。

## 本地 Qwen3-VL + SAM 2.1 真实推理（2026-11-16）

`scripts/verify_local_detection.py` 用固定权重在本机 GPU 跑真实流程，不是 Mock：

```powershell
.\.venv\Scripts\python.exe scripts/verify_local_detection.py
```

- 输入：`图片/01-cma-1963.252.jpg`，工作图 900×626；权重：Qwen3-VL-2B-Instruct（4.27 GB）+ SAM 2.1 Hiera Tiny（156 MB）；RTX 4060 Laptop，Qwen 在子进程推理后释放显存，再按需加载 SAM。
- 结果：**Qwen 真实推理成功**，返回 6 个候选（裂隙、颜料缺失、起甲三类）；**SAM 真实推理成功**，其中 4 个掩膜通过校验标为 `sam`，掩膜尺寸均为 900×626，越框像素比例 0.0–0.0161；2 个掩膜因范围远超候选框（疑似选中整条边线）被丢弃，只保留候选框并提示人工修正。
- 耗时：整轮（含 Qwen 进程启动、生成和随后 4 次 SAM 分割）187.83 秒。
- 报告：`artifacts/local-detection-verification.json`；叠加预览：`artifacts/local-detection-preview.png`；独立 SAM 探针（中心框）掩膜 900×626、覆盖率 21.6%、IoU 0.666，见 `artifacts/sam-probe-center.png`。
- 坐标：Qwen 实际按百分比回答（例如 `[1,1,6,6]`），解析层自动换算到 0–1000 再映射到原图像素（`[9,6,54,38]`），与工作图尺寸一致。
- 结论边界：只证明“权重能加载、两个模型能真实运行、掩膜坐标与工作图一致”，**不证明**病害识别准确率，也不证明这些候选框在真实壁画上定位正确。

## 静态检查

```powershell
node --check ui/app.js
.\.venv\Scripts\python.exe -m compileall -q mural_agent
git diff --check
```

## 发货壳验收

`scripts/verify_desktop.py` 在隔离离屏窗口加载 `ui/index.html`，通过 QWebChannel 完成真实 LaMa CUDA 修复并断言工具/设备/掩膜外差异：

- 结果：exit 0，`page_loaded=true`，job=`review`，`tool=lama`，`device=cuda`，`outside_max_difference=0`；
- 报告：`artifacts/desktop-web-verification.json`；档案：`artifacts/verified-web-restoration.zip`；预览：`artifacts/desktop-web-preview.png`。

## 本轮修复的安全断言（已用隔离脚本复现）

- 旧行为：professional_step + mask `region_class='uncertainty'` + 未传 `has_uncertain` → 可建任务；本轮修复后 → 拒绝（要求专家强制放行 + 理由）。
- 旧行为：`workflow='standard'` + `mode='enhanced'` + 无任何勾选 → 可建任务；本轮修复后 → 拒绝（须勾选掩膜核对/允许加强/风险知悉）。
- 旧行为：驳回或未标记的假设性结果仍可导出/保存；本轮修复后 → 拒绝。
- 合成：<4 px 薄掩膜按完整预测写回；厚掩膜只在边界环带羽化；掩膜外像素仍为 0 差异。

## 尚未验证 / 已知限制

- 真实壁画上的缺损识别准确率、风险分类准确率、修复质量与历史可信度没有专家标注数据验证。
- Qwen3-VL-2B 的定位能力有限：真实运行中 6 个候选框都落在画面左侧条带，存在漏检与框不准；当前只承诺“候选 + 人工核对”，不承诺定位精度。要提升需换更大模型、加 grounding 微调或改用 SAM 点选补框。
- SAM 掩膜被判定越框时只保留候选框（无掩膜），必须人工修正后才能运行修复；这是有意设计，不是遗漏。
- 坐标单位识别是启发式：模型返回的框全部 ≤105 时按百分比乘以 10。若某个 0–1000 坐标框恰好整体落在画面左上 10.5% 内，会被误判为百分比，需人工核对。
- 视觉 API 需要用户自己的密钥与服务；没有记录真实付费调用。
- 逐项/批量修复的完整桌面交互路径仍缺少端到端自动化覆盖（API、桥接层和主要 UI 状态已有回归）。
- 缺损清单按用户要求不再按风险分类拦截，`uncertain` / `unrestorable` 也会尝试修复；出现坏结果需要人工重跑，这是产品选择而非缺陷。
- 云端全图重绘依赖外部 API Key、网络与额度；本机测试通过本地代理 `127.0.0.1:12450` 访问 xAI，直接连接会超时。云端结果同样是推测，不是历史复原。
- 比赛观察模式（观察包、human_gate、contest 门禁、演示脚本）已从代码与界面移除；本地 `data/demo_competition`、`data/human-gate` 旧目录未被引用，也未自动删除。
- 本次未做交互式人工桌面点击验收（以离屏 WebEngine 外壳与 API/桥接回归代替）。
- 本机 `dist/Shibi-1.0` 是 2026-10-31 的旧构建，包含已删除的旧 UI 代码，不能作为当前交付物；需要重新构建或删除。
- 工程像素指标与合成缺损测试不能证明补全内容符合壁画历史原貌。应用输出仅供数字修复实验与人工复核。

## 可复现入口

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m "not web"
.\.venv\Scripts\python.exe scripts/verify_desktop.py
```

CI：`.github/workflows/tests.yml` 运行非 WebEngine 测试、`node --check` 与 `compileall`。
