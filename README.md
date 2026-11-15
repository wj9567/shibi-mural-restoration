# 拾壁：壁画数字修复工作室

拾壁是一款 Windows 桌面应用，用于检查壁画图像中的疑似缺损、比较数字修补方案并保存可追溯的修复记录。它使用 PySide6 和 Qt WebEngine 显示本地界面，工作区与修补结果保存在本机。使用云端分析时，应用在点击分析后发送当前图像的缩小副本。

## 功能

- 导入壁画图像，并用画笔或辅助分割工具选择待处理区域。
- 使用本地 Qwen3-VL（默认）或 DeepSeek 云端分析疑似缺损和可见文化线索。本地流程会对裂隙、起甲/剥落和颜料缺失再调用 SAM 2.1 生成掩膜候选；褪色、污染只保留候选框。分析结果进入同一个缺损清单。
- 点击清单条目可在主图定位；支持手动拖框添加缺损区域；每条缺损可选择建议方法或单独指定 LaMa、Telea、Navier–Stokes、局部色调匹配，再逐项或全部运行。
- 可选实验性 AI 尝试性补全：允许大面积缺损走扩散生成，或让缺损清单直接生成；结果强制按假设归档，不作为历史定稿。
- 三分屏对照：原图 / AI 大面积修补 / 全 AI 重绘，用于方案比较；后两栏均为推测性结果。
- 全图重绘可接 xAI Grok Imagine 等 OpenAI 兼容图片编辑服务（可选，需自备 API Key）；未配置时回退本地扩散。
- 对照原图、修复候选和差异图，复核后保存图像或导出 ZIP 档案。
- 缺损清单可对已确认掩膜生成本地修复候选：整图修复、区域修复和差异图，支持逐区域查看与重新生成；几何缺损优先 LaMa，褪色/污染不强制补洞。
- 修复候选可交给 DeepSeek 做**辅助评估**：按本地区域编号返回观察、修复评价、建议、依据、风险与人工复核原因；模型只给建议，不改掩膜、像素或审核状态，接口失败也不影响本地复核。
- 保存历史任务，支持查看、续跑、参考资料配准和工作空间备份。

## 安装

要求 Windows 10/11 x64 和 Python 3.10 或更高版本。克隆仓库后，在项目目录运行：

```powershell
.\start.cmd
```

也可以双击项目目录中的 `start.cmd`。首次启动时，脚本会创建 `.venv` 并安装桌面运行依赖。手动安装和启动：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start.cmd
```

基础图像处理可使用 CPU。LaMa、缺损识别、SAM 和扩散模型需要各自的权重；模型权重不随 Git 仓库分发。LaMa 权重可单独下载：

```powershell
.\.venv\Scripts\python.exe scripts/download_lama.py
```

本地视觉模型和可选分割、扩散模型的安装方式见 [使用说明](使用说明.md#可选模型安装) 与 [模型下载脚本](scripts/download_models.py)。NVIDIA GPU 用户可按本机驱动环境安装 CUDA 版 PyTorch，参考 `scripts/install_gpu.ps1` 和 [requirements-models.txt](requirements-models.txt)。

## 使用

1. 打开应用并载入一幅壁画图像。
2. 默认选择本地模型（Qwen3-VL）。点击“AI 分析缺损与文化线索”后，本地 Qwen3-VL 给出候选框，SAM 2.1 对几何缺损生成掩膜候选；未安装权重时界面会提示所需文件，手动画选不受影响。切换到 DeepSeek 云端需在“API 设置”的“DeepSeek 缺损分析”中填写密钥；只在点击分析时发送当前图像的缩小副本。
3. 打开缺损清单，点击条目在主图定位；候选框与半透明掩膜、区域编号、病害类型会同时显示。逐项确认、修改掩膜或删除后再运行；DeepSeek 的定位框是粗略建议，须手动圈选实际修补区域后才能运行对应条目。
4. 逐项运行，或运行全部。需要本地自动修复时，点缺损清单里的“生成修复候选（本地补全）”，或逐项点“生成修复候选”；结果页会显示整图、区域和差异图，并可按区域“重新生成此区域”。
5. 需要 AI 辅助评价时，在结果页点“AI 修复评估（DeepSeek）”；可按区域查看建议、风险和人工复核原因。该结果仅为建议，不改变本地图像与审核状态。
6. 需要时打开“三分屏对照”，分别生成 AI 大面积修补和全 AI 图。
6. 对比结果与差异，复核后保存图像或导出档案。

手动圈选及本地规则修补无需密钥。缺损识别和文化观察是待核对的模型建议，适用于数字修复实验。大面积语义缺失需要参考资料或专家判断。

完整操作说明、常见问题和边界说明见 [使用说明](使用说明.md)。工程验证信息见 [VALIDATION.md](VALIDATION.md)。

## API 规划（可选）

右侧分析来源只控制缺损识别。DeepSeek 视觉分析使用 `https://api.deepseek.com/chat/completions` 和 `deepseek-flash`，独立密钥保存在本机 `.env` 的 `DEEPSEEK_API_KEY` 中。界面保存后立即生效。修复规划由工作台上的规划设置控制。

应用可调用支持图像输入的视觉 API 生成修复规划。配置入口位于应用的“API 设置”；密钥保存在本地 `.env` 文件中，该文件不会提交到 Git。API 规划是可选项，本地规则规划不需要密钥。图像发送范围与接口格式说明见使用说明中的 [API 设置](使用说明.md#10-api-设置怎么填)。

## 开发

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_desktop.py
.\.venv\Scripts\python.exe scripts/verify_local_detection.py
.\.venv\Scripts\python.exe scripts/verify_region_repair.py
.\.venv\Scripts\python.exe scripts/verify_repair_assessment.py
```

测试包含本地业务逻辑、API 开发适配器和 Qt WebEngine 桌面桥接。CI（`.github/workflows/tests.yml`）在 Windows 上运行非 WebEngine 测试、`node --check` 与 `compileall`。完整依赖清单见 [requirements.txt](requirements.txt)、[requirements-models.txt](requirements-models.txt) 和 [requirements-dev.txt](requirements-dev.txt)。

变更记录见 [CHANGELOG.md](CHANGELOG.md)，验证基线见 [VALIDATION.md](VALIDATION.md)。

## 目录

```text
mural_agent/       修复引擎、桌面桥接、图像处理与工作空间
ui/                随桌面应用加载的 HTML、CSS、JavaScript 界面
图片/              带来源与许可说明的示例壁画图像
datasets/          示例数据清单与公开图像测试集
scripts/           启动构建、模型下载和验证工具
tests/             自动化回归测试
assets/            桌面应用图标
```

任务数据库、导入图像、修复结果和模型权重保存在本地 `data/` 与 `models/` 目录中，不包含在仓库内。

## 许可与图像来源

代码以 **MIT License** 发布，见 [`LICENSE`](LICENSE)。模型权重、示例图像与数据集**不在 MIT 范围内**：示例壁画图像各自的来源和许可见 [`图片/来源与许可.md`](图片/来源与许可.md) 及 `datasets/public-fresco-smoke/` 中的元数据；各模型权重许可见 `models/*/provenance.json` 与上游模型卡。使用前请逐项核对。
