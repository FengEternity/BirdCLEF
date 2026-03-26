# BirdCLEF+ 2026 项目进度总览

<!--
  🤖 AI 阅读指引：
  这是 BirdCLEF 竞赛项目的入口文件。扫描此文件即可获取项目全貌。
-->

## 当前阶段

> **竞赛截止**: 2026-06-03（约 10 周）
> **当前排名**: LB Score = 0.779（CV AUC=0.9776）
> **目标排名**: Top 5%（~Top 57 / 1134 队伍）

| 阶段 | 状态 |
|------|------|
| 竞赛调研与方案设计 | 🟢 已完成 |
| 数据 EDA | 🟢 已完成 |
| 基线模型搭建 | 🟢 已完成（V8, CV AUC=0.9776） |
| 模型调优 + 伪标签 | 🔴 未开始 |
| SED 模型 | 🔴 未开始 |
| 集成融合 | 🔴 未开始 |
| 推理优化 + 提交 | 🟡 进行中（推理 Notebook 通过，待 LB 提交） |
| Working Note 论文 | 🔴 未开始 |

## 活跃文档

| 编号 | 类型 | 标题 | 状态 | 最后更新 |
|------|------|------|------|---------|
| ANAL-001 | analyze | [竞赛深度分析](analyze/ANAL-001-competition-analysis.md) | ✅ done | 2026-03-24 |
| ANAL-002 | analyze | [数据 EDA 分析](analyze/ANAL-002-data-eda.md) | ✅ done | 2026-03-25 |
| DES-001 | design | [模型架构设计方案](design/DES-001-model-architecture.md) | 🔧 implementing | 2026-03-25 |
| JOUR-0325 | journal | [3月25日工作日志](journal/2026-03-25.md) | ✅ done | 2026-03-25 |

## 文档目录结构

```
docs/
├── PROGRESS.md              # 项目进度总览（核心入口）
├── analyze/                 # 分析文档（ANAL-XXX）
│   ├── ANAL-001 竞赛分析
│   ├── ANAL-002 数据 EDA
│   └── ...
├── design/                  # 设计方案（DES-XXX）
│   ├── DES-001 模型架构方案
│   ├── DES-002 数据增强策略
│   └── ...
├── experiment/              # 实验记录（EXP-XXX）
│   ├── EXP-001 基线模型
│   ├── EXP-002 频谱参数对比
│   └── ...
├── journal/                 # 每日工作日志（YYYY-MM-DD.md）
└── archive/                 # 归档文档
```

## 近期工作（最新在上）

### 2026-03-26（推理 Notebook 调试 + 首次提交准备）

- V8 训练 Notebook 在 Kaggle "Save & Run All" 成功完成：**CV AUC = 0.9776**
- 推理 Notebook 调试（V7→V10）：
  - V7: Cell 5 语法错误（shell 命令意外拼接到代码行）
  - V8: 模型路径未找到 — kernel_sources 挂载在 `/kaggle/input/notebooks/montyeternity/` 下，非顶层
  - V9: 增强路径搜索（`rglob` 递归搜索）+ 调试输出，模型成功加载
  - V9: `test_soundscapes/` 为空（公开 LB 环境），提交验证 assert 失败
  - V10: 空 test 时使用 `sample_submission.csv` 作为占位，验证逻辑容错
- 推理 Notebook V10 成功运行，生成 `submission.csv`（3 rows placeholder）
- **首次 LB 提交：Score = 0.779**（CV AUC=0.9776 vs LB 0.779 差距分析：单 fold、简化模型、域偏移）

### 2026-03-25（Kaggle 训练调试 + Git 仓库建立）

#### Kaggle 训练迭代（V1→V7b）

- **V1-V3**: 初始推送，遇到 Kaggle 环境兼容性问题
  - V3 被分配 P100 GPU（SM 6.0），PyTorch 2.10+cu128 不支持 → `CUDA error: no kernel image`
- **V4**: 移除耗时的音频时长扫描（8 min → 0），`load_audio` 加异常处理
  - 仍被分配 P100，同样 CUDA 报错
- **V5**: 添加 P100 自动检测+兼容 PyTorch 安装（`torch.cuda.get_device_capability()[0] < 7` 时安装 2.5.1+cu124）
- **V6**: 全面性能优化
  - **librosa → torchaudio**：音频加载和 mel 频谱计算
  - **insect_energy 从 mel 估算**：移除额外 STFT 开销
  - `NUM_WORKERS=4`, `persistent_workers=True`, `prefetch_factor=3`
  - `cudnn.benchmark=True`, `zero_grad(set_to_none=True)`
  - 但训练 12 epoch 后 **AUC 始终 ~0.50**（等同随机猜测）
- **关键 Bug 发现**: `torchaudio 2.10` 移除了 `torchaudio.info()` API
  - 导致 `load_audio_fast` 的 `except` 分支捕获 `AttributeError`
  - **所有音频返回零向量**，模型在训练纯静音
  - 这是 AUC=0.50 的根本原因
- **V7b（当前版本）**: 修复 `torchaudio.info()` → 直接 `torchaudio.load()` + 重采样后截取
  - 简化模型：移除 GeM / hour_embed / insect_energy，使用标准 `global_pool='avg'`
  - 简化损失：纯 BCEWithLogitsLoss，无 class weights / focal loss
  - 添加诊断打印（mel 统计、logits 分布、正负样本预测对比）
  - **Epoch 1 AUC = 0.8124**（191 个有效物种），确认训练正常
  - **最终结果：10 epoch, Best AUC = 0.9755**（epoch 9）
  - 正负样本分离度：342x（Pred@pos=0.615 vs Pred@neg=0.0018）
  - 总训练时间：~82 分钟（单卡 T4）

#### 性能调试经验

- **DataParallel 对小模型无效**：B0（4.8M 参数）GPU 计算极快，瓶颈在 CPU 数据加载
  - 单卡 4.28s/it → DataParallel 双卡 6.36s/it（更慢）
  - 通信开销 > 计算并行收益
- **数据加载是真正瓶颈**：librosa / torchaudio 的 CPU mel 计算、OGG 解码
- 当前速度：~604s/epoch（单卡 T4），10 epoch 预计 ~100 min

#### 仓库管理

- `git init` + 关联远程仓库 `git@github.com:FengEternity/BirdCLEF.git`（SSH）
- 初始提交推送到 `main` 分支（39 files, notebooks / src / docs / configs）

### 2026-03-25（数据 EDA + 深度分析 + 跨年对比 + 伪标签预估）

- 配置 Python 虚拟环境（`.venv`），安装全部依赖（numpy, pandas, librosa, torch, timm 等）
- 运行 EDA 脚本，分析竞赛数据集
- 关键发现：实际 234 种（非 650+），含 5 纲动物；28 个零样本物种；1,478 条已标注声景
- 深度分析：
  - 音频时长分布（中位 21.7s，全部 32kHz 单声道）
  - 频谱特征对比（鸟 4047Hz / 蛙 3502Hz / 虫 3836Hz±1963）
  - 物种共现矩阵（前 15 对全是蛙类，物种 65380 出现在 45% 片段）
  - 时间模式（鸟类黎明黄昏，蛙类夜间 18-23 时）
  - 零样本物种可用性评估（2 种充足 / 5 种勉强 / 20 种高重叠 / 1 种极少）
  - 地理域偏移量化（仅 2.4% 训练数据来自潘塔纳尔本地）
  - 昆虫 sonotype 相似度分析（多组余弦相似度=1.0，全段特征无法区分）
  - 录音质量与来源分析（XC 均分 4.01 vs iNat 全部 0 未评级）
- **BirdCLEF 2025 物种重叠分析**：41 种重叠（17.5%），全为广布种，外部数据价值有限
- **无标注声景伪标签预估**：10,592 文件 → ~127k 窗口，但夜间偏重、14/23 站点无标注、仅覆盖 32.1% 物种
- 编写并持续更新数据 EDA 分析文档（`ANAL-002`），含图表分析说明和策略指导
- 编写模型架构设计文档（`DES-001`）：三阶段训练、双分辨率频谱、SED 帧级模型、后处理策略
- 实现 Stage 1 基线代码：
  - `src/` 本地模块（config, utils, dataset, model, train, inference, postprocess）
  - Kaggle 训练 Notebook（`notebooks/train_baseline_b0.ipynb`）— 自包含，可直接在 Kaggle GPU 上运行
  - Kaggle 推理 Notebook（`notebooks/inference_submission.ipynb`）— 加载权重 + 后处理 + 生成 submission.csv
  - 配置文件（`configs/baseline_b0.yaml`）

### 2026-03-24（竞赛启动）

- 完成竞赛深度分析文档（`docs/analyze/ANAL-001-competition-analysis.md`）
- 分析排行榜分数分布、历年获胜方案
- 制定 5 阶段技术路线和 8 周时间规划
- 设计项目目录结构

## 待办事项

- [x] 注册 Kaggle 竞赛
- [x] 下载训练数据
- [x] 数据 EDA + 物种分布分析
- [x] 搭建本地开发环境（.venv + 依赖）
- [x] 编写 Stage 1 基线代码 + Kaggle Notebook
- [x] 搭建训练环境（Kaggle Notebook GPU）
- [x] 在 Kaggle 上运行 Stage 1 训练 → **CV AUC=0.9755**
- [x] Git 仓库建立并推送到 GitHub
- [x] Stage 1 训练完成，最终 CV AUC=0.9755
- [x] 首次提交，LB Score = 0.779
- [ ] 恢复完整模型（class weights + focal loss + GeM + insect_energy）
- [ ] Stage 2 模型（B3/B5 + SED）

---

## 文档统计

| 类型 | 数量 |
|------|------|
| 分析文档 (analyze/) | 2 |
| 设计方案 (design/) | 1 |
| 实验记录 (experiment/) | 0 |
| 工作日志 (journal/) | 2 |
| 归档 (archive/) | 0 |
