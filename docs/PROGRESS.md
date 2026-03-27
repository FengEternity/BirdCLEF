# BirdCLEF+ 2026 项目进度总览

<!--
  🤖 AI 阅读指引：
  这是 BirdCLEF 竞赛项目的入口文件。扫描此文件即可获取项目全貌。
-->

## 当前阶段

> **竞赛截止**: 2026-06-03（约 10 周）
> **当前排名**: LB Score = 0.830（V14 A1 声景混入, CV AUC=0.9721）
> **目标排名**: Top 5%（~Top 57 / 1134 队伍）

| 阶段 | 状态 |
|------|------|
| 竞赛调研与方案设计 | 🟢 已完成 |
| 数据 EDA | 🟢 已完成 |
| 基线模型搭建 | 🟢 已完成（V8, CV AUC=0.9776） |
| 模型调优 + 伪标签 | 🔴 未开始 |
| SED 模型 | 🔴 未开始 |
| 集成融合 | 🔴 未开始 |
| 推理优化 + 提交 | 🟢 已完成（LB 0.779） |
| LB 诊断分析 | 🟢 已完成（ANAL-003） |
| Working Note 论文 | 🔴 未开始 |

## 活跃文档

| 编号 | 类型 | 标题 | 状态 | 最后更新 |
|------|------|------|------|---------|
| ANAL-001 | analyze | [竞赛深度分析](analyze/ANAL-001-competition-analysis.md) | ✅ done | 2026-03-24 |
| ANAL-002 | analyze | [数据 EDA 分析](analyze/ANAL-002-data-eda.md) | ✅ done | 2026-03-25 |
| DES-001 | design | [模型架构设计方案](design/DES-001-model-architecture.md) | 🔧 implementing | 2026-03-25 |
| JOUR-0325 | journal | [3月25日工作日志](journal/2026-03-25.md) | ✅ done | 2026-03-25 |
| ANAL-003 | analyze | [LB 0.779 诊断分析](analyze/ANAL-003-lb-diagnostic.md) | ✅ done | 2026-03-26 |
| ANAL-004 | analyze | [LB 提分方案深度调研](analyze/ANAL-004-optimization-research.md) | ✅ done | 2026-03-26 |
| EXP-001 | experiment | [提交实验日志](experiment/EXP-001-submission-log.md) | 🟡 active | 2026-03-26 |
| DES-002 | design | [增量实验计划](design/DES-002-incremental-experiment-plan.md) | 🟡 active | 2026-03-26 |
| ANAL-005 | analyze | [A1 域适应效果深度分析](analyze/ANAL-005-a1-domain-adaptation-analysis.md) | ✅ done | 2026-03-26 |

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

### 2026-03-26（增量实验 A1 成功：LB 0.830，+0.041）

- **V13 训练失败**：Kaggle Internet 关闭导致 HuggingFace 权重下载失败
- **V14 修复 + 训练成功**：
  - 添加 `_create_backbone()` 多级 fallback + Internet 连通性检查
  - 训练 10 epoch，Best CV AUC = **0.9721**（Ep 09），对比 V8 的 0.9776（-0.005）
- **LB 提交结果：0.830（+0.041 vs 0.789 基线）**
  - 域适应验证成功：CV 略降但 LB 显著提升
  - CV-LB 差距从 0.189 缩小到 0.142
  - 仅通过 30% 声景弱标签混入（无架构改动）即获得 +0.041

### 2026-03-26（增量实验 A1：基线 + 声景混入，Train V13）

- **方法论转变**：V11/V12 多变量同时改动导致失败，转为逐步单变量消融实验
- 创建 DES-002 增量实验计划文档
- V13 = 基线（5s, BCE, avg pool, batch 64）+ **唯一变量：声景弱标签 30% 混入**
- 推理 notebook 回退到基线模型架构（avg pool + 单层 FC + Dropout 0.5）
- 训练已推送 Kaggle，等待结果

### 2026-03-26（Stage 2 训练 V12：NaN 修复 + 10s 训练对齐—失败）

- **修复 V11 训练 loss=NaN 问题**：
  - 根因：ASL 的 `eps=1e-8` 在 AMP float16 下溢出为 0，`log(0)=-inf` → NaN
  - 修复 1: ASL forward 添加 `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` 强制 float32 计算
  - 修复 2: `eps` 从 `1e-8` 增大到 `1e-4`
  - 修复 3: `pos_weight` 裁剪到 `[0.1, 10.0]` 范围，防止极端权重
- **训练-推理 10s 窗口对齐**（2024 冠军 Kefir 验证的方案）：
  - `CLIP_DURATION` 5s → 10s（mel 从 128×501 → 128×1001）
  - `BATCH_SIZE` 64 → 32（GPU 内存翻倍，对应减半）
  - `TMASK_W` 50 → 100（时间掩码宽度随 mel 宽度等比增大）
  - 训练数据偏移上限调整：25s → 20s，噪声偏移 55s → 50s
- Stage 1 深度分析写入 EXP-001 实验日志（三条学习点 L1/L2/L3）

### 2026-03-26（Stage 2 域适应训练 V11—失败，loss=NaN）

- 在对话中交付 BirdCLEF 2026 Stage 2 训练用独立 notebook cell：`BirdCLEFB0V2`（GeM + 两层 FC/Dropout）、`AsymmetricLoss`（ASL）、`BirdDatasetV2`（train_audio + train_soundscapes 按比例混入、5s 对齐、`end_time`/`start_time` 兼容）、声景背景噪声 SNR 混合、`compute_pos_weight_from_taxonomy`（纲别 × 频率）
- 与现有 `load_audio_fast` / `audio_to_melspec_fast` 及 `CFG` 路径约定对齐；声景标签支持 `;` 分隔的 eBird / iNaturalist ID 字符串
- **结果：Ep 01 loss=nan, AUC=0.4964（随机水平），训练失败**

### 2026-03-26（Stage 1 推理优化）

- 推理 notebook V16：10s 上下文窗口 + 5s 标准双路推理 + 时间平滑 + 改进后处理
- 移除有害的 sonotype 除法后处理，降低共现增强阈值至 0.05
- ANAL-004 增补通用 ML 前沿方案（UDA/DG/TTA/SFDA、零样本、校准），12 章完整版
- 已推送至 Kaggle，等待执行结果

### 2026-03-26（LB 诊断 + 深度调研）

- 创建 ANAL-003 诊断分析文档和 Kaggle 分析 Notebook，生成 7 张可视化图表
- 诊断结果：**域偏移是 LB 差距的主因**
  - CV macro AUC: 0.9444 → 声景 macro AUC: 0.6831（域偏移 0.26）
  - Amphibia 域偏移最严重（0.35），Mammalia CV 本身就偏低
  - 28 个零样本物种预测概率本质为 0
  - 声景正样本中 62.7% 预测概率 < 0.01（严重欠校准）
- 创建 ANAL-004 深度调研报告：调研 BirdCLEF 2022-2025 获胜方案 + 5 个问题的开源解决方案
  - 域偏移：伪标签声景训练（+0.03-0.05）、背景噪声增强、PANNs/BEATs 预训练
  - 零样本：声景弱标签挖掘、CLAP/NatureLM 零样本迁移
  - 校准：逐物种阈值、温度缩放、ASL loss
  - SED：Attention Pooling、多窗口推理、邻域平滑
  - 形成三阶段实施计划，预期 LB 0.779 → 0.92~0.95

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
- [x] ANAL-003 诊断分析完成：域偏移 0.26 为主因
- [ ] 恢复完整模型（class weights + focal loss + GeM + insect_energy）
- [ ] Stage 2 模型（B3/B5 + SED）

---

## 文档统计

| 类型 | 数量 |
|------|------|
| 分析文档 (analyze/) | 5 |
| 设计方案 (design/) | 1 |
| 实验记录 (experiment/) | 1 |
| 工作日志 (journal/) | 2 |
| 归档 (archive/) | 0 |
