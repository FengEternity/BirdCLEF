# BirdCLEF+ 2026 项目进度总览

<!--
  🤖 AI 阅读指引：
  这是 BirdCLEF 竞赛项目的入口文件。扫描此文件即可获取项目全貌。
-->

## 当前阶段

> **竞赛截止**: 2026-06-03（约 10 周）
> **当前排名**: 未提交
> **目标排名**: Top 5%（~Top 57 / 1134 队伍）

| 阶段 | 状态 |
|------|------|
| 竞赛调研与方案设计 | 🟢 已完成 |
| 数据 EDA | 🟢 已完成 |
| 基线模型搭建 | 🔴 未开始 |
| 模型调优 + 伪标签 | 🔴 未开始 |
| SED 模型 | 🔴 未开始 |
| 集成融合 | 🔴 未开始 |
| 推理优化 + 提交 | 🔴 未开始 |
| Working Note 论文 | 🔴 未开始 |

## 活跃文档

| 编号 | 类型 | 标题 | 状态 | 最后更新 |
|------|------|------|------|---------|
| ANAL-001 | analyze | [竞赛深度分析](analyze/ANAL-001-competition-analysis.md) | ✅ done | 2026-03-24 |
| ANAL-002 | analyze | [数据 EDA 分析](analyze/ANAL-002-data-eda.md) | ✅ done | 2026-03-25 |
| DES-001 | design | [模型架构设计方案](design/DES-001-model-architecture.md) | 🔧 implementing | 2026-03-25 |

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

- [ ] 注册 Kaggle 竞赛
- [x] 下载训练数据
- [x] 数据 EDA + 物种分布分析
- [x] 搭建本地开发环境（.venv + 依赖）
- [x] 编写 Stage 1 基线代码 + Kaggle Notebook
- [ ] 搭建训练环境（Kaggle Notebook GPU）
- [ ] 在 Kaggle 上运行 Stage 1 训练，获得 CV AUC
- [ ] 首次提交，验证 LB 分数

---

## 文档统计

| 类型 | 数量 |
|------|------|
| 分析文档 (analyze/) | 2 |
| 设计方案 (design/) | 1 |
| 实验记录 (experiment/) | 0 |
| 工作日志 (journal/) | 1 |
| 归档 (archive/) | 0 |
