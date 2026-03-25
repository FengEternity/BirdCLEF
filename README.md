# BirdCLEF+ 2026

> Acoustic Species Identification in the Pantanal, South America
> Kaggle Research Code Competition

## 项目结构

```
BirdCLEF/
├── README.md                    # 项目说明
├── docs/                        # 文档
│   ├── PROGRESS.md              # 项目进度总览（核心入口）
│   ├── analyze/                 # 分析文档（ANAL-XXX: 竞赛分析、数据 EDA 等）
│   ├── design/                  # 设计方案（DES-XXX: 模型架构、增强策略等）
│   ├── experiment/              # 实验记录（EXP-XXX: 每轮实验参数与结果）
│   ├── journal/                 # 每日工作日志
│   └── archive/                 # 归档文档
├── src/                         # 源代码
│   ├── data/                    # 数据加载、预处理、增强
│   ├── models/                  # 模型定义（CNN, SED, Ensemble）
│   ├── training/                # 训练循环、调度器、损失函数
│   ├── inference/               # 推理、后处理、提交生成
│   └── utils/                   # 工具函数
├── configs/                     # 训练配置文件（YAML）
├── notebooks/                   # 探索性分析 Jupyter Notebooks
├── experiments/                 # 实验记录（每次实验一个子目录）
├── scripts/                     # 一键运行脚本
│   ├── train.sh
│   ├── inference.sh
│   └── submit.sh
└── data/                        # 数据目录（不入版本控制）
    ├── raw/                     # 原始数据（Kaggle 下载）
    ├── processed/               # 预处理后数据（频谱图等）
    └── external/                # 外部数据（历年 BirdCLEF 等）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据（需先配置 Kaggle API）
kaggle competitions download -c birdclef-2026 -p data/raw/

# 3. 数据预处理
python scripts/preprocess.py

# 4. 训练基线模型
python -m src.training.train --config configs/baseline.yaml

# 5. 推理 + 生成提交文件
python -m src.inference.predict --config configs/baseline.yaml
```

## 技术路线

详见 [docs/analyze/ANAL-001-competition-analysis.md](docs/analyze/ANAL-001-competition-analysis.md)

| 阶段 | 内容 | 目标 LB |
|------|------|---------|
| Phase 1 | EfficientNet-B0 基线 | ~0.82 |
| Phase 2 | 伪标签 + 模型改进 | ~0.87 |
| Phase 3 | SED 模型 | ~0.87 |
| Phase 4 | 集成融合 | ~0.92+ |
| Phase 5 | 冲刺 + 论文 | Top 5% |

## 竞赛信息

- **竞赛页面**: https://www.kaggle.com/competitions/birdclef-2026
- **截止日期**: 2026-06-03
- **评测指标**: AUC-ROC（推测）
- **提交限制**: 每天 5 次
