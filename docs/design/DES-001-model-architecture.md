<!--
  📋 状态卡片
  id: DES-001
  title: BirdCLEF+ 2026 模型架构设计方案
  type: design
  status: implementing
  created: 2026-03-25
  updated: 2026-03-25
  author: fangyj0708
  tags: [BirdCLEF, 模型架构, EfficientNet, SED, 伪标签, 域适应]
  depends_on: [ANAL-001, ANAL-002]
-->

# DES-001: BirdCLEF+ 2026 模型架构设计方案

> **日期**: 2026-03-25
> **状态**: implementing (Stage 1 代码已完成)
> **目标**: 设计从基线到竞赛最终提交的完整模型方案，每条决策标注数据依据

## 摘要

基于 ANAL-002 数据 EDA 结论和 ANAL-001 历年方案分析，设计三阶段渐进式训练流程：EfficientNet-B0 基线 → 声景域适应微调 → 伪标签迭代扩展。核心创新点：跨纲均衡采样、双分辨率频谱输入、时间戳辅助特征、蛙类共现后处理。

**预期目标**：

| 阶段 | 模型 | 预期 Public LB |
|------|------|---------------|
| Stage 1 基线 | EfficientNet-B0 | ~0.82 |
| Stage 2 域适应 | + 声景微调 | ~0.86 |
| Stage 3 伪标签 | + 迭代伪标签 | ~0.88 |
| 集成 | 多模型融合 | ~0.92+ |

---

## 一、输入流水线

### 1.1 音频预处理

```python
AUDIO_CONFIG = {
    "sample_rate": 32000,      # 原始采样率，不重采样
    "clip_duration": 5.0,      # 5s 窗口（与评测粒度一致）
    "clip_samples": 160000,    # 32000 * 5
}
```

**裁剪策略**（依据 ANAL-002 §三 3.1：中位时长 21.7s，6% < 5s）：

| 音频时长 | 处理方式 | 理由 |
|---------|---------|------|
| >= 5s | 随机裁剪 5s（训练）/ 滑窗（推理） | 长录音多窗口增强 [ANAL-002 §三] |
| < 5s | **循环 padding** 到 5s | 非零填充，保留频谱连续性 [ANAL-002 §三] |
| 声景片段 | 精确 5s 对齐（start/end 时间戳） | 与标注对齐 |

### 1.2 梅尔频谱图

**双分辨率设计**（依据 ANAL-002 §三 3.2：各纲频率中心和带宽差异大）：

```python
SPEC_CONFIG_FINE = {
    "n_fft": 1024,
    "hop_length": 320,       # 10ms per frame → 500 frames / 5s
    "n_mels": 128,
    "fmin": 50,
    "fmax": 14000,
    "power": 2.0,
}

SPEC_CONFIG_COARSE = {
    "n_fft": 2048,
    "hop_length": 640,       # 20ms per frame → 250 frames / 5s
    "n_mels": 128,
    "fmin": 50,
    "fmax": 14000,
    "power": 2.0,
}
```

| 参数组 | 时间分辨率 | 频率分辨率 | 适合物种 |
|--------|-----------|-----------|---------|
| Fine (1024) | 10ms | ~31 Hz | 鸟类短促鸣叫、快速调频 |
| Coarse (2048) | 20ms | ~16 Hz | 蛙类低频持续呼叫、昆虫窄带信号 |

基线阶段用 Fine，Stage 2+ 引入 Coarse 或双通道拼接。

### 1.3 辅助特征

| 特征 | 编码方式 | 依据 |
|------|---------|------|
| **时间戳 (hour)** | 24-dim one-hot → MLP embedding | 鸟晨/蛙夜差异 [ANAL-002 §四 4.3] |
| **4-6 kHz 能量比** | scalar (0-1) | 昆虫 70.5% 能量 [ANAL-002 §三 3.3] |
| **地理位置** | lat/lon → 2-dim （仅推理时） | 域偏移 [ANAL-002 §五] |

---

## 二、模型架构

### 2.1 基线模型：BirdCLEF-B0

```
Input: Mel spectrogram (1, 128, 500)  # (channels, n_mels, time_frames)
  │
  ├─ EfficientNet-B0 (ImageNet pretrained, in_channels=1)
  │    └─ 提取 features: (1280, 4, 16)
  │
  ├─ GeM Pooling (p=3.0)  → (1280,)
  │
  ├─ Dropout (0.3)
  │
  ├─ FC (1280 → 512) + ReLU
  │
  ├─ Auxiliary Input: [hour_embed(32) + insect_energy(1)]
  │    └─ Concat → (512 + 33,)
  │
  └─ FC (545 → 234)  # 234 物种
      └─ Sigmoid (multi-label)
```

**设计理由**：
- **EfficientNet-B0**：ANAL-001 §三确认历年冠军方案均使用，参数少（5.3M）推理快
- **GeM Pooling**：优于 Global Average Pooling，对稀有物种更鲁棒 [ANAL-001 §三 Top 2%]
- **辅助输入在分类头注入**：不污染 backbone 特征提取，保留 ImageNet 预训练权重
- **Sigmoid 而非 Softmax**：多标签分类，89.4% 声景含 2+ 物种 [ANAL-002 §四 4.2]

### 2.2 SED 模型（Stage 3+）

```
Input: Mel spectrogram (1, 128, 500)
  │
  ├─ EfficientNet-B0 backbone → (1280, 4, 16)
  │
  ├─ Reshape → (1280, 64)  # 64 time frames
  │
  ├─ Attention Pooling (time dim)
  │    ├─ att = Sigmoid(FC(1280 → 234))     # (234, 64) attention weights
  │    ├─ framewise = Sigmoid(FC(1280 → 234)) # (234, 64) frame predictions
  │    └─ clipwise = sum(att * framewise, dim=time)  # (234,)
  │
  └─ Output: clipwise predictions (234,)
```

**设计理由**：
- **帧级预测**：25 个 sonotype 在 5s 全段无法区分 [ANAL-002 §六 6.3]，帧级可捕捉时间差异
- **Attention Pooling**：比 max/avg pooling 更能聚焦有声段，忽略静默段
- **与基线共享 backbone**：可从 Stage 1 权重初始化

### 2.3 多模型集成（最终提交）

| 模型 | backbone | 频谱参数 | 训练数据 |
|------|----------|---------|---------|
| M1 | EfficientNet-B0 | Fine (1024) | train_audio + soundscape |
| M2 | EfficientNet-B0 | Coarse (2048) | train_audio + soundscape |
| M3 | EfficientNetV2-S | Fine (1024) | train_audio + soundscape + pseudo |
| M4 | SED-B0 | Fine (1024) | train_audio + soundscape + pseudo |

**融合策略**：Quantile-Mix（α=0.5）

```python
# Quantile-Mix 融合
def quantile_mix(preds_list, alpha=0.5):
    mean_pred = np.mean(preds_list, axis=0)
    rank_pred = np.mean([rankdata(p) for p in preds_list], axis=0)
    return alpha * mean_pred + (1 - alpha) * rank_pred / rank_pred.max()
```

---

## 三、训练流程

### 3.1 Stage 1: 基线训练

**数据**: train_audio（35,549 条）

```python
TRAIN_CONFIG_S1 = {
    "epochs": 30,
    "batch_size": 64,
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingWarmRestarts",
    "T_0": 10,
    "warmup_epochs": 2,
}
```

**损失函数**（依据 ANAL-002 §二：长尾分布 + 多标签）：

```python
# BCE + Focal 混合
loss = 0.7 * BCEWithLogitsLoss(pos_weight=class_weights) \
     + 0.3 * FocalLoss(gamma=2.0, alpha=0.25)
```

**类别权重计算**（依据 ANAL-002 §二 2.1：Aves 97.9% vs Insecta 0.6%）：

```python
# 纲级均衡 + 物种频率逆权重
class_weights = []
class_multiplier = {"Aves": 1.0, "Amphibia": 20.0, "Insecta": 50.0,
                    "Mammalia": 30.0, "Reptilia": 100.0}
for species in taxonomy:
    base_weight = max_count / species_count[species]
    class_weight = base_weight * class_multiplier[species.class_name]
    class_weights.append(min(class_weight, 50.0))  # 上限截断
```

**数据增强**：

| 增强 | 概率 | 参数 | 依据 |
|------|------|------|------|
| Mixup | 0.3 | α=0.15 | ANAL-001 §三 历年有效 |
| TimeMask | 0.3 | max_width=50 frames | 标准增强 |
| FreqMask | 0.3 | max_width=20 bins | 标准增强 |
| GaussianNoise | 0.2 | std=0.005 | 模拟噪声 |
| RandomGain | 0.3 | gain_range=(-6, 6) dB | 音量变化 |

**Secondary labels 处理**（依据 ANAL-002 §二 2.3）：

```python
# 12.3% 录音含 secondary_labels → soft target
target = torch.zeros(234)
target[primary_idx] = 1.0
for sec_idx in secondary_indices:
    target[sec_idx] = 0.3  # 弱监督信号
```

### 3.2 Stage 2: 声景域适应

**数据**: train_soundscapes_labels（1,478 个 5s 片段，来自 66 个文件）

这是**最关键的阶段**——声景数据全部来自目标域 [ANAL-002 §五]。

```python
TRAIN_CONFIG_S2 = {
    "epochs": 15,
    "batch_size": 32,
    "lr": 1e-4,               # 比 S1 低 10x
    "freeze_layers": "stem + block[0:4]",  # 冻结前 4 个 block
    "data_mix_ratio": 0.3,    # 30% soundscape + 70% train_audio
}
```

**多标签训练**（依据 ANAL-002 §四 4.2：平均 4.2 物种/片段）：

```python
# 声景标注是 ";" 分隔的多标签
# "22961;23158;24321;517063;65380" → 5 个物种标签全设为 1.0
for label in labels_str.split(";"):
    target[label_to_idx[label.strip()]] = 1.0
```

### 3.3 Stage 3: 伪标签

**数据**: train_soundscapes 无标注（优先 S22 站点，~3,343 文件）[ANAL-002 §九 9.2]

```python
PSEUDO_LABEL_CONFIG = {
    "confidence_threshold": 0.7,    # 低于此不纳入 [ANAL-002 §九 9.5]
    "site_priority": ["S22", "S13", "S15", "S19", "S18"],  # 有标注参考的站点优先
    "iterations": 2,                # 最多 2 轮伪标签
    "window_stride": 2.5,           # 5s 窗口，2.5s 步长 → 50% 重叠
}
```

**伪标签生成流程**：

```
1. 用 Stage 2 模型推理 S22 站点全部 3,343 文件
2. 每文件滑窗切分 5s（2.5s 步长），每窗口得到 234 维概率
3. 过滤：仅保留 max(prob) >= 0.7 的窗口
4. 生成 soft label CSV
5. 混合 train_audio + soundscape_labels + pseudo_labels 重新训练
6. 第 2 轮：用新模型更新伪标签（置信度阈值提升到 0.8）
```

---

## 四、后处理

### 4.1 共现先验

依据 ANAL-002 §四 4.2：蛙类共现率极高，65380 是枢纽物种。

```python
# 从 train_soundscapes_labels 构建条件概率矩阵
# P(B|A) = count(A ∩ B) / count(A)
cond_prob = np.zeros((234, 234))
for segment_labels in all_labeled_segments:
    for a in segment_labels:
        for b in segment_labels:
            if a != b:
                cond_prob[a][b] += 1
cond_prob /= cond_prob.sum(axis=1, keepdims=True) + 1e-8

# 推理时：如果检测到物种 A (prob > 0.5)，提升共现物种 B
def apply_cooccurrence(preds, cond_prob, boost=0.15):
    detected = np.where(preds > 0.5)[0]
    for a in detected:
        for b in range(234):
            if cond_prob[a][b] > 0.3:
                preds[b] = max(preds[b], preds[a] * cond_prob[a][b] * boost)
    return preds
```

### 4.2 时间先验

依据 ANAL-002 §四 4.3：鸟类 04-07h，蛙类 18-23h。

```python
# 24h × 5纲 基线概率调整因子
TIME_PRIOR = {
    "Aves":     [0.2, 0.2, 0.3, 0.5, 0.9, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3, 0.3,
                 0.3, 0.3, 0.3, 0.4, 0.5, 0.8, 1.0, 0.6, 0.3, 0.2, 0.2, 0.2],
    "Amphibia": [0.8, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
                 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 0.9, 0.9],
    "Insecta":  [0.5, 0.5, 0.5, 0.8, 0.6, 0.4, 1.0, 1.0, 0.4, 0.3, 0.3, 0.3,
                 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5],
}

def apply_time_prior(preds, hour, taxonomy, prior=TIME_PRIOR, weight=0.1):
    for idx, species in enumerate(taxonomy):
        cls = species["class_name"]
        if cls in prior:
            factor = prior[cls][hour]
            preds[idx] = preds[idx] * (1 - weight) + preds[idx] * factor * weight
    return preds
```

### 4.3 零样本物种处理

依据 ANAL-002 §六：28 零样本物种分三档处理。

| 类型 | 物种 | 策略 |
|------|------|------|
| 🟢 蛙类 | 517063, 1491113 | 声景切片直接训练，正常推理 |
| 🟡 独特 sonotype | son07 | 规则检测：2-3 kHz 能量 > 阈值 |
| 🔴 不可区分 sonotype | 20 种 | 超类合并训练 + 推理时按先验拆分 |
| 🔴 极少 | son05 | 共现先验推断（不独立训练） |

**Sonotype 超类合并**（依据 ANAL-002 §六 6.3：cosine=1.0 的组）：

```python
# 训练时将不可区分的 sonotype 合并为超类
SONOTYPE_GROUPS = {
    "super_A": ["son08", "son11", "son20"],
    "super_B": ["son13", "son22", "son23"],
    "super_C": ["son15", "son16", "son25"],
    "super_D": ["son04", "son10"],
}
# 推理时：超类预测概率均匀拆分给成员
# pred[son08] = pred[son11] = pred[son20] = pred[super_A] / 3
```

---

## 五、推理流水线

### 5.1 Kaggle 提交推理流程

```
test_soundscapes/
  └── soundscape_XXXXXX.ogg (1 min each)
        │
        ├── 切分为 12 个 5s 窗口 (0-5s, 5-10s, ..., 55-60s)
        │
        ├── 每窗口生成 Mel spectrogram
        │
        ├── 模型推理 (4 模型集成)
        │     ├── M1: EfficientNet-B0 Fine
        │     ├── M2: EfficientNet-B0 Coarse
        │     ├── M3: EfficientNetV2-S Fine
        │     └── M4: SED-B0 Fine
        │
        ├── Quantile-Mix 融合
        │
        ├── 后处理
        │     ├── 时间先验 (从文件名解析时间)
        │     ├── 共现先验
        │     └── Sonotype 超类拆分
        │
        └── 输出: row_id, species_1, ..., species_234
```

### 5.2 推理时间预算

```
Kaggle 限制: ~2 hours for all test files
预估测试文件: ~700 files × 12 windows = 8,400 推理
每次推理: 4 模型 × ~15ms/模型 = 60ms
总推理时间: 8,400 × 60ms = ~504s ≈ 8.4 min
后处理 + I/O: ~2 min
总计: ~11 min (远低于限制)
```

---

## 六、实验计划

### 6.1 消融实验清单

| 实验 | 对比 | 衡量指标 |
|------|------|---------|
| A1 | BCE vs BCE+Focal vs Focal only | 5-fold CV AUC |
| A2 | 纲均衡采样 vs 无均衡 | Per-class AUC (重点看 Amphibia/Insecta) |
| A3 | Fine spec vs Coarse vs 双通道 | CV AUC |
| A4 | 时间戳特征 on/off | CV AUC |
| A5 | 4-6 kHz 能量特征 on/off | Insecta recall |
| A6 | Secondary labels soft target vs 忽略 | CV AUC |
| A7 | GeM Pooling vs GAP vs Attention | CV AUC |
| A8 | 共现后处理 on/off | LB AUC |
| A9 | 时间先验 on/off | LB AUC |
| A10 | 伪标签阈值 0.5 vs 0.7 vs 0.9 | CV AUC |

### 6.2 验证策略

**5-Fold StratifiedGroupKFold**：
- Stratify: 按物种频率分层
- Group: 按 `author` 字段分组（同一录音者的数据不能同时出现在训练和验证中，防止录音环境泄漏）
- 指标: macro AUC-ROC (与 LB 对齐)

---

## 七、目录结构

```
BirdCLEF/
├── configs/
│   └── baseline_b0.yaml         # Stage 1 超参配置（已实现）
├── src/
│   ├── __init__.py
│   ├── config.py                # 路径 + 超参（自动检测 Kaggle/本地）
│   ├── utils.py                 # 频谱图生成、指标、权重计算
│   ├── dataset.py               # Dataset + augmentation + Mixup
│   ├── model.py                 # BirdCLEF-B0, SED-B0, GeM, FocalLoss
│   ├── train.py                 # K-Fold 训练循环
│   ├── inference.py             # Kaggle 推理流水线
│   ├── postprocess.py           # 共现/时间先验/sonotype 拆分
│   └── pseudo_label.py          # 伪标签生成（待实现）
├── scripts/
│   ├── eda.py                   # 数据探索（已完成）
│   └── test_pipeline.py         # 端到端测试（已验证通过）
├── notebooks/
│   ├── train_baseline_b0.ipynb  # Kaggle 训练（自包含）
│   └── inference_submission.ipynb # Kaggle 推理提交（自包含）
├── data/
│   └── raw/                     # 竞赛原始数据
├── models/                      # 训练好的模型权重
└── docs/
```

---

## 八、Kaggle 使用步骤

### 8.1 Stage 1 训练

1. **上传训练 Notebook** → `notebooks/train_baseline_b0.ipynb`
2. **添加竞赛数据集**：在 Notebook 右侧面板搜索 `birdclef-2026`，添加为 Input
3. **开启 GPU**：Settings → Accelerator → GPU P100
4. **运行训练**：
   - 默认先跑 **Fold 0**（`CFG.TRAIN_FOLDS = [0]`），验证流程和 AUC 水平
   - 确认效果后，修改为 `CFG.TRAIN_FOLDS = [0, 1, 2, 3, 4]` 跑全 5 fold
5. **保存模型权重为 Dataset**：
   - 训练完成后，点击 "Save Version" → "Quick Save"
   - 将 `/kaggle/working/` 下的 `.pth` 文件和 `train_meta.json` 保存为 Kaggle Dataset
   - 命名为 `birdclef2026-weights`

### 8.2 推理提交

1. **上传推理 Notebook** → `notebooks/inference_submission.ipynb`
2. **添加数据源**：
   - 竞赛数据集 `birdclef-2026`
   - 模型权重 Dataset `birdclef2026-weights`（上一步保存的）
3. **开启 GPU**：Settings → Accelerator → GPU P100
4. **运行推理**：自动生成 `submission.csv`
5. **提交**：点击 "Submit" 查看 Public LB 分数

### 8.3 注意事项

- **GPU 时间**：Kaggle 每周 30 小时 GPU 配额，单 fold 训练约 15-30 分钟
- **Internet**：训练 Notebook 首次运行需要联网下载 EfficientNet-B0 预训练权重，之后可关闭
- **推理 Notebook 必须关闭 Internet**：Kaggle 竞赛提交要求 Internet Off
- **模型权重路径**：推理 Notebook 中 `MODEL_DIR` 默认为 `/kaggle/input/birdclef2026-weights`，如 Dataset 名称不同需修改

---

## 九、风险与缓解

| 风险 | 严重度 | 缓解策略 | 数据依据 |
|------|--------|---------|---------|
| 域偏移导致 LB 大幅低于 CV | **高** | Stage 2 声景微调是核心 | 仅 2.4% 本地数据 [§五] |
| 25 sonotype 提交全错 | **高** | 超类合并 + 共现先验 | cosine=1.0 [§六 6.3] |
| Public→Private LB 洗牌 | **高** | 关注 CV 而非 LB | ANAL-001 §二 |
| GPU 时间不足 | 中 | B0 优先，EfficientNetV2-S 可选 | — |
| 推理超时 | 低 | 预估 11 min << 2h | — |
| 蛙类过度后处理 | 中 | 共现先验权重需调参 | — |

---

## 参考

- [ANAL-001: 竞赛深度分析](../analyze/ANAL-001-competition-analysis.md)
- [ANAL-002: 数据 EDA 分析](../analyze/ANAL-002-data-eda.md)
- [LB 0.792 Starter Notebook](https://www.kaggle.com/code/baidalinadilzhan/lb-0-792-birdclef-2026-starter)
- [BirdCLEF 2025 Top 2% 方案](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df)
