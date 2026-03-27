<!--
 📋 状态卡片
 id: ANAL-004
 title: LB 提分方案深度调研
 type: analyze
 status: done
 created: 2026-03-26
 updated: 2026-03-26
 author: fangyj0708
 tags: [BirdCLEF, 调研, 域适应, 零样本, SED, 校准]
 depends_on: [ANAL-003]
-->

# ANAL-004: LB 提分方案深度调研报告

> **日期**: 2026-03-26
> **基于**: ANAL-003 诊断结论（域偏移 0.26 为主因，零样本 28 种失效，预测严重欠校准）
> **目标**: 为每个诊断问题搜集学术界 + 工业界（BirdCLEF 历届方案）的解决方案，形成可执行优化计划

---

## 一、BirdCLEF 历届获胜方案总结（2022-2025）

### 1.1 架构演进

| 年份 | 冠军 | 核心架构 | 关键技术 |
|------|------|---------|---------|
| 2022 | Kefir | EfficientNet + 多模型集成 | 外部 no-call 数据降假阳性，5-fold |
| 2023 | VSydorskyy | ConvNeXt-S + ConvNeXtV2-T + ECA-NFNet-L0 | 三架构 ONNX 集成，外部数据 |
| 2024 | Kefir | EfficientNet-B0 + RegNetY-008 | 伪标签声景，10s 上下文，sigmoid 推理 |
| 2025 | Nikita Babych | "Multi-Iterative Noisy Student" | 迭代式噪声学生半监督 |

**核心发现**：EfficientNet-B0 始终是基线和决赛方案的主力，但获胜靠的不是更大的模型，而是**训练策略**（伪标签、域适应、集成）。

### 1.2 跨年共性模式

**架构层面**：
- **Mel CNN** 始终主导：EfficientNet / RegNet / ConvNeXt / NFNet / EfficientViT
- SED 模型单模约 0.86，不如精调 CNN 集成
- **BirdNET / Google 鸟类分类器** 常用作教师模型或过滤器，而非直接提交

**域适应层面**：
- **声景伪标签** 是几乎所有 Top 方案的标配（2024 冠军直接贡献 +0.05 private LB）
- **教师-学生蒸馏**：大模型集成 GPU 生成伪标签 → 小模型 CPU 推理
- **数据质量过滤**：分位数过滤（保留 80% 最安静样本）、VAD 去人声

**增强层面**：
- **SpecAugment**（时间/频率掩码）+ **Mixup**（mel 图像级）
- **CutMix**（水平拼接）、背景噪声混合
- **Checkpoint Soup**（多 epoch 权重平均，2024 第 2 名）

**集成层面**：
- 多架构多种子多 mel 参数 → **late fusion**
- 2024 冠军使用 **min** 聚合（而非 mean），降低不确定正样本
- **邻域平滑**：时间窗口 t 的预测 += 0.5 × 相邻窗口预测

**推理层面**：
- **训练用 softmax，推理用 sigmoid**（2024 冠军，+0.044 private LB）
- **10s 上下文**（5s 窗口 + 两侧 2.5s padding）
- **多窗口推理** + 聚合

### 1.3 量化收益参考（2024 冠军 Kefir 的消融实验）

| 技术 | Private LB |
|------|-----------|
| 基线 (softmax) | 0.544 |
| + sigmoid 推理 | 0.588 (+0.044) |
| + 伪标签声景 | 0.640 (+0.052) |
| + 10s 输入 | 0.670 (+0.030) |
| + min 集成 (6× B0) | 0.689 (+0.019) |
| + 多架构 (3 Eff + 3 RegNet) | 0.690 (+0.001) |

### 1.4 2025 第 38 名公开数据（EfficientNet-B0 基线）

| 技术 | AUC |
|------|-----|
| B0 mel 基线 | 0.817 |
| + 伪标签 train_soundscapes | 0.835 (+0.018) |
| + 粗粒度 mel + 多层 GeM | 0.855 (+0.020) |
| + 3 CNN 平均 | 0.868 (+0.013) |
| + SED 模型混合 (Quantile-Mix) | 0.893 (+0.025) |
| + BirdCLEF 2021-2024 预训练 | 0.894 (+0.001) |

---

## 二、问题 1：域偏移（贡献 ~0.26 AUC）

### 2.1 问题本质

训练数据 97.6% 为干净焦点录音，测试数据为野外声景。模型学到的特征在噪声环境下失效，表现为"不敢预测"（正样本中位预测 0.002）。

### 2.2 解决方案矩阵

#### 方案 A：声景伪标签训练（P0，预计 +0.03~0.05）

**原理**：用当前模型在 `train_soundscapes` 上生成伪标签，将声景数据混入训练集。

**实施步骤**：
1. 用当前 best_fold0.pth 对 train_soundscapes 的每个 5s 段推理
2. 对高置信度预测（> 阈值）生成硬/软伪标签
3. 训练时按概率（25-45%）混入伪标签样本
4. 迭代 2-3 轮（每轮用新模型更新伪标签）

**关键参数**（来自 2024 第 2 名）：
- 混入概率：25-45%
- 伪标签与真实标签合并方式：`max(one_hot, pseudo_vector)`
- 伪标签权重系数：0.05（过高会伤害，来自 Kefir）

**开源参考**：
- [jfpuget/birdclef-2024](https://github.com/jfpuget/birdclef-2024)（第 3 名，两阶段蒸馏）
- [TheoViel/kaggle_birdclef2024](https://github.com/TheoViel/kaggle_birdclef2024)

#### 方案 B：背景噪声数据增强（P0，预计 +0.02~0.03）

**原理**：将干净录音与各类环境噪声混合，缩小训练/测试域差距。

**噪声数据源**：

| 数据集 | 大小 | 内容 | 用途 |
|--------|------|------|------|
| ESC-50 | 2000 clips | 50 类环境声 | 小规模通用噪声 |
| FSD50K | 51k clips | Freesound 多事件 | 大规模非鸟声干扰 |
| MUSAN | 1000h+ | 音乐/语音/噪声 | 通用噪声混合 |
| train_soundscapes 切片 | 66 files | 目标域背景 | **最匹配**的噪声源 |

**增强库**：
- [audiomentations](https://github.com/iver56/audiomentations)：`AddBackgroundNoise`（SNR 控制）、`AddGaussianNoise`、`TimeStretch`
- [torch-audiomentations](https://github.com/iver56/torch-audiomentations)：GPU 友好的 `nn.Module` 变换

**实施**：
```python
from audiomentations import AddBackgroundNoise, Compose
augment = Compose([
    AddBackgroundNoise(sounds_path="noise/", min_snr_in_db=3, max_snr_in_db=30, p=0.5),
])
```

#### 方案 C：PCEN 前端归一化（P1，预计 +0.01~0.02）

**原理**：Per-Channel Energy Normalization 自动补偿信道间能量差异，对静态噪声有天然鲁棒性。

**论文**：[Springer - 声学域不匹配鸟类检测](https://link.springer.com/article/10.1007/s10772-022-09957-w)

**实施**：替换当前 `AmplitudeToDB` 为 `torchaudio.transforms.PCEN` 或手动实现。

#### 方案 D：预训练模型迁移（P1，预计 +0.02~0.04）

**候选预训练模型**：

| 模型 | 类型 | 预训练数据 | 特点 | 代码 |
|------|------|-----------|------|------|
| PANNs (CNN14) | CNN | AudioSet | 快速、SED 友好 | [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| AST | ViT | AudioSet | 强分类 | [YuanGongND/ast](https://github.com/YuanGongND/ast) |
| BEATs | Tokenizer+Transformer | AudioSet | 广谱最强 | [microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats) |
| HTS-AT | 层级 Transformer | AudioSet | 分类+定位 | [RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer) |
| OpenBEATs | Transformer | 生物声学 | 生物声学特化 | [espnet/openbeats](https://huggingface.co/collections/espnet/openbeats) |

**建议**：优先尝试 PANNs（简单集成）和 BEATs（性能上限），用声景验证集而非 CV 选模型。

#### 方案 E：多年数据预训练（P2，预计 +0.01~0.02）

**原理**：BirdCLEF 2021-2024 的训练数据与 2026 有物种重叠，可作为预训练数据。

**来自 2025 第 38 名数据**：单模型 0.855 → 预训练后 0.868（+0.013）。

---

## 三、问题 2：零样本物种（28 种完全失效）

### 3.1 问题本质

25 个 insecta sonotypes + 2 个两栖类 + 1 个爬行类在训练集中无任何样本。模型对它们的预测概率为 1e-7~8e-5，本质上是 0。

### 3.2 解决方案矩阵

#### 方案 F：声景弱标签挖掘（P0，预计最高优先级）

**原理**：`train_soundscapes_labels.csv` 包含这 28 个物种的出现标注，可直接作为弱监督训练数据。

**实施**：
1. 从声景音频中切出对应 5s 片段
2. 作为额外训练样本加入训练集
3. 标签权重可适当降低（如 0.5）以反映弱标签噪声

**关键**：这是最直接、最低成本的方案，因为数据和标签已经有了。

#### 方案 G：CLAP / NatureLM 零样本迁移（P1）

**原理**：利用音频-文本对齐模型，通过物种名称/描述进行零样本分类。

**候选模型**：

| 模型 | 特点 | 代码 |
|------|------|------|
| CLAP | 音频-文本对比学习，通用零样本 | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) |
| ReCLAP | 描述性提示优于类别名 | [arXiv:2409.09213](https://arxiv.org/abs/2409.09213) |
| NatureLM-audio | 生物声学特化，跨物种泛化 | [earthspecies/NatureLM-audio](https://github.com/earthspecies/NatureLM-audio) |
| AudioCLIP | CLIP 扩展到音频 | [AndreyGuzhov/AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) |

**论文**：
- [arXiv:2206.04769](https://arxiv.org/abs/2206.04769) — CLAP 原论文
- [arXiv:2309.08398](https://arxiv.org/abs/2309.08398) — 音频+元信息零样本鸟类分类（直接相关）
- [Scientific Reports 2025](https://www.nature.com/articles/s41598-025-89153-3) — 多模态 LM 在生物声学零样本中的案例

**限制**：CLAP 在粗粒度分类上有效，但物种级细粒度识别效果有限，且对 insecta sonotype 可能无法区分。

#### 方案 H：Sonotype 处理策略（P1）

**背景**：25 个 sonotype（47158son01-25）都属于单一昆虫物种（iNat ID 47158）。

| 策略 | 适用场景 | 风险 |
|------|---------|------|
| 保持 25 个独立类 | 评测按 sonotype 计分 | 数据饥饿 |
| 合并为单一物种标签 | 评测按物种计分 | 丢失 sonotype 粒度 |
| 层级模型（物种→sonotype） | 两级预测 | 需要部分标注 |
| 共享主干 + 25 个 sonotype 头 | 特征共享 | 实现复杂 |

**建议**：检查 BirdCLEF 2026 评测规则，如果提交矩阵要求 25 个独立列，则不能合并输出。可在模型内部共享 backbone 特征，用 taxonomy 节点约束。

#### 方案 I：Few-shot 和原型网络（P2）

**适用条件**：如果竞赛规则允许外部数据（Xeno-Canto、iNaturalist）。

**方法**：
- 原型网络：[arXiv:2012.01573](https://arxiv.org/abs/2012.01573) — 少样本音频分类
- DCASE 少样本生物声学任务：[arXiv:2306.09223](https://arxiv.org/abs/2306.09223)
- 自监督预训练 + 少样本微调：[arXiv:2312.15824](https://arxiv.org/abs/2312.15824)

#### 方案 J：BirdNET 嵌入最近邻（P2）

**原理**：利用 BirdNET 的预训练嵌入空间，将零样本物种映射到最近的已知物种。

**论文**：[Frontiers in Ecology and Evolution 2024](https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2024.1409407/full) — BirdNET 发现新声类

---

## 四、问题 3：预测欠校准（正样本中位预测 0.002）

### 4.1 问题本质

模型在声景上输出的概率全局被压制，62.7% 的正样本预测 < 0.01。这不是"分错"而是"不敢分"。

### 4.2 解决方案矩阵

#### 方案 K：逐物种阈值优化（P0，快速部署）

**原理**：不用全局 0.5 阈值，而是在声景验证集上为每个物种优化独立阈值。

**实施**：
```python
from sklearn.metrics import f1_score
for species_idx in range(234):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.001, 0.5, 0.001):
        f1 = f1_score(targets[:, species_idx], preds[:, species_idx] > t)
        if f1 > best_f1: best_t, best_f1 = t, f1
    thresholds[species_idx] = best_t
```

**参考**：BirdNET 官方指南强调"分数不是校准概率，逐物种阈值常优于全局阈值"。

#### 方案 L：温度缩放（P0，后处理）

**原理**：在 sigmoid 之前对 logits 进行温度缩放 `p = σ(z / T)`，如果概率太低则降低 T（< 1 使 logits 更极端）。

**经典论文**：Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)

**多标签版本**：逐类温度 + 偏置 `p' = σ(a_c * z_c + b_c)`（Platt scaling）

#### 方案 M：Focal Loss 替换 BCE（P1，训练时）

**原理**：Focal Loss 降低易分类样本的权重，聚焦于难样本。

**注意**：我们当前的问题是**欠置信**（正样本预测太低），而 Focal Loss 的 γ 可能进一步压制"容易"的正样本。需要谨慎调 γ。

**更好的替代 — Asymmetric Loss (ASL)**：
- 对正样本和负样本使用不同的 γ
- 正样本 γ+ 较小（少压制），负样本 γ- 较大（多压制假阳性）
- 论文：[arXiv:2009.14119](https://arxiv.org/abs/2009.14119)

**Distribution Balanced Loss (DBL)**：
- 专为长尾多标签设计
- 论文：[arXiv:2007.09654](https://arxiv.org/abs/2007.09654) (ECCV 2020)
- 代码：[wutong16/DistributionBalancedLoss](https://github.com/wutong16/DistributionBalancedLoss)

#### 方案 N：Sigmoid 推理修正（P0，零成本）

**来自 2024 冠军**：训练用 softmax（CE loss），推理用 sigmoid。收益 +0.044 private LB。

**原理**：Softmax 训练鼓励物种间竞争（适合焦点录音，单物种为主），但声景中多物种同时出现时 softmax 会互相压制。Sigmoid 推理解除这种竞争。

---

## 五、问题 4：模型架构不足

### 5.1 当前简化点

| 缺失组件 | 预计收益 | 实现难度 |
|---------|---------|---------|
| GeM Pooling | +0.01 | 低 |
| Focal Loss / ASL | +0.01~0.02 | 低 |
| Class Weights | +0.005~0.01 | 低 |
| Hour Embedding | +0.005 | 低 |
| Insect Energy 特征 | +0.005 | 中 |
| 5-fold 集成 | +0.01~0.02 | 中（训练时间×5） |
| 多 Dropout 层 | +0.005 | 低 |

### 5.2 恢复计划

参考 DES-001 完整设计方案，逐步恢复：

1. **Phase 1**（快速）：GeM + 5× Dropout + ASL → 重新训练 Fold 0
2. **Phase 2**（中等）：Hour Embedding + Class Weights + 5-fold
3. **Phase 3**（完整）：Insect Energy + Secondary Labels Soft Target

---

## 六、问题 5：SED 架构升级

### 6.1 Clip-level vs Frame-level SED

| 维度 | Clip-level（当前） | Frame-level SED |
|------|-------------------|----------------|
| 输出 | 每段一个向量 | 每帧一个向量 |
| 重叠物种 | 全局池化混合 | 帧级独立，更好处理 |
| 弱标签 | 自然适配 | 需要 MIL/Attention |
| 短暂叫声 | 被平均稀释 | 局部高分保留 |

### 6.2 推荐 SED 架构

**Attention-based Pooling（PSLA 风格）**：
- 论文：[arXiv:2102.01243](https://arxiv.org/abs/2102.01243) — Gong et al.
- 代码：[YuanGongND/psla](https://github.com/YuanGongND/psla)
- CNN backbone 输出 frame-level features → Attention 加权聚合 → clip-level 预测

**CRNN（CNN + BiGRU/Transformer）**：
- DCASE 弱标签 SED 标准架构
- 适合时序建模，捕捉叫声的时间结构

**PANNs SED 模式**：
- PANNs 内置 decision-level attention 变体
- 可直接用 `Cnn14_DecisionLevelAtt` 预训练权重
- 代码：[qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)

### 6.3 多窗口推理优化

当前推理：每个 5s 窗口独立。

**改进方案**：
- **滑动窗口 + 重叠**：步长 2s，窗口 5s → 每个时间点有 2-3 次预测
- **聚合策略**：max（偏好稀疏叫声）/ mean（偏好持续声）/ attention 加权
- **上下文扩展**：使用 10s 输入（中间 5s 预测 + 两侧 2.5s 上下文），收益 +0.03（来自 2024 冠军）
- **邻域平滑**：`pred[t] += 0.5 * (pred[t-1] + pred[t+1])` / 2

---

## 七、综合优化路线图

### 7.1 优先级排序（ROI 最大化）

| 优先级 | 方案 | 预计收益 | 实现复杂度 | 依赖 |
|--------|------|---------|-----------|------|
| **P0-1** | 方案 N: Sigmoid 推理 | +0.02~0.04 | 极低（改 1 行） | 无 |
| **P0-2** | 方案 K: 逐物种阈值 | +0.01~0.02 | 低 | 声景验证集 |
| **P0-3** | 方案 F: 声景弱标签训练 | +0.03~0.05 | 中 | 训练流程修改 |
| **P0-4** | 方案 A: 伪标签迭代 | +0.02~0.04 | 中 | 模型推理 |
| **P0-5** | 方案 B: 背景噪声增强 | +0.02~0.03 | 低 | 噪声数据集 |
| **P1-1** | Phase 1 模型恢复 | +0.02~0.03 | 低 | 无 |
| **P1-2** | 方案 L: 温度缩放 | +0.01 | 极低 | 声景验证集 |
| **P1-3** | 多窗口 + 邻域平滑 | +0.01~0.02 | 低 | 推理修改 |
| **P2-1** | 5-fold 集成 | +0.01~0.02 | 中 | 5x 训练时间 |
| **P2-2** | SED 架构 (PANNs) | +0.02~0.04 | 高 | 新模型 |
| **P2-3** | 方案 G: CLAP 零样本 | +0.01~0.02 | 高 | 新依赖 |

### 7.2 三阶段实施计划

#### Stage 1: 快速提分（1-2 天，预计 +0.05~0.10）

不需要重新训练，只修改推理和后处理：

1. 推理时使用 sigmoid（而非当前已用 sigmoid 的话则跳过）
2. 在 train_soundscapes 上做逐物种阈值优化
3. 实现 10s 上下文窗口推理
4. 实现邻域时间平滑
5. 温度缩放校准

#### Stage 2: 域适应训练（3-5 天，预计 +0.05~0.08）

1. 恢复完整模型（GeM + ASL + Dropout 堆叠）
2. 添加声景弱标签数据（train_soundscapes 切片）到训练集
3. 添加背景噪声数据增强（audiomentations）
4. 伪标签迭代训练（2-3 轮）
5. 5-fold 训练 + 集成

#### Stage 3: 高级方案（1-2 周，预计 +0.03~0.05）

1. PANNs / BEATs 预训练 backbone 对比
2. SED 架构实现（Attention Pooling）
3. CLAP / NatureLM 零样本辅助
4. 多年 BirdCLEF 数据预训练
5. 多架构集成（CNN + SED + Transformer）

### 7.3 预期 LB 分数轨迹

| 阶段 | 措施 | 预期 LB |
|------|------|---------|
| 当前 | 基线 B0, 单 fold, 无后处理 | 0.779 |
| Stage 1 完成 | 推理优化 + 阈值 + 温度 | 0.82~0.85 |
| Stage 2 完成 | 域适应 + 完整模型 + 集成 | 0.87~0.92 |
| Stage 3 完成 | SED + 预训练 + 多架构 | 0.92~0.95 |
| Top 5% 目标 | — | ≥ 0.93 |

---

## 八、关键开源资源汇总

### 8.1 代码仓库

| 资源 | 链接 | 用途 |
|------|------|------|
| BirdCLEF 2024 3rd | [jfpuget/birdclef-2024](https://github.com/jfpuget/birdclef-2024) | 两阶段蒸馏 |
| BirdCLEF 2024 3rd | [TheoViel/kaggle_birdclef2024](https://github.com/TheoViel/kaggle_birdclef2024) | 完整训练流程 |
| BirdCLEF 2023 1st | [VSydorskyy/BirdCLEF_2023_1st_place](https://github.com/VSydorskyy/BirdCLEF_2023_1st_place) | ONNX 集成 |
| PANNs | [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) | SED 预训练 |
| BEATs | [microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats) | 预训练 backbone |
| CLAP | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) | 零样本分类 |
| NatureLM-audio | [earthspecies/NatureLM-audio](https://github.com/earthspecies/NatureLM-audio) | 生物声学零样本 |
| audiomentations | [iver56/audiomentations](https://github.com/iver56/audiomentations) | 音频增强 |
| DBL | [wutong16/DistributionBalancedLoss](https://github.com/wutong16/DistributionBalancedLoss) | 长尾多标签 loss |
| PSLA | [YuanGongND/psla](https://github.com/YuanGongND/psla) | Attention Pooling |

### 8.2 关键论文

| 论文 | 主题 | 链接 |
|------|------|------|
| PANNs (Kong et al.) | 预训练音频网络 | [arXiv:1912.10211](https://arxiv.org/abs/1912.10211) |
| BEATs | 音频 Tokenizer | [microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats) |
| CLAP | 音频-文本对比学习 | [arXiv:2206.04769](https://arxiv.org/abs/2206.04769) |
| ASL | 非对称多标签 Loss | [arXiv:2009.14119](https://arxiv.org/abs/2009.14119) |
| PSLA | 音频标注 Aggregation | [arXiv:2102.01243](https://arxiv.org/abs/2102.01243) |
| 零样本鸟类分类 | 音频+元信息 | [arXiv:2309.08398](https://arxiv.org/abs/2309.08398) |
| NatureLM-audio | 生物声学基础模型 | [earthspecies/NatureLM-audio](https://github.com/earthspecies/NatureLM-audio) |
| 模型合并零样本泛化 | 基础模型合并 | [arXiv:2511.05171](https://arxiv.org/abs/2511.05171) |
| Focal Loss 与校准 | 校准理论 | [arXiv:2408.11598](https://arxiv.org/abs/2408.11598) |
| BirdSet 基准 | 生物声学评测 | [arXiv:2403.10380](https://arxiv.org/abs/2403.10380) |

---

---

## 九、通用前沿方案：域偏移

> 本节超越 BirdCLEF 竞赛经验，调研 ML 学术界对域偏移问题的通用前沿方案。

### 9.1 无监督域适应（UDA）

#### 9.1.1 基于分布对齐

| 方法 | 核心原理 | 论文 | 代码 | 适用性 | 复杂度 |
|------|----------|------|------|--------|--------|
| **DAN / MK-MMD** | RKHS 中对齐源/目标特征分布矩，界定目标风险上界 | [ICML 2015](https://proceedings.mlr.press/v37/long15.html) | 社区实现 | 中 | 中 |
| **Deep CORAL** | 对齐深层激活协方差（二阶统计），比单矩匹配更细 | [ECCV 2016, arXiv:1607.01719](https://arxiv.org/abs/1607.01719) | [VisionLearningGroup/CORAL](https://github.com/VisionLearningGroup/CORAL) | 中 | 低-中 |
| **DANN** | 特征提取器 + 梯度反转对抗域分类器 | [JMLR 2016](https://jmlr.org/papers/volume17/15-239/15-239.html) | 广泛 | 中-高 | 中 |
| **CDAN** | DANN + 条件分布/多线性映射，强化类别结构对齐 | 见 UDA 综述 | 常见 | 中 | 中-高 |

**BirdCLEF 适用分析**：对齐方法需要目标域无标签数据（train_soundscapes 可充当）。风险在于野外噪声可能让对齐器把"噪声特征"而非"鸟声特征"对齐。建议与伪标签筛选结合。

#### 9.1.2 基于伪标签 / 自训练

| 方法 | 核心原理 | 论文 | 代码 | 适用性 | 复杂度 |
|------|----------|------|------|--------|--------|
| **Mean Teacher** | EMA 教师 + 学生一致性正则，降低伪标签方差 | [arXiv:1703.01780](https://arxiv.org/abs/1703.01780) | [CuriousAI/mean-teachers](https://github.com/CuriousAI/mean-teachers) | **高** | 中 |
| **FixMatch** | 高置信伪标签 + 强弱增广一致性 | [NeurIPS 2020, arXiv:2001.07685](https://arxiv.org/abs/2001.07685) | [google-research/fixmatch](https://github.com/google-research/fixmatch) | **高** | 中 |
| **Noisy Student** | 大模型自训练 + dropout/噪声，迭代伪标签 | [CVPR 2020, arXiv:1911.04252](https://arxiv.org/abs/1911.04252) | 参考实现 | 中 | 高 |

**BirdCLEF 关联**：2025 冠军 Nikita Babych 正是使用 "Multi-Iterative Noisy Student"。这类方法与竞赛实践高度契合。

### 9.2 域泛化（DG，无目标域数据）

| 方法 | 核心原理 | 论文 | 代码 | 适用性 | 复杂度 |
|------|----------|------|------|--------|--------|
| **GroupDRO** | 最小化最坏组风险，对齐子群偏移 | [ICLR 2020, arXiv:1911.08731](https://arxiv.org/abs/1911.08731) | [wilds](https://github.com/p-lambda/wilds) | **高**：可把不同 SNR/场景划为组 | 中-高 |
| **SWAD** | 密集随机权重平均找平坦极小，减小域泛化间隙 | [NeurIPS 2021, arXiv:2102.08604](https://arxiv.org/abs/2102.08604) | [khanrc/swad](https://github.com/khanrc/swad) | **高**：叠加 ERM 开销可控 | 中 |
| **IRM** | 在各环境上学习同时最优的不变预测器 | [arXiv:1907.02893](https://arxiv.org/abs/1907.02893) | [facebookresearch/IRM](https://github.com/facebookresearch/InvariantRiskMinimization) | 中：需明确的多环境划分 | 高 |
| **FISH** | 跨域梯度匹配（一阶近似），使更新方向一致 | [arXiv:2104.09937](https://arxiv.org/abs/2104.09937) | [YugeTen/fish](https://github.com/YugeTen/fish) | 中 | 中-高 |

**环境划分策略**：对 BirdCLEF，可将训练数据按 SNR、录音设备、生境类型划分为多个"环境"。SWAD 最实用（无需显式环境定义，直接在训练过程中做密集权重平均）。

### 9.3 测试时适应（TTA）

| 方法 | 核心原理 | 论文 | 代码 | 适用性 | 复杂度 |
|------|----------|------|------|--------|--------|
| **TENT** | 测试时仅优化 BN 仿射参数，最小化预测熵 | [ICLR 2021](https://openreview.net/forum?id=uXl3bZLkr3c) | [DequanWang/tent](https://github.com/DequanWang/tent) | **高**：在线适应无需标签 | 低-中 |
| **CoTTA** | 连续 TTA：教师-学生 + 随机恢复权重缓解误差累积 | [CVPR 2022, arXiv:2203.13591](https://arxiv.org/abs/2203.13591) | [qinenergy/cotta](https://github.com/qinenergy/cotta) | **高**：流式野外数据 | 中 |
| **SAR** | 锐度感知 + 可靠样本选择，缓解 TTA 不稳定 | [ICLR 2023, arXiv:2302.12400](https://arxiv.org/abs/2302.12400) | 社区 TTA 套件 | 中-高 | 中 |
| **EATA** | 选择性更新 + Fisher 正则减轻遗忘与冗余 | [ICML 2022](https://proceedings.mlr.press/v162/niu22a.html) | [mr-eggplant/EATA](https://github.com/mr-eggplant/EATA) | 中-高 | 中 |
| **T3A** | 无反传：用测试流构造类原型，调整最后线性层 | [NeurIPS 2021](https://openreview.net/forum?id=e_yvNqkJKAW) | [matsuolab/T3A](https://github.com/matsuolab/T3A) | 中：类别数大时原型噪声大 | **低** |
| **TTT** | 测试时用自监督辅助任务更新特征提取器 | [NeurIPS 2020, arXiv:1909.13231](https://arxiv.org/abs/1909.13231) | 社区实现 | 中-高：需设计音频自监督任务 | 中-高 |

**BirdCLEF 竞赛适用性**：竞赛推理有 2 小时时间限制。TENT/T3A 可在推理时对声景数据做轻量适应（几十次梯度步）。需注意批次过小时 TENT 易塌缩。

### 9.4 Source-Free 域适应（SFDA）

| 方法 | 核心原理 | 论文 | 代码 | 适用性 | 复杂度 |
|------|----------|------|------|--------|--------|
| **SHOT** | 冻结分类器，仅优化特征提取器；信息最大化 + 伪标签 | [ICML 2020, arXiv:2002.08504](https://arxiv.org/abs/2002.08504) | [tim-learn/SHOT](https://github.com/tim-learn/SHOT) | **高**：只需 checkpoint + 目标数据 | 中 |
| **NRC** | 利用特征邻域结构做伪标签传播 | [NeurIPS 2021, arXiv:2006.11326](https://arxiv.org/abs/2006.11326) | [Albert0147/NRC_SFDA](https://github.com/Albert0147/NRC_SFDA) | 中-高 | 中 |
| **BMD** | 多中心动态原型 + 类平衡，缓解长尾偏置 | [ECCV 2022, arXiv:2204.02811](https://arxiv.org/abs/2204.02811) | [ispc-lab/BMD](https://github.com/ispc-lab/BMD) | 中-高：长尾场景 | 中 |

**竞赛优势**：Kaggle 推理环境无法访问训练数据，只有模型权重。SFDA 天然适配——在测试音频上做在线适应。

### 9.5 音频/生物声学特化

| 方法 | 核心原理 | 论文 | 代码 | 适用性 |
|------|----------|------|------|--------|
| **无训练频域自适应** | 不调权重，通过中间层频域滤波恢复时频结构 | [arXiv:2412.17212](https://arxiv.org/abs/2412.17212) | 见论文 | 中：低成本 |
| **Bird-MAE / BirdSet** | 大规模鸟类声学预训练，从表征上减小域差 | [arXiv:2504.12880](https://arxiv.org/abs/2504.12880), [arXiv:2403.10380](https://arxiv.org/abs/2403.10380) | 随论文 | **高** |
| **ProtoCLR** | 对比学习 + 域不变原型，焦点 vs PAM 迁移 | [arXiv:2409.08589](https://arxiv.org/abs/2409.08589) | 分散 | **高** |

### 9.6 综述入口

| 综述 | 链接 | 覆盖范围 |
|------|------|---------|
| DG 综述（经典） | [arXiv:2103.02503](https://arxiv.org/abs/2103.02503) | 对齐/元学习/增广/集成 |
| DG + Meta-learning（2024） | [arXiv:2404.02785](https://arxiv.org/abs/2404.02785) | 元学习视角 |
| TTA 广谱综述（2024） | [arXiv:2411.03687](https://arxiv.org/abs/2411.03687) | 400+ 篇 TTA 方法归类 |

---

## 十、通用前沿方案：零样本与少样本学习

### 10.1 音频-语言基础模型（零样本核心路线）

| 模型 | 核心原理 | 论文 | 代码 | 适用性 | 复杂度 |
|------|----------|------|------|--------|--------|
| **CLAP** | 音频编码器 + 文本编码器对比学习对齐；零样本 = 类别文本与音频嵌入相似度 | [arXiv:2309.05767](https://arxiv.org/abs/2309.05767) | [microsoft/CLAP](https://github.com/microsoft/CLAP) | **高**：最直接 | 低-中 |
| **ReCLAP** | 描述性文本（而非类名）改进 CLAP 零样本判别力 | [arXiv:2409.09213](https://arxiv.org/abs/2409.09213) | HuggingFace | **高**：物种级描述 | 低-中 |
| **Pengi** | 音频 + 文本前缀 → LLM 生成式求解 | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3a2e5889b4bbef997ddb13b55d5acf77-Abstract-Conference.html) | [microsoft/Pengi](https://github.com/microsoft/Pengi) | 中-高 | 中-高 |
| **SALMONN** | 双编码器（Whisper + BEATs）+ Q-Former + LLM | [ICLR 2024, arXiv:2310.13289](https://arxiv.org/abs/2310.13289) | [bytedance/SALMONN](https://github.com/bytedance/SALMONN) | 中 | 高 |

**零样本提示设计**：
- 物种名多模板：`"the sound of [species_name]"`, `"[species_name] bird call"`
- 声学描述：`"high-pitched insect chirping at night"`, `"rhythmic clicking pulse at 8kHz"`
- Taxonomy 层级：`"a bird in the family [family_name]"`
- ReCLAP 风格：用 LLM 为每个物种生成 5-10 条声学细节描述

### 10.2 生成式零样本学习

| 方法 | 核心原理 | 论文 | 代码 | 适用性 | 复杂度 |
|------|----------|------|------|--------|--------|
| **f-CLSWGAN** | 条件 GAN 在语义条件下生成视觉特征 | [CVPR 2018, arXiv:1712.00981](https://arxiv.org/abs/1712.00981) | 社区实现 | 低-中 | 中-高 |
| **TF-VAEGAN** | VAE+GAN + 语义解码反馈 | [ECCV 2020, arXiv:2008.08335](https://arxiv.org/abs/2008.08335) | [akshitac8/tfvaegan](https://github.com/akshitac8/tfvaegan) | 低-中 | 中-高 |
| **AudioLDM** | CLAP 潜空间 + 扩散，文本条件音频生成 | [ICML 2023, arXiv:2301.12503](https://arxiv.org/abs/2301.12503) | [haoheliu/AudioLDM](https://github.com/haoheliu/AudioLDM) | **中（高风险）**：可生成合成训练数据 | 中-高 |
| **Make-An-Audio** | 提示增强 + 频谱自编码 + 扩散 | [arXiv:2305.18474](https://arxiv.org/abs/2305.18474) | [Text-to-Audio/Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio) | 中（高风险） | 中-高 |

**结论**：为 28 个零样本物种生成合成训练数据理论可行但**高风险**——物种特异性叫声（尤其昆虫 sonotype 的脉冲间隔等微结构）难以准确合成。建议仅作为辅助正则化，不宜作为唯一监督。

### 10.3 语义空间 / Taxonomy 方法

| 方法 | 核心原理 | 适用性 |
|------|----------|--------|
| **DAP / IAP** | 预测声学属性向量 → 与未见类属性模板匹配 | 中：需定义可学习声学属性 |
| **ALE / SJE / DEVISE** | 学习音频-标签嵌入兼容性函数 | 中：CLAP 的前身式框架 |
| **分层 Taxonomy** | 利用种-属-科层级共享统计量；层级 softmax 或图网络 | **高**：近缘种特征共享 |

**Taxonomy 应用**（高优先级）：
- BirdCLEF 中近缘种可共享属/科级文本描述
- 对昆虫 sonotype：虽与鸟不同纲，但可共享"夜行昆虫高频脉冲"等声学属性
- SHiNe（[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_SHiNe_Semantic_Hierarchy_Nexus_for_Open-vocabulary_Object_Detection_CVPR_2024_paper.html)）提供层级 + 开放词表的联合框架

### 10.4 Prompt Learning 方法

| 方法 | 核心原理 | 论文 | 代码 | 适用性 |
|------|----------|------|------|--------|
| **CoOp** | 学习固定可微 prompt 替代手工模板 | [IJCV 2022, arXiv:2109.01134](https://arxiv.org/abs/2109.01134) | [KaiyangZhou/CoOp](https://github.com/KaiyangZhou/CoOp) | 中：需适配 CLAP |
| **CoCoOp** | 实例条件 prompt，减轻新类过拟合 | [CVPR 2022, arXiv:2203.05557](https://arxiv.org/abs/2203.05557) | 同上 | 中：用已见类学，未见类泛化 |
| **VLM + 文本描述** | 外部文本描述适配 VLM | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Saha_Improved_Zero-Shot_Classification_by_Adapting_VLMs_with_Text_Descriptions_CVPR_2024_paper.html) | 作者页 | 中-高 |

### 10.5 检索增强少样本

| 方法 | 核心原理 | 论文 | 适用性 |
|------|----------|------|--------|
| **RAFIC** | 从大规模库检索近邻 + 元学习 | [arXiv:2312.06868](https://arxiv.org/abs/2312.06868) | **高**（若允许外部数据） |
| **COBRA** | 组合式检索，平衡相似度与多样性 | [arXiv:2412.17684](https://arxiv.org/abs/2412.17684) | **高**（若允许外部数据） |

**实施策略**：从 Xeno-Canto / iNaturalist 建物种级检索库 → CLAP 嵌入空间 k-NN / 原型分类。

### 10.6 零样本方案优先级建议

| 优先级 | 方向 | 理由 |
|--------|------|------|
| **1** | CLAP/ReCLAP 开放词表 + 丰富文本描述 | 直接解决闭集对未见类概率塌缩 |
| **2** | Taxonomy 层级知识迁移 | 缓解文本稀疏，近缘种有效 |
| **3** | 检索增强（若规则允许外部音频） | 真实原型比纯生成稳定 |
| **4** | 生成式合成数据（AudioLDM） | 仅作辅助，昆虫 sonotype 期望保守 |

---

## 十一、通用前沿方案：预测校准

### 11.1 后处理校准方法

| 方法 | 核心原理 | 论文 | 代码 | 多标签适用性 | 复杂度 |
|------|----------|------|------|-------------|--------|
| **温度缩放 (TS)** | `p = σ(z/T)`，单标量 T 最小化验证 NLL | [Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599) | [gpleiss/temperature_scaling](https://github.com/gpleiss/temperature_scaling) | 中：单 T 无法处理逐类差异 | 低 |
| **向量缩放** | 每类 `z'_k = a_k * z_k + b_k` | 同上讨论 | NetCal | **高**：多标签 BR 灵活 | 低-中 |
| **Platt Scaling** | 逐类 logistic 映射 `p = σ(a*s + b)` | Platt 经典 | sklearn `CalibratedClassifierCV` | **高**：天然适配 BR | 低-中 |
| **Beta Calibration** | Beta 分布族 logit 变换，比 Platt 多形状自由度 | [Kull et al., AISTATS 2017](http://proceedings.mlr.press/v54/kull17a.html) | [betacal/python](https://github.com/betacal/python) | **高**：非对称 miscalibration | 低-中 |
| **Isotonic Regression** | 非参数单调映射（PAV 算法） | Zadrozny & Elkan | sklearn | 中：长尾类样本少时不稳 | 低-中 |
| **Dirichlet Calibration** | 概率单纯形上 Dirichlet 形式校准 | [Kull et al., NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/8cb0f6767e7b2d2ab5e0e4076a97744d-Abstract.html) | [dirichletcal/dirichlet_python](https://github.com/dirichletcal/dirichlet_python) | 低-中：面向多类 softmax | 中-高 |

**统一工具库**：[NetCal](https://github.com/EFS-OpenSource/calibration-framework)（`pip install netcal`）集成 TS、Logistic、Beta、Binning 等，便于批量实验。

### 11.2 训练时校准方法

| 方法 | 核心原理 | 论文 | 适用性 | 备注 |
|------|----------|------|--------|------|
| **Label Smoothing** | 软标签防过度自信 | [arXiv:1906.02629](https://arxiv.org/abs/1906.02629) | 中：我们是**欠置信**，过度 smoothing 可能更压概率 | 需小网格调参 |
| **Mixup** | 凸组合 + 软标签改善校准 | [arXiv:1905.11001](https://arxiv.org/abs/1905.11001) | 中：对欠置信不一定单向 | 音频可在 mel 上做 |
| **Focal Loss** | 难例聚焦 + 比 CE 更好校准（Mukhoti et al.） | [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html) | 中：**注意**我们欠置信，盲目 focal 可能加重 | [torrvision/focal_calibration](https://github.com/torrvision/focal_calibration) |
| **Logit Adjustment** | 用类先验修正 logits，长尾公平与校准 | [Menon et al., ICLR 2021](https://openreview.net/forum?id=37nvvqkCo5) | **高**：测试先验与训练不一致时强相关 | [Google Research 代码](https://github.com/google-research/google-research/tree/master/logit_adjustment) |
| **SPA + LPR** | 从 proper scoring rule 角度解释多标签损失的校准差 | [Cheng & Vasconcelos, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html) | **高**：多标签专项 | 复现中高 |

### 11.3 分布偏移下的校准（核心需求）

| 方法 | 核心原理 | 论文 | 代码 | 适用性 |
|------|----------|------|------|--------|
| **TransCal** | 域适应场景下重要性加权 + 控制变量降低校准估计偏差 | [NeurIPS 2020, arXiv:2007.08259](https://arxiv.org/abs/2007.08259) | [thuml/TransCal](https://github.com/thuml/TransCal) | 中：有 UDA 设定时 |
| **LaSCal** | Label shift 下无目标域标签估计校准误差 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/783c5986e1d6112cb4688d9b2105609a-Abstract-Conference.html) | [tpopordanoska/label-shift-calibration](https://github.com/tpopordanoska/label-shift-calibration) | 中-高：物种频率变化时 |
| **梯度修正鲁棒校准** | 频域/梯度角度缓解域移错校准 | [arXiv:2508.19830](https://arxiv.org/abs/2508.19830) | 查作者仓库 | 中 |

**关键洞察**：我们的"全概率压到 ~0"更可能是 **logit 尺度整体偏小**（训练域太干净 → 声景特征不在 learned manifold 上 → logit 幅度小）。这不是标准"过置信"问题，而是**域偏移导致的欠置信**。因此：

1. **先诊断**：是 logit 尺度问题还是排序问题（AUC 是否合理）
2. **后处理**：逐类 Platt/Beta（数据够的类）+ 全局共享 T（数据少的类）
3. **偏移修正**：TransCal / LaSCal 用目标域无标签数据估计校准参数
4. **训练修正**：Logit Adjustment 是最相关的训练时方法

### 11.4 评估指标

| 指标 | 含义 | 多标签建议 |
|------|------|-----------|
| **ECE** | 分箱后 \|置信度 - 准确率\| 加权平均 | 每类 ECE + macro 平均 |
| **MCE** | 各箱最大校准误差 | 报告分位数与类数 |
| **Brier Score** | 概率 proper scoring rule | macro / micro 均报 |
| **Reliability Diagram** | 可视化 | 每类或按置信度分层 |
| **TACE** | 改进分箱（Nixon et al., [arXiv:1904.01685](https://arxiv.org/abs/1904.01685)） | 高类数时更稳 |

**关于过/欠置信同时存在**：[Ao et al., ICML 2023](https://proceedings.mlr.press/v216/ao23a.html) 提出区分两侧 miscalibration 的方法，对设计校准策略有参考价值。

---

## 十二、更新后的综合路线图

基于通用前沿方案的补充，优化路线图调整如下：

### 12.1 Stage 1: 快速提分（零成本/低成本，1-2 天）

| 序号 | 方案 | 来源 | 预计收益 |
|------|------|------|---------|
| 1 | Sigmoid 推理（若训练用 softmax） | BirdCLEF 2024 冠军 | +0.02~0.04 |
| 2 | 逐物种阈值优化 | 工业实践 | +0.01~0.02 |
| 3 | 逐类向量缩放/Platt Scaling | 校准前沿 | +0.01~0.02 |
| 4 | 10s 上下文窗口 + 邻域平滑 | BirdCLEF 2024 冠军 | +0.01~0.03 |
| 5 | T3A 无反传测试时适应 | TTA 前沿 | +0.005~0.01 |

### 12.2 Stage 2: 域适应训练（3-5 天）

| 序号 | 方案 | 来源 | 预计收益 |
|------|------|------|---------|
| 1 | 恢复完整模型（GeM + ASL + Dropout） | DES-001 | +0.02~0.03 |
| 2 | 声景弱标签数据混入训练 | BirdCLEF 实践 + UDA | +0.03~0.05 |
| 3 | 背景噪声增强（audiomentations） | DG 数据增强 | +0.02~0.03 |
| 4 | SWAD 权重平均 | DG 前沿 | +0.01~0.02 |
| 5 | Logit Adjustment（类先验修正） | 校准前沿 | +0.01~0.02 |
| 6 | Mean Teacher / FixMatch 伪标签迭代 | UDA 前沿 | +0.02~0.04 |
| 7 | 5-fold 集成 | 竞赛实践 | +0.01~0.02 |

### 12.3 Stage 3: 高级方案（1-2 周）

| 序号 | 方案 | 来源 | 预计收益 |
|------|------|------|---------|
| 1 | CLAP/ReCLAP 零样本辅助 | 零样本前沿 | +0.01~0.02 |
| 2 | PANNs/BEATs/Bird-MAE 预训练 backbone | 音频预训练 | +0.02~0.04 |
| 3 | SED 架构（Attention Pooling） | SED 前沿 | +0.01~0.03 |
| 4 | TENT/CoTTA 测试时适应 | TTA 前沿 | +0.01~0.02 |
| 5 | Taxonomy 层级知识迁移 | 零样本前沿 | +0.005~0.01 |
| 6 | 多架构集成（CNN + SED + Transformer） | 竞赛实践 | +0.01~0.02 |

---

## 参考

- [ANAL-003: LB 0.779 诊断分析](ANAL-003-lb-diagnostic.md)
- [DES-001: 模型架构设计方案](../design/DES-001-model-architecture.md)
- [ANAL-001: 竞赛深度分析](ANAL-001-competition-analysis.md)
