"""
端到端流水线测试：验证数据加载 → 模型前向 → 损失计算 → 反向传播全部能跑通
使用 CPU + 少量样本，不做真正训练
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config as CFG
from src.utils import set_seed, load_taxonomy, load_audio, audio_to_melspec, compute_insect_energy_ratio, compute_class_weights
from src.dataset import BirdCLEFDataset, SoundscapeDataset, mixup_data
from src.model import BirdCLEFB0, SEDB0, BirdCLEFLoss
from src.postprocess import apply_cooccurrence, apply_time_prior, apply_sonotype_split

set_seed()
DEVICE = torch.device('cpu')
NUM_TEST_SAMPLES = 8


def test_data_loading():
    print("=" * 50)
    print("1. 数据加载测试")
    print("=" * 50)

    taxonomy, species_cols, label_to_idx = load_taxonomy()
    print(f"  物种数: {len(species_cols)}")
    print(f"  分类表行数: {len(taxonomy)}")
    assert len(species_cols) == 234, f"Expected 234 species, got {len(species_cols)}"
    assert len(label_to_idx) == 234

    train_df = pd.read_csv(CFG.TRAIN_CSV)
    print(f"  训练集行数: {len(train_df)}")
    print(f"  列: {list(train_df.columns)}")

    sample_row = train_df.iloc[0]
    filepath = CFG.TRAIN_AUDIO_DIR / sample_row['filename']
    print(f"  测试文件: {filepath}")
    assert filepath.exists(), f"File not found: {filepath}"

    audio = load_audio(str(filepath))
    print(f"  音频形状: {audio.shape}, 时长: {len(audio)/CFG.SR:.2f}s")
    assert len(audio) == CFG.CLIP_SAMPLES, f"Expected {CFG.CLIP_SAMPLES}, got {len(audio)}"

    mel = audio_to_melspec(audio)
    print(f"  频谱图形状: {mel.shape}")  # (128, 500)
    assert mel.shape[0] == 128, f"Expected 128 mels, got {mel.shape[0]}"

    ie = compute_insect_energy_ratio(audio)
    print(f"  4-6kHz 能量比: {ie:.4f}")
    assert 0 <= ie <= 1

    print("  PASS\n")
    return taxonomy, species_cols, label_to_idx, train_df


def test_dataset(train_df, label_to_idx):
    print("=" * 50)
    print("2. Dataset 测试")
    print("=" * 50)

    small_df = train_df.head(NUM_TEST_SAMPLES)
    ds = BirdCLEFDataset(small_df, label_to_idx, is_train=True)
    print(f"  Dataset 大小: {len(ds)}")

    t0 = time.time()
    item = ds[0]
    dt = time.time() - t0
    mel = item['mel']
    target = item['target']
    ie = item['insect_energy']
    print(f"  mel: {mel.shape}, dtype={mel.dtype}")      # (1, 128, ~500)
    print(f"  target: {target.shape}, sum={target.sum():.1f}")
    print(f"  insect_energy: {ie.shape}")
    print(f"  加载耗时: {dt:.2f}s")
    assert mel.dim() == 3
    assert target.shape[0] == 234

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"  Batch mel: {batch['mel'].shape}")
    print(f"  Batch target: {batch['target'].shape}")

    mixed_mel, mixed_target = mixup_data(batch['mel'], batch['target'])
    print(f"  Mixup mel: {mixed_mel.shape}")
    assert mixed_mel.shape == batch['mel'].shape

    print("  PASS\n")
    return loader


def test_model():
    print("=" * 50)
    print("3. 模型前向测试")
    print("=" * 50)

    model = BirdCLEFB0(pretrained=False).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  BirdCLEF-B0 参数: {params / 1e6:.1f}M")

    dummy_mel = torch.randn(2, 1, 128, 500)
    dummy_ie = torch.randn(2, 1)
    dummy_hour = torch.tensor([6, 20])

    logits = model(dummy_mel, insect_energy=dummy_ie, hour=dummy_hour)
    print(f"  输出形状: {logits.shape}")  # (2, 234)
    assert logits.shape == (2, 234)

    logits_no_aux = model(dummy_mel)
    print(f"  无辅助特征输出: {logits_no_aux.shape}")
    assert logits_no_aux.shape == (2, 234)

    sed = SEDB0(pretrained=False).to(DEVICE)
    sed_params = sum(p.numel() for p in sed.parameters())
    print(f"  SED-B0 参数: {sed_params / 1e6:.1f}M")

    sed_out = sed(dummy_mel)
    print(f"  SED 输出形状: {sed_out.shape}")
    assert sed_out.shape == (2, 234)

    print("  PASS\n")
    return model


def test_loss_and_backward(model, train_df, label_to_idx, taxonomy):
    print("=" * 50)
    print("4. 损失 + 反向传播测试")
    print("=" * 50)

    class_weights = compute_class_weights(train_df, taxonomy, label_to_idx)
    print(f"  类别权重: min={class_weights.min():.2f}, max={class_weights.max():.2f}, mean={class_weights.mean():.2f}")

    criterion = BirdCLEFLoss(pos_weight=class_weights)
    dummy_logits = torch.randn(4, 234, requires_grad=True)
    dummy_target = torch.zeros(4, 234)
    dummy_target[0, 10] = 1.0
    dummy_target[1, 50] = 1.0
    dummy_target[2, 100] = 1.0
    dummy_target[3, 200] = 1.0

    loss = criterion(dummy_logits, dummy_target)
    print(f"  损失值: {loss.item():.4f}")
    assert not torch.isnan(loss)

    loss.backward()
    print(f"  梯度: {dummy_logits.grad.abs().mean():.6f}")
    assert dummy_logits.grad is not None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    mel = torch.randn(2, 1, 128, 500)
    target = torch.zeros(2, 234)
    target[0, 5] = 1.0
    target[1, 100] = 1.0

    logits = model(mel)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()
    print(f"  模型训练步骤完成, loss={loss.item():.4f}")

    print("  PASS\n")


def test_postprocess():
    print("=" * 50)
    print("5. 后处理测试")
    print("=" * 50)

    taxonomy, species_cols, label_to_idx = load_taxonomy()
    tax_map = dict(zip(taxonomy['primary_label'].astype(str), taxonomy['class_name']))

    preds = np.random.rand(234) * 0.3
    preds[10] = 0.8

    cond_prob = np.zeros((234, 234))
    cond_prob[10, 20] = 0.5
    result = apply_cooccurrence(preds, cond_prob)
    print(f"  共现后处理: pred[20] {preds[20]:.4f} → {result[20]:.4f}")
    assert result[20] >= preds[20]

    result2 = apply_time_prior(preds, hour=5, species_cols=species_cols, tax_map=tax_map)
    print(f"  时间先验: 5am 调整完成")

    result3 = apply_sonotype_split(preds, species_cols)
    print(f"  Sonotype 拆分完成")

    print("  PASS\n")


def test_real_data_pipeline(train_df, label_to_idx, taxonomy):
    print("=" * 50)
    print("6. 真实数据端到端测试（CPU, 4 样本）")
    print("=" * 50)

    small_df = train_df.head(4)
    ds = BirdCLEFDataset(small_df, label_to_idx, is_train=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    model = BirdCLEFB0(pretrained=False).to(DEVICE)
    class_weights = compute_class_weights(train_df, taxonomy, label_to_idx)
    criterion = BirdCLEFLoss(pos_weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for i, batch in enumerate(loader):
        mel = batch['mel'].to(DEVICE)
        target = batch['target'].to(DEVICE)
        ie = batch['insect_energy'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(mel, insect_energy=ie)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(logits).detach()
        print(f"  Batch {i}: loss={loss.item():.4f}, "
              f"pred_mean={preds.mean():.4f}, pred_max={preds.max():.4f}")

    print("  端到端测试 PASS\n")


if __name__ == '__main__':
    print("\n🔬 BirdCLEF Pipeline 端到端测试\n")

    taxonomy, species_cols, label_to_idx, train_df = test_data_loading()
    test_dataset(train_df, label_to_idx)
    model = test_model()
    test_loss_and_backward(model, train_df, label_to_idx, taxonomy)
    test_postprocess()
    test_real_data_pipeline(train_df, label_to_idx, taxonomy)

    print("=" * 50)
    print("✅ 所有测试通过！代码流水线已验证。")
    print("=" * 50)
