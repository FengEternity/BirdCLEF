"""
训练循环：Stage 1 基线
支持 5-Fold StratifiedGroupKFold + mixup + CosineAnnealing
"""
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedGroupKFold

from . import config as CFG
from .utils import set_seed, load_taxonomy, compute_metrics, compute_class_weights
from .dataset import BirdCLEFDataset, mixup_data
from .model import BirdCLEFB0, BirdCLEFLoss


def get_folds(train_df: pd.DataFrame, n_folds: int = CFG.N_FOLDS):
    """StratifiedGroupKFold: 按物种分层，按 author 分组"""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=CFG.SEED)
    labels = train_df["primary_label"].astype(str).values
    groups = train_df["author"].astype(str).values
    folds = []
    for train_idx, val_idx in sgkf.split(train_df, labels, groups):
        folds.append((train_idx, val_idx))
    return folds


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_mixup=True):
    model.train()
    total_loss = 0.0
    for batch in loader:
        mel = batch["mel"].to(device)
        target = batch["target"].to(device)
        insect_energy = batch["insect_energy"].to(device)

        if use_mixup and np.random.rand() < CFG.MIXUP_PROB:
            mel, target = mixup_data(mel, target)

        optimizer.zero_grad()
        with autocast():
            logits = model(mel, insect_energy=insect_energy)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * mel.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        mel = batch["mel"].to(device)
        target = batch["target"]
        insect_energy = batch["insect_energy"].to(device)

        with autocast():
            logits = model(mel, insect_energy=insect_energy)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(target.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    return metrics


def train_fold(fold_idx: int, train_idx, val_idx, train_df, label_to_idx,
               taxonomy, device, spec_cfg=None):
    """训练单个 fold"""
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1}/{CFG.N_FOLDS}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"{'='*60}\n")

    train_ds = BirdCLEFDataset(train_df.iloc[train_idx], label_to_idx, spec_cfg, is_train=True)
    val_ds = BirdCLEFDataset(train_df.iloc[val_idx], label_to_idx, spec_cfg, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True,
    )

    model = BirdCLEFB0(pretrained=True).to(device)
    class_weights = compute_class_weights(train_df.iloc[train_idx], taxonomy, label_to_idx)
    criterion = BirdCLEFLoss(pos_weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    scaler = GradScaler()

    best_auc = 0.0
    best_path = CFG.OUTPUT_DIR / f"best_fold{fold_idx}.pth"

    for epoch in range(CFG.EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        metrics = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        auc = metrics["macro_auc"]
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:02d}/{CFG.EPOCHS} | "
              f"loss={train_loss:.4f} | auc={auc:.4f} | "
              f"lr={lr:.6f} | {elapsed:.0f}s")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f"  ★ Best AUC: {best_auc:.4f} → saved {best_path.name}")

    return best_auc


def run_training(fold_list=None):
    """运行完整的 K-Fold 训练"""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    taxonomy, species_cols, label_to_idx = load_taxonomy()
    train_df = pd.read_csv(CFG.TRAIN_CSV)

    audio_durations = {}
    for _, row in train_df.iterrows():
        fp = CFG.TRAIN_AUDIO_DIR / row["filename"]
        if fp.exists():
            import soundfile as sf
            info = sf.info(str(fp))
            audio_durations[row["filename"]] = info.duration
    train_df["duration"] = train_df["filename"].map(audio_durations)

    folds = get_folds(train_df)
    fold_list = fold_list or list(range(CFG.N_FOLDS))

    results = []
    for fold_idx in fold_list:
        train_idx, val_idx = folds[fold_idx]
        auc = train_fold(fold_idx, train_idx, val_idx, train_df,
                         label_to_idx, taxonomy, device)
        results.append(auc)
        print(f"\n  Fold {fold_idx+1} Best AUC: {auc:.4f}")

    print(f"\n{'='*60}")
    print(f"  Mean CV AUC: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    run_training(fold_list=[0])
