"""
Kaggle 推理流水线
测试声景 → 5s 窗口 → 模型预测 → 后处理 → submission.csv
"""
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from pathlib import Path

from . import config as CFG
from .utils import load_audio, audio_to_melspec, compute_insect_energy_ratio, load_taxonomy
from .model import BirdCLEFB0
from .postprocess import apply_cooccurrence, apply_time_prior, apply_sonotype_split


def load_models(model_dir: Path, device, n_folds: int = CFG.N_FOLDS):
    """加载所有 fold 模型"""
    models = []
    for fold_idx in range(n_folds):
        model = BirdCLEFB0(pretrained=False).to(device)
        ckpt_path = model_dir / f"best_fold{fold_idx}.pth"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            models.append(model)
    print(f"Loaded {len(models)} models")
    return models


def parse_soundscape_filename(filename: str):
    """从声景文件名解析站点和时间: BC2026_Test_XXXX_S05_20250227_010002"""
    parts = Path(filename).stem.split("_")
    site = parts[3] if len(parts) >= 4 else ""
    if len(parts) >= 6:
        time_str = parts[5]
        hour = int(time_str[:2]) if len(time_str) >= 2 else 0
    else:
        hour = 12
    return site, hour


@torch.no_grad()
def predict_file(filepath: str, models, device, spec_cfg=None):
    """对单个声景文件做 12 个 5s 窗口的预测"""
    spec_cfg = spec_cfg or CFG.SPEC_FINE
    _, hour = parse_soundscape_filename(filepath)
    predictions = []

    for start_sec in range(0, 60, 5):
        audio = load_audio(filepath, offset=float(start_sec), duration=CFG.CLIP_DURATION)
        mel = audio_to_melspec(audio, spec_cfg)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)
        insect_e = torch.tensor([[compute_insect_energy_ratio(audio)]],
                                dtype=torch.float32, device=device)

        fold_preds = []
        for model in models:
            with autocast():
                logits = model(mel_tensor, insect_energy=insect_e)
            pred = torch.sigmoid(logits).cpu().numpy()[0]
            fold_preds.append(pred)

        avg_pred = np.mean(fold_preds, axis=0)
        predictions.append((start_sec + 5, avg_pred, hour))

    return predictions


def run_inference(model_dir: Path = None, output_path: Path = None):
    """完整推理流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = model_dir or CFG.OUTPUT_DIR
    output_path = output_path or CFG.OUTPUT_DIR / "submission.csv"

    models = load_models(model_dir, device)
    taxonomy, species_cols, label_to_idx = load_taxonomy()

    tax_map = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"]))

    cond_prob = build_cooccurrence_matrix(label_to_idx)

    test_files = sorted(CFG.TEST_SOUNDSCAPES_DIR.glob("*.ogg"))
    print(f"Test files: {len(test_files)}")

    rows = []
    for filepath in test_files:
        predictions = predict_file(str(filepath), models, device)
        stem = filepath.stem

        for end_sec, preds, hour in predictions:
            preds = apply_cooccurrence(preds, cond_prob)
            preds = apply_time_prior(preds, hour, species_cols, tax_map)
            preds = apply_sonotype_split(preds, species_cols)

            row_id = f"{stem}_{end_sec}"
            row = {"row_id": row_id}
            for i, col in enumerate(species_cols):
                row[col] = preds[i]
            rows.append(row)

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path} ({len(submission)} rows)")
    return submission


def build_cooccurrence_matrix(label_to_idx: dict) -> np.ndarray:
    """从声景标注构建共现条件概率矩阵"""
    labels_df = pd.read_csv(CFG.SOUNDSCAPE_LABELS_CSV)
    n = CFG.NUM_CLASSES
    cond_prob = np.zeros((n, n))

    for _, row in labels_df.iterrows():
        labels = [l.strip() for l in str(row["primary_label"]).split(";")]
        indices = [label_to_idx[l] for l in labels if l in label_to_idx]
        for a in indices:
            for b in indices:
                if a != b:
                    cond_prob[a][b] += 1

    row_sums = cond_prob.sum(axis=1, keepdims=True) + 1e-8
    cond_prob /= row_sums
    return cond_prob
