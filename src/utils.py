"""
工具函数：频谱图生成、指标计算、种子设置
"""
import random
import os
import numpy as np
import pandas as pd
import torch
import librosa
from sklearn.metrics import roc_auc_score

from . import config as CFG


def set_seed(seed: int = CFG.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_taxonomy() -> pd.DataFrame:
    """加载物种分类表，返回按 submission 列顺序排列的 DataFrame"""
    taxonomy = pd.read_csv(CFG.TAXONOMY_CSV)
    submission = pd.read_csv(CFG.SAMPLE_SUBMISSION_CSV, nrows=0)
    species_cols = [c for c in submission.columns if c != "row_id"]
    label_to_idx = {label: i for i, label in enumerate(species_cols)}
    taxonomy["label_idx"] = taxonomy["primary_label"].astype(str).map(label_to_idx)
    return taxonomy, species_cols, label_to_idx


def load_audio(path: str, sr: int = CFG.SR, duration: float = CFG.CLIP_DURATION,
               offset: float = 0.0) -> np.ndarray:
    """加载音频并裁剪/循环 padding 到指定时长"""
    audio, _ = librosa.load(path, sr=sr, offset=offset, duration=duration, mono=True)
    target_len = int(sr * duration)
    if len(audio) < target_len:
        repeats = (target_len // len(audio)) + 1
        audio = np.tile(audio, repeats)[:target_len]
    else:
        audio = audio[:target_len]
    return audio


def audio_to_melspec(audio: np.ndarray, spec_cfg: dict = None) -> np.ndarray:
    """将音频波形转换为 log-mel 频谱图，返回 (n_mels, time_frames)"""
    if spec_cfg is None:
        spec_cfg = CFG.SPEC_FINE
    mel = librosa.feature.melspectrogram(
        y=audio, sr=CFG.SR,
        n_fft=spec_cfg["n_fft"],
        hop_length=spec_cfg["hop_length"],
        n_mels=spec_cfg["n_mels"],
        fmin=spec_cfg["fmin"],
        fmax=spec_cfg["fmax"],
        power=spec_cfg["power"],
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db


def compute_insect_energy_ratio(audio: np.ndarray, sr: int = CFG.SR,
                                low: int = 4000, high: int = 6000) -> float:
    """计算 4-6 kHz 子带能量占比（昆虫辅助特征）"""
    S = np.abs(librosa.stft(audio, n_fft=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mask = (freqs >= low) & (freqs <= high)
    band_energy = float(S[mask].sum())
    total_energy = float(S.sum()) + 1e-8
    return band_energy / total_energy


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 macro AUC-ROC（竞赛主指标）"""
    valid_cols = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            valid_cols.append(i)
    if not valid_cols:
        return {"macro_auc": 0.0, "valid_classes": 0}
    auc = roc_auc_score(
        y_true[:, valid_cols], y_pred[:, valid_cols], average="macro"
    )
    return {"macro_auc": auc, "valid_classes": len(valid_cols)}


def compute_class_weights(train_df: pd.DataFrame, taxonomy: pd.DataFrame,
                          label_to_idx: dict) -> torch.Tensor:
    """计算纲级均衡 + 物种频率逆权重"""
    species_counts = train_df["primary_label"].astype(str).value_counts()
    max_count = species_counts.max()
    tax_map = dict(zip(
        taxonomy["primary_label"].astype(str),
        taxonomy["class_name"]
    ))
    weights = torch.ones(CFG.NUM_CLASSES)
    for label, idx in label_to_idx.items():
        count = species_counts.get(label, 1)
        class_name = tax_map.get(label, "Aves")
        multiplier = CFG.CLASS_MULTIPLIER.get(class_name, 1.0)
        w = (max_count / max(count, 1)) * multiplier
        weights[idx] = min(w, 50.0)
    return weights
