"""
数据集：train_audio + train_soundscapes 统一接口
支持 mixup、SpecAugment、高斯噪声、随机增益
"""
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as CFG
from .utils import load_audio, audio_to_melspec, compute_insect_energy_ratio


class BirdCLEFDataset(Dataset):
    """Stage 1 训练集：单标签录音数据"""

    def __init__(self, df: pd.DataFrame, label_to_idx: dict,
                 spec_cfg: dict = None, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.spec_cfg = spec_cfg or CFG.SPEC_FINE
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = CFG.TRAIN_AUDIO_DIR / row["filename"]

        duration = row.get("duration", None)
        if self.is_train and duration and duration > CFG.CLIP_DURATION:
            max_offset = max(0, duration - CFG.CLIP_DURATION)
            offset = np.random.uniform(0, max_offset)
        else:
            offset = 0.0

        audio = load_audio(str(filepath), offset=offset)

        if self.is_train:
            audio = self._augment_waveform(audio)

        mel = audio_to_melspec(audio, self.spec_cfg)

        if self.is_train:
            mel = self._augment_spectrogram(mel)

        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)  # (1, n_mels, T)

        target = self._make_target(row)
        insect_energy = compute_insect_energy_ratio(audio)

        return {
            "mel": mel_tensor,
            "target": target,
            "insect_energy": torch.tensor([insect_energy], dtype=torch.float32),
        }

    def _make_target(self, row) -> torch.Tensor:
        target = torch.zeros(CFG.NUM_CLASSES, dtype=torch.float32)
        primary = str(row["primary_label"])
        if primary in self.label_to_idx:
            target[self.label_to_idx[primary]] = 1.0

        sec_raw = row.get("secondary_labels", "[]")
        if isinstance(sec_raw, str) and sec_raw not in ("[]", ""):
            try:
                sec_list = ast.literal_eval(sec_raw)
            except (ValueError, SyntaxError):
                sec_list = []
            for sec in sec_list:
                sec_str = str(sec)
                if sec_str in self.label_to_idx:
                    target[self.label_to_idx[sec_str]] = CFG.SECONDARY_LABEL_WEIGHT
        return target

    def _augment_waveform(self, audio: np.ndarray) -> np.ndarray:
        if np.random.rand() < CFG.NOISE_PROB:
            audio = audio + np.random.randn(len(audio)) * CFG.NOISE_STD
        if np.random.rand() < CFG.GAIN_PROB:
            gain_db = np.random.uniform(*CFG.GAIN_RANGE)
            audio = audio * (10 ** (gain_db / 20))
        return audio

    def _augment_spectrogram(self, mel: np.ndarray) -> np.ndarray:
        n_mels, n_frames = mel.shape
        if np.random.rand() < CFG.TIME_MASK_PROB:
            t = np.random.randint(0, min(CFG.TIME_MASK_WIDTH, n_frames))
            t0 = np.random.randint(0, max(1, n_frames - t))
            mel[:, t0:t0 + t] = 0.0
        if np.random.rand() < CFG.FREQ_MASK_PROB:
            f = np.random.randint(0, min(CFG.FREQ_MASK_WIDTH, n_mels))
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[f0:f0 + f, :] = 0.0
        return mel


class SoundscapeDataset(Dataset):
    """声景数据集：多标签 5s 片段"""

    def __init__(self, labels_df: pd.DataFrame, label_to_idx: dict,
                 spec_cfg: dict = None, is_train: bool = True):
        self.df = labels_df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.spec_cfg = spec_cfg or CFG.SPEC_FINE
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = CFG.TRAIN_SOUNDSCAPES_DIR / row["filename"]

        start_parts = str(row["start"]).split(":")
        offset = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])

        audio = load_audio(str(filepath), offset=float(offset))

        if self.is_train:
            audio = self._augment_waveform(audio)

        mel = audio_to_melspec(audio, self.spec_cfg)
        if self.is_train:
            mel = self._augment_spectrogram(mel)

        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)

        target = torch.zeros(CFG.NUM_CLASSES, dtype=torch.float32)
        labels_str = str(row["primary_label"])
        for label in labels_str.split(";"):
            label = label.strip()
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0

        insect_energy = compute_insect_energy_ratio(audio)

        return {
            "mel": mel_tensor,
            "target": target,
            "insect_energy": torch.tensor([insect_energy], dtype=torch.float32),
        }

    _augment_waveform = BirdCLEFDataset._augment_waveform
    _augment_spectrogram = BirdCLEFDataset._augment_spectrogram


def mixup_data(mel, target, alpha=CFG.MIXUP_ALPHA):
    """Mixup 数据增强：混合两个样本"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = mel.size(0)
    perm = torch.randperm(batch_size)
    mixed_mel = lam * mel + (1 - lam) * mel[perm]
    mixed_target = lam * target + (1 - lam) * target[perm]
    return mixed_mel, mixed_target
