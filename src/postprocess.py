"""
后处理：共现先验、时间先验、零样本 sonotype 处理
基于 DES-001 §四
"""
import numpy as np

from . import config as CFG

# ── 时间先验（24h × 纲） ─────────────────────────────────────
TIME_PRIOR = {
    "Aves":     [0.2, 0.2, 0.3, 0.5, 0.9, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3, 0.3,
                 0.3, 0.3, 0.3, 0.4, 0.5, 0.8, 1.0, 0.6, 0.3, 0.2, 0.2, 0.2],
    "Amphibia": [0.8, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
                 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 0.9, 0.9],
    "Insecta":  [0.5, 0.5, 0.5, 0.8, 0.6, 0.4, 1.0, 1.0, 0.4, 0.3, 0.3, 0.3,
                 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5],
}

# ── 不可区分 sonotype 超类分组 ────────────────────────────────
SONOTYPE_GROUPS = {
    "super_A": ["47158son08", "47158son11", "47158son20"],
    "super_B": ["47158son13", "47158son22", "47158son23"],
    "super_C": ["47158son15", "47158son16", "47158son25"],
    "super_D": ["47158son04", "47158son10"],
}


def apply_cooccurrence(preds: np.ndarray, cond_prob: np.ndarray,
                       detect_thresh: float = 0.5, boost: float = 0.15,
                       min_cond_prob: float = 0.3) -> np.ndarray:
    """
    共现后处理：检测到物种 A 时提升高共现物种 B
    """
    preds = preds.copy()
    detected = np.where(preds > detect_thresh)[0]
    for a in detected:
        for b in range(len(preds)):
            if cond_prob[a][b] > min_cond_prob:
                preds[b] = max(preds[b], preds[a] * cond_prob[a][b] * boost)
    return preds


def apply_time_prior(preds: np.ndarray, hour: int,
                     species_cols: list, tax_map: dict,
                     weight: float = 0.1) -> np.ndarray:
    """
    时间先验：根据一天中的时间调整不同纲的预测概率
    """
    preds = preds.copy()
    hour = hour % 24
    for idx, col in enumerate(species_cols):
        class_name = tax_map.get(col, "")
        if class_name in TIME_PRIOR:
            factor = TIME_PRIOR[class_name][hour]
            preds[idx] = preds[idx] * (1 - weight) + preds[idx] * factor * weight
    return preds


def apply_sonotype_split(preds: np.ndarray, species_cols: list) -> np.ndarray:
    """
    将超类预测概率均匀拆分给组内成员
    """
    preds = preds.copy()
    col_to_idx = {c: i for i, c in enumerate(species_cols)}

    for group_name, members in SONOTYPE_GROUPS.items():
        member_indices = [col_to_idx[m] for m in members if m in col_to_idx]
        if not member_indices:
            continue
        max_pred = max(preds[i] for i in member_indices)
        split_val = max_pred / len(member_indices)
        for i in member_indices:
            preds[i] = split_val

    return preds
