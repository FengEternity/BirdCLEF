#!/usr/bin/env python3
"""
BirdCLEF 2026 - Exploratory Data Analysis
Run: python scripts/eda.py [--data-dir data/raw]
Output: EDA summary printed to stdout + figures saved to docs/analyze/figures/
"""
import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

def check_deps():
    missing = []
    for mod in ["numpy", "pandas", "matplotlib"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请安装: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_data_dir(base: str) -> Path:
    """Locate the actual data directory (handle zip and extracted)."""
    base_path = Path(base)
    if not base_path.exists():
        print(f"错误: 数据目录不存在: {base_path}")
        sys.exit(1)

    zips = list(base_path.glob("*.zip"))
    if zips:
        print(f"发现 {len(zips)} 个 zip 文件:")
        for z in zips:
            size_mb = z.stat().st_size / 1024 / 1024
            print(f"  {z.name}: {size_mb:.1f} MB")

        for z in zips:
            if not (base_path / z.stem).exists():
                print(f"\n解压 {z.name}...")
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(base_path)
                print("  解压完成")

    return base_path


def analyze_taxonomy(data_dir: Path) -> pd.DataFrame | None:
    """Analyze taxonomy CSV if present."""
    tax_files = list(data_dir.rglob("taxonomy.csv")) + list(data_dir.rglob("*taxonomy*.csv"))
    if not tax_files:
        print("未找到 taxonomy.csv")
        return None

    tax_path = tax_files[0]
    print(f"\n{'='*60}")
    print(f"📋 Taxonomy: {tax_path}")
    print(f"{'='*60}")

    df = pd.read_csv(tax_path)
    print(f"  总物种数: {len(df)}")
    print(f"  列: {list(df.columns)}")
    print(f"\n  前 5 行:")
    print(df.head().to_string(index=False))

    if "order" in df.columns:
        order_counts = df["order"].value_counts()
        print(f"\n  目 (Order) 分布 (前 10):")
        for order, count in order_counts.head(10).items():
            print(f"    {order}: {count} 种")

    if "family" in df.columns:
        family_counts = df["family"].value_counts()
        print(f"\n  科 (Family) 数量: {len(family_counts)}")

    return df


def analyze_train_metadata(data_dir: Path) -> pd.DataFrame | None:
    """Analyze training metadata CSV."""
    meta_files = (
        list(data_dir.rglob("train_metadata.csv"))
        + list(data_dir.rglob("train.csv"))
    )
    if not meta_files:
        print("\n未找到训练元数据文件")
        return None

    meta_path = meta_files[0]
    print(f"\n{'='*60}")
    print(f"📊 训练元数据: {meta_path}")
    print(f"{'='*60}")

    df = pd.read_csv(meta_path)
    print(f"  总记录数: {len(df):,}")
    print(f"  列: {list(df.columns)}")

    species_col = None
    for col in ["primary_label", "species", "ebird_code", "label"]:
        if col in df.columns:
            species_col = col
            break

    if species_col:
        species_counts = df[species_col].value_counts()
        n_species = len(species_counts)
        print(f"\n  物种列: {species_col}")
        print(f"  唯一物种数: {n_species}")
        print(f"  每物种样本数:")
        print(f"    最多: {species_counts.iloc[0]:,} ({species_counts.index[0]})")
        print(f"    最少: {species_counts.iloc[-1]:,} ({species_counts.index[-1]})")
        print(f"    中位数: {species_counts.median():.0f}")
        print(f"    平均: {species_counts.mean():.1f}")
        print(f"    标准差: {species_counts.std():.1f}")

        q25, q75 = species_counts.quantile(0.25), species_counts.quantile(0.75)
        print(f"    Q25: {q25:.0f}, Q75: {q75:.0f}")

        rare_threshold = 10
        rare_species = (species_counts < rare_threshold).sum()
        print(f"\n  稀有物种 (<{rare_threshold} 样本): {rare_species} ({rare_species/n_species*100:.1f}%)")

        abundant_threshold = 100
        abundant = (species_counts >= abundant_threshold).sum()
        print(f"  充足物种 (>={abundant_threshold} 样本): {abundant} ({abundant/n_species*100:.1f}%)")

    if "duration" in df.columns or "length" in df.columns:
        dur_col = "duration" if "duration" in df.columns else "length"
        print(f"\n  音频时长 ({dur_col}):")
        print(f"    最短: {df[dur_col].min():.1f}s")
        print(f"    最长: {df[dur_col].max():.1f}s")
        print(f"    平均: {df[dur_col].mean():.1f}s")
        print(f"    中位数: {df[dur_col].median():.1f}s")
        print(f"    总时长: {df[dur_col].sum()/3600:.1f}h")

    if "latitude" in df.columns and "longitude" in df.columns:
        valid_geo = df[df["latitude"].notna() & df["longitude"].notna()]
        print(f"\n  地理分布:")
        print(f"    有坐标记录: {len(valid_geo):,} ({len(valid_geo)/len(df)*100:.1f}%)")
        if len(valid_geo) > 0:
            print(f"    纬度范围: {valid_geo['latitude'].min():.2f} ~ {valid_geo['latitude'].max():.2f}")
            print(f"    经度范围: {valid_geo['longitude'].min():.2f} ~ {valid_geo['longitude'].max():.2f}")

    if "rating" in df.columns:
        print(f"\n  录音质量评分 (rating):")
        print(f"    分布: {df['rating'].value_counts().sort_index().to_dict()}")
        print(f"    平均: {df['rating'].mean():.2f}")

    for col in ["source", "type", "author"]:
        if col in df.columns:
            vc = df[col].value_counts()
            print(f"\n  {col}: {len(vc)} 个唯一值, 前 5: {dict(vc.head())}")

    return df


def analyze_audio_files(data_dir: Path):
    """Analyze audio file structure."""
    audio_exts = {".ogg", ".mp3", ".wav", ".flac"}
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(data_dir.rglob(f"*{ext}"))

    print(f"\n{'='*60}")
    print(f"🎵 音频文件分析")
    print(f"{'='*60}")
    print(f"  音频文件总数: {len(audio_files):,}")

    if not audio_files:
        print("  未找到音频文件（可能需要先解压）")
        return

    ext_counts = Counter(f.suffix for f in audio_files)
    print(f"  格式分布: {dict(ext_counts)}")

    sizes = [f.stat().st_size for f in audio_files[:1000]]
    if sizes:
        print(f"\n  文件大小 (采样 {len(sizes)} 个):")
        print(f"    最小: {min(sizes)/1024:.1f} KB")
        print(f"    最大: {max(sizes)/1024:.1f} KB")
        print(f"    平均: {np.mean(sizes)/1024:.1f} KB")

    dir_counts = Counter(f.parent.name for f in audio_files)
    print(f"\n  按目录分布 (物种/类别): {len(dir_counts)} 个目录")
    top_dirs = dir_counts.most_common(5)
    bottom_dirs = dir_counts.most_common()[-5:]
    print(f"  前 5 多:")
    for d, c in top_dirs:
        print(f"    {d}: {c} 个文件")
    print(f"  后 5 少:")
    for d, c in bottom_dirs:
        print(f"    {d}: {c} 个文件")

    total_size = sum(f.stat().st_size for f in audio_files) / 1024 / 1024 / 1024
    print(f"\n  音频文件总大小: {total_size:.2f} GB")


def analyze_test_data(data_dir: Path):
    """Analyze test/unlabeled soundscape data."""
    soundscape_dirs = list(data_dir.rglob("*soundscape*")) + list(data_dir.rglob("*test*"))
    soundscape_dirs = [d for d in soundscape_dirs if d.is_dir()]

    if soundscape_dirs:
        print(f"\n{'='*60}")
        print(f"🌿 声景/测试数据")
        print(f"{'='*60}")
        for sd in soundscape_dirs:
            files = list(sd.rglob("*"))
            audio = [f for f in files if f.suffix in {".ogg", ".mp3", ".wav", ".flac"}]
            print(f"  {sd.relative_to(data_dir)}: {len(audio)} 音频文件, {len(files)} 总文件")


def analyze_sample_submission(data_dir: Path):
    """Analyze sample submission file."""
    sub_files = list(data_dir.rglob("sample_submission.csv"))
    if sub_files:
        print(f"\n{'='*60}")
        print(f"📝 Sample Submission")
        print(f"{'='*60}")
        df = pd.read_csv(sub_files[0])
        print(f"  行数: {len(df):,}")
        print(f"  列: {list(df.columns)}")
        print(f"  前 3 行:")
        print(df.head(3).to_string(index=False))


def plot_species_distribution(df: pd.DataFrame, species_col: str, output_dir: Path):
    """Plot species sample count distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    species_counts = df[species_col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(range(len(species_counts)), species_counts.values, color="#4ECDC4", width=1.0)
    axes[0].set_xlabel("Species (sorted by count)")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title("Species Distribution (Long-tail)")
    axes[0].set_yscale("log")
    axes[0].axhline(y=10, color="red", linestyle="--", alpha=0.7, label="Rare threshold (10)")
    axes[0].legend()

    axes[1].hist(species_counts.values, bins=50, color="#FF6B6B", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Samples per species")
    axes[1].set_ylabel("Number of species")
    axes[1].set_title("Histogram of samples per species")

    plt.tight_layout()
    out_path = output_dir / "species_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 图表已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 EDA")
    parser.add_argument("--data-dir", default="data/raw", help="数据目录路径")
    parser.add_argument("--fig-dir", default="docs/analyze/figures", help="图表输出目录")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parent.parent
    data_dir = proj_root / args.data_dir
    fig_dir = proj_root / args.fig_dir

    print("=" * 60)
    print("  BirdCLEF 2026 - 数据探索分析 (EDA)")
    print("=" * 60)
    print(f"  数据目录: {data_dir}")
    print(f"  图表目录: {fig_dir}")

    data_dir = find_data_dir(str(data_dir))

    print(f"\n  顶层内容:")
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            n_files = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"    📁 {item.name}/ ({n_files} 个文件)")
        else:
            size_mb = item.stat().st_size / 1024 / 1024
            print(f"    📄 {item.name} ({size_mb:.1f} MB)")

    tax_df = analyze_taxonomy(data_dir)
    train_df = analyze_train_metadata(data_dir)
    analyze_audio_files(data_dir)
    analyze_test_data(data_dir)
    analyze_sample_submission(data_dir)

    if train_df is not None:
        species_col = None
        for col in ["primary_label", "species", "ebird_code", "label"]:
            if col in train_df.columns:
                species_col = col
                break
        if species_col:
            plot_species_distribution(train_df, species_col, fig_dir)

    print(f"\n{'='*60}")
    print("  EDA 完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
