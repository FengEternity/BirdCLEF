"""
BirdCLEF+ 2026 配置
基于 DES-001 模型架构设计方案
"""
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    DATA_DIR = Path("/kaggle/input/birdclef-2026")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"

TRAIN_CSV = DATA_DIR / "train.csv"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"
SOUNDSCAPE_LABELS_CSV = DATA_DIR / "train_soundscapes_labels.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
TEST_SOUNDSCAPES_DIR = DATA_DIR / "test_soundscapes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 音频 ──────────────────────────────────────────────────────
SR = 32_000
CLIP_DURATION = 5.0
CLIP_SAMPLES = int(SR * CLIP_DURATION)  # 160_000

# ── 频谱图（Fine - Stage 1 基线） ────────────────────────────
SPEC_FINE = dict(
    n_fft=1024,
    hop_length=320,   # 10ms → 500 frames / 5s
    n_mels=128,
    fmin=50,
    fmax=14000,
    power=2.0,
)

SPEC_COARSE = dict(
    n_fft=2048,
    hop_length=640,   # 20ms → 250 frames / 5s
    n_mels=128,
    fmin=50,
    fmax=14000,
    power=2.0,
)

# ── 训练（Stage 1 基线） ─────────────────────────────────────
NUM_CLASSES = 234
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

# ── 纲级类别权重乘数 ─────────────────────────────────────────
CLASS_MULTIPLIER = {
    "Aves": 1.0,
    "Amphibia": 20.0,
    "Insecta": 50.0,
    "Mammalia": 30.0,
    "Reptilia": 100.0,
}

# ── 损失函数权重 ──────────────────────────────────────────────
BCE_WEIGHT = 0.7
FOCAL_WEIGHT = 0.3
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# ── Secondary label 软标签权重 ────────────────────────────────
SECONDARY_LABEL_WEIGHT = 0.3

# ── 数据增强概率 ──────────────────────────────────────────────
MIXUP_PROB = 0.3
MIXUP_ALPHA = 0.15
TIME_MASK_PROB = 0.3
TIME_MASK_WIDTH = 50
FREQ_MASK_PROB = 0.3
FREQ_MASK_WIDTH = 20
NOISE_PROB = 0.2
NOISE_STD = 0.005
GAIN_PROB = 0.3
GAIN_RANGE = (-6, 6)

# ── 交叉验证 ──────────────────────────────────────────────────
N_FOLDS = 5
