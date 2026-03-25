#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "  BirdCLEF 2026 环境设置与数据下载"
echo "=========================================="

# 1. Kaggle 认证
echo -e "\n[1/5] 配置 Kaggle 认证..."
mkdir -p ~/.kaggle
if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo '{"username":"MontyEternity","key":"KGAT_8065decb316dba165829697468ad4abf"}' > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  echo "  ✅ kaggle.json 已创建"
else
  echo "  ✅ kaggle.json 已存在"
fi

# 2. 安装 kaggle CLI
echo -e "\n[2/5] 安装 Kaggle CLI..."
if command -v kaggle >/dev/null 2>&1; then
  echo "  ✅ kaggle 已安装: $(kaggle --version 2>&1)"
else
  pip install -q kaggle
  echo "  ✅ kaggle 安装完成: $(kaggle --version 2>&1)"
fi

# 3. 检查磁盘空间
echo -e "\n[3/5] 磁盘空间..."
df -h /home/theia/ | head -3

# 4. 检查 ML 库
echo -e "\n[4/5] 检查 ML 库..."
python3 -c "
import importlib
libs = ['torch','torchaudio','librosa','numpy','pandas','matplotlib','sklearn','timm','soundfile']
for m in libs:
    try:
        mod = importlib.import_module(m)
        print(f'  ✅ {m}: {getattr(mod, \"__version__\", \"ok\")}')
    except ImportError:
        print(f'  ❌ {m}: NOT INSTALLED')
"

# 5. 下载数据
DATA_DIR="/home/theia/code/Tbaymax/BirdCLEF/data/raw"
echo -e "\n[5/5] 下载 BirdCLEF 2026 竞赛数据..."
mkdir -p "$DATA_DIR"

echo "  目标目录: $DATA_DIR"
echo "  开始下载（可能需要几分钟）..."

kaggle competitions download -c birdclef-2026 -p "$DATA_DIR"

echo -e "\n  下载完成！文件列表:"
ls -lah "$DATA_DIR"

echo -e "\n=========================================="
echo "  设置完成！请将以上输出贴回聊天"
echo "=========================================="
