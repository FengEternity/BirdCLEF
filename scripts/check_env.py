#!/usr/bin/env python3
"""Check environment for BirdCLEF competition."""
import os
import shutil
import subprocess
import sys

print("=" * 50)
print("BirdCLEF Environment Check")
print("=" * 50)

# Python version
print(f"\nPython: {sys.version}")

# Disk space
usage = shutil.disk_usage("/home/theia")
print(f"Disk: total={usage.total // 1024**3}GB, used={usage.used // 1024**3}GB, free={usage.free // 1024**3}GB")

# Check kaggle CLI
kaggle_path = shutil.which("kaggle")
print(f"\nKaggle CLI: {kaggle_path or 'NOT FOUND'}")

if kaggle_path:
    r = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
    print(f"Kaggle version: {r.stdout.strip()}")

# Check kaggle credentials
creds_path = os.path.expanduser("~/.kaggle/kaggle.json")
print(f"Kaggle credentials: {'EXISTS' if os.path.exists(creds_path) else 'NOT FOUND'}")

# Check key libraries
libs = ["torch", "torchaudio", "librosa", "soundfile", "numpy", "pandas", 
        "matplotlib", "sklearn", "timm", "audiomentations"]
print("\nLibraries:")
for lib in libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "?")
        print(f"  {lib}: {ver}")
    except ImportError:
        print(f"  {lib}: NOT INSTALLED")

# Check pip for kaggle
r = subprocess.run([sys.executable, "-m", "pip", "show", "kaggle"], capture_output=True, text=True)
if r.returncode == 0:
    for line in r.stdout.split("\n"):
        if line.startswith(("Name:", "Version:")):
            print(f"  pip kaggle: {line}")
else:
    print("  pip kaggle: NOT INSTALLED")

print("\n" + "=" * 50)
print("Done.")
