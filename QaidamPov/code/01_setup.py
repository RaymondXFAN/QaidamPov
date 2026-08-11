#!/usr/bin/env python3
"""
QaidamPov - 01 环境检查 + 数据链接
检查 Python/PyTorch/GPU，安装缺失包，创建数据软链接
"""
import os
import sys
import subprocess

# ========== 项目根目录（绝对路径，AutoDL 上固定）==========
PROJECT_ROOT = "/root/autodl-tmp/QaidamPov"
OLD_DATA_ROOT = "/root/autodl-tmp/qaidam-salinization-project/data"

def run_cmd(cmd):
    """运行命令并打印结果"""
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(f"    {r.stdout.strip()}")
    if r.returncode != 0 and r.stderr.strip():
        print(f"    ⚠️ {r.stderr.strip()}")
    return r.returncode == 0

def check_python():
    print("\n[1/6] Python 环境")
    print(f"  Python: {sys.version}")
    print(f"  路径: {sys.executable}")

def check_pytorch():
    print("\n[2/6] PyTorch & CUDA")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError:
        print("  ❌ PyTorch 未安装！")
        sys.exit(1)

def check_packages():
    print("\n[3/6] 必需包检查 & 安装")
    required = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'timm': 'timm',
        'tqdm': 'tqdm',
    }
    for mod, pkg in required.items():
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except ImportError:
            print(f"  ⚠️ {mod} 缺失，正在安装 {pkg}...")
            run_cmd(f"pip install {pkg} -q")

def create_dirs():
    print("\n[4/6] 创建项目目录")
    dirs = ['results', 'checkpoints', 'logs']
    for d in dirs:
        path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(path, exist_ok=True)
        print(f"  ✅ {path}")

def link_data():
    print("\n[5/6] 链接数据")
    data_link = os.path.join(PROJECT_ROOT, "data")

    # 检查旧数据是否存在
    old_dual = os.path.join(OLD_DATA_ROOT, "processed_dual")
    if not os.path.exists(old_dual):
        print(f"  ❌ 旧数据不存在: {old_dual}")
        print(f"  💡 请确认数据位置，或手动创建软链接")
        return

    # 如果 data 链接已存在，先删除
    if os.path.islink(data_link):
        os.unlink(data_link)
        print(f"  🗑️ 删除旧软链接: {data_link}")
    elif os.path.isdir(data_link):
        # 如果是真实目录，不删除
        print(f"  ⚠️ {data_link} 是真实目录，不覆盖")
        # 检查 processed_dual 是否在里面
        if os.path.exists(os.path.join(data_link, "processed_dual")):
            print(f"  ✅ 数据已就位")
            return

    # 创建软链接
    os.symlink(OLD_DATA_ROOT, data_link)
    print(f"  ✅ 软链接: {data_link} -> {OLD_DATA_ROOT}")

def verify_data():
    print("\n[6/6] 验证数据完整性")
    dual_dir = os.path.join(PROJECT_ROOT, "data", "processed_dual")
    required_files = [
        'train_x.npy', 'train_y_static.npy', 'train_y_change.npy',
        'val_x.npy', 'val_y_static.npy', 'val_y_change.npy',
        'test_x.npy', 'test_y_static.npy', 'test_y_change.npy',
    ]
    all_ok = True
    for f in required_files:
        path = os.path.join(dual_dir, f)
        if os.path.exists(path):
            import numpy as np
            arr = np.load(path, mmap_mode='r')
            print(f"  ✅ {f}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  ❌ {f}: 缺失!")
            all_ok = False

    if all_ok:
        print(f"\n{'='*60}")
        print(f"🎉 环境检查完成！所有数据就位！")
        print(f"{'='*60}")
        print(f"项目目录: {PROJECT_ROOT}")
        print(f"数据目录: {dual_dir}")
        print(f"\n下一步: python 02_train_dual.py --mode both")
    else:
        print(f"\n⚠️ 有数据缺失，请检查数据目录")

if __name__ == '__main__':
    print("=" * 60)
    print("🏠 QaidamPov 项目环境检查")
    print("=" * 60)
    check_python()
    check_pytorch()
    check_packages()
    create_dirs()
    link_data()
    verify_data()
