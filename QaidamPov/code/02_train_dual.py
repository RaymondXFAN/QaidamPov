#!/usr/bin/env python3
"""
QaidamPov - 02 双任务模型训练（多种子 + 任务权重消融）
支持 --mode {seed|ablation|both|single}
  - seed:      多种子统计检验 (seeds: 42, 123, 2024, λ=1.0)
  - ablation:  任务权重消融 (λ: 0.3, 0.5, 0.7, seed=42)
  - both:      两种都跑
  - single:    单次训练 (seed=42, λ=1.0)
"""
import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# ========== 项目路径（与 01_setup.py 一致）==========
PROJECT = "/root/autodl-tmp/QaidamPov"
DATA_DIR = os.path.join(PROJECT, "data", "processed_dual")
CKPT_DIR = os.path.join(PROJECT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT, "results")


# ============================================================
#  工具函数
# ============================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
#  数据集
# ============================================================
class DualDataset(Dataset):
    """双任务 patch 数据集（归一化 + 随机翻转增强）"""
    def __init__(self, x_path, ys_path, yc_path, mean=None, std=None, augment=True):
        self.x = np.load(x_path).astype(np.float32)
        self.ys = np.load(ys_path).astype(np.int64)
        self.yc = np.load(yc_path).astype(np.int64)
        self.augment = augment

        # 归一化：用训练集统计量
        if mean is not None and std is not None:
            self.mean = mean.reshape(1, -1, 1, 1)
            self.std = std.reshape(1, -1, 1, 1)
        else:
            self.mean = self.x.mean(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            self.std = (self.x.std(axis=(0, 2, 3)) + 1e-8).reshape(1, -1, 1, 1)
        self.x = (self.x - self.mean) / self.std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].copy()
        ys = self.ys[idx].copy()
        yc = self.yc[idx].copy()
        if self.augment:
            if np.random.random() > 0.5:
                x, ys, yc = x[:, :, ::-1].copy(), ys[:, ::-1].copy(), yc[:, ::-1].copy()
            if np.random.random() > 0.5:
                x, ys, yc = x[:, ::-1, :].copy(), ys[::-1, :].copy(), yc[::-1, :].copy()
        return torch.FloatTensor(x), torch.LongTensor(ys), torch.LongTensor(yc)


# ============================================================
#  模型定义（与之前完全一致，确保 checkpoint 兼容）
# ============================================================
class DualTaskModel(nn.Module):
    """双任务 CNN：编码器 + 静态/变化双解码器"""
    def __init__(self, in_channels=6, num_classes=2, num_change_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.decoder_static = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
        )
        self.decoder_change = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, num_change_classes, 1),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder_static(feat), self.decoder_change(feat)


# ============================================================
#  训练 + 评估
# ============================================================
def train_one_run(seed, lambda_change, epochs, batch_size, lr, device):
    """单次训练 + 测试集评估，返回结果字典"""
    set_seed(seed)
    tag = f"seed{seed}_lam{lambda_change}"
    print(f"\n{'='*60}")
    print(f"🌱 Seed={seed} | λ_change={lambda_change} | epochs={epochs}")
    print(f"{'='*60}")

    # --- 数据 ---
    train_ds = DualDataset(
        os.path.join(DATA_DIR, 'train_x.npy'),
        os.path.join(DATA_DIR, 'train_y_static.npy'),
        os.path.join(DATA_DIR, 'train_y_change.npy'),
        augment=True,
    )
    mean_1d = train_ds.mean.flatten()
    std_1d = train_ds.std.flatten()

    val_ds = DualDataset(
        os.path.join(DATA_DIR, 'val_x.npy'),
        os.path.join(DATA_DIR, 'val_y_static.npy'),
        os.path.join(DATA_DIR, 'val_y_change.npy'),
        mean=mean_1d, std=std_1d, augment=False,
    )

    test_x = np.load(os.path.join(DATA_DIR, 'test_x.npy')).astype(np.float32)
    test_ys = np.load(os.path.join(DATA_DIR, 'test_y_static.npy')).astype(np.int64)
    test_yc = np.load(os.path.join(DATA_DIR, 'test_y_change.npy')).astype(np.int64)
    test_x_n = (test_x - mean_1d.reshape(1, -1, 1, 1)) / std_1d.reshape(1, -1, 1, 1)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_x_n), torch.LongTensor(test_ys), torch.LongTensor(test_yc)),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 模型 ---
    model = DualTaskModel(in_channels=6, num_classes=2, num_change_classes=4).to(device)

    # 类别权重
    train_ys_arr = np.load(os.path.join(DATA_DIR, 'train_y_static.npy'))
    n_pos = (train_ys_arr == 1).sum()
    n_neg = (train_ys_arr == 0).sum()
    ws = torch.FloatTensor([n_pos / (n_neg + 1e-8), 1.0]).to(device)
    ws = torch.clamp(ws, max=5.0)
    crit_s = nn.CrossEntropyLoss(weight=ws)

    train_yc_arr = np.load(os.path.join(DATA_DIR, 'train_y_change.npy'))
    counts = [(train_yc_arr == c).sum() for c in range(4)]
    total = sum(counts)
    wc = torch.FloatTensor([total / (4 * c + 1e-8) for c in counts]).to(device)
    wc = torch.clamp(wc, max=10.0)
    crit_c = nn.CrossEntropyLoss(weight=wc)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # --- 训练循环 ---
    best_combined = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0
        for x, ys, yc in train_loader:
            x, ys, yc = x.to(device), ys.to(device), yc.to(device)
            opt.zero_grad()
            os_, oc = model(x)
            loss = crit_s(os_, ys) + lambda_change * crit_c(oc, yc)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
        scheduler.step()

        # 验证
        model.eval()
        cs = cc = tv = 0
        with torch.no_grad():
            for x, ys, yc in val_loader:
                x, ys, yc = x.to(device), ys.to(device), yc.to(device)
                os_, oc = model(x)
                cs += (os_.argmax(1) == ys).sum().item()
                cc += (oc.argmax(1) == yc).sum().item()
                tv += ys.numel()
        vas = cs / tv
        vac = cc / tv
        combined = (vas + vac) / 2
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Ep {epoch:2d}/{epochs} | Loss {loss_sum/len(train_ds):.3f} | "
                  f"ValS {vas:.3f} | ValC {vac:.3f}")
        if combined > best_combined:
            best_combined = combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- 测试集评估 ---
    model.load_state_dict(best_state)
    model.eval()
    ps_list, pc_list = [], []
    with torch.no_grad():
        for x, ys, yc in test_loader:
            os_, oc = model(x.to(device))
            ps_list.append(os_.argmax(1).cpu().numpy())
            pc_list.append(oc.argmax(1).cpu().numpy())
    pred_s = np.concatenate(ps_list)
    pred_c = np.concatenate(pc_list)

    # 静态分类指标
    acc_s = accuracy_score(test_ys.flatten(), pred_s.flatten())
    f1_s = f1_score(test_ys.flatten(), pred_s.flatten(), average='weighted')
    f1m_s = f1_score(test_ys.flatten(), pred_s.flatten(), average='macro')
    kappa_s = cohen_kappa_score(test_ys.flatten(), pred_s.flatten())
    cm_s = confusion_matrix(test_ys.flatten(), pred_s.flatten())
    tp, fp, fn = cm_s[1, 1], cm_s[0, 1], cm_s[1, 0]
    iou_s = tp / (tp + fp + fn + 1e-8)

    # 变化检测指标
    acc_c = accuracy_score(test_yc.flatten(), pred_c.flatten())
    f1_c = f1_score(test_yc.flatten(), pred_c.flatten(), average='weighted')
    f1m_c = f1_score(test_yc.flatten(), pred_c.flatten(), average='macro')
    cm_c = confusion_matrix(test_yc.flatten(), pred_c.flatten())
    f1_per_c = f1_score(test_yc.flatten(), pred_c.flatten(), average=None, labels=[0, 1, 2, 3])

    result = {
        'seed': seed, 'lambda_change': lambda_change,
        'static': {
            'accuracy': float(acc_s), 'f1_weighted': float(f1_s),
            'f1_macro': float(f1m_s), 'kappa': float(kappa_s),
            'iou_saline': float(iou_s), 'confusion_matrix': cm_s.tolist(),
        },
        'change': {
            'accuracy': float(acc_c), 'f1_weighted': float(f1_c),
            'f1_macro': float(f1m_c), 'f1_per_class': f1_per_c.tolist(),
            'confusion_matrix': cm_c.tolist(),
        },
        'best_val_combined': float(best_combined),
    }

    # 保存 checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"dual_{tag}.pth")
    torch.save({'model_state_dict': best_state, 'tag': tag, **result}, ckpt_path)
    print(f"  ✅ Test | IoU={iou_s:.4f} | AccC={acc_c:.4f} | F1mC={f1m_c:.4f}")
    print(f"     Change F1 per class: {f1_per_c}")
    print(f"     Checkpoint: {ckpt_path}")
    return result


def aggregate_results(results, out_path):
    """聚合多次运行结果，输出 mean ± std"""
    if not results:
        return
    keys_s = ['accuracy', 'f1_weighted', 'f1_macro', 'kappa', 'iou_saline']
    keys_c = ['accuracy', 'f1_weighted', 'f1_macro']

    agg = {'n_runs': len(results), 'static': {}, 'change': {}}
    for k in keys_s:
        vals = [r['static'][k] for r in results]
        agg['static'][k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                            'values': [float(v) for v in vals]}
    for k in keys_c:
        vals = [r['change'][k] for r in results]
        agg['change'][k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                            'values': [float(v) for v in vals]}

    # 打印汇总表
    print(f"\n{'='*60}")
    print(f"📊 汇总结果 (n={len(results)})")
    print(f"{'='*60}")
    print(f"  静态分类:")
    for k in keys_s:
        m, s = agg['static'][k]['mean'], agg['static'][k]['std']
        print(f"    {k:15s}: {m:.4f} ± {s:.4f}")
    print(f"  变化检测:")
    for k in keys_c:
        m, s = agg['change'][k]['mean'], agg['change'][k]['std']
        print(f"    {k:15s}: {m:.4f} ± {s:.4f}")

    with open(out_path, 'w') as f:
        json.dump({'individual': results, 'aggregated': agg}, f, indent=2)
    print(f"\n✅ 结果保存到: {out_path}")


# ============================================================
#  main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='QaidamPov 双任务训练')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['seed', 'ablation', 'both', 'single'],
                        help='seed=多种子 | ablation=消融 | both=都跑 | single=单次')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_results = []

    if args.mode in ('seed', 'both'):
        print("\n" + "=" * 60)
        print("🔬 多种子统计检验 (seeds: 42, 123, 2024, λ=1.0)")
        print("=" * 60)
        seed_results = []
        for seed in [42, 123, 2024]:
            r = train_one_run(seed, 1.0, args.epochs, args.batch_size, args.lr, device)
            seed_results.append(r)
            all_results.append(r)
        aggregate_results(seed_results, os.path.join(RESULT_DIR, 'seed_results.json'))

    if args.mode in ('ablation', 'both'):
        print("\n" + "=" * 60)
        print("🔬 任务权重消融 (λ: 0.3, 0.5, 0.7, seed=42)")
        print("=" * 60)
        abl_results = []
        for lam in [0.3, 0.5, 0.7]:
            r = train_one_run(42, lam, args.epochs, args.batch_size, args.lr, device)
            abl_results.append(r)
            all_results.append(r)
        # 加上 λ=1.0 的结果（如果 seed=42 已跑过）
        if args.mode == 'both':
            # seed_results[0] 就是 seed=42, λ=1.0
            abl_results.insert(0, seed_results[0])
        else:
            r = train_one_run(42, 1.0, args.epochs, args.batch_size, args.lr, device)
            abl_results.insert(0, r)
            all_results.append(r)
        aggregate_results(abl_results, os.path.join(RESULT_DIR, 'ablation_results.json'))

    if args.mode == 'single':
        r = train_one_run(42, 1.0, args.epochs, args.batch_size, args.lr, device)
        all_results.append(r)
        with open(os.path.join(RESULT_DIR, 'single_result.json'), 'w') as f:
            json.dump(r, f, indent=2)

    # 全部结果汇总
    if len(all_results) > 1:
        aggregate_results(all_results, os.path.join(RESULT_DIR, 'all_dual_results.json'))

    print(f"\n🎉 全部训练完成！")


if __name__ == '__main__':
    main()
