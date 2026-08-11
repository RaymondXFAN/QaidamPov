#!/usr/bin/env python3
"""
QaidamPov - 04 统一评估所有模型
加载 DualTaskModel 和 PrithviDualTaskModel 的 checkpoint，在测试集上评估，生成对比表格
"""
import os
import sys
import json
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# ========== 项目路径 ==========
PROJECT = "/root/autodl-tmp/QaidamPov"
DATA_DIR = os.path.join(PROJECT, "data", "processed_dual")
CKPT_DIR = os.path.join(PROJECT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT, "results")


# ============================================================
#  DualTaskModel（与 02_train_dual.py 完全一致）
# ============================================================
class DualTaskModel(nn.Module):
    """双任务 CNN"""
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
#  PrithviDualTaskModel（与 03_train_prithvi.py 完全一致）
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        d_in = original_linear.in_features
        d_out = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(d_in, r))
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original(x) + self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling


def apply_lora_to_model(model, r=8, alpha=16, target_modules=['qkv', 'proj']):
    for name, module in model.named_modules():
        if any(t in name.lower() for t in target_modules) and isinstance(module, nn.Linear):
            lora_layer = LoRALinear(module, r=r, alpha=alpha)
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_layer)
    return model


class ViTSegDecoder(nn.Module):
    def __init__(self, embed_dim=768, img_size=128, patch_size=16, num_classes=2):
        super().__init__()
        self.img_size = img_size
        self.num_patches_side = img_size // patch_size
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, patch_tokens):
        B = patch_tokens.shape[0]
        H = W = self.num_patches_side
        x = patch_tokens.transpose(1, 2).reshape(B, -1, H, W)
        x = self.decoder(x)
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return x


class PrithviDualTaskModel(nn.Module):
    def __init__(self, img_size=128, in_channels=6, lora_r=8, lora_alpha=16):
        super().__init__()
        import timm
        self.img_size = img_size
        self.encoder = timm.create_model(
            'vit_small_patch16_224', pretrained=False, img_size=img_size,
            in_chans=in_channels, num_classes=0, embed_dim=768, depth=12, num_heads=12,
        )
        apply_lora_to_model(self.encoder, r=lora_r, alpha=lora_alpha)
        self.decoder_static = ViTSegDecoder(embed_dim=768, img_size=img_size, num_classes=2)
        self.decoder_change = ViTSegDecoder(embed_dim=768, img_size=img_size, num_classes=4)

    def forward(self, x):
        features = self.encoder.forward_features(x)
        patch_tokens = features[:, 1:]
        return self.decoder_static(patch_tokens), self.decoder_change(patch_tokens)


# ============================================================
#  评估函数
# ============================================================
def evaluate_model(model, test_loader, device):
    """评估模型，返回指标字典"""
    model.eval()
    ps_list, pc_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            os_, oc = model(x)
            ps_list.append(os_.argmax(1).cpu().numpy())
            pc_list.append(oc.argmax(1).cpu().numpy())
    pred_s = np.concatenate(ps_list)
    pred_c = np.concatenate(pc_list)
    return pred_s, pred_c


def compute_metrics(test_ys, test_yc, pred_s, pred_c):
    """计算所有指标"""
    # 静态分类
    acc_s = accuracy_score(test_ys.flatten(), pred_s.flatten())
    f1_s = f1_score(test_ys.flatten(), pred_s.flatten(), average='weighted')
    f1m_s = f1_score(test_ys.flatten(), pred_s.flatten(), average='macro')
    kappa_s = cohen_kappa_score(test_ys.flatten(), pred_s.flatten())
    cm_s = confusion_matrix(test_ys.flatten(), pred_s.flatten())
    tp, fp, fn = cm_s[1, 1], cm_s[0, 1], cm_s[1, 0]
    iou_s = tp / (tp + fp + fn + 1e-8)

    # 变化检测
    acc_c = accuracy_score(test_yc.flatten(), pred_c.flatten())
    f1_c = f1_score(test_yc.flatten(), pred_c.flatten(), average='weighted')
    f1m_c = f1_score(test_yc.flatten(), pred_c.flatten(), average='macro')
    f1_per_c = f1_score(test_yc.flatten(), pred_c.flatten(), average=None, labels=[0, 1, 2, 3])

    return {
        'static': {
            'accuracy': float(acc_s), 'f1_weighted': float(f1_s),
            'f1_macro': float(f1m_s), 'kappa': float(kappa_s),
            'iou_saline': float(iou_s),
        },
        'change': {
            'accuracy': float(acc_c), 'f1_weighted': float(f1_c),
            'f1_macro': float(f1m_c), 'f1_per_class': f1_per_c.tolist(),
        },
    }


# ============================================================
#  生成对比表格
# ============================================================
def generate_comparison_table(all_results):
    """生成 Markdown 对比表格"""
    # 按模型分组
    models = {}
    for r in all_results:
        m = r['model']
        if m not in models:
            models[m] = []
        models[m].append(r)

    lines = []
    lines.append("| Model | Static Acc | Static IoU | Static F1 | Change Acc | Change F1m |")
    lines.append("|-------|-----------|-----------|-----------|-----------|-----------|")

    for model_name, results in models.items():
        # 每个种子一行
        for r in results:
            s = r['static']
            c = r['change']
            seed = r.get('seed', '?')
            lines.append(
                f"| {model_name} (seed{seed}) | {s['accuracy']:.3f} | {s['iou_saline']:.3f} | "
                f"{s['f1_weighted']:.3f} | {c['accuracy']:.3f} | {c['f1_macro']:.3f} |"
            )

        # mean±std 行
        if len(results) > 1:
            keys_s = ['accuracy', 'iou_saline', 'f1_weighted']
            keys_c = ['accuracy', 'f1_macro']
            vals = []
            for k in keys_s:
                v = [r['static'][k] for r in results]
                vals.append(f"{np.mean(v):.3f}±{np.std(v):.3f}")
            for k in keys_c:
                v = [r['change'][k] for r in results]
                vals.append(f"{np.mean(v):.3f}±{np.std(v):.3f}")
            lines.append(
                f"| **{model_name} (mean±std)** | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} |"
            )

    return '\n'.join(lines)


# ============================================================
#  main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='QaidamPov 统一评估')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--no-prithvi', action='store_true', help='跳过 Prithvi 模型评估')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")

    # 加载测试数据
    print("\n📊 加载测试数据...")
    test_x = np.load(os.path.join(DATA_DIR, 'test_x.npy')).astype(np.float32)
    test_ys = np.load(os.path.join(DATA_DIR, 'test_y_static.npy')).astype(np.int64)
    test_yc = np.load(os.path.join(DATA_DIR, 'test_y_change.npy')).astype(np.int64)

    # 用训练集的 mean/std 归一化
    train_x = np.load(os.path.join(DATA_DIR, 'train_x.npy')).astype(np.float32)
    mean = train_x.mean(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
    std = (train_x.std(axis=(0, 2, 3)) + 1e-8).reshape(1, -1, 1, 1)
    test_x_n = (test_x - mean) / std

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_x_n), torch.LongTensor(test_ys), torch.LongTensor(test_yc)),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"  测试集: {len(test_x)} 样本")

    all_results = []

    # ========== 评估 DualTaskModel ==========
    print("\n" + "=" * 60)
    print("🏋️ 评估 DualTaskModel (CNN)")
    print("=" * 60)
    dual_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'dual_*.pth')))
    if dual_ckpts:
        for ckpt_path in dual_ckpts:
            ckpt_name = os.path.basename(ckpt_path)
            print(f"\n  📂 {ckpt_name}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            model = DualTaskModel(in_channels=6, num_classes=2, num_change_classes=4).to(device)
            model.load_state_dict(ckpt['model_state_dict'])

            pred_s, pred_c = evaluate_model(model, test_loader, device)
            metrics = compute_metrics(test_ys, test_yc, pred_s, pred_c)
            result = {
                'model': 'DualTask',
                'seed': ckpt.get('seed', '?'),
                'lambda_change': ckpt.get('lambda_change', '?'),
                **metrics,
            }
            all_results.append(result)
            print(f"    ✅ Static IoU={metrics['static']['iou_saline']:.4f} | "
                  f"Change Acc={metrics['change']['accuracy']:.4f} | "
                  f"Change F1m={metrics['change']['f1_macro']:.4f}")
    else:
        print("  ⚠️ 未找到 dual_*.pth checkpoint")

    # ========== 评估 PrithviDualTaskModel ==========
    if not args.no_prithvi:
        print("\n" + "=" * 60)
        print("🌍 评估 PrithviDualTaskModel (ViT+LoRA)")
        print("=" * 60)
        prithvi_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'prithvi_lora_*.pth')))
        if prithvi_ckpts:
            try:
                import timm
                for ckpt_path in prithvi_ckpts:
                    ckpt_name = os.path.basename(ckpt_path)
                    print(f"\n  📂 {ckpt_name}")
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                    model = PrithviDualTaskModel(
                        img_size=128, in_channels=6, lora_r=8, lora_alpha=16
                    ).to(device)
                    model.load_state_dict(ckpt['model_state_dict'])

                    pred_s, pred_c = evaluate_model(model, test_loader, device)
                    metrics = compute_metrics(test_ys, test_yc, pred_s, pred_c)
                    result = {
                        'model': 'Prithvi+LoRA',
                        'seed': ckpt.get('seed', '?'),
                        **metrics,
                    }
                    all_results.append(result)
                    print(f"    ✅ Static IoU={metrics['static']['iou_saline']:.4f} | "
                          f"Change Acc={metrics['change']['accuracy']:.4f} | "
                          f"Change F1m={metrics['change']['f1_macro']:.4f}")
            except ImportError:
                print("  ⚠️ timm 未安装，跳过 Prithvi 评估")
        else:
            print("  ⚠️ 未找到 prithvi_lora_*.pth checkpoint")

    # ========== 生成对比表格 ==========
    if all_results:
        print("\n" + "=" * 60)
        print("📊 对比表格")
        print("=" * 60)
        table = generate_comparison_table(all_results)
        print(table)

        # 保存结果
        with open(os.path.join(RESULT_DIR, 'eval_all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        with open(os.path.join(RESULT_DIR, 'comparison_table.md'), 'w') as f:
            f.write("# QaidamPov 模型对比结果\n\n")
            f.write(table + "\n")
        print(f"\n✅ 结果保存到: {RESULT_DIR}/eval_all_results.json")
        print(f"✅ 表格保存到: {RESULT_DIR}/comparison_table.md")
    else:
        print("\n⚠️ 没有可评估的模型")

    print("\n🎉 评估完成！")


if __name__ == '__main__':
    main()
