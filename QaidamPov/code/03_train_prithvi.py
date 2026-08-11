#!/usr/bin/env python3
"""
QaidamPov - 03 Prithvi-EO + LoRA 双任务训练
核心：使用 timm ViT（非 CustomViT），确保 LoRA 正确应用到 qkv 和 proj 层
"""
import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# ========== 必须有 timm ==========
try:
    import timm
except ImportError:
    print("❌ timm 未安装！请先运行: pip install timm")
    print("   本脚本必须使用 timm ViT，不支持 CustomViT fallback")
    sys.exit(1)

# ========== 项目路径 ==========
PROJECT = "/root/autodl-tmp/QaidamPov"
DATA_DIR = os.path.join(PROJECT, "data", "processed_dual")
CKPT_DIR = os.path.join(PROJECT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT, "results")

# Prithvi checkpoint 搜索路径
PRITHVI_SEARCH_PATHS = [
    "/root/autodl-tmp/qaidam-salinization-project/models/prithvi-100M",
    "/root/autodl-tmp/qaidam-salinization-project/models/Prithvi-100M",
    "/root/autodl-tmp/prithvi-100M",
    "/root/autodl-tmp/models/prithvi-100M",
]


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
#  数据集（与 02_train_dual.py 完全一致）
# ============================================================
class DualDataset(Dataset):
    """双任务 patch 数据集"""
    def __init__(self, x_path, ys_path, yc_path, mean=None, std=None, augment=True):
        self.x = np.load(x_path).astype(np.float32)
        self.ys = np.load(ys_path).astype(np.int64)
        self.yc = np.load(yc_path).astype(np.int64)
        self.augment = augment

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
#  LoRA
# ============================================================
class LoRALinear(nn.Module):
    """LoRA: 冻结原始 Linear + 低秩矩阵 A,B"""
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA 低秩矩阵
        d_in = original_linear.in_features
        d_out = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(d_in, r))
        self.lora_B = nn.Parameter(torch.zeros(r, d_out))
        self.dropout = nn.Dropout(0.1)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original(x) + self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling


def apply_lora_to_model(model, r=8, alpha=16, target_modules=['qkv', 'proj']):
    """将 LoRA 应用到 ViT 的 attention 层（timm ViT 的 qkv 和 proj 是 nn.Linear）"""
    lora_count = 0
    lora_names = []
    for name, module in model.named_modules():
        if any(t in name.lower() for t in target_modules):
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(module, r=r, alpha=alpha)
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], lora_layer)
                lora_count += 1
                lora_names.append(name)

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA 应用: {lora_count} 个层")
    for n in lora_names:
        print(f"    - {n}")
    print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# ============================================================
#  ViT 分割解码器
# ============================================================
class ViTSegDecoder(nn.Module):
    """ViT patch tokens → 像素级分割"""
    def __init__(self, embed_dim=768, img_size=128, patch_size=16, num_classes=2):
        super().__init__()
        self.img_size = img_size
        self.num_patches_side = img_size // patch_size  # 8

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
        # patch_tokens: (B, num_patches, embed_dim) e.g. (B, 64, 768)
        B = patch_tokens.shape[0]
        H = W = self.num_patches_side
        x = patch_tokens.transpose(1, 2).reshape(B, -1, H, W)
        x = self.decoder(x)
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return x


# ============================================================
#  Prithvi 双任务模型
# ============================================================
class PrithviDualTaskModel(nn.Module):
    """Prithvi-EO (timm ViT) + LoRA + 双任务解码器"""
    def __init__(self, img_size=128, in_channels=6, lora_r=8, lora_alpha=16):
        super().__init__()
        self.img_size = img_size

        # 1. 创建 timm ViT（不用 CustomViT！）
        self.encoder = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )

        # 2. 应用 LoRA（必须在创建 encoder 后、加载 checkpoint 前应用）
        apply_lora_to_model(self.encoder, r=lora_r, alpha=lora_alpha)

        # 3. 双任务解码器
        self.decoder_static = ViTSegDecoder(embed_dim=768, img_size=img_size, num_classes=2)
        self.decoder_change = ViTSegDecoder(embed_dim=768, img_size=img_size, num_classes=4)

    def forward(self, x):
        features = self.encoder.forward_features(x)  # (B, 65, 768)
        patch_tokens = features[:, 1:]  # 去掉 cls token → (B, 64, 768)
        out_s = self.decoder_static(patch_tokens)
        out_c = self.decoder_change(patch_tokens)
        return out_s, out_c


# ============================================================
#  Prithvi checkpoint 加载
# ============================================================
def find_prithvi_checkpoint():
    """搜索 Prithvi checkpoint"""
    for p in PRITHVI_SEARCH_PATHS:
        if os.path.isdir(p):
            for fname in ['Prithvi_100M.pt', 'Prithvi_100M.pth', 'prithvi_100M.pt']:
                fpath = os.path.join(p, fname)
                if os.path.exists(fpath):
                    print(f"  📦 找到 Prithvi checkpoint: {fpath}")
                    return fpath
    return None


def interpolate_pos_embed(pos_embed, new_size=8):
    """将 pos_embed 插值到新的 patch grid 大小
    pos_embed: (1, N+1, C)  N+1 = cls_token + patch_tokens
    new_size: 新的 patch grid 边长 (img_size/patch_size)

    支持两种情况：
    1. 标准 2D ViT: N = H*W (完全平方数)
    2. 多时相 ViT (如 Prithvi): N = T*H*W (3帧×14×14=588)
    """
    cls_token = pos_embed[:, :1, :]  # (1, 1, C)
    patch_pos = pos_embed[:, 1:, :]  # (1, N, C)

    N = patch_pos.shape[1]
    C = patch_pos.shape[2]

    # 检查是否是标准 2D ViT（N 是完全平方数）
    old_size = int(N ** 0.5)

    if old_size * old_size == N:
        # ===== 标准 2D ViT =====
        if old_size == new_size:
            return pos_embed
        print(f"  📐 pos_embed 插值 (2D): {old_size}×{old_size} → {new_size}×{new_size}")
        patch_pos = patch_pos.reshape(1, old_size, old_size, C).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, C)
    else:
        # ===== 多时相 ViT (如 Prithvi-100M) =====
        # 尝试找到 temporal factor: N = T * H * W
        temporal = None
        old_spatial = None
        for T in range(1, 20):
            spatial_n = N // T
            s = int(spatial_n ** 0.5)
            if s * s == spatial_n and T * s * s == N:
                temporal = T
                old_spatial = s
                break

        if temporal is None:
            print(f"  ⚠️ 无法解析 pos_embed 形状: N={N}，跳过 pos_embed 加载")
            print(f"     将使用随机初始化的 pos_embed")
            return None

        print(f"  📐 检测到 temporal pos_embed: T={temporal}, spatial={old_spatial}×{old_spatial}")
        print(f"  📐 提取中间帧 → 插值: {old_spatial}×{old_spatial} → {new_size}×{new_size}")

        # 提取中间帧的空间 pos_embed
        mid_t = temporal // 2
        try:
            patch_pos = patch_pos.reshape(1, temporal, old_spatial, old_spatial, C)
            patch_pos = patch_pos[0, mid_t]  # (old_spatial, old_spatial, C)
            patch_pos = patch_pos.unsqueeze(0)  # (1, old_spatial, old_spatial, C)
        except RuntimeError:
            print(f"  ⚠️ (T,H,W) 重排失败，尝试 (H,W,T) 重排...")
            patch_pos_raw = pos_embed[:, 1:, :]
            patch_pos = patch_pos_raw.reshape(1, old_spatial, old_spatial, temporal, C)
            patch_pos = patch_pos[0, :, :, mid_t, :]
            patch_pos = patch_pos.unsqueeze(0)

        if old_spatial == new_size:
            patch_pos = patch_pos.reshape(1, new_size * new_size, C)
        else:
            patch_pos = patch_pos.permute(0, 3, 1, 2)  # (1, C, H, W)
            patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, C)

    # 拼回 cls_token
    new_pos_embed = torch.cat([cls_token, patch_pos], dim=1)
    print(f"  ✅ pos_embed 处理完成: {pos_embed.shape} → {new_pos_embed.shape}")
    return new_pos_embed


def load_prithvi_weights(model, ckpt_path):
    """加载 Prithvi 预训练权重到 timm ViT encoder
    处理以下差异：
    1. 多时相 pos_embed (3帧×14×14=588) → 单帧空间 pos_embed
    2. patch_embed 通道数差异 (Prithvi 18ch vs 我们 6ch)
    3. key 前缀差异
    """
    print(f"  📂 加载 Prithvi checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 提取 state_dict
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # 打印前 10 个 key，帮助调试
    all_keys = list(state_dict.keys())
    print(f"  📋 Checkpoint 共 {len(all_keys)} 个 key，前 10 个:")
    for k in all_keys[:10]:
        print(f"     {k}: {state_dict[k].shape}")

    # 去掉前缀，统一 key 格式
    new_state_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace('encoder.', '').replace('backbone.', '').replace('module.', '')
        new_state_dict[k_clean] = v

    # ===== 处理 patch_embed 通道数差异 =====
    pe_key = 'patch_embed.proj.weight'
    if pe_key in new_state_dict:
        pe_weight = new_state_dict[pe_key]
        if pe_weight.shape[1] != 6:
            old_ch = pe_weight.shape[1]
            print(f"  🔧 patch_embed 通道数: {old_ch} → 6")
            new_state_dict[pe_key] = pe_weight[:, :6, :, :].clone()
            print(f"     ✅ 已提取第一帧的6通道 (shape: {new_state_dict[pe_key].shape})")

    # ===== 处理 pos_embed（多时相 → 单帧） =====
    if 'pos_embed' in new_state_dict:
        pe = new_state_dict['pos_embed']
        expected_tokens = 1 + (model.img_size // 16) ** 2
        if pe.shape[1] != expected_tokens:
            print(f"  ⚠️ pos_embed 形状不匹配: {pe.shape} vs 期望 ({1}, {expected_tokens}, {pe.shape[2]})")
            result = interpolate_pos_embed(pe, new_size=model.img_size // 16)
            if result is not None:
                new_state_dict['pos_embed'] = result
            else:
                del new_state_dict['pos_embed']
                print(f"  ⚠️ 跳过 pos_embed，将使用随机初始化")

    # ===== 加载到 encoder =====
    encoder_state = model.encoder.state_dict()
    matched_keys = []
    skipped_keys = []

    for k, v in new_state_dict.items():
        if k in encoder_state:
            if encoder_state[k].shape == v.shape:
                encoder_state[k] = v
                matched_keys.append(k)
            else:
                skipped_keys.append(f"{k}: ckpt{v.shape} vs model{encoder_state[k].shape}")

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)

    print(f"  ✅ Prithvi 权重加载: {len(matched_keys)} 个 key 匹配")
    print(f"     Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    if skipped_keys:
        print(f"  ⚠️ 形状不匹配跳过的 key ({len(skipped_keys)} 个):")
        for s in skipped_keys[:5]:
            print(f"     - {s}")
        if len(skipped_keys) > 5:
            print(f"     ... 还有 {len(skipped_keys)-5} 个")

    if len(matched_keys) < 10:
        print(f"  ⚠️ 匹配的 key 太少 ({len(matched_keys)})，可能 key 前缀不匹配")
        print(f"  💡 列出 checkpoint 的 key 供参考:")
        for k in all_keys[:20]:
            print(f"     ckpt: {k}")
        print(f"  💡 列出 model encoder 的 key 供参考:")
        model_keys = list(encoder_state.keys())
        for k in model_keys[:20]:
            print(f"     model: {k}")
    else:
        print(f"  🎉 成功加载 {len(matched_keys)} 个预训练权重！")


# ============================================================
#  训练 + 评估
# ============================================================
def train_one_run(seed, epochs, batch_size, lr, lambda_change, lora_r, lora_alpha, no_pretrained, device):
    """单次训练 + 测试集评估"""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"🌍 Prithvi+LoRA | Seed={seed} | epochs={epochs} | lr={lr}")
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
    model = PrithviDualTaskModel(
        img_size=128, in_channels=6, lora_r=lora_r, lora_alpha=lora_alpha
    ).to(device)

    # 加载 Prithvi 预训练权重
    if not no_pretrained:
        ckpt_path = find_prithvi_checkpoint()
        if ckpt_path:
            load_prithvi_weights(model, ckpt_path)
        else:
            print("  ⚠️ 未找到 Prithvi checkpoint，使用 timm ViT 从头训练")
            print("  💡 仍然是有效贡献：使用 ViT+LoRA 架构进行盐渍化监测")

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

    # 只训练 LoRA + decoder 参数
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
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
        'seed': seed,
        'model': 'Prithvi+LoRA',
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
    ckpt_path = os.path.join(CKPT_DIR, f"prithvi_lora_seed{seed}.pth")
    torch.save({'model_state_dict': best_state, 'tag': f'prithvi_seed{seed}', **result}, ckpt_path)
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

    print(f"\n{'='*60}")
    print(f"📊 Prithvi+LoRA 汇总结果 (n={len(results)})")
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
    parser = argparse.ArgumentParser(description='QaidamPov Prithvi+LoRA 训练')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lambda-change', type=float, default=1.0)
    parser.add_argument('--no-pretrained', action='store_true', help='不加载 Prithvi checkpoint')
    parser.add_argument('--seeds', type=str, default='42,123,2024', help='训练种子，逗号分隔')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    print(f"   LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"   Seeds: {seeds}")
    print(f"   Pretrained: {not args.no_pretrained}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_results = []
    for seed in seeds:
        r = train_one_run(
            seed, args.epochs, args.batch_size, args.lr,
            args.lambda_change, args.lora_r, args.lora_alpha,
            args.no_pretrained, device
        )
        all_results.append(r)

    aggregate_results(all_results, os.path.join(RESULT_DIR, 'prithvi_results.json'))
    print(f"\n🎉 Prithvi+LoRA 训练完成！")


if __name__ == '__main__':
    main()
