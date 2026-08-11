#!/usr/bin/env python3
"""补丁脚本：修复 03_train_prithvi.py 的 pos_embed 多时相问题
用法：python patch_03.py
"""
import os

TARGET = "/root/autodl-tmp/QaidamPov/03_train_prithvi.py"

# 读取原文件
with open(TARGET, 'r') as f:
    content = f.read()

# ========== 1. 替换 interpolate_pos_embed 函数 ==========
OLD_INTERPOLATE = '''def interpolate_pos_embed(pos_embed, new_size=8):
    """将 pos_embed 插值到新的 patch grid 大小
    pos_embed: (1, N+1, C)  N+1 = cls_token + patch_tokens
    new_size: 新的 patch grid 边长 (img_size/patch_size)
    """
    cls_token = pos_embed[:, :1, :]  # (1, 1, C)
    patch_pos = pos_embed[:, 1:, :]  # (1, N, C)

    N = patch_pos.shape[1]
    C = patch_pos.shape[2]
    old_size = int(N ** 0.5)

    if old_size == new_size:
        return pos_embed

    print(f"  📐 pos_embed 插值: {old_size}×{old_size} → {new_size}×{new_size}")

    # (1, N, C) → (1, old_size, old_size, C) → (1, C, old_size, old_size)
    patch_pos = patch_pos.reshape(1, old_size, old_size, C).permute(0, 3, 1, 2)
    # Bilinear 插值
    patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
    # (1, C, new_size, new_size) → (1, new_size*new_size, C)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, C)

    # 拼回 cls_token
    new_pos_embed = torch.cat([cls_token, patch_pos], dim=1)
    print(f"  ✅ pos_embed 插值完成: {pos_embed.shape} → {new_pos_embed.shape}")
    return new_pos_embed'''

NEW_INTERPOLATE = '''def interpolate_pos_embed(pos_embed, new_size=8):
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
    return new_pos_embed'''

if OLD_INTERPOLATE in content:
    content = content.replace(OLD_INTERPOLATE, NEW_INTERPOLATE)
    print("✅ interpolate_pos_embed 函数已替换")
else:
    print("⚠️ 未找到旧版 interpolate_pos_embed，可能已更新或格式不同")
    # 检查是否已经是新版
    if "多时相 ViT" in content:
        print("   检测到已包含多时相处理代码，跳过")
    else:
        print("   尝试按行号定位替换...")
        # 找到函数定义行
        lines = content.split('\n')
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if 'def interpolate_pos_embed' in line:
                start_idx = i
            if start_idx is not None and end_idx is None:
                if i > start_idx and line.startswith('def ') and 'interpolate_pos_embed' not in line:
                    end_idx = i
                    break
        if start_idx is not None and end_idx is not None:
            print(f"   找到函数位置: 行 {start_idx+1} 到 {end_idx}")
            lines[start_idx:end_idx] = NEW_INTERPOLATE.split('\n')
            content = '\n'.join(lines)
            print("✅ 按行号替换成功")
        else:
            print("❌ 无法定位函数，请手动替换")

# ========== 2. 替换 load_prithvi_weights 函数 ==========
OLD_LOAD = '''def load_prithvi_weights(model, ckpt_path):
    """加载 Prithvi 预训练权重到 timm ViT encoder"""
    print(f"  📂 加载 Prithvi checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 提取 state_dict
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # 去掉前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('encoder.', '').replace('backbone.', '')
        new_state_dict[k] = v

    # pos_embed 插值（如果需要）
    if 'pos_embed' in new_state_dict:
        pe = new_state_dict['pos_embed']
        # 期望的 pos_embed shape: (1, 65, 768) for img_size=128
        expected_tokens = 1 + (model.img_size // 16) ** 2  # 1 + 64 = 65
        if pe.shape[1] != expected_tokens:
            print(f"  ⚠️ pos_embed 形状不匹配: {pe.shape[1]} vs 期望 {expected_tokens}")
            new_state_dict['pos_embed'] = interpolate_pos_embed(pe, new_size=model.img_size // 16)

    # 加载到 encoder（strict=False，因为 LoRA 替换后 key 会变）
    # 先只尝试匹配 encoder 的 key
    encoder_state = model.encoder.state_dict()
    matched_keys = []
    for k, v in new_state_dict.items():
        if k in encoder_state and encoder_state[k].shape == v.shape:
            encoder_state[k] = v
            matched_keys.append(k)

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"  ✅ Prithvi 权重加载: {len(matched_keys)} 个 key 匹配")
    print(f"     Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    if len(matched_keys) < 10:
        print(f"  ⚠️ 匹配的 key 太少，可能权重未正确加载")
        print(f"  💡 但 LoRA + 从头训练仍然有效")'''

NEW_LOAD = '''def load_prithvi_weights(model, ckpt_path):
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
        print(f"  🎉 成功加载 {len(matched_keys)} 个预训练权重！")'''

if OLD_LOAD in content:
    content = content.replace(OLD_LOAD, NEW_LOAD)
    print("✅ load_prithvi_weights 函数已替换")
else:
    print("⚠️ 未找到旧版 load_prithvi_weights，可能已更新或格式不同")
    if "patch_embed 通道数差异" in content:
        print("   检测到已包含通道数处理代码，跳过")
    else:
        print("   尝试按行号定位替换...")
        lines = content.split('\n')
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if 'def load_prithvi_weights' in line:
                start_idx = i
            if start_idx is not None and end_idx is None:
                if i > start_idx and line.startswith('def ') and 'load_prithvi_weights' not in line:
                    end_idx = i
                    break
        if start_idx is not None and end_idx is not None:
            print(f"   找到函数位置: 行 {start_idx+1} 到 {end_idx}")
            lines[start_idx:end_idx] = NEW_LOAD.split('\n')
            content = '\n'.join(lines)
            print("✅ 按行号替换成功")
        else:
            print("❌ 无法定位函数，请手动替换")

# 写回文件
with open(TARGET, 'w') as f:
    f.write(content)

print(f"\n✅ 补丁完成！文件: {TARGET}")

# 验证
with open(TARGET, 'r') as f:
    content = f.read()
if "多时相 ViT" in content and "patch_embed 通道数差异" in content:
    print("✅ 验证通过：包含多时相处理 + 通道数处理")
else:
    print("❌ 验证失败：补丁可能未完全生效")
