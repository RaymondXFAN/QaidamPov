#!/bin/bash
set -e

PROJECT="/root/autodl-tmp/QaidamPov"
cd $PROJECT

echo "=========================================="
echo "🚀 QaidamPov 全流程运行"
echo "=========================================="

# Step 1: 环境检查
echo ""
echo "[1/4] 🔍 环境检查 + 数据链接"
python 01_setup.py

# Step 2: 双任务训练（多种子 + 消融）
echo ""
echo "[2/4] 🏋️ 双任务模型训练 (CNN DualTask)"
echo "  预计耗时: ~30分钟 (6次训练 × 30 epochs)"
python 02_train_dual.py --mode both

# Step 3: Prithvi+LoRA训练
echo ""
echo "[3/4] 🌍 Prithvi-EO + LoRA 训练"
echo "  预计耗时: ~60分钟 (3次训练 × 50 epochs)"
python 03_train_prithvi.py

# Step 4: 统一评估
echo ""
echo "[4/4] 📊 统一评估 + 生成对比表"
python 04_eval_all.py

echo ""
echo "=========================================="
echo "🎉 全部完成！"
echo "  结果: $PROJECT/results/"
echo "  模型: $PROJECT/checkpoints/"
echo "=========================================="
