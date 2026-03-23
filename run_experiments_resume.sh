#!/bin/bash
# RecAutoPilot 续跑脚本 - 从 baseline_v2 验证开始

cd /mnt/workspace/open_research/rec-autopilot
mkdir -p log

export PYTHONPATH="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python:./src:$PYTHONPATH"
export PYSPARK_PYTHON=/root/anaconda3/envs/spore/bin/python
export PYSPARK_DRIVER_PYTHON=/root/anaconda3/envs/spore/bin/python

PYTHON=/root/anaconda3/envs/spore/bin/python

echo "=========================================="
echo "RecAutoPilot 续跑开始"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 步骤1: baseline_v2 验证（使用已有模型 model_2026-03-08）
echo ""
echo "[1/3] baseline_v2 验证 (model_2026-03-08, val_date=2026-03-11)"
echo "开始时间: $(date '+%H:%M:%S')"
$PYTHON src/autopilot_runner.py \
    --base_conf ./conf/base.yaml \
    --exp_conf ./conf/experiments/phase1_baseline/exp1_baseline.yaml \
    --name baseline_v2 \
    --hypothesis "Baseline on v16 data (2026)" \
    --validation True \
    --model_date 2026-03-08 \
    --sample_date 2026-03-11 \
    2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR"
echo "结束时间: $(date '+%H:%M:%S')"

# 步骤2: Deeper DNN 完整训练+验证
echo ""
echo "[2/3] 实验2: deeper_dnn (更深网络)"
echo "开始时间: $(date '+%H:%M:%S')"
$PYTHON src/autopilot_runner.py \
    --base_conf ./conf/base.yaml \
    --exp_conf ./conf/experiments/phase1_baseline/exp2_deeper.yaml \
    --name deeper_dnn \
    --hypothesis "Deeper DNN: [1024,512,256,128]" \
    2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR"
echo "结束时间: $(date '+%H:%M:%S')"

# 步骤3: Wider + Larger Embedding 完整训练+验证
echo ""
echo "[3/3] 实验3: wider_emb16 (更宽网络+大embedding)"
echo "开始时间: $(date '+%H:%M:%S')"
$PYTHON src/autopilot_runner.py \
    --base_conf ./conf/base.yaml \
    --exp_conf ./conf/experiments/phase1_baseline/exp3_wider.yaml \
    --name wider_emb16 \
    --hypothesis "Wider DNN + emb_size=16 + lr=5e-6" \
    2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR"
echo "结束时间: $(date '+%H:%M:%S')"

echo ""
echo "=========================================="
echo "全部完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

echo ""
echo "📊 Leaderboard:"
cat ./leaderboard.json 2>/dev/null || echo "No leaderboard yet"
