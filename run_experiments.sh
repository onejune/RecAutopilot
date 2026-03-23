#!/bin/bash
# RecAutoPilot 批量实验脚本
# 串行执行避免 Spark 端口冲突

cd /mnt/workspace/open_research/rec-autopilot
mkdir -p log

export PYTHONPATH="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python:./src:$PYTHONPATH"
export PYSPARK_PYTHON=/root/anaconda3/envs/spore/bin/python
export PYSPARK_DRIVER_PYTHON=/root/anaconda3/envs/spore/bin/python

PYTHON=/root/anaconda3/envs/spore/bin/python

echo "=========================================="
echo "RecAutoPilot 批量实验开始"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 实验1: Baseline
echo ""
echo "[1/3] 实验1: baseline_v2 (标准 WideDeep)"
echo "开始时间: $(date '+%H:%M:%S')"
cp ./conf/experiments/exp1_baseline/readme ./readme 2>/dev/null || true
$PYTHON src/autopilot_runner.py \
    --base_conf ./conf/base.yaml \
    --exp_conf ./conf/experiments/exp1_baseline.yaml \
    --name baseline_v2 \
    --hypothesis "Baseline on v16 data (2026)" \
    2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR"
echo "结束时间: $(date '+%H:%M:%S')"

# 实验2: Deeper DNN
echo ""
echo "[2/3] 实验2: deeper_dnn (更深网络)"
echo "开始时间: $(date '+%H:%M:%S')"
cp ./conf/experiments/exp2_deeper/readme ./readme 2>/dev/null || true
$PYTHON src/autopilot_runner.py \
    --base_conf ./conf/base.yaml \
    --exp_conf ./conf/experiments/exp2_deeper.yaml \
    --name deeper_dnn \
    --hypothesis "Deeper DNN: [1024,512,256,128]" \
    2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR"
echo "结束时间: $(date '+%H:%M:%S')"

# 实验3: Wider + Larger Embedding
echo ""
echo "[3/3] 实验3: wider_emb16 (更宽网络+大embedding)"
echo "开始时间: $(date '+%H:%M:%S')"
cp ./conf/experiments/exp3_wider/readme ./readme 2>/dev/null || true
$PYTHON src/autopilot_runner.py \
    --base_conf ./conf/base.yaml \
    --exp_conf ./conf/experiments/exp3_wider.yaml \
    --name wider_emb16 \
    --hypothesis "Wider DNN + emb_size=16 + lr=5e-6" \
    2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR"
echo "结束时间: $(date '+%H:%M:%S')"

echo ""
echo "=========================================="
echo "全部实验完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 显示 leaderboard
echo ""
echo "📊 Leaderboard:"
cat ./leaderboard.json 2>/dev/null || echo "No leaderboard yet"
