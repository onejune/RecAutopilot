#!/bin/bash
# Phase 5: 组合超参实验脚本
# 4个实验，2组并行（每组2个串行，错开30秒）
# 组1: comb_lr5e5_drop03 → comb_lr5e5_drop03_emb16
# 组2: comb_lr5e5_emb16 → comb_lr1e4_drop03

cd /mnt/workspace/open_research/rec-autopilot
mkdir -p log

export PYTHONPATH="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python:./src:$PYTHONPATH"
export PYSPARK_PYTHON=/root/anaconda3/envs/spore/bin/python
export PYSPARK_DRIVER_PYTHON=/root/anaconda3/envs/spore/bin/python
PYTHON=/root/anaconda3/envs/spore/bin/python

echo "============================================================"
echo "Phase 5 组合超参实验开始 (2组并行)"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 组1
run_group1() {
    for exp in comb_lr5e5_drop03 comb_lr5e5_drop03_emb16; do
        echo "[Group 1] $exp 开始 $(date '+%H:%M:%S')"
        $PYTHON src/autopilot_runner.py \
            --base_conf ./conf/base.yaml \
            --exp_conf ./conf/experiments/phase5_combined/${exp}.yaml \
            --output_dir ./output/${exp} \
            2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/${exp}.log
        echo "[Group 1] $exp 完成 $(date '+%H:%M:%S')"
    done
}

# 组2
run_group2() {
    for exp in comb_lr5e5_emb16 comb_lr1e4_drop03; do
        echo "[Group 2] $exp 开始 $(date '+%H:%M:%S')"
        $PYTHON src/autopilot_runner.py \
            --base_conf ./conf/base.yaml \
            --exp_conf ./conf/experiments/phase5_combined/${exp}.yaml \
            --output_dir ./output/${exp} \
            2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/${exp}.log
        echo "[Group 2] $exp 完成 $(date '+%H:%M:%S')"
    done
}

run_group1 &
PID1=$!

sleep 30
run_group2 &
PID2=$!

wait $PID1 && echo "[Group 1] 完成"
wait $PID2 && echo "[Group 2] 完成"

echo ""
echo "============================================================"
echo "所有实验完成！时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
SUCCESS=0
FAILED=()
for exp in comb_lr5e5_drop03 comb_lr5e5_drop03_emb16 comb_lr5e5_emb16 comb_lr1e4_drop03; do
    if grep -q "实验完成" log/${exp}.log 2>/dev/null; then
        SUCCESS=$((SUCCESS+1))
    else
        FAILED+=($exp)
    fi
done
echo "✅ 成功: $SUCCESS"
[ ${#FAILED[@]} -gt 0 ] && echo "❌ 失败: ${FAILED[*]}"
