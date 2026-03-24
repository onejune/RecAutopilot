#!/bin/bash
# Phase 4: 超参优化实验脚本
# 7个实验，分3组并行（每组串行，组间错开30秒）
# 组1: emb12 → emb16 → emb24
# 组2: lr_5e5 → lr_1e4
# 组3: dropout03 → dropout07

cd /mnt/workspace/open_research/rec-autopilot
mkdir -p log

export PYTHONPATH="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python:./src:$PYTHONPATH"
export PYSPARK_PYTHON=/root/anaconda3/envs/spore/bin/python
export PYSPARK_DRIVER_PYTHON=/root/anaconda3/envs/spore/bin/python
PYTHON=/root/anaconda3/envs/spore/bin/python

echo "============================================================"
echo "Phase 4 超参优化实验开始 (3组并行)"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 组1: embedding_size 探索（串行）
run_group1() {
    for exp in hp_emb12 hp_emb16 hp_emb24; do
        echo "[Group 1] $exp 开始 $(date '+%H:%M:%S')"
        $PYTHON src/autopilot_runner.py \
            --base_conf ./conf/base.yaml \
            --exp_conf ./conf/experiments/phase4_hyperparam/${exp}.yaml \
            --output_dir ./output/${exp} \
            2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/${exp}.log
        echo "[Group 1] $exp 完成 $(date '+%H:%M:%S')"
    done
}

# 组2: learning_rate 探索（串行）
run_group2() {
    for exp in hp_lr_5e5 hp_lr_1e4; do
        echo "[Group 2] $exp 开始 $(date '+%H:%M:%S')"
        $PYTHON src/autopilot_runner.py \
            --base_conf ./conf/base.yaml \
            --exp_conf ./conf/experiments/phase4_hyperparam/${exp}.yaml \
            --output_dir ./output/${exp} \
            2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/${exp}.log
        echo "[Group 2] $exp 完成 $(date '+%H:%M:%S')"
    done
}

# 组3: dropout 探索（串行）
run_group3() {
    for exp in hp_dropout03 hp_dropout07; do
        echo "[Group 3] $exp 开始 $(date '+%H:%M:%S')"
        $PYTHON src/autopilot_runner.py \
            --base_conf ./conf/base.yaml \
            --exp_conf ./conf/experiments/phase4_hyperparam/${exp}.yaml \
            --output_dir ./output/${exp} \
            2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/${exp}.log
        echo "[Group 3] $exp 完成 $(date '+%H:%M:%S')"
    done
}

# 并行启动3组，错开30秒
run_group1 &
PID1=$!

sleep 30
run_group2 &
PID2=$!

sleep 30
run_group3 &
PID3=$!

wait $PID1 && echo "[Group 1] 完成"
wait $PID2 && echo "[Group 2] 完成"
wait $PID3 && echo "[Group 3] 完成"

echo ""
echo "============================================================"
echo "所有实验完成！时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
SUCCESS=0
FAILED=()
for exp in hp_emb12 hp_emb16 hp_emb24 hp_lr_5e5 hp_lr_1e4 hp_dropout03 hp_dropout07; do
    if grep -q "实验完成" log/${exp}.log 2>/dev/null; then
        SUCCESS=$((SUCCESS+1))
    else
        FAILED+=($exp)
    fi
done
echo "✅ 成功: $SUCCESS — $(echo ${!FAILED[@]} | tr ' ' ',')"
[ ${#FAILED[@]} -gt 0 ] && echo "❌ 失败: ${FAILED[*]}"
