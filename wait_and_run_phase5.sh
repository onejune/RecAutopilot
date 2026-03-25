#!/bin/bash
# 等待 Phase 4 全部完成后自动启动 Phase 5

cd /mnt/workspace/open_research/rec-autopilot

PHASE4_EXPS="hp_emb12 hp_emb16 hp_emb24 hp_lr_5e5 hp_lr_1e4 hp_dropout03 hp_dropout07"

echo "[$(date '+%H:%M:%S')] 等待 Phase 4 完成..."

while true; do
    all_done=true
    for exp in $PHASE4_EXPS; do
        if ! grep -q "实验完成" log/${exp}.log 2>/dev/null; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        echo "[$(date '+%H:%M:%S')] Phase 4 全部完成！启动 Phase 5..."
        bash run_phase5_experiments.sh 2>&1 | tee log/phase5_all.log
        break
    fi

    sleep 60
done
