#!/bin/bash
# 特征筛选实验脚本 - 2组并行，共4个实验
# 每组2个实验串行（避免Spark端口冲突），2组并行

cd /mnt/workspace/open_research/rec-autopilot
mkdir -p log

export PYTHONPATH="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python:./src:$PYTHONPATH"
export PYSPARK_PYTHON=/root/anaconda3/envs/spore/bin/python
export PYSPARK_DRIVER_PYTHON=/root/anaconda3/envs/spore/bin/python
PYTHON=/root/anaconda3/envs/spore/bin/python

echo "=========================================="
echo "特征筛选实验开始 (2组并行)"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 组1: 实验A + 实验B（串行）
run_group1() {
    echo "[Group1] 实验A: fea_no_outer_dev 开始 $(date '+%H:%M:%S')"
    $PYTHON src/autopilot_runner.py \
        --base_conf ./conf/base.yaml \
        --exp_conf ./conf/experiments/exp_fea_A.yaml \
        --name fea_no_outer_dev \
        --hypothesis "去掉外部行为特征duf_outer_dev_*，验证其贡献" \
        2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/fea_A.log
    echo "[Group1] 实验A结束 $(date '+%H:%M:%S')"

    echo "[Group1] 实验B: fea_short_window 开始 $(date '+%H:%M:%S')"
    $PYTHON src/autopilot_runner.py \
        --base_conf ./conf/base.yaml \
        --exp_conf ./conf/experiments/exp_fea_B.yaml \
        --name fea_short_window \
        --hypothesis "只保留<=7d时间窗口特征，去掉15d特征" \
        2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/fea_B.log
    echo "[Group1] 实验B结束 $(date '+%H:%M:%S')"
}

# 组2: 实验C + 实验D（串行）
run_group2() {
    echo "[Group2] 实验C: fea_more_cross 开始 $(date '+%H:%M:%S')"
    $PYTHON src/autopilot_runner.py \
        --base_conf ./conf/base.yaml \
        --exp_conf ./conf/experiments/exp_fea_C.yaml \
        --name fea_more_cross \
        --hypothesis "增加9个基础特征交叉，如bundle#devicetype等" \
        2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/fea_C.log
    echo "[Group2] 实验C结束 $(date '+%H:%M:%S')"

    echo "[Group2] 实验D: fea_no_ruf2 开始 $(date '+%H:%M:%S')"
    $PYTHON src/autopilot_runner.py \
        --base_conf ./conf/base.yaml \
        --exp_conf ./conf/experiments/exp_fea_D.yaml \
        --name fea_no_ruf2 \
        --hypothesis "去掉ruf2_*实时流式特征，验证其贡献" \
        2>&1 | grep -v "bkdr_hash\|add expr\|StringBKDR" | tee log/fea_D.log
    echo "[Group2] 实验D结束 $(date '+%H:%M:%S')"
}

# 并行启动两组
run_group1 &
PID1=$!

# 错开30秒启动，避免Spark同时初始化冲突
sleep 30
run_group2 &
PID2=$!

# 等待两组都完成
wait $PID1
echo "Group1 完成"
wait $PID2
echo "Group2 完成"

echo ""
echo "=========================================="
echo "全部特征实验完成!"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
