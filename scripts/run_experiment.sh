#!/bin/bash
# ============================================================
# RecAutoPilot 单次实验运行脚本
# ============================================================
set -e

# 配置区域
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
PYTHON_ENV_DIR="/root/anaconda3/envs/spore/bin"
METASPORE_DIR="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python"

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 参数解析
EXP_NAME="${1:-autopilot}"
EVAL_KEYS="${2:-business_type}"
HYPOTHESIS="${3:-}"
EXP_ID="${4:-}"

echo "=============================================="
echo "RecAutoPilot 实验启动"
echo "=============================================="
echo "项目目录: $PROJECT_DIR"
echo "实验名称: $EXP_NAME"
echo "评估分组: $EVAL_KEYS"
echo "实验假设: ${HYPOTHESIS:-无}"
echo "=============================================="

# 进入项目目录
cd "$PROJECT_DIR"

# 确保环境初始化
if [ ! -f "python.zip" ]; then
    echo "首次运行，初始化环境..."
    bash scripts/init_env.sh
fi

# 确保中断控制文件存在
touch train_interrupt.flag

# 设置环境变量
export PYTHONPATH=$METASPORE_DIR:$PROJECT_DIR/src:$PYTHONPATH
export PATH="$PYTHON_ENV_DIR:$PATH"
export PYSPARK_PYTHON=$PYTHON_ENV
export PYSPARK_DRIVER_PYTHON=$PYTHON_ENV

# 生成日志文件名
LOG_FILE="log/exp_$(date +%Y%m%d_%H%M%S).log"

# 构建运行命令
CMD="$PYTHON_ENV src/autopilot_runner.py \
    --base_conf ./conf/base.yaml \
    --exp_conf ./conf/experiment.yaml \
    --name $EXP_NAME \
    --eval_keys $EVAL_KEYS"

if [ -n "$HYPOTHESIS" ]; then
    CMD="$CMD --hypothesis \"$HYPOTHESIS\""
fi

if [ -n "$EXP_ID" ]; then
    CMD="$CMD --exp_id $EXP_ID"
fi

# 运行实验
echo ""
echo "开始运行实验..."
echo "日志文件: $LOG_FILE"
echo ""

# 使用 nohup 后台运行，同时输出到终端和日志
nohup env PYTHONUNBUFFERED=1 $CMD 2>&1 | tee "$LOG_FILE" &

TRAIN_PID=$!
echo "训练进程 PID: $TRAIN_PID"
echo ""
echo "=============================================="
echo "实验已启动！"
echo "=============================================="
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "停止训练: rm train_interrupt.flag"
echo "查看排行: cat leaderboard.json"
