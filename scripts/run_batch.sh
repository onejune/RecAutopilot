#!/bin/bash
# ============================================================
# RecAutoPilot 通用批量实验启动脚本
#
# 用法:
#   bash scripts/run_batch.sh <plan_file>
#
# 示例:
#   bash scripts/run_batch.sh conf/plans/phase2_feature.yaml
#   bash scripts/run_batch.sh conf/plans/phase3_architecture.yaml
#
# plan_file 格式见 conf/plans/ 目录下的示例文件。
# ============================================================
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
METASPORE_DIR="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python"

cd "$PROJECT_DIR"

PLAN_FILE="${1:-}"
if [ -z "$PLAN_FILE" ]; then
    echo "用法: bash scripts/run_batch.sh <plan_file.yaml>"
    echo ""
    echo "可用的实验计划:"
    ls conf/plans/*.yaml 2>/dev/null || echo "  (暂无计划文件)"
    exit 1
fi

echo "=============================================="
echo "RecAutoPilot 批量实验"
echo "计划文件: $PLAN_FILE"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# 确保中断控制文件存在
touch train_interrupt.flag

# 设置环境变量
export PYTHONPATH="$METASPORE_DIR:$PROJECT_DIR/src:$PYTHONPATH"
export PYSPARK_PYTHON=$PYTHON_ENV
export PYSPARK_DRIVER_PYTHON=$PYTHON_ENV
export PYTHONUNBUFFERED=1

# 运行批量调度器
$PYTHON_ENV scripts/batch_runner.py --plan "$PLAN_FILE"
