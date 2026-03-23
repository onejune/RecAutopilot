#!/bin/bash
# ============================================================
# RecAutoPilot 环境初始化脚本
# ============================================================
set -e

# 配置区域
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
PYTHON_ENV_DIR="/root/anaconda3/envs/spore/bin"
METASPORE_DIR="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python"

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "RecAutoPilot 环境初始化"
echo "=============================================="
echo "项目目录: $PROJECT_DIR"
echo "Python: $PYTHON_ENV"
echo "MetaSpore: $METASPORE_DIR"

# 进入项目目录
cd "$PROJECT_DIR"

# 创建必要目录
mkdir -p output
mkdir -p log
mkdir -p experiments
mkdir -p temp

# 创建中断控制文件
touch train_interrupt.flag
echo "创建 train_interrupt.flag"

# 设置环境变量
export PYTHONPATH=$METASPORE_DIR:$PROJECT_DIR/src:$PYTHONPATH
export PATH="$PYTHON_ENV_DIR:$PATH"
export PYSPARK_PYTHON=$PYTHON_ENV
export PYSPARK_DRIVER_PYTHON=$PYTHON_ENV

# 打包 MetaSpore python 模块
echo "打包 python.zip..."
cd $METASPORE_DIR/..
rm -f python.zip
zip -rq python.zip python -x "*.pyc" -x "__pycache__/*"
mv python.zip "$PROJECT_DIR/"
cd "$PROJECT_DIR"

# 复制 movas_logger 到 MetaSpore
cp "$PROJECT_DIR/src/movas_logger.py" "$METASPORE_DIR/metaspore/"

# 环境检查
echo ""
echo "=============================================="
echo "环境检查"
echo "=============================================="
echo "Python 版本:"
$PYTHON_ENV --version

echo ""
echo "PySpark 版本:"
$PYTHON_ENV -c "import pyspark; print(pyspark.__version__)" 2>/dev/null || echo "未安装"

echo ""
echo "PyTorch 版本:"
$PYTHON_ENV -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装"

echo ""
echo "MetaSpore 路径:"
$PYTHON_ENV -c "import metaspore; print(metaspore.__file__)" 2>/dev/null || echo "未找到"

echo ""
echo "=============================================="
echo "初始化完成！"
echo "=============================================="
echo ""
echo "运行实验: bash scripts/run_experiment.sh"
