#!/bin/bash

# 在线推荐系统启动脚本

set -e

echo "=== 在线推荐系统启动脚本 ==="

# 检查Python版本
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Python版本: $python_version"

# 检查是否安装了依赖
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

echo "激活虚拟环境..."
source venv/bin/activate

echo "安装依赖..."
pip install -r requirements.txt

# 检查数据目录
if [ ! -d "data/ml-1m" ]; then
    echo "MovieLens数据不存在，开始下载..."
    python scripts/train_model.py --download
fi

# 检查模型文件
if [ ! -f "models/saved/lightfm_model.pkl" ]; then
    echo "模型文件不存在，开始训练..."
    python scripts/train_model.py
fi

# 检查Redis是否运行
if ! pgrep -x "redis-server" > /dev/null; then
    echo "启动Redis服务..."
    redis-server --daemonize yes
    sleep 2
fi

echo "启动推荐API服务..."
python app.py
