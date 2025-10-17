#!/bin/bash
"""
自动部署脚本
用于在远程服务器上自动配置HLS实验环境
"""

set -e  # 遇到错误立即退出

echo "🚀 HLS实验环境自动部署脚本"
echo "=================================="

# 检查是否在HLS目录中
if [ ! -f "src/main.py" ]; then
    echo "❌ 错误：请在HLS项目根目录中运行此脚本"
    exit 1
fi

# 1. 硬件配置检测
echo "📊 步骤1：检测硬件配置..."
python check_hardware.py

# 2. 创建conda环境
echo ""
echo "🐍 步骤2：创建conda环境..."
ENV_NAME="hls_env"

# 检查环境是否已存在
if conda env list | grep -q $ENV_NAME; then
    echo "⚠️  环境 $ENV_NAME 已存在，是否重新创建？(y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "✅ 使用现有环境 $ENV_NAME"
    fi
fi

# 创建新环境
if ! conda env list | grep -q $ENV_NAME; then
    echo "创建conda环境: $ENV_NAME"
    conda create -n $ENV_NAME python=3.8 -y
fi

# 激活环境
echo "激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 3. 安装PyTorch
echo ""
echo "🔥 步骤3：安装PyTorch..."

# 检测GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU，安装GPU版本PyTorch"
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
else
    echo "⚠️  未检测到GPU，安装CPU版本PyTorch"
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
fi

# 4. 安装其他依赖
echo ""
echo "📦 步骤4：安装其他依赖..."
pip install -r requirements_base.txt

# 5. 验证安装
echo ""
echo "✅ 步骤5：验证安装..."
python -c "
import torch
import transformers
import networkx
import numpy
import tqdm
import sklearn
print('✅ 所有依赖安装成功！')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
"

# 6. 数据预处理
echo ""
echo "📊 步骤6：数据预处理..."
if [ ! -f "preprocessed_data/label_map.json" ]; then
    echo "执行数据预处理..."
    PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --do_preprocess
    echo "✅ 数据预处理完成"
else
    echo "✅ 预处理数据已存在，跳过预处理步骤"
fi

# 7. 测试运行
echo ""
echo "🧪 步骤7：测试运行..."
echo "运行1个epoch的测试..."

# 根据硬件推荐batch size
MEMORY_GB=$(free -g | grep Mem | awk '{print $2}')
if [ $MEMORY_GB -ge 32 ]; then
    BATCH_SIZE=16
elif [ $MEMORY_GB -ge 16 ]; then
    BATCH_SIZE=8
elif [ $MEMORY_GB -ge 8 ]; then
    BATCH_SIZE=4
else
    BATCH_SIZE=2
fi

echo "推荐batch_size: $BATCH_SIZE"

# 运行测试
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 1 --batch_size $BATCH_SIZE

echo ""
echo "🎉 部署完成！"
echo "=================================="
echo "环境名称: $ENV_NAME"
echo "推荐batch_size: $BATCH_SIZE"
echo ""
echo "运行实验命令："
echo "conda activate $ENV_NAME"
echo "PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 15 --batch_size $BATCH_SIZE"
echo ""
echo "使用screen运行："
echo "screen -S hls_experiment"
echo "conda activate $ENV_NAME"
echo "PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 15 --batch_size $BATCH_SIZE"
echo "# 按Ctrl+A+D退出screen"
echo "screen -r hls_experiment  # 重新连接"
