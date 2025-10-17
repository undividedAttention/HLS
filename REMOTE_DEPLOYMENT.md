# 🚀 HLS实验远程部署指南

## 📋 部署步骤

### 1. 连接到远程服务器
```bash
ssh jingxiaozhu@172.28.23.21
# 输入密码: newuser
```

### 2. 克隆代码
```bash
cd /home/jingxiaozhu
git clone <your-git-repo-url> HLS
cd HLS
```

### 3. 运行自动部署脚本
```bash
chmod +x deploy.sh
./deploy.sh
```

## 🔧 手动配置（如果自动脚本失败）

### 1. 硬件检测
```bash
python check_hardware.py
```

### 2. 创建conda环境
```bash
conda create -n hls_env python=3.8 -y
conda activate hls_env
```

### 3. 安装PyTorch（根据硬件选择）

**GPU版本（如果有NVIDIA GPU）:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**CPU版本:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. 安装其他依赖
```bash
pip install -r requirements_base.txt
```

### 5. 数据预处理
```bash
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --do_preprocess
```

### 6. 运行实验
```bash
# 测试运行
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 1 --batch_size 8

# 完整训练
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 15 --batch_size 8
```

## 🖥️ 硬件配置建议

### GPU配置
- **RTX 4090/3090 (24GB)**: batch_size=16-32
- **RTX 4080/3080 (12GB)**: batch_size=8-16  
- **RTX 4070/3070 (8GB)**: batch_size=4-8
- **RTX 4060/3060 (6GB)**: batch_size=2-4

### CPU配置
- **32GB+ 内存**: batch_size=16-32
- **16GB 内存**: batch_size=8-16
- **8GB 内存**: batch_size=4-8
- **4GB 内存**: batch_size=2-4

## 📊 监控命令

### 查看GPU使用情况
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### 查看系统资源
```bash
htop
free -h
df -h
```

### 查看训练进程
```bash
ps aux | grep python
```

## 🎯 使用Screen运行长时间实验

```bash
# 创建screen会话
screen -S hls_experiment

# 激活环境并运行实验
conda activate hls_env
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 15 --batch_size 8

# 退出screen（保持运行）
# 按 Ctrl+A，然后按 D

# 重新连接screen
screen -r hls_experiment

# 查看所有screen会话
screen -ls
```

## 🚨 故障排除

### 1. 内存不足
- 减小batch_size
- 使用CPU版本PyTorch
- 检查其他进程占用内存

### 2. GPU内存不足
- 减小batch_size
- 使用gradient checkpointing
- 检查GPU使用情况

### 3. 依赖安装失败
- 更新pip: `pip install --upgrade pip`
- 使用conda安装: `conda install package_name`
- 检查Python版本兼容性

### 4. 数据预处理失败
- 检查数据文件路径
- 确保有足够磁盘空间
- 检查文件权限

## 📈 性能优化建议

### 1. 数据加载优化
- 使用SSD存储数据
- 增加DataLoader的num_workers
- 使用pin_memory=True

### 2. 训练优化
- 使用混合精度训练
- 启用梯度累积
- 使用学习率调度器

### 3. 系统优化
- 关闭不必要的服务
- 使用高性能文件系统
- 优化网络配置
