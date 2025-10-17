# HLS (Hierarchical Label Similarity) TCM Syndrome Classification System

## 🎯 项目简介

本项目实现了一个基于BERT的中医证候分类系统，使用层次化标签相似性损失函数(HLS Loss)进行训练。

## 🚀 快速开始

### 本地部署

1. **克隆仓库**
```bash
git clone https://github.com/undividedAttention/HLS.git
cd HLS
```

2. **运行自动部署脚本**
```bash
chmod +x deploy.sh
./deploy.sh
```

### 远程服务器部署

1. **连接到远程服务器**
```bash
ssh jingxiaozhu@172.28.23.21
# 密码: newuser
```

2. **克隆代码**
```bash
cd /home/jingxiaozhu
git clone https://github.com/undividedAttention/HLS.git
cd HLS
```

3. **运行部署脚本**
```bash
chmod +x deploy.sh
./deploy.sh
```

## 📊 硬件要求

### 推荐配置
- **GPU**: NVIDIA RTX 4090/3090 (24GB) 或更高
- **内存**: 32GB+ RAM
- **存储**: 100GB+ 可用空间
- **CPU**: 8核+ 处理器

### 最低配置
- **GPU**: NVIDIA RTX 3060 (6GB) 或 CPU
- **内存**: 8GB+ RAM
- **存储**: 50GB+ 可用空间

## 🔧 环境配置

### 自动配置
运行 `deploy.sh` 脚本会自动：
- 检测硬件配置
- 创建conda环境
- 安装合适的PyTorch版本
- 安装所有依赖
- 执行数据预处理

### 手动配置
```bash
# 创建conda环境
conda create -n hls_env python=3.8 -y
conda activate hls_env

# 安装PyTorch (GPU版本)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements_base.txt

# 数据预处理
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --do_preprocess
```

## 📈 使用方法

### 训练模型
```bash
# 激活环境
conda activate hls_env

# 运行训练
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 15 --batch_size 8
```

### 使用Screen运行长时间训练
```bash
# 创建screen会话
screen -S hls_training

# 运行训练
conda activate hls_env
PYTHONPATH=/home/jingxiaozhu/HLS python src/main.py --epochs 15 --batch_size 8

# 退出screen (Ctrl+A, D)
# 重新连接: screen -r hls_training
```

## 📊 性能指标

- **Macro-F1**: 0.3396 (最佳)
- **准确率**: 75.14%
- **平均层次距离**: 2.0651
- **训练时间**: 约15-20小时 (15 epochs)

## 🔍 项目结构

```
HLS/
├── src/                    # 源代码
│   ├── main.py            # 主训练脚本
│   ├── model.py           # BERT模型定义
│   ├── dataset.py         # 数据集处理
│   ├── loss.py            # HLS损失函数
│   └── utils.py           # 工具函数
├── data/                   # 数据文件
├── preprocessed_data/      # 预处理数据
├── output/                 # 输出结果
├── ZY-BERT/               # 预训练BERT模型
├── deploy.sh              # 自动部署脚本
├── check_hardware.py      # 硬件检测脚本
├── requirements_base.txt  # 依赖列表
└── REMOTE_DEPLOYMENT.md   # 远程部署指南
```

## 🛠️ 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 使用CPU版本PyTorch

2. **GPU内存不足**
   - 减小batch_size
   - 检查GPU使用情况

3. **依赖安装失败**
   - 更新pip: `pip install --upgrade pip`
   - 使用conda安装

4. **数据预处理失败**
   - 检查数据文件路径
   - 确保有足够磁盘空间

## 📝 更新日志

### v1.0.0 (2025-10-17)
- 初始版本发布
- 实现HLS损失函数
- 支持多GPU训练
- 添加自动部署脚本
- 支持远程服务器部署

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请提交Issue或联系项目维护者。
