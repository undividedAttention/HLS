# HLS项目Conda环境配置说明

## 环境信息
- **环境名称**: hls_env
- **Python版本**: 3.9.24
- **Conda版本**: 25.3.0
- **CUDA版本**: 12.9 (服务器支持)
- **GPU**: 3x NVIDIA RTX 5880 Ada Generation (49GB显存)

## 已安装的依赖包
| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.1.0+cu121 | PyTorch深度学习框架，支持CUDA 12.1 |
| torchvision | 0.16.0+cu121 | 计算机视觉工具包 |
| torchaudio | 2.1.0+cu121 | 音频处理工具包 |
| transformers | 4.35.2 | Hugging Face Transformers库 |
| networkx | 3.2.1 | 图网络分析库 |
| numpy | 1.26.3 | 数值计算库 |
| tqdm | 4.66.1 | 进度条库 |
| scikit-learn | 1.3.2 | 机器学习库 |

## 环境激活命令
```bash
# 激活环境
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hls_env

# 验证环境
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 环境验证结果
✅ PyTorch版本: 2.1.0+cu121  
✅ CUDA可用: True  
✅ CUDA版本: 12.1  
✅ GPU数量: 3  
✅ 所有依赖包版本正确安装  

## 注意事项
1. 环境已配置为使用CUDA 12.1版本的PyTorch，与服务器CUDA 12.9兼容
2. 所有依赖包版本严格按照requirements.txt要求安装
3. 环境路径: `/opt/miniconda/envs/hls_env`
4. 建议在每次使用前先激活环境: `conda activate hls_env`
