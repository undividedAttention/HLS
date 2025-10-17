#!/usr/bin/env python3
"""
硬件配置检测脚本
用于检测远程服务器的硬件配置并推荐合适的PyTorch版本
"""

import subprocess
import sys
import platform

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_gpu():
    """检查GPU配置"""
    print("=== GPU 配置检查 ===")
    nvidia_smi = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    if "Error" not in nvidia_smi and nvidia_smi:
        print("✅ 检测到NVIDIA GPU:")
        for line in nvidia_smi.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # 检查CUDA版本
        cuda_version = run_command("nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-")
        if cuda_version and "Error" not in cuda_version:
            print(f"✅ CUDA版本: {cuda_version}")
        else:
            print("⚠️  无法检测CUDA版本")
    else:
        print("❌ 未检测到NVIDIA GPU，将使用CPU版本")

def check_cpu():
    """检查CPU配置"""
    print("\n=== CPU 配置检查 ===")
    cpu_info = run_command("lscpu | grep -E 'Model name|CPU\\(s\\)|Thread\\(s\\)|Core\\(s\\)'")
    if cpu_info:
        print("✅ CPU信息:")
        for line in cpu_info.split('\n'):
            if line.strip():
                print(f"   {line.strip()}")
    else:
        print("❌ 无法获取CPU信息")

def check_memory():
    """检查内存配置"""
    print("\n=== 内存配置检查 ===")
    memory_info = run_command("free -h | grep Mem")
    if memory_info:
        print(f"✅ {memory_info}")
    else:
        print("❌ 无法获取内存信息")

def check_disk():
    """检查磁盘空间"""
    print("\n=== 磁盘空间检查 ===")
    disk_info = run_command("df -h /home")
    if disk_info:
        print("✅ 磁盘空间:")
        for line in disk_info.split('\n'):
            if line.strip():
                print(f"   {line.strip()}")
    else:
        print("❌ 无法获取磁盘信息")

def recommend_pytorch():
    """推荐PyTorch版本"""
    print("\n=== PyTorch版本推荐 ===")
    
    # 检查是否有GPU
    nvidia_smi = run_command("nvidia-smi")
    has_gpu = "Error" not in nvidia_smi and nvidia_smi
    
    if has_gpu:
        # 检查CUDA版本
        cuda_version = run_command("nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-")
        if cuda_version and "Error" not in cuda_version:
            cuda_major = cuda_version.split('.')[0]
            if cuda_major == "12":
                print("✅ 推荐PyTorch版本:")
                print("   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121")
            elif cuda_major == "11":
                print("✅ 推荐PyTorch版本:")
                print("   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118")
            else:
                print("⚠️  CUDA版本较老，建议使用CPU版本")
                print("   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0")
        else:
            print("✅ 推荐PyTorch版本 (GPU):")
            print("   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("✅ 推荐PyTorch版本 (CPU):")
        print("   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu")

def recommend_batch_size():
    """推荐batch size"""
    print("\n=== Batch Size推荐 ===")
    
    # 检查内存
    memory_info = run_command("free -m | grep Mem | awk '{print $2}'")
    if memory_info and "Error" not in memory_info:
        total_memory = int(memory_info)
        if total_memory >= 32000:  # 32GB+
            print("✅ 内存充足，推荐batch_size: 16-32")
        elif total_memory >= 16000:  # 16GB+
            print("✅ 内存中等，推荐batch_size: 8-16")
        elif total_memory >= 8000:   # 8GB+
            print("✅ 内存较少，推荐batch_size: 4-8")
        else:
            print("⚠️  内存较少，推荐batch_size: 2-4")
    
    # 检查GPU
    nvidia_smi = run_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    if nvidia_smi and "Error" not in nvidia_smi:
        gpu_memory = int(nvidia_smi.split('\n')[0])
        if gpu_memory >= 24000:  # 24GB+
            print("✅ GPU内存充足，推荐batch_size: 16-32")
        elif gpu_memory >= 12000:  # 12GB+
            print("✅ GPU内存中等，推荐batch_size: 8-16")
        elif gpu_memory >= 6000:   # 6GB+
            print("✅ GPU内存较少，推荐batch_size: 4-8")
        else:
            print("⚠️  GPU内存较少，推荐batch_size: 2-4")

def main():
    print("🔍 硬件配置检测脚本")
    print("=" * 50)
    
    check_gpu()
    check_cpu()
    check_memory()
    check_disk()
    recommend_pytorch()
    recommend_batch_size()
    
    print("\n" + "=" * 50)
    print("✅ 检测完成！请根据推荐配置环境。")

if __name__ == "__main__":
    main()
