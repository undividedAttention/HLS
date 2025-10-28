#!/bin/bash

# 消融实验完整运行脚本
# 依次执行：实验1 -> 实验2 -> 结果对比

echo "========================================="
echo "  消融实验自动化脚本"
echo "========================================="
echo ""

# 检查GPU
echo "【步骤0】检查GPU状态..."
nvidia-smi
echo ""

# 读取用户输入
echo "请选择要运行的实验："
echo "  1) 仅运行实验1 (单流架构)"
echo "  2) 仅运行实验2 (仅Focal Loss)"
echo "  3) 运行完整消融实验 (实验1 + 实验2)"
echo "  4) 仅生成结果对比报告"
read -p "请输入选项 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "【实验1】单流架构训练..."
        echo "开始时间: $(date)"
        cd /home/jingxiaozhu/HLS
        CUDA_VISIBLE_DEVICES=2 python ablation_exp1_single_stream.py \
            --epochs 20 --batch_size 16 \
            --output_dir output_ablation_exp1 2>&1 | tee ablation_exp1.log
        echo "结束时间: $(date)"
        ;;
    2)
        echo ""
        echo "【实验2】仅Focal Loss训练..."
        echo "开始时间: $(date)"
        cd /home/jingxiaozhu/HLS
        CUDA_VISIBLE_DEVICES=2 python ablation_exp2_no_hls.py \
            --epochs 20 --batch_size 16 \
            --output_dir output_ablation_exp2 2>&1 | tee ablation_exp2.log
        echo "结束时间: $(date)"
        ;;
    3)
        echo ""
        echo "【完整消融实验】"
        
        echo ""
        echo ">>> 实验1: 单流架构"
        echo "开始时间: $(date)"
        cd /home/jingxiaozhu/HLS
        CUDA_VISIBLE_DEVICES=2 python ablation_exp1_single_stream.py \
            --epochs 20 --batch_size 16 \
            --output_dir output_ablation_exp1 2>&1 | tee ablation_exp1.log
        echo "实验1结束时间: $(date)"
        
        echo ""
        echo ">>> 实验2: 仅Focal Loss"
        echo "开始时间: $(date)"
        cd /home/jingxiaozhu/HLS
        CUDA_VISIBLE_DEVICES=2 python ablation_exp2_no_hls.py \
            --epochs 20 --batch_size 16 \
            --output_dir output_ablation_exp2 2>&1 | tee ablation_exp2.log
        echo "实验2结束时间: $(date)"
        
        echo ""
        echo ">>> 生成结果对比"
        python compare_ablation_results.py
        
        echo ""
        echo "完整消融实验完成！"
        echo "最终时间: $(date)"
        ;;
    4)
        echo ""
        echo "【生成结果对比报告】"
        cd /home/jingxiaozhu/HLS
        python compare_ablation_results.py
        ;;
    *)
        echo "无效选项，退出"
        exit 1
        ;;
esac

echo ""
echo "实验完成！"



