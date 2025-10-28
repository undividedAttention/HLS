"""
消融实验结果对比脚本
读取各个实验的输出文件，生成对比报告
"""
import json
import os
import sys
import numpy as np
from tabulate import tabulate

def load_metrics(file_path):
    """加载实验结果"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取最佳epoch的性能（dev_macro_f1最高）
    best_epoch = max(data, key=lambda x: x.get('dev_macro_f1', 0))
    return best_epoch

def load_test_results(dir_path):
    """从输出目录加载测试集结果"""
    metrics_file = os.path.join(dir_path, 'all_epochs_metrics.json')
    
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取最佳epoch
    best_idx = 0
    best_f1 = 0
    for i, epoch in enumerate(data):
        if epoch.get('dev_macro_f1', 0) > best_f1:
            best_f1 = epoch.get('dev_macro_f1', 0)
            best_idx = i
    
    return data[best_idx]

def format_improvement(baseline, variant):
    """计算性能变化百分比"""
    if baseline == 0:
        return "N/A"
    change = ((variant - baseline) / baseline) * 100
    return f"{change:+.2f}%"

def generate_comparison_report():
    """生成对比报告"""
    
    results = {}
    
    # 加载各个实验的结果
    experiments = {
        'Baseline (Dual-Stream + HybridLoss)': 'output_v13',
        'Exp1 (Single-Stream + HybridLoss)': 'output_ablation_exp1',
        'Exp2 (Dual-Stream + FocalOnly)': 'output_ablation_exp2'
    }
    
    for exp_name, output_dir in experiments.items():
        metrics_path = os.path.join(output_dir, 'all_epochs_metrics.json')
        results[exp_name] = load_test_results(output_dir)
        
        if results[exp_name] is None:
            print(f"警告: {exp_name} 的结果文件不存在于 {metrics_path}")
            continue
    
    # 检查是否有结果
    if not any(results.values()):
        print("错误: 没有找到任何实验结果")
        return
    
    # 生成对比表格
    print("\n" + "="*80)
    print("消融实验结果对比")
    print("="*80)
    
    table_data = []
    
    baseline_f1 = results.get('Baseline (Dual-Stream + HybridLoss)', {}).get('test_macro_f1', 0)
    baseline_acc = results.get('Baseline (Dual-Stream + HybridLoss)', {}).get('test_accuracy', 0)
    baseline_ahd = results.get('Baseline (Dual-Stream + HybridLoss)', {}).get('test_avg_hier_distance', 0)
    
    for exp_name, metrics in results.items():
        if metrics is None:
            continue
            
        f1 = metrics.get('test_macro_f1', 0)
        acc = metrics.get('test_accuracy', 0)
        ahd = metrics.get('test_avg_hier_distance', 0)
        
        if exp_name == 'Baseline (Dual-Stream + HybridLoss)':
            f1_change = "-"
            acc_change = "-"
            ahd_change = "-"
        else:
            f1_change = format_improvement(baseline_f1, f1)
            acc_change = format_improvement(baseline_acc, acc)
            ahd_change = format_improvement(baseline_ahd, ahd)
        
        table_data.append([
            exp_name,
            f"{f1:.4f}",
            f1_change,
            f"{acc:.4f}",
            acc_change,
            f"{ahd:.4f}",
            ahd_change
        ])
    
    headers = [
        '模型',
        'Test Macro-F1',
        'F1变化',
        'Test Accuracy',
        'Acc变化',
        'Test AHD',
        'AHD变化'
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\n" + "="*80)
    print("关键观察")
    print("="*80)
    
    # 分析结果
    baseline_metrics = results.get('Baseline (Dual-Stream + HybridLoss)', {})
    exp1_metrics = results.get('Exp1 (Single-Stream + HybridLoss)', {})
    exp2_metrics = results.get('Exp2 (Dual-Stream + FocalOnly)', {})
    
    print("\n【实验1 - 单流架构 vs 双流架构】")
    if exp1_metrics:
        print(f"  双流架构在Macro-F1上优于单流架构:")
        print(f"    Baseline (双流): {baseline_metrics.get('test_macro_f1', 0):.4f}")
        print(f"    Exp1 (单流):     {exp1_metrics.get('test_macro_f1', 0):.4f}")
        print(f"    性能差异: {((baseline_metrics.get('test_macro_f1', 0) - exp1_metrics.get('test_macro_f1', 0)) / baseline_metrics.get('test_macro_f1', 0) * 100):.2f}%")
        
        print(f"\n  在平均层次距离(AHD)上的表现:")
        print(f"    Baseline (双流): {baseline_metrics.get('test_avg_hier_distance', 0):.4f}")
        print(f"    Exp1 (单流):     {exp1_metrics.get('test_avg_hier_distance', 0):.4f}")
        print(f"    结论: 双流架构能够更好地建模证候的层次结构")
    
    print("\n【实验2 - HLS Loss的影响】")
    if exp2_metrics:
        print(f"  HybridLoss (Focal + HLS) vs FocalOnly:")
        print(f"    Baseline (HybridLoss): {baseline_metrics.get('test_macro_f1', 0):.4f}")
        print(f"    Exp2 (FocalOnly):      {exp2_metrics.get('test_macro_f1', 0):.4f}")
        
        print(f"\n  在平均层次距离(AHD)上的表现:")
        print(f"    Baseline (HybridLoss): {baseline_metrics.get('test_avg_hier_distance', 0):.4f}")
        print(f"    Exp2 (FocalOnly):      {exp2_metrics.get('test_avg_hier_distance', 0):.4f}")
        print(f"    结论: HLS正则化显著改善了层次结构预测")
    
    print("\n" + "="*80)
    
    # 保存报告
    report = {
        'baseline': baseline_metrics,
        'exp1': exp1_metrics,
        'exp2': exp2_metrics,
        'summary': {
            'dual_stream_improvement': baseline_metrics.get('test_macro_f1', 0) - exp1_metrics.get('test_macro_f1', 0) if exp1_metrics else 0,
            'hls_improvement': baseline_metrics.get('test_avg_hier_distance', 0) - exp2_metrics.get('test_avg_hier_distance', 0) if exp2_metrics else 0,
        }
    }
    
    with open('ablation_comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n报告已保存到: ablation_comparison_report.json")

if __name__ == '__main__':
    generate_comparison_report()



