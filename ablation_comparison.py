#!/usr/bin/env python3
"""
消融实验结果可视化对比
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 加载三个实验的结果
with open('output_v13/all_epochs_metrics.json', 'r') as f:
    baseline = json.load(f)

with open('output_ablation_exp1/all_epochs_metrics.json', 'r') as f:
    exp1 = json.load(f)

with open('output_ablation_exp2/all_epochs_metrics.json', 'r') as f:
    exp2 = json.load(f)

# 提取测试集指标
epochs = [e['epoch'] for e in baseline]

baseline_acc = [e['test_accuracy'] for e in baseline]
baseline_f1 = [e['test_macro_f1'] for e in baseline]
baseline_ahd = [e['test_avg_hier_distance'] for e in baseline]

exp1_acc = [e['test_accuracy'] for e in exp1]
exp1_f1 = [e['test_macro_f1'] for e in exp1]
exp1_ahd = [e['test_avg_hier_distance'] for e in exp1]
exp1_auprc = [e['test_auprc'] for e in exp1]

exp2_acc = [e['test_accuracy'] for e in exp2]
exp2_f1 = [e['test_macro_f1'] for e in exp2]
exp2_ahd = [e['test_avg_hier_distance'] for e in exp2]
exp2_auprc = [e['test_auprc'] for e in exp2]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('消融实验结果对比', fontsize=16, fontweight='bold')

# 1. Accuracy对比
axes[0, 0].plot(epochs, baseline_acc, 'o-', label='主实验 (Dual-Stream + HybridLoss)', linewidth=2)
axes[0, 0].plot(epochs, exp1_acc, 's-', label='实验1 (Single-Stream + HybridLoss)', linewidth=2)
axes[0, 0].plot(epochs, exp2_acc, '^-', label='实验2 (Dual-Stream + FocalOnly)', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('测试集准确率对比', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Macro-F1对比
axes[0, 1].plot(epochs, baseline_f1, 'o-', label='主实验', linewidth=2)
axes[0, 1].plot(epochs, exp1_f1, 's-', label='实验1', linewidth=2)
axes[0, 1].plot(epochs, exp2_f1, '^-', label='实验2', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Macro-F1', fontsize=12)
axes[0, 1].set_title('测试集Macro-F1对比', fontsize=13)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. AHD对比（越低越好）
axes[1, 0].plot(epochs, baseline_ahd, 'o-', label='主实验', linewidth=2)
axes[1, 0].plot(epochs, exp1_ahd, 's-', label='实验1', linewidth=2)
axes[1, 0].plot(epochs, exp2_ahd, '^-', label='实验2', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('平均层次距离 (AHD)', fontsize=12)
axes[1, 0].set_title('测试集层次距离对比 (越小越好)', fontsize=13)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. AUPRC对比（只有实验1和实验2有数据）
axes[1, 1].plot(epochs, exp1_auprc, 's-', label='实验1 (Single-Stream)', linewidth=2)
axes[1, 1].plot(epochs, exp2_auprc, '^-', label='实验2 (FocalOnly)', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('AUPRC', fontsize=12)
axes[1, 1].set_title('测试集AUPRC对比', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_comparison.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存: ablation_comparison.png")

# 打印最佳结果
print("\n" + "="*80)
print("最佳结果汇总")
print("="*80)

print("\n【主实验 - Dual-Stream + HybridLoss】")
best_epoch = np.argmax(baseline_acc)
print(f"最佳Epoch: {best_epoch + 1}")
print(f"  Accuracy: {baseline_acc[best_epoch]:.4f}")
print(f"  Macro-F1: {baseline_f1[best_epoch]:.4f}")
print(f"  AHD: {baseline_ahd[best_epoch]:.4f}")

print("\n【实验1 - Single-Stream + HybridLoss】")
best_epoch = np.argmax(exp1_acc)
print(f"最佳Epoch: {best_epoch + 1}")
print(f"  Accuracy: {exp1_acc[best_epoch]:.4f}")
print(f"  Macro-F1: {exp1_f1[best_epoch]:.4f}")
print(f"  AHD: {exp1_ahd[best_epoch]:.4f}")
print(f"  AUPRC: {exp1_auprc[best_epoch]:.4f}")

print("\n【实验2 - Dual-Stream + FocalOnly】")
best_epoch = np.argmax(exp2_acc)
print(f"最佳Epoch: {best_epoch + 1}")
print(f"  Accuracy: {exp2_acc[best_epoch]:.4f}")
print(f"  Macro-F1: {exp2_f1[best_epoch]:.4f}")
print(f"  AHD: {exp2_ahd[best_epoch]:.4f}")
print(f"  AUPRC: {exp2_auprc[best_epoch]:.4f}")

print("\n" + "="*80)


