import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取三个数据集的指标
base_dir = os.path.dirname(os.path.abspath(__file__))
datasets = {
    'internal test set': os.path.join(base_dir, 'output_trans', 'metrics', 'all_patients_summary_metrics.csv'),
    'external validation set 1': os.path.join(base_dir, 'output_trans_B', 'metrics', 'all_patients_summary_metrics.csv'),
    'external validation set 2': os.path.join(base_dir, 'output_trans_C', 'metrics', 'all_patients_summary_metrics.csv'),
}

metrics = ['ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']
metric_labels = ['MS-SSIM', 'MI', 'NRMSE', 'SMAPE', 'LogAC', 'MedSymAC']

# 读取数据
data_dict = {}
for name, path in datasets.items():
    df = pd.read_csv(path)
    data_dict[name] = df

dataset_names = list(datasets.keys())

# 绘制6个指标的箱线图
fig, axes = plt.subplots(2, 3, figsize=(19, 10))
fig.subplots_adjust(wspace=0.35, hspace=0.35)
axes = axes.flatten()

BLUE = '#3A79C0'

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    
    # 收集每个数据集的该指标数据
    all_data = []
    for name in dataset_names:
        values = data_dict[name][metric].dropna().values
        all_data.append(values)
    
    # 位置间隔
    positions = [0, 1.5, 3]
    
    # 画箱线图（黑色边框，中位线保留）
    bp = ax.boxplot(all_data, positions=positions, widths=0.7,
                    patch_artist=True, showcaps=True,
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2),
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='', markersize=0))
    
    # 画每个数据点（深蓝色实心圆点，加抖动）
    for pos, values in zip(positions, all_data):
        jitter = np.random.normal(0, 0.04, size=len(values))
        ax.scatter(np.full(len(values), pos) + jitter, values, 
                   color=BLUE, s=12, zorder=5, edgecolors='none')
    
    # 画均值点（红色小圆点，与中位线含义不同）
    for pos, values in zip(positions, all_data):
        mean_val = np.mean(values)
        ax.scatter(pos, mean_val, color='#C0392B', s=25, zorder=6, marker='o', edgecolors='#922B21', linewidths=0.8)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(dataset_names, fontsize=10)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = os.path.join(base_dir, 'vis', 'boxplot_6metrics_3centers.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"箱线图已保存到: {save_path}")
plt.show()
