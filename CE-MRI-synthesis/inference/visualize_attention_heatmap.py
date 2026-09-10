#!/usr/bin/env python3
"""
注意力热图可视化脚本

功能:
1. 加载训练好的 trans-GAN 模型
2. 对测试数据进行推理，提取像素级模态注意力权重
3. 将注意力权重可视化为热图，叠加到原图上

使用示例:
    python visualize_attention_heatmap.py --model_dir results/model1 --output_dir output/attention_vis --num_samples 10
"""

import os
import sys
import glob
import argparse
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.transforms import Compose
from monai.data import Dataset

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.test_param import config
from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD
from training_project.custom_transform import get_2d_test_transform
from training_project.utils.my_transform import LoadH5
from monai.transforms import MapTransform


class LoadH5Debug(MapTransform):
    """从 H5 文件加载数据，保留原始序列数据用于可视化"""
    def __init__(self, path_key, keys):
        super().__init__(keys)
        self.path_key = path_key

    def __call__(self, data):
        d = dict(data)
        with h5py.File(d[self.path_key], 'r') as h5_file:
            for key in self.keys:
                if key in h5_file:
                    d[key] = h5_file[key][()]
                else:
                    raise KeyError(f"Key '{key}' not found in H5 file: {d[self.path_key]}")
        return d


class ConcatKeysDebug(MapTransform):
    """将多个序列键拼接成一个image键，同时保留原始序列"""
    def __init__(self, keys, output_key="image"):
        super().__init__(keys)
        self.output_key = output_key
    
    def __call__(self, data):
        d = dict(data)
        arrays = []
        for key in self.keys:
            arr = d[key]
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            arrays.append(arr)
        d[self.output_key] = np.concatenate(arrays, axis=0)
        # 保留原始序列
        for i, key in enumerate(self.keys):
            d[f"raw_{key}"] = arrays[i]
        return d


class EnsureChannelFirstDebug(MapTransform):
    """确保指定键的数据有通道维度"""
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d[key]
            if arr.ndim == 2:
                d[key] = np.expand_dims(arr, axis=0)
        return d


def find_checkpoint(model_dir):
    """在模型文件夹及其子目录中查找checkpoint.ckpt"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型文件夹不存在: {model_dir}")
    
    ckpt_files = glob.glob(os.path.join(model_dir, "**/*.ckpt"), recursive=True)
    if not ckpt_files:
        raise FileNotFoundError(f"在 {model_dir} 及其子目录中未找到任何.ckpt文件")
    
    exact_matches = [f for f in ckpt_files if os.path.basename(f) == "checkpoint.ckpt"]
    if exact_matches:
        exact_matches.sort(key=lambda x: x.count(os.sep), reverse=True)
        return exact_matches[0]
    
    best_matches = [f for f in ckpt_files if "best" in os.path.basename(f).lower() or "epoch" in os.path.basename(f).lower()]
    if best_matches:
        pattern = r"epoch=(\d+)"
        versions = []
        for f in best_matches:
            match = re.search(pattern, os.path.basename(f))
            if match:
                versions.append((int(match.group(1)), f))
        
        if versions:
            versions.sort(key=lambda x: x[0])
            return versions[-1][1]
        else:
            return best_matches[0]
    
    return ckpt_files[0]


def normalize_for_display(arr, percentile=(1, 99)):
    """归一化到[0,1]用于显示"""
    p_low, p_high = np.percentile(arr, percentile)
    arr = np.clip(arr, p_low, p_high)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    return arr


def gamma_enhance(arr, gamma=0.5):
    """Gamma 校正增强对比度"""
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.power(arr_norm, gamma)


def percentile_clip(arr, low=1, high=99):
    """百分位数裁剪，去除极端值"""
    p_low, p_high = np.percentile(arr, [low, high])
    return np.clip(arr, p_low, p_high)


def visualize_attention_heatmap(
    model,
    sample_data,
    device,
    output_dir,
    sample_idx,
    train_keys,
    overlay_alpha=0.5,
    cmap_name='jet'
):
    """
    对单个样本进行推理并可视化注意力权重（增强对比度版本）
    
    Args:
        model: 训练好的模型
        sample_data: 包含 "image" 和 "t1ce" 的字典
        device: 设备
        output_dir: 输出目录
        sample_idx: 样本索引
        train_keys: 序列名称列表
        overlay_alpha: 叠加透明度
        cmap_name: 热图颜色映射
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入数据 [1, C, H, W]
    image = sample_data["image"].unsqueeze(0).to(device)  # [1, 5, H, W]
    t1ce = sample_data["t1ce"].to(device)
    
    # 推理并获取注意力权重
    model.eval()
    with torch.no_grad():
        pred, attn_weights = model(image, return_attention=True)
    
    # attn_weights: [1, num_series, H, W]
    if attn_weights is None:
        print(f"  [Sample {sample_idx}] 该模型未返回注意力权重，请确认 fusion_mode='transformer'")
        return
    
    attn_weights = attn_weights[0].cpu().numpy()  # [num_series, H, W]
    pred = pred[0, 0].cpu().numpy()  # [H, W]
    image_np = image[0].cpu().numpy()  # [num_series, H, W]
    t1ce_np = t1ce[0].cpu().numpy() if t1ce.dim() > 2 else t1ce.cpu().numpy()
    
    num_series = len(train_keys)
    
    # ============ 图1: 原始输入 + 叠加热图（全局范围） ============
    fig, axes = plt.subplots(2, num_series + 1, figsize=(4 * (num_series + 1), 8))
    
    for i in range(num_series):
        ax = axes[0, i]
        img = image_np[i]
        img_norm = normalize_for_display(img)
        ax.imshow(img_norm, cmap='gray')
        ax.set_title(f"Input: {train_keys[i]}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    axes[0, -1].imshow(normalize_for_display(t1ce_np), cmap='gray')
    axes[0, -1].set_title("Ground Truth T1CE", fontsize=10, fontweight='bold')
    axes[0, -1].axis('off')
    
    for i in range(num_series):
        ax = axes[1, i]
        img = image_np[i]
        attn = attn_weights[i]
        img_norm = normalize_for_display(img)
        ax.imshow(img_norm, cmap='gray')
        im = ax.imshow(attn, cmap=cmap_name, alpha=overlay_alpha, vmin=0, vmax=1)
        ax.set_title(f"Attention: {train_keys[i]}\nmean={attn.mean():.3f}, max={attn.max():.3f}", 
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    axes[1, -1].imshow(normalize_for_display(pred), cmap='gray')
    axes[1, -1].set_title("Predicted T1CE", fontsize=10, fontweight='bold')
    axes[1, -1].axis('off')
    
    fig.suptitle(f"Sample {sample_idx} | Pixel-wise Modal Attention (Global 0-1)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"attention_global_sample_{sample_idx:04d}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # ============ 图2: 归一化热图（每个序列独立归一化，最大化对比度） ============
    fig, axes = plt.subplots(2, num_series + 1, figsize=(4 * (num_series + 1), 8))
    
    for i in range(num_series):
        ax = axes[0, i]
        img = image_np[i]
        img_norm = normalize_for_display(img)
        ax.imshow(img_norm, cmap='gray')
        ax.set_title(f"Input: {train_keys[i]}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    axes[0, -1].imshow(normalize_for_display(t1ce_np), cmap='gray')
    axes[0, -1].set_title("Ground Truth T1CE", fontsize=10, fontweight='bold')
    axes[0, -1].axis('off')
    
    for i in range(num_series):
        ax = axes[1, i]
        img = image_np[i]
        attn = attn_weights[i]
        img_norm = normalize_for_display(img)
        ax.imshow(img_norm, cmap='gray')
        # 使用自身min/max归一化，最大化颜色对比度
        vmin, vmax = attn.min(), attn.max()
        im = ax.imshow(attn, cmap=cmap_name, alpha=overlay_alpha, vmin=vmin, vmax=vmax)
        ax.set_title(f"Norm.Attn: {train_keys[i]}\nmin={vmin:.3f}, max={vmax:.3f}", 
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    axes[1, -1].imshow(normalize_for_display(pred), cmap='gray')
    axes[1, -1].set_title("Predicted T1CE", fontsize=10, fontweight='bold')
    axes[1, -1].axis('off')
    
    fig.suptitle(f"Sample {sample_idx} | Normalized Attention (per-sequence min/max)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"attention_norm_sample_{sample_idx:04d}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # ============ 图3: 差异增强热图（gamma校正 + 百分位裁剪） ============
    fig, axes = plt.subplots(1, num_series, figsize=(4 * num_series, 4))
    if num_series == 1:
        axes = [axes]
    for i in range(num_series):
        ax = axes[i] if num_series > 1 else axes
        attn = attn_weights[i]
        # 1. 百分位裁剪去除极端值
        attn_clipped = percentile_clip(attn, low=2, high=98)
        # 2. Gamma校正增强对比度（gamma < 1 增强暗部差异）
        attn_enhanced = gamma_enhance(attn_clipped, gamma=0.4)
        im = ax.imshow(attn_enhanced, cmap=cmap_name, vmin=0, vmax=1)
        ax.set_title(f"{train_keys[i]} Enhanced\nclip[2,98]+gamma=0.4", 
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Sample {sample_idx} | Contrast-Enhanced Attention Maps", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path3 = os.path.join(output_dir, f"attention_enhanced_sample_{sample_idx:04d}.png")
    plt.savefig(out_path3, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # ============ 图4: 注意力权重分布直方图 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：每个序列的权重分布
    for i in range(num_series):
        attn = attn_weights[i].flatten()
        axes[0].hist(attn, bins=50, alpha=0.5, label=f"{train_keys[i]} (μ={attn.mean():.3f})")
    axes[0].set_xlabel("Attention Weight", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Attention Weight Distribution", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 右图：每个序列的均值对比（bar图）
    means = [attn_weights[i].mean() for i in range(num_series)]
    stds = [attn_weights[i].std() for i in range(num_series)]
    bars = axes[1].bar(train_keys, means, yerr=stds, capsize=4, 
                        color=plt.cm.tab10(np.linspace(0, 1, num_series)), alpha=0.8)
    axes[1].set_ylabel("Mean Attention Weight", fontsize=12)
    axes[1].set_title("Mean Attention per Modality", fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    # 标注数值
    for bar, mean in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(f"Sample {sample_idx} | Attention Statistics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path4 = os.path.join(output_dir, f"attention_stats_sample_{sample_idx:04d}.png")
    plt.savefig(out_path4, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # 保存注意力权重的统计信息
    stats = {
        "sample_idx": sample_idx,
    }
    for i, key in enumerate(train_keys):
        attn = attn_weights[i]
        stats[f"{key}_mean"] = float(attn.mean())
        stats[f"{key}_std"] = float(attn.std())
        stats[f"{key}_max"] = float(attn.max())
        stats[f"{key}_min"] = float(attn.min())
    
    return stats


def visualize_patient_summary_stats(all_layer_stats, train_keys, output_dir, patient_id, cmap_name='jet'):
    """
    汇总患者所有层的统计信息，生成患者级别的汇总图表
    
    Args:
        all_layer_stats: 列表，每个元素是单层的stats字典
        train_keys: 序列名称列表
        output_dir: 输出目录
        patient_id: 患者ID
        cmap_name: 热图颜色映射
    """
    import matplotlib.pyplot as plt
    num_series = len(train_keys)
    
    # 收集所有层的统计数据
    layer_means = {key: [] for key in train_keys}
    layer_stds = {key: [] for key in train_keys}
    layer_mins = {key: [] for key in train_keys}
    layer_maxs = {key: [] for key in train_keys}
    
    for stats in all_layer_stats:
        for key in train_keys:
            layer_means[key].append(stats.get(f"{key}_mean", 0))
            layer_stds[key].append(stats.get(f"{key}_std", 0))
            layer_mins[key].append(stats.get(f"{key}_min", 0))
            layer_maxs[key].append(stats.get(f"{key}_max", 0))
    
    # ============ 图1: 患者汇总统计图（替代逐层stats图） ============
    fig = plt.figure(figsize=(16, 10))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # 左上图：各序列在所有层上的均值变化趋势
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(all_layer_stats))
    for i, key in enumerate(train_keys):
        ax1.plot(x, layer_means[key], marker='o', label=key, linewidth=1.5, markersize=3)
    ax1.set_xlabel("Layer Index", fontsize=11)
    ax1.set_ylabel("Mean Attention Weight", fontsize=11)
    ax1.set_title("Mean Attention per Modality across Layers", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 右中图：各序列在所有层上的均值柱状图（带误差条）
    ax2 = fig.add_subplot(gs[1, 0])
    overall_means = [np.mean(layer_means[key]) for key in train_keys]
    overall_stds = [np.std(layer_means[key]) for key in train_keys]
    bars = ax2.bar(train_keys, overall_means, yerr=overall_stds, capsize=4,
                   color=plt.cm.tab10(np.linspace(0, 1, num_series)), alpha=0.8)
    ax2.set_ylabel("Mean Attention Weight", fontsize=11)
    ax2.set_title("Overall Mean Attention (Patient-level)", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, overall_means):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 右中图：各序列在所有层上的标准差分布
    ax3 = fig.add_subplot(gs[1, 1])
    overall_stds_mean = [np.mean(layer_stds[key]) for key in train_keys]
    bars = ax3.bar(train_keys, overall_stds_mean,
                   color=plt.cm.tab10(np.linspace(0, 1, num_series)), alpha=0.8)
    ax3.set_ylabel("Std Dev", fontsize=11)
    ax3.set_title("Average Std Dev across Layers", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, std in zip(bars, overall_stds_mean):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{std:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 底部：箱线图展示每层各序列的分布
    ax4 = fig.add_subplot(gs[2, :])
    box_data = []
    box_labels = []
    for key in train_keys:
        box_data.append(layer_means[key])
        box_labels.append(key)
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, num_series))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel("Mean Attention Weight", fontsize=11)
    ax4.set_title("Distribution of Mean Attention across Layers (Box Plot)", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Patient {patient_id} | Attention Summary ({len(all_layer_stats)} layers)", 
                 fontsize=14, fontweight='bold')
    
    out_path = os.path.join(output_dir, "attention_stats_patient.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [Patient {patient_id}] Summary stats saved to: {out_path}")


def run_attention_visualization(model_dir, output_dir, num_samples=10, 
                                test_data_dir=None, train_keys=None,
                                overlay_alpha=0.5, cmap_name='jet',
                                patient_id=None):
    """运行注意力热图可视化（支持按患者ID分组）"""
    
    # 查找checkpoint
    ckpt_path = find_checkpoint(model_dir)
    print(f"[Attention Viz] Using checkpoint: {ckpt_path}")
    
    # 加载模型
    target_device = f"cuda:{config.cuda_idx_list[0]}"
    model = Pix2Pix_2d_MulD.load_from_checkpoint(
        ckpt_path,
        map_location={"cuda:0": target_device},
        weights_only=False
    )
    model.eval()
    device = next(model.parameters()).device
    
    # 检查模型是否使用 transformer 融合模式
    if model.fusion_mode != "transformer":
        print(f"警告: 当前模型融合模式为 '{model.fusion_mode}'，不是 'transformer'，可能无法获取注意力权重")
    
    # 设置路径
    dir_prefix = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if test_data_dir is None:
        test_data_dir = os.path.join(dir_prefix, config.test_data_dir.replace('/images_ts', ''))
    test_dir = os.path.join(test_data_dir, "images_ts")
    
    # 获取h5文件
    h5_files = []
    if patient_id is not None:
        # 指定了患者ID，只扫描该患者的文件
        patient_dir = os.path.join(test_dir, patient_id)
        if not os.path.exists(patient_dir):
            print(f"错误: 患者目录不存在: {patient_dir}")
            return
        h5_files = sorted([os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith('.h5')])
        print(f"[Attention Viz] Patient {patient_id}: Found {len(h5_files)} h5 files")
    else:
        # 未指定患者ID，扫描所有
        for root, dirs, files in os.walk(test_dir):
            for f in files:
                if f.endswith('.h5'):
                    h5_files.append(os.path.join(root, f))
        h5_files = sorted(h5_files)
        print(f"[Attention Viz] Found {len(h5_files)} h5 files (all patients), processing {num_samples} samples...")
    
    if not h5_files:
        print(f"错误: 未找到任何h5文件")
        return
    
    # 限制样本数
    h5_files = h5_files[:num_samples]
    
    # 创建数据变换
    if train_keys is None:
        train_keys = ["pret1", "t1", "t2", "dwi", "adc"]
    
    # 确定实际输出目录
    if patient_id is not None:
        output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = []
    
    for idx, h5_path in enumerate(h5_files):
        print(f"  [{idx+1}/{len(h5_files)}] Processing {h5_path}...")
        
        # 加载单个样本
        with h5py.File(h5_path, 'r') as f:
            sample_data = {}
            for key in train_keys + ["t1ce", "mask"]:
                if key in f:
                    sample_data[key] = f[key][()]
        
        # 应用变换
        for key in train_keys + ["t1ce", "mask"]:
            if key in sample_data and sample_data[key].ndim == 2:
                sample_data[key] = np.expand_dims(sample_data[key], axis=0)
        
        # 拼接输入
        image = np.concatenate([sample_data[k] for k in train_keys], axis=0)
        sample_data["image"] = torch.from_numpy(image).float()
        sample_data["t1ce"] = torch.from_numpy(sample_data["t1ce"]).float()
        
        # 可视化
        stats = visualize_attention_heatmap(
            model=model,
            sample_data=sample_data,
            device=device,
            output_dir=output_dir,
            sample_idx=idx,
            train_keys=train_keys,
            overlay_alpha=overlay_alpha,
            cmap_name=cmap_name
        )
        if stats:
            stats["file_path"] = h5_path
            all_stats.append(stats)
    
    # 如果有指定患者ID且处理了多层，生成患者汇总统计图
    if patient_id is not None and len(all_stats) > 1:
        print(f"  [Patient {patient_id}] Generating patient-level summary...")
        visualize_patient_summary_stats(
            all_layer_stats=all_stats,
            train_keys=train_keys,
            output_dir=output_dir,
            patient_id=patient_id,
            cmap_name=cmap_name
        )
    
    # 保存统计信息
    if all_stats:
        import json
        stats_path = os.path.join(output_dir, "attention_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"[Attention Viz] Stats saved to: {stats_path}")
    
    print(f"[Attention Viz] Done! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention heatmap for trans-GAN")
    parser.add_argument('--model_dir', type=str, required=True, help='训练好的模型文件夹路径')
    parser.add_argument('--output_dir', type=str, default='output/attention_vis', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='要可视化的样本数（不指定patient_id时生效）')
    parser.add_argument('--test_data_dir', type=str, default=None, help='测试数据目录')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='热图叠加透明度')
    parser.add_argument('--cmap', type=str, default='jet', help='热图颜色映射')
    parser.add_argument('--patient_id', type=str, default=None, help='指定患者ID，只处理该患者的所有层')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_attention_visualization(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        test_data_dir=args.test_data_dir,
        overlay_alpha=args.overlay_alpha,
        cmap_name=args.cmap,
        patient_id=args.patient_id
    )
