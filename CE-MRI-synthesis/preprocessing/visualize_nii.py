#!/usr/bin/env python3
"""
NII数据可视化脚本

功能:
1. 输入患者nii文件夹路径，读取各模态nii.gz文件
2. 将同一层的不同模态拼接为一张图（6个模态排列为2×3）
3. 每张小图右下角标明模态名称

使用方式:
    # 可视化单个患者所有层
    python CE-MRI-synthesis/preprocessing/visualize_nii.py --input pre-data/02_resample/P000695856 --output ./vis_nii_output

    # 只可视化指定层
    python visualize_nii.py --input /mnt/data/KASR/Dengsiyi/MRI_GAN/pre-data/01_nii/P000030031 --output ./vis_nii_output --layers 50 60 70

    # 自定义图像尺寸
    python visualize_nii.py --input /path/to/patient --output ./vis_nii_output --figsize 15 15
"""

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

# ==================== 配置 ====================
# 模态列表: (文件名前缀, 显示名称, 颜色映射)
MODALITIES = [
    ('T1', 'T1', 'gray'),
    ('T2', 'T2', 'gray'),
    ('T1CE', 'T1CE', 'gray'),
    ('DWI', 'DWI', 'gray'),
    ('ADC', 'ADC', 'gray'),
    ('PreT1', 'PreT1', 'gray'),
]


def read_nii_modality(patient_dir, modality_name):
    """读取指定模态的nii.gz文件

    Args:
        patient_dir: 患者nii文件夹路径
        modality_name: 模态名称 (如 'T1')

    Returns:
        np.ndarray: 3D numpy数组，若文件不存在返回None
    """
    file_path = os.path.join(patient_dir, f'{modality_name}.nii.gz')
    if not os.path.exists(file_path):
        return None

    nii_img = nib.load(file_path)
    data = nii_img.get_fdata()
    return data.astype(np.float32)


def normalize_for_display(arr):
    """将数组归一化到 [0, 1] 用于显示"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        # 裁剪到 1% - 99% 分位数，增强对比度
        p1, p99 = np.percentile(arr, [1, 99])
        arr = np.clip(arr, p1, p99)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        arr = np.zeros_like(arr)
    return arr


def visualize_single_layer(modality_data_dict, layer_num, output_path, figsize=(18, 12)):
    """将单个layer的多个模态拼接为一张2×3图像

    Args:
        modality_data_dict: {模态名: 2D数组} 的字典
        layer_num: 层号，用于标题
        output_path: 输出图像路径
        figsize: 图像尺寸
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Layer {layer_num}', fontsize=16, fontweight='bold', y=0.98)

    # 扁平化axes便于遍历
    axes_flat = axes.flatten()

    for idx, (mod_file, mod_name, cmap) in enumerate(MODALITIES):
        ax = axes_flat[idx]

        if mod_file in modality_data_dict and modality_data_dict[mod_file] is not None:
            img = modality_data_dict[mod_file]
            img_display = normalize_for_display(img)

            ax.imshow(img_display, cmap=cmap, aspect='equal')
            ax.set_title(mod_name, fontsize=12, fontweight='bold', pad=5)

            # 右下角标注模态名称（黄色文字带黑色底框）
            ax.text(
                0.98, 0.02, mod_name,
                transform=ax.transAxes,
                fontsize=10,
                color='yellow',
                fontweight='bold',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='none')
            )
        else:
            ax.set_title(f'{mod_name} (缺失)', fontsize=12, color='red')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    fontsize=14, ha='center', va='center', color='red')

        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"  已保存: {output_path}")


def visualize_patient(patient_nii_dir, output_dir, layers=None, figsize=(18, 12)):
    """可视化单个患者的所有nii数据

    Args:
        patient_nii_dir: 患者nii文件夹路径
        output_dir: 输出目录
        layers: 指定要可视化的层号列表，None表示全部
        figsize: 图像尺寸
    """
    if not os.path.exists(patient_nii_dir):
        print(f"错误: 输入路径不存在: {patient_nii_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 读取所有模态数据
    modality_data = {}
    num_slices = None
    for mod_file, mod_name, _ in MODALITIES:
        data = read_nii_modality(patient_nii_dir, mod_file)
        modality_data[mod_file] = data
        if data is not None:
            num_slices = data.shape[2]  # nii通常是 (x, y, z)
            print(f"  读取 {mod_name}: shape={data.shape}")
        else:
            print(f"  缺失 {mod_name}")

    if num_slices is None:
        print("错误: 未找到任何有效的nii文件")
        return

    # 确定要处理的层号
    if layers is not None:
        slice_indices = [l for l in layers if 0 <= l < num_slices]
        if not slice_indices:
            print(f"错误: 指定的层号无效或超出范围 (0-{num_slices-1}): {layers}")
            return
    else:
        slice_indices = range(num_slices)

    patient_id = os.path.basename(os.path.normpath(patient_nii_dir))
    print(f"\n患者: {patient_id}")
    print(f"总层数: {num_slices}")
    print(f"待处理层数: {len(slice_indices)}")
    print(f"输出目录: {output_dir}")
    print("-" * 40)

    for z in slice_indices:
        # 提取当前层的各模态2D图像
        layer_data = {}
        for mod_file, _, _ in MODALITIES:
            data = modality_data[mod_file]
            if data is not None:
                # 提取第z层，并转置为 (y, x) 以适应imshow
                layer_data[mod_file] = data[:, :, z].T
            else:
                layer_data[mod_file] = None

        os.makedirs(os.path.join(output_dir,patient_id), exist_ok=True)
        output_path = os.path.join(output_dir,patient_id,f'layer_{z:03d}.png')
        visualize_single_layer(layer_data, z, output_path, figsize=figsize)

    print(f"\n完成! 共生成 {len(slice_indices)} 张可视化图像")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='NII数据可视化工具')
    parser.add_argument('--input', '-i', required=True,
                        help='患者nii文件夹路径 (如: /path/to/pre-data/01_nii/P000030031)')
    parser.add_argument('--output', '-o', required=True,
                        help='可视化图像输出目录')
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=None,
                        help='指定要可视化的层号，空格分隔 (如: 50 60 70)，不指定则处理全部')
    parser.add_argument('--figsize', nargs=2, type=int, default=[18, 12],
                        help='图像尺寸 (宽 高)，默认 18 12')

    args = parser.parse_args()

    figsize = tuple(args.figsize)
    visualize_patient(args.input, args.output, layers=args.layers, figsize=figsize)


if __name__ == '__main__':
    main()
