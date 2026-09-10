#!/usr/bin/env python3
"""
H5数据可视化脚本

功能:
1. 输入患者h5文件夹路径，读取h5数据并转化为图像
2. 将同一层的不同模态拼接为一张图（6个模态+mask排列为3×3）
3. 每张小图右下角标明模态名称

使用方式:
    # 可视化单个患者所有层
    python visualize_h5.py --input /mnt/data/KASR/Dengsiyi/MRI_GAN/train-data/P000030031 --output ./vis_output

    # 只可视化指定层
    python visualize_h5.py --input /mnt/data/KASR/Dengsiyi/MRI_GAN/train-data/P000030031 --output ./vis_output --layers 0 1 2

    # 自定义图像尺寸
    python visualize_h5.py --input /path/to/patient --output ./vis_output --figsize 15 15
"""

import os
import argparse
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

# ==================== 配置 ====================
# 模态列表: (h5键名, 显示名称, 颜色映射)
# Mask不单独绘制，以轮廓形式叠加在每张模态图上
MODALITIES = [
    ('t1', 'T1', 'gray'),
    ('t2', 'T2', 'gray'),
    ('t1ce', 'T1CE', 'gray'),
    ('dwi', 'DWI', 'gray'),
    ('adc', 'ADC', 'gray'),
    ('pret1', 'PreT1', 'gray'),
]


def get_layer_number(h5_filename):
    """从文件名 layer_{num}.h5 中提取层号"""
    basename = os.path.basename(h5_filename)
    # 去掉前缀 layer_ 和后缀 .h5
    num_str = basename.replace('layer_', '').replace('.h5', '')
    try:
        return int(num_str)
    except ValueError:
        return -1


def read_h5_layer(h5_path):
    """读取单个h5文件中的所有模态数据

    Returns:
        dict: {键名: 2D numpy数组}
    """
    data = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            data[key] = f[key][()]
    return data


def normalize_for_display(arr, modality_key):
    """将数组归一化到 [0, 1] 用于显示"""
    arr = arr.astype(np.float32)

    # mask 用不同处理：二值化
    if modality_key == 'mask':
        arr = (arr > 0).astype(np.float32)
        return arr

    # 其他模态：去除异常值后归一化
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        # 裁剪到 1% - 99% 分位数，增强对比度
        p1, p99 = np.percentile(arr, [1, 99])
        arr = np.clip(arr, p1, p99)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        arr = np.zeros_like(arr)
    return arr


def visualize_single_layer(data_dict, layer_num, output_path, figsize=(12, 12)):
    """将单个layer的多个模态拼接为一张2*3图像，Mask以轮廓叠加

    Args:
        data_dict: h5文件读取的数据字典 {键名: 2D数组}
        layer_num: 层号，用于标题
        output_path: 输出图像路径
        figsize: 图像尺寸
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Layer {layer_num}', fontsize=16, fontweight='bold', y=0.98)

    # 扁平化axes便于遍历
    axes_flat = axes.flatten()

    # 预读取mask数据（如果存在）用于叠加轮廓
    mask_data = None
    if 'mask' in data_dict:
        mask_data = data_dict['mask']

    for idx, (mod_key, mod_name, cmap) in enumerate(MODALITIES):
        ax = axes_flat[idx]

        if mod_key in data_dict:
            img = data_dict[mod_key]
            img_display = normalize_for_display(img, mod_key)

            ax.imshow(img_display, cmap=cmap, aspect='equal')
            ax.set_title(mod_name, fontsize=12, fontweight='bold', pad=5)

            # 叠加Mask轮廓（红色）
            if mask_data is not None:
                try:
                    ax.contour(
                        mask_data,
                        levels=[0.5],
                        colors='red',
                        linewidths=1.5,
                        alpha=0.9
                    )
                except Exception:
                    pass  # 轮廓绘制失败时静默跳过

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


def visualize_patient(patient_h5_dir, output_dir, layers=None, figsize=(12, 12)):
    """可视化单个患者的所有h5数据

    Args:
        patient_h5_dir: 患者h5文件夹路径
        output_dir: 输出目录
        layers: 指定要可视化的层号列表，None表示全部
        figsize: 图像尺寸
    """
    if not os.path.exists(patient_h5_dir):
        print(f"错误: 输入路径不存在: {patient_h5_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 获取所有h5文件并按层号排序
    h5_files = glob.glob(os.path.join(patient_h5_dir, 'layer_*.h5'))
    if not h5_files:
        print(f"错误: 在 {patient_h5_dir} 中未找到 layer_*.h5 文件")
        return

    h5_files.sort(key=lambda x: get_layer_number(x))

    # 如果指定了layers，过滤文件
    if layers is not None:
        h5_files = [f for f in h5_files if get_layer_number(f) in layers]
        if not h5_files:
            print(f"错误: 未找到指定的层: {layers}")
            return

    patient_id = os.path.basename(os.path.normpath(patient_h5_dir))
    print(f"\n患者: {patient_id}")
    print(f"共找到 {len(h5_files)} 个layer文件")
    print(f"输出目录: {output_dir}")
    print("-" * 40)

    for h5_file in h5_files:
        layer_num = get_layer_number(h5_file)
        data_dict = read_h5_layer(h5_file)

        output_path = os.path.join(output_dir, patient_id,f'layer_{layer_num:03d}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        visualize_single_layer(data_dict, layer_num, output_path, figsize=figsize)

    print(f"\n完成! 共生成 {len(h5_files)} 张可视化图像")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='H5数据可视化工具')
    parser.add_argument('--input', '-i', required=True,
                        help='患者h5文件夹路径 (如: /path/to/train-data/P000030031)')
    parser.add_argument('--output', '-o', required=True,
                        help='可视化图像输出目录')
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=None,
                        help='指定要可视化的层号，空格分隔 (如: 0 1 2)，不指定则处理全部')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 12],
                        help='图像尺寸 (宽 高)，默认 18 12')

    args = parser.parse_args()

    figsize = tuple(args.figsize)
    visualize_patient(args.input, args.output, layers=args.layers, figsize=figsize)


if __name__ == '__main__':
    main()
