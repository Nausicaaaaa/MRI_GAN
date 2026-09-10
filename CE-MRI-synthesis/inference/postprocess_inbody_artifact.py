#!/usr/bin/env python3
"""
后处理：检测并去除预测图中body内部的孤立马赛克伪影斑块

算法:
1. 对每层2D切片，用形态学开运算（opening）得到"干净基准图"（去除了小亮斑）
2. 计算 residual = prediction - clean_baseline，找到 residual > threshold 的亮区域
3. 按连通域大小过滤：只保留面积在 [min_area, max_area] 范围内的候选
4. 交叉验证：检查输入序列（T1等）在同一位置是否也有高信号
   - 有对应信号 → 真实增强结构，保留
   - 无对应信号 → GAN幻觉伪影，去除
5. 将伪影像素替换为 clean_baseline 的值
6. 保存清理后的 NIfTI 并生成对比可视化

使用方式:
    # 对指定患者测试
    python postprocess_inbody_artifact.py --patients P003634336 P003607486
    
    # 对所有output_trans患者处理
    python postprocess_inbody_artifact.py --pred_dir output_trans/pred_nii --all
    
    # 自定义参数
    python postprocess_inbody_artifact.py --patients P003634336 --bright_threshold 0.20 --min_area 80 --max_area 600
"""

import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label
import h5py
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def remove_inbody_artifacts_slice(pred_2d, mask_2d, inputs_2d=None,
                                   bright_threshold=0.15,
                                   min_area=50,
                                   max_area=500,
                                   opening_radius=15,
                                   dilate_radius=3,
                                   input_check_threshold=0.05):
    """
    对单张2D切片检测并去除body内部的伪影斑块
    
    Parameters:
        pred_2d: (H, W) 预测值, [-1, 1]
        mask_2d: (H, W) body mask, 0/1
        inputs_2d: dict of {seq_name: (H,W) array}, 输入序列（可选，用于交叉验证）
        bright_threshold: 残差阈值，超过此值视为候选伪影
        min_area: 最小面积（像素数），太小的忽略
        max_area: 最大面积（像素数），太大的视为真实结构
        opening_radius: 形态学开运算半径（像素），用于生成干净基准
        dilate_radius: 检测mask膨胀半径，扩大替换范围
        input_check_threshold: 输入序列交叉验证阈值
    
    Returns:
        cleaned: (H, W) 清理后的预测
        artifact_mask: (H, W) 检测到的伪影mask
        info: dict 检测信息
    """
    H, W = pred_2d.shape
    body = mask_2d > 0.5
    
    if body.sum() < 100:
        return pred_2d.copy(), np.zeros_like(pred_2d, dtype=bool), {'n_artifacts': 0}
    
    # --- Step 1: 形态学开运算得到"干净基准" ---
    # 开运算 = 先腐蚀后膨胀，会去掉小亮斑但保留大结构
    # 使用disk kernel
    kernel_size = 2 * opening_radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    cy, cx = opening_radius, opening_radius
    for dy in range(-opening_radius, opening_radius + 1):
        for dx in range(-opening_radius, opening_radius + 1):
            if dy * dy + dx * dx <= opening_radius * opening_radius:
                kernel[cy + dy, cx + dx] = 1
    
    # 灰度开运算: scipy grey_opening
    from scipy.ndimage import grey_opening, grey_closing
    clean_baseline = grey_opening(pred_2d, footprint=kernel)
    
    # --- Step 2: 计算残差，找亮区域 ---
    residual = pred_2d - clean_baseline
    bright_mask = (residual > bright_threshold) & body
    
    # --- Step 3: 膨胀候选区域（扩大替换范围，覆盖边缘过渡区）---
    if dilate_radius > 0:
        dilate_kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1))
        bright_mask = binary_dilation(bright_mask, structure=dilate_kernel) & body
    
    # --- Step 4: 连通域分析，按面积过滤 ---
    labeled, n_features = label(bright_mask)
    artifact_mask = np.zeros_like(bright_mask)
    info = {'n_candidates': n_features, 'n_artifacts': 0, 'areas': [], 'removed_areas': []}
    
    for i in range(1, n_features + 1):
        component = labeled == i
        area = component.sum()
        info['areas'].append(int(area))
        
        # 面积过滤
        if area < min_area or area > max_area:
            continue
        
        # --- Step 5: 交叉验证（如果提供了输入序列）---
        is_real = False
        if inputs_2d is not None:
            for seq_name in ['t1', 'pret1']:  # T1相关序列最可靠
                if seq_name in inputs_2d:
                    inp = inputs_2d[seq_name]
                    # 检查该区域在输入中是否也有相对高信号
                    region_vals = inp[component]
                    body_vals = inp[body]
                    if len(region_vals) > 0 and len(body_vals) > 0:
                        region_mean = region_vals.mean()
                        body_p75 = np.percentile(body_vals, 75)
                        # 如果输入序列在该区域也有较高信号，可能是真实结构
                        if region_mean > body_p75 - input_check_threshold:
                            is_real = True
                            break
        
        if not is_real:
            artifact_mask |= component
            info['n_artifacts'] += 1
            info['removed_areas'].append(int(area))
    
    # --- Step 6: 替换伪影像素 ---
    cleaned = pred_2d.copy()
    cleaned[artifact_mask] = clean_baseline[artifact_mask]
    
    return cleaned, artifact_mask, info


def process_patient(pred_path, mask_path, output_path, h5_dir=None,
                    bright_threshold=0.15, min_area=50, max_area=500,
                    vis_dir=None):
    """处理单个患者的3D预测"""
    pred_img = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.float32)
    
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    
    if pred_arr.shape != mask_arr.shape:
        print(f"  shape不匹配: pred={pred_arr.shape} vs mask={mask_arr.shape}")
        return False
    
    D = pred_arr.shape[0]  # number of slices
    
    # 加载输入序列（用于交叉验证）
    inputs_3d = {}
    if h5_dir and os.path.isdir(h5_dir):
        h5_files = sorted(glob.glob(os.path.join(h5_dir, 'layer_*.h5')))
        for key in ['t1', 'pret1']:
            seq_vol = []
            for hf in h5_files:
                with h5py.File(hf, 'r') as f:
                    if key in f:
                        seq_vol.append(f[key][:].astype(np.float32))
                    else:
                        seq_vol.append(np.zeros((pred_arr.shape[1], pred_arr.shape[2])))
            if len(seq_vol) == D:
                inputs_3d[key] = np.stack(seq_vol, axis=0)
    
    # 逐层处理
    cleaned_arr = np.zeros_like(pred_arr)
    total_artifacts = 0
    artifact_slices = []
    
    for z in range(D):
        inputs_z = {}
        for key, vol in inputs_3d.items():
            inputs_z[key] = vol[z]
        
        cleaned_z, art_mask_z, info_z = remove_inbody_artifacts_slice(
            pred_arr[z], mask_arr[z],
            inputs_2d=inputs_z if inputs_3d else None,
            bright_threshold=bright_threshold,
            min_area=min_area,
            max_area=max_area
        )
        cleaned_arr[z] = cleaned_z
        if info_z['n_artifacts'] > 0:
            total_artifacts += info_z['n_artifacts']
            artifact_slices.append((z, info_z))
    
    # 保存清理后的NIfTI
    cleaned_img = sitk.GetImageFromArray(cleaned_arr)
    cleaned_img.CopyInformation(pred_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(cleaned_img, output_path)
    
    # 打印结果
    pid = os.path.basename(pred_path).replace('_pred.nii.gz', '')
    print(f"  [{pid}] 检测到 {total_artifacts} 个伪影，分布在 {len(artifact_slices)} 层:")
    for z, info in artifact_slices:
        areas_str = ', '.join([f'{a}px' for a in info['removed_areas']])
        print(f"    Layer {z:3d}: {info['n_artifacts']} 个伪影 (面积: {areas_str})")
    
    # 可视化对比
    if vis_dir and artifact_slices:
        os.makedirs(vis_dir, exist_ok=True)
        for z, info in artifact_slices:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始预测
            norm_pred = normalize_display(pred_arr[z])
            axes[0].imshow(norm_pred, cmap='gray')
            axes[0].set_title('Original Prediction', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # 清理后
            norm_clean = normalize_display(cleaned_arr[z])
            axes[1].imshow(norm_clean, cmap='gray')
            axes[1].set_title('Cleaned (artifacts removed)', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # 伪影标记
            norm_pred2 = normalize_display(pred_arr[z])
            axes[2].imshow(norm_pred2, cmap='gray')
            # 叠加伪影mask轮廓
            art_mask_z = (cleaned_arr[z] != pred_arr[z])  # 被修改的区域
            if art_mask_z.any():
                from scipy.ndimage import binary_dilation as bd
                contour = bd(art_mask_z, iterations=2) ^ art_mask_z
                overlay = np.zeros((*art_mask_z.shape, 3))
                overlay[contour] = [1, 0, 0]  # 红色轮廓
                axes[2].imshow(overlay, alpha=0.8)
            axes[2].set_title('Artifacts marked (red)', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            areas_str = '+'.join([str(a) for a in info['removed_areas']])
            fig.suptitle(f'{pid} | Layer {z:03d} | {info["n_artifacts"]} artifacts (areas: {areas_str}px)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{pid}_artifact_layer_{z:03d}.png'),
                        dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        print(f"  [{pid}] 可视化保存到 {vis_dir}")
    
    return True


def normalize_display(arr, percentile=(1, 99)):
    p_low, p_high = np.percentile(arr, percentile)
    arr = np.clip(arr, p_low, p_high)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    return arr


def main():
    parser = argparse.ArgumentParser(description='后处理去除body内部马赛克伪影')
    parser.add_argument('--pred_dir', default='output_trans/pred_nii', help='预测NIfTI目录')
    parser.add_argument('--gt_dir', default='pre-data/05_normalized', help='GT目录')
    parser.add_argument('--h5_dir_base', default='train-data/images_ts',
                        help='h5输入数据基目录（用于交叉验证）')
    parser.add_argument('--output_dir', default='output_trans_cleaned/pred_nii',
                        help='输出目录')
    parser.add_argument('--vis_dir', default='vis_artifact_removal',
                        help='可视化输出目录')
    parser.add_argument('--patients', nargs='+', default=None, help='指定患者ID')
    parser.add_argument('--all', action='store_true', help='处理所有患者')
    parser.add_argument('--bright_threshold', type=float, default=0.15,
                        help='残差阈值 (默认0.15)')
    parser.add_argument('--min_area', type=int, default=50,
                        help='最小伪影面积 (默认50px)')
    parser.add_argument('--max_area', type=int, default=500,
                        help='最大伪影面积 (默认500px)')
    
    args = parser.parse_args()
    
    # 获取患者列表
    if args.patients:
        pred_files = [f'{p}_pred.nii.gz' for p in args.patients]
    elif args.all:
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('_pred.nii.gz')])
    else:
        print("请指定 --patients 或 --all")
        sys.exit(1)
    
    print(f"处理 {len(pred_files)} 个患者")
    print(f"参数: bright_threshold={args.bright_threshold}, min_area={args.min_area}, max_area={args.max_area}")
    print(f"输出: {args.output_dir}")
    print()
    
    success = 0
    for pred_file in pred_files:
        pid = pred_file.replace('_pred.nii.gz', '')
        pred_path = os.path.join(args.pred_dir, pred_file)
        
        # body mask
        mask_path = os.path.join(args.gt_dir, pid, 'body_mask.nii.gz')
        if not os.path.exists(mask_path):
            print(f"  [{pid}] 未找到body mask，跳过")
            continue
        
        output_path = os.path.join(args.output_dir, pred_file)
        h5_dir = os.path.join(args.h5_dir_base, pid)
        vis_dir = os.path.join(args.vis_dir, pid)
        
        if process_patient(pred_path, mask_path, output_path,
                           h5_dir=h5_dir if os.path.isdir(h5_dir) else None,
                           bright_threshold=args.bright_threshold,
                           min_area=args.min_area,
                           max_area=args.max_area,
                           vis_dir=vis_dir):
            success += 1
    
    print(f"\n完成! {success}/{len(pred_files)} 个患者处理成功")


if __name__ == '__main__':
    main()
