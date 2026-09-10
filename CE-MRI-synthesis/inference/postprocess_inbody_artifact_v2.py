#!/usr/bin/env python3
"""
后处理 v2：基于输入序列交叉验证，去除预测中GAN幻觉伪影

核心原理:
  真实T1CE增强结构在输入序列（特别是T1）中也有信号基础
  GAN幻觉伪影是预测中独有的亮斑，在输入序列中无对应信号
  
算法:
  1. 计算"预测增强图" = prediction - max(各输入通道)
  2. 对增强图做局部异常检测：找出增强值远高于局部邻域的区域
  3. 按面积过滤：只保留小孤立区域（50-500px），排除大解剖结构
  4. 替换为局部基准值（形态学开运算结果）

使用:
    python postprocess_inbody_artifact_v2.py --patients P003634336 P003607486
    python postprocess_inbody_artifact_v2.py --all
"""

import os, sys, argparse, glob
import numpy as np
import SimpleITK as sitk
import h5py
from scipy.ndimage import grey_opening, binary_dilation, label, gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_disk_kernel(radius):
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                kernel[radius + dy, radius + dx] = 1
    return kernel


def remove_artifacts_v2(pred_2d, mask_2d, inputs_2d,
                        enhance_threshold=0.20,
                        min_area=50,
                        max_area=600,
                        opening_radius=12,
                        dilate_radius=2):
    """
    对单张2D切片检测并去除body内部GAN幻觉伪影
    
    Parameters:
        pred_2d: (H,W) 预测 [-1,1]
        mask_2d: (H,W) body mask 0/1
        inputs_2d: dict {seq_name: (H,W)} 输入序列
        enhance_threshold: 增强残差阈值
        min_area: 最小伪影面积
        max_area: 最大伪影面积
        opening_radius: 开运算半径
        dilate_radius: 检测mask膨胀半径
    
    Returns:
        cleaned, artifact_mask, info
    """
    H, W = pred_2d.shape
    body = mask_2d > 0.5
    if body.sum() < 100:
        return pred_2d.copy(), np.zeros((H, W), dtype=bool), {'n_artifacts': 0}

    # --- Step 1: 构建"最大输入图"（取各通道的最大值） ---
    available_inputs = []
    for key in ['t1', 'pret1', 't2', 'dwi', 'adc']:
        if key in inputs_2d and inputs_2d[key] is not None:
            available_inputs.append(inputs_2d[key])
    
    if not available_inputs:
        # 无输入数据时退化为纯预测检测
        max_input = np.full_like(pred_2d, -1.0)
    else:
        max_input = np.maximum.reduce(available_inputs)

    # --- Step 2: 计算"预测增强图" ---
    # 真实T1CE增强：pred >> input（对比增强后变亮）
    # GAN幻觉：pred >> input（无输入支撑的虚假增强）
    # 但两者在enhancement map上表现不同：
    #   真实增强：增强值与周围组织一致（如血管周围也有增强）
    #   GAN幻觉：增强值是孤立的异常尖峰
    enhancement = pred_2d - max_input

    # --- Step 3: 对增强图做局部异常检测 ---
    # 用开运算得到"平滑增强基准"
    kernel = build_disk_kernel(opening_radius)
    smooth_enhancement = grey_opening(enhancement, footprint=kernel)
    
    # 增强残差 = 增强值 - 平滑增强基准
    # 高残差 = 孤立的增强尖峰（可能是伪影）
    enhance_residual = enhancement - smooth_enhancement
    
    # 候选区域：增强残差超过阈值的区域
    candidates = (enhance_residual > enhance_threshold) & body

    # --- Step 4: 膨胀候选区域 ---
    if dilate_radius > 0:
        dk = build_disk_kernel(dilate_radius)
        candidates = binary_dilation(candidates, structure=dk) & body

    # --- Step 5: 连通域分析 + 面积过滤 ---
    labeled_arr, n_features = label(candidates)
    artifact_mask = np.zeros((H, W), dtype=bool)
    info = {'n_candidates': n_features, 'n_artifacts': 0,
            'areas': [], 'removed_areas': [], 'removed_enhance': []}

    for i in range(1, n_features + 1):
        comp = labeled_arr == i
        area = int(comp.sum())
        info['areas'].append(area)
        
        if area < min_area or area > max_area:
            continue
        
        # 记录该区域的平均增强残差
        mean_res = float(enhance_residual[comp].mean())
        
        # 通过面积+残差双重条件
        artifact_mask |= comp
        info['n_artifacts'] += 1
        info['removed_areas'].append(area)
        info['removed_enhance'].append(mean_res)

    # --- Step 6: 替换伪影像素 ---
    # 用开运算结果（去除了小亮斑的平滑版本）替换
    clean_pred = grey_opening(pred_2d, footprint=kernel)
    cleaned = pred_2d.copy()
    cleaned[artifact_mask] = clean_pred[artifact_mask]
    
    # 在artifact边界做轻微高斯模糊，避免锐利边缘
    # （只对artifact mask边缘的少量像素做blur）
    edge = binary_dilation(artifact_mask, iterations=2) ^ artifact_mask
    if edge.any():
        blurred = gaussian_filter(cleaned, sigma=1.5)
        cleaned[edge] = blurred[edge]

    return cleaned, artifact_mask, info


def load_inputs_for_patient(h5_dir, n_slices):
    """加载h5输入数据，按layer编号顺序排列"""
    h5_files = sorted(glob.glob(os.path.join(h5_dir, 'layer_*.h5')),
                      key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if len(h5_files) != n_slices:
        # 如果数量不匹配，可能layer编号和NIfTI slice编号不一致
        # 此时用sorted顺序
        h5_files = sorted(glob.glob(os.path.join(h5_dir, 'layer_*.h5')))
        if len(h5_files) != n_slices:
            return None
    
    inputs_vol = {key: [] for key in ['pret1', 't1', 't2', 'dwi', 'adc']}
    for hf in h5_files:
        with h5py.File(hf, 'r') as f:
            for key in inputs_vol:
                if key in f:
                    inputs_vol[key].append(f[key][:].astype(np.float32))
                else:
                    inputs_vol[key].append(None)
    
    result = {}
    for key in inputs_vol:
        if all(v is not None for v in inputs_vol[key]):
            result[key] = np.stack(inputs_vol[key], axis=0)
    
    return result


def process_patient_v2(pred_path, mask_path, output_path, h5_dir=None,
                       enhance_threshold=0.20, min_area=50, max_area=600,
                       vis_dir=None):
    """处理单个患者的3D预测"""
    pred_img = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.float32)
    
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    
    if pred_arr.shape != mask_arr.shape:
        print(f"  shape不匹配: pred={pred_arr.shape} vs mask={mask_arr.shape}")
        return False
    
    D = pred_arr.shape[0]
    pid = os.path.basename(pred_path).replace('_pred.nii.gz', '')
    
    # 加载输入序列
    inputs_3d = None
    if h5_dir and os.path.isdir(h5_dir):
        inputs_3d = load_inputs_for_patient(h5_dir, D)
        if inputs_3d:
            print(f"  [{pid}] 已加载输入序列: {list(inputs_3d.keys())}")
        else:
            print(f"  [{pid}] 警告: h5文件数量({len(glob.glob(os.path.join(h5_dir, 'layer_*.h5')))})与切片数({D})不匹配，跳过交叉验证")
    
    # 逐层处理
    cleaned_arr = np.zeros_like(pred_arr)
    total_artifacts = 0
    artifact_slices = []
    
    for z in range(D):
        inputs_z = {}
        if inputs_3d:
            for key, vol in inputs_3d.items():
                inputs_z[key] = vol[z]
        
        cleaned_z, art_mask_z, info_z = remove_artifacts_v2(
            pred_arr[z], mask_arr[z],
            inputs_2d=inputs_z,
            enhance_threshold=enhance_threshold,
            min_area=min_area,
            max_area=max_area
        )
        cleaned_arr[z] = cleaned_z
        if info_z['n_artifacts'] > 0:
            total_artifacts += info_z['n_artifacts']
            artifact_slices.append((z, info_z))
    
    # 保存
    cleaned_img = sitk.GetImageFromArray(cleaned_arr)
    cleaned_img.CopyInformation(pred_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(cleaned_img, output_path)
    
    print(f"  [{pid}] 检测到 {total_artifacts} 个伪影，分布在 {len(artifact_slices)} 层:")
    for z, info in artifact_slices:
        areas = ', '.join([f'{a}' for a in info['removed_areas']])
        enhances = ', '.join([f'{e:.3f}' for e in info['removed_enhance']])
        print(f"    Layer {z:3d}: {info['n_artifacts']} 个 (面积: {areas}px, 增强残差: {enhances})")
    
    # 可视化
    if vis_dir and artifact_slices:
        os.makedirs(vis_dir, exist_ok=True)
        for z, info in artifact_slices:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 原始预测
            norm = lambda a: np.clip((a - np.percentile(a, 1)) / (np.percentile(a, 99) - np.percentile(a, 1) + 1e-8), 0, 1)
            
            axes[0].imshow(norm(pred_arr[z]), cmap='gray')
            axes[0].set_title('Original Prediction')
            axes[0].axis('off')
            
            # 清理后
            axes[1].imshow(norm(cleaned_arr[z]), cmap='gray')
            axes[1].set_title('Cleaned')
            axes[1].axis('off')
            
            # 伪影标记
            art_mask_z = (cleaned_arr[z] != pred_arr[z])
            overlay_rgb = np.stack([norm(pred_arr[z])] * 3, axis=-1)
            if art_mask_z.any():
                overlay_rgb[art_mask_z, 0] = 1.0  # 红色
                overlay_rgb[art_mask_z, 1] = 0.0
                overlay_rgb[art_mask_z, 2] = 0.0
            axes[2].imshow(overlay_rgb)
            axes[2].set_title('Artifacts (red)')
            axes[2].axis('off')
            
            # 输入T1参考
            if inputs_3d and 't1' in inputs_3d:
                axes[3].imshow(norm(inputs_3d['t1'][z]), cmap='gray')
                axes[3].set_title('Input T1 (reference)')
            else:
                axes[3].text(0.5, 0.5, 'No input data', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].axis('off')
            
            areas_str = '+'.join([str(a) for a in info['removed_areas']])
            fig.suptitle(f'{pid} | Layer {z:03d} | {info["n_artifacts"]} artifacts ({areas_str}px)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{pid}_v2_layer_{z:03d}.png'),
                        dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        print(f"  [{pid}] 可视化 → {vis_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='后处理v2：交叉验证去除body内伪影')
    parser.add_argument('--pred_dir', default='output_trans/pred_nii')
    parser.add_argument('--gt_dir', default='pre-data/05_normalized')
    parser.add_argument('--h5_dir_base', default='train-data/images_ts')
    parser.add_argument('--output_dir', default='output_trans_cleaned_v2/pred_nii')
    parser.add_argument('--vis_dir', default='vis_artifact_removal_v2')
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--enhance_threshold', type=float, default=0.20)
    parser.add_argument('--min_area', type=int, default=50)
    parser.add_argument('--max_area', type=int, default=600)
    
    args = parser.parse_args()
    
    if args.patients:
        pred_files = [f'{p}_pred.nii.gz' for p in args.patients]
    elif args.all:
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('_pred.nii.gz')])
    else:
        print("请指定 --patients 或 --all")
        sys.exit(1)
    
    print(f"处理 {len(pred_files)} 个患者")
    print(f"参数: enhance_threshold={args.enhance_threshold}, min_area={args.min_area}, max_area={args.max_area}\n")
    
    success = 0
    for pf in pred_files:
        pid = pf.replace('_pred.nii.gz', '')
        pred_path = os.path.join(args.pred_dir, pf)
        mask_path = os.path.join(args.gt_dir, pid, 'body_mask.nii.gz')
        if not os.path.exists(mask_path):
            print(f"  [{pid}] 未找到body mask，跳过")
            continue
        
        h5_dir = os.path.join(args.h5_dir_base, pid)
        output_path = os.path.join(args.output_dir, pf)
        vis_dir = os.path.join(args.vis_dir, pid)
        
        if process_patient_v2(pred_path, mask_path, output_path,
                              h5_dir=h5_dir,
                              enhance_threshold=args.enhance_threshold,
                              min_area=args.min_area,
                              max_area=args.max_area,
                              vis_dir=vis_dir):
            success += 1
    
    print(f"\n完成! {success}/{len(pred_files)}")


if __name__ == '__main__':
    main()
