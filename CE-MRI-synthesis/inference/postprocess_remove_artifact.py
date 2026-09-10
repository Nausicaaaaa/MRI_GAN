#!/usr/bin/env python3
"""
对已有的预测结果应用body mask后处理，去除体外伪影

功能:
1. 读取 output_trans/pred_nii 中的预测NIfTI文件
2. 加载对应的body mask
3. 将body外部区域的预测值设为-1.0（背景值）
4. 覆盖保存或保存到新目录

使用示例:
    # 处理所有患者，覆盖原文件
    python postprocess_remove_artifact.py --pred_dir output_trans/pred_nii --gt_dir pre-data/05_normalized --overwrite

    # 处理指定患者，保存到新目录
    python postprocess_remove_artifact.py --pred_dir output_trans/pred_nii --gt_dir pre-data/05_normalized --output_dir output_trans_clean/pred_nii --patients P000030031 P003607486
"""

import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk


def apply_body_mask(pred_path, mask_path, output_path, background_value=-1.0):
    """应用body mask去除体外伪影"""
    pred_img = sitk.ReadImage(pred_path)
    pred_arr = sitk.GetArrayFromImage(pred_img)
    
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    
    # 确保shape一致
    if pred_arr.shape != mask_arr.shape:
        print(f"  警告: shape不匹配 pred={pred_arr.shape} vs mask={mask_arr.shape}，跳过")
        return False
    
    # 应用mask: body外部区域设为background_value
    pred_arr = pred_arr * mask_arr + background_value * (1 - mask_arr)
    
    # 保存
    out_img = sitk.GetImageFromArray(pred_arr)
    out_img.CopyInformation(pred_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(out_img, output_path)
    return True


def main():
    parser = argparse.ArgumentParser(description='对预测结果应用body mask去除体外伪影')
    parser.add_argument('--pred_dir', default='output_trans/pred_nii', help='预测NIfTI目录')
    parser.add_argument('--gt_dir', default='pre-data/05_normalized', help='GT目录（包含body_mask.nii.gz）')
    parser.add_argument('--mask_dir', default=None, help='body mask目录（可选，默认在gt_dir下查找）')
    parser.add_argument('--output_dir', default=None, help='输出目录（不指定则使用pred_dir）')
    parser.add_argument('--patients', nargs='+', default=None, help='指定患者ID')
    parser.add_argument('--overwrite', action='store_true', help='覆盖原文件')
    parser.add_argument('--background_value', type=float, default=-1.0, help='背景值（默认-1.0）')
    
    args = parser.parse_args()
    
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    mask_dir = args.mask_dir or gt_dir
    output_dir = args.output_dir if args.output_dir and not args.overwrite else pred_dir
    
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('_pred.nii.gz')])
    if args.patients:
        pred_files = [f for f in pred_files if f.replace('_pred.nii.gz', '') in args.patients]
    
    print(f"处理 {len(pred_files)} 个患者的预测结果...")
    print(f"  pred_dir: {pred_dir}")
    print(f"  mask_dir: {mask_dir}")
    print(f"  output_dir: {output_dir}")
    
    success = 0
    failed = 0
    no_mask = 0
    
    for pred_file in pred_files:
        pid = pred_file.replace('_pred.nii.gz', '')
        pred_path = os.path.join(pred_dir, pred_file)
        
        # 查找body mask
        mask_path = os.path.join(gt_dir, pid, 'body_mask.nii.gz')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, 'body_masks', f'{pid}_body_mask.nii.gz')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(gt_dir, pid, 'mask.nii.gz')
        
        if not os.path.exists(mask_path):
            print(f"  [{pid}] 未找到body mask，跳过")
            no_mask += 1
            continue
        
        output_path = os.path.join(output_dir, pred_file)
        if apply_body_mask(pred_path, mask_path, output_path, args.background_value):
            success += 1
            print(f"  [{pid}] ✓ 已处理")
        else:
            failed += 1
    
    print(f"\n完成! 成功: {success}, 失败: {failed}, 无mask: {no_mask}")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
