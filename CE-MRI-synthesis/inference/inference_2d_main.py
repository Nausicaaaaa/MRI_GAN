#!/usr/bin/env python3
"""
批量推理脚本 - 支持模型文件夹批处理和指定患者推理

功能:
1. 支持输入模型文件夹，自动查找checkpoint.ckpt进行推理
2. 支持批处理所有患者或指定患者
3. 计算指标: ms_ssim, mi, nrmse, smape, logac, medsymac
4. 结果保存到output目录（包含CSV表格和可视化对比图）

使用示例:
    # 推理所有患者，生成可视化
    python inference_2d_main.py --model_dir results/model1 --output_dir output --visualize

    # 只推理指定患者
    python inference_2d_main.py --model_dir results/model1 --patients P003248663 P003248664 --output_dir output
    
    # 推理并采样16层进行可视化
    python inference_2d_main.py --model_dir results/model1 --output_dir output --visualize --sample_layers 16

    # 指定特定层可视化
    python inference_2d_main.py --model_dir results/model1 --output_dir output --visualize --specify_layers 30 40 50
"""

import os.path
import re
import sys
import argparse
import glob
import csv
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from monai.utils import set_determinism

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.test_param import config
from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD

torch.multiprocessing.set_sharing_strategy('file_system')
set_determinism(2023)
seed_everything(2023, workers=True)


# ==================== 指标计算函数 ====================

def scale12bit(img):
    """scale to 12 bit range"""
    new_mean = 2048.
    new_std = 400.
    std = np.std(img)
    if std == 0:
        std = 1e-8
    return np.clip(((img - np.mean(img)) / (std / new_std)) + new_mean, 1e-10, 4095)


def compute_nrmse(gt_array, pred_array, mask=None):
    """归一化均方根误差"""
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(gt_array, dtype=bool)
    
    mse_val = np.mean((gt_array[mask_img] - pred_array[mask_img]) ** 2)
    rmse = np.sqrt(mse_val)
    data_range = gt_array[mask_img].max() - gt_array[mask_img].min()
    if data_range == 0:
        return 0.0
    return rmse / data_range


def compute_smape(gt_array, pred_array, mask=None):
    """对称平均绝对百分比误差"""
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(gt_array, dtype=bool)
    
    true_img = scale12bit(gt_array[mask_img])
    pred_img = scale12bit(pred_array[mask_img])
    return np.mean(np.fabs(pred_img - true_img) / (np.fabs(true_img) + np.fabs(pred_img)))


def compute_logac(gt_array, pred_array, mask=None):
    """对数准确率比"""
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(gt_array, dtype=bool)
    
    true_img = scale12bit(gt_array[mask_img])
    pred_img = scale12bit(pred_array[mask_img])
    return np.mean(np.fabs(np.log(pred_img / true_img)))


def compute_medsymac(gt_array, pred_array, mask=None):
    """中位数对称准确率"""
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(gt_array, dtype=bool)
    
    true_img = scale12bit(gt_array[mask_img])
    pred_img = scale12bit(pred_array[mask_img])
    return np.exp(np.median(np.fabs(np.log(pred_img / true_img)))) - 1


def compute_mi(gt_array, pred_array, mask=None, bins=256):
    """直方图互信息"""
    if mask is not None:
        mask_img = mask.astype(bool)
    else:
        mask_img = np.ones_like(gt_array, dtype=bool)
    
    gt_flat = gt_array[mask_img].flatten()
    pred_flat = pred_array[mask_img].flatten()
    joint_hist, _, _ = np.histogram2d(gt_flat, pred_flat, bins=bins)
    pxy = joint_hist / float(np.sum(joint_hist))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


def compute_ms_sim_slice(gt_slice, pred_slice):
    """计算单层的MS-SSIM"""
    ms_ssim_fn = torchmetrics.functional.image.multiscale_structural_similarity_index_measure
    gt_tensor = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred_slice).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        val = ms_ssim_fn(preds=pred_tensor, target=gt_tensor)
    return val.item()


def compute_patient_metrics(gt_nii_path, pred_nii_path, mask_nii_path=None):
    """计算患者级别的所有指标"""
    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_nii_path))
    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_nii_path))
    
    mask_img = None
    if mask_nii_path and os.path.exists(mask_nii_path):
        mask_img = sitk.GetArrayFromImage(sitk.ReadImage(mask_nii_path))
    
    # 计算3D指标
    nrmse_val = compute_nrmse(gt_img, pred_img, mask_img)
    smape_val = compute_smape(gt_img, pred_img, mask_img)
    logac_val = compute_logac(gt_img, pred_img, mask_img)
    medsymac_val = compute_medsymac(gt_img, pred_img, mask_img)
    mi_val = compute_mi(gt_img, pred_img, mask_img)
    
    # 计算MS-SSIM（逐层平均）
    num_slices = gt_img.shape[0]
    ms_ssim_sum = 0
    for i in range(num_slices):
        gt_slice = gt_img[i, :, :]
        pred_slice = pred_img[i, :, :]
        ms_ssim_sum += compute_ms_sim_slice(gt_slice, pred_slice)
    
    ms_ssim_val = ms_ssim_sum / num_slices
    
    return {
        'ms_ssim': ms_ssim_val,
        'mi': mi_val,
        'nrmse': nrmse_val,
        'smape': smape_val,
        'logac': logac_val,
        'medsymac': medsymac_val
    }


# ==================== 可视化函数 ====================

def normalize_for_display(arr, percentile=(1, 99)):
    """归一化到[0,1]用于显示"""
    p_low, p_high = np.percentile(arr, percentile)
    arr = np.clip(arr, p_low, p_high)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    return arr


def visualize_patient_comparison(gt_nii_path, pred_nii_path, patient_id, output_dir,
                                  sample_layers=None, specify_layers=None):
    """生成患者对比可视化图"""
    gt_data = sitk.GetArrayFromImage(sitk.ReadImage(gt_nii_path))
    pred_data = sitk.GetArrayFromImage(sitk.ReadImage(pred_nii_path))
    
    num_slices = gt_data.shape[0]
    
    # 确定要可视化的层号
    if specify_layers is not None:
        slice_indices = [l for l in specify_layers if 0 <= l < num_slices]
    elif sample_layers is not None and sample_layers < num_slices:
        slice_indices = np.linspace(0, num_slices - 1, sample_layers, dtype=int).tolist()
    else:
        slice_indices = list(range(num_slices))
    
    patient_out = os.path.join(output_dir, patient_id)
    os.makedirs(patient_out, exist_ok=True)
    
    for z in slice_indices:
        gt_slice = gt_data[z, :, :]
        pred_slice = pred_data[z, :, :]
        
        gt_norm = normalize_for_display(gt_slice)
        pred_norm = normalize_for_display(pred_slice)
        diff = np.abs(gt_norm - pred_norm)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(gt_norm, cmap='gray', aspect='equal')
        axes[0].set_title('Original T1CE', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(pred_norm, cmap='gray', aspect='equal')
        axes[1].set_title('Predicted T1CE', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        im = axes[2].imshow(diff, cmap='hot', aspect='equal', vmin=0, vmax=1)
        axes[2].set_title('Absolute Difference', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        fig.suptitle(f'{patient_id} | Layer {z:03d}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_path = os.path.join(patient_out, f'{patient_id}_layer_{z:03d}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    print(f"  [{patient_id}] 已保存 {len(slice_indices)} 张对比图到 {patient_out}")


# ==================== 推理流程 ====================

def find_checkpoint(model_dir):
    """在模型文件夹及其子目录中查找checkpoint.ckpt"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型文件夹不存在: {model_dir}")
    
    # 递归查找所有.ckpt文件
    ckpt_files = glob.glob(os.path.join(model_dir, "**/*.ckpt"), recursive=True)
    if not ckpt_files:
        raise FileNotFoundError(f"在 {model_dir} 及其子目录中未找到任何.ckpt文件")
    
    # 优先查找名为checkpoint.ckpt的文件
    exact_matches = [f for f in ckpt_files if os.path.basename(f) == "checkpoint.ckpt"]
    if exact_matches:
        # 如果有多个checkpoint.ckpt，选择路径最深的（通常在checkpoint子目录中）
        exact_matches.sort(key=lambda x: x.count(os.sep), reverse=True)
        return exact_matches[0]
    
    # 如果没有精确匹配，查找包含"best"或"epoch"的文件
    best_matches = [f for f in ckpt_files if "best" in os.path.basename(f).lower() or "epoch" in os.path.basename(f).lower()]
    if best_matches:
        # 优先选择epoch最大的
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
    
    # 返回第一个找到的ckpt文件
    return ckpt_files[0]


def run_inference(model_dir, output_dir, patients=None, visualize=False,
                   sample_layers=None, specify_layers=None):
    """运行推理流程"""
    # 查找checkpoint
    ckpt_path = find_checkpoint(model_dir)
    print(f"[Inference] Using checkpoint: {ckpt_path}")
    
    # 加载模型
    target_device = f"cuda:{config.cuda_idx_list[0]}"
    model = Pix2Pix_2d_MulD.load_from_checkpoint(
        ckpt_path,
        map_location={"cuda:0": target_device},
        weights_only=False
    )
    
    # 设置路径
    dir_prefix = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config.filepath_img = os.path.join(dir_prefix, config.gt_data_dir)  # GT数据目录
    config.h5_2d_img_dir = os.path.join(dir_prefix, os.path.dirname(config.test_data_dir))  # h5数据根目录
    
    model.template_dir = config.filepath_img
    model.data_dir = config.h5_2d_img_dir
    model.test_dir = os.path.join(model.data_dir, os.path.basename(config.test_data_dir))  # 测试数据目录
    model.pred_result_dir = os.path.join(output_dir, "pred_nii")
    
    if not os.path.exists(model.pred_result_dir):
        os.makedirs(model.pred_result_dir)
    
    print(f"[Inference] Output dir: {model.pred_result_dir}")
    
    # 如果指定了患者，创建临时测试目录
    if patients is not None:
        print(f"[Inference] Running inference for {len(patients)} specified patients: {patients}")
        # 这里需要修改测试数据集逻辑，暂时先全部推理后过滤
        # TODO: 实现指定患者推理
    else:
        print(f"[Inference] Running inference for all patients")
    
    # 运行推理
    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[config.cuda_idx_list[0]],
        enable_progress_bar=True,
    )
    
    predictions = trainer.predict(model)
    print("[Inference] Inference completed!")
    
    # 计算指标
    print("\n[Metrics] Computing metrics...")
    all_metrics = []
    
    pred_files = sorted([f for f in os.listdir(model.pred_result_dir) if f.endswith('_pred.nii.gz')])
    
    for idx, pred_file in enumerate(pred_files):
        patient_id = pred_file.replace('_pred.nii.gz', '')
        
        # 如果指定了患者列表，只处理指定的患者
        if patients is not None and patient_id not in patients:
            continue
        
        gt_path = os.path.join(config.filepath_img, patient_id, "T1CE.nii.gz")
        pred_path = os.path.join(model.pred_result_dir, pred_file)
        mask_path = os.path.join(config.filepath_img, patient_id, "body_mask.nii.gz")
        
        if not os.path.exists(gt_path):
            print(f"  Warning: GT file not found for {patient_id}, skipping...")
            continue
        
        print(f"  [{idx+1}/{len(pred_files)}] Computing metrics for {patient_id}...")
        
        metrics = compute_patient_metrics(gt_path, pred_path, mask_path if os.path.exists(mask_path) else None)
        metrics['patient_id'] = patient_id
        all_metrics.append(metrics)
        
        # 可视化
        if visualize:
            print(f"  [{patient_id}] Generating visualization...")
            visualize_patient_comparison(
                gt_path, pred_path, patient_id,
                os.path.join(output_dir, "compare"),
                sample_layers=sample_layers,
                specify_layers=specify_layers
            )
    
    # 保存指标CSV
    if all_metrics:
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        df = pd.DataFrame(all_metrics)
        df = df[['patient_id', 'ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']]
        
        # 添加平均值行
        mean_row = {'patient_id': 'MEAN'}
        for col in ['ms_ssim', 'mi', 'nrmse', 'smape', 'logac', 'medsymac']:
            mean_row[col] = df[col].mean()
        
        df_with_mean = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        
        csv_path = os.path.join(metrics_dir, "all_patients_summary_metrics.csv")
        df_with_mean.to_csv(csv_path, index=False)
        print(f"\n[Metrics] Summary saved to: {csv_path}")
        
        # 打印汇总
        print("\n" + "="*80)
        print("Metrics Summary:")
        print("="*80)
        print(df_with_mean.to_string(index=False))
        print("="*80)
    else:
        print("\n[Metrics] No metrics computed.")
    
    print("\n[Done] All tasks completed!")


# ==================== 主函数 ====================

if __name__ == "__main__":
    # 使用config中定义的参数
    # 如果需要额外的参数解析，可以在test_param.py中添加
    
    model_dir = config.model_dir
    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', config.output_dir
    ))
    patients = config.patients
    visualize = config.visualize
    sample_layers = config.sample_layers
    specify_layers = config.specify_layers
    
    if model_dir is None:
        print("Error: --model_dir is required!")
        print("Usage: python inference_2d_main.py --model_dir <path_to_model_folder> [options]")
        sys.exit(1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"=" * 80)
    print(f"Batch Inference Configuration:")
    print(f"  Model dir: {model_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Patients: {patients if patients else 'All'}")
    print(f"  Visualize: {visualize}")
    if visualize:
        if specify_layers:
            print(f"  Specified layers: {specify_layers}")
        elif sample_layers:
            print(f"  Sample layers: {sample_layers}")
        else:
            print(f"  Sample layers: All")
    print(f"=" * 80)
    
    run_inference(
        model_dir=model_dir,
        output_dir=output_dir,
        patients=patients,
        visualize=visualize,
        sample_layers=sample_layers,
        specify_layers=specify_layers
    )
