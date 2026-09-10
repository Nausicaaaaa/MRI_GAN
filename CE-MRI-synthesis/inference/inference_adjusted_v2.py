#!/usr/bin/env python3
"""
Center C 线性偏移校正 V2 - per-patient mean+std matching
对每个患者的每个序列，应用仿射变换: new = (old - patient_mean) * scale + ref_mean
其中 scale = ref_std / patient_std
"""
import os, sys, glob
import numpy as np
import h5py
import SimpleITK as sitk
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD
from training_project.custom_transform import get_2d_test_transform
from monai.inferers import sliding_window_inference

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PATIENTS = ['冯英', '陈德伦', '秦霞', '冉启玲']
SRC_H5_DIR = os.path.join(PROJECT_ROOT, 'train-data/images_ts_C')
ADJ_H5_DIR = os.path.join(PROJECT_ROOT, 'train-data/images_ts_C_adjusted_v2')
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'pre-data/C/05_normalized')
CKPT_PATH = os.path.join(PROJECT_ROOT, 'results/CE_MRI_simulate_PCa_1_trans_fold5-1-2/checkpoint/checkpoint.ckpt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_adjusted_v2_C')
INPUT_KEYS = ["pret1", "t1", "t2", "dwi", "adc"]
ALL_KEYS = ["pret1", "t1", "t2", "dwi", "adc", "t1ce"]
DEVICE = 'cuda:4'

# 训练集参考统计
REF_MEAN = {'pret1':-0.5124,'t1':-0.1450,'t2':-0.4210,'dwi':-0.7031,'adc':-0.2484,'t1ce':-0.5186}
REF_STD  = {'pret1': 0.2451,'t1': 0.3832,'t2': 0.5455,'dwi': 0.1864,'adc': 0.4108,'t1ce': 0.2573}

# scale 因子: 1.0=完全匹配std, 0.0=不匹配std(仅偏移). 设0.5为折中
SCALE_FACTOR = 0.5

print("=" * 80)
print(f"Center C per-patient mean+std 校正 V2 (scale_factor={SCALE_FACTOR})")
print(f"  患者: {PATIENTS}")
print("=" * 80)


# ==================== Step 1: Per-patient 仿射校正 ====================
def create_adjusted_h5():
    print("\n[Step 1] Per-patient mean+std 校正...")
    os.makedirs(ADJ_H5_DIR, exist_ok=True)

    for pid in PATIENTS:
        src_dir = os.path.join(SRC_H5_DIR, pid)
        dst_dir = os.path.join(ADJ_H5_DIR, pid)
        os.makedirs(dst_dir, exist_ok=True)

        h5_files = sorted(glob.glob(os.path.join(src_dir, 'layer_*.h5')))

        # 先计算该患者的 per-sequence 统计 (用中间层)
        mid_file = h5_files[len(h5_files) // 2]
        patient_stats = {}
        with h5py.File(mid_file, 'r') as f:
            mask = f['mask'][:]
            body = mask > 0.5
            for k in ALL_KEYS:
                if k in f:
                    bd = f[k][:].astype(np.float32)[body]
                    patient_stats[k] = (float(bd.mean()), float(bd.std()))

        # 计算每个序列的仿射参数
        affine_params = {}
        for k in ALL_KEYS:
            if k not in patient_stats or k not in REF_MEAN:
                continue
            pm, ps = patient_stats[k]
            rm, rs = REF_MEAN[k], REF_STD[k]
            if ps > 0 and rs > 0:
                raw_scale = rs / ps
                # 混合: 实际 scale = SCALE_FACTOR * raw_scale + (1 - SCALE_FACTOR) * 1.0
                actual_scale = SCALE_FACTOR * raw_scale + (1 - SCALE_FACTOR) * 1.0
            else:
                actual_scale = 1.0
            affine_params[k] = (pm, actual_scale, rm)
            print(f"  {pid}/{k}: mean={pm:.4f}->{rm:.4f}, scale={actual_scale:.3f} (raw={rs/ps:.3f})")

        # 对所有层应用变换
        for h5_path in h5_files:
            fname = os.path.basename(h5_path)
            dst_path = os.path.join(dst_dir, fname)
            with h5py.File(h5_path, 'r') as f_in:
                mask = f_in['mask'][:]
                body = mask > 0.5
                with h5py.File(dst_path, 'w') as f_out:
                    f_out.create_dataset('mask', data=mask)
                    for k in ALL_KEYS:
                        if k not in f_in:
                            continue
                        data = f_in[k][:].astype(np.float32)
                        if k in affine_params:
                            pm, scale, rm = affine_params[k]
                            adjusted = data.copy()
                            # body 区域: new = (old - pm) * scale + rm
                            adjusted[body] = (data[body] - pm) * scale + rm
                            adjusted = np.clip(adjusted, -1.0, 1.0)
                            adjusted[~body] = data[~body]
                        else:
                            adjusted = data
                        f_out.create_dataset(k, data=adjusted)

        print(f"  {pid}: {len(h5_files)} layers -> {dst_dir}")
    print("[Step 1] Done!")


# ==================== Step 2: 推理 ====================
def run_inference():
    print("\n[Step 2] 加载模型并推理...")
    model = Pix2Pix_2d_MulD.load_from_checkpoint(
        CKPT_PATH, map_location={"cuda:0": DEVICE}, weights_only=False)
    model.eval(); model.to(DEVICE)
    model.template_dir = TEMPLATE_DIR
    model.data_dir = os.path.dirname(ADJ_H5_DIR)
    model.test_dir = ADJ_H5_DIR
    pred_dir = os.path.join(OUTPUT_DIR, 'pred_nii')
    os.makedirs(pred_dir, exist_ok=True)

    transform = get_2d_test_transform(keys=INPUT_KEYS)
    all_files = []
    for pid in PATIENTS:
        all_files.extend(sorted(glob.glob(os.path.join(ADJ_H5_DIR, pid, 'layer_*.h5'))))
    print(f"  共 {len(all_files)} 层")

    pred_results = {pid: {} for pid in PATIENTS}
    with torch.no_grad():
        for i, h5_path in enumerate(all_files):
            pid = os.path.basename(os.path.dirname(h5_path))
            slice_idx = int(os.path.basename(h5_path).split('_')[-1].split('.')[0])
            sample = {"path": h5_path}
            transformed = transform(sample)
            image = transformed["image"].unsqueeze(0).to(DEVICE)
            roi_x = int(np.ceil(image.shape[2] / 32) * 32)
            roi_y = int(np.ceil(image.shape[3] / 32) * 32)
            output = sliding_window_inference(image, (roi_x, roi_y), 4, model.forward,
                                               overlap=0.25, mode='gaussian')
            output = torch.clamp(output, -1.0, 1.0)
            with h5py.File(h5_path, 'r') as f:
                mask = f['mask'][:] if 'mask' in f else None
            if mask is not None:
                mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                output = output * mask_t + (-1.0) * (1 - mask_t)
            pred_results[pid][slice_idx] = output[0, 0].cpu().numpy()
            if (i + 1) % 10 == 0: print(f"    [{i+1}/{len(all_files)}]")

    for pid in PATIENTS:
        tp = os.path.join(TEMPLATE_DIR, pid, "T1CE.nii.gz")
        if not os.path.exists(tp): continue
        tn = sitk.ReadImage(tp); ta = sitk.GetArrayFromImage(tn)
        pa = np.zeros_like(ta, dtype=np.float32)
        for si, si_img in pred_results[pid].items(): pa[si] = si_img
        bmp = os.path.join(TEMPLATE_DIR, pid, "body_mask.nii.gz")
        if os.path.exists(bmp):
            bm = sitk.GetArrayFromImage(sitk.ReadImage(bmp)).astype(np.float32)
            pa = pa * bm + (-1.0) * (1 - bm)
        pn = sitk.GetImageFromArray(pa); pn.CopyInformation(tn)
        sitk.WriteImage(pn, os.path.join(pred_dir, f"{pid}_pred.nii.gz"))
    print("[Step 2] Done!")


# ==================== Step 3: 可视化 (三方对比) ====================
def visualize_results():
    print("\n[Step 3] 三方对比可视化...")
    vis_dir = os.path.join(OUTPUT_DIR, 'vis')
    orig_pred_dir = os.path.join(PROJECT_ROOT, 'output_trans_C/pred_nii')
    v1_pred_dir = os.path.join(PROJECT_ROOT, 'output_adjusted_C/pred_nii')

    for pid in PATIENTS:
        d = os.path.join(vis_dir, pid); os.makedirs(d, exist_ok=True)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(TEMPLATE_DIR, pid, "T1CE.nii.gz")))
        orig = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_pred_dir, f"{pid}_pred.nii.gz")))
        v1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(v1_pred_dir, f"{pid}_pred.nii.gz")))
        v2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(OUTPUT_DIR, 'pred_nii', f"{pid}_pred.nii.gz")))
        ns = gt.shape[0]; zm = ns // 2

        # 汇总图: 2行4列
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        hp = os.path.join(SRC_H5_DIR, pid, f'layer_{zm}.h5')
        with h5py.File(hp, 'r') as f:
            sq = {k: f[k][:] for k in INPUT_KEYS + ['t1ce'] if k in f}
        labels = ['PreT1', 'T1', 'T2', 'DWI', 'ADC']
        for ci in range(4):
            k = INPUT_KEYS[ci]
            if k in sq: ax[0, ci].imshow((sq[k]+1)/2, cmap='gray', vmin=0, vmax=1)
            ax[0, ci].set_title(f'Input: {labels[ci]}', fontsize=13, fontweight='bold'); ax[0, ci].axis('off')
        if 'adc' in sq: ax[1, 0].imshow((sq['adc']+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[1, 0].set_title('Input: ADC', fontsize=13); ax[1, 0].axis('off')
        ax[1, 1].imshow((orig[zm]+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[1, 1].set_title('Original (no correction)', fontsize=12); ax[1, 1].axis('off')
        ax[1, 2].imshow((v1[zm]+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[1, 2].set_title('V1: Mean-only offset', fontsize=12); ax[1, 2].axis('off')
        # V2 + GT 并排
        # 用subplot_mosaic太复杂，改成2行5列
        plt.close(fig)

        fig, ax = plt.subplots(2, 5, figsize=(25, 10))
        for ci in range(4):
            k = INPUT_KEYS[ci]
            if k in sq: ax[0, ci].imshow((sq[k]+1)/2, cmap='gray', vmin=0, vmax=1)
            ax[0, ci].set_title(f'Input: {labels[ci]}', fontsize=13, fontweight='bold'); ax[0, ci].axis('off')
        if 'adc' in sq: ax[0, 4].imshow((sq['adc']+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[0, 4].set_title('Input: ADC', fontsize=13, fontweight='bold'); ax[0, 4].axis('off')

        ax[1, 0].imshow((orig[zm]+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[1, 0].set_title('Original\n(no correction)', fontsize=12); ax[1, 0].axis('off')
        ax[1, 1].imshow((v1[zm]+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[1, 1].set_title('V1: Mean offset', fontsize=12); ax[1, 1].axis('off')
        ax[1, 2].imshow((v2[zm]+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[1, 2].set_title(f'V2: Mean+Std\n(sf={SCALE_FACTOR})', fontsize=12); ax[1, 2].axis('off')
        ax[1, 3].imshow((gt[zm]+1)/2, cmap='gray', vmin=0, vmax=1)
        ax[1, 3].set_title('Ground Truth', fontsize=13, fontweight='bold'); ax[1, 3].axis('off')
        # 差异图
        diff = np.abs(v2[zm] - v1[zm])
        im = ax[1, 4].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        plt.colorbar(im, ax=ax[1, 4], shrink=0.8)
        ax[1, 4].set_title('|V2 - V1|', fontsize=12); ax[1, 4].axis('off')

        fig.suptitle(f'{pid} - Layer {zm}', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(d, 'summary_v2_compare.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {pid}: saved")
    print("[Step 3] Done!")


if __name__ == "__main__":
    create_adjusted_h5()
    run_inference()
    visualize_results()
    print("\n" + "=" * 80)
    print(f"V2 完成! 输出: {OUTPUT_DIR}")
    print("=" * 80)
