#!/usr/bin/env python3
"""
对 Center C 的 h5 数据应用线性偏移校正，然后推理并可视化。
用法: python inference_adjusted.py
"""
import os, sys, glob, shutil
import numpy as np
import h5py
import SimpleITK as sitk
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 中文字体支持
_cjk_fonts = [f.name for f in fm.fontManager.ttflist if any(k in f.name.lower() for k in ['noto', 'cjk', 'wqy', 'simhei', 'simsun', 'droid sans fallback', 'wenquanyi'])]
if _cjk_fonts:
    plt.rcParams['font.sans-serif'] = [_cjk_fonts[0], 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD
from training_project.custom_transform import get_2d_test_transform
from monai.transforms import LoadImaged, EnsureChannelFirstd, ToTensord, Compose, MapTransform
from monai.inferers import sliding_window_inference

# ==================== 配置 ====================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 患者列表
PATIENTS = ['冯英', '陈德伦', '秦霞', '冉启玲']

# 源 h5 数据
SRC_H5_DIR = os.path.join(PROJECT_ROOT, 'train-data/images_ts_C')

# 调整后 h5 输出目录
ADJ_H5_DIR = os.path.join(PROJECT_ROOT, 'train-data/images_ts_C_adjusted')

# GT 模板目录（Center C）
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'pre-data/C/05_normalized')

# 模型 checkpoint
CKPT_PATH = os.path.join(PROJECT_ROOT, 'results/CE_MRI_simulate_PCa_1_trans_fold5-1-2/checkpoint/checkpoint.ckpt')

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_adjusted_C')

# 序列及偏移量 (训练集 body_mean - Center C body_mean)
# 训练集参考值: pret1=-0.5124, t1=-0.1450, t2=-0.4210, dwi=-0.7031, adc=-0.2484, t1ce=-0.5186
# Center C 当前: pret1=-0.403, t1=-0.289, t2=-0.515, dwi=-0.505, adc=-0.429, t1ce=-0.479
OFFSETS = {
    'pret1': -0.5124 - (-0.4030),  # -0.1094
    't1':    -0.1450 - (-0.2889),  # +0.1439
    't2':    -0.4210 - (-0.5145),  # +0.0935
    'dwi':   -0.7031 - (-0.5051),  # -0.1980
    'adc':   -0.2484 - (-0.4294),  # +0.1810
    't1ce':  -0.5186 - (-0.4789),  # -0.0397
}

# 输入序列（模型使用5个序列作为输入）
INPUT_KEYS = ["pret1", "t1", "t2", "dwi", "adc"]
DEVICE = 'cuda:4'

print("=" * 80)
print("Center C 线性偏移校正推理")
print(f"  患者: {PATIENTS}")
print(f"  偏移量: {OFFSETS}")
print(f"  模型: {CKPT_PATH}")
print(f"  输出: {OUTPUT_DIR}")
print("=" * 80)


# ==================== Step 1: 创建调整后的 h5 数据 ====================
def create_adjusted_h5():
    """为每个患者创建偏移校正后的 h5 文件"""
    print("\n[Step 1] 创建偏移校正后的 h5 数据...")
    os.makedirs(ADJ_H5_DIR, exist_ok=True)

    for pid in PATIENTS:
        src_dir = os.path.join(SRC_H5_DIR, pid)
        dst_dir = os.path.join(ADJ_H5_DIR, pid)
        os.makedirs(dst_dir, exist_ok=True)

        h5_files = sorted(glob.glob(os.path.join(src_dir, 'layer_*.h5')))
        print(f"  {pid}: {len(h5_files)} 层")

        for h5_path in h5_files:
            fname = os.path.basename(h5_path)
            dst_path = os.path.join(dst_dir, fname)

            # 读取原始 h5
            with h5py.File(h5_path, 'r') as f_in:
                mask = f_in['mask'][:] if 'mask' in f_in else None
                body = mask > 0.5 if mask is not None else np.ones(mask.shape, dtype=bool)

                with h5py.File(dst_path, 'w') as f_out:
                    # 写入 mask（不变）
                    f_out.create_dataset('mask', data=mask)

                    # 对每个序列应用偏移
                    for key in ['pret1', 't1', 't2', 'dwi', 'adc', 't1ce']:
                        if key in f_in:
                            data = f_in[key][:].astype(np.float32)
                            if key in OFFSETS:
                                # 只在 body 区域应用偏移
                                adjusted = data.copy()
                                adjusted[body] = data[body] + OFFSETS[key]
                                # 限制到 [-1, 1]
                                adjusted = np.clip(adjusted, -1.0, 1.0)
                                # body 外保持 -1
                                adjusted[~body] = data[~body]
                            else:
                                adjusted = data
                            f_out.create_dataset(key, data=adjusted)

        print(f"    -> 已保存到 {dst_dir}")

    print("[Step 1] Done!")


# ==================== Step 2: 推理 ====================
def run_inference():
    """对调整后的数据进行推理"""
    print("\n[Step 2] 加载模型并推理...")

    # 加载模型
    print(f"  加载 checkpoint: {CKPT_PATH}")
    model = Pix2Pix_2d_MulD.load_from_checkpoint(
        CKPT_PATH,
        map_location={"cuda:0": DEVICE},
        weights_only=False
    )
    model.eval()
    model.to(DEVICE)

    # 设置路径
    model.template_dir = TEMPLATE_DIR
    model.data_dir = os.path.dirname(ADJ_H5_DIR)
    model.test_dir = ADJ_H5_DIR

    # 创建输出目录
    pred_dir = os.path.join(OUTPUT_DIR, 'pred_nii')
    os.makedirs(pred_dir, exist_ok=True)

    # 收集所有 h5 文件
    all_files = []
    for pid in PATIENTS:
        pid_dir = os.path.join(ADJ_H5_DIR, pid)
        layer_files = sorted(glob.glob(os.path.join(pid_dir, 'layer_*.h5')))
        all_files.extend(layer_files)

    print(f"  共 {len(all_files)} 层需要推理")

    # 构建 transform
    transform = get_2d_test_transform(keys=INPUT_KEYS)

    # 逐层推理
    pred_results = {}  # {patient_id: {slice_idx: array}}
    for pid in PATIENTS:
        pred_results[pid] = {}

    with torch.no_grad():
        for i, h5_path in enumerate(all_files):
            pid = os.path.basename(os.path.dirname(h5_path))
            slice_idx = int(os.path.basename(h5_path).split('_')[-1].split('.')[0])

            # 加载并 transform
            sample = {"path": h5_path}
            transformed = transform(sample)
            image = transformed["image"].unsqueeze(0).to(DEVICE)  # [1, C, H, W]

            # sliding window inference
            roi_x = int(np.ceil(image.shape[2] / 32) * 32)
            roi_y = int(np.ceil(image.shape[3] / 32) * 32)
            output = sliding_window_inference(
                image, (roi_x, roi_y), 4, model.forward,
                overlap=0.25, mode='gaussian'
            )

            # clamp + body mask
            output = torch.clamp(output, -1.0, 1.0)

            # 读取 mask
            with h5py.File(h5_path, 'r') as f:
                mask = f['mask'][:] if 'mask' in f else None

            if mask is not None:
                mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                output = output * mask_t + (-1.0) * (1 - mask_t)

            pred_array = output[0, 0].cpu().numpy()
            pred_results[pid][slice_idx] = pred_array

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(all_files)}] done")

    # 保存为 NIfTI
    print("\n  保存预测 NIfTI...")
    for pid in PATIENTS:
        template_path = os.path.join(TEMPLATE_DIR, pid, "T1CE.nii.gz")
        if not os.path.exists(template_path):
            print(f"    警告: {pid} 模板不存在，跳过")
            continue

        template_nii = sitk.ReadImage(template_path)
        template_array = sitk.GetArrayFromImage(template_nii)
        pred_array = np.zeros_like(template_array, dtype=np.float32)

        for slice_idx, slice_img in pred_results[pid].items():
            pred_array[slice_idx] = slice_img

        # body mask 后处理
        body_mask_path = os.path.join(TEMPLATE_DIR, pid, "body_mask.nii.gz")
        if os.path.exists(body_mask_path):
            bm = sitk.GetArrayFromImage(sitk.ReadImage(body_mask_path)).astype(np.float32)
            pred_array = pred_array * bm + (-1.0) * (1 - bm)

        pred_nii = sitk.GetImageFromArray(pred_array)
        pred_nii.CopyInformation(template_nii)
        out_path = os.path.join(pred_dir, f"{pid}_pred.nii.gz")
        sitk.WriteImage(pred_nii, out_path)
        print(f"    {pid}: {out_path}")

    print("[Step 2] Done!")
    return pred_results


# ==================== Step 3: 可视化 ====================
def visualize_results():
    """对比可视化: 输入序列 + 原始预测 + 校正后预测 + GT"""
    print("\n[Step 3] 可视化...")

    vis_dir = os.path.join(OUTPUT_DIR, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # 读取原始预测（未校正）
    orig_pred_dir = os.path.join(PROJECT_ROOT, 'output_trans_C/pred_nii')

    for pid in PATIENTS:
        pid_vis_dir = os.path.join(vis_dir, pid)
        os.makedirs(pid_vis_dir, exist_ok=True)

        # 加载各种数据
        template_path = os.path.join(TEMPLATE_DIR, pid)

        # GT T1CE
        gt_path = os.path.join(template_path, "T1CE.nii.gz")
        gt_data = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)) if os.path.exists(gt_path) else None

        # 原始预测（未校正）
        orig_pred_path = os.path.join(orig_pred_dir, f"{pid}_pred.nii.gz")
        orig_pred = sitk.GetArrayFromImage(sitk.ReadImage(orig_pred_path)) if os.path.exists(orig_pred_path) else None

        # 校正后预测
        adj_pred_path = os.path.join(OUTPUT_DIR, 'pred_nii', f"{pid}_pred.nii.gz")
        adj_pred = sitk.GetArrayFromImage(sitk.ReadImage(adj_pred_path)) if os.path.exists(adj_pred_path) else None

        # 输入序列 - 从原始 h5 读取
        h5_files = sorted(glob.glob(os.path.join(SRC_H5_DIR, pid, 'layer_*.h5')))

        num_slices = gt_data.shape[0] if gt_data is not None else len(h5_files)

        # 选中间5层可视化
        mid = num_slices // 2
        vis_slices = list(range(max(0, mid-2), min(num_slices, mid+3)))

        for z in vis_slices:
            h5_path = os.path.join(SRC_H5_DIR, pid, f'layer_{z}.h5')
            if not os.path.exists(h5_path):
                continue

            with h5py.File(h5_path, 'r') as f:
                seqs = {}
                for k in INPUT_KEYS + ['t1ce']:
                    if k in f:
                        seqs[k] = f[k][:]

            # 构建可视化面板
            n_cols = len(INPUT_KEYS) + 3  # 5个输入 + 原始预测 + 校正预测 + GT
            fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))

            for ci, k in enumerate(INPUT_KEYS):
                if k in seqs:
                    img = seqs[k]
                    # 归一化显示
                    img_disp = (img + 1) / 2  # [-1,1] -> [0,1]
                    axes[ci].imshow(img_disp, cmap='gray', aspect='equal', vmin=0, vmax=1)
                axes[ci].set_title(k.upper(), fontsize=10, fontweight='bold')
                axes[ci].axis('off')

            # 原始预测
            ci = len(INPUT_KEYS)
            if orig_pred is not None and z < orig_pred.shape[0]:
                img = (orig_pred[z] + 1) / 2
                axes[ci].imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
            axes[ci].set_title('Original\nPred', fontsize=10, fontweight='bold')
            axes[ci].axis('off')

            # 校正预测
            ci += 1
            if adj_pred is not None and z < adj_pred.shape[0]:
                img = (adj_pred[z] + 1) / 2
                axes[ci].imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
            axes[ci].set_title('Adjusted\nPred', fontsize=10, fontweight='bold')
            axes[ci].axis('off')

            # GT
            ci += 1
            if gt_data is not None and z < gt_data.shape[0]:
                img = (gt_data[z] + 1) / 2
                axes[ci].imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
            axes[ci].set_title('GT\nT1CE', fontsize=10, fontweight='bold')
            axes[ci].axis('off')

            fig.suptitle(f'{pid} - Layer {z}', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            out_path = os.path.join(pid_vis_dir, f'layer_{z:03d}_compare.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # 生成一张汇总对比图（中间层大图）
        z_mid = num_slices // 2
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        h5_path = os.path.join(SRC_H5_DIR, pid, f'layer_{z_mid}.h5')
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                seqs = {}
                for k in INPUT_KEYS + ['t1ce']:
                    if k in f:
                        seqs[k] = f[k][:]

        # Row 1: PreT1, T1, T2, DWI  (前4个输入)
        input_labels = ['PreT1', 'T1', 'T2', 'DWI', 'ADC']
        for ci in range(4):
            k = INPUT_KEYS[ci]
            if k in seqs:
                img = (seqs[k] + 1) / 2
                axes[0, ci].imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
            axes[0, ci].set_title(f'Input: {input_labels[ci]}', fontsize=13, fontweight='bold')
            axes[0, ci].axis('off')

        # Row 2: ADC, 原始预测, 校正预测, GT
        if 'adc' in seqs:
            img = (seqs['adc'] + 1) / 2
            axes[1, 0].imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Input: {input_labels[4]}', fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')

        if orig_pred is not None and z_mid < orig_pred.shape[0]:
            img_orig = (orig_pred[z_mid] + 1) / 2
            axes[1, 1].imshow(img_orig, cmap='gray', aspect='equal', vmin=0, vmax=1)
        axes[1, 1].set_title('Original Prediction', fontsize=13, fontweight='bold')
        axes[1, 1].axis('off')

        if adj_pred is not None and z_mid < adj_pred.shape[0]:
            img_adj = (adj_pred[z_mid] + 1) / 2
            axes[1, 2].imshow(img_adj, cmap='gray', aspect='equal', vmin=0, vmax=1)
        axes[1, 2].set_title('Adjusted Prediction', fontsize=13, fontweight='bold')
        axes[1, 2].axis('off')

        if gt_data is not None and z_mid < gt_data.shape[0]:
            img_gt = (gt_data[z_mid] + 1) / 2
            axes[1, 3].imshow(img_gt, cmap='gray', aspect='equal', vmin=0, vmax=1)
        axes[1, 3].set_title('Ground Truth', fontsize=13, fontweight='bold')
        axes[1, 3].axis('off')

        fig.suptitle(f'{pid} - Layer {z_mid} (Mid Slice)', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        out_path = os.path.join(pid_vis_dir, 'summary_mid_compare.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  {pid}: 可视化保存到 {pid_vis_dir}")

    print("[Step 3] Done!")


# ==================== Main ====================
if __name__ == "__main__":
    # Step 1: 创建调整后的 h5
    create_adjusted_h5()

    # Step 2: 推理
    run_inference()

    # Step 3: 可视化
    visualize_results()

    print("\n" + "=" * 80)
    print("全部完成！")
    print(f"  调整后 h5: {ADJ_H5_DIR}")
    print(f"  预测 NIfTI: {os.path.join(OUTPUT_DIR, 'pred_nii')}")
    print(f"  可视化: {os.path.join(OUTPUT_DIR, 'vis')}")
    print("=" * 80)
