#!/usr/bin/env python3
"""
将 output_trans/pred_nii 中所有生成结果与对应的真实T1CE（预处理后）
转换为DICOM格式，用于医生盲评比对。

输出结构:
  output_dicom/{anon_id}/1/  -> 其中一组DICOM切片
  output_dicom/{anon_id}/2/  -> 另一组DICOM切片
  (不标注哪组为真实/生成，用于盲评)
"""

import os
import uuid
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid as dicom_generate_uid
from pydicom.sequence import Sequence
from datetime import datetime

# ==================== 配置 ====================
PROJECT_ROOT = "/mnt/data/KASR/Dengsiyi/MRI_GAN"
PRED_DIR = os.path.join(PROJECT_ROOT, "output_trans_C", "pred_nii")
GT_DIR = os.path.join(PROJECT_ROOT, "pre-data/C", "05_normalized")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_dicom_C")



def generate_uid():
    """生成DICOM合规UID，保证全局唯一"""
    return dicom_generate_uid()


def nifti_to_dicom_series(nii_path, output_dicom_dir, patient_anon_id="P001",
                          study_uid=None, series_uid=None, series_number=1,
                          frame_ref_uid=None, series_desc="MRI",
                          scale_min=None, scale_max=None,
                          window_center=None, window_width=None,
                          mask_nii_path=None):
    """
    将NIfTI文件转换为DICOM序列并保存。
    使用pydicom写入，完全控制Study/Series UID，确保读片软件正确识别序列。

    Args:
        scale_min, scale_max: 统一的缩放范围（用于将像素映射到0-65535）。
            如果为None，则使用本图像自身的百分位范围。
        window_center, window_width: DICOM显示窗宽窗位。
            如果为None，则自动根据缩放后的数据计算。
        mask_nii_path: body mask NIfTI路径，用于在计算缩放参数时排除背景像素。
    Returns:
        (scale_min, scale_max, window_center, window_width) 供另一组使用。
    """
    os.makedirs(output_dicom_dir, exist_ok=True)

    # 读取NIfTI
    nii_image = sitk.ReadImage(nii_path)
    nii_array = sitk.GetArrayFromImage(nii_image)
    num_slices = nii_array.shape[0]

    # 使用min/max确定缩放范围，保留完整动态范围供医生调节对比度
    if scale_min is None:
        scale_min = float(nii_array.min())
    if scale_max is None:
        scale_max = float(nii_array.max())
    if scale_max - scale_min < 1e-8:
        scale_max = scale_min + 1.0

    # 缩放到DICOM 16-bit无符号整数范围 (0~65535)
    scaled = (nii_array - scale_min) / (scale_max - scale_min) * 65535
    scaled = np.clip(scaled, 0, 65535).astype(np.uint16)

    # 使用全16-bit范围的窗宽窗位，让医生在读片软件中可自由调节对比度
    if window_center is None:
        window_center = 32768
    if window_width is None:
        window_width = 65536

    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")

    spacing = nii_image.GetSpacing()
    origin = nii_image.GetOrigin()
    direction = nii_image.GetDirection()
    rows, cols = scaled.shape[1], scaled.shape[2]

    for z in range(num_slices):
        slice_data = scaled[z, :, :]
        sop_instance_uid = generate_uid()
        pos_z = origin[2] + z * spacing[2]

        # 创建pydicom FileDataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        ds = FileDataset(
            os.path.join(output_dicom_dir, f"slice_{z:04d}.dcm"),
            {}, file_meta=file_meta, preamble=b"\x00" * 128
        )
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # DICOM标签
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.InstanceCreationDate = date_str
        ds.InstanceCreationTime = time_str
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.SOPInstanceUID = sop_instance_uid
        ds.StudyDate = date_str
        ds.SeriesDate = date_str
        ds.StudyTime = time_str
        ds.SeriesTime = time_str
        ds.AccessionNumber = "ACC-001"
        ds.Modality = "MR"
        ds.Manufacturer = "Anonymous"
        ds.InstitutionName = "Anonymous"
        ds.StudyDescription = "MRI Study"
        ds.SeriesDescription = series_desc
        ds.PatientName = patient_anon_id
        ds.PatientID = patient_anon_id
        ds.PatientBirthDate = "19000101"
        ds.PatientSex = "O"
        ds.SliceThickness = round(float(spacing[2]), 6)
        ds.ProtocolName = "MRI"

        # 关键UID：确保同Study共享、同Series共享
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_ref_uid

        ds.StudyID = "1"
        ds.SeriesNumber = series_number
        ds.AcquisitionNumber = z + 1
        ds.InstanceNumber = z + 1
        ds.ImagePositionPatient = [origin[0], origin[1], pos_z]
        ds.ImageOrientationPatient = [
            direction[0], direction[1], direction[2],
            direction[3], direction[4], direction[5]
        ]

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.WindowCenter = round(window_center, 1)
        ds.WindowWidth = round(window_width, 1)

        ds.PixelData = slice_data.tobytes()

        ds.save_as(os.path.join(output_dicom_dir, f"slice_{z:04d}.dcm"))

    print(f"    已保存 {num_slices} 张DICOM到 {output_dicom_dir}")
    return scale_min, scale_max, window_center, window_width


def main():
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('_pred.nii.gz')])

    print(f"将处理 {len(pred_files)} 个患者:")
    for f in pred_files:
        print(f"  - {f}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0
    for i, pred_file in enumerate(pred_files, 1):
        patient_id = pred_file.replace('_pred.nii.gz', '')
        pred_path = os.path.join(PRED_DIR, pred_file)
        gt_path = os.path.join(GT_DIR, patient_id, "T1CE.nii.gz")
        # 查找body mask
        mask_path = os.path.join(GT_DIR, patient_id, "body_mask.nii.gz")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(os.path.dirname(GT_DIR), "body_masks", f"{patient_id}_body_mask.nii.gz")
        if not os.path.exists(mask_path):
            mask_path = None

        print(f"[{i}/{len(pred_files)}] 患者: {patient_id}")

        if not os.path.exists(pred_path):
            print(f"  ✗ 预测文件不存在: {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  ✗ 真实文件不存在: {gt_path}")
            continue

        # 保留原始患者ID作为文件夹名和DICOM元数据
        patient_out = os.path.join(OUTPUT_DIR, patient_id)
        dir_1 = os.path.join(patient_out, "1")
        dir_2 = os.path.join(patient_out, "2")

        # 同一患者共享Study UID，读片软件会将其归为同一Study下的两个Series
        # 支持上下翻看对比；每组Series有独立的Series UID
        study_uid = generate_uid()
        series_uid_1 = generate_uid()
        series_uid_2 = generate_uid()
        frame_ref_uid = generate_uid()

        # 序列1 (GT) 和序列2 (Pred) 各自独立计算缩放参数
        # 这样两组都能充分利用16-bit动态范围，避免预测图因共用GT参数而变暗
        print(f"  转换序列 1 -> {dir_1}")
        nifti_to_dicom_series(
            nii_path=gt_path,
            output_dicom_dir=dir_1,
            patient_anon_id=patient_id,
            study_uid=study_uid,
            series_uid=series_uid_1,
            series_number=1,
            frame_ref_uid=frame_ref_uid,
            series_desc="Series 1",
            mask_nii_path=mask_path
        )

        print(f"  转换序列 2 -> {dir_2}")
        nifti_to_dicom_series(
            nii_path=pred_path,
            output_dicom_dir=dir_2,
            patient_anon_id=patient_id,
            study_uid=study_uid,
            series_uid=series_uid_2,
            series_number=2,
            frame_ref_uid=frame_ref_uid,
            series_desc="Series 2",
            mask_nii_path=mask_path
        )

        success_count += 1
        print(f"  ✓ 完成\n")

    print(f"\n全部完成! 成功转换 {success_count}/{len(pred_files)} 个患者")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n目录结构:")
    print(f"  {OUTPUT_DIR}/")
    for f in pred_files:
        pid = f.replace('_pred.nii.gz', '')
        print(f"  ├── {pid}/")
        print(f"  │   ├── 1/    (DICOM序列)")
        print(f"  │   └── 2/    (DICOM序列)")


if __name__ == "__main__":
    main()
