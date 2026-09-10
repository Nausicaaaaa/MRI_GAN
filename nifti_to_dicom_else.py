#!/usr/bin/env python3
"""
Convert other sequences (ADC, DWI, PreT1, T1, T2) for patients in output_dicom
to DICOM format. Output to output_dicom_else/{sequence}/{patient_id}/
"""

import os
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid as dicom_generate_uid
from datetime import datetime

PROJECT_ROOT = "/mnt/data/KASR/Dengsiyi/MRI_GAN"
OUTPUT_DICOM_DIR = os.path.join(PROJECT_ROOT, "output_dicom")
GT_DIR = os.path.join(PROJECT_ROOT, "pre-data", "05_normalized")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_dicom_else")

SEQUENCES = ["ADC", "DWI", "PreT1", "T1", "T2"]


def generate_uid():
    return dicom_generate_uid()


def nifti_to_dicom_series(nii_path, output_dicom_dir, patient_id="P001",
                          study_uid=None, series_uid=None, series_number=1,
                          frame_ref_uid=None, series_desc="MRI"):
    os.makedirs(output_dicom_dir, exist_ok=True)
    nii_image = sitk.ReadImage(nii_path)

    # 重采样到平面内等间距，避免DICOM查看器中图像被拉伸
    orig_spacing = list(nii_image.GetSpacing())
    iso_spacing = min(orig_spacing[0], orig_spacing[1])
    target_spacing = [iso_spacing, iso_spacing, orig_spacing[2]]
    orig_size = nii_image.GetSize()
    new_size = [
        int(round(orig_size[0] * orig_spacing[0] / target_spacing[0])),
        int(round(orig_size[1] * orig_spacing[1] / target_spacing[1])),
        orig_size[2]
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(nii_image.GetOrigin())
    resampler.SetOutputDirection(nii_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    nii_image = resampler.Execute(nii_image)

    nii_array = sitk.GetArrayFromImage(nii_image)
    num_slices = nii_array.shape[0]

    data_min = nii_array.min()
    data_max = nii_array.max()
    if data_max - data_min > 1e-8:
        scaled = (nii_array - data_min) / (data_max - data_min) * 65535
    else:
        scaled = np.zeros_like(nii_array)
    scaled = np.clip(scaled, 0, 65535).astype(np.uint16)

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
        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = "19000101"
        ds.PatientSex = "O"
        ds.SliceThickness = round(float(spacing[2]), 6)
        ds.ProtocolName = "MRI"

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
        ds.PixelSpacing = [spacing[0], spacing[1]]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.WindowCenter = 32768
        ds.WindowWidth = 65536

        ds.PixelData = slice_data.tobytes()
        ds.save_as(os.path.join(output_dicom_dir, f"slice_{z:04d}.dcm"))

    print(f"    saved {num_slices} DICOM slices to {output_dicom_dir}")


def main():
    patient_ids = sorted([
        d for d in os.listdir(OUTPUT_DICOM_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DICOM_DIR, d))
    ])
    print(f"output_dicom has {len(patient_ids)} patients")

    total_tasks = 0
    for pid in patient_ids:
        norm_dir = os.path.join(GT_DIR, pid)
        if os.path.isdir(norm_dir):
            for seq in SEQUENCES:
                nii_path = os.path.join(norm_dir, f"{seq}.nii.gz")
                if os.path.exists(nii_path):
                    total_tasks += 1

    print(f"Total conversion tasks: {total_tasks}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0
    for seq in SEQUENCES:
        seq_out_dir = os.path.join(OUTPUT_DIR, seq)
        os.makedirs(seq_out_dir, exist_ok=True)
        print(f"=== Sequence: {seq} ===")

        for i, pid in enumerate(patient_ids, 1):
            norm_dir = os.path.join(GT_DIR, pid)
            nii_path = os.path.join(norm_dir, f"{seq}.nii.gz")

            if not os.path.exists(nii_path):
                print(f"  [{i}/{len(patient_ids)}] {pid}: NIfTI not found, skip")
                continue

            patient_out_dir = os.path.join(seq_out_dir, pid)
            study_uid = generate_uid()
            series_uid = generate_uid()
            frame_ref_uid = generate_uid()

            print(f"  [{i}/{len(patient_ids)}] {pid}: converting {seq}")
            try:
                nifti_to_dicom_series(
                    nii_path=nii_path,
                    output_dicom_dir=patient_out_dir,
                    patient_id=pid,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    series_number=1,
                    frame_ref_uid=frame_ref_uid,
                    series_desc=seq
                )
                success_count += 1
            except Exception as e:
                print(f"    FAILED: {e}")
                fail_count += 1

        print()

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}, Total: {total_tasks}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
