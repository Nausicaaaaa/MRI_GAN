"""
MRI 身体区域获取 - 适用于 DICOM 格式数据
数据结构: data/患者ID/AX_T1/IM*
"""
import glob
import os

import SimpleITK as sitk
import cv2
import numpy as np


def morph_operation(img, kernel_size=(5, 5), anchor=(-1, -1), operation_type=cv2.MORPH_OPEN):
    body_mask = sitk.GetArrayFromImage(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    mask_final = body_mask.copy()

    for i in range(body_mask.shape[0]):
        if np.max(body_mask[i, :, :]) > 0:
            if operation_type == "erode":
                mask_final[i, :, :] = cv2.erode(body_mask[i, :, :], kernel, anchor=anchor, iterations=1)
            elif operation_type == "dilate":
                mask_final[i, :, :] = cv2.dilate(body_mask[i, :, :], kernel, anchor=anchor, iterations=1)
            else:
                mask_final[i, :, :] = cv2.morphologyEx(body_mask[i, :, :], operation_type, kernel, iterations=1)
    mask_final = sitk.GetImageFromArray(mask_final.astype(np.uint8))
    return mask_final


def fill_inter_bone(mask):
    # 对一张图像做孔洞填充，读入的是一层
    mask = mask_fill = mask.astype(np.uint8)
    if np.sum(mask[:]) != 0:  # 即读入图层有值
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)
        mask_fill = sum(contour_list)
        mask_fill[mask_fill >= 1] = 1
    return mask_fill.astype(np.uint8)


def fill_inter_3D(mask, other_axis=True):
    # 对3D图像做孔洞填充，即三个维度的fill_inter_bone
    if not isinstance(mask, np.ndarray):
        mask = sitk.GetArrayFromImage(mask)
    mask_final = mask.copy()
    for i in range(mask.shape[0]):
        if np.max(mask[i, :, :]) > 0:
            mask_final[i, :, :] = fill_inter_bone(mask_final[i, :, :])
    if other_axis:
        for i in range(mask.shape[1]):
            if np.max(mask[:, i, :]) > 0:
                mask_final[:, i, :] = fill_inter_bone(mask_final[:, i, :])
        for i in range(mask.shape[2]):
            if np.max(mask[:, :, i]) > 0:
                mask_final[:, :, i] = fill_inter_bone(mask_final[:, :, i])
    return mask_final.astype(np.uint8)

def fill_inter_3D_with_wall(mask, other_axis=True, wall_dim=1):
    if not isinstance(mask, np.ndarray):
        mask = sitk.GetArrayFromImage(mask)
    for ind in range(mask.shape[0]):
        if wall_dim == 1:
            mask[ind, :, 0] = 1
            mask[ind, :, -1] = 1
        else:
            mask[ind, 0, :] = 1
    return fill_inter_3D(mask, other_axis)


def getmaxcomponent(mask_array, min_size=1e4, check_num=10000, print_num=False, id_num=None):
    """
    获取最大连通域
    :reg_param mask_array: 输入的二值mask image模式
    :reg_param min_size: 最大连通域最小包含的体素数量
    :reg_param check_num: 检查多少个连通域
    :reg_param print_num: 说明有多少连通域
    :return:
    """
    # sitk方法，更快,得到的是相当于ski的connectivity=3的结果
    # 建立报错文件
    error_f = r'./body_error.txt'
    if isinstance(mask_array, np.ndarray):
        mask_array = sitk.GetImageFromArray(mask_array)
    cca = sitk.ConnectedComponentImageFilter()
    # cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = mask_array
    output_ex = cca.Execute(_input)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    if num <= check_num:
        check_num = num
        # print(check_num)
    # 获得连通域(ski方法)
    # labeled_img, num = ski.measure.label(mask_array, connectivity=3, return_num=True)
    max_label = 1
    max_num = 0
    for i in range(1, check_num + 1):  # 不必全部遍历，一般在前面就有对应的label，减少计算时间
        if np.sum(labeled_img == i) < min_size:  # 小于设置的最小体素数量，直接不计
            continue
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    if print_num:
        print(str(num) + '/' + str(max_label) + ':' + str(np.sum(labeled_img == max_label)))  # 看第几个是最大的
    if np.sum(labeled_img == max_label) < min_size:  # 最终大小还是小于设定值，说明check_num设太小了
        print("Don't get the right component!! size：" + str(np.sum(labeled_img == max_label)))
        if id_num:
            with open(error_f, mode="a+") as f:
                print(id_num, file=f)
            f.close()

    maxcomponent = (labeled_img == max_label).astype(np.uint8)
    maxcomponent = sitk.GetImageFromArray(maxcomponent)
    return maxcomponent


def read_dicom_series(dicom_dir):
    """
    读取 DICOM 序列并返回 3D SimpleITK 图像
    :param dicom_dir: DICOM 文件夹路径
    :return: SimpleITK Image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


if __name__ == '__main__':
    # ==================== 配置参数 ====================
    data_root = "/mnt/data/KASR/Dengsiyi/MRI_GAN/data"
    series_name = "AX_T1"
    body_save_folder_path = "/mnt/data/KASR/Dengsiyi/MRI_GAN/pre-data/body_masks"
    # ==================================================

    # 创建保存目录
    if not os.path.exists(body_save_folder_path):
        os.makedirs(body_save_folder_path)

    # 获取所有患者ID文件夹
    patient_dirs = [d for d in glob.glob(os.path.join(data_root, "P*")) if os.path.isdir(d)]
    patient_dirs.sort()

    print(f"找到 {len(patient_dirs)} 个患者文件夹")
    print(f"处理序列: {series_name}")
    print(f"保存路径: {body_save_folder_path}")
    print("-" * 50)

    for i, patient_dir in enumerate(patient_dirs, 1):
        patient_id = os.path.basename(patient_dir)
        dicom_dir = os.path.join(patient_dir, series_name)

        # 检查 DICOM 文件夹是否存在
        if not os.path.exists(dicom_dir):
            print(f"[{i}/{len(patient_dirs)}] 跳过 {patient_id}: 未找到 {series_name} 序列")
            continue

        # 检查是否有 DICOM 文件
        dicom_files = glob.glob(os.path.join(dicom_dir, "IM*"))
        if len(dicom_files) == 0:
            print(f"[{i}/{len(patient_dirs)}] 跳过 {patient_id}: {series_name} 文件夹为空")
            continue

        print(f"[{i}/{len(patient_dirs)}] 处理 {patient_id}: 找到 {len(dicom_files)} 个 DICOM 文件")

        try:
            # 读取 DICOM 序列
            img = read_dicom_series(dicom_dir)
            print(f"    图像尺寸: {img.GetSize()}")

            # 获取图像数组
            img_array = sitk.GetArrayFromImage(img)

            # 阈值分割 - 获取身体mask
            ret = 150  
            new_img = (img_array > ret).astype(np.uint8)

            # 填补孔洞
            new_img = fill_inter_3D(new_img)

            # 做一次开运算 去伪影
            new_img = sitk.GetImageFromArray(new_img)
            new_img = morph_operation(new_img, kernel_size=(7, 7), operation_type=cv2.MORPH_OPEN)

            # 取最大连通域
            new_img = getmaxcomponent(new_img,
                                      min_size=1e4,
                                      check_num=50,
                                      print_num=False,
                                      id_num=None)

            # 填补孔洞
            new_img = fill_inter_3D_with_wall(new_img, other_axis=True, wall_dim=1)

            # 转sitk操作
            new_img = sitk.GetImageFromArray(new_img)

            # 腐蚀 去掉细伪影
            new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="erode")
            new_img = sitk.GetArrayFromImage(new_img)
            new_img = fill_inter_3D_with_wall(new_img, other_axis=True, wall_dim=2)

            # 开运算 多一次去掉细伪影
            new_img = sitk.GetImageFromArray(new_img)
            new_img = morph_operation(new_img, kernel_size=(9, 9), operation_type=cv2.MORPH_OPEN)

            # 去除细伪影后去除小连通域
            new_img = getmaxcomponent(new_img, min_size=1e4, check_num=50, print_num=False)

            # 去除连通域后膨胀再闭运算
            new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type="dilate")

            # 闭运算 填补可能出现的细节缺漏
            new_img = morph_operation(new_img, kernel_size=(11, 11), operation_type=cv2.MORPH_CLOSE)

            # 膨胀（左右少,上下多，前后更少）获得大一点的mask 提高容错
            new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type="dilate")

            # 闭开运算
            new_img = morph_operation(new_img, kernel_size=(5, 5), operation_type=cv2.MORPH_CLOSE)
            new_img = sitk.BinaryMorphologicalOpening(new_img, (3, 3, 3))

            # 保存身体掩码
            save_nii = os.path.join(body_save_folder_path, patient_id + "_body.nii.gz")
            sitk.WriteImage(new_img, save_nii)

            print(f"    ✓ 已保存: {save_nii}")

        except Exception as e:
            print(f"    ✗ 处理失败: {str(e)}")
            continue

    print("-" * 50)
    print("处理完成!")
