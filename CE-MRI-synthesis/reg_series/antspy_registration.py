import json
import logging
import os

import SimpleITK
import ants
import nibabel as nib


def get_nii_file_name(sdict, workdir):
    series_file_list = os.listdir(workdir)
    tmp_list = series_file_list.copy()
    
    # 先处理T1，确保T1优先匹配，避免被其他序列（如PreT1）占用
    series_order = sorted(sdict.keys())
    if "T1" in series_order:
        series_order.remove("T1")
        series_order.insert(0, "T1")
    
    for series in series_order:
        if "filename" not in sdict[series].keys():
            # 遍历json的可能的名字序列
            for series_dict_name in sdict[series]["name"]:
                if "filename" in sdict[series].keys():
                    continue
                # 找对应的实际文件名
                for series_file_name in series_file_list:
                    # json名字对应上实际文件名
                    # 使用更精确的匹配：检查文件名是否以该序列名开头（避免T1匹配到PreT1）
                    if series_dict_name in series_file_name:
                        # 额外检查：如果是T1，确保不是PreT1
                        if series == "T1" and "PreT1" in series_file_name:
                            continue
                        sdict[series].update({"filename": os.path.join(workdir, series_file_name)})
                        break
        # 若没读到filename，标记为缺失而不是报错
        if "filename" not in sdict[series].keys():
            logger = logging.getLogger("my_logger")
            logger.warning("  警告: 未找到序列 {} 的文件".format(series))
            sdict[series].update({"filename": "None", "missing": True})
    return sdict


def get_series(workdir, params, after_resample_t1=True):
    assert os.path.isfile(params), "Param file does not exist at " + params
    with open(params, 'r') as f:
        json_str = f.read()
    sdict = json.loads(json_str)
    sdict = get_nii_file_name(sdict, workdir)
    sdict.update({"info": {
        "filename": "None",
        "dcmdir": workdir,
        "id": workdir.split(os.sep)[-1],
    }})
    # t1 redirect to resampled_t1
    # t1 dict的文件名改成重采样的t1，使用重采样后的作为模板
    if after_resample_t1:
        dir_temp = sdict["info"]["dcmdir"].split('/')
        dir_temp[-2] = "CE-MRI-PCa-resample-gz"
        resample_name = '/'.join(dir_temp)
        ### 对上resample里的正确的名字
        for name in os.listdir(resample_name):
            if "T1.nii.gz" in name or "t1fs" in name:
                dir_temp.append(name)
                break
        resample_name = '/'.join(dir_temp)
        ###
        sdict["T1"].update({"filename": resample_name})

        assert os.path.isfile(resample_name), "Resampled T1 does not exist"

    return sdict


def make_log(work_dir, repeat=False):
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    # make log file, append to existing
    idno = os.path.basename(work_dir)
    log_file = os.path.join(work_dir, idno + "_log_PCa.txt")
    if repeat:
        open(log_file, 'w').close()
    else:
        open(log_file, 'a').close()
    # make logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # should this be DEBUG?
    # set all existing handlers to null to prevent duplication
    logger.handlers = []
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterch = logging.Formatter('%(message)s')
    formatterfh = logging.Formatter("[%(asctime)s]  [%(levelname)s]:     %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatterch)
    fh.setFormatter(formatterfh)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info("####################### STARTING NEW LOG #######################")


def reg_series(ser_dict, repeat=False):
    # logging
    logger = logging.getLogger("my_logger")
    logger.info("REGISTERING IMAGES:")
    # dcm_dir prep
    dcm_dir = ser_dict["info"]["dcmdir"]
    # sort serdict keys so that the atlas reg comes up first - this makes sure atlas registration is first
    # T1必须是第一个作为参考图像
    sorted_keys = ["T1"] if "T1" in ser_dict.keys() else []
    for key in sorted(ser_dict.keys()):
        # 跳过info、T1和缺失的序列
        if key == "info" or key == "T1":
            continue
        # 让ADC和DWI最后处理
        if key == "ADC" or key == "DWI":
            continue
        else:
            sorted_keys.append(key)
    # DWI在ADC之前处理（因为ADC可能继承DWI的变换）
    if "DWI" in ser_dict.keys():
        sorted_keys.append("DWI")
    if "ADC" in ser_dict.keys():
        sorted_keys.append("ADC")

    # if reg is false, or if there is no input file found, then just make the reg filename same as unreg filename
    # 处理所有序列（包括T1）
    all_series = [k for k in ser_dict.keys() if k != "info"]
    for ser in all_series:
        # 跳过缺失的序列
        if ser_dict[ser].get("missing", False) or ser_dict[ser]["filename"] == "None":
            ser_dict[ser].update({"filename_reg": "None"})
            ser_dict[ser].update({"transform": "None"})
            ser_dict[ser].update({"reg": False})
            continue
        # first, if there is no filename, set to None
        if "filename" not in ser_dict[ser].keys():
            ser_dict[ser].update({"filename": "None"})
        # T1作为参考图像，filename_reg就是原始文件名
        if ser == "T1":
            ser_dict[ser].update({"filename_reg": ser_dict[ser]["filename"]})
            ser_dict[ser].update({"transform": "None"})
        elif ser_dict[ser]["filename"] == "None" or "reg" not in ser_dict[ser].keys() or not ser_dict[ser]["reg"]:
            ser_dict[ser].update({"filename_reg": ser_dict[ser]["filename"]})
            ser_dict[ser].update({"transform": "None"})
            ser_dict[ser].update({"reg": False})
        # if reg True, then do the registration using translation, affine, nonlin, or just applying existing transform
    # handle registration
    for ser in sorted_keys:
        # 跳过缺失的序列
        if ser_dict[ser].get("missing", False) or ser_dict[ser]["filename"] == "None":
            continue
        # 跳过没有reg属性的序列（如T1参考图像）
        if not ser_dict[ser].get("reg"):
            continue
        if ser_dict[ser]["reg"] not in sorted_keys:
            trans_method = ser_dict[ser]["reg"]
            if os.path.isfile(ser_dict[ser]["reg_target"]):
                template = ser_dict[ser]["reg_target"]
            else:
                template = ser_dict[ser_dict[ser]["reg_target"]]["filename_reg"]
            # handle surrogate moving image
            if "reg_moving" in ser_dict[ser]:
                movingr = ser_dict[ser]["reg_moving"]
                movinga = ser_dict[ser]["filename"]
            else:
                movingr = ser_dict[ser]["filename"]
                movinga = ser_dict[ser]["filename"]
            # handle registration options here
            if "reg_option" in ser_dict[ser].keys():
                option = ser_dict[ser]["reg_option"]
            else:
                option = None
            transforms = get_reg_transform(moving_nii=movingr,
                                           template_nii=template,
                                           work_dir=dcm_dir,
                                           type_of_transform=trans_method,
                                           option=option, )
            transforms_file = transforms['fwdtransforms']
            # handle interp option
            if "interp" in ser_dict[ser].keys():
                interp = ser_dict[ser]["interp"]
            else:
                interp = 'linear'
            niiout = ants_apply(movinga, template, interp, transforms_file, dcm_dir)
            ser_dict[ser].update({"filename_reg": niiout})
            ser_dict[ser].update({"transform": transforms_file})
        # 如果是用别的序列的配准方法
        elif ser_dict[ser]["reg"] in sorted_keys:
            # 检查目标序列是否存在且已配准
            target_ser = ser_dict[ser]["reg"]
            if target_ser not in ser_dict or ser_dict[target_ser].get("missing", False):
                logger.warning("  目标序列 {} 缺失，跳过 {} 的配准".format(target_ser, ser))
                ser_dict[ser].update({"filename_reg": ser_dict[ser]["filename"]})
                ser_dict[ser].update({"transform": "None"})
                continue
            transforms_file = ser_dict[target_ser]["transform"]
            template = ser_dict[target_ser]["filename_reg"]
            moving = ser_dict[ser]["filename"]
            # 插值
            if "interp" in ser_dict[ser].keys():
                interp = ser_dict[ser]["interp"]
            else:
                interp = 'linear'
            niiout = ants_apply(moving, template, interp, transforms_file, dcm_dir)
            ser_dict[ser].update({"filename_reg": niiout})
            ser_dict[ser].update({"transform": transforms_file})
    return ser_dict


def get_reg_transform(moving_nii, template_nii, work_dir, type_of_transform, option=None):
    logger = logging.getLogger("my_logger")
    # 给transform文件建文件夹
    transform_file_folder = os.path.join(work_dir, 'trans_file')
    if not os.path.exists(transform_file_folder):
        os.makedirs(transform_file_folder)
    # moving和模板
    moving_name = os.path.basename(moving_nii).split(".")[0]
    template_name = os.path.basename(template_nii).split(".")[0]
    # transform文件名
    outprefix = os.path.join(transform_file_folder, moving_name + "_2_" + template_name + "_")
    # initial_transform_option = option["reg_com"] if isinstance(option, dict) and "reg_com" in option.keys() else 1 #不知道什么参数
    moving_img = ants.image_read(moving_nii)
    template_img = ants.image_read(template_nii)
    transformation = ants.registration(fixed=template_img,
                                       moving=moving_img,
                                       outprefix=outprefix,
                                       type_of_transform=type_of_transform,
                                       write_composite_transform=True,
                                       random_seed=2023,
                                       # aff_smoothing_sigmas=[6, 4, 1, 0],  # 有问题  出现对不齐  缺角
                                       reg_iterations=(1000, 500, 250, 50),  # 效果差不多
                                       aff_iterations=(1000, 1000, 1000, 1000),  # 效果差不多
                                       aff_shrink_factors=(4, 3, 2, 1),  # 效果差不多
                                       grad_step=0.1,  # 就那样
                                       # aff_metric="CC",
                                       # # syn_metric="CC",
                                       aff_sampling=32,  # 差不多
                                       syn_sampling=32,  # 效果差不多
                                       aff_random_sampling_rate=0.25,
                                       verbose=False,
                                       multivariate_extras=None,
                                       restrict_transformation=None,
                                       smoothing_in_mm=False,
                                       )
    logger.info("- Registering image " + moving_nii + " to " + template_nii)
    return transformation


def ants_apply(moving, fixed, interp, transform_list, work_dir, reg_dir=None):
    # logging
    logger = logging.getLogger("my_logger")
    # 获取registration保存的位置
    if reg_dir is None:
        # 尝试从环境变量获取，或使用默认路径
        import os
        reg_dir = os.environ.get('MRI_REG_DIR', None)
        if reg_dir is None:
            dir_temp = work_dir.split('/')
            dir_temp[-2] = "CE-MRI-PCa-new-reg"
            reg_dir = '/'.join(dir_temp)
    # 不存在就建造
    if not os.path.exists(reg_dir):
        os.makedirs(reg_dir)
    # enforce list
    if not isinstance(moving, list):
        moving = [moving]
    if not isinstance(transform_list, list):
        transform_list = [transform_list]
    # create output list of same shape
    output_nii = moving
    # define extension
    ext = ".nii.gz"
    # for loop for applying reg
    for ind, mvng in enumerate(moving, 0):
        # get ants img
        moving_img = ants.image_read(mvng)
        fixed_img = ants.image_read(fixed)
        # define output path
        output_nii[ind] = os.path.join(reg_dir, os.path.basename(mvng).split(ext)[0] + '_w.nii.gz')
        # do registration if not already done
        warped_image = ants.apply_transforms(fixed=fixed_img, moving=moving_img, transformlist=transform_list,
                                             interpolator=interp)
        logger.info("- Creating warped image " + output_nii[ind])
        # 保存配准后的图像 (使用ants.image_write)
        ants.image_write(warped_image, output_nii[ind])
    # if only 1 label, don't return array
    if len(output_nii) == 1:
        output_nii = output_nii[0]
    return output_nii


def move_resampled_t1_to_regfolder(series_dict):
    logger = logging.getLogger("my_logger")
    t1_image_name = series_dict["T1"]["filename"]
    t1_image = SimpleITK.ReadImage(t1_image_name)
    
    # 调试信息
    logger.info("- Moving T1 to reg folder")
    logger.info("  T1 input filename: " + t1_image_name)
    
    # 获取输出目录：优先使用T1CE的目录，其次是PreT1、T2，最后是环境变量
    reg_new_dir = None
    target_ser = None
    
    for ser_name in ["T1CE", "PreT1", "T2"]:
        if ser_name in series_dict and series_dict[ser_name].get("filename_reg"):
            filename_reg = series_dict[ser_name]["filename_reg"]
            if filename_reg and filename_reg != "None" and os.path.exists(filename_reg):
                reg_new_dir = os.path.dirname(filename_reg)
                target_ser = ser_name
                break
    
    if reg_new_dir is None:
        # 使用环境变量或原始目录
        reg_new_dir = os.environ.get('MRI_REG_DIR', os.path.dirname(t1_image_name))
        logger.info("  Using ENV/T1 directory: " + reg_new_dir)
    else:
        logger.info(f"  Using {target_ser} directory: " + reg_new_dir)
    
    # 确保输出目录存在
    if not os.path.exists(reg_new_dir):
        os.makedirs(reg_new_dir, exist_ok=True)
        logger.info("  Created directory: " + reg_new_dir)
    
    reg_filename = os.path.join(reg_new_dir, "T1_w.nii.gz")
    logger.info("  Writing T1 to: " + reg_filename)
    
    try:
        SimpleITK.WriteImage(t1_image, reg_filename)
        series_dict["T1"].update({"filename_reg": reg_filename})
        logger.info("- Successfully copied T1 to " + reg_filename)
    except Exception as e:
        logger.error("  Failed to write T1: " + str(e))
        raise

    tmp_dir = series_dict["info"]["dcmdir"]
    tmp_list = tmp_dir.split('/')
    tmp_list[-2] = os.path.basename(os.path.dirname(reg_new_dir))
    tmp_dir = "/".join(tmp_list)
    series_dict["info"].update({"dcmdir": tmp_dir})
    return series_dict


def window_width_and_level(series_dict):
    print("set window and level")
    for ser in series_dict.keys():
        if ser == "B0":
            img = SimpleITK.ReadImage(series_dict[ser]["filename"])
            img_array = SimpleITK.GetArrayFromImage(img)
            img_array[img_array > 500] = 500
            img_new = SimpleITK.GetImageFromArray(img_array)
            img_new.CopyInformation(img)
            SimpleITK.WriteImage(img_new, series_dict[ser]["filename"])
        if ser == "DWI":
            img = SimpleITK.ReadImage(series_dict[ser]["filename"])
            img_array = SimpleITK.GetArrayFromImage(img)
            img_array[img_array > 40] = 40
            img_new = SimpleITK.GetImageFromArray(img_array)
            img_new.CopyInformation(img)
            SimpleITK.WriteImage(img_new, series_dict[ser]["filename"])
    return series_dict
