import argparse

parser = argparse.ArgumentParser()
# =============================常改的参数=============================
parser.add_argument('--Task_name', type=str, default='CE_MRI_simulate_PCa')  # 任务名,也是文件名
parser.add_argument('--Task_id', type=str, default='1')
parser.add_argument('--cuda_idx', type=int, default=0)  # 用几号卡的显存（单卡推理时生效，多卡时以 cuda_idx_list 为准）
parser.add_argument('--cuda_idx_list', type=int, nargs='+', default=[4, 5], help='推理使用的GPU列表，例如 [4,5]')
parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')  # 交叉验证的折数
parser.add_argument('--fold_idx', type=int, default=1)  # 跑第几折的数据 1开始
parser.add_argument('--ckpt_name', type=str, default="checkpoint.ckpt", help="best/checkpoint/val_loss_best")
parser.add_argument('--net_mode', type=str, default="pix2pix_mulD")
# image
# parser.add_argument('--filepath_img', type=str, default=r'/data/newnas/MJY_file/CE-MRI/nii_data_norm_pre')
parser.add_argument('--filepath_img', type=str, default=r'pre-data/05_normalized')
# result&save
parser.add_argument('--result_path', type=str, default=r'results')
# training hyper-parameters
parser.add_argument('--seed', type=int, default=2023)  # 随机数的种子点，一般不变

# ==========新增：批量推理和可视化参数==========
parser.add_argument('--model_dir', type=str, default=None, help='模型文件夹路径，自动查找其中的checkpoint.ckpt')
parser.add_argument('--patients', type=str, nargs='+', default=None, help='指定要推理的患者ID列表，不指定则推理所有患者')
parser.add_argument('--output_dir', type=str, default=r'output', help='推理结果输出目录')
parser.add_argument('--visualize', action='store_true', help='是否生成输入输出的对比可视化图')
parser.add_argument('--sample_layers', type=int, default=None, help='可视化时均匀采样N层（不指定则处理全部）')
parser.add_argument('--specify_layers', type=int, nargs='+', default=None, help='指定要可视化的层号，空格分隔')
parser.add_argument('--test_data_dir', type=str, default=r'train-data/images_ts', help='推理测试数据目录，默认train-data/images_ts')
parser.add_argument('--gt_data_dir', type=str, default=r'pre-data/05_normalized', help='Ground Truth数据目录（用于计算指标和可视化）')

config, _ = parser.parse_known_args()
