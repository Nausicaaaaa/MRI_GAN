import os
import h5py
import numpy as np

# 配置路径
train_dir = '/mnt/data/KASR/Dengsiyi/MRI_GAN/train-data/images_tr'
target_key = 't1ce'

# 统计变量
all_values = []
min_vals, max_vals, mean_vals, std_vals = [], [], [], []
sample_count = 0
max_samples = 100  # 限制采样数量，避免内存溢出

print(f"开始扫描目录: {train_dir}")
print(f"目标键名: {target_key}")

# 遍历病人文件夹
for patient_id in sorted(os.listdir(train_dir)):
    if sample_count >= max_samples:
        break
    
    patient_path = os.path.join(train_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue
        
    # 遍历切片文件
    for layer_file in sorted(os.listdir(patient_path)):
        if sample_count >= max_samples:
            break
            
        if not layer_file.endswith('.h5'):
            continue
            
        file_path = os.path.join(patient_path, layer_file)
        try:
            with h5py.File(file_path, 'r') as f:
                if target_key in f:
                    data = f[target_key][()]
                    
                    # 收集统计信息
                    min_vals.append(np.min(data))
                    max_vals.append(np.max(data))
                    mean_vals.append(np.mean(data))
                    std_vals.append(np.std(data))
                    
                    # 随机采样部分像素值用于整体分布分析
                    flat_data = data.flatten()
                    if len(flat_data) > 1000:
                        sample_indices = np.random.choice(len(flat_data), 1000, replace=False)
                        all_values.extend(flat_data[sample_indices])
                    else:
                        all_values.extend(flat_data)
                        
                    sample_count += 1
                    if sample_count % 10 == 0:
                        print(f"已处理 {sample_count} 个样本...")
                        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

if all_values:
    all_values = np.array(all_values)
    
    print("\n" + "="*50)
    print(f"统计结果 (基于 {sample_count} 个切片样本):")
    print("="*50)
    
    print(f"\n单切片统计:")
    print(f"  最小值范围: [{np.min(min_vals):.6f}, {np.max(min_vals):.6f}]")
    print(f"  最大值范围: [{np.min(max_vals):.6f}, {np.max(max_vals):.6f}]")
    print(f"  均值范围:   [{np.min(mean_vals):.6f}, {np.max(mean_vals):.6f}]")
    print(f"  标准差范围: [{np.min(std_vals):.6f}, {np.max(std_vals):.6f}]")
    
    print(f"\n  平均最小值: {np.mean(min_vals):.6f}")
    print(f"  平均最大值: {np.mean(max_vals):.6f}")
    print(f"  平均均值:   {np.mean(mean_vals):.6f}")
    print(f"  平均标准差: {np.mean(std_vals):.6f}")
    
    print(f"\n全局像素统计 (采样 {len(all_values)} 个像素点):")
    print(f"  全局最小值: {np.min(all_values):.6f}")
    print(f"  全局最大值: {np.max(all_values):.6f}")
    print(f"  全局均值:   {np.mean(all_values):.6f}")
    print(f"  全局标准差: {np.std(all_values):.6f}")
    
    # 百分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n全局像素百分位数:")
    for p in percentiles:
        print(f"  {p}%: {np.percentile(all_values, p):.6f}")
        
    # 检查是否有异常值
    q1, q3 = np.percentile(all_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((all_values < lower_bound) | (all_values > upper_bound)).sum()
    print(f"\n异常值检测 (IQR方法):")
    print(f"  下界: {lower_bound:.6f}")
    print(f"  上界: {upper_bound:.6f}")
    print(f"  异常值数量: {outliers} ({outliers/len(all_values)*100:.2f}%)")
    
else:
    print("未找到任何有效数据！")
