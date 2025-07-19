import pandas as pd
from pathlib import Path
import re
import shutil
from datetime import datetime, timedelta
import numpy as np
import time
import os
import sys
import xarray as xr

# 配置设置
VA = 'SST'
NA = r"_L2MG_SST[ND]Q_010c_SST"
SOURCE_DIR = Path(r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\sst")
TARGET_DIR = Path(r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\data_m\sst_m")
YEAR = 2018
DATE_PATTERN = re.compile(r"GC1SG1_(\d{8})\d+" + NA + r"\.nc$")
OUTPUT_FILE = f"combined_sst_{YEAR}_full_year.nc"
ENGINE = "netcdf4"  # 指定读取引擎

# 创建目标目录
TARGET_DIR.mkdir(exist_ok=True, parents=True)
print(f"源目录: {SOURCE_DIR}")
print(f"目标目录: {TARGET_DIR}")
print(f"使用引擎: {ENGINE}")

# 记录开始时间
start_time = time.time()

# 1. 检查引擎可用性
print("验证引擎可用性...")
try:
    # 尝试打开样本文件验证引擎
    sample_path = next(SOURCE_DIR.glob("*.nc"))
    with xr.open_dataset(sample_path, engine=ENGINE) as ds:
        print(f"引擎验证成功! 样本文件变量: {list(ds.data_vars)}")
except Exception as e:
    print(f"引擎验证失败: {e}")
    print("可能的原因:")
    print("1. 缺少netCDF4库: 请运行 'pip install netCDF4'")
    print("2. 文件格式错误")
    sys.exit(1)

# 2. 预先处理所有文件日期映射
print("分析文件名并建立日期映射...")
available_files = list(SOURCE_DIR.glob("*.nc"))
if not available_files:
    raise FileNotFoundError(f"在目录 {SOURCE_DIR} 中没有找到任何.nc文件")

date_to_path = {}
for f in available_files:
    if match := DATE_PATTERN.match(f.name):
        date_str = match.group(1)
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            # 记录文件路径 - 不实际打开文件
            date_to_path[file_date] = f
        except Exception as e:
            print(f"忽略无效文件 {f.name}: {e}")
            continue

if not date_to_path:
    raise ValueError("没有找到任何有效的数据文件")

print(f"找到 {len(date_to_path)} 个有效文件")

# 3. 生成全年日期序列
start_date = datetime(YEAR, 1, 1)
all_dates = [start_date + timedelta(days=i) for i in range(365)]
target_dates = [d.date() for d in all_dates]

# 4. 用于查找最近日期的辅助列表
sorted_available_dates = sorted(date_to_path.keys())
print(f"日期范围: {sorted_available_dates[0]} 至 {sorted_available_dates[-1]}")

# 5. 复制和重命名文件，并进行验证
print(f"开始创建365天完整数据集并进行文件验证...")
valid_count = 0
invalid_files = []

for i, target_date in enumerate(target_dates):
    # 创建目标文件名
    target_filename = f"GC1SG1_{target_date.strftime('%Y%m%d')}AAAAAAAAAA" + NA + ".nc"
    target_path = TARGET_DIR / target_filename
    
    if target_date in date_to_path:
        source_path = date_to_path[target_date]
        action = "复制原始文件"
    else:
        # 查找最近日期（向前查找）
        pos = np.searchsorted(sorted_available_dates, target_date)
        closest_date = sorted_available_dates[0] if pos == 0 else sorted_available_dates[pos - 1]
        source_path = date_to_path[closest_date]
        action = f"基于 {closest_date.strftime('%Y-%m-%d')} 创建"
    
    # 复制文件
    shutil.copy2(source_path, target_path)
    
    # 验证文件能否读取
    try:
        with xr.open_dataset(target_path, engine=ENGINE) as ds:
            # 检查必要变量是否存在
            if VA not in ds.data_vars:
                raise ValueError(f"缺失关键变量" + VA)
                
        valid_count += 1
        valid_str = "✓ 验证成功"
    except Exception as e:
        invalid_files.append(target_path.name)
        valid_str = f"✗ 验证失败: {str(e)}"
        # 删除无效文件（可选）
        try:
            os.remove(target_path)
            valid_str += " (已删除)"
        except:
            valid_str += " (删除失败)"
    
    # 进度显示
    if (i+1) % 10 == 0 or (i+1) == 365:
        percent = (i+1) / 365 * 100
        status = f"{i+1}/365 ({percent:.1f}%) | {valid_str}"
        print(f"{target_date.strftime('%Y-%m-%d')}: {action.ljust(35)} | {status}")

# 结果统计
elapsed_time = time.time() - start_time
print("\n" + "="*60)
print(f"处理完成! 总耗时: {elapsed_time:.2f} 秒")
print(f"成功创建并验证文件: {valid_count}/365")
print(f"平均每文件处理时间: {elapsed_time/365:.4f} 秒")

if invalid_files:
    print("\n警告: 以下文件验证失败:")
    for f in invalid_files:
        print(f"  - {f}")
else:
    print("\n所有文件验证成功!")

# 最终验证统计
print("\n最终状态:")
print(f"目标目录文件数: {len(list(TARGET_DIR.glob('*.nc')))}/365")
if valid_count == 365:
    print("准备进行多文件合并操作")
else:
    print("警告: 存在无效文件，不推荐进行多文件合并")