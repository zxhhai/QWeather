import xarray as xr
import pandas as pd
from pathlib import Path
import re
import numpy as np
import time
import os
import sys
from datetime import datetime

# O3专用配置
SOURCE_DIR = Path(r"C:/RIP_D/Codes/Python/QWeather/DataProcess/datas/o3")
YEAR = 2018
DATE_PATTERN = re.compile(r"(\d{6})_omi_mls_tco\.nc$")  # 匹配月度文件
OUTPUT_FILE = f"combined_o3_{YEAR}_full_year.nc"
ENGINE = "netcdf4"
VAR_NAME = 'tco'  # O3数据变量名

print(f"源目录: {SOURCE_DIR}")
print(f"输出文件: {OUTPUT_FILE}")
print(f"处理变量: {VAR_NAME}")

# 记录开始时间
start_time = time.time()

# 1. 引擎验证
print("验证引擎可用性...")
try:
    sample_path = next(SOURCE_DIR.glob("*.nc"))
    with xr.open_dataset(sample_path, engine=ENGINE) as ds:
        print(f"引擎验证成功! 样本文件变量: {list(ds.data_vars)}")
        # 保存坐标信息用于后续创建完整数据集
        sample_lat = ds.latitude.values
        sample_lon = ds.longitude.values
except Exception as e:
    print(f"引擎验证失败: {e}")
    sys.exit(1)

# 2. 创建全年日期序列
print(f"生成{YEAR}年日期序列...")
start_date = datetime(YEAR, 1, 1)
end_date = datetime(YEAR, 12, 31)
full_dates = pd.date_range(start_date, end_date, freq='D')
print(f"共{len(full_dates)}天")

# 3. 创建空的全年数据集
# 根据样本文件确定网格大小
n_lat = len(sample_lat)
n_lon = len(sample_lon)
n_time = len(full_dates)

# 创建空数据立方体 (时间, 纬度, 纬度)
data_cube = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

# 创建xarray数据集
combined_ds = xr.Dataset(
    data_vars={
        VAR_NAME: (('time', 'latitude', 'longitude'), data_cube)
    },
    coords={
        'time': full_dates,
        'latitude': sample_lat,
        'longitude': sample_lon
    },
    attrs={
        'title': f'OMI/MLS Tropospheric Column O3 - Full Year {YEAR}',
        'institution': 'NASA/Goddard Space Flight Center',
        'source': 'Monthly OMI/MLS composites',
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'comment': 'Full year generated from monthly composites'
    }
)

# 4. 处理各月数据
print(f"开始处理月度数据并填充全年数据集...")
monthly_datasets = []

for month in range(1, 13):
    # [保持不变] 生成月份文件名
    month_str = f"{YEAR}{month:02d}"
    filename = f"{month_str}_omi_mls_tco.nc"
    file_path = SOURCE_DIR / filename
    
    if not file_path.exists():
        print(f"警告: 缺少{month_str}月份文件，使用NaN填充")
        continue
        
    try:
        # 打开月度文件
        with xr.open_dataset(file_path, engine=ENGINE) as monthly_ds:
            ###################################################
            # 关键修改：提取正确的二维数据形状
            # 原始方法：monthly_data = monthly_ds[VAR_NAME].values -> (1, 120, 288)
            ###################################################
            # 新方法：获取第一个时间步的数据 -> (120, 288)
            monthly_data = monthly_ds[VAR_NAME].isel(time=0).values
            
            # [保持不变] 获取月度文件的时间点
            monthly_time = pd.to_datetime(monthly_ds.time.values[0])
            
            # [保持不变] 确定当月的天数范围
            month_start = datetime(YEAR, month, 1)
            if month == 12:
                month_end = datetime(YEAR+1, 1, 1) - pd.Timedelta(days=1)
            else:
                month_end = datetime(YEAR, month+1, 1) - pd.Timedelta(days=1)
            
            # [保持不变] 计算填充的日期索引
            time_idx = (full_dates >= month_start) & (full_dates <= month_end)
            
            ###################################################
            # 关键修改：简化赋值逻辑
            # 原始方法：循环赋值 -> 效率低且可能导致形状不匹配
            ###################################################
            # 新方法：直接赋值给选定时间范围
            combined_ds[VAR_NAME][time_idx] = monthly_data
            
            print(f"处理 {month_str}: {month_start.strftime('%Y-%m-%d')} 到 {month_end.strftime('%Y-%m-%d')} 共 {time_idx.sum()} 天")
            
    except Exception as e:
        print(f"处理 {filename} 时出错: {str(e)[:100]}")

# 5. 保存合并后的数据集
print(f"保存合并数据到 {OUTPUT_FILE}...")
combined_ds.to_netcdf(OUTPUT_FILE, engine=ENGINE)
print("保存完成!")

# 验证输出文件
try:
    with xr.open_dataset(OUTPUT_FILE, engine=ENGINE) as check_ds:
        print("文件验证成功! 数据结构:")
        print(check_ds)
        print(f"时间维度: {len(check_ds.time)}天")
        print(f"空间分辨率: {len(check_ds.latitude)}x{len(check_ds.longitude)}")
except Exception as e:
    print(f"输出文件验证失败: {e}")

# 结果统计
elapsed_time = time.time() - start_time
print("\n" + "="*60)
print(f"处理完成! 总耗时: {elapsed_time:.2f} 秒")
print(f"平均每日数据处理时间: {elapsed_time/365:.4f} 秒")