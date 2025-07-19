import xarray as xr
import numpy as np
import pandas as pd
import os
import re
import time
from pathlib import Path
import netCDF4 as nc
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 配置设置 - 修改SOURCE_DIR为你的SST数据目录
SOURCE_DIR = Path(r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\data_m\sst_m")
YEAR = 2018
OUTPUT_FILE = f"combined_sst_{YEAR}_full_year.nc"
CHUNK_SIZE = 50  # 每批处理的文件数

logger.info(f"开始合并 SST 数据集...")
start_time = time.time()

# 1. 获取文件列表并识别时间信息
logger.info("分析文件名并提取时间信息...")
files = list(SOURCE_DIR.glob("*.nc"))
if not files:
    raise FileNotFoundError(f"在目录 {SOURCE_DIR} 中没有找到任何.nc文件")

# 更灵活的文件名解析方法
def extract_date_from_filename(filename):
    """从文件名中提取日期信息"""
    # 尝试匹配8位数字序列（YYYYMMDD格式）
    match = re.search(r"(\d{8})", filename)
    if match:
        date_str = match.group(1)
        try:
            # 将日期字符串转换为datetime64
            return np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        except:
            return None
    return None

# 创建时间索引字典
file_info = {}
for file in files:
    date_obj = extract_date_from_filename(file.name)
    if date_obj is not None:
        file_info[date_obj] = file

if not file_info:
    logger.error("没有找到任何包含有效时间信息的文件")
    # 打印所有文件名以便调试
    logger.info("所有文件名:")
    for file in files:
        logger.info(f" - {file.name}")
    raise ValueError("没有找到任何包含有效时间信息的文件")

logger.info(f"找到 {len(file_info)} 个有效文件")
logger.info(f"日期范围: {min(file_info.keys())} 到 {max(file_info.keys())}")

# 2. 创建365天的完整时间序列
all_dates = [np.datetime64(f"{YEAR}-01-01") + np.timedelta64(i, 'D') for i in range(365)]

# 3. 创建输出文件结构（使用样本文件）
logger.info("创建输出文件结构...")
sample_path = next(iter(file_info.values()))
with xr.open_dataset(sample_path) as sample_ds:
    # 获取维度信息
    lat_dim = len(sample_ds.Latitude)
    lon_dim = len(sample_ds.Longitude)
    time_dim = len(all_dates)
    
    logger.info(f"网格大小: {lat_dim}×{lon_dim}, 时间步长: {time_dim}天")
    
    # 创建输出文件
    ds_out = nc.Dataset(OUTPUT_FILE, 'w', format='NETCDF4')
    
    # 创建维度
    ds_out.createDimension('time', None)  # 无限维度
    ds_out.createDimension('lat', lat_dim)
    ds_out.createDimension('lon', lon_dim)
    
    # 创建时间变量
    times = ds_out.createVariable('time', 'f8', ('time',))
    times.units = 'days since 1970-01-01 00:00:00'
    times.calendar = 'gregorian'
    times[:] = nc.date2num([pd.Timestamp(d).to_pydatetime() for d in all_dates], 
                          units=times.units, calendar=times.calendar)
    
    # 创建纬度变量 - 使用原始坐标名称但转为小写
    lats = ds_out.createVariable('lat', 'f4', ('lat',))
    lats[:] = sample_ds.Latitude.values
    lats.units = 'degrees_north'
    lats.long_name = 'latitude'
    lats.standard_name = 'latitude'
    
    # 创建经度变量 - 使用原始坐标名称但转为小写
    lons = ds_out.createVariable('lon', 'f4', ('lon',))
    lons[:] = sample_ds.Longitude.values
    lons.units = 'degrees_east'
    lons.long_name = 'longitude'
    lons.standard_name = 'longitude'
    
    # 创建SST变量 - 主要修改点
    sst = ds_out.createVariable('SST', 'f4', ('time', 'lat', 'lon'), 
                               zlib=True, complevel=4, fill_value=-9999.0)
    sst.units = 'deg C'  # 明确设置单位为摄氏度
    sst.long_name = 'Sea Surface Temperature'
    sst.standard_name = 'sea_surface_temperature'
    
    # 添加原始文件中的元数据
    sst.scale_factor = sample_ds.SST.attrs.get('scale_factor', 0.0012)
    sst.add_offset = sample_ds.SST.attrs.get('add_offset', -10.0)
    
    # 设置全局属性
    ds_out.title = f"Combined SST Data for {YEAR}"
    ds_out.history = f"Created on {time.ctime()} by data processing script"
    ds_out.source = "Original data from Japan Aerospace Exploration Agency"
    ds_out.Conventions = "CF-1.8"
    ds_out.spatial_resolution = "250 m"
    ds_out.satellite = sample_ds.attrs.get('Satellite', 'Global Change Observation Mission - Climate (GCOM-C)')
    ds_out.data_provider = sample_ds.attrs.get('Data provider', 'Japan Aerospace Exploration Agency')
    
    # 添加地理范围信息
    ds_out.geospatial_lat_min = sample_ds.attrs.get('Lower_left_latitude', 20.0)
    ds_out.geospatial_lat_max = sample_ds.attrs.get('Upper_left_latitude', 60.0)
    ds_out.geospatial_lon_min = sample_ds.attrs.get('Lower_left_longitude', 115.0)
    ds_out.geospatial_lon_max = sample_ds.attrs.get('Upper_right_longitude', 155.0)
    
    # 关闭文件，稍后以追加模式打开
    ds_out.close()

# 4. 分块处理数据并增量写入
logger.info(f"开始分块处理数据 (每批 {CHUNK_SIZE} 个文件)...")
total_days = len(all_dates)
processed_count = 0

# 打开输出文件进行追加
ds_out = nc.Dataset(OUTPUT_FILE, 'a')

for chunk_start in range(0, total_days, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_days)
    chunk_dates = all_dates[chunk_start:chunk_end]
    
    logger.info(f"处理日期范围: {chunk_dates[0]} 到 {chunk_dates[-1]} ({len(chunk_dates)} 天)")
    
    # 处理当前块的所有日期
    for i, date in enumerate(chunk_dates):
        idx = chunk_start + i
        
        if date in file_info:
            # 直接使用存在的文件
            source_path = file_info[date]
            action = "原始文件"
        else:
            # 查找最近的可用日期（向前查找）
            available_dates = sorted(file_info.keys())
            pos = np.searchsorted(available_dates, date)
            if pos == 0:
                nearest_date = available_dates[0]
            else:
                # 向后查找一天范围内的有效数据
                candidate_dates = [d for d in available_dates if d <= date]
                nearest_date = max(candidate_dates) if candidate_dates else available_dates[0]
            
            source_path = file_info[nearest_date]
            action = f"基于 {str(nearest_date)[:10]} 复制"
        
        logger.info(f"处理 {date} ({idx+1}/{total_days}): {action}")
        
        try:
            # 使用分块读取避免内存问题
            with xr.open_dataset(source_path, chunks={'Latitude': 1000}) as ds:
                # 确保数据维度匹配
                if ds.SST.shape != (lat_dim, lon_dim):
                    logger.warning(f"文件 {source_path.name} 维度不匹配: {ds.SST.shape} vs ({lat_dim}, {lon_dim})")
                    # 尝试调整维度
                    if ds.SST.shape == (lon_dim, lat_dim):
                        sst_data = ds.SST.values.T  # 转置以匹配维度
                    else:
                        raise ValueError(f"无法自动调整维度: {ds.SST.shape}")
                else:
                    sst_data = ds.SST.values
                
                # 应用缩放因子和偏移量
                scaled_data = sst_data * ds.SST.attrs.get('scale_factor', 1) + ds.SST.attrs.get('add_offset', 0)
                
                # 写入输出文件
                ds_out['SST'][idx, :, :] = scaled_data
                
                # 递增处理计数
                processed_count += 1
        except Exception as e:
            logger.error(f"处理文件 {source_path.name} 时出错: {str(e)}")
            # 使用填充值
            ds_out['SST'][idx, :, :] = np.full((lat_dim, lon_dim), -9999.0)
            processed_count += 1

# 关闭输出文件
ds_out.close()
logger.info(f"成功处理 {processed_count}/{total_days} 天的数据")

# 5. 验证结果
logger.info("合并成功! 验证数据集...")
try:
    # 打开合并后的文件
    with xr.open_dataset(OUTPUT_FILE) as final_ds:
        # 输出摘要信息
        logger.info(f"合并后的数据集:\n{final_ds}")
        logger.info(f"时间维度大小: {len(final_ds.time)}")
        
        # 检查数据质量
        valid_count = np.sum(final_ds.SST.values != -9999.0)
        total_cells = len(final_ds.time) * len(final_ds.lat) * len(final_ds.lon)
        coverage = valid_count / total_cells * 100
        logger.info(f"数据覆盖率: {coverage:.2f}%")
        
        # 统计每个时间步的数据来源
        source_types = {}
        for date in all_dates:
            idx = np.where(final_ds.time.values == np.datetime64(date))[0][0]
            sst_data = final_ds.SST.isel(time=idx)
            
            # 计算缺失值比例
            missing_percent = np.mean(sst_data == -9999.0) * 100
            
            if date in file_info:
                source_types.setdefault("原始文件", 0)
                source_types["原始文件"] += 1
                if missing_percent > 5:  # 如果有较多缺失值
                    logger.warning(f"日期 {date} 有 {missing_percent:.1f}% 缺失值")
            else:
                source_types.setdefault("复制文件", 0)
                source_types["复制文件"] += 1
                if missing_percent > 5:  # 如果有较多缺失值
                    logger.warning(f"日期 {date} (复制)有 {missing_percent:.1f}% 缺失值")
        
        logger.info("数据来源统计:")
        for stype, count in source_types.items():
            logger.info(f" - {stype}: {count} 天 ({count/len(all_dates)*100:.1f}%)")
        
        logger.info(f"空间范围: 纬度 {final_ds.lat.min().item():.2f}-{final_ds.lat.max().item():.2f}°, "
              f"经度 {final_ds.lon.min().item():.2f}-{final_ds.lon.max().item():.2f}°")
        
        # 添加日期范围验证
        start_date = pd.Timestamp(final_ds.time[0].values).strftime('%Y-%m-%d')
        end_date = pd.Timestamp(final_ds.time[-1].values).strftime('%Y-%m-%d')
        logger.info(f"日期范围: {start_date} 到 {end_date}")
except Exception as e:
    logger.error(f"验证数据集时出错: {str(e)}")

# 性能统计
elapsed_time = time.time() - start_time
try:
    output_size = os.path.getsize(OUTPUT_FILE) / (1024 ** 3)  # GB
    logger.info(f"输出文件大小: {output_size:.2f} GB")
except:
    output_size = 0

logger.info(f"处理完成! 总耗时: {elapsed_time/60:.2f} 分钟")
logger.info(f"平均处理速度: {len(all_dates)/(elapsed_time/60):.1f} 文件/分钟")