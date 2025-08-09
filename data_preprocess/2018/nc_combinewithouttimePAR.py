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

# 配置设置 - 修改为你的 HCHO 数据目录
SOURCE_DIR = Path(r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\hcho")
YEAR = 2018
OUTPUT_FILE = f"combined_hcho_{YEAR}_full_year.nc"
CHUNK_SIZE = 50  # 每批处理的文件数
PRIMARY_VAR = "key_science_data_column_amount"  # 主要科学数据变量

logger.info(f"开始合并 HCHO 数据集...")
start_time = time.time()

# 1. 获取文件列表并识别时间信息
logger.info("分析文件名并提取时间信息...")
files = list(SOURCE_DIR.glob("*.nc4"))
if not files:
    raise FileNotFoundError(f"在目录 {SOURCE_DIR} 中没有找到任何.nc文件")

# 针对OMI HCHO文件名的专用日期解析方法
def extract_date_from_filename_hcho(filename):
    """从OMI HCHO文件名中提取日期信息"""
    # 尝试匹配OMI HCHO文件名的日期格式: YYYYmMMDD
    match = re.search(r"(\d{4})m(\d{4})", filename)
    if match:
        year = match.group(1)
        month_day = match.group(2)
        try:
            # 提取月份和日期
            month = month_day[:2]
            day = month_day[2:]
            # 将日期字符串转换为datetime64
            return np.datetime64(f"{year}-{month}-{day}")
        except:
            return None
    return None

# 创建时间索引字典
file_info = {}
for file in files:
    date_obj = extract_date_from_filename_hcho(file.name)
    if date_obj is not None:
        file_info[date_obj] = file

if not file_info:
    logger.error("没有找到任何包含有效时间信息的文件")
    # 打印所有文件名以便调试
    logger.info("所有文件名:")
    for file in files[:10]:  # 只打印前10个文件避免太多输出
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
    # 获取维度信息 - 注意HCHO数据中的维度名称是小写
    lat_dim = len(sample_ds.latitude)
    lon_dim = len(sample_ds.longitude)
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
    times.standard_name = 'time'
    times[:] = nc.date2num([pd.Timestamp(d).to_pydatetime() for d in all_dates], 
                          units=times.units, calendar=times.calendar)
    
    # 创建纬度变量
    lats = ds_out.createVariable('lat', 'f4', ('lat',))
    lats[:] = sample_ds.latitude.values
    lats.units = 'degrees_north'
    lats.long_name = 'latitude'
    lats.standard_name = 'latitude'
    
    # 创建经度变量
    lons = ds_out.createVariable('lon', 'f4', ('lon',))
    lons[:] = sample_ds.longitude.values
    lons.units = 'degrees_east'
    lons.long_name = 'longitude'
    lons.standard_name = 'longitude'
    
    # 创建HCHO主变量 - 使用标准命名
    hcho = ds_out.createVariable('HCHO', 'f4', ('time', 'lat', 'lon'), 
                               zlib=True, complevel=4, fill_value=-9999.0)
    hcho.units = 'mol m-2'  # 甲醛总量的标准单位
    hcho.long_name = 'Formaldehyde total column'
    hcho.standard_name = 'atmosphere_moles_per_square_meter_of_formaldehyde'
    
    # 添加重要辅助变量（根据需要选择）
    selected_aux_vars = [
        'key_science_data_column_uncertainty',
        'support_data_cloud_fraction',
        'support_data_amf'
    ]
    
    # 为每个选定的辅助变量创建存储
    aux_vars = {}
    for var_name in selected_aux_vars:
        if var_name in sample_ds.data_vars:
            var = ds_out.createVariable(var_name, 'f4', ('time', 'lat', 'lon'), 
                                     zlib=True, complevel=4, fill_value=-9999.0)
            # 从原始文件中复制属性和单位
            var.units = sample_ds[var_name].attrs.get('units', 'unknown')
            var.long_name = sample_ds[var_name].attrs.get('long_name', var_name)
            
            aux_vars[var_name] = var
    
    # 设置全局属性 - 保留重要元数据
    ds_out.title = f"Combined HCHO Data for {YEAR}"
    ds_out.history = f"Created on {time.ctime()} by data processing script"
    ds_out.Conventions = "CF-1.8"
    ds_out.source = "Original data from OMI/Aura"
    
    # 复制重要属性
    key_attrs = [
        'IdentifierProductDOI',
        'ProductGenerationAlgorithm',
        'LongName',
        'ShortName',
        'InstrumentName',
        'Platform',
        'ProcessingLevel',
        'spatial_resolution'
    ]
    
    for attr in key_attrs:
        if attr in sample_ds.attrs:
            ds_out.setncattr(attr, sample_ds.attrs[attr])
    
    # 添加地理范围信息
    ds_out.geospatial_lat_min = float(sample_ds.latitude.min())
    ds_out.geospatial_lat_max = float(sample_ds.latitude.max())
    ds_out.geospatial_lon_min = float(sample_ds.longitude.min())
    ds_out.geospatial_lon_max = float(sample_ds.longitude.max())
    
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
                # 向后查找
                candidate_dates = [d for d in available_dates if d <= date]
                nearest_date = max(candidate_dates) if candidate_dates else available_dates[0]
            
            source_path = file_info[nearest_date]
            action = f"基于 {str(nearest_date)[:10]} 复制"
            logger.info(f"日期 {date} 无数据，使用最近的 {nearest_date} 代替")
        
        logger.info(f"处理 {date} ({idx+1}/{total_days}): {action}")
        
        try:
            # 使用分块读取避免内存问题
            with xr.open_dataset(source_path, chunks={'latitude': 500, 'longitude': 500}) as ds:
                # 处理主变量 (HCHO)
                if PRIMARY_VAR in ds.data_vars:
                    var_data = ds[PRIMARY_VAR].values
                    
                    # 确保数据维度匹配
                    if var_data.shape != (lat_dim, lon_dim):
                        logger.warning(f"文件 {source_path.name} 主变量维度不匹配: {var_data.shape} vs ({lat_dim}, {lon_dim})")
                        # 尝试调整维度
                        if var_data.shape == (lon_dim, lat_dim):
                            var_data = var_data.T  # 转置以匹配维度
                        elif var_data.shape == (1, lat_dim, lon_dim):
                            var_data = var_data[0, :, :]
                        elif var_data.shape == (lat_dim, lon_dim, 1):
                            var_data = var_data[:, :, 0]
                        else:
                            # 调整大小以匹配目标网格 (最后手段)
                            logger.warning(f"使用插值调整主变量尺寸")
                            var_data = xr.DataArray(var_data, dims=('latitude', 'longitude')).interp(
                                latitude=ds_out['lat'][:],
                                longitude=ds_out['lon'][:]
                            ).values
                    
                    # 应用质量标志筛选
                    if 'qa_statistics_data_quality_flag' in ds.data_vars:
                        qa_flag = ds['qa_statistics_data_quality_flag'].values
                        # 仅保留高质量数据 (假设标志为0代表高质量)
                        var_data[qa_flag != 0] = -9999.0
                    
                    # 写入输出文件
                    ds_out['HCHO'][idx, :, :] = var_data
                else:
                    logger.error(f"文件 {source_path.name} 缺少主变量 {PRIMARY_VAR}")
                    raise KeyError(f"主变量 {PRIMARY_VAR} 不存在")
                
                # 处理选定的辅助变量
                for var_name in aux_vars.keys():
                    if var_name in ds.data_vars:
                        aux_data = ds[var_name].values
                        
                        # 确保辅助变量维度匹配
                        if aux_data.shape != (lat_dim, lon_dim):
                            # 尝试调整维度
                            if aux_data.shape == (lon_dim, lat_dim):
                                aux_data = aux_data.T  # 转置
                            elif aux_data.shape == (1, lat_dim, lon_dim):
                                aux_data = aux_data[0, :, :]
                            elif aux_data.shape == (lat_dim, lon_dim, 1):
                                aux_data = aux_data[:, :, 0]
                            elif aux_data.shape == var_data.shape:
                                # 使用与主变量相同的调整
                                pass
                            else:
                                logger.warning(f"无法自动调整辅助变量 {var_name} 的维度，跳过")
                                aux_data = np.full((lat_dim, lon_dim), -9999.0)
                        
                        # 写入辅助变量
                        aux_vars[var_name][idx, :, :] = aux_data
                    else:
                        logger.warning(f"文件 {source_path.name} 缺少辅助变量 {var_name}")
                
                # 递增处理计数
                processed_count += 1
        except Exception as e:
            logger.error(f"处理文件 {source_path.name} 时出错: {str(e)}")
            # 使用填充值
            ds_out['HCHO'][idx, :, :] = np.full((lat_dim, lon_dim), -9999.0)
            # 填充辅助变量
            for var_name in aux_vars.keys():
                aux_vars[var_name][idx, :, :] = np.full((lat_dim, lon_dim), -9999.0)
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
        valid_count = np.sum(final_ds.HCHO.values != -9999.0)
        total_cells = len(final_ds.time) * len(final_ds.lat) * len(final_ds.lon)
        coverage = valid_count / total_cells * 100
        logger.info(f"主变量数据覆盖率: {coverage:.2f}%")
        
        # 统计每个时间步的数据来源
        source_types = {}
        for date in all_dates:
            idx = np.where(final_ds.time.values == np.datetime64(date))[0][0]
            hcho_data = final_ds.HCHO.isel(time=idx)
            
            # 计算缺失值比例
            missing_percent = np.mean(hcho_data == -9999.0) * 100
            
            if date in file_info:
                source_types.setdefault("原始文件", 0)
                source_types["原始文件"] += 1
                if missing_percent > 50:  # 如果有较多缺失值
                    logger.warning(f"日期 {date} 有 {missing_percent:.1f}% 缺失值")
            else:
                source_types.setdefault("复制文件", 0)
                source_types["复制文件"] += 1
                if missing_percent > 50:  # 如果有较多缺失值
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
        
        # 统计HCHO值的统计信息
        hcho_values = final_ds.HCHO.values
        valid_values = hcho_values[hcho_values != -9999.0]
        if len(valid_values) > 0:
            logger.info(f"HCHO值统计:")
            logger.info(f" - 最小值: {np.nanmin(valid_values):.3e} mol m-2")
            logger.info(f" - 最大值: {np.nanmax(valid_values):.3e} mol m-2")
            logger.info(f" - 平均值: {np.nanmean(valid_values):.3e} mol m-2")
            logger.info(f" - 标准差: {np.nanstd(valid_values):.3e} mol m-2")
            logger.info(f" - 有效数据点数量: {len(valid_values)}/{len(hcho_values.flatten())} ({len(valid_values)/len(hcho_values.flatten())*100:.1f}%)")
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