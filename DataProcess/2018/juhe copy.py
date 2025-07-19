import xarray as xr
import numpy as np
import os
import re
from glob import glob
import pandas as pd
from pathlib import Path
import dask
import time
import logging
from datetime import datetime
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("aggregation.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_date_from_filename(filename):
    """
    健壮的文件名时间提取函数
    """
    filename = Path(filename).stem
    
    # 尝试匹配多种日期格式
    patterns = [
        r'(\d{8})',  # YYYYMMDD
        r'(\d{4})m(\d{2})(\d{2})',  # YYYYmMMDD
        r'(\d{4})_(\d{2})_(\d{2})',  # YYYY_MM_DD
        r'(\d{4})(\d{2})(\d{2})\d{6}',  # YYYYMMDDHHMMSS
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if pattern == r'(\d{8})':
                date_str = match.group(1)
                return pd.to_datetime(date_str, format='%Y%m%d').to_numpy().astype('datetime64[s]')
            else:
                year, month, day = match.groups()[:3]
                return np.datetime64(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
    
    # 如果文件名中找不到日期，使用文件修改时间
    file_time = os.path.getmtime(filename)
    return np.datetime64(datetime.utcfromtimestamp(file_time), 'D')

def create_clean_filepath(original_path):
    """
    创建无特殊字符的临时文件路径
    """
    clean_path = str(original_path).replace('[ND]', '_ND_').replace('[', '_').replace(']', '_')
    return Path(clean_path)

def robust_aggregation(file_pattern, variable, output_dir, 
                      target_resolution=0.25, target_bounds=(115.0, 155.0, 20.0, 60.0),
                      chunk_size=10):
    """
    优化版高分辨率数据聚合 - 直接处理特殊字符
    """
    start_time = time.time()
    logger.info(f"开始处理 {variable} 数据聚合")
    
    # 获取文件列表
    files = sorted(glob(file_pattern))
    if not files:
        logger.error(f"未找到匹配文件: {file_pattern}")
        return
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    temp_dir = output_dir / f"temp_{variable}_{int(time.time())}"
    temp_dir.mkdir(exist_ok=True)
    logger.info(f"找到 {len(files)} 个文件需要处理")
    
    # 目标网格
    lon_min, lon_max, lat_min, lat_max = target_bounds
    target_lon = np.arange(lon_min + target_resolution/2, lon_max, target_resolution)
    target_lat = np.arange(lat_min + target_resolution/2, lat_max, target_resolution)
    logger.info(f"目标网格尺寸: {len(target_lat)}×{len(target_lon)}")
    
    # 确定聚合因子（使用第一个有效文件）
    for file in files:
        try:
            with xr.open_dataset(file) as sample:
                lat_diff = np.abs(np.diff(sample.Latitude.values[:2])[0])
                lon_diff = np.abs(np.diff(sample.Longitude.values[:2])[0])
                factor = int(np.round(target_resolution / min(lat_diff, lon_diff)))
                logger.info(f"原始分辨率: {lat_diff:.6f}×{lon_diff:.6f}°, 聚合因子: {factor}")
                break
        except Exception as e:
            logger.warning(f"无法打开 {Path(file).name} 确定分辨率: {str(e)}")
    
    # 分块处理函数
    def process_batch(file_batch, batch_id):
        batch_results = []
        for file_path in file_batch:
            file_name = Path(file_path).name
            clean_file_path = create_clean_filepath(file_path)
            
            try:
                # 1. 提取时间
                date_val = extract_date_from_filename(file_path)
                
                # 2. 直接处理原始文件（不使用符号链接）
                with xr.open_dataset(file_path, chunks={'Latitude': 1000, 'Longitude': 1000}) as ds:
                    if variable not in ds:
                        logger.warning(f"{file_name} 中未找到变量 {variable}")
                        continue
                        
                    da = ds[variable]
                    
                    # 均值聚合
                    da_agg = da.coarsen(
                        Latitude=factor, 
                        Longitude=factor, 
                        boundary="pad"
                    ).mean()
                    
                    # 重命名坐标
                    da_agg = da_agg.rename({'Latitude': 'lat', 'Longitude': 'lon'})
                    
                    # 插值到目标网格
                    da_agg = da_agg.interp(
                        lat=target_lat,
                        lon=target_lon,
                        method="nearest"
                    )
                    
                    # 添加时间维度
                    da_agg = da_agg.expand_dims(time=[date_val])
                    
                    # 转换为数据集
                    ds_agg = da_agg.to_dataset(name=variable)
                    ds_agg.attrs['source_file'] = file_name
                    
                    batch_results.append(ds_agg)
                    
            except Exception as e:
                logger.error(f"处理 {file_name} 失败: {str(e)}", exc_info=True)
        
        if batch_results:
            # 合并当前批次并保存
            batch_ds = xr.concat(batch_results, dim='time', coords='minimal')
            batch_path = temp_dir / f"batch_{batch_id:03d}.nc"
            
            # 优化保存设置
            encoding = {variable: {'zlib': True, 'complevel': 1}}
            batch_ds.to_netcdf(batch_path, encoding=encoding)
            
            logger.info(f"保存批次 {batch_id}: {len(batch_results)}个文件 -> {batch_path}")
        
        return len(batch_results)
    
    # 分块处理主循环
    logger.info(f"开始分块处理 ({len(files)} 文件, 批次大小: {chunk_size})")
    processed_count = 0
    for i in range(0, len(files), chunk_size):
        batch_files = files[i:i+chunk_size]
        batch_id = i // chunk_size
        processed_in_batch = process_batch(batch_files, batch_id)
        processed_count += processed_in_batch
        
        # 进度报告
        progress = processed_count / len(files) * 100
        time_per_file = (time.time() - start_time) / processed_count if processed_count else 0
        remaining_time = (len(files) - processed_count) * time_per_file
        
        logger.info(
            f"进度: {processed_count}/{len(files)} ({progress:.1f}%) | "
            f"剩余时间: {remaining_time/60:.1f}分钟"
        )
    
    # 检查是否生成批次文件
    batch_files = sorted(temp_dir.glob("batch_*.nc"))
    if not batch_files:
        logger.error("未生成任何批次文件，处理失败")
        return
    
    logger.info(f"开始合并 {len(batch_files)} 个批次文件")
    
    # 并行合并所有批次
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        final_ds = xr.open_mfdataset(
            batch_files,
            combine='nested',
            concat_dim='time',
            parallel=True,
            chunks={'time': 100}
        )
    
    # 添加全局属性
    final_ds.attrs = {
        'variable': variable,
        'resolution': f'{target_resolution} degree',
        'aggregation_factor': factor,
        'n_files_processed': processed_count,
        'n_files_total': len(files),
        'processing_date': datetime.now().isoformat(),
        'region': f"Lon: {lon_min}-{lon_max}°E, Lat: {lat_min}-{lat_max}°N"
    }
    
    # 保存最终结果
    output_path = output_dir / f"{variable}_agg_{int(target_resolution*100)}deg_{len(target_lat)}x{len(target_lon)}.nc"
    
    # 使用高效压缩
    encoding = {}
    for var in final_ds.data_vars:
        encoding[var] = {
            'zlib': True, 
            'complevel': 3, 
            'chunksizes': (100, 100, 100)
        }
    
    final_ds.to_netcdf(output_path, encoding=encoding)
    logger.info(f"最终结果保存至: {output_path}")
    
    # 清理临时文件
    for batch_file in batch_files:
        batch_file.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        logger.warning(f"无法删除临时目录 {temp_dir}，可能非空")
    
    # 性能报告
    total_time = time.time() - start_time
    logger.info(f"处理完成! 总用时: {total_time/60:.1f}分钟, 平均每文件: {total_time/len(files):.1f}秒")
    
    return output_path

# ===================== 主执行模块 =====================
if __name__ == "__main__":
    # 配置参数 - 调整为您的实际设置
    CONFIG = {
        'target_resolution': 0.25,            # 目标分辨率(度)
        'target_bounds': (115.0, 155.0, 20.0, 60.0),  # 处理区域边界
        'output_dir': "aggregated_results",   # 输出目录
        'chunk_size': 5,                     # 批次大小（减小以降低内存需求）
        'variables': {
            # 'SST': {
            #     'pattern': r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\data_m\sst_m\*.nc",
            # },
            # 'PAR': {
            #     'pattern': r"C:\RIP_D\Codes\Python\QWeather\DataProcess\2018\*.nc"
            # },
            'CHLA': {
                'pattern': r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\data_m\chla_m\*.nc"
            }
        }
    }
    
    # 确保输出目录存在
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # 处理每个变量
    results = {}
    for var_name, var_config in CONFIG['variables'].items():
        logger.info(f"开始处理 {var_name} 数据")
        
        # 直接使用原始文件路径
        result_path = robust_aggregation(
            file_pattern=var_config['pattern'],
            variable=var_name,
            output_dir=output_dir,
            target_resolution=CONFIG['target_resolution'],
            target_bounds=CONFIG['target_bounds'],
            chunk_size=CONFIG['chunk_size']
        )
        
        if result_path:
            results[var_name] = str(result_path)
    
    # 生成最终报告
    logger.info("======= 聚合处理完成 =======")
    logger.info(f"输出目录: {output_dir}")
    for var, path in results.items():
        logger.info(f"{var}: {path}")
    
    print(f"\n所有处理完成! 结果保存在 {output_dir} 目录")