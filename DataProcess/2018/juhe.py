import xarray as xr
import numpy as np
import os
from glob import glob
import dask
from dask.diagnostics import ProgressBar
import time

# 禁用Dask的线程池警告
dask.config.set(**{'array.slicing.split_large_chunks': False})

def aggregate_high_res_data(file_pattern, variable, output_dir, target_resolution=0.25, target_bounds=(115, 155, 20, 60)):
    """
    聚合高分辨率数据到中等分辨率网格
    
    参数:
    file_pattern - 文件路径模式 (e.g., "/path/to/data/*.nc")
    variable - 需要提取的变量名 (e.g., "CHLA")
    output_dir - 输出目录
    target_resolution - 目标分辨率（单位：度）
    target_bounds - 目标区域边界 (lon_min, lon_max, lat_min, lat_max)
    """
    # 记录开始时间
    start_time = time.time()
    
    # 获取文件列表
    files = sorted(glob(file_pattern))
    if not files:
        print(f"未找到匹配文件: {file_pattern}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义目标网格
    lon_min, lon_max, lat_min, lat_max = target_bounds
    target_lon = np.arange(lon_min + target_resolution/2, lon_max, target_resolution)
    target_lat = np.arange(lat_min + target_resolution/2, lat_max, target_resolution)
    
    print(f"开始处理 {variable} 数据...")
    print(f"找到 {len(files)} 个文件, 目标网格尺寸: {len(target_lat)}×{len(target_lon)}")
    
    # 计算聚合因子
    with xr.open_dataset(files[0]) as sample:
        # 自动计算聚合因子（将原始分辨率聚合到目标分辨率）
        lat_diff = np.abs(np.diff(sample.Latitude.values[:2])[0])
        lon_diff = np.abs(np.diff(sample.Longitude.values[:2])[0])
        factor = int(np.round(target_resolution / min(lat_diff, lon_diff)))
        print(f"计算聚合因子: {factor}x ({target_resolution}° / {min(lat_diff, lon_diff):.5f}°)")
    
    # 初始化结果数据集列表
    datasets = []
    
    # 使用Dask并行处理每个文件
    for file in files:
        # 使用内存映射方式打开文件
        with xr.open_dataset(file, chunks={'Latitude': 1000, 'Longitude': 1000}) as ds:
            # 创建数据选择器（仅选择目标变量）
            da = ds[variable]
            
            # 应用聚合（均值）
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                da_agg = (
                    da
                    .coarsen(Latitude=factor, Longitude=factor, boundary="pad")
                    .mean()
                    .rename({'Latitude': 'lat', 'Longitude': 'lon'})
                )
            
            # 保留坐标信息
            da_agg = da_agg.assign_coords({
                'lat': da_agg.lat.compute(),
                'lon': da_agg.lon.compute()
            })
            
            # 添加文件时间信息
            if 'time' not in da.dims:
                try:
                    # 尝试从文件名提取时间
                    date_str = os.path.basename(file).split('_')[-1].split('.')[0]
                    date = np.datetime64(date_str)
                    da_agg = da_agg.expand_dims(time=[date])
                except:
                    pass
            
            datasets.append(da_agg)
    
    # 合并所有时间片
    with ProgressBar():
        print(f"合并 {len(datasets)} 个时间片...")
        combined = xr.concat(datasets, dim='time', coords='minimal', compat='override')
        
        # 插值到统一的目标网格
        print("插值到统一网格...")
        combined = combined.interp(
            lat=target_lat,
            lon=target_lon,
            method='nearest',
            kwargs={"fill_value": np.nan}
        )
        
        # 转换为Dataset并添加元数据
        ds_out = xr.Dataset({variable: combined})
        ds_out.attrs = {
            'aggregation_method': 'coarsen mean',
            'target_resolution': f'{target_resolution} degree',
            'source_files': f"{len(files)} files",
            'original_resolution': f"{lat_diff:.5f} × {lon_diff:.5f} degree",
            'processing_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{variable}_aggregated_{int(target_resolution*100)}deg.nc")
        ds_out.to_netcdf(output_path)
        print(f"处理完成! 结果保存至: {output_path}")
        print(f"用时: {time.time() - start_time:.2f} 秒")
        print(f"最终数据集大小: {ds_out.nbytes / 1024**2:.2f} MB")

# ===================== 参数配置 =====================
# 注意：以下路径需要根据实际数据位置修改

# 配置数据处理参数
PARAMS = {
    'target_resolution': 0.25,  # 目标分辨率 (单位: 度)
    'target_bounds': (115.0, 155.0, 20.0, 60.0),  # 区域边界 (lon_min, lon_max, lat_min, lat_max)
    'output_dir': "aggregated_data"
}

# ===================== 执行处理 =====================
if __name__ == "__main__":
    # 1. 处理叶绿素数据 (CHLA)
    aggregate_high_res_data(
        file_pattern=r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\data_m\chla_m\*.nc",
        variable="CHLA",
        output_dir=PARAMS['output_dir'],
        target_resolution=PARAMS['target_resolution'],
        target_bounds=PARAMS['target_bounds']
    )
    print("所有高分辨率数据聚合完成!")