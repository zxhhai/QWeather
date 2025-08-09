import xarray as xr
import numpy as np
import os
from dask.diagnostics import ProgressBar

# 1. 定义目标网格参数
TARGET_RES = 0.1  # 目标分辨率(度)
LAT_MIN, LAT_MAX = 20.05, 59.95  # 目标纬度范围
LON_MIN, LON_MAX = 115.05, 154.95  # 目标经度范围

# 创建目标网格坐标
target_lat = np.round(np.arange(LAT_MIN, LAT_MAX + TARGET_RES/2, TARGET_RES), 2)
target_lon = np.round(np.arange(LON_MIN, LON_MAX + TARGET_RES/2, TARGET_RES), 2)

# 2. 通用处理函数
def process_dataset(ds, dataset_type):
    """统一处理不同数据集到目标网格"""
    # 重命名时间维度（如有需要）
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
    # 统一坐标名称
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    # 区域裁剪（针对全球数据）
    if dataset_type in ['SLA', 'O3', 'HCHO', 'WINDS']:
        # 风速数据纬度方向特殊处理
        if dataset_type == 'WINDS':
            ds = ds.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
            ds = ds.sortby('lat')  # 确保纬度递增
        else:
            ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    
    # 分辨率适配处理
    if dataset_type in ['CHLA', 'PAR', 'SST']:
        # 高分辨率数据聚合
        ds = ds.coarsen(lat=40, lon=40, boundary='trim').mean()
        ds = ds.interp(lat=target_lat, lon=target_lon)
    elif dataset_type == 'SLA':
        # 中等分辨率线性插值
        ds = ds.interp(lat=target_lat, lon=target_lon, method='linear')
    elif dataset_type == 'O3':
        # 低分辨率最近邻插值
        ds = ds.interp(lat=target_lat, lon=target_lon, method='nearest')
    elif dataset_type == 'WINDS':
        # 风速数据经度转换和插值
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360 - 180))
        ds = ds.sortby('lon')
        ds = ds.interp(lat=target_lat, lon=target_lon, method='linear')
    else:  # HCHO和其他
        ds = ds.interp(lat=target_lat, lon=target_lon)
    
    # 变量重命名（风速特殊处理）
    if dataset_type == 'WINDS':
        ds = ds.rename({'si10': 'windspeed'})
    
    return ds

# 3. 数据集处理配置
DATASET_CONFIG = {
    'CHLA': {
        'path': r'combined_chla_2018_full_year.nc',
        'variables': ['CHLA'],
        'chunks': {'time': 10}  # 分块大小
    },
    'PAR': {
        'path': r'combined_par_2018_full_year.nc',
        'variables': ['PAR'],
        'chunks': {'time': 10}
    },
    'SST': {
        'path': r'combined_sst_2018_full_year.nc',
        'variables': ['SST'],
        'chunks': {'time': 5}  # 更大数据量，更小的分块
    },
    'SLA': {
        'path': r'combined_ssh_2018_full_year.nc',
        'variables': ['sla'],
        'chunks': {'time': 20}
    },
    'O3': {
        'path': r'combined_o3_2018_full_year.nc',
        'variables': ['tco'],
        'chunks': {'time': 50}  # 小数据集，大分块
    },
    'HCHO': {
        'path': r'combined_hcho_2018_full_year.nc',
        'variables': ['HCHO'],
        'chunks': {'time': 15}
    },
    'WINDS': {
        'path': r'combined_ws_2018_full_year.nc',
        'variables': ['si10'],
        'chunks': {'time': 30}
    }
}

# 4. 主处理函数
def process_all_datasets(output_dir='processed_data'):
    """处理所有数据集并保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    for ds_name, config in DATASET_CONFIG.items():
        print(f"Processing {ds_name} dataset...")
        
        # 使用Dask分块加载数据
        with xr.open_dataset(config['path'], chunks=config['chunks']) as ds:
            # 选择需要的变量
            ds_subset = ds[config['variables']]
            
            # 处理数据集
            processed_ds = process_dataset(ds_subset, ds_name)
            
            # 设置输出路径
            output_path = os.path.join(output_dir, f"{ds_name}_regridded.nc")
            
            # 编码设置（压缩和类型转换）
            encoding = {}
            for var in processed_ds.data_vars:
                encoding[var] = {
                    'dtype': 'float32',
                    'zlib': True,
                    'complevel': 1
                }
            
            # 使用进度条保存
            with ProgressBar():
                processed_ds.to_netcdf(
                    output_path,
                    encoding=encoding,
                    compute=True
                )
        
        print(f"✅ {ds_name} processed and saved to {output_path}")
    
    print("All datasets processed successfully!")

# 5. 验证函数
def validate_results(output_dir):
    """验证处理后的数据集一致性"""
    datasets = []
    
    # 加载所有处理后的数据集
    for ds_name in DATASET_CONFIG.keys():
        path = os.path.join(output_dir, f"{ds_name}_regridded.nc")
        ds = xr.open_dataset(path)
        datasets.append(ds)
    
    # 检查网格对齐
    for i in range(1, len(datasets)):
        assert np.allclose(datasets[0].lat, datasets[i].lat)
        assert np.allclose(datasets[0].lon, datasets[i].lon)
    
    # 检查时间对齐
    for i in range(1, len(datasets)):
        assert np.array_equal(datasets[0].time, datasets[i].time)
    
    print("✅ All datasets aligned correctly!")
    
    # 数据范围检查
    for ds in datasets:
        if 'SST' in ds:
            assert -5 < ds.SST.min() < 40
        if 'CHLA' in ds:
            assert 0 <= ds.CHLA.min() < 50
        if 'windspeed' in ds:
            assert 0 <= ds.windspeed.min() < 50
    
    print("✅ All data within expected ranges")

# 6. 执行处理
if __name__ == "__main__":
    # 处理所有数据集
    #process_all_datasets()
    
    # 验证结果
    # validate_results('processed_data')
    
    #可选：创建合并数据集
    combined_ds = xr.merge([xr.open_dataset(f"processed_data/{name}_regridded.nc") 
                            for name in DATASET_CONFIG.keys()])
    combined_ds.to_netcdf("combined_data.nc")