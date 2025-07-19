import xarray as xr

# 指定需要合并的文件列表（支持通配符）
file_paths = r"C:/RIP_D/Codes/Python/QWeather/DataProcess/datas/data_m/chla_m/*.nc"  # 或用列表明确指定文件路径

# 合并文件（按时间维度自动对齐）
ds = xr.open_mfdataset(
    file_paths,
    engine='netcdf4',  # 明确指定引擎（使用您单文件成功的引擎）
    combine="by_coords",
    parallel=True,
    chunks={"time": 20}
)

# 将合并后的数据集保存为新文件
ds.to_netcdf("2018_chala_data_365.nc")