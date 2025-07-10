import xarray as xr

# 指定需要合并的文件列表（支持通配符）
file_paths = r"C:/RIP_D/Codes/Python/QWeather/DataProcess/datas/ssh/*.nc"  # 或用列表明确指定文件路径

# 合并文件（按时间维度自动对齐）
ds = xr.open_mfdataset(
    file_paths,
    combine="by_coords",    # 根据坐标自动对齐
    parallel=True,          # 并行读取加速
    chunks={"time": 20}     # 分块处理节省内存（可选）
)

# 将合并后的数据集保存为新文件
ds.to_netcdf("combined_data.nc")