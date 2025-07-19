import xarray as xr

# 读取.nc文件（替换为你的文件路径）
ds = xr.open_dataset(
    r'DataProcess\datas\combined_data.nc')
# 打印数据集摘要（类似NCO的ncdump）
print(ds)

# 打印详细数据变量（可选）
# print(ds['variable_name'][:])  # 替换为实际变量名

# 关闭文件连接（在with语句中可自动关闭）
ds.close()