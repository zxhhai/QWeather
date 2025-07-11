import xarray as xr

# 读取.nc文件（替换为你的文件路径）
ds = xr.open_dataset(r'C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\chla\GC1SG1_201801010021003300_L2MG_NWLRQ_2007_PAR.nc')

# 打印数据集摘要（类似NCO的ncdump）
print(ds)

# 打印详细数据变量（可选）
# print(ds['variable_name'][:])  # 替换为实际变量名

# 关闭文件连接（在with语句中可自动关闭）
ds.close()