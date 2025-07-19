import xarray as xr
import glob
import os

# 1. 指定目录路径
directory = r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\ssh"  # 替换为你的目录路径

# 2. 获取所有.nc文件路径并排序
file_paths = sorted(glob.glob(os.path.join(directory, "*.nc")))
# 如果文件名包含时间信息，需确保排序正确（如按文件名排序r

# 3. 合并文件
ds = xr.open_mfdataset(
    file_paths,       # 文件路径列表
    combine="by_coords",  # 按坐标自动合并（默认）
    parallel=True,    # 并行读取加速（可选）
    chunks={"time": 10}  # 使用Dask分块处理大文件（可选）
)

# 4. 使用合并后的数据集
print(ds)  # 查看数据
ds.to_netcdf("combined.nc")  # 保存为单个文件（可选）