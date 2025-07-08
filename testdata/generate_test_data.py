import numpy as np
import xarray as xr

# 设置维度大小
days = np.arange(365)                     # time
lat = np.linspace(-90, 90, 180)           # latitude
lon = np.linspace(-180, 180, 360)         # longitude
variables = ['a', 'b', 'c', 'x']          # 4 variables (a,b,c are input, x is target)

# 生成模拟数据
np.random.seed(42)  # 固定种子，确保可复现
data = np.random.rand(len(days), len(variables), len(lat), len(lon)).astype(np.float32)

# 创建 xarray Dataset
ds = xr.Dataset(
    {
        "data": (("day", "variable", "lat", "lon"), data)
    },
    coords={
        "day": days,
        "variable": variables,
        "lat": lat,
        "lon": lon
    }
)

# 可选元信息
ds["data"].attrs["description"] = "Synthetic 4D tensor"
ds.attrs["title"] = "Generated test dataset for ML model: a,b,c → x"
ds.attrs["creator"] = "ChatGPT"

# 保存为 NetCDF 文件
ds.to_netcdf("test_data.nc")

print("✅ 文件已保存为 test_data.nc，shape:", data.shape)

