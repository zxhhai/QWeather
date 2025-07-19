import pandas as pd
import xarray as xr

# 1. 加载已有月度数据
ds = xr.open_dataset(r'DataProcess\datas\wind\monthly_windspeed.nc')  # 替换为你的文件路径

# 2. 创建2018年全年的时间轴（每日）
full_time = pd.date_range('2018-01-01', '2018-12-31', freq='D')

# 3. 将每月第一天数据扩展到全月每日
monthly_da = ds.sel(valid_time=full_time, method='ffill')

# 4. 修正时间坐标确保对齐
monthly_da = monthly_da.assign_coords(valid_time=full_time)

# 5. 保存扩展后的数据
monthly_da.to_netcdf('full_year_2018.nc')

# 验证结果
print("新数据集时间范围:", monthly_da.valid_time.min().values, "至", monthly_da.valid_time.max().values)
print("时间点数量:", len(monthly_da.valid_time))