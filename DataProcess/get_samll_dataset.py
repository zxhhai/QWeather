import xarray as xr

def get_small_dataset(ds):
    """
    """
    ds_small = ds.isel(
        lat=slice(0, 40),
        lon=slice(0, 40)
    )
    
    print(f"Original shape: {ds.sizes}")
    print(f"Downsampled shape: {ds_small.sizes}")
    print(f"Size reduction: {ds.nbytes / ds_small.nbytes:.1f}x")
    
    return ds_small


ds = xr.open_dataset('/home/zxh/CQ/dataset/isoprene_results.nc')

ds_small = get_small_dataset(ds)

# 保存小数据集
ds_small.to_netcdf('/home/zxh/CQ/dataset/data_small.nc')