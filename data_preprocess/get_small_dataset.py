import xarray as xr

def get_small_dataset(ds):
    """
    """
    ds_small = ds.isel(
        latitude=slice(120, 240),
        longitude=slice(240, 480),
    )
    
    print(f"Original shape: {ds.sizes}")
    print(f"Downsampled shape: {ds_small.sizes}")
    print(f"Size reduction: {ds.nbytes / ds_small.nbytes:.1f}x")
    
    return ds_small


ds = xr.open_dataset('./datasets/data/data_large.nc')

ds_small = get_small_dataset(ds)

# 保存小数据集
ds_small.to_netcdf('./datasets/data/data_small.nc')