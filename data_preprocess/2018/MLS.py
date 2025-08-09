import numpy as np
import xarray as xr
import os
import re
from datetime import datetime
import logging
import glob

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('omi_mls_conversion.log')
    ]
)
logger = logging.getLogger('omi_mls_converter')

# ===================== 配置参数 =====================
# 输入目录路径
INPUT_DIR = r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\o3\o3_ori"

# 输出目录
OUTPUT_DIR = r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\o3"

# 数据年份 (根据您的上下文)
DATA_YEAR = 2018

# 是否详细输出
VERBOSE = True

# ===================== 基于 IDL 读取器的解析函数 =====================
def read_lvl3(fname):
    """
    完全复制 IDL 读取器逻辑的 Python 实现
    
    参数:
        fname (str): 文件路径
        
    返回:
        tuple: (latitudes, longitudes, ozone_data, date)
    """
    try:
        # 从文件名提取日期
        date = extract_date_from_filename(fname)
        
        # 固定网格参数
        nlon = 288  # 经度点数
        nlat = 120  # 纬度点数
        
        # 创建经度数组 (固定值)
        longitudes = np.around(np.arange(-179.375, 180.0, 1.25), decimals=3)
        
        # 创建纬度数组 (固定值)
        latitudes = np.arange(-59.5, 60.0, 1.0)
        
        # 初始化臭氧数据数组
        ozone_data = np.full((nlon, nlat), np.nan)  # 注意: IDL 使用 (lon, lat) 顺序
        
        # 读取文件内容
        with open(fname, 'r') as f:
            # 跳过前3行标题
            for _ in range(3):
                f.readline()
            
            # 处理每个纬度带
            for lat_idx in range(nlat):
                # 处理前11行 (每行25个值)
                for irow in range(11):
                    line = f.readline().strip()
                    
                    # 解析25个整数值
                    values = []
                    # 跳过行首空格 (IDL格式中的'1X')
                    clean_line = line.lstrip()
                    
                    # 解析25个3位整数
                    for i in range(25):
                        if len(clean_line) < 3:
                            values.append(999)
                        else:
                            str_val = clean_line[:3]
                            clean_line = clean_line[3:]
                            
                            try:
                                values.append(int(str_val))
                            except ValueError:
                                values.append(999)
                    
                    # 添加到数据数组
                    start_lon = irow * 25
                    for lon_idx in range(25):
                        ozone_data[start_lon + lon_idx, lat_idx] = values[lon_idx]
                
                # 处理最后一行 (13个值 + 纬度标签)
                line = f.readline().strip()
                
                # 解析13个整数值
                values = []
                # 跳过行首空格 (IDL格式中的'1X')
                clean_line = line.lstrip()
                
                # 解析13个3位整数
                for i in range(13):
                    if len(clean_line) < 3:
                        values.append(999)
                    else:
                        str_val = clean_line[:3]
                        clean_line = clean_line[3:]
                        
                        try:
                            values.append(int(str_val))
                        except ValueError:
                            values.append(999)
                
                # 添加到数据数组
                start_lon = 275
                for lon_idx in range(13):
                    ozone_data[start_lon + lon_idx, lat_idx] = values[lon_idx]
        
        # 处理缩放和缺失值
        # 原始值乘以10存储，需要除以10
        # 缺失值999转换为NaN
        ozone_data = ozone_data.astype(float)
        ozone_data[ozone_data == 999] = np.nan
        ozone_data /= 10.0
        
        # 转置为 (lat, lon) 顺序以符合CF标准
        ozone_data = ozone_data.T  # 现在形状为 (nlat, nlon)
        
        return latitudes, longitudes, ozone_data, date
    
    except Exception as e:
        logger.error(f"解析文件失败: {fname} - {str(e)}")
        return None

def extract_date_from_filename(filename):
    """
    从文件名中提取日期信息 - 针对数字文件名优化
    
    参数:
        filename (str): 文件路径
        
    返回:
        datetime: 解析出的日期
    """
    # 从文件名提取数字部分
    filename_base = os.path.basename(filename)
    match = re.search(r'(\d+)', filename_base)
    
    if match:
        try:
            month_num = int(match.group(1))
            # 确保月份在1-12范围内
            if 1 <= month_num <= 12:
                return datetime(DATA_YEAR, month_num, 15)
        except ValueError:
            pass
    
    logger.warning(f"无法从文件名解析日期: {filename}, 使用当前日期")
    return datetime.now()

# ===================== NetCDF 创建函数 =====================
def create_netcdf_dataset(lats, lons, ozone, date, source_file):
    """
    创建NetCDF数据集
    
    参数:
        lats (array): 纬度数组
        lons (array): 经度数组
        ozone (array): 臭氧数据数组
        date (datetime): 日期
        source_file (str): 源文件名
        
    返回:
        xarray.Dataset: NetCDF数据集
    """
    # 创建时间数组 (确保为datetime64类型)
    time_array = np.array([date], dtype='datetime64[ns]')
    
    return xr.Dataset(
        data_vars={
            'tco': (('time', 'latitude', 'longitude'), ozone[np.newaxis, :, :], {
                'long_name': 'Tropospheric Column Ozone',
                'units': 'Dobson Units',
                'description': 'OMI/MLS tropospheric ozone column',
                'source': 'NASA OMI/MLS retrieval',
                'processing_level': 'Monthly Mean',
                'scaling': 'Values stored in ASCII were multiplied by 10',
                'missing_value': 'NaN',
                'source_file': os.path.basename(source_file)
            })
        },
        coords={
            'latitude': ('latitude', lats, {
                'long_name': 'Latitude',
                'units': 'degrees_north',
                'axis': 'Y',
                'standard_name': 'latitude'
            }),
            'longitude': ('longitude', lons, {
                'long_name': 'Longitude',
                'units': 'degrees_east',
                'axis': 'X',
                'standard_name': 'longitude'
            }),
            'time': ('time', time_array, {
                'long_name': 'Time',
                'standard_name': 'time',
            }),
        },
        attrs={
            'title': f'OMI/MLS Tropospheric Column Ozone ({date.strftime("%B %Y")})',
            'institution': 'NASA/Goddard Space Flight Center',
            'source': 'OMI total ozone minus MLS stratospheric ozone',
            'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} from {os.path.basename(source_file)}',
            'references': 'Ziemke, J. R., et al. (2006), Tropospheric ozone determined from Aura OMI and MLS: Evaluation of measurements and comparison with the Global Modeling Initiative\'s Chemical Transport Model, J. Geophys. Res., 111, D19303, doi:10.1029/2006JD007089.',
            'comment': 'Data filtered for near clear-sky conditions (OMI reflectivity < 0.3)',
            'Conventions': 'CF-1.8',
            'geospatial_lat_min': float(np.min(lats)),
            'geospatial_lat_max': float(np.max(lats)),
            'geospatial_lon_min': float(np.min(lons)),
            'geospatial_lon_max': float(np.max(lons)),
            'time_coverage_start': date.strftime("%Y-%m-%d"),
            'time_coverage_end': date.strftime("%Y-%m-%d"),
            'creator_name': 'NASA/GSFC',
            'creator_email': 'data@ozone.gsfc.nasa.gov',
            'project': 'Aura Data Products',
        }
    )

def save_as_netcdf(ds, output_path):
    """
    将数据集保存为NetCDF文件 - 修复时间序列序列化问题
    
    参数:
        ds (xarray.Dataset): 数据集
        output_path (str): 输出文件路径
    """
    try:
        # 设置编码 - 彻底解决时间序列序列化问题
        encoding = {
            'tco': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
                'chunksizes': (1, 60, 96),
                '_FillValue': -9999.0
            },
            'latitude': {'dtype': 'float32'},
            'longitude': {'dtype': 'float32'},
            'time': {
                'dtype': 'float64',  # 使用浮点数而不是整数
                'units': 'days since 1970-01-01',
                'calendar': 'proleptic_gregorian'
            }
        }
        
        # 保存为NetCDF
        ds.to_netcdf(
            path=output_path,
            format='NETCDF4',
            encoding=encoding,
            engine='netcdf4'
        )
        
        return True
    
    except Exception as e:
        logger.error(f"保存NetCDF失败: {output_path} - {str(e)}")
        return False

# ===================== 文件处理函数 =====================
def convert_file(input_file):
    """
    转换单个文件
    
    参数:
        input_file (str): 输入文件路径
        
    返回:
        bool: 是否转换成功
    """
    if VERBOSE:
        logger.info(f"开始处理: {input_file}")
    
    # 解析数据
    result = read_lvl3(input_file)
    if result is None:
        logger.error(f"解析失败: {input_file}")
        return False
    
    lats, lons, ozone, date = result
    
    if VERBOSE:
        logger.info(f"成功读取数据: {input_file}")
        logger.info(f"日期: {date.strftime('%Y-%m')}")
        
        # 计算统计信息
        valid_mask = ~np.isnan(ozone)
        if np.any(valid_mask):
            min_val = np.nanmin(ozone)
            max_val = np.nanmax(ozone)
            missing_percent = 100 * (1 - np.sum(valid_mask) / ozone.size)
            logger.info(f"数据范围: {min_val:.1f} - {max_val:.1f} DU")
            logger.info(f"缺失值比例: {missing_percent:.2f}%")
        else:
            logger.warning("没有有效数据!")
    
    # 创建数据集
    ds = create_netcdf_dataset(lats, lons, ozone, date, input_file)
    
    # 确定输出路径
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建输出文件名
    output_filename = f"{date.strftime('%Y%m')}_omi_mls_tco.nc"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 保存文件
    if save_as_netcdf(ds, output_path):
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"创建文件: {output_path} ({file_size:.2f} MB)")
            return True
        else:
            logger.error(f"文件创建失败: {output_path}")
            return False
    else:
        return False

def process_directory():
    """
    处理目录中的所有文件
    """
    # 获取文件列表
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    
    if not input_files:
        logger.error(f"在 {INPUT_DIR} 中未找到文件")
        return
    
    if VERBOSE:
        logger.info(f"找到 {len(input_files)} 个文件需要转换")
    
    # 转换每个文件
    success_count = 0
    for i, file_path in enumerate(input_files):
        if VERBOSE:
            logger.info(f"处理文件 {i+1}/{len(input_files)}: {file_path}")
        
        try:
            if convert_file(file_path):
                success_count += 1
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
    
    logger.info(f"处理完成! 成功转换 {success_count}/{len(input_files)} 个文件")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    logger.info("开始 OMI/MLS 数据转换")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"数据年份: {DATA_YEAR}")
    logger.info(f"详细模式: {'是' if VERBOSE else '否'}")
    
    process_directory()
    
    logger.info("处理结束")