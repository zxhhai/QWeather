import numpy as np
import xarray as xr
from scipy.integrate import quad
from functools import partial
import dask

def calculate_isoprene_concentration(
    # 核心遥感输入参数
    Chla_surface: float,       # 海表叶绿素浓度 [mg/m³] (卫星反演)
    PAR_0: float,              # 海表光合有效辐射 [μE/m²/s] (卫星反演)
    SST: float,                # 海表温度 [°C] (卫星反演)
    
    
    # 海洋物理参数
    wind_speed: float,         # 风速 [m/s] (再分析数据)
    D_ML: float = 1,               # 混合层深度 [m] (再分析数据)
    ## D_ML待定

    # 浮游植物特征参数
    delta: float = 1,              # 温度修正项 δ = 23.375 - T_opt [°C]
    ## delta待定，暂未找到来源

    # 模型系数 (默认值来自论文)
    EF: float = 0.042,                 # 排放因子 (查表Supplementary Table 1)
    ## EF待定，与海域有关
    a1: float = 3.6402,        # 温度响应系数1
    a2: float = -46.75,        # 温度响应系数2
    a3: float = 618.2,         # 温度响应系数3
    beta: float = 86.8798,     # 源强缩放因子
    mu: float = 0.14,          # 生物消耗系数
    k_chem: float = 0.05,      # 化学消耗率 [d⁻¹]
    k_mix: float = -0.005      # 混合损失系数 [d⁻¹]
) -> float:
    """
    计算海水异戊二烯浓度 (Cₘ) [nmol/L]
    
    参数:
    Chla_surface -- 海表叶绿素浓度 [mg/m³]
    PAR_0 -- 海表光合有效辐射 [μE/m²/s]
    SST -- 海表温度 [°C]
    EF -- 浮游植物排放因子
    delta -- 温度修正项 [°C]
    D_ML -- 混合层深度 [m]
    wind_speed -- 风速 [m/s]
    (其余为模型系数，可使用默认值)
    
    返回:
    C_m -- 海水异戊二烯浓度 [nmol/L]
    """
    # 1. 计算真光层叶绿素总量 [mg/m²]
    if Chla_surface <= 0.5:
        Chla_tot = 38.0 * (Chla_surface ** 0.425)
    else:
        Chla_tot = 40.2 * (Chla_surface ** 0.507)
    
    # 2. 计算PAR衰减系数k_d [m⁻¹]
    if Chla_tot <= 13.62:
        k_d = (4.6 / 426.3) * (Chla_tot ** 0.547)
    else:
        k_d = (4.6 / 912.5) * (Chla_tot ** 0.839)
    
    # 3. 计算真光层深度H_max [m]
    I_cutoff = 2.5 * 4.57  # 将2.5 W/m²转换为μE/m²/s (转换因子4.57)
    H_max = -np.log(I_cutoff / PAR_0) / k_d if PAR_0 > I_cutoff else 0
    
    # 4. 计算有效深度H [m]
    H = min(H_max, D_ML)
    
    # 5. 计算平均叶绿素浓度 [mg/m³]
    Chla_avg = Chla_tot / H_max if H_max > 0 else 0
    
    # 6. 计算异戊二烯产生率积分
    def production_rate(h):
        """深度h处的异戊二烯产生率"""
        # 水下光辐射衰减
        I_h = PAR_0 * np.exp(-k_d * h)
        # 温度响应函数分母
        T_term = (SST + delta)**2 + a2*(SST + delta) + a3
        # 完整产生率公式
        return EF * (a1 / T_term) * (np.log(I_h)**2)
    
    # 对产生率从0到H进行积分
    integral, _ = quad(production_rate, 0, H)
    
    # 7. 计算源强P [nmol/m³/d]
    P = (beta / D_ML) * Chla_avg * integral * 24  # 24为小时转天
    
    # 8. 计算生物消耗率 [d⁻¹]
    k_bio = mu * (Chla_avg ** 1.28)
    
    # 9. 计算海气通量系数 [d⁻¹]
    # 先计算施密特数
    Sc = 2675.0 - 147.12*SST + 3.726*SST**2 - 0.038*SST**3
    # 气体传输速度 [m/d]
    k_w = (0.222 * wind_speed**2 + 0.333 * wind_speed) * (Sc/600)**(-0.5) * 0.01 * 24
    
    # 10. 计算异戊二烯浓度Cₘ [nmol/L]
    denominator = k_bio + k_chem + (k_w / D_ML) - k_mix
    C_m = P / denominator if denominator != 0 else 0
    
    return C_m


# 包装函数，处理NaN和异常
def isoprene_wrapper(chla, par, sst, wind, D_ML=1, delta=1, EF=0.042):
    if any(np.isnan([chla, par, sst, wind])):
        return np.nan
    try:
        return calculate_isoprene_concentration(
            Chla_surface=chla,
            PAR_0=par,
            SST=sst,
            wind_speed=wind,
            D_ML=D_ML,
            delta=delta,
            EF=EF
        )
    except Exception as e:
        print(f"Error: {e} at values {chla}, {par}, {sst}, {wind}")
        return np.nan

# 主处理流程
def process_netcdf(input_path, output_path):
    # 1. 加载数据
    ds = xr.open_dataset(input_path, chunks={'time': 1, 'lat': 100, 'lon': 100})
    
    # 2. 创建固定参数的包装函数
    calc_isoprene = partial(isoprene_wrapper, D_ML=1, delta=1, EF=0.042)
    
    # 3. 并行计算异戊二烯浓度
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        isoprene = xr.apply_ufunc(
            calc_isoprene,
            ds['CHLA'],
            ds['PAR'],
            ds['SST'],
            ds['windspeed'],
            input_core_dims=[[], [], [], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            output_core_dims=[[]]
        )
    
    # 4. 创建包含原始变量+异戊二烯的结果文件
    ds_with_isoprene = ds.copy()
    ds_with_isoprene['isoprene'] = isoprene
    ds_with_isoprene.to_netcdf(output_path)
    
    # 5. 创建仅含异戊二烯的新文件
    isoprene_ds = xr.Dataset({'isoprene': isoprene})
    isoprene_ds.to_netcdf(output_path.replace('.nc', '_isoprene_only.nc'))

# 执行处理
if __name__ == "__main__":
    input_nc = r"DataProcess\datas\combined_data.nc"   # 替换为实际输入路径
    output_nc = "isoprene_results.nc" # 输出文件路径
    process_netcdf(input_nc, output_nc)
    print("处理完成！生成了两个新文件:")
    print(f"- 包含所有变量的结果: {output_nc}")
    print(f"- 仅含异戊二烯的结果: {output_nc.replace('.nc', '_isoprene_only.nc')}")