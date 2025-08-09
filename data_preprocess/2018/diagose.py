import numpy as np
import os
import logging
import re
import matplotlib.pyplot as plt
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('omi_mls_diagnostic.log')
    ]
)
logger = logging.getLogger('omi_mls_diagnostic')

def diagnose_file(filepath):
    """
    诊断 OMI/MLS 数据文件问题
    
    参数:
        filepath (str): 文件路径
    """
    logger.info(f"开始诊断文件: {filepath}")
    
    # 读取文件内容
    with open(filepath, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    
    logger.info(f"文件行数: {len(lines)}")
    
    # 1. 检查文件头部
    logger.info("\n===== 文件头部检查 =====")
    for i in range(min(5, len(lines))):
        logger.info(f"行 {i+1}: {lines[i]}")
    
    # 2. 检查数据起始位置
    logger.info("\n===== 数据起始位置检查 =====")
    data_start_idx = 3  # 默认从第4行开始
    for i in range(min(10, len(lines))):
        if len(lines[i]) > 200 and any(char.isdigit() for char in lines[i]):
            data_start_idx = i
            logger.info(f"在行 {i+1} 发现可能的数据行: {lines[i][:50]}...")
            break
    
    logger.info(f"数据起始行: {data_start_idx+1}")
    
    # 3. 检查数据行格式
    logger.info("\n===== 数据行格式检查 =====")
    for i in range(data_start_idx, min(data_start_idx+5, len(lines))):
        line = lines[i]
        logger.info(f"行 {i+1}: 长度={len(line)}, 数字字符数={sum(c.isdigit() for c in line)}")
        
        # 检查行首是否有空格
        if line.startswith(' '):
            logger.info("  行首有空格 (符合IDL格式)")
        else:
            logger.info("  行首无空格 (可能不符合IDL格式)")
    
    # 4. 解析示例数据行
    logger.info("\n===== 数据行解析示例 =====")
    if data_start_idx < len(lines):
        sample_line = lines[data_start_idx]
        logger.info(f"示例数据行: {sample_line[:50]}...")
        
        # 尝试解析前10个值
        values = []
        clean_line = sample_line.lstrip()  # 跳过行首空格
        
        for i in range(10):
            if len(clean_line) < 3:
                logger.info(f"  位置 {i}: 行长度不足")
                break
            
            chunk = clean_line[:3]
            clean_line = clean_line[3:]
            
            try:
                int_val = int(chunk)
                values.append(int_val)
                logger.info(f"  位置 {i}: 值={int_val}")
            except ValueError:
                logger.info(f"  位置 {i}: 无效值 '{chunk}'")
    
    # 5. 检查纬度标签行
    logger.info("\n===== 纬度标签行检查 =====")
    if data_start_idx + 1 < len(lines):
        lat_line = lines[data_start_idx + 1]
        logger.info(f"行 {data_start_idx+2}: {lat_line}")
        
        # 尝试提取纬度值
        match = re.search(r'lat\s*=\s*([-+]?\d*\.?\d+)', lat_line)
        if match:
            lat_value = float(match.group(1))
            logger.info(f"  提取的纬度值: {lat_value}")
        else:
            logger.info("  未找到纬度值")
    
    # 6. 创建数据可视化
    logger.info("\n===== 创建数据可视化 =====")
    try:
        # 创建经度数组 (固定值)
        longitudes = np.around(np.arange(-179.375, 180.0, 1.25), decimals=3)
        
        # 创建纬度数组 (固定值)
        latitudes = np.arange(-59.5, 60.0, 1.0)
        
        # 初始化臭氧数据数组
        ozone_data = np.full((len(latitudes), len(longitudes)), np.nan)
        
        # 处理每个纬度带
        current_lat_idx = 0
        line_idx = data_start_idx
        
        for lat_idx in range(len(latitudes)):
            if line_idx >= len(lines):
                break
            
            # 处理前11行 (每行25个值)
            for irow in range(11):
                if line_idx >= len(lines):
                    break
                
                line = lines[line_idx]
                line_idx += 1
                
                # 解析25个整数值
                values = []
                clean_line = line.lstrip()
                
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
                    ozone_data[lat_idx, start_lon + lon_idx] = values[lon_idx]
            
            # 处理最后一行 (13个值 + 纬度标签)
            if line_idx >= len(lines):
                break
            
            line = lines[line_idx]
            line_idx += 1
            
            # 解析13个整数值
            values = []
            clean_line = line.lstrip()
            
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
                ozone_data[lat_idx, start_lon + lon_idx] = values[lon_idx]
        
        # 处理缩放和缺失值
        ozone_data = ozone_data.astype(float)
        ozone_data[ozone_data == 999] = np.nan
        ozone_data /= 10.0
        
        # 计算有效值比例
        valid_count = np.sum(~np.isnan(ozone_data))
        total_count = ozone_data.size
        valid_percent = (valid_count / total_count) * 100 if total_count > 0 else 0
        
        logger.info(f"解析完成: {valid_count}/{total_count} 个有效值 ({valid_percent:.2f}%)")
        
        # 创建可视化
        plt.figure(figsize=(12, 6))
        
        # 纬向平均值
        zonal_mean = np.nanmean(ozone_data, axis=1)
        plt.subplot(1, 2, 1)
        plt.plot(latitudes, zonal_mean)
        plt.title('纬向平均臭氧柱总量')
        plt.xlabel('纬度')
        plt.ylabel('臭氧柱总量 (DU)')
        plt.grid(True)
        
        # 数据分布直方图
        plt.subplot(1, 2, 2)
        valid_values = ozone_data[~np.isnan(ozone_data)]
        if len(valid_values) > 0:
            plt.hist(valid_values, bins=50)
            plt.title('臭氧值分布')
            plt.xlabel('臭氧柱总量 (DU)')
            plt.ylabel('频率')
        else:
            plt.text(0.5, 0.5, '无有效数据', ha='center', va='center')
        
        plt.tight_layout()
        
        # 保存图像
        filename = Path(filepath).stem
        output_path = f"{filename}_diagnostic.png"
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"创建诊断图像: {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"创建可视化失败: {str(e)}")
        return False

def main():
    # 输入文件路径
    filepath = r"C:\RIP_D\Codes\Python\QWeather\DataProcess\datas\o3\o3_ori\1.txt"  # 替换为您的文件路径
    
    diagnose_file(filepath)
    logger.info("诊断完成")

if __name__ == "__main__":
    main()