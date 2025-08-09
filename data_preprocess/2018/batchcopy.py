import os
import ftplib
from datetime import datetime, timedelta

def download_single_file_per_day():
    # 配置参数 ================================
    base_ftp_path = "/pub/SGLI_NRT/L2_In-water_properties/CHLA/"
    year = 2018  # 要下载的年份
    local_dir = "C:/RIP_D/Codes/Python/QWeather/DataProcess/datas/chla"  # 所有文件直接放在这里
    file_suffix = "CHLA.nc"  # 文件后缀 (如 ".nc", ".hdf", ".dat")   
    file_prefix = "GC1SG1_"  # 文件前缀
    host = "apollo.eorc.jaxa.jp"
    timeout = 60  # FTP连接超时（秒）
    max_retries = 3  # 失败重试次数
    
    # 匿名FTP登录凭证
    username = "anonymous"
    password = "your_email@example.com"  # 使用您的邮箱
    
    # 创建本地存储目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 建立FTP连接
    print(f"正在连接到 {host}...")
    try:
        ftp = ftplib.FTP(host, timeout=timeout)
        ftp.login(username, password)
        print("FTP登录成功")
    except Exception as e:
        print(f"连接失败: {e}")
        return

    # 计算日期范围 ==============================
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    total_days = (end_date - start_date).days + 1
    files_downloaded = 0

    print(f"开始处理 {year} 年的数据，共 {total_days} 天")
    
    # 遍历每一天
    current_date = start_date
    while current_date <= end_date:
        # 构建日期部分 (YYYYMMDD)
        date_str = current_date.strftime("%Y%m%d")
        file_name_start = f"{file_prefix}{date_str}"  # 如 "GC1SG1_20180121"
        
        # 检查本地是否已有相同日期的文件
        existing_files = [f for f in os.listdir(local_dir) 
                          if f.startswith(file_name_start) and f.endswith(file_suffix)]
        
        if existing_files:
            print(f"跳过 {current_date.strftime('%Y-%m-%d')}: 本地已有文件 {existing_files[0]}")
            current_date += timedelta(days=1)
            continue
        
        # 构建远程路径
        remote_dir = os.path.join(
            base_ftp_path,
            str(current_date.year),
            f"{current_date.month:02d}",
            f"{current_date.day:02d}"
        ).replace("\\", "/")  # 确保使用正斜杠
        
        # 检查远程目录是否存在
        print(f"处理 {current_date.strftime('%Y-%m-%d')}: ", end="")
        
        try:
            # 尝试切换目录 (目录不存在会抛出异常)
            ftp.cwd(remote_dir)
            print(f"目录存在")
        except ftplib.error_perm:
            print(f"目录不存在")
            current_date += timedelta(days=1)
            continue
        
        # 获取文件列表
        try:
            file_list = []
            ftp.retrlines("NLST", file_list.append)
        except Exception as e:
            print(f"  获取文件列表失败: {e}")
            current_date += timedelta(days=1)
            continue
        
        # 后缀匹配文件名
        target_files = [f for f in file_list 
                        if f.startswith(file_prefix)  # 匹配前缀
                        and f.endswith(file_suffix)]   # 匹配后缀
        
        if not target_files:
            print(f"  没有匹配 {file_name_start}*{file_suffix} 的文件")
            current_date += timedelta(days=1)
            continue
        
        # 只取第一个匹配的文件
        remote_file = target_files[0]
        print(f"  找到匹配文件: {remote_file}")
        
        local_path = os.path.join(local_dir, remote_file)
            
        # 带重试机制的下载
        for attempt in range(max_retries):
            try:
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {remote_file}", f.write)
                print(f"    下载成功: {remote_file}")
                files_downloaded += 1
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
                else:
                    print(f"    ❌ 最终下载失败: {remote_file} - {e}")
        
        current_date += timedelta(days=1)
    
    # 关闭连接和统计结果
    ftp.quit()
    print("\n" + "=" * 50)
    print(f"处理完成!")
    print(f"  遍历天数: {total_days}")
    print(f"  成功下载文件: {files_downloaded}")
    print(f"  所有文件保存在: {os.path.abspath(local_dir)}")
    print("=" * 50)

if __name__ == "__main__":
    download_single_file_per_day()