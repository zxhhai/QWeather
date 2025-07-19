import os
import urllib.request
import urllib.error
import base64
import time
import shutil
import ssl
import http.cookiejar
from urllib.parse import urlparse

def download_nasa_gesdisc_files():
    # ======= 配置参数 ========
    input_file = r"DataProcess\datas\subset_OMHCHOd_003_20250714_075208_.txt"      # 包含下载链接的文本文件
    username = r"h3licopter"   # 替换为实际用户名
    password = r"daiGANG.6260"   # 替换为实际密码
    local_dir = r"DataProcess\datas\hcho"     # 下载文件保存目录
    max_retries = 5              # 增加重试次数（NASA服务器常需多次尝试）
    retry_delay = 10             # 增加重试延迟（秒）
    timeout = 120                # 延长超时时间（处理大文件）
    # ========================
    
    # 创建具有NASA兼容性的SSL上下文（解决SSL问题）
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # 创建cookie处理器（处理NASA重定向）
    cookie_jar = http.cookiejar.CookieJar()
    cookie_processor = urllib.request.HTTPCookieProcessor(cookie_jar)
    
    # 创建本地存储目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 从文件读取URL列表
    try:
        with open(input_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取文件出错: {e}")
        return
    
    total_files = len(urls)
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    print(f"开始处理 {total_files} 个NASA EarthData文件...")
    print("=" * 70)
    
    # 创建基本认证头
    credentials = f"{username}:{password}"
    base64_credentials = base64.b64encode(credentials.encode()).decode()
    auth_header = f"Basic {base64_credentials}"
    
    # NASA认证端点（解决重定向问题）
    auth_url = "https://urs.earthdata.nasa.gov/users/auth"
    
    # 创建opener（包含cookie处理器）
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context),
        cookie_processor
    )
    opener.addheaders = [("Authorization", auth_header)]
    
    # 首先进行NASA认证（解决401和重定向问题）
    print("正在尝试连接NASA EarthData身份验证服务...")
    for auth_retry in range(max_retries):
        try:
            req = urllib.request.Request(auth_url)
            response = opener.open(req, timeout=timeout)
            if response.getcode() == 200:
                print("✅ NASA EarthData身份验证成功")
                break
            else:
                print(f"⚠️ 认证尝试 {auth_retry+1}/{max_retries}: HTTP {response.getcode()}")
        except Exception as e:
            print(f"⚠️ 认证尝试 {auth_retry+1}/{max_retries}: {type(e).__name__} - {str(e)}")
        
        time.sleep(retry_delay)
    else:
        print("❌ 无法完成NASA EarthData身份验证，请检查用户名和密码")
        return
    
    # 文件下载主循环
    for idx, url in enumerate(urls, 1):
        filename = url.split('/')[-1]
        save_path = os.path.join(local_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(save_path):
            print(f"[{idx}/{total_files}] 跳过已存在文件: {filename}")
            skip_count += 1
            continue
        
        # 提取URL域名
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # 文件大小未知（NASA没有Content-Length头部）
        print(f"[{idx}/{total_files}] 开始下载: {filename} [大小未知 - NASA不提供文件大小信息]")
        
        # NASA数据下载需要特殊处理的标志
        is_nasa_data_file = "gesdisc.eosdis.nasa.gov" in domain or "acdisc.gesdisc.eosdis.nasa.gov" in domain
        
        # 带重试机制的下载
        for attempt in range(1, max_retries + 1):
            try:
                # NASA数据文件需要使用自定义Opener（处理重定向）
                if is_nasa_data_file:
                    req = urllib.request.Request(url)
                    response = opener.open(req, timeout=timeout)
                # 其他文件（如PDF）使用标准请求
                else:
                    req = urllib.request.Request(url)
                    req.add_header("Authorization", auth_header)
                    response = urllib.request.urlopen(req, timeout=timeout, context=ssl_context)
                
                # 使用临时文件下载（避免中断导致文件损坏）
                temp_path = save_path + ".tmp"
                downloaded_bytes = 0
                
                # 处理重定向
                if response.geturl() != url:
                    print(f"   ↳ 重定向: {url} → {response.geturl()}")
                
                # 块下载（NASA大文件可能需要块读取）
                with open(temp_path, 'wb') as f:
                    while True:
                        chunk = response.read(16384)  # 增大块大小
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        print(f"\r   ↳ 已下载: {downloaded_bytes / (1024 * 1024):.2f} MB", end='')
                
                # 重命名临时文件
                shutil.move(temp_path, save_path)
                success_count += 1
                print(f"\n   ✅ 下载成功: {filename} ({downloaded_bytes / (1024 * 1024):.2f} MB)")
                break
            
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    print(f"\n   ❌ 认证失败: 请检查用户名和密码")
                    fail_count += 1
                    break
                elif e.code == 302:
                    print(f"\n   ⚠️ 尝试 {attempt}/{max_retries}: 重定向错误 (302)")
                else:
                    print(f"\n   ⚠️ 尝试 {attempt}/{max_retries}: HTTP错误 {e.code}")
            
            except Exception as e:
                print(f"\n   ⚠️ 尝试 {attempt}/{max_retries}: {type(e).__name__} - {str(e)}")
            
            if attempt < max_retries:
                print(f"   等待 {retry_delay * attempt} 秒后重试...")
                time.sleep(retry_delay * attempt)
        else:
            print(f"\n   ❌ 下载失败: {filename}")
            fail_count += 1
            
            # 清理可能存在的临时文件
            if os.path.exists(save_path + ".tmp"):
                os.remove(save_path + ".tmp")
    
    # 打印摘要信息
    print("\n" + "=" * 70)
    print("NASA EarthData下载摘要:")
    print(f"  总文件数: {total_files}")
    print(f"  成功下载: {success_count}")
    print(f"  跳过已存在: {skip_count}")
    print(f"  下载失败: {fail_count}")
    
    if fail_count > 0:
        print("\n故障排除建议:")
        print("1. 检查NASA EarthData账户有效性 - 访问 https://urs.earthdata.nasa.gov")
        print("2. 确保账户有权限访问OMNO2数据集")
        print("3. 网络问题导致 - 尝试使用VPN或不同网络")
        print("4. 服务器端问题 - 稍后重试或分批下载")
    
    print(f"\n文件保存位置: {os.path.abspath(local_dir)}")
    print("=" * 70)

if __name__ == "__main__":
    download_nasa_gesdisc_files()