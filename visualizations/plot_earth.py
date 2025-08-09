import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. 创建球体网格（地球模型）
sphere = pv.Sphere(radius=1.0, theta_resolution=360, phi_resolution=180)

# 2. 生成纹理图像（示例：纬度相关热力图）
# 经纬度范围：theta: 0~360°, phi: 0~180°
width, height = 360, 180

# 构造热力图数据，示例为纬度相关，模拟异戊二烯排放强度（归一化0~1）
phi = np.linspace(0, np.pi, height)  # 纬度，从北极到南极
theta = np.linspace(0, 2*np.pi, width)  # 经度

# meshgrid for texture image
phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')

# 示例热力图数据：假设排放与纬度成函数关系
heatmap = np.cos(phi_grid)**2  # 0~1，最大在赤道

# 转换成RGB图像
cmap = cm.plasma
heatmap_rgb = cmap(heatmap)  # RGBA，shape=(height, width, 4)
heatmap_rgb = (heatmap_rgb[:, :, :3] * 255).astype(np.uint8)  # 去alpha，转255色阶

# 3. 创建 Pyvista 纹理对象
texture = pv.Texture(heatmap_rgb)

# 4. 纹理映射
sphere.texture_map_to_sphere(inplace=True)

# 5. 渲染
plotter = pv.Plotter()
plotter.add_mesh(sphere, texture=texture)
plotter.add_text("3D Earth with Isoprene Heatmap Texture", font_size=12)

# 设置背景色为黑色更显眼
plotter.set_background('black')

# 6. 显示交互窗口
plotter.show()
