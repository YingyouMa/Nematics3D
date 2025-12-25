# import numpy as np
# import pyvista as pv

# # -----------------------------
# # 1. 构造两条三维曲线
# # -----------------------------
# t = np.linspace(0, 4 * np.pi, 400)

# curve1 = np.c_[
#     np.cos(t),
#     np.sin(t),
#     0.3 * t
# ]

# curve2 = np.c_[
#     np.cos(t + np.pi),
#     np.sin(t + np.pi),
#     0.3 * t
# ]

# # 转为 PolyData
# line1 = pv.Spline(curve1, len(curve1))
# line2 = pv.Spline(curve2, len(curve2))

# # 管道化（这是“质感”的关键）
# tube1 = line1.tube(radius=0.05)
# tube2 = line2.tube(radius=0.05)

# # -----------------------------
# # 2. 创建 Plotter（启用阴影）
# # -----------------------------
# pl = pv.Plotter(lighting="three lights")
# pl.enable_shadows()

# # 背景色（微灰，增强对比）
# pl.set_background("#e6e6e6")

# # -----------------------------
# # 3. 添加第一条线（红色、偏金属）
# # -----------------------------
# pl.add_mesh(
#     tube1,
#     color="#b22222",
#     smooth_shading=True,
#     pbr=True,
#     metallic=0.4,
#     roughness=0.25,
#     specular=0.8,
#     specular_power=60,
# )

# # -----------------------------
# # 4. 添加第二条线（蓝色、偏塑料）
# # -----------------------------
# pl.add_mesh(
#     tube2,
#     color="#1f77b4",
#     smooth_shading=True,
#     pbr=True,
#     metallic=0.1,
#     roughness=0.45,
#     specular=0.6,
#     specular_power=40,
# )

# # -----------------------------
# # 5. 相机设置（非常重要）
# # -----------------------------
# pl.camera_position = [
#     (6, 6, 4),   # camera position
#     (0, 0, 3),   # focal point
#     (0, 0, 1),   # view up
# ]

# # -----------------------------
# # 6. 地面投影（增强空间感）
# # -----------------------------
# # pl.add_floor(
# #     face="z",
# #     color="white",
# #     lighting=True,
# #     pad=1.0,
# # )

# # -----------------------------
# # 7. 显示
# # -----------------------------
# pl.show()

import numpy as np
import pyvista as pv
import sys
sys.path.insert(0, 'D:/Document/GitHub/')
from Nematics3D import PlotTube

# 假设你的类都定义在当前环境或已导入
# from your_module import PlotTube, OptsTube

# 1. 创建非阻塞画布
# 使用 .show(interactive_update=True) 或直接简单的 show() 
# 在 Jupyter 中通常是非阻塞的，在脚本中我们需要启动交互
plotter = pv.Plotter()
plotter.show(interactive_update=True) 

# 准备三条线的坐标数据（简单的螺旋线或直线）
z = np.linspace(0, 10, 50)
x = np.sin(z)
y = np.cos(z)

coords1 = np.column_stack((x, y, z))          # 第一条线：左侧
coords2 = np.column_stack((x + 5, y, z))      # 第二条线：中间
coords3 = np.column_stack((x + 10, y, z))     # 第三条线：右侧

# --- 案例 1：使用元组着色 (单一颜色) ---
# 逻辑：OptsTube 会通过 as_ColorRGB 校验，然后 _helper_resolve_color 会执行 np.tile
tube1 = PlotTube(
    coords=coords1, 
    plotter=plotter, 
    name="solid_blue",
    color_rule=(0.2, 0.2, 0.8), # 蓝色
    radius_rule=0.3,
    sides=12
)

# --- 案例 2：使用函数着色 (动态颜色) ---
# 逻辑：传入一个函数，根据 Z 轴高度映射颜色
def my_color_func(coords):
    # 根据高度生成从红到绿的渐变
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    normalized_z = (coords[:, 2] - z_min) / (z_max - z_min)
    colors = np.zeros((len(coords), 3))
    colors[:, 0] = normalized_z      # 红色分量随高度增加
    colors[:, 1] = 1 - normalized_z  # 绿色分量随高度减少
    return colors

tube2 = PlotTube(
    coords=coords2, 
    plotter=plotter, 
    name="func_gradient",
    color_rule=my_color_func,
    radius_rule=0.4,
    sides=16
)

# --- 案例 3：使用 manual 模式 (手动输入数据) ---
# 逻辑：设置 manual 模式，并手动提供 color_values
# 我们故意给一个随机颜色数组
manual_colors = np.random.rand(len(coords3), 3)

tube3 = PlotTube(
    coords=coords3, 
    plotter=plotter, 
    name="manual_random",
    color_rule="manual",
    color_values=manual_colors, # 必须手动提供，否则会被 recovery 救回成黑色
    radius_rule=0.2,
    sides=8,
)

# 调整视角并渲染
plotter.reset_camera()
plotter.render()



# 如果你在纯 Python 脚本中运行，可以加一个简单的循环保持窗口
# while True:
#     plotter.update()
#     time.sleep(0.01)


