import sys
sys.path.insert(0, 'D:/Document/GitHub/')
from Nematics3D import PlotTube

import numpy as np
import pyvista as pv


# --- 准备测试数据和函数 ---
def get_path(offset_y=0):
    z = np.linspace(0, 10, 50)
    return np.column_stack((np.sin(z), np.cos(z) + offset_y, z))

def radius_wave(coords):
    return 0.1 + 0.2 * np.abs(np.sin(coords[:, 2]))

def color_func(coords):
    z_norm = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min())
    return np.column_stack((z_norm, np.zeros_like(z_norm), 1 - z_norm))

def opacity_func(coords):
    opacity = np.abs(np.sin(coords[:, 2]))
    return opacity

plotter = pv.Plotter()

# --- 案例 1: 固定参数 (类似于你给的例子) ---
tube1 = PlotTube(
    coords=get_path(offset_y=0), 
    plotter=plotter, 
    name="solid_blue",
    color_rule=(0,0,1), # 蓝色
    radius_rule=0.3,
    sides=12,
    is_capping=False,
)

# --- 案例 2: 函数驱动 (一次性传入函数) ---
tube2 = PlotTube(
    coords=get_path(offset_y=5),
    plotter=plotter,
    name="functional_pbr",
    color_rule=color_func,     # 渐变色函数
    radius_rule=radius_wave,   # 波动半径函数
    opacity_rule=opacity_func,
)

# --- 案例 3: 手动模式 + 透明度 ---
tube3 = PlotTube(
    coords=get_path(offset_y=10),
    plotter=plotter,
    name="manual_alpha",
    color_rule="manual",
    color_values=np.random.rand(50, 3), # 随机色数组
    radius_rule=0.15,
    opacity_rule=0.5,
    shading_type='pbr',
    metallic=1,
    roughness=0.4    
)

tube4 = PlotTube(
    coords=get_path(offset_y=15),
    plotter=plotter,
    name="func_scalars",
    sides=20,
    color_rule="scalars",
    radius_rule=0.25,
    opacity_rule=1,     
    scalars_rule=radius_wave, 
    cmap='plasma'     
)

# tube5 = PlotTube(
#     coords=get_path(offset_y=20),
#     plotter=plotter,
#     name="func_scalars2",
#     sides=20,
#     color_rule="scalars",
#     radius_rule=0.25,
#     opacity_rule=1,     
#     scalars_rule=radius_wave,  
#     shading_type='pbr',
#     metallic=1,
#     roughness=0.4,
#     clim=(0,0.2)
# )

plotter.show(interactive_update=True)