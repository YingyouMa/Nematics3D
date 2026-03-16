import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

import numpy as np
import pyvista as pv


# --- 准备测试数据和函数 ---
def get_path(offset_y=0):
    z = np.linspace(0, 10, 50)
    line1 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    z = np.linspace(15, 20, 25)
    line2 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    return np.concatenate([line1, line2])

def radius_wave(coords):
    return 0.1 + 0.2 * np.abs(np.sin(coords[:, 2]))

def color_func(coords):
    z_norm = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min())
    return np.column_stack((z_norm, np.zeros_like(z_norm), 1 - z_norm))

def opacity_func(coords):
    opacity = np.abs(np.sin(coords[:, 2]))
    return opacity

figure = Nematics3D.PlotFigure(name="test_figure")


# --- 案例 1: 固定参数 (类似于你给的例子) ---
tube1 = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=0),  
    name="solid_blue",
    color=(0,0,1), # 蓝色
    radius=0.3,
    sides=12,
)

# --- 案例 2: 函数驱动 (一次性传入函数) ---
tube2 = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=5),
    name="functional",
    color=color_func,     # 渐变色函数
    radius=radius_wave,   # 波动半径函数
    opacity=opacity_func,
)

# --- 案例 3: 手动模式 + 透明度 ---
tube3 = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=10),
    name="manual_color",
    color=np.random.rand(75, 3), # 随机色数组
    radius=0.15,
    opacity=1,
    shading_type='pbr',
    metallic=1,
    roughness=0.4    
)

tube4 = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=15),
    name="func_scalars",
    sides=20,
    radius=0.25,
    opacity=opacity_func,     
    scalars=radius_wave, 
    scalars_cmap='plasma'     
)

tube5 = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=20),
    name="manual_scalars2",
    sides=20,
    radius=0.25,
    opacity=1,     
    scalars=radius_wave(get_path(offset_y=20)),  
    shading_type='pbr',
    metallic=1,
    roughness=0.4,
    scalars_clim=(0,0.2),
    scalar_bar_title='test'
)
'''
tube1.opts.color = 'scalars'
tube1.opts.scalars = lambda x: radius_wave(x)+1
tube1._helper_resolver_spec('scalars')
tube1._helper_update_scalars()

tube2.opts.color = (1,0,0)
tube2.opts.opacity = 1
tube2._helper_resolver_spec('color')
tube2._helper_resolver_spec('opacity')
tube2._helper_update_rgba()

tube4.opts.color = (0,1,0)
tube4._helper_resolver_spec('color')
tube4._helper_update_rgba()
tube4._helper_switch_scalars_to_rgba()

tube5.opts.color = 'scalars'
tube5.opts.opacity = 1
tube5.opts.scalars = lambda x: np.sin(5*radius_wave(x))
tube5.opts.scalars_cmap = 'plasma'
tube5.opts.scalars_clim = None
tube5._helper_resolver_spec('scalars')
tube5._helper_resolver_spec('opacity')
tube5._helper_update_scalars()
tube5._entities.prop.interpolation = 'phong'
'''

'''
tube1.act_commit(scalars=lambda x: radius_wave(x)+1, opacity=0.5)
tube1.opts.opacity = opacity_func

tube2.opts.color = (1,0,0)
tube2.opts.shading_type = 'pbr'
tube2.opts.roughness = 0.2

tube2.act_commit(color=(0,1,0), opacity=0.8, sides=4, is_reset_camera=False)
'''
