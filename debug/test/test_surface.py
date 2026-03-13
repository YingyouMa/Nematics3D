import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

import numpy as np
import pyvista as pv

# -----------------------------
# 1. 构造一个三角平面（在 3D 中）
# -----------------------------

# 三角形三个顶点（不共轴，定义一个平面）
A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 0.5])
C = np.array([0.2, 1.0, 0.2])

# 在三角形内随机采样点（重心坐标法，保证凸）
N = 300
u = np.random.rand(N, 1)
v = np.random.rand(N, 1)
mask = (u + v > 1.0)
u[mask] = 1.0 - u[mask]
v[mask] = 1.0 - v[mask]

coords = A + u * (B - A) + v * (C - A)   # (N, 3)

# -----------------------------
# 2. 给每个点一个颜色（示例：按高度 z）
# -----------------------------

z = coords[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min())

# RGB：蓝 -> 红
colors = np.column_stack([
    z_norm,
    np.zeros_like(z_norm),
    1.0 - z_norm
])


figure = Nematics3D.PlotFigure()
plane = Nematics3D.PlotSurface(coords, color=colors, figure=figure)
# plane = Nematics3D.PlotSurface(coords, color='scalars',
#                                 scalars= lambda x: np.linalg.norm(x, axis=-1),
#                                 figure=figure)
# plane = Nematics3D.PlotSurface(coords, color='scalars', scalars=np.sin,  figure=figure)
