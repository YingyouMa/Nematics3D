import numpy as np

import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

# -----------------------------
# 构造测试数据
# -----------------------------

N = 12

# 中心点：排成一条线
coords = np.zeros((N, 3))
coords[:, 0] = np.linspace(0, 10, N)

# 方向：在 xy 平面转一圈（每根 rod 不同方向）
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
orient = np.stack([
    np.cos(theta),
    np.sin(theta),
    np.zeros_like(theta)
], axis=1)

# -----------------------------
# 画图
# -----------------------------

figure = Nematics3D.PlotFigure()

opts = Nematics3D.OptsRod(
    length=1.5,          # 每根 rod 长度
    radius=0.15,         # 粗细
    color=(0.2, 0.6, 0.9),
    shading_type="phong",
    sides=16,
)

rod = Nematics3D.PlotRod(
    figure=figure,
    coords=coords,
    orient=orient,
    opts=opts,
)

