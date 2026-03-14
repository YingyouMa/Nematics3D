import numpy as np
import pyvista as pv

# ---------------------------
# 1 生成一条穿过球的曲线
# ---------------------------
t = np.linspace(-10, 10, 500)

points = np.c_[
    1.6*np.sin(t),
    0.6*np.sin(2.5*t),
    0.4*np.cos(1.7*t)
]

line = pv.lines_from_points(points)

# ---------------------------
# 2 tube
# ---------------------------
tube = line.tube(radius=0.05)

# ---------------------------
# 3 球体
# ---------------------------
sphere = pv.Sphere(radius=1.0)

# ---------------------------
# 4 clip
# ---------------------------
tube_inside = tube.clip_surface(sphere, invert=False)
tube_outside = tube.clip_surface(sphere, invert=True)

# ---------------------------
# 5 绘制
# ---------------------------
pl = pv.Plotter()

pl.add_mesh(sphere, opacity=0.25, color="white")

pl.add_mesh(tube_outside, color="blue")
pl.add_mesh(tube_inside, color="red")

pl.show()