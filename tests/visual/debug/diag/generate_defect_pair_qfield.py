#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import nematics3d as n3d


NX = NY = NZ = 128
S_VALUE = 1.0
DEFECT_SEPARATION = 48.0
MID_Z = 0.5 * (NZ - 1)
NOISE_AMPLITUDE = 1e-3
RNG_SEED = 7

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "generated_defect_pair_qfield"
output_dir.mkdir(parents=True, exist_ok=True)


x = np.arange(NX, dtype=np.float64) - 0.5 * (NX - 1)
y = np.arange(NY, dtype=np.float64) - 0.5 * (NY - 1)
z = np.arange(NZ, dtype=np.float64) - 0.5 * (NZ - 1)

x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

x_plus = 0.0
y_plus = -0.5 * DEFECT_SEPARATION
x_minus = 0.0
y_minus = 0.5 * DEFECT_SEPARATION

# +1/2 defect below and -1/2 defect above.
# The far field is along +x, and the +1/2 polarity points toward the -1/2 defect.
theta_xy = 0.5 * np.arctan2(y_grid - y_plus, x_grid - x_plus) - 0.5 * np.arctan2(
    y_grid - y_minus, x_grid - x_minus
)

director_xy = np.stack(
    [np.cos(theta_xy), np.sin(theta_xy), np.zeros_like(theta_xy)],
    axis=-1,
)
director = np.repeat(director_xy[:, :, None, :], NZ, axis=2)
S = np.full((NX, NY, NZ), S_VALUE, dtype=np.float64)
Q = n3d.getQ(director, S=S)

rng = np.random.default_rng(RNG_SEED)
noise_2d = rng.standard_normal((NX, NY, 5))
noise_2d /= np.max(np.abs(noise_2d))
noise_2d *= NOISE_AMPLITUDE
noise = np.repeat(noise_2d[:, :, None, :], NZ, axis=2)

Q[..., 0, 0] += noise[..., 0]
Q[..., 0, 1] += noise[..., 1]
Q[..., 1, 0] += noise[..., 1]
Q[..., 0, 2] += noise[..., 2]
Q[..., 2, 0] += noise[..., 2]
Q[..., 1, 1] += noise[..., 3]
Q[..., 1, 2] += noise[..., 4]
Q[..., 2, 1] += noise[..., 4]
Q[..., 2, 2] = -(Q[..., 0, 0] + Q[..., 1, 1])

input_q = n3d.classes.q_field_object.InputQ(
    Q=Q,
    box_periodic_flag=[True, True, True],
)
qobj = n3d.QFieldObject(
    is_detect_defects=False,
    is_classify_lines=False,
    inputValue=input_q,
    name="defect_pair_qfield",
)

np.save(output_dir / "Q_full_128x128x128.npy", Q)
np.save(output_dir / "director_128x128x128.npy", director)
Q[..., 0, 0].astype(np.float64).tofile(output_dir / "Qxx_0.dat")
Q[..., 0, 1].astype(np.float64).tofile(output_dir / "Qxy_0.dat")
Q[..., 0, 2].astype(np.float64).tofile(output_dir / "Qxz_0.dat")
Q[..., 1, 1].astype(np.float64).tofile(output_dir / "Qyy_0.dat")
Q[..., 1, 2].astype(np.float64).tofile(output_dir / "Qyz_0.dat")

figure = n3d.PlotFigure(
    is_off_screen=False,
    name="defect pair xy plane",
    opts=n3d.OptsFigure(size=(1600, 1000), bg_color=(1, 1, 1)),
)

if qobj.interpolator is None:
    qobj.act_add_interpolator()

plane = n3d.QPlane(
    qobj.interpolator,
    name="mid_z_xy_plane",
    opts=n3d.OptsPlaneGrid(
        normal=(0, 0, 1),
        axis1=(1, 0, 0),
        origin=(0.5 * (NX - 1), 0.5 * (NY - 1), MID_Z),
        spacing=4,
        size=NX - 1,
        alignment="center",
    ),
    bounds=qobj.calc_bounds,
)

plane.act_visualize_n(
    figure=figure,
    is_defect=False,
    opts_nb=n3d.OptsRod(
        length=2.8,
        radius=0.18,
        color=(0.08, 0.08, 0.08),
        opacity=1.0,
        ambient=0.25,
        diffuse=0.75,
        specular=0.05,
    ),
    opts_nd=n3d.OptsRod(
        length=2.8,
        radius=0.18,
        color=(0.85, 0.2, 0.2),
        opacity=1.0,
        ambient=0.25,
        diffuse=0.75,
        specular=0.05,
    ),
)

figure.act_view_xy()
figure.opts.focal_point = (0.5 * (NX - 1), 0.5 * (NY - 1), MID_Z)
figure.opts.distance = 170
figure.opts.roll = 0
figure.opts.azimuth = 0
figure.opts.elevation = 0
figure.act_commit()
figure.pl.add_text(
    "z-invariant +/-1/2 defect pair, far field along +x",
    position="upper_left",
    font_size=12,
    color="black",
)

print(f"Q shape = {Q.shape}")
print(f"director shape = {director.shape}")
print(f"Output dir = {output_dir}")
print(
    "Global variables available: x, y, z, theta_xy, director_xy, director, S, Q, input_q, qobj, figure, plane"
)
