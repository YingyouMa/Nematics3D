#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from nematics3d import q_diagonalize
from nematics3d.field import get_q

NX = NY = NZ = 128
S_VALUE = 1.0
DEFECT_SEPARATION = 48.0
MID_Z = 0.5 * (NZ - 1)

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "generated_defect_pair_qfield_no_noise_diag"
output_dir.mkdir(parents=True, exist_ok=True)


def main():
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
    S_input = np.full((NX, NY, NZ), S_VALUE, dtype=np.float64)
    Q = get_q(director, S=S_input)

    diagonalization = q_diagonalize(Q)
    S_diag, n_diag = diagonalization.S, diagonalization.n

    is_finite_S = np.isfinite(S_diag)
    is_finite_n = np.isfinite(n_diag).all(axis=-1)
    dot_abs = np.abs(np.einsum("...i,...i->...", director, n_diag))
    dot_abs = np.clip(dot_abs, -1.0, 1.0)
    angle_error_deg = np.degrees(np.arccos(dot_abs))
    angle_error_deg[~is_finite_n] = np.nan

    q_xx = Q[..., 0, 0]
    q_yy = Q[..., 1, 1]
    q_xy = Q[..., 0, 1]

    mask_axis_aligned_x = np.isclose(
        np.abs(director[..., 0]), 1.0, atol=1e-12
    ) & np.isclose(director[..., 1], 0.0, atol=1e-12)
    mask_director_z_zero = np.isclose(director[..., 2], 0.0, atol=1e-12)
    mask_q_axis_x_form = (
        np.isclose(q_xy, 0.0, atol=1e-12)
        & np.isclose(Q[..., 0, 2], 0.0, atol=1e-12)
        & np.isclose(Q[..., 1, 2], 0.0, atol=1e-12)
        & np.isclose(q_xx, 2.0 / 3.0, atol=1e-12)
        & np.isclose(q_yy, -1.0 / 3.0, atol=1e-12)
    )

    np.save(output_dir / "Q_full_128x128x128.npy", Q)
    np.save(output_dir / "director_input_128x128x128.npy", director)
    np.save(output_dir / "S_diag_128x128x128.npy", S_diag)
    np.save(output_dir / "director_diag_128x128x128.npy", n_diag)
    np.save(output_dir / "angle_error_deg_128x128x128.npy", angle_error_deg)

    print(f"Q shape = {Q.shape}")
    print(f"director shape = {director.shape}")
    print(f"S_diag shape = {S_diag.shape}")
    print(f"n_diag shape = {n_diag.shape}")
    print(f"Output dir = {output_dir}")
    print()
    print("Finite diagnostics:")
    print(f"  finite S points     = {np.count_nonzero(is_finite_S)} / {S_diag.size}")
    print(
        f"  finite n points     = {np.count_nonzero(is_finite_n)} / {is_finite_n.size}"
    )
    print(f"  non-finite S points = {np.count_nonzero(~is_finite_S)}")
    print(f"  non-finite n points = {np.count_nonzero(~is_finite_n)}")
    print()
    print("Director agreement diagnostics:")
    print(f"  min |n_in . n_out|       = {np.nanmin(dot_abs):.16f}")
    print(f"  max angle error (deg)    = {np.nanmax(angle_error_deg):.16e}")
    print(f"  mean angle error (deg)   = {np.nanmean(angle_error_deg):.16e}")
    print(
        f"  points with angle > 1e-8 deg = "
        f"{np.count_nonzero(np.nan_to_num(angle_error_deg) > 1e-8)}"
    )
    print()
    print("Special masks:")
    print(f"  axis-aligned x director points = {np.count_nonzero(mask_axis_aligned_x)}")
    print(
        f"  director z == 0 points         = {np.count_nonzero(mask_director_z_zero)}"
    )
    print(f"  exact x-axis Q-form points     = {np.count_nonzero(mask_q_axis_x_form)}")
    print()
    print("Representative points:")
    sample_points = [
        (0, 0, 0),
        (NX // 2, NY // 2, NZ // 2),
        (NX // 2, 0, NZ // 2),
        (0, NY // 2, NZ // 2),
    ]
    for idx in sample_points:
        ix, iy, iz = idx
        print(f"  idx = {idx}")
        print(f"    director_in  = {director[ix, iy, iz]}")
        print(f"    Q            =\n{Q[ix, iy, iz]}")
        print(f"    S_diag       = {S_diag[ix, iy, iz]}")
        print(f"    director_out = {n_diag[ix, iy, iz]}")
        print(f"    angle_err_deg= {angle_error_deg[ix, iy, iz]}")

    print()
    print(
        "Global variables available: "
        "x, y, z, theta_xy, director_xy, director, S_input, Q, S_diag, n_diag, "
        "dot_abs, angle_error_deg, mask_axis_aligned_x, mask_director_z_zero, "
        "mask_q_axis_x_form"
    )

    return {
        "x": x,
        "y": y,
        "z": z,
        "theta_xy": theta_xy,
        "director_xy": director_xy,
        "director": director,
        "S_input": S_input,
        "Q": Q,
        "S_diag": S_diag,
        "n_diag": n_diag,
        "dot_abs": dot_abs,
        "angle_error_deg": angle_error_deg,
        "mask_axis_aligned_x": mask_axis_aligned_x,
        "mask_director_z_zero": mask_director_z_zero,
        "mask_q_axis_x_form": mask_q_axis_x_form,
    }


GLOBALS = main()
globals().update(GLOBALS)
