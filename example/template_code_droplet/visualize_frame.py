"""Visualize one processed time-sequence frame.

Usage
-----
    from visualize_frame import visualize_frame

    # interactive window
    q_obj = visualize_frame(77)

    # save to file (off-screen)
    q_obj = visualize_frame(77, sigma=1.45, save_path="figures/frame_077")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage


# ── display constants ─────────────────────────────────────────────────────────
GRID_SPACING = 3
DIRECTOR_LENGTH = 2
DIRECTOR_RADIUS = 0.15

BETA_LINEFUNC_SAMPLES = 51
BETA_SCALARS_CMAP = "turbo"
BETA_SCALARS_CLIM = (0.0, 90.0)

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT_ROOT = Path(r"E:\Research\AN\time_sequence\data\npy")

import nematics3d
from nematics3d.grid import apply_linear_transform


# ── helpers ───────────────────────────────────────────────────────────────────

def _sigma_tag(sigma: float) -> str:
    sigma_str = str(int(sigma)) if sigma == int(sigma) else str(sigma).replace(".", "p")
    return f"sigma{sigma_str}um"


def _frame_dir(frame_number: int) -> Path:
    pattern = f"*Frame{int(frame_number):04d}_DT_diag"
    matches = sorted(p for p in INPUT_ROOT.glob(pattern) if p.is_dir())
    if not matches:
        raise FileNotFoundError(
            f"No processed frame directory matched {pattern!r} under {INPUT_ROOT}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple directories matched frame {frame_number}: "
            + ", ".join(p.name for p in matches)
        )
    return matches[0]


def _load_frame(frame_number: int, sigma: float) -> dict:
    tag = _sigma_tag(sigma)
    frame_dir = _frame_dir(frame_number)

    metadata = json.loads((frame_dir / f"metadata_weighted_gaussian_{tag}_xyz.json").read_text(encoding="utf-8"))
    n_values = np.load(frame_dir / f"n_qfield_xyz_weighted_gaussian_{tag}_xyz.npy").astype(np.float32)
    mask_values = np.load(frame_dir / "mask_xyz_largest_component.npy").astype(np.uint8)
    bounds = json.loads((frame_dir / "mask_xyz_largest_component_bounds.json").read_text(encoding="utf-8"))

    crop = (
        slice(bounds["x"]["start"], bounds["x"]["stop"]),
        slice(bounds["y"]["start"], bounds["y"]["stop"]),
        slice(bounds["z"]["start"], bounds["z"]["stop"]),
    )
    return {
        "frame_number": int(frame_number),
        "mask_values": mask_values[crop],
        "n_values": n_values[crop],
        "grid_transform": np.asarray(metadata["grid_transform_for_QFieldObject"], dtype=float),
        "grid_offset": np.asarray(metadata["grid_offset_for_QFieldObject"], dtype=float),
    }


def _fill_invalid_directors(n_values: np.ndarray, mask_values: np.ndarray) -> np.ndarray:
    """Fill mask-exterior voxels from nearest valid boundary director on each z slice."""
    filled = n_values.copy()
    valid_3d = mask_values > 0
    structure = np.ones((3, 3), dtype=bool)

    for iz in range(valid_3d.shape[2]):
        valid = valid_3d[:, :, iz]
        if np.all(valid):
            continue
        if not np.any(valid):
            raise ValueError(f"z slice {iz} has no valid voxels.")

        eroded = ndimage.binary_erosion(valid, structure=structure, border_value=0)
        boundary = valid & ~eroded
        if not np.any(boundary):
            boundary = valid

        _, nearest = ndimage.distance_transform_edt(~boundary, return_distances=True, return_indices=True)
        inv = np.argwhere(~valid)
        if len(inv) == 0:
            continue
        bx = nearest[0][inv[:, 0], inv[:, 1]]
        by = nearest[1][inv[:, 0], inv[:, 1]]
        filled[inv[:, 0], inv[:, 1], iz, :] = filled[bx, by, iz, :]

    return filled


def _mask_opacity(coords_physical, mask_values, grid_transform, grid_offset, base_opacity):
    coords_index = apply_linear_transform(
        np.asarray(coords_physical, dtype=float),
        transform=grid_transform,
        offset=grid_offset,
        is_inv=True,
    )
    ijk = np.rint(coords_index).astype(int)
    for axis, size in enumerate(mask_values.shape):
        ijk[:, axis] = np.clip(ijk[:, axis], 0, size - 1)
    return (mask_values[ijk[:, 0], ijk[:, 1], ijk[:, 2]] > 0).astype(np.float32) * base_opacity


def _build_scene(frame_number: int, sigma: float, *, is_off_screen: bool):
    data = _load_frame(frame_number, sigma)
    mask_values = data["mask_values"]
    n_values = _fill_invalid_directors(data["n_values"], mask_values)
    grid_transform = data["grid_transform"]
    grid_offset = data["grid_offset"]

    q_obj = nematics3d.QFieldObject(
        n=n_values,
        grid_transform=grid_transform,
        grid_offset=grid_offset,
        is_detect_defects=True,
        is_classify_lines=True,
        name=f"Frame {int(frame_number):04d}",
    )
    q_obj.act_lines_smooth()

    figure = nematics3d.PlotFigure(
        name=f"Frame {int(frame_number):04d}",
        is_off_screen=is_off_screen,
    )

    length_list = [getattr(q_obj.dataset.calc_bounds.opts, f"length{i}") for i in range(1, 4)]

    q_obj.act_visualize_n_plane(
        figure=figure,
        is_extent=False,
        grid_normal=(0, 0, 1),
        grid_origin=q_obj.dataset.calc_center,
        grid_size=max(length_list),
        grid_spacing=GRID_SPACING,
        n_length=DIRECTOR_LENGTH,
        n_radius=DIRECTOR_RADIUS,
        title=f"frame {int(frame_number):04d} director, {_sigma_tag(sigma)}",
    )

    plane = q_obj.objs["n-plane"]

    if getattr(plane.visual_nb, "opts", None) is not None and plane.visual_nb.opts.is_visible:
        base_opacity_nb = float(np.asarray(plane.visual_nb.calc_opacity)[0])
        plane.visual_nb.act_commit(
            opacity=lambda coords: _mask_opacity(coords, mask_values, grid_transform, grid_offset, base_opacity_nb),
            resolver_source_opacity="coords",
        )

    if getattr(plane.visual_nd, "opts", None) is not None and plane.visual_nd.opts.is_visible:
        base_opacity_nd = float(np.asarray(plane.visual_nd.calc_opacity)[0])
        plane.visual_nd.act_commit(
            opacity=lambda coords: _mask_opacity(coords, mask_values, grid_transform, grid_offset, base_opacity_nd),
            resolver_source_opacity="coords",
        )

    q_obj.act_visualize_disclination_lines(figure=figure, is_extent=False, line_radius=0.2)

    is_first_scalar_bar = True
    for line in q_obj.lines:
        if not line.smooths:
            continue
        smooth = line.smooths[-1]
        tube = smooth.visual_tube
        if tube is None:
            continue

        is_wrap = getattr(smooth.opts, "mode", None) == "wrap"
        u_samples = np.linspace(0.0, 100.0, int(BETA_LINEFUNC_SAMPLES), endpoint=not is_wrap)
        beta_linefunc = smooth.act_add_beta_interpolator(u_samples=u_samples)
        tube_u = np.linspace(0.0, 100.0, len(tube.raw_coords), endpoint=not is_wrap)
        beta_values = np.asarray(beta_linefunc(tube_u), dtype=np.float32)

        tube.act_commit(
            opacity=lambda coords: _mask_opacity(coords, mask_values, grid_transform, grid_offset, 1.0),
            paint_by="scalars",
            scalars=beta_values,
            scalars_cmap=BETA_SCALARS_CMAP,
            scalars_clim=BETA_SCALARS_CLIM,
            is_scalar_bar=is_first_scalar_bar,
            scalar_bar_title="beta",
        )
        if is_first_scalar_bar:
            figure.scalar_bars[0].act_commit(is_interactive=True)
            is_first_scalar_bar = False

    figure.pl.enable_depth_peeling()

    nematics3d.ContourSurfaceSet(
        mask_values,
        levels=0.5,
        name="mask_boundary",
        grid_transform=grid_transform,
        is_extract=True,
    ).act_plot_all(figure=figure, color=(1, 1, 1), opacity=0.1, ambient=0.35, diffuse=0.7, specular=0.1)

    figure.act_commit(
        azimuth=45,
        elevation=35.26,
        roll=0,
        distance=116,
        focal_point=[30.5946, 28.8153, 32.1477],
    )

    return q_obj, figure


# ── public API ────────────────────────────────────────────────────────────────

def visualize_frame(
    frame_number: int,
    sigma: float = 1.45,
    save_path: str | Path | None = None,
    *,
    is_overwrite: bool = True,
) -> nematics3d.QFieldObject:
    """Visualize one processed frame.

    Parameters
    ----------
    frame_number:
        Frame number, e.g. ``77``. Resolved to a directory like
        ``*Frame0077_DT_diag`` under INPUT_ROOT.
    sigma:
        Gaussian smoothing radius in µm used when the data was preprocessed.
        Must match the sigma used in ``process_mat_file``. Default 1.45.
    save_path:
        If None, open an interactive window. Otherwise render off-screen and
        save to this path (extension optional; figure decides format).
    is_overwrite:
        If True (default), overwrite an existing file at save_path. If False,
        raise FileExistsError when the file already exists.
    """
    q_obj, figure = _build_scene(frame_number, sigma, is_off_screen=(save_path is not None))

    if save_path is not None:
        save_path = Path(save_path).resolve()
        if not is_overwrite and save_path.exists():
            raise FileExistsError(
                f"Output already exists; pass is_overwrite=True to replace: {save_path}"
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.act_savefig(save_path)

    return q_obj


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize one time-sequence frame.")
    parser.add_argument("frame_number", type=int)
    parser.add_argument("--sigma", type=float, default=1.45)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--no-overwrite", dest="is_overwrite", action="store_false")
    a = parser.parse_args()
    visualize_frame(a.frame_number, sigma=a.sigma, save_path=a.save_path, is_overwrite=a.is_overwrite)
