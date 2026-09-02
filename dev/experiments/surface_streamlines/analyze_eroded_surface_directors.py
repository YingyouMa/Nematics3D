"""Sample, project, analyze, and save directors on the eroded example surface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

from director_projection import project_surface_directors


EXAMPLE_DIR = Path(r"E:\Research\AN\example")
INPUT_DIR = EXAMPLE_DIR / "qfield_input"
OUTPUT_DIR = EXAMPLE_DIR / "streamline" / "1"

DIRECTOR_PATH = INPUT_DIR / "n_qfield_xyz_weighted_gaussian_sigma12_xyz.npy"
BOUNDS_PATH = INPUT_DIR / "mask_xyz_largest_component_bounds.json"
METADATA_PATH = INPUT_DIR / "metadata_weighted_gaussian_sigma12_xyz.json"
SURFACE_PATH = OUTPUT_DIR / "mask_xyz_shrinked_level12.vtp"

POSITIONS_OUTPUT_PATH = OUTPUT_DIR / "surface_director_positions.npy"
SAMPLED_OUTPUT_PATH = OUTPUT_DIR / "surface_directors_sampled.npy"
PROJECTED_OUTPUT_PATH = OUTPUT_DIR / "surface_directors_projected.npy"
RESULT_OUTPUT_PATH = OUTPUT_DIR / "surface_director_projection.npz"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "surface_director_projection_summary.json"


def _physical_to_local_indices(
    positions: np.ndarray,
    transform: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Apply the inverse of ``points @ transform + offset``."""
    return np.linalg.solve(transform.T, (positions - offset).T).T


def _normalized(values: np.ndarray) -> np.ndarray:
    """Normalize vectors along their final axis, preserving exact zeros."""
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    result = np.zeros_like(values, dtype=float)
    np.divide(values, norms, out=result, where=norms > 1.0e-12)
    return result


def sample_directors_through_q(
    director_field: np.ndarray,
    index_positions: np.ndarray,
) -> np.ndarray:
    """Trilinearly interpolate Q from eight neighbors and diagonalize it."""
    shape = np.asarray(director_field.shape[:3], dtype=int)
    if np.any(index_positions < 0.0) or np.any(index_positions > shape - 1):
        invalid = np.flatnonzero(
            np.any(
                (index_positions < 0.0) | (index_positions > shape - 1),
                axis=1,
            )
        )
        raise ValueError(
            "Surface positions fall outside the cropped director field at "
            f"indices including {invalid[:10].tolist()}."
        )

    lower = np.floor(index_positions).astype(int)
    upper = np.minimum(lower + 1, shape - 1)
    fractions = index_positions - lower

    q5 = np.zeros((len(index_positions), 5), dtype=float)
    reference_directors = np.zeros((len(index_positions), 3), dtype=float)
    one_third = 1.0 / 3.0

    for corner_x in (0, 1):
        for corner_y in (0, 1):
            for corner_z in (0, 1):
                corner = np.array([corner_x, corner_y, corner_z])
                indices = np.where(corner, upper, lower)
                weights = np.prod(np.where(corner, fractions, 1.0 - fractions), axis=1)
                directors = _normalized(
                    np.asarray(
                        director_field[
                            indices[:, 0],
                            indices[:, 1],
                            indices[:, 2],
                        ],
                        dtype=float,
                    )
                )
                nx, ny, nz = directors.T
                q5[:, 0] += weights * (nx * nx - one_third)
                q5[:, 1] += weights * nx * ny
                q5[:, 2] += weights * nx * nz
                q5[:, 3] += weights * (ny * ny - one_third)
                q5[:, 4] += weights * ny * nz
                reference_directors += weights[:, None] * directors

    q9 = np.empty((len(q5), 3, 3), dtype=float)
    q9[:, 0, 0] = q5[:, 0]
    q9[:, 0, 1] = q9[:, 1, 0] = q5[:, 1]
    q9[:, 0, 2] = q9[:, 2, 0] = q5[:, 2]
    q9[:, 1, 1] = q5[:, 3]
    q9[:, 1, 2] = q9[:, 2, 1] = q5[:, 4]
    q9[:, 2, 2] = -q5[:, 0] - q5[:, 3]

    _, eigenvectors = np.linalg.eigh(q9)
    sampled_directors = eigenvectors[:, :, -1]

    reference_norms = np.linalg.norm(reference_directors, axis=1)
    alignment = np.einsum("ij,ij->i", sampled_directors, reference_directors)
    is_flip = (reference_norms > 1.0e-12) & (alignment < 0.0)
    sampled_directors[is_flip] *= -1.0
    return sampled_directors


def _tilt_statistics(tilt_angles: np.ndarray) -> dict[str, float | int]:
    """Return compact finite-angle statistics for the saved JSON summary."""
    finite = tilt_angles[np.isfinite(tilt_angles)]
    summary: dict[str, float | int] = {
        "count": int(tilt_angles.size),
        "finite_count": int(finite.size),
    }
    if finite.size == 0:
        return summary

    summary.update(
        {
            "minimum_degrees": float(np.min(finite)),
            "mean_degrees": float(np.mean(finite)),
            "median_degrees": float(np.median(finite)),
            "p90_degrees": float(np.percentile(finite, 90)),
            "p95_degrees": float(np.percentile(finite, 95)),
            "p99_degrees": float(np.percentile(finite, 99)),
            "maximum_degrees": float(np.max(finite)),
        }
    )
    for threshold in (5, 10, 15, 30, 45, 60):
        summary[f"count_at_most_{threshold}_degrees"] = int(
            np.count_nonzero(finite <= threshold)
        )
        summary[f"fraction_at_most_{threshold}_degrees"] = float(
            np.mean(finite <= threshold)
        )
    return summary


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    bounds = json.loads(BOUNDS_PATH.read_text(encoding="utf-8"))
    select_region = tuple(
        slice(bounds[axis]["start"], bounds[axis]["stop"]) for axis in ("x", "y", "z")
    )

    surface = pv.read(SURFACE_PATH)
    positions = np.asarray(surface.points, dtype=float)
    transform = np.asarray(metadata["grid_transform_for_QFieldObject"], dtype=float)
    offset = np.asarray(metadata["grid_offset_for_QFieldObject"], dtype=float)
    index_positions = _physical_to_local_indices(positions, transform, offset)

    full_director_field = np.load(DIRECTOR_PATH, mmap_mode="r")
    director_field = full_director_field[select_region]
    sampled_directors = sample_directors_through_q(
        director_field,
        index_positions,
    )
    projection = project_surface_directors(surface, sampled_directors)

    np.save(POSITIONS_OUTPUT_PATH, positions)
    np.save(SAMPLED_OUTPUT_PATH, sampled_directors)
    np.save(PROJECTED_OUTPUT_PATH, projection.projected_directors)
    np.savez_compressed(
        RESULT_OUTPUT_PATH,
        positions=positions,
        sampled_directors=sampled_directors,
        projected_directors=projection.projected_directors,
        surface_normals=projection.surface_normals,
        tilt_angles_degrees=projection.tilt_angles_degrees,
        normal_fractions=projection.normal_fractions,
        tangent_fractions=projection.tangent_fractions,
        is_projectable=projection.is_projectable,
        index_positions=index_positions,
    )

    summary = {
        "surface_path": str(SURFACE_PATH),
        "director_path": str(DIRECTOR_PATH),
        "surface_point_count": int(surface.n_points),
        "surface_cell_count": int(surface.n_cells),
        "projectable_count": int(np.count_nonzero(projection.is_projectable)),
        "unprojectable_count": int(
            projection.is_projectable.size - np.count_nonzero(projection.is_projectable)
        ),
        "tilt": _tilt_statistics(projection.tilt_angles_degrees),
        "outputs": {
            "positions": str(POSITIONS_OUTPUT_PATH),
            "sampled_directors": str(SAMPLED_OUTPUT_PATH),
            "projected_directors": str(PROJECTED_OUTPUT_PATH),
            "complete_projection": str(RESULT_OUTPUT_PATH),
        },
    }
    SUMMARY_OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
