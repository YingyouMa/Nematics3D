"""Convert a single MAT director file into a smoothed director field.

Usage
-----
    from mat_to_smoothed_director import process_mat_file

    process_mat_file(
        mat_path=r"E:/data/sample.mat",
        output_dir=r"E:/data/processed",
        sigma=12.0,           # optional, default 12
    )

The function creates a subdirectory named after the MAT stem inside
``output_dir`` and writes:

- ``mask_xyz_largest_component.npy``
- ``mask_xyz_largest_component_bounds.json``
- ``n_qfield_xyz_weighted_gaussian_sigma1.45um_xyz.npy``  (name reflects sigma)
- ``metadata_weighted_gaussian_sigma1.45um_xyz.json``
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.ndimage import label as ndimage_label


# ── tuneable constants ────────────────────────────────────────────────────────
_TRUNCATE = None          # Gaussian truncation radius (None = scipy default)
_WEIGHTS_FLOOR = 1e-12    # floor weight to avoid divide-by-zero in smoothing
_DIAG_Z_CHUNK = 6         # z-slice chunk size during Q diagonalisation
_READ_Z_CHUNK = 8         # z-slice chunk size when reading the MAT file
_IMAG_WARN_TOL = 1e-3     # |imag| threshold that triggers a per-voxel warning
_IMAG_WARN_COUNT = 32     # number of offending voxels before we raise an error


# ── internal helpers ──────────────────────────────────────────────────────────

from nematics3d.classes.grid_field import GridFieldDataset, InputGridField
from nematics3d.field import Q_diagonalize, align_directors


def _iter_chunks(length: int, size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, length, size):
        yield start, min(start + size, length)


def _read_scalar_or_vector(dataset: h5py.Dataset) -> list[float]:
    values = np.asarray(dataset).reshape(-1, order="F")
    return [float(v) for v in values]


def _to_float32(array: np.ndarray, *, label: str) -> np.ndarray:
    """Cast to float32, handling MATLAB complex structured storage."""
    if array.dtype.fields is None:
        return np.asarray(array, dtype=np.float32)

    if {"real", "imag"}.issubset(set(array.dtype.fields)):
        real = np.asarray(array["real"], dtype=np.float32)
        imag = np.asarray(array["imag"], dtype=np.float32)
        n_bad = int(np.count_nonzero(np.abs(imag) > _IMAG_WARN_TOL))
        if n_bad >= _IMAG_WARN_COUNT:
            raise ValueError(
                f"{label} has {n_bad} voxels with |imag| > {_IMAG_WARN_TOL}; "
                "cannot safely drop the imaginary part."
            )
        if n_bad > 0:
            print(
                f"  Warning: {label} has {n_bad} voxels with "
                f"|imag| > {_IMAG_WARN_TOL}; using real part only. "
                f"Max |imag|: {float(np.max(np.abs(imag))):.6g}"
            )
        return real

    raise TypeError(f"Unsupported structured dtype for {label}: {array.dtype!r}")


def _extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a uint8 mask keeping only the largest 26-connected region."""
    valid = np.asarray(mask > 0, dtype=bool)
    if valid.ndim != 3:
        raise ValueError(f"Expected 3-D mask, got shape {valid.shape}.")

    labeled, _ = ndimage_label(valid, structure=np.ones((3, 3, 3), dtype=int))
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # exclude background
    largest_label = int(sizes.argmax())

    out = np.zeros_like(mask, dtype=np.uint8)
    out[labeled == largest_label] = 1
    return out


def _bounding_box(mask: np.ndarray) -> tuple[slice, ...]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("Mask has no valid voxels after component filtering.")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _bounds_to_dict(bounds: tuple[slice, ...]) -> dict:
    names = ("x", "y", "z")
    d = {n: {"start": s.start, "stop": s.stop} for n, s in zip(names, bounds)}
    d["slice_expression"] = (
        f"[{bounds[0].start}:{bounds[0].stop}, "
        f"{bounds[1].start}:{bounds[1].stop}, "
        f"{bounds[2].start}:{bounds[2].stop}]"
    )
    return d


def _q5_component(n: np.ndarray, i: int) -> np.ndarray:
    nx, ny, nz_ = (np.asarray(n[..., k], dtype=np.float32) for k in range(3))
    components = [
        nx * nx - np.float32(1 / 3),
        nx * ny,
        nx * nz_,
        ny * ny - np.float32(1 / 3),
        ny * nz_,
    ]
    return components[i]


def _build_smoothed_q5(
    n_raw: np.ndarray,
    weights: np.ndarray,
    dataset,
    tmp_path: Path,
    sigma: float,
) -> Path:
    q = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32, shape=n_raw.shape[:3] + (5,)
    )
    for i in range(5):
        q_raw = _q5_component(n_raw, i)
        q_s = dataset.act_gaussian_smooth(
            q_raw,
            sigma=(sigma, sigma, sigma),
            coord="physical",
            boundary="reflect",
            truncate=_TRUNCATE,
            weights=weights,
            weights_floor=_WEIGHTS_FLOOR,
            is_result=False,
        )
        q_s = np.asarray(q_s, dtype=np.float32)
        q_s[weights <= 0] = q_raw[weights <= 0]
        q[..., i] = q_s
        print(f"  Smoothed Q5 component {i + 1}/5.")
        del q_raw, q_s
        gc.collect()
    q.flush()
    del q
    gc.collect()
    return tmp_path


def _diagonalize(
    q_path: Path, n_raw: np.ndarray, mask: np.ndarray, out_path: Path,
    Q_diagonalize, align_directors,
) -> None:
    q = np.load(q_path, mmap_mode="r")
    n_out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=n_raw.shape
    )
    for z0, z1 in _iter_chunks(q.shape[2], _DIAG_Z_CHUNK):
        q_c = np.asarray(q[:, :, z0:z1, :], dtype=np.float32)
        n_c = np.asarray(n_raw[:, :, z0:z1, :], dtype=np.float32)
        m_c = np.asarray(mask[:, :, z0:z1] > 0, dtype=bool)
        _, n_d = Q_diagonalize(q_c)
        n_d = np.asarray(align_directors(n_c, n_d), dtype=np.float32)
        n_d[~m_c] = n_c[~m_c]
        n_out[:, :, z0:z1, :] = n_d
        print(f"  Diagonalised z [{z0}, {z1}).")
        del q_c, n_c, m_c, n_d
        gc.collect()
    n_out.flush()
    del q, n_out
    gc.collect()


def _read_mat(mat_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(mat_path, "r") as f:
        refs = f["v"][:, 0]
        vy, vx, vz, mask_h5 = (f[refs[k]] for k in range(4))

        shapes = {vy.shape, vx.shape, vz.shape, mask_h5.shape}
        if len(shapes) != 1:
            raise ValueError(
                f"Components and mask have inconsistent HDF5 shapes: "
                f"vy={vy.shape}, vx={vx.shape}, vz={vz.shape}, mask={mask_h5.shape}"
            )

        nz, nx, ny = vy.shape
        n_out = np.empty((nx, ny, nz, 3), dtype=np.float32)
        m_out = np.empty((nx, ny, nz), dtype=np.uint8)

        for z0, z1 in _iter_chunks(nz, _READ_Z_CHUNK):
            s = np.s_[:, :, z0:z1]
            m_block = np.transpose(np.asarray(mask_h5[z0:z1], dtype=np.uint8), (1, 2, 0))
            m_out[s] = m_block
            for comp, src in enumerate((vx, vy, vz)):
                block = _to_float32(src[z0:z1], label=f"component {comp} z[{z0}:{z1}]")
                n_out[s + (comp,)] = np.transpose(block, (1, 2, 0))
            print(f"  Read z {z0}:{z1} / {nz}")

        res_yxz = _read_scalar_or_vector(f["f_config"]["res"])
        res_xyz = [res_yxz[1], res_yxz[0], res_yxz[2]]

    meta = {
        "source_mat_path": str(mat_path),
        "source_h5_shape_zyx": [int(nz), int(nx), int(ny)],
        "qfield_shape_xyz": [int(nx), int(ny), int(nz)],
        "director_components_source_order": ["y", "x", "z"],
        "director_components_output_order": ["x", "y", "z"],
        "spatial_axes_source_h5": ["z", "x", "y"],
        "spatial_axes_output": ["x", "y", "z"],
        "voxel_size_source_yxz_um": res_yxz,
        "voxel_size_output_xyz_um": res_xyz,
        "grid_transform_for_QFieldObject": np.diag(res_xyz).tolist(),
        "grid_offset_for_QFieldObject": [0.0, 0.0, 0.0],
        "mat_complex_handling": {
            "rule": "use real part if MATLAB complex storage detected",
            "imag_warn_tolerance": _IMAG_WARN_TOL,
            "imag_warn_count_threshold": _IMAG_WARN_COUNT,
        },
    }
    return n_out, m_out, meta


# ── public API ────────────────────────────────────────────────────────────────

def process_mat_file(
    mat_path: str | Path,
    output_dir: str | Path,
    sigma: float = 1.45,
    *,
    is_overwrite: bool = True,
) -> dict[str, str]:
    """Convert a MAT director file into a smoothed director field.

    Parameters
    ----------
    mat_path:
        Path to the source MATLAB v7.3 / HDF5 MAT file.
    output_dir:
        Parent directory.  A subdirectory named after the MAT stem is created
        there and all outputs are written into it.
    sigma:
        Isotropic Gaussian smoothing radius in physical units (µm, default 1.45).
        Applied with ``coord="physical"`` so each axis is automatically scaled
        by its voxel size — gives equal smoothing in physical space.
    is_overwrite:
        If True (default), overwrite existing output files. If False, raise
        FileExistsError when outputs already exist.

    Returns
    -------
    dict with keys ``mat``, ``output_dir``, ``mask``, ``bounds``,
    ``director``, ``metadata``.
    """
    mat_path = Path(mat_path).resolve()
    output_dir = Path(output_dir).resolve() / mat_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma_str = str(int(sigma)) if sigma == int(sigma) else str(sigma).replace(".", "p")
    sigma_tag = f"sigma{sigma_str}um"

    mask_path = output_dir / "mask_xyz_largest_component.npy"
    bounds_path = output_dir / "mask_xyz_largest_component_bounds.json"
    director_path = output_dir / f"n_qfield_xyz_weighted_gaussian_{sigma_tag}_xyz.npy"
    meta_path = output_dir / f"metadata_weighted_gaussian_{sigma_tag}_xyz.json"
    tmp_q_path = output_dir / f"_tmp_qfield5_weighted_gaussian_{sigma_tag}_xyz.npy"

    if not is_overwrite:
        existing = [p for p in (mask_path, bounds_path, director_path, meta_path) if p.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist; pass is_overwrite=True to replace: "
                + ", ".join(str(p) for p in existing)
            )

    print(f"Reading {mat_path} ...")
    n_raw, mask_raw, meta = _read_mat(mat_path)

    print("Extracting largest connected component ...")
    mask_large = _extract_largest_component(mask_raw)
    bounds = _bounding_box(mask_large)
    np.save(mask_path, mask_large)
    bounds_path.write_text(
        json.dumps(_bounds_to_dict(bounds), indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Mask saved → {mask_path}")

    dataset = GridFieldDataset(
        inputValue=InputGridField(
            shape=n_raw.shape[:3],
            grid_transform=np.asarray(meta["grid_transform_for_QFieldObject"], dtype=float),
            grid_offset=np.asarray(meta["grid_offset_for_QFieldObject"], dtype=float),
        ),
        name=f"{mat_path.stem} weighted Gaussian sigma={sigma}",
    )

    try:
        print(f"Smoothing Q5 field (sigma={sigma}) ...")
        weights = np.asarray(mask_large, dtype=np.float32)
        _build_smoothed_q5(n_raw, weights, dataset, tmp_q_path, sigma)

        print("Diagonalising ...")
        _diagonalize(tmp_q_path, n_raw, mask_large, director_path, Q_diagonalize, align_directors)
    finally:
        if tmp_q_path.exists():
            try:
                tmp_q_path.unlink()
            except OSError:
                print(f"  Warning: temp file still locked by Windows: {tmp_q_path}")

    meta_out = {
        **meta,
        "generated_director_path": str(director_path),
        "generated_mask_path": str(mask_path),
        "generated_bounds_path": str(bounds_path),
        "generation_note": (
            "Director field after axis/component reordering, largest-26-connected "
            "mask extraction, weighted Gaussian smoothing in Q5 form, Q "
            "diagonalisation, and director-sign alignment."
        ),
        "largest_component": {
            "connectivity": 26,
            "bounds_xyz": _bounds_to_dict(bounds),
            "input_valid_voxels": int(np.count_nonzero(mask_raw)),
            "largest_component_valid_voxels": int(np.count_nonzero(mask_large)),
        },
        "gaussian_smoothing": {
            "coord": "physical",
            "sigma_um": float(sigma),
            "sigma_xyz": [float(sigma)] * 3,
            "truncate": _TRUNCATE,
            "boundary": "reflect",
            "weights": "largest_component_mask > 0",
            "weights_floor": _WEIGHTS_FLOOR,
        },
        "diagonalisation": {"z_chunk_size": _DIAG_Z_CHUNK},
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Director saved → {director_path}")
    print(f"  Metadata saved → {meta_path}")

    return {
        "mat": str(mat_path),
        "output_dir": str(output_dir),
        "mask": str(mask_path),
        "bounds": str(bounds_path),
        "director": str(director_path),
        "metadata": str(meta_path),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a MAT director file into a smoothed director field."
    )
    parser.add_argument("mat_path", type=Path, help="Source MAT file.")
    parser.add_argument("output_dir", type=Path, help="Parent output directory.")
    parser.add_argument("--sigma", type=float, default=1.45, help="Gaussian sigma in um.")
    parser.add_argument("--no-overwrite", dest="is_overwrite", action="store_false")
    a = parser.parse_args()
    result = process_mat_file(a.mat_path, a.output_dir, a.sigma, is_overwrite=a.is_overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))
