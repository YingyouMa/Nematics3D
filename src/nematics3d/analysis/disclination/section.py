"""Analysis of defect-centered polar Q-tensor sections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ...core.result_base import ResultBase
from ...misc import align_directors
from ...geometry import find_rotation_axis
from ...q_field.diagonalization import q_diagonalize


@dataclass(slots=True, frozen=True, repr=False)
class OmegaResult(ResultBase):
    """Inspectable result of an in-plane rotation-axis evaluation."""

    __result_name__: ClassVar[str] = "local omega"
    __field_docs__: ClassVar[dict[str, str]] = {
        "omega": "Estimated average in-plane rotation axis on the selected polar ring.",
        "metric": "Quality metrics and defect-domain flags returned by the omega fit.",
        "layer": "Polar ring layer index used for the omega evaluation.",
        "num_directors": "Number of sampled directors used on the selected ring.",
        "R": "Physical ring radius associated with the selected layer.",
        "opts": "Frozen copy of the polar-grid opts used to generate the section.",
    }

    omega: np.ndarray
    metric: dict[str, Any]
    layer: int
    num_directors: int
    R: float
    opts: Any


@dataclass(slots=True, frozen=True, repr=False)
class DefectSectionOmegaResult(OmegaResult):
    """Omega analysis result for one local section of a smoothed defect line."""

    __result_name__ = "defect-section omega"
    __field_docs__ = {
        **OmegaResult.__field_docs__,
        "beta": "Tilt angle in degrees between line tangent and omega axis.",
        "u_percent": "Normalized line parameter of the sampled section.",
        "position": "Wrapped real-space origin of the sampled section.",
        "tangent": "Unit tangent of the smoothed line at the sampled section.",
    }

    beta: float
    u_percent: float
    position: np.ndarray
    tangent: np.ndarray


def project_defect_radii_on_polar_plane(*, defect_positions, origin, normal, axis1):
    """Project physical defect positions into a polar plane and return radii."""
    if defect_positions is None or len(defect_positions) == 0:
        return np.array([], dtype=float)

    delta = np.asarray(defect_positions, dtype=float) - np.asarray(origin, dtype=float)
    axis1 = np.asarray(axis1, dtype=float)
    axis2 = np.cross(np.asarray(normal, dtype=float), axis1)
    return np.hypot(delta @ axis1, delta @ axis2)


def calc_polar_plane_omega(
    *,
    grid,
    interpolator,
    defect_positions_all,
    layer,
    logger=None,
) -> OmegaResult:
    """Estimate the average director rotation axis on one polar sampling ring."""
    ring_offsets = grid.calc_ring_offsets
    layer = int(layer)
    if layer < 0 or layer >= ring_offsets.shape[0] - 1:
        raise ValueError(
            f"`layer` must be between 0 and {ring_offsets.shape[0] - 2}, got {layer}."
        )

    s, e = ring_offsets[layer], ring_offsets[layer + 1]
    if (e - s) < 2:
        raise ValueError(
            f"Layer {layer} contains fewer than 2 directors and cannot define a rotation axis."
        )

    radius = float(grid.entity_polar[s, 0])
    q_layer, out_points = interpolator.interpolate(
        grid.entity_grid_all[s:e],
        is_out_warning=True,
    )
    directors = np.asarray(q_diagonalize(q_layer).n, dtype=float).copy()
    for i in range(1, len(directors)):
        directors[i] = align_directors(directors[i - 1], directors[i])

    rotation_axis = find_rotation_axis(directors)
    defect_radii = project_defect_radii_on_polar_plane(
        defect_positions=defect_positions_all,
        origin=grid.opts.origin,
        normal=grid.opts.normal,
        axis1=grid.opts.theta0_axis,
    )
    center_tol = max(1e-8, 1e-6 * max(1.0, radius))
    is_defect_center = defect_radii <= center_tol
    flags = {
        "is_out_of_domain": len(out_points) > 0,
        "is_defect_inside_R": bool(
            np.any((defect_radii <= radius) & ~is_defect_center)
        ),
        "is_defect_at_center": bool(np.any(is_defect_center)),
    }

    if logger is not None:
        if not flags["is_defect_at_center"]:
            logger.warning("No defect is detected at the center of this polar plane.")
        if flags["is_defect_inside_R"]:
            logger.warning(
                f"Defects are detected inside or on omega layer {layer} (R={radius})."
            )

    return OmegaResult(
        omega=rotation_axis.axis,
        metric={**rotation_axis.metric, **flags},
        layer=layer,
        num_directors=int(len(directors)),
        R=radius,
        opts=deepcopy(grid.opts),
    )


def calc_defect_section_omega(
    smooth, u_percent, *, opts_grid=None, opts_grid_defaults_override=None, **kwargs
) -> DefectSectionOmegaResult:
    """Evaluate omega and beta on a tangent-normal section of a smoothed line."""
    from ...grid import wrap_points_to_box
    from ...sample.plane_grid_polar import PlaneGridPolar
    from ...sample.q_plane import QPlanePolar

    opts_grid_keys = set() if opts_grid is None else set(opts_grid.act_asdict())
    opts_grid_default_keys = (
        set()
        if opts_grid_defaults_override is None
        else set(opts_grid_defaults_override)
    )
    pose_keys = {"origin", "normal"} & (
        set(kwargs) | opts_grid_keys | opts_grid_default_keys
    )
    pose_keys |= {"grid_offset", "grid_transform"} & set(kwargs)
    if pose_keys:
        raise ValueError(
            "calc_defect_section_omega resolves the section pose from u_percent; "
            "do not pass " + ", ".join(sorted(pose_keys)) + "."
        )

    q_host = getattr(getattr(smooth.owner, "registry", None), "owner", None)
    if q_host is None:
        raise RuntimeError("Cannot resolve the owning Q object for omega analysis.")
    tangent, origin = smooth.act_calc_tangent(u_percent, is_return_coord=True)
    origin = wrap_points_to_box(
        origin,
        smooth.owner.raw_box_size_periodic_index,
        transform=smooth.owner.raw_grid_transform,
        offset=smooth.owner.raw_grid_offset,
    )
    grid = PlaneGridPolar(
        normal=tangent,
        origin=origin,
        opts=opts_grid,
        opts_defaults_override=opts_grid_defaults_override,
        **kwargs,
    )
    q_plane = QPlanePolar(
        interpolator=q_host.act_add_interpolator(),
        grid=grid,
        name=f"omega plane of {smooth.name!r}",
    )
    layer = int(q_plane.grid.calc_ring_offsets.shape[0] - 2)
    if layer < 0:
        raise ValueError("The local polar section has no valid ring layer.")
    result = q_plane.act_calc_omega(layer)
    omega = np.asarray(result.omega, dtype=float)
    cos_beta = abs(float(np.dot(tangent, omega)))
    beta = (
        np.nan
        if not np.isfinite(cos_beta)
        else float(np.degrees(np.arccos(np.clip(cos_beta, -1.0, 1.0))))
    )
    return DefectSectionOmegaResult(
        omega=result.omega,
        metric=result.metric,
        layer=result.layer,
        num_directors=result.num_directors,
        R=result.R,
        opts=result.opts,
        beta=beta,
        u_percent=float(u_percent),
        position=origin,
        tangent=np.asarray(tangent, dtype=float),
    )


def sample_beta_from_smooth(
    u_percent: float,
    *,
    smooth,
    opts_grid=None,
    opts_grid_defaults_override=None,
    **grid_kwargs,
):
    """Line-function callback returning local defect-section omega/beta analysis."""
    return calc_defect_section_omega(
        smooth,
        u_percent,
        opts_grid=opts_grid,
        opts_grid_defaults_override=opts_grid_defaults_override,
        **grid_kwargs,
    )


__all__ = [
    "DefectSectionOmegaResult",
    "OmegaResult",
    "calc_defect_section_omega",
    "calc_polar_plane_omega",
    "project_defect_radii_on_polar_plane",
    "sample_beta_from_smooth",
]
