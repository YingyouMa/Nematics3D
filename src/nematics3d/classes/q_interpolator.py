"""Q-tensor interpolation helpers for sampled Q-field objects."""

from typing import Any, ClassVar, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nematics3d.field import apply_linear_transform
from ..logging_decorator import logging_and_warning_decorator
from .class_base import ClassBase


# QInterpolator wraps the scipy backend as a structured repository object so
# Q-field owners can bind it through the standard relation model.
#
# Subclasses should preserve the coupling between the owner relation, the
# periodic-boundary expansion rules, and the backend interpolator state.
class QInterpolator(ClassBase):
    """
    Interpolator object specialized for QFieldObject sampling.

    For normal users this object is usually created by `QFieldObject` and then
    accessed through `Q.interpolator`. It converts real-space query points into
    lattice-index coordinates and handles periodic-boundary interpolation when
    required by the owning Q field.
    """

    # fmt: off
    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this Q-field interpolator.",
        },
        "owner": {
            **dict(ClassBase.__attr_defs__["owner"]),
            "doc": "The QFieldObject whose field values are sampled by this interpolator.",
        },
        "entity_backend": {
            "doc": "The scipy RegularGridInterpolator backend used to evaluate Q values.",
            "kind": "entity",
        },
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in ClassBase.__slots__
    )

    def __init__(self, owner, name: str | None = None):
        if name is None:
            name = f"{owner.name} interpolator"
        super().__init__(name=name, name_replace="Q interpolator")
        self.act_bind_relation_base("owner", owner, is_weak=True)

        values = owner.raw_Q
        shape = np.shape(values)[:3]
        periodic = np.asarray(owner.raw_box_periodic_flag, dtype=bool)

        grid_axes = [np.arange(n, dtype=float) for n in shape]
        values_interp = values
        for dim, is_periodic in enumerate(periodic):
            if not is_periodic:
                continue
            grid_axes[dim] = np.arange(shape[dim] + 1, dtype=float)
            values_interp = np.concatenate(
                [values_interp, np.take(values_interp, [0], axis=dim)],
                axis=dim,
            )

        backend = RegularGridInterpolator(
            tuple(grid_axes),
            values_interp,
            method="linear",
            bounds_error=True,
        )
        object.__setattr__(self, "entity_backend", backend)

    @logging_and_warning_decorator()
    def interpolate(
        self,
        points: np.ndarray,
        is_index=False,
        is_out_warning=False,
        logger=None,
    ):
        """
        Interpolate Q values at query points.

        If `is_out_warning` is True, also return input points outside
        non-periodic dimensions and warn when any are found.
        """

        pts = np.asarray(points, dtype=float).copy()
        points_input = pts.copy()

        if not is_index:
            pts = apply_linear_transform(
                pts,
                transform=np.linalg.inv(self.owner.raw_grid_transform),
                offset=-self.owner.raw_grid_offset,
            )

        shape = np.shape(self.owner.raw_Q)[:3]
        periodic = np.asarray(self.owner.raw_box_periodic_flag, dtype=bool)

        out_mask = np.zeros(len(pts), dtype=bool)
        for d in range(3):
            if not periodic[d]:
                out_mask |= (pts[:, d] < 0) | (pts[:, d] > shape[d] - 1)
        out_points = points_input[out_mask]

        if is_out_warning and len(out_points) > 0:
            out_points_text = np.array2string(
                out_points,
                precision=6,
                separator=", ",
                suppress_small=False,
            )
            logger.warning(
                "Some interpolation query points are outside the non-periodic "
                "Q-field domain and will be clipped to the boundary.\n"
                f"Out-of-domain points ({len(out_points)}):\n{out_points_text}"
            )

        for d in range(3):
            if periodic[d]:
                pts[:, d] = np.mod(pts[:, d], shape[d])
            else:
                pts[:, d] = np.clip(pts[:, d], 0, shape[d] - 1)

        values = self.entity_backend(pts)
        if is_out_warning:
            return values, out_points
        return values
