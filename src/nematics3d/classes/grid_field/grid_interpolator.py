"""Generic interpolation helpers for fields living on a shared grid dataset."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nematics3d.field import apply_linear_transform
from ..class_base import ClassBase
from ...logging_decorator import logging_and_warning_decorator


class GridInterpolator(ClassBase):
    """
    Generic interpolator for one `FieldData` object on a shared grid dataset.

    The owning field provides the numeric values to be sampled, while the
    owning dataset provides the grid transform, offset, and periodic-boundary
    rules used to interpret sample points.
    """

    # fmt: off
    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this shared-grid interpolator.",
        },
        "entity_backend": {
            "doc": "The scipy RegularGridInterpolator backend used to evaluate field values.",
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
        super().__init__(name=name, name_replace="grid interpolator")
        self.act_bind_relation_base("owner", owner, is_weak=True)

        dataset = owner.owner
        if dataset is None:
            raise RuntimeError(
                "GridInterpolator requires a FieldData owner that is already "
                "bound to a GridFieldDataset."
            )

        values = owner.raw_values
        shape = np.shape(values)[:3]
        periodic = np.asarray(dataset.raw_box_periodic_flag, dtype=bool)

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
        is_index: bool = False,
        is_out_warning: bool = False,
        logger=None,
    ):
        """
        Interpolate field values at arbitrary sample points.

        If `is_out_warning` is True, also return the input points that lie
        outside non-periodic dimensions before clipping.
        """

        pts = np.asarray(points, dtype=float).copy()
        points_input = pts.copy()

        dataset = self.owner.owner
        values = self.owner.raw_values

        if not is_index:
            pts = apply_linear_transform(
                pts,
                transform=dataset.raw_grid_transform,
                offset=dataset.raw_grid_offset,
                is_inv=True,
            )

        shape = np.shape(values)[:3]
        periodic = np.asarray(dataset.raw_box_periodic_flag, dtype=bool)

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
                f"domain of field {self.owner.name!r} and will be clipped to the boundary.\n"
                f"Out-of-domain points ({len(out_points)}):\n{out_points_text}"
            )

        for d in range(3):
            if periodic[d]:
                pts[:, d] = np.mod(pts[:, d], shape[d])
            else:
                pts[:, d] = np.clip(pts[:, d], 0, shape[d] - 1)

        result = self.entity_backend(pts)
        if is_out_warning:
            return result, out_points
        return result
