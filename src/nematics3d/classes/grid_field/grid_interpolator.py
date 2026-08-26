"""Generic interpolation helpers for fields living on a shared grid dataset."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nematics3d.datatypes import as_points
from nematics3d.grid import VALIDITY_FIELD_NAME, apply_linear_transform
from ..class_base import AttrDef, ClassBase
from ...logging_decorator import logging_and_warning_decorator

# A trilinear blend of a 0/1 mask equals 1.0 only when all eight supporting
# voxels are valid. This tolerance absorbs floating-point summation error so an
# all-valid support still counts as fully valid under the strict rule.
_VALIDITY_EPS = 1e-9


class GridInterpolator(ClassBase):
    """
    Generic interpolator for one `FieldData` object on a shared grid dataset.

    The owning field provides the numeric values to be sampled, while the
    owning dataset provides the grid transform, offset, and periodic-boundary
    rules used to interpret sample points.
    """

    # fmt: off
    __attr_defs__: ClassVar = {
        "entity_backend": AttrDef(
            doc="The scipy RegularGridInterpolator backend used to evaluate field values.",
            kind="entity",
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
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
        is_return_validity: bool = False,
        logger=None,
    ):
        """
        Interpolate field values at arbitrary sample points.

        The returned values are always the plain interpolated field values; no
        validity masking is applied to them, so existing callers are unaffected.

        Extra outputs are appended in a fixed order when requested:

        - If ``is_out_warning`` is True, also return the input points that lie
          outside non-periodic dimensions before clipping.
        - If ``is_return_validity`` is True, also return a boolean array marking
          which query points are physically valid. A point is valid only when
          all voxels supporting its trilinear interpolation are valid in the
          dataset validity mask and the point is inside the non-periodic domain.
          When the dataset carries no mask, every in-domain point is valid.
        """

        pts = as_points(points, name="interpolation query points", d=3)
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

        outputs = [result]
        if is_out_warning:
            outputs.append(out_points)
        if is_return_validity:
            outputs.append(self._helper_validity_at(pts, out_mask))

        if len(outputs) == 1:
            return result
        return tuple(outputs)

    def _helper_validity_at(
        self,
        pts_index: np.ndarray,
        out_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Return strict per-point validity for already-prepared index points.

        ``pts_index`` are lattice-index coordinates already wrapped/clipped onto
        the grid; ``out_mask`` flags points that were outside the non-periodic
        domain before clipping. A point is valid only when its trilinear support
        is fully valid in the dataset mask and it lies inside the domain.
        """
        dataset = self.owner.owner
        try:
            mask_field = dataset.fields[VALIDITY_FIELD_NAME]
        except KeyError:
            # No mask: every in-domain point is valid.
            return ~out_mask

        support = mask_field.act_add_interpolator().interpolate(
            pts_index,
            is_index=True,
        )
        return (np.asarray(support) >= 1.0 - _VALIDITY_EPS) & ~out_mask
