from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping

import numpy as np
import pyvista as pv

from Nematics3D.datatypes import Number, UNSET, Unset, Vect, as_Number, as_Vect, as_str
from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.general import get_box_corners
from .host_base import HostBase, OptsBase


@dataclass(slots=True, repr=False)
class OptsBounds(OptsBase):
    origin: Vect(3) | Unset = UNSET
    axis1: Vect(3) | Unset = UNSET
    axis2: Vect(3) | None | Unset = UNSET

    length1: Number | Unset = UNSET
    length2: Number | None | Unset = UNSET
    length3: Number | None | Unset = UNSET

    alignment: Literal["min_corner", "center"] | Unset = UNSET

    __descriptions__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__descriptions__,
        "origin": (
            "The anchor point of the bounds box. "
            "Its geometric meaning is determined by ``alignment``."
        ),
        "axis1": (
            "The first direction of the bounds box. "
            "It is typically normalized by the host before use."
        ),
        "axis2": (
            "The second direction of the bounds box. "
            "The third direction is derived from the cross product of ``axis1`` and ``axis2``. "
            "If None, the host may infer a default orthogonal direction."
        ),
        "length1": "The box length along ``axis1``.",
        "length2": "The box length along ``axis2``.",
        "length3": (
            "The box length along the third direction derived from "
            "``axis1 x axis2``."
        ),
        "alignment": (
            "How ``origin`` is interpreted relative to the box. "
            'Typical values include ``"min_corner"`` and ``"center"``.'
        ),
    }

    _validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase._validators,
        "origin": lambda v, d: as_Vect(v, name=d, dim=3),
        "axis1": lambda v, d: as_Vect(v, name=d, dim=3, is_norm=True),
        "axis2": lambda v, d: None if v is None else as_Vect(v, name=d, dim=3, is_norm=True),
        "length1": lambda v, d: as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "length2": lambda v, d: None if v is None else as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "length3": lambda v, d: None if v is None else as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "alignment": lambda v, d: as_str(v, name=d, pool=("min_corner", "center")),
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsBase._DEFAULTS_FROZEN),
            "tag": "bounds options",
            "origin": (0.0, 0.0, 0.0),
            "axis1": (1.0, 0.0, 0.0),
            "axis2": None,
            "length2": None,
            "length3": None,
            "alignment": "min_corner",
        }
    )


class Bounds(HostBase):
    __descriptions__ = {
        **dict(HostBase.__descriptions__),
        "_entity_corners": "Corner coordinates of the bounds box in real space as an (8, 3) array.",
        "_entity_clip_geometry": "PyVista PolyData surface used for clipping other meshes inside this bounds.",
        "_calc_axis2": "Resolved second axis used by the bounds box.",
        "_calc_axis3": "Resolved third axis used by the bounds box.",
    }

    __slots__ = tuple(
        k for k, v in __descriptions__.items() if not v.startswith("Property:") and k not in HostBase.__slots__
    )

    def __init__(
        self,
        name: str | None = None,
        name_replace: str = "bounds",
        opts: OptsBounds | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            OptsBounds,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs,
        )

        object.__setattr__(self, "_entity_corners", None)
        object.__setattr__(self, "_entity_clip_geometry", None)
        object.__setattr__(self, "_calc_axis2", None)
        object.__setattr__(self, "_calc_axis3", None)

        for attr_name, value in {
            "length1": self.opts.length1,
        }.items():
            if value is UNSET:
                raise ValueError(f"Missing required variable {attr_name!r} to generate bounds")

        self.opts.act_finalize(defaults=self._opts_defaults)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, logger=None, **kwargs):
        if not is_reapply_opts and not kwargs:
            return

        with self.opts._helper_internal_update():
            for key, value in kwargs.items():
                setattr(self.opts, key, value)

        origin = self.opts.origin
        axis1 = self.opts.axis1
        axis2 = self.opts.axis2
        length1 = self.opts.length1
        length2 = length1 if self.opts.length2 is None else self.opts.length2
        length3 = length1 if self.opts.length3 is None else self.opts.length3
        alignment = self.opts.alignment

        if axis2 is not None:
            dot_product = axis1 @ axis2
            if not np.isclose(dot_product, 0, atol=1e-8):
                old_axis2 = axis2.copy()
                axis2 = axis2 - dot_product * axis1
                axis2 /= np.linalg.norm(axis2)
                logger.warning(
                    f"Invalid geometry: axis2 is not perpendicular to axis1 (dot product: {dot_product:.4e}). "
                    f"Projecting original axis2 {old_axis2} onto the plane normal to axis1 {axis1}. "
                    f"New orthonormal axis2: {axis2}."
                )
        else:
            from Nematics3D.general import rotation_matrix_from_vectors

            rotation_matrix = rotation_matrix_from_vectors((1, 0, 0), axis1)
            axis2 = rotation_matrix @ np.array([0.0, 1.0, 0.0])
            logger.debug(
                f"axis2 not provided. Automatically generated a reference axis2 {axis2} "
                f"from axis1 {axis1}."
            )

        axis3 = np.cross(axis1, axis2)
        corners_local = get_box_corners(length1, length2, length3)

        if alignment == "min_corner":
            origin_min_corner = origin
        elif alignment == "center":
            origin_min_corner = origin - 0.5 * (
                length1 * axis1 + length2 * axis2 + length3 * axis3
            )
        else:
            raise ValueError(f"Unsupported alignment {alignment!r}.")

        corners = (
            origin_min_corner
            + corners_local[:, [0]] * axis1
            + corners_local[:, [1]] * axis2
            + corners_local[:, [2]] * axis3
        )

        faces = np.hstack(
            [
                [4, 0, 1, 4, 2],
                [4, 3, 5, 7, 6],
                [4, 0, 1, 5, 3],
                [4, 2, 4, 7, 6],
                [4, 0, 2, 6, 3],
                [4, 1, 4, 7, 5],
            ]
        )
        clip_geometry = pv.PolyData(corners, faces).triangulate().clean()

        object.__setattr__(self, "_calc_axis2", axis2)
        object.__setattr__(self, "_calc_axis3", axis3)
        object.__setattr__(self, "_entity_corners", corners)
        object.__setattr__(self, "_entity_clip_geometry", clip_geometry)
