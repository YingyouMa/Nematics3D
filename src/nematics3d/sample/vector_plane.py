"""Vector-field sampling on physical-space plane grids."""

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from ..grid.field import GridInterpolator
from ..core.class_base import AttrDef
from .interpolate_plane import InterpolatePlane
from .plane_grid import OptsPlaneGrid, PlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar


class VectorPlane(InterpolatePlane):
    """Sample a 3-vector field on a Cartesian or polar physical-space plane."""

    __attr_defs__ = {
        "calc_magnitude_all": AttrDef(
            doc="Vector magnitudes on the complete sampling plane before Bounds selection.",
            kind="calc",
        ),
        "calc_magnitude": AttrDef(
            doc="Vector magnitudes on the currently selected plane points.",
            kind="calc",
        ),
        "state_is_interactable": AttrDef(
            doc="Whether plane controls may be opened from an attached visual.",
            kind="state",
        ),
        "default_visual_opts": AttrDef(
            doc="Default option overrides used by vector-plane visualization.",
            kind="default",
        ),
        "visual": AttrDef(
            doc="Attached PlotVector visual, if one has been created.",
            kind="relation",
            is_weak_by_default=False,
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in InterpolatePlane.__slots__
    )

    _origin_default_visual_opts = {
        "vector": {
            "resolver_source": "orient",
            "length": lambda orient: np.linalg.norm(orient, axis=1),
        }
    }

    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "vector-plane",
        grid: PlaneGrid | PlaneGridPolar | None = None,
        opts: OptsPlaneGrid | OptsPlaneGridPolar | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        visual_default: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        default_visual_opts = deepcopy(self._origin_default_visual_opts)
        visual_default = {} if visual_default is None else dict(visual_default)
        unexpected_visual_keys = set(visual_default) - {"vector"}
        if unexpected_visual_keys:
            raise ValueError(
                "`visual_default` must only contain the key 'vector'. "
                f"Got unexpected keys: {sorted(unexpected_visual_keys)!r}."
            )

        override = visual_default.get("vector", {})
        if not isinstance(override, Mapping):
            raise TypeError(
                "`visual_default['vector']` must be a mapping of option overrides."
            )
        default_visual_opts["vector"] |= dict(override)

        object.__setattr__(self, "default_visual_opts", default_visual_opts)
        object.__setattr__(self, "state_is_interactable", True)

        super().__init__(
            interpolator=interpolator,
            name=name,
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

    def act_refresh(self):
        """Re-sample vectors, validate the vector contract, and refresh derived data."""
        super().act_refresh()

        result_all = np.asarray(self.calc_result_all, dtype=float)
        if result_all.ndim != 2 or result_all.shape[1] != 3:
            raise ValueError(
                "VectorPlane requires interpolated values with shape (N, 3). "
                f"Got shape {result_all.shape} instead."
            )

        mask = self.grid.calc_box_mask
        object.__setattr__(self, "calc_result_all", result_all)
        object.__setattr__(self, "calc_result", result_all[mask])
        object.__setattr__(
            self, "calc_magnitude_all", np.linalg.norm(result_all, axis=1)
        )
        object.__setattr__(
            self,
            "calc_magnitude",
            self.calc_magnitude_all[mask],
        )

        from ..visual.vector_plane import update_vector_plane_visual

        update_vector_plane_visual(self)

    def act_visualize_vector(self, *args, **kwargs):
        """Create the vector glyph visual through the visualization layer."""
        from ..visual.vector_plane import visualize_vector_plane

        return visualize_vector_plane(self, *args, **kwargs)


__all__ = ["VectorPlane"]
