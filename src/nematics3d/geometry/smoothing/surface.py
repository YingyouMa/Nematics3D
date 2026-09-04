"""PyVista-backed windowed-sinc surface smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from ...core.class_base import AttrDef
from ...core.host_base import HostBase, OptsBase
from ...core.opts import cover_value
from ...datatypes import (
    Number,
    UNSET,
    Unset,
    as_bool,
    as_number,
    as_readonly_array,
)
from ...logging_decorator import logging_and_warning_decorator
from ..polydata import as_polydata_input, copy_polydata_geometry


@dataclass(slots=True, repr=False)
class OptsSmoothedSurface(OptsBase):
    """Options passed to PyVista PolyData.smooth_taubin."""

    n_iter: int | Unset = UNSET
    pass_band: Number | Unset = UNSET
    boundary_smoothing: bool | Unset = UNSET
    feature_smoothing: bool | Unset = UNSET
    feature_angle: Number | Unset = UNSET
    edge_angle: Number | Unset = UNSET
    normalize_coordinates: bool | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "n_iter": (
            "non-negative degree of the polynomial approximation used by "
            "windowed-sinc smoothing"
        ),
        "pass_band": (
            "windowed-sinc pass-band in [0, 2]; lower values produce more smoothing"
        ),
        "boundary_smoothing": "whether boundary vertices may be smoothed",
        "feature_smoothing": (
            "whether sharp feature edges are identified and smoothed under "
            "feature constraints"
        ),
        "feature_angle": (
            "angle in degrees above which an interior edge is classified as a feature"
        ),
        "edge_angle": (
            "angle in degrees controlling smoothing along feature and boundary edges"
        ),
        "normalize_coordinates": (
            "whether coordinates are normalized before smoothing and restored afterward"
        ),
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase.impl_validators,
        "n_iter": lambda v, d: as_number(
            v,
            name=d,
            is_integer=True,
            value_range=(0, np.inf),
        ),
        "pass_band": lambda v, d: as_number(v, name=d, value_range=(0, 2)),
        "boundary_smoothing": lambda v, d: as_bool(v, name=d),
        "feature_smoothing": lambda v, d: as_bool(v, name=d),
        "feature_angle": lambda v, d: as_number(
            v,
            name=d,
            value_range=(0, 180),
        ),
        "edge_angle": lambda v, d: as_number(v, name=d, value_range=(0, 180)),
        "normalize_coordinates": lambda v, d: as_bool(v, name=d),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(getattr(OptsBase, "impl_defaults_frozen", {})),
            "tag": "smoothed surface options",
            "n_iter": 20,
            "pass_band": 0.1,
            "boundary_smoothing": True,
            "feature_smoothing": False,
            "feature_angle": 45.0,
            "edge_angle": 15.0,
            "normalize_coordinates": False,
        }
    )


class SurfaceSmoothingConfigError(ValueError):
    """Invalid configuration or geometry for surface smoothing."""


class SmoothedSurface(HostBase):
    """Smooth a surface through PyVista's windowed-sinc Taubin wrapper.

    The raw input is normalized to PyVista PolyData. Each smoothing pass starts
    from an independent geometry-only copy of that input, so repeated option
    commits are history-independent and never modify the caller's mesh.

    result is the smoothed geometry-only PolyData. vertices, error_vectors, and
    error_scalars expose read-only NumPy views. Displacement follows the explicit
    convention error_vectors = result.points - raw_surface.points.
    """

    __attr_defs__ = {
        "raw_surface": AttrDef(
            doc="Raw input surface normalized to pyvista.PolyData.",
            kind="raw",
            validator=lambda v, d: as_polydata_input(v, name=d),
            is_reapply_opts_after_raw=True,
        ),
        "impl_surface_result": AttrDef(
            doc="Internal storage for the current smoothed PolyData result.",
            kind="impl",
        ),
        "impl_error_scalars": AttrDef(
            doc="Internal read-only per-vertex displacement magnitudes.",
            kind="impl",
        ),
        "impl_error_vectors": AttrDef(
            doc="Internal read-only per-vertex displacement vectors.",
            kind="impl",
        ),
        "result": AttrDef(
            doc="Read-only: final smoothed geometry-only PolyData.",
            kind="property",
            is_public_settable=False,
        ),
        "vertices": AttrDef(
            doc="Read-only: final smoothed vertex coordinates.",
            kind="property",
            is_public_settable=False,
        ),
        "error_scalars": AttrDef(
            doc="Read-only: per-vertex smoothing displacement magnitudes.",
            kind="property",
            is_public_settable=False,
        ),
        "error_vectors": AttrDef(
            doc="Read-only: per-vertex smoothing displacement vectors.",
            kind="property",
            is_public_settable=False,
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
    )

    def __init__(
        self,
        surface,
        name: str | None = None,
        opts: OptsSmoothedSurface | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            OptsSmoothedSurface,
            opts,
            opts_defaults_override,
            name=name,
            name_replace="surface",
            raw_surface=surface,
            **kwargs,
        )
        self.opts.act_finalize(self.opts_defaults)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    def _helper_prepare_input(self):
        """Return a validated independent geometry-only PolyData input."""
        poly = copy_polydata_geometry(
            as_polydata_input(self.raw_surface, name="surface")
        )
        if poly.n_points == 0 or poly.n_cells == 0:
            raise SurfaceSmoothingConfigError(
                "Surface smoothing requires a non-empty PolyData with points and cells."
            )

        points = np.asarray(poly.points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise SurfaceSmoothingConfigError(
                "Surface point coordinates must have shape (n_points, 3)."
            )
        if not np.all(np.isfinite(points)):
            raise SurfaceSmoothingConfigError(
                "Surface point coordinates must contain only finite values."
            )
        return poly

    def _helper_run_smoothing(self) -> None:
        """Run PyVista smoothing from the raw geometry and store diagnostics."""
        poly = self._helper_prepare_input()
        points_initial = np.asarray(poly.points, dtype=float).copy()

        if self.opts.n_iter == 0:
            result = poly.copy(deep=True)
        else:
            result = poly.smooth_taubin(
                n_iter=self.opts.n_iter,
                pass_band=self.opts.pass_band,
                edge_angle=self.opts.edge_angle,
                feature_angle=self.opts.feature_angle,
                boundary_smoothing=self.opts.boundary_smoothing,
                feature_smoothing=self.opts.feature_smoothing,
                non_manifold_smoothing=False,
                normalize_coordinates=self.opts.normalize_coordinates,
                inplace=False,
                progress_bar=False,
            ).copy(deep=True)

        points_result = np.asarray(result.points, dtype=float)
        if points_result.shape != points_initial.shape:
            raise RuntimeError(
                "PyVista smooth_taubin changed the number of surface points."
            )
        if not np.all(np.isfinite(points_result)):
            raise RuntimeError(
                "PyVista smooth_taubin produced non-finite point coordinates."
            )

        error_vectors = as_readonly_array(
            points_result - points_initial,
            dtype=None,
            copy=False,
        )
        error_scalars = as_readonly_array(
            np.linalg.norm(error_vectors, axis=1),
            dtype=None,
            copy=False,
        )

        object.__setattr__(self, "impl_surface_result", result)
        object.__setattr__(self, "impl_error_vectors", error_vectors)
        object.__setattr__(self, "impl_error_scalars", error_scalars)

    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self,
        is_reapply_opts=False,
        logger=None,
        **kwargs,
    ):
        if not is_reapply_opts and not kwargs:
            return

        if kwargs:
            with self.opts.act_internal_update():
                cover_value(
                    self.opts,
                    is_allow_cover_target_set=True,
                    is_allow_unset_source=False,
                    **kwargs,
                )

        logger.debug(
            f"Smoothing surface {self.name!r} with PyVista smooth_taubin: "
            f"n_iter={self.opts.n_iter}, pass_band={self.opts.pass_band}, "
            f"boundary_smoothing={self.opts.boundary_smoothing}, "
            f"feature_smoothing={self.opts.feature_smoothing}, "
            f"feature_angle={self.opts.feature_angle}, "
            f"edge_angle={self.opts.edge_angle}, "
            f"normalize_coordinates={self.opts.normalize_coordinates}."
        )
        self._helper_run_smoothing()

    @property
    def result(self):
        return self.impl_surface_result

    @property
    def vertices(self):
        return as_readonly_array(
            np.asarray(self.impl_surface_result.points),
            dtype=None,
            copy=False,
        )

    @property
    def error_scalars(self):
        return self.impl_error_scalars

    @property
    def error_vectors(self):
        return self.impl_error_vectors


__all__ = [
    "OptsSmoothedSurface",
    "SmoothedSurface",
    "SurfaceSmoothingConfigError",
]
