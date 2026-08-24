"""Scalar-bar object model for figure-level scalar legend management."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import datetime
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from nematics3d.datatypes import (
    UNSET,
    Unset,
    as_ColorRGB,
    as_Number,
    as_vector,
    as_bool,
    as_str,
)

from ..class_base import AttrDef
from ..host_base import HostBase, OptsBase


def _as_optional_str(value, desc, *, pool=None):
    if value is None:
        return None
    return as_str(value, name=desc, pool=pool)


def _as_optional_int(value, desc, *, value_range=None):
    if value is None:
        return None
    return as_Number(
        value,
        name=desc,
        is_int=True,
        value_range=value_range,
        bounded=True,
    )


def _as_optional_float(value, desc, *, value_range=None):
    if value is None:
        return None
    return as_Number(
        value,
        name=desc,
        value_range=value_range,
        bounded=True,
    )


def _as_optional_bool(value, desc):
    if value is None:
        return None
    return as_bool(value, name=desc)


def _as_optional_color(value, desc):
    if value is None:
        return None
    color = as_ColorRGB(value, name=desc)
    return tuple(float(x) for x in color)


def _as_optional_position(value, desc):
    if value is None:
        return None
        position = tuple(float(x) for x in as_vector(value, d=2, name=desc))
    if any((x < 0.0) or (x > 1.0) for x in position):
        raise ValueError(f"{desc} must lie inside the normalized viewport [0, 1].")
    return position


@dataclass(slots=True, repr=False)
class OptsScalarBar(OptsBase):
    """Configuration object for one figure-level scalar bar."""

    is_visible: bool | Unset = UNSET
    is_vertical: bool | Unset = UNSET
    width: float | None | Unset = UNSET
    height: float | None | Unset = UNSET
    position: tuple[float, float] | None | Unset = UNSET
    n_labels: int | None | Unset = UNSET
    fmt: str | None | Unset = UNSET
    n_colors: int | None | Unset = UNSET
    font_family: str | None | Unset = UNSET
    title_font_size: int | None | Unset = UNSET
    label_font_size: int | None | Unset = UNSET
    color: tuple[float, float, float] | None | Unset = UNSET
    background_color: tuple[float, float, float] | None | Unset = UNSET
    is_fill: bool | None | Unset = UNSET
    is_outline: bool | None | Unset = UNSET
    is_bold: bool | None | Unset = UNSET
    is_italic: bool | None | Unset = UNSET
    is_shadow: bool | None | Unset = UNSET
    is_interactive: bool | None | Unset = UNSET
    is_use_opacity: bool | None | Unset = UNSET
    is_unconstrained_font_size: bool | None | Unset = UNSET
    is_nan_annotation: bool | None | Unset = UNSET
    below_label: str | None | Unset = UNSET
    above_label: str | None | Unset = UNSET
    # fmt: off
    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsBase.__attrs__),
        "is_visible":              "Whether this scalar bar should currently be shown.",
        "is_vertical":             "Whether the scalar bar should use vertical orientation.",
        "width":                   "Normalized scalar-bar width in the figure viewport.",
        "height":                  "Normalized scalar-bar height in the figure viewport.",
        "position":                "Normalized (x, y) scalar-bar anchor position in the figure viewport.",
        "n_labels":                "Number of numeric labels shown on the scalar bar.",
        "fmt":                     "Numeric label format string used by the scalar bar.",
        "n_colors":                "Number of discrete colors displayed in the scalar bar.",
        "font_family":             "Font family used for scalar-bar text.",
        "title_font_size":         "Font size of the scalar-bar title.",
        "label_font_size":         "Font size of the scalar-bar numeric labels.",
        "color":                   "Text and outline color used by the scalar bar.",
        "background_color":        "Background fill color of the scalar-bar box.",
        "is_fill":                 "Whether the scalar-bar background box should be filled.",
        "is_outline":              "Whether the scalar bar should draw an outline frame.",
        "is_bold":                 "Whether scalar-bar text should use a bold font weight.",
        "is_italic":               "Whether scalar-bar text should use italics.",
        "is_shadow":               "Whether scalar-bar text should use drop shadows.",
        "is_interactive":          "Whether the scalar bar should be interactive/draggable when supported.",
        "is_use_opacity":          "Whether to display opacity information in the scalar bar.",
        "is_unconstrained_font_size": (
            "Whether font size may exceed the scalar-bar box constraints."
        ),
        "is_nan_annotation":       "Whether NaN values should be annotated in the scalar bar.",
        "below_label":             "Label shown for values below the displayed range.",
        "above_label":             "Label shown for values above the displayed range.",
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsBase.impl_validators),
        "is_visible":              lambda v, d: as_bool(v, name=d),
        "is_vertical":             lambda v, d: as_bool(v, name=d),
        "width":                   lambda v, d: _as_optional_float(v, d, value_range=(0.0, 1.0)),
        "height":                  lambda v, d: _as_optional_float(v, d, value_range=(0.0, 1.0)),
        "position":                lambda v, d: _as_optional_position(v, d),
        "n_labels":                lambda v, d: _as_optional_int(v, d, value_range=(0, np.inf)),
        "fmt":                     lambda v, d: _as_optional_str(v, d),
        "n_colors":                lambda v, d: _as_optional_int(v, d, value_range=(1, np.inf)),
        "font_family":             lambda v, d: _as_optional_str(v, d, pool=("arial", "courier", "times")),
        "title_font_size":         lambda v, d: _as_optional_int(v, d, value_range=(1, np.inf)),
        "label_font_size":         lambda v, d: _as_optional_int(v, d, value_range=(1, np.inf)),
        "color":                   lambda v, d: _as_optional_color(v, d),
        "background_color":        lambda v, d: _as_optional_color(v, d),
        "is_fill":                 lambda v, d: _as_optional_bool(v, d),
        "is_outline":              lambda v, d: _as_optional_bool(v, d),
        "is_bold":                 lambda v, d: _as_optional_bool(v, d),
        "is_italic":               lambda v, d: _as_optional_bool(v, d),
        "is_shadow":               lambda v, d: _as_optional_bool(v, d),
        "is_interactive":          lambda v, d: _as_optional_bool(v, d),
        "is_use_opacity":          lambda v, d: _as_optional_bool(v, d),
        "is_unconstrained_font_size": lambda v, d: _as_optional_bool(v, d),
        "is_nan_annotation":       lambda v, d: _as_optional_bool(v, d),
        "below_label":             lambda v, d: _as_optional_str(v, d),
        "above_label":             lambda v, d: _as_optional_str(v, d),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "is_visible": True,
            "is_vertical": True,
            "width": None,
            "height": None,
            "position": None,
            "n_labels": 5,
            "fmt": None,
            "n_colors": None,
            "font_family": None,
            "title_font_size": None,
            "label_font_size": None,
            "color": None,
            "background_color": None,
            "is_fill": None,
            "is_outline": None,
            "is_bold": None,
            "is_italic": None,
            "is_shadow": None,
            "is_interactive": None,
            "is_use_opacity": None,
            "is_unconstrained_font_size": None,
            "is_nan_annotation": None,
            "below_label": None,
            "above_label": None,
        }
    )
    # fmt: on


# ScalarBar developer conventions:
# - ScalarBar is a figure-owned HostBase object whose independent user inputs
#   live in OptsScalarBar and whose derived backend-facing state lives in
#   calc_/entity_ fields on the host.
# - Keep `owner` bound to the owning PlotFigure and `source` bound to the
#   glyph-like object that currently provides mapper semantics for this bar.
# - Treat `raw_name` as the registry identity and `impl_name_pv` as the stable
#   backend-facing PyVista key. User-facing scalar metadata such as title,
#   cmap, and clim are source-derived rather than scalar-bar-owned inputs.
# - `calc_pyvista_kwargs` should remain the resolved backend kwargs payload
#   derived from the supported explicit opts, while `entity_backend` stores the
#   live backend actor/handle when one exists.


class ScalarBar(HostBase):
    """Host-style declaration object for one figure-level scalar bar."""

    __attr_defs__ = {
        "source": AttrDef(
            doc="The glyph-like object that currently provides the scalar-bar mapper semantics.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "raw_mapper_name": AttrDef(
            doc="Optional readable name of the mapper/source associated with this scalar bar.",
            kind="raw",
            validator=lambda v, d: _as_optional_str(v, d),
        ),
        "impl_name_pv": AttrDef(
            doc="Stable unique identifier of this scalar bar stored in the PyVista plotter.",
            kind="impl",
        ),
        "calc_pyvista_kwargs": AttrDef(
            doc="Resolved PyVista scalar-bar keyword arguments derived from the current opts.",
            kind="calc",
        ),
        "entity_backend": AttrDef(
            doc="Live backend scalar-bar handle managed by the rendering layer, or None if not created.",
            kind="entity",
        ),
        "entity_backend_widget": AttrDef(
            doc="Live backend scalar-bar widget handle when interactive mode is enabled.",
            kind="entity",
        ),
        "impl_backend_widget_observer_tag": AttrDef(
            doc="Internal observer id attached to the interactive widget end-event.",
            kind="impl",
        ),
        "impl_backend_rebuild_signature": AttrDef(
            doc="Internal signature of backend settings that require backend recreation when changed.",
            kind="impl",
        ),
        "impl_is_syncing_backend": AttrDef(
            doc="Internal guard preventing recursive opts/backend synchronization loops.",
            kind="impl",
        ),
        "backend": AttrDef(
            doc="Read-only: Alias of `entity_backend`.",
            kind="property",
        ),
        "backend_widget": AttrDef(
            doc="Read-only: Alias of `entity_backend_widget`.",
            kind="property",
        ),
    }

    __slots__ = (
        "raw_mapper_name",
        "impl_name_pv",
        "calc_pyvista_kwargs",
        "entity_backend",
        "entity_backend_widget",
        "impl_backend_widget_observer_tag",
        "impl_backend_rebuild_signature",
        "impl_is_syncing_backend",
    )

    _PYVISTA_COMMON_OPT_KEYS = (
        "is_vertical",
        "n_labels",
        "is_italic",
        "is_bold",
        "is_shadow",
        "fmt",
        "font_family",
        "label_font_size",
        "title_font_size",
        "color",
        "position",
        "width",
        "height",
        "n_colors",
        "background_color",
        "is_fill",
        "is_outline",
        "is_interactive",
        "is_use_opacity",
        "is_unconstrained_font_size",
        "is_nan_annotation",
        "below_label",
        "above_label",
    )

    _PYVISTA_BOOL_KEY_MAP = {
        "is_vertical": "vertical",
        "is_fill": "fill",
        "is_outline": "outline",
        "is_bold": "bold",
        "is_italic": "italic",
        "is_shadow": "shadow",
        "is_interactive": "interactive",
        "is_use_opacity": "use_opacity",
        "is_unconstrained_font_size": "unconstrained_font_size",
        "is_nan_annotation": "nan_annotation",
    }

    def __init__(
        self,
        *,
        name: str | None = None,
        mapper_name: str | None = None,
        opts: OptsScalarBar | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        backend=None,
        **kwargs,
    ):
        mapper_name_validated = _as_optional_str(
            mapper_name,
            "Optional readable name of the mapper/source associated with this scalar bar.",
        )
        name_replace = "scalar_bar"
        if mapper_name_validated is not None:
            name_replace = f"{mapper_name_validated}_scalarbar"

        super().__init__(
            OptsScalarBar,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs,
        )
        object.__setattr__(
            self,
            "raw_mapper_name",
            mapper_name_validated,
        )
        str_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        unique_id = self.name + str_now
        object.__setattr__(self, "impl_name_pv", unique_id)
        object.__setattr__(self, "entity_backend", backend)
        object.__setattr__(self, "entity_backend_widget", None)
        object.__setattr__(self, "impl_backend_widget_observer_tag", None)
        object.__setattr__(self, "impl_backend_rebuild_signature", None)
        object.__setattr__(self, "impl_is_syncing_backend", False)
        object.__setattr__(self, "calc_pyvista_kwargs", {})
        self.opts.act_finalize(self.opts_defaults)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    @property
    def backend(self):
        """Return the live backend handle currently attached to this scalar bar."""
        return self.entity_backend

    @property
    def backend_widget(self):
        """Return the live backend widget handle when interactive mode is enabled."""
        return self.entity_backend_widget

    def _helper_set_backend(self, backend):
        """Attach or replace the live backend handle for this scalar bar."""
        object.__setattr__(self, "entity_backend", backend)
        return backend

    def _helper_set_backend_widget(self, widget):
        """Attach or replace the live backend widget handle for this scalar bar."""
        object.__setattr__(self, "entity_backend_widget", widget)
        return widget

    def _helper_set_backend_widget_observer_tag(self, tag):
        """Attach or replace the widget end-interaction observer id."""
        object.__setattr__(self, "impl_backend_widget_observer_tag", tag)
        return tag

    def _helper_set_backend_rebuild_signature(self, signature):
        """Store the applied backend-rebuild signature for future sync checks."""
        object.__setattr__(self, "impl_backend_rebuild_signature", signature)
        return signature

    def _helper_clear_backend(self):
        """Forget the live backend handle without changing the declarative state."""
        object.__setattr__(self, "entity_backend", None)
        object.__setattr__(self, "entity_backend_widget", None)
        object.__setattr__(self, "impl_backend_widget_observer_tag", None)
        object.__setattr__(self, "impl_backend_rebuild_signature", None)

    def _helper_set_backend_sync_guard(self, is_syncing):
        """Set the internal backend-sync guard used to avoid recursive updates."""
        object.__setattr__(self, "impl_is_syncing_backend", bool(is_syncing))

    def _helper_request_owner_sync(self):
        """Request owner-side backend synchronization after opts/state changes."""
        if self.impl_is_syncing_backend:
            return

        registry = getattr(self, "registry", None)
        act_sync = getattr(registry, "act_sync", None)
        if callable(act_sync):
            act_sync(self.name)
            return

        owner = getattr(self, "owner", None)
        if owner is None:
            return

        act_sync_scalar_bar = getattr(owner, "act_sync_scalar_bar", None)
        if callable(act_sync_scalar_bar):
            act_sync_scalar_bar(self.name)

    def _helper_build_pyvista_kwargs(self) -> dict[str, Any]:
        """Build the common PyVista keyword payload from current opts values."""
        kwargs_resolved: dict[str, Any] = {}
        opts_dict = self.opts.act_asdict()

        for key in self._PYVISTA_COMMON_OPT_KEYS:
            value = opts_dict.get(key)
            if value is None:
                continue
            if key == "position":
                kwargs_resolved["position_x"] = value[0]
                kwargs_resolved["position_y"] = value[1]
                continue
            target_key = self._PYVISTA_BOOL_KEY_MAP.get(key, key)
            kwargs_resolved[target_key] = value

        return kwargs_resolved

    # ==================== OVERRIDE ====================
    # ScalarBar overrides HostBase._helper_commit_apply_opts_main so opts
    # updates are normalized into a resolved backend kwargs payload and then
    # forwarded into owner-side scalar-bar synchronization.
    # ==================================================
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        del is_reapply_opts

        kwargs_left = {}
        kwargs_applied_opts_main = {}
        for key, value in kwargs.items():
            if key not in type(self.opts).__attrs__:
                kwargs_left[key] = value
                continue
            object.__setattr__(self.opts, key, value)
            kwargs_applied_opts_main[key] = getattr(self.opts, key)

        object.__setattr__(
            self, "calc_pyvista_kwargs", self._helper_build_pyvista_kwargs()
        )
        self._helper_request_owner_sync()
        return kwargs_left, kwargs_applied_opts_main

    def __str__(self):
        """Return the compact identity-style string form of this scalar bar."""
        return f"{type(self).__name__}({self.name!r})"

    def __repr__(self):
        """Return a short readable summary of this scalar-bar declaration."""
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"impl_name_pv={self.impl_name_pv!r}, "
            f"mapper_name={self.raw_mapper_name!r}, "
            f"is_visible={self.opts.is_visible!r}, "
            f"pyvista_kwargs={self.calc_pyvista_kwargs!r})"
        )
