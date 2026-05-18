"""Contour-surface domain objects."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Mapping

import numpy as np
import pyvista as pv

from nematics3d.datatypes import (
    DimensionFlagInput,
    Tensor,
    Vect,
    as_Number,
    as_bool,
    as_readonly_array,
    as_real_lattice_field,
)
from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    apply_linear_transform,
    as_readonly_grid_offset,
    as_readonly_grid_transform,
)

from .class_base import ClassBase
from .bounds import as_bounds
from .grid_field.input_grid_field import InputGridField, as_grid_shape
from .registry_base import RegistryBase


def _as_contour_level(value, *, name: str) -> float:
    """Validate one contour level and normalize it to float."""
    return float(as_Number(value, name=name))


class ContourSurface(ClassBase):
    """One contour surface bound to one contour level."""

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this contour surface.",
        },
        "raw_level": {
            "doc": "Contour value represented by this contour surface.",
            "validator": lambda v, d: _as_contour_level(v, name=d),
        },
        "calc_surface_index": {
            "doc": "Index of this contour surface inside its owner contour set.",
            "kind": "calc",
        },
        "entity_mesh_cache": {
            "doc": "Cached extracted contour mesh stored as PyVista PolyData.",
            "kind": "entity",
        },
        "visual": {
            "doc": "The one-to-one contour visual currently associated with this surface.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "impl_sync_func": {
            "doc": "Internal sync callbacks triggered after contour mesh updates.",
        },
        "mesh": {
            "doc": "Read-only: extracted contour mesh, if already cached.",
            "kind": "property",
        },
        "is_extracted": {
            "doc": "Read-only: whether this contour surface already has a cached mesh.",
            "kind": "property",
        },
    }

    __slots__ = ("raw_level", "calc_surface_index", "entity_mesh_cache", "impl_sync_func")

    def __init__(
        self,
        level,
        *,
        owner,
        surface_index: int,
        name: str | None = None,
    ):
        super().__init__(
            name=name,
            name_replace=f"contour_surface_{surface_index}",
            is_fixed=True,
        )
        object.__setattr__(
            self,
            "raw_level",
            self.impl_attrs["raw_level"]["validator"](
                level,
                self.impl_attrs["raw_level"]["doc"],
            ),
        )
        object.__setattr__(self, "calc_surface_index", int(surface_index))
        object.__setattr__(self, "entity_mesh_cache", None)
        object.__setattr__(self, "impl_sync_func", {})
        self.act_bind_relation_base(
            "owner",
            owner,
            doc="The ContourSurfaceSet that owns this contour surface.",
            is_weak=True,
        )

    @property
    def mesh(self):
        """Return the cached contour mesh, if available."""
        return self.entity_mesh_cache

    @property
    def is_extracted(self) -> bool:
        """Return whether this contour surface already has an extracted mesh."""
        return self.entity_mesh_cache is not None

    def act_extract(self, *, is_overwrite: bool = False):
        """Extract and cache the contour mesh for this surface."""
        if self.entity_mesh_cache is not None and not is_overwrite:
            return self.entity_mesh_cache
        owner = self.owner
        if owner is None:
            raise RuntimeError("Cannot extract a contour mesh without a live owner.")
        mesh = owner._helper_extract_contour_mesh(self.raw_level)
        object.__setattr__(self, "entity_mesh_cache", mesh)
        self._helper_trigger_sync_batch(
            mesh=mesh,
            level=float(self.raw_level),
            source=self,
            event="mesh_updated",
        )
        return mesh

    def act_clear_mesh_cache(self):
        """Drop the cached contour mesh for this surface."""
        object.__setattr__(self, "entity_mesh_cache", None)

    def act_attach_sync_task(self, name: str, func: Callable) -> None:
        """Register one post-update sync callback on this contour surface."""
        if not callable(func):
            raise TypeError(f"The sync task {name!r} must be callable.")
        self.impl_sync_func[str(name)] = func

    def act_detach_sync_task(self, name: str) -> None:
        """Detach one registered contour-surface sync callback."""
        self.impl_sync_func.pop(str(name), None)

    def _helper_trigger_sync_batch(self, **kwargs) -> None:
        """Run all registered sync callbacks with the merged sync payload."""
        for func in tuple(self.impl_sync_func.values()):
            try:
                func(**kwargs)
            except (
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
                RuntimeError,
            ):
                continue

    def act_set_level(self, level: float) -> float:
        """Update the contour level and immediately refresh the cached mesh."""
        level_value = self.impl_attrs["raw_level"]["validator"](
            level,
            self.impl_attrs["raw_level"]["doc"],
        )
        object.__setattr__(self, "raw_level", level_value)
        owner = self.owner
        if owner is not None:
            owner._helper_on_surface_level_changed(self)
            self.act_extract(is_overwrite=True)
        return level_value

    def _helper_resolve_current_visual(self):
        """
        Return the currently managed live visual for this surface.

        This helper enforces the single-visual contract used by contour
        surfaces. If the remembered visual no longer belongs to any figure, or
        its figure backend has already died, the stale owner/figure/sync links
        are cleared and `None` is returned.
        """
        visual = self.visual
        if visual is None:
            return None

        fig = getattr(visual, "fig", None)
        if fig is None:
            sync_name = getattr(visual, "impl_owner_sync_name", None)
            if sync_name is not None:
                self.act_detach_sync_task(sync_name)
            if getattr(visual, "owner", None) is self:
                visual.act_unbind_relation_base("owner")
            self.act_unbind_relation_base("visual")
            return None

        if not fig.is_alive:
            sync_name = getattr(visual, "impl_owner_sync_name", None)
            if sync_name is not None:
                self.act_detach_sync_task(sync_name)
            if getattr(visual, "owner", None) is self:
                visual.act_unbind_relation_base("owner")
            visual.act_unbind_relation_base("fig")
            self.act_unbind_relation_base("visual")
            return None

        return visual

    def act_plot(
        self,
        *,
        figure=None,
        opts=None,
        bounds=None,
        name: str | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_extract: bool = True,
        is_replace: bool = False,
        **kwargs,
    ):
        """
        Create or replace the one managed contour visualization for this surface.

        Stale remembered visuals are discarded automatically. If a live visual
        still exists, this method refuses to overwrite it unless
        `is_replace=True`.
        """
        from .visual.plot_contour_surface import PlotContourSurface

        visual_old = self._helper_resolve_current_visual()
        if visual_old is not None:
            if not is_replace:
                raise RuntimeError(
                    f"ContourSurface {self.name!r} already has a live visualization. "
                    "Pass is_replace=True to replace it."
                )
            visual_old.act_remove()

        visual = PlotContourSurface(
            surface=self,
            figure=figure,
            opts=opts,
            bounds=bounds,
            name=name,
            opts_defaults_override=opts_defaults_override,
            is_extract=is_extract,
            **kwargs,
        )
        self.act_bind_relation_base("visual", visual, is_weak=False)
        return visual

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return (
            f"{cls_name}({self.name!r}, level={self.raw_level!r}, "
            f"index={self.calc_surface_index})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class ContourSurfaceSet(ClassBase):
    """Validated host for one scalar field and its per-level contour surfaces."""

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this contour-surface set.",
        },
        "raw_values": {
            "doc": "Validated 3D scalar field values with shape (Nx, Ny, Nz).",
        },
        "impl_init_levels": {
            "doc": "Initialization-time contour levels recorded for reference.",
        },
        "raw_box_periodic_flag": {
            "doc": "Periodic-boundary-condition flags for the source grid.",
        },
        "raw_grid_offset": {
            "doc": "Grid translation offset mapping lattice indices into real space.",
        },
        "raw_grid_transform": {
            "doc": "Grid transform matrix mapping lattice indices into real space.",
        },
        "bounds": {
            "doc": "Optional shared bounds used as the default clipping context for contour visuals.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "impl_visual_default": {
            "doc": "Stored default visual option overrides used when contour plots are created.",
        },
        "impl_plot_opts_defaults_override": {
            "doc": "Stored default opts-default overrides forwarded to contour plot creation.",
        },
        "surface_registry": {
            "doc": "Registry storing the contour surfaces owned by this set.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "surfaces": {
            "doc": "Read-only: contour surfaces owned by this set in level order.",
            "kind": "property",
        },
        "calc_levels": {
            "doc": "Read-only: current contour levels in the current surface order.",
            "kind": "property",
        },
    }

    __slots__ = (
        "raw_values",
        "impl_init_levels",
        "raw_box_periodic_flag",
        "raw_grid_offset",
        "raw_grid_transform",
        "impl_visual_default",
        "impl_plot_opts_defaults_override",
    )

    def __init__(
        self,
        values,
        levels,
        *,
        name: str | None = None,
        grid_offset: Vect(3) | None = None,
        grid_transform: Tensor((3, 3)) = GRID_TRANSFORM_IDENTITY,
        box_periodic_flag: DimensionFlagInput = False,
        bounds=None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        visual_default: Mapping[str, Any] | None = None,
        figure=None,
        is_plot: bool = False,
    ):
        super().__init__(
            name=name,
            name_replace="contour-surface-set",
            is_fixed=True,
        )
        values_use = as_real_lattice_field(
            values,
            name="Contour-surface field values",
            extra_ndim=0,
        )
        shape = as_grid_shape(values_use.shape, name="Contour-surface grid shape")
        grid_info = InputGridField(
            shape=shape,
            box_periodic_flag=box_periodic_flag,
            grid_offset=grid_offset,
            grid_transform=grid_transform,
        )
        if visual_default is None:
            visual_default_use = {}
        elif isinstance(visual_default, Mapping):
            visual_default_use = dict(visual_default)
        else:
            raise TypeError(
                "`visual_default` must be a mapping of plot option overrides."
            )
        if (
            opts_defaults_override is not None
            and not isinstance(opts_defaults_override, Mapping)
        ):
            raise TypeError(
                "`opts_defaults_override` must be a mapping of default opts overrides."
            )

        object.__setattr__(self, "raw_values", as_readonly_array(values_use, dtype=float))
        object.__setattr__(self, "impl_init_levels", self._helper_as_levels(levels))
        object.__setattr__(
            self,
            "raw_box_periodic_flag",
            tuple(bool(v) for v in grid_info.box_periodic_flag),
        )
        object.__setattr__(
            self,
            "raw_grid_offset",
            as_readonly_grid_offset(grid_info.grid_offset),
        )
        object.__setattr__(
            self,
            "raw_grid_transform",
            as_readonly_grid_transform(grid_info.grid_transform),
        )
        object.__setattr__(self, "impl_visual_default", visual_default_use)
        object.__setattr__(
            self,
            "impl_plot_opts_defaults_override",
            None
            if opts_defaults_override is None
            else dict(opts_defaults_override),
        )
        if bounds is not None:
            self.act_bind_relation_base(
                "bounds",
                as_bounds(bounds, name="Contour-surface shared bounds"),
                is_weak=True,
            )

        registry = RegistryBase(
            name=f"{self.name} surfaces",
            info="ContourSurface objects owned by one ContourSurfaceSet",
        )
        self.act_bind_relation_base("surface_registry", registry, is_weak=False)
        self._helper_build_surfaces()
        if as_bool(is_plot, name="Whether to create contour visuals immediately"):
            self.act_plot_all(figure=figure)

    @property
    def surfaces(self) -> tuple[ContourSurface, ...]:
        """Return the contour surfaces in stored level order."""
        registry = self.surface_registry
        if registry is None:
            return ()
        return tuple(registry.entity)

    @property
    def calc_levels(self) -> tuple[float, ...]:
        """Return the current contour levels in the current surface order."""
        return tuple(surface.raw_level for surface in self.surfaces)

    def _helper_as_levels(self, levels) -> tuple[float, ...]:
        """Validate contour levels and preserve the first-seen user order."""
        if np.isscalar(levels):
            raw_levels = [levels]
        else:
            raw_levels = list(levels)
        if len(raw_levels) == 0:
            raise ValueError("Contour levels must contain at least one value.")

        normalized: list[float] = []
        seen: set[float] = set()
        for index, level in enumerate(raw_levels):
            level_value = _as_contour_level(
                level,
                name=f"Contour level {index}",
            )
            if level_value in seen:
                continue
            seen.add(level_value)
            normalized.append(level_value)
        return tuple(normalized)

    def _helper_make_surface_name(self, level: float, surface_index: int) -> str:
        """Return one default child-surface name."""
        return f"{self.name}_level_{surface_index}_{level:g}"

    def _helper_merge_visual_plot_kwargs(self, opts, kwargs):
        """Merge stored contour-visual defaults when no explicit opts object is passed."""
        if opts is not None:
            return dict(kwargs)
        merged = dict(self.impl_visual_default)
        merged.update(dict(kwargs))
        return merged

    def _helper_resolve_plot_bounds(self, bounds):
        """Return explicit plot bounds or the set-level shared bounds when omitted."""
        return self.bounds if bounds is None else bounds

    def _helper_resolve_plot_opts_defaults_override(self, opts_defaults_override):
        """Return explicit or stored opts-default overrides for contour visuals."""
        if opts_defaults_override is None:
            return self.impl_plot_opts_defaults_override
        return opts_defaults_override

    def _helper_update_surface_name(self, surface: ContourSurface) -> str:
        """Refresh one child-surface name from its current level and index."""
        name = self._helper_make_surface_name(
            surface.raw_level,
            int(surface.calc_surface_index),
        )
        surface.act_set_name(name)
        return name

    def _helper_build_contour_image_data(self) -> pv.ImageData:
        """Build one point-centered ImageData object in lattice index space."""
        grid = pv.ImageData(dimensions=tuple(int(v) for v in self.raw_values.shape))
        grid.point_data["contour_values"] = np.asarray(
            self.raw_values,
            dtype=float,
        ).ravel(order="F")
        return grid

    def _helper_transform_contour_mesh_points(self, mesh):
        """Map one extracted contour mesh from index space into physical space."""
        if mesh.n_points == 0:
            return mesh
        points = apply_linear_transform(
            np.asarray(mesh.points, dtype=float),
            transform=self.raw_grid_transform,
            offset=self.raw_grid_offset,
        )
        mesh.points = np.asarray(points, dtype=float)
        return mesh

    def _helper_extract_contour_mesh(self, level: float):
        """Extract one contour mesh at one level and map it into physical space."""
        level_value = _as_contour_level(level, name="Contour extraction level")
        grid = self._helper_build_contour_image_data()
        mesh = grid.contour(isosurfaces=[level_value], scalars="contour_values")
        mesh = self._helper_transform_contour_mesh_points(mesh)
        mesh.field_data["contour_level"] = np.array([level_value], dtype=float)
        return mesh

    def _helper_build_surfaces(self) -> None:
        """Rebuild the child contour surfaces from the current level list."""
        registry = self.surface_registry
        if registry is None:
            raise RuntimeError("ContourSurfaceSet surface registry is not initialized.")
        registry.act_clear(is_show_existing=False)
        for surface_index, level in enumerate(self.impl_init_levels):
            surface = ContourSurface(
                level,
                owner=self,
                surface_index=surface_index,
                name=self._helper_make_surface_name(level, surface_index),
            )
            registry.act_register(surface)

    def _helper_normalize_surface_order(self) -> None:
        """Refresh cached indices and default names after one structural change."""
        for surface_index, surface in enumerate(self.surfaces):
            object.__setattr__(surface, "calc_surface_index", int(surface_index))
            self._helper_update_surface_name(surface)

    def _helper_on_surface_level_changed(self, surface: ContourSurface) -> None:
        """Handle one child-surface level update from the owning container."""
        if surface not in self.surfaces:
            raise ValueError("The contour surface does not belong to this contour set.")
        surface.act_clear_mesh_cache()
        self._helper_update_surface_name(surface)

    def act_get_surface(self, index: int | None):
        """Return one contour surface by insertion index."""
        if index is None:
            return None
        if not isinstance(index, (int, np.integer)):
            raise TypeError(
                "`index` must be int or None when reading a contour surface by index. "
                f"Got {type(index).__name__} instead."
            )
        return self.surfaces[int(index)]

    def act_get_surface_by_name(self, name: str | None):
        """Return one contour surface by registered name."""
        if name is None:
            return None
        return self.surface_registry[name]

    def act_get_surface_by_level(self, level: float):
        """Return one contour surface by exact contour level."""
        level_value = _as_contour_level(level, name="Contour surface level lookup")
        for surface in self.surfaces:
            if surface.raw_level == level_value:
                return surface
        raise KeyError(f"No contour surface with level {level_value!r} exists.")

    def act_refresh(self, *, levels=None):
        """Return a refreshed contour-surface set on the same scalar field."""
        levels_use = self.calc_levels if levels is None else levels
        return type(self)(
            self.raw_values,
            levels_use,
            name=self.name,
            grid_offset=self.raw_grid_offset,
            grid_transform=self.raw_grid_transform,
            box_periodic_flag=self.raw_box_periodic_flag,
            bounds=self.bounds,
            opts_defaults_override=self.impl_plot_opts_defaults_override,
            visual_default=self.impl_visual_default,
        )

    def act_extract_surface(self, index: int, *, is_overwrite: bool = False):
        """Extract one contour surface by insertion index."""
        return self.act_get_surface(index).act_extract(is_overwrite=is_overwrite)

    def act_extract_surface_by_name(self, name: str, *, is_overwrite: bool = False):
        """Extract one contour surface by registered name."""
        return self.act_get_surface_by_name(name).act_extract(is_overwrite=is_overwrite)

    def act_extract_surface_by_level(
        self,
        level: float,
        *,
        is_overwrite: bool = False,
    ):
        """Extract one contour surface by exact contour level."""
        return self.act_get_surface_by_level(level).act_extract(
            is_overwrite=is_overwrite
        )

    def act_extract_all(self, *, is_overwrite: bool = False):
        """Extract and cache contour meshes for every stored level."""
        return tuple(
            surface.act_extract(is_overwrite=is_overwrite) for surface in self.surfaces
        )

    def act_add_surface(self, level: float, *, name: str | None = None):
        """Add one new contour surface at the end of the current family."""
        level_value = _as_contour_level(level, name="Contour surface level")
        try:
            self.act_get_surface_by_level(level_value)
        except KeyError:
            pass
        else:
            raise ValueError(
                f"A contour surface with level {level_value!r} already exists."
            )

        surface_index = len(self.surfaces)
        surface = ContourSurface(
            level_value,
            owner=self,
            surface_index=surface_index,
            name=(
                self._helper_make_surface_name(level_value, surface_index)
                if name is None
                else name
            ),
        )
        self.surface_registry.act_register(surface)
        self._helper_normalize_surface_order()
        return surface

    def act_plot_surface(
        self,
        index: int,
        *,
        figure=None,
        opts=None,
        bounds=None,
        name: str | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_extract: bool = True,
        **kwargs,
    ):
        """Create one PlotContourSurface from the surface at one insertion index."""
        bounds_use = self._helper_resolve_plot_bounds(bounds)
        opts_defaults_override_use = self._helper_resolve_plot_opts_defaults_override(
            opts_defaults_override
        )
        kwargs_use = self._helper_merge_visual_plot_kwargs(opts, kwargs)
        return self.act_get_surface(index).act_plot(
            figure=figure,
            opts=opts,
            bounds=bounds_use,
            name=name,
            opts_defaults_override=opts_defaults_override_use,
            is_extract=is_extract,
            **kwargs_use,
        )

    def act_plot_surface_by_name(
        self,
        name: str,
        *,
        figure=None,
        opts=None,
        bounds=None,
        plot_name: str | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_extract: bool = True,
        **kwargs,
    ):
        """Create one PlotContourSurface from the surface with one registered name."""
        bounds_use = self._helper_resolve_plot_bounds(bounds)
        opts_defaults_override_use = self._helper_resolve_plot_opts_defaults_override(
            opts_defaults_override
        )
        kwargs_use = self._helper_merge_visual_plot_kwargs(opts, kwargs)
        return self.act_get_surface_by_name(name).act_plot(
            figure=figure,
            opts=opts,
            bounds=bounds_use,
            name=plot_name,
            opts_defaults_override=opts_defaults_override_use,
            is_extract=is_extract,
            **kwargs_use,
        )

    def act_plot_surface_by_level(
        self,
        level: float,
        *,
        figure=None,
        opts=None,
        bounds=None,
        name: str | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_extract: bool = True,
        **kwargs,
    ):
        """Create one PlotContourSurface from the surface at one exact contour level."""
        bounds_use = self._helper_resolve_plot_bounds(bounds)
        opts_defaults_override_use = self._helper_resolve_plot_opts_defaults_override(
            opts_defaults_override
        )
        kwargs_use = self._helper_merge_visual_plot_kwargs(opts, kwargs)
        return self.act_get_surface_by_level(level).act_plot(
            figure=figure,
            opts=opts,
            bounds=bounds_use,
            name=name,
            opts_defaults_override=opts_defaults_override_use,
            is_extract=is_extract,
            **kwargs_use,
        )

    def act_plot_all(
        self,
        *,
        figure=None,
        opts=None,
        bounds=None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_extract: bool = True,
        **kwargs,
    ):
        """Create PlotContourSurface wrappers for every stored contour surface."""
        bounds_use = self._helper_resolve_plot_bounds(bounds)
        opts_defaults_override_use = self._helper_resolve_plot_opts_defaults_override(
            opts_defaults_override
        )
        kwargs_use = self._helper_merge_visual_plot_kwargs(opts, kwargs)
        visuals = []
        for surface in self.surfaces:
            visuals.append(
                surface.act_plot(
                    figure=figure,
                    opts=opts,
                    bounds=bounds_use,
                    opts_defaults_override=opts_defaults_override_use,
                    is_extract=is_extract,
                    **kwargs_use,
                )
            )
        return tuple(visuals)

    def _helper_remove_surface(self, surface: ContourSurface) -> ContourSurface:
        """Remove one resolved contour surface and rebuild the child collection."""
        self.surface_registry.act_unregister(surface, is_missing_ok=True)
        self._helper_normalize_surface_order()
        return surface

    def act_remove_surface(self, index: int):
        """Remove one contour surface by insertion index."""
        return self._helper_remove_surface(self.act_get_surface(index))

    def act_remove_surface_by_name(self, name: str):
        """Remove one contour surface by registered name."""
        return self._helper_remove_surface(self.act_get_surface_by_name(name))

    def act_remove_surface_by_level(self, level: float):
        """Remove one contour surface by exact contour level."""
        return self._helper_remove_surface(self.act_get_surface_by_level(level))

    def __len__(self) -> int:
        return len(self.surfaces)

    def __iter__(self):
        return iter(self.surfaces)

    def __getitem__(self, index: int | None):
        return self.act_get_surface(index)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return (
            f"{cls_name}({self.name!r}, values_shape={self.raw_values.shape}, "
            f"levels={self.calc_levels}, n_surfaces={len(self)})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
