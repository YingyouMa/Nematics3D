"""Contour-surface domain objects and isosurface extraction."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Mapping

import numpy as np
import pyvista as pv

from nematics3d.analysis.bounds import as_bounds
from nematics3d.grid.field import InputGridField
from nematics3d.core.class_base import AttrDef, ClassBase
from nematics3d.core.registry_base import RegistryBase
from nematics3d.datatypes import (
    DimensionInfo,
    Vect,
    as_bool,
    as_grid_shape,
    as_number,
    as_readonly_array,
    as_real_lattice_field,
)
from nematics3d.misc import add_periodic_boundary
from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    GridTransform,
    apply_linear_transform,
    as_grid_offset,
    as_grid_transform,
)


def _as_contour_level(value, *, name: str) -> float:
    """Validate one contour level and normalize it to ``float``."""
    return float(as_number(value, name=name))


class ContourSurface(ClassBase):
    """One cached isosurface at one scalar-field level."""

    __attr_defs__: ClassVar = {
        "raw_level": AttrDef(
            doc="Contour value represented by this contour surface.",
            kind="raw",
            validator=lambda v, d: _as_contour_level(v, name=d),
        ),
        "calc_surface_index": AttrDef(
            doc="Index of this contour surface inside its owner contour set.",
            kind="calc",
        ),
        "entity_mesh_cache": AttrDef(
            doc="Cached extracted contour mesh stored as PyVista PolyData.",
            kind="entity",
        ),
        "visual": AttrDef(
            doc="The one-to-one contour visual currently associated with this surface.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "impl_sync_func": AttrDef(
            doc="Internal sync callbacks triggered after contour mesh updates.",
            kind="impl",
        ),
        "mesh": AttrDef(
            doc="Read-only: extracted contour mesh, if already cached.",
            kind="property",
        ),
        "is_extracted": AttrDef(
            doc="Read-only: whether this contour surface already has a cached mesh.",
            kind="property",
        ),
    }

    __slots__ = (
        "raw_level",
        "calc_surface_index",
        "entity_mesh_cache",
        "impl_sync_func",
    )

    def __init__(self, level, *, owner, surface_index: int, name: str | None = None):
        super().__init__(
            name=name,
            name_replace=f"contour_surface_{surface_index}",
            is_fixed=True,
        )
        object.__setattr__(
            self,
            "raw_level",
            type(self)
            .__attr_defs__["raw_level"]
            .validator(level, type(self).__attr_defs__["raw_level"].doc),
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
        return self.entity_mesh_cache

    @property
    def is_extracted(self) -> bool:
        return self.entity_mesh_cache is not None

    def act_extract(self, *, is_overwrite: bool = False):
        """Extract and cache the isosurface mesh for this level."""
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
        object.__setattr__(self, "entity_mesh_cache", None)

    def act_attach_sync_task(self, name: str, func: Callable) -> None:
        if not callable(func):
            raise TypeError(f"The sync task {name!r} must be callable.")
        self.impl_sync_func[str(name)] = func

    def act_detach_sync_task(self, name: str) -> None:
        self.impl_sync_func.pop(str(name), None)

    def _helper_trigger_sync_batch(self, **kwargs) -> None:
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
        """Set the contour level and refresh the cached mesh immediately."""
        level_value = (
            type(self)
            .__attr_defs__["raw_level"]
            .validator(level, type(self).__attr_defs__["raw_level"].doc)
        )
        object.__setattr__(self, "raw_level", level_value)
        owner = self.owner
        if owner is not None:
            owner._helper_on_surface_level_changed(self)
            self.act_extract(is_overwrite=True)
        return level_value

    def _helper_resolve_current_visual(self):
        visual = self.visual
        if visual is None:
            return None
        fig = getattr(visual, "fig", None)
        if fig is None or not fig.is_alive:
            sync_name = getattr(visual, "impl_owner_sync_name", None)
            if sync_name is not None:
                self.act_detach_sync_task(sync_name)
            if getattr(visual, "owner", None) is self:
                visual.act_unbind_relation_base("owner")
            if fig is not None:
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
        """Create or replace the one managed visualization for this surface."""
        from nematics3d.visual.plot_contour_surface import PlotContourSurface

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
        return (
            f"{type(self).__name__}({self.name!r}, level={self.raw_level!r}, "
            f"index={self.calc_surface_index})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class ContourSurfaceSet(ClassBase):
    """Scalar-field host owning a family of per-level contour surfaces."""

    __attr_defs__: ClassVar = {
        "raw_values": AttrDef(
            doc="Validated 3D scalar field values with shape (Nx, Ny, Nz).",
            kind="raw",
        ),
        "impl_init_levels": AttrDef(
            doc="Initialization-time contour levels recorded for reference.",
            kind="impl",
        ),
        "raw_box_periodic_flag": AttrDef(
            doc="Periodic-boundary-condition flags for the source grid.",
            kind="raw",
        ),
        "raw_grid_offset": AttrDef(
            doc="Grid translation offset mapping lattice indices into real space.",
            kind="raw",
        ),
        "raw_grid_transform": AttrDef(
            doc="Grid transform matrix mapping lattice indices into real space.",
            kind="raw",
        ),
        "bounds": AttrDef(
            doc="Optional shared bounds used as the default clipping context for contour visuals.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "impl_visual_default": AttrDef(
            doc="Stored default visual option overrides used when contour plots are created.",
            kind="impl",
        ),
        "impl_plot_opts_defaults_override": AttrDef(
            doc="Stored default opts-default overrides forwarded to contour plot creation.",
            kind="impl",
        ),
        "surface_registry": AttrDef(
            doc="Registry storing the contour surfaces owned by this set.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "surfaces": AttrDef(
            doc="Read-only: contour surfaces owned by this set in level order.",
            kind="property",
        ),
        "calc_levels": AttrDef(
            doc="Read-only: current contour levels in the current surface order.",
            kind="property",
        ),
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
        grid_transform: GridTransform = GRID_TRANSFORM_IDENTITY,
        box_periodic_flag: DimensionInfo = False,
        bounds=None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        visual_default: Mapping[str, Any] | None = None,
        figure=None,
        is_extract: bool = False,
        is_plot: bool = False,
    ):
        super().__init__(name=name, name_replace="contour-surface-set", is_fixed=True)
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
        if opts_defaults_override is not None and not isinstance(
            opts_defaults_override, Mapping
        ):
            raise TypeError(
                "`opts_defaults_override` must be a mapping of default opts overrides."
            )

        object.__setattr__(
            self, "raw_values", as_readonly_array(values_use, dtype=float)
        )
        object.__setattr__(self, "impl_init_levels", self._helper_as_levels(levels))
        object.__setattr__(
            self,
            "raw_box_periodic_flag",
            tuple(bool(v) for v in grid_info.box_periodic_flag),
        )
        object.__setattr__(
            self,
            "raw_grid_offset",
            as_grid_offset(grid_info.grid_offset, is_readonly=True),
        )
        object.__setattr__(
            self,
            "raw_grid_transform",
            as_grid_transform(grid_info.grid_transform, is_readonly=True),
        )
        object.__setattr__(self, "impl_visual_default", visual_default_use)
        object.__setattr__(
            self,
            "impl_plot_opts_defaults_override",
            None if opts_defaults_override is None else dict(opts_defaults_override),
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
        if as_bool(is_extract, name="Whether to extract contour meshes immediately"):
            self.act_extract_all()
        if as_bool(is_plot, name="Whether to create contour visuals immediately"):
            self.act_plot_all(figure=figure)

    @property
    def surfaces(self) -> tuple[ContourSurface, ...]:
        registry = self.surface_registry
        if registry is None:
            return ()
        return tuple(registry.entity)

    @property
    def calc_levels(self) -> tuple[float, ...]:
        return tuple(surface.raw_level for surface in self.surfaces)

    def _helper_as_levels(self, levels) -> tuple[float, ...]:
        raw_levels = [levels] if np.isscalar(levels) else list(levels)
        if not raw_levels:
            raise ValueError("Contour levels must contain at least one value.")
        normalized: list[float] = []
        seen: set[float] = set()
        for index, level in enumerate(raw_levels):
            level_value = _as_contour_level(level, name=f"Contour level {index}")
            if level_value in seen:
                continue
            seen.add(level_value)
            normalized.append(level_value)
        return tuple(normalized)

    def _helper_make_surface_name(self, level: float, surface_index: int) -> str:
        return f"{self.name}_level_{surface_index}_{level:g}"

    def _helper_merge_visual_plot_kwargs(self, opts, kwargs):
        if opts is not None:
            return dict(kwargs)
        merged = dict(self.impl_visual_default)
        merged.update(dict(kwargs))
        return merged

    def _helper_resolve_plot_bounds(self, bounds):
        return self.bounds if bounds is None else bounds

    def _helper_resolve_plot_opts_defaults_override(self, opts_defaults_override):
        if opts_defaults_override is None:
            return self.impl_plot_opts_defaults_override
        return opts_defaults_override

    def _helper_update_surface_name(self, surface: ContourSurface) -> str:
        name = self._helper_make_surface_name(
            surface.raw_level, int(surface.calc_surface_index)
        )
        surface.act_set_name(name)
        return name

    def _helper_build_contour_image_data(self, values) -> pv.ImageData:
        """Build point-centered ``ImageData`` in lattice-index space."""
        values_use = as_real_lattice_field(
            values,
            name="Contour extraction field values",
            extra_ndim=0,
        )
        grid = pv.ImageData(dimensions=tuple(int(v) for v in values_use.shape))
        grid.point_data["contour_values"] = np.asarray(values_use, dtype=float).ravel(
            order="F"
        )
        return grid

    def _helper_transform_contour_mesh_points(self, mesh):
        if mesh.n_points == 0:
            return mesh
        mesh.points = np.asarray(
            apply_linear_transform(
                np.asarray(mesh.points, dtype=float),
                transform=self.raw_grid_transform,
                offset=self.raw_grid_offset,
            ),
            dtype=float,
        )
        return mesh

    def _helper_extract_contour_mesh(self, level: float):
        """Extract one isosurface and map it from index to physical space."""
        level_value = _as_contour_level(level, name="Contour extraction level")
        values_contour = add_periodic_boundary(
            self.raw_values,
            self.raw_box_periodic_flag,
        )
        grid = self._helper_build_contour_image_data(values_contour)
        mesh = grid.contour(isosurfaces=[level_value], scalars="contour_values")
        mesh = self._helper_transform_contour_mesh_points(mesh)
        mesh.field_data["contour_level"] = np.array([level_value], dtype=float)
        return mesh

    def _helper_build_surfaces(self) -> None:
        registry = self.surface_registry
        if registry is None:
            raise RuntimeError("ContourSurfaceSet surface registry is not initialized.")
        registry.act_clear(is_show_existing=False)
        for surface_index, level in enumerate(self.impl_init_levels):
            registry.act_register(
                ContourSurface(
                    level,
                    owner=self,
                    surface_index=surface_index,
                    name=self._helper_make_surface_name(level, surface_index),
                )
            )

    def _helper_normalize_surface_order(self) -> None:
        for surface_index, surface in enumerate(self.surfaces):
            object.__setattr__(surface, "calc_surface_index", int(surface_index))
            self._helper_update_surface_name(surface)

    def _helper_on_surface_level_changed(self, surface: ContourSurface) -> None:
        if surface not in self.surfaces:
            raise ValueError("The contour surface does not belong to this contour set.")
        surface.act_clear_mesh_cache()
        self._helper_update_surface_name(surface)

    def act_get_surface(self, index: int | None):
        if index is None:
            return None
        if not isinstance(index, (int, np.integer)):
            raise TypeError(
                "`index` must be int or None when reading a contour surface by index. "
                f"Got {type(index).__name__} instead."
            )
        return self.surfaces[int(index)]

    def act_get_surface_by_name(self, name: str | None):
        if name is None:
            return None
        return self.surface_registry[name]

    def act_get_surface_by_level(self, level: float):
        level_value = _as_contour_level(level, name="Contour surface level lookup")
        for surface in self.surfaces:
            if surface.raw_level == level_value:
                return surface
        raise KeyError(f"No contour surface with level {level_value!r} exists.")

    def act_refresh(self, *, levels=None):
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
        return self.act_get_surface(index).act_extract(is_overwrite=is_overwrite)

    def act_extract_surface_by_name(self, name: str, *, is_overwrite: bool = False):
        return self.act_get_surface_by_name(name).act_extract(is_overwrite=is_overwrite)

    def act_extract_surface_by_level(self, level: float, *, is_overwrite: bool = False):
        return self.act_get_surface_by_level(level).act_extract(
            is_overwrite=is_overwrite
        )

    def act_extract_all(self, *, is_overwrite: bool = False):
        return tuple(
            surface.act_extract(is_overwrite=is_overwrite) for surface in self.surfaces
        )

    def act_add_surface(self, level: float, *, name: str | None = None):
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

    def _helper_plot_surface(
        self,
        surface: ContourSurface,
        *,
        figure=None,
        opts=None,
        bounds=None,
        name: str | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_extract: bool = True,
        **kwargs,
    ):
        """Apply set-level visual defaults and plot one resolved child surface."""
        return surface.act_plot(
            figure=figure,
            opts=opts,
            bounds=self._helper_resolve_plot_bounds(bounds),
            name=name,
            opts_defaults_override=self._helper_resolve_plot_opts_defaults_override(
                opts_defaults_override
            ),
            is_extract=is_extract,
            **self._helper_merge_visual_plot_kwargs(opts, kwargs),
        )

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
        return self._helper_plot_surface(
            self.act_get_surface(index),
            figure=figure,
            opts=opts,
            bounds=bounds,
            name=name,
            opts_defaults_override=opts_defaults_override,
            is_extract=is_extract,
            **kwargs,
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
        return self._helper_plot_surface(
            self.act_get_surface_by_name(name),
            figure=figure,
            opts=opts,
            bounds=bounds,
            name=plot_name,
            opts_defaults_override=opts_defaults_override,
            is_extract=is_extract,
            **kwargs,
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
        return self._helper_plot_surface(
            self.act_get_surface_by_level(level),
            figure=figure,
            opts=opts,
            bounds=bounds,
            name=name,
            opts_defaults_override=opts_defaults_override,
            is_extract=is_extract,
            **kwargs,
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
        return tuple(
            self._helper_plot_surface(
                surface,
                figure=figure,
                opts=opts,
                bounds=bounds,
                opts_defaults_override=opts_defaults_override,
                is_extract=is_extract,
                **kwargs,
            )
            for surface in self.surfaces
        )

    def _helper_remove_surface(self, surface: ContourSurface) -> ContourSurface:
        visual = surface._helper_resolve_current_visual()
        if visual is not None:
            visual.act_remove()
        surface.act_unbind_relation_base("visual")
        surface.act_unbind_relation_base("owner")
        self.surface_registry.act_unregister(surface, is_missing_ok=True)
        self._helper_normalize_surface_order()
        return surface

    def act_remove_surface(self, index: int):
        return self._helper_remove_surface(self.act_get_surface(index))

    def act_remove_surface_by_name(self, name: str):
        return self._helper_remove_surface(self.act_get_surface_by_name(name))

    def act_remove_surface_by_level(self, level: float):
        return self._helper_remove_surface(self.act_get_surface_by_level(level))

    def __len__(self) -> int:
        return len(self.surfaces)

    def __iter__(self):
        return iter(self.surfaces)

    def __getitem__(self, index: int | None):
        return self.act_get_surface(index)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.name!r}, values_shape={self.raw_values.shape}, "
            f"levels={self.calc_levels}, n_surfaces={len(self)})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


__all__ = ["ContourSurface", "ContourSurfaceSet"]
