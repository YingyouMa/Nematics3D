"""Smoothed trajectory object for disclination lines."""

from __future__ import annotations

import weakref
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import splprep

from ...core.class_base import AttrDef
from ...core.opts import merge_opts_all
from ...core.registry_base import RegistryBase
from ...geometry.smoothing.line import OptsSmoothedLine, SmoothedLine
from ...grid import apply_linear_transform, unwrap_trajectory
from ...logging_decorator import logging_and_warning_decorator
from .line import DisclinationLine
from .section import calc_defect_section_omega, sample_beta_from_smooth


class DisclinationLineSmooth(SmoothedLine):
    """Smoothed, parameterized representation of a :class:`DisclinationLine`."""

    __attr_defs__ = {
        "calc_coords_index": AttrDef(
            doc="Index-space trajectory entering smoothing.", kind="calc"
        ),
        "calc_result_index": AttrDef(
            doc="Smoothed trajectory in lattice-index space.", kind="calc"
        ),
        "calc_result_coords": AttrDef(
            doc="Compatibility alias of calc_result in real space.", kind="calc"
        ),
        "calc_padding_num": AttrDef(
            doc="Padding used for cross-boundary smoothing.", kind="calc"
        ),
        "impl_owner_init_ref": AttrDef(
            doc="Temporary owner weakref used during initialization.", kind="impl"
        ),
        "owner": AttrDef(
            doc="Owning raw disclination line.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "visual": AttrDef(
            doc="Managed visualization wrapper, if any.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "visual_tube": AttrDef(
            doc="Wrapped PlotTube of the current visualization.",
            kind="property",
            is_public_settable=False,
        ),
        "sections": AttrDef(
            doc="Registry of derived cross-section grids.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "linefunc_mode": AttrDef(
            doc="Interpolation mode for functions sampled along this line.",
            kind="property",
            is_public_settable=False,
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in SmoothedLine.__slots__
    )

    def __init__(
        self,
        line: DisclinationLine,
        opts: OptsSmoothedLine | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        name: str | None = None,
        **kwargs,
    ):
        if not isinstance(line, DisclinationLine):
            raise TypeError(
                "`line` must be a DisclinationLine instance; "
                f"got {type(line).__name__}."
            )
        if name is None:
            name = line.name

        object.__setattr__(self, "impl_owner_init_ref", weakref.ref(line))
        super().__init__(
            line.calc_defect_coords,
            name=name,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )
        self.act_bind_relation_base("owner", line, is_weak=True)
        object.__setattr__(self, "impl_owner_init_ref", None)

        sections = RegistryBase(
            name="Planes",
            info=f"cross-section grids of smoothed disclination line {self.name!r}",
        )
        self.act_bind_relation_base("sections", sections, is_weak=False)
        sections.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "calc_padding_num", 0)
        self.act_register_protected_attr(["coords", "mode"])

    def _helper_get_owner_during_init(self) -> DisclinationLine:
        owner = getattr(self, "owner", None)
        if owner is not None:
            return owner
        owner_ref = getattr(self, "impl_owner_init_ref", None)
        if isinstance(owner_ref, weakref.ReferenceType):
            owner = owner_ref()
        if owner is None:
            raise RuntimeError("Cannot resolve the owning DisclinationLine.")
        return owner

    @property
    def linefunc_mode(self):
        owner = self.owner
        if owner is not None and owner.kind in ("loop", "cross"):
            return "wrap"
        return self.opts.mode

    @property
    def visual_tube(self):
        visual = self.visual
        return None if visual is None else visual.wrapped

    def _helper_resolve_coords(self):
        owner = self._helper_get_owner_during_init()
        indices = owner.raw_defect_indices.copy()
        padding_num = 0
        smooth_mode = "interp"

        if owner.kind == "loop":
            smooth_mode = "wrap"
        elif owner.kind == "cross":
            box_size = owner.raw_box_size_periodic_index
            if self.opts.window_ratio is not None:
                padding_num = int(len(indices) / self.opts.window_ratio / 2)
            else:
                padding_num = int((self.opts.window_length or len(indices)) / 2)

            origin_indices = owner.raw_defect_indices.copy()
            tail = origin_indices[:padding_num].copy()
            head = origin_indices[-padding_num:].copy()
            indices = np.concatenate([head, origin_indices, tail])
            indices = unwrap_trajectory(indices, box_size_periodic=box_size)

            start_origin = owner.raw_defect_indices[0]
            start_now = indices[padding_num]
            periodic = np.isfinite(box_size)
            shift = np.zeros(3, dtype=float)
            shift[periodic] = np.round(
                (start_origin[periodic] - start_now[periodic]) / box_size[periodic]
            )
            indices += shift * box_size

        object.__setattr__(self, "calc_padding_num", padding_num)
        object.__setattr__(self, "calc_coords_index", indices)
        object.__setattr__(
            self,
            "calc_coords",
            apply_linear_transform(
                indices,
                transform=owner.raw_grid_transform,
                offset=owner.raw_grid_offset,
            ),
        )
        object.__setattr__(self.opts, "mode", smooth_mode)

    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):
        owner = self._helper_get_owner_during_init()
        if "mode" in kwargs:
            kwargs.pop("mode")
            logger.warning(
                "`mode` is controlled by the disclination-line kind and was ignored."
            )

        super()._helper_commit_apply_opts_main(
            is_reapply_opts=is_reapply_opts, **kwargs
        )

        padding_num = int(self.calc_padding_num)
        if self.calc_is_smoothed and padding_num > 0 and len(self.calc_result) > 0:
            trim = int(round(padding_num * float(self.opts.num_out_ratio)))
            if trim > 0 and 2 * trim < len(self.calc_result):
                result = self.calc_result[trim:-trim]
                object.__setattr__(self, "calc_result", result)
                object.__setattr__(
                    self,
                    "entity_tck",
                    splprep(
                        result.T.copy(),
                        u=np.linspace(0.0, 1.0, len(result)),
                        s=0,
                        per=0,
                    )[0],
                )

        if not self.calc_is_smoothed:
            result_index = owner.raw_defect_indices.copy()
            self._helper_set_result(
                apply_linear_transform(
                    result_index,
                    transform=owner.raw_grid_transform,
                    offset=owner.raw_grid_offset,
                )
            )
            result = self.calc_result
        else:
            result = self.calc_result
            result_index = apply_linear_transform(
                result,
                transform=owner.raw_grid_transform,
                offset=owner.raw_grid_offset,
                is_inv=True,
            )

        object.__setattr__(self, "calc_result_index", result_index)
        object.__setattr__(self, "calc_result_coords", result)
        if self.visual:
            self.visual.act_commit()

    def act_add_beta_interpolator(
        self,
        u_samples: np.ndarray | None = None,
        name: str | None = None,
        opts_grid=None,
        opts_grid_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        if opts_grid is None:
            from ...sample.plane_grid_polar import OptsPlaneGridPolar

            opts_grid = OptsPlaneGridPolar()
        opts_grid = merge_opts_all({"grid_": opts_grid}, kwargs, type(self).__name__)[
            "grid_"
        ]
        if u_samples is None:
            u_samples = (
                np.arange(0, 100, 5, dtype=float)
                if self.linefunc_mode == "wrap"
                else np.linspace(0, 100, 21, dtype=float)
            )
        if name is None:
            try:
                index_smooth = self.owner.smooths.index(self)
            except (AttributeError, ValueError):
                index_smooth = -1
            name = f"beta_smooth_{index_smooth}"
        return self.act_create_linefunc(
            func=sample_beta_from_smooth,
            u_samples=u_samples,
            func_kwargs={
                "smooth": self,
                "opts_grid": opts_grid,
                "opts_grid_defaults_override": opts_grid_defaults_override,
            },
            result_value_attr="beta",
            name=name,
        )

    def act_visualize(self, *args, **kwargs):
        from ...visual.disclination_line import DisclinationLineSmoothPlot

        visual_old = self._helper_resolve_current_visual()
        is_replace = kwargs.pop("is_replace", False)
        if visual_old is not None:
            if not is_replace:
                raise RuntimeError(
                    f"DisclinationLineSmooth {self.name!r} already has a live visualization. "
                    "Pass is_replace=True to replace it."
                )
            visual_old.act_remove()
        visual = DisclinationLineSmoothPlot(self, *args, **kwargs)
        self.act_bind_relation_base("visual", visual, is_weak=False)
        return visual

    def _helper_resolve_current_visual(self):
        visual = self.visual
        if visual is None:
            return None
        wrapped = getattr(visual, "wrapped", None)
        fig = None if wrapped is None else getattr(wrapped, "fig", None)
        if wrapped is not None and fig is not None and fig.is_alive:
            return visual
        if wrapped is not None and getattr(wrapped, "wrapper", None) is visual:
            wrapped.act_unbind_wrapper()
        if getattr(visual, "owner", None) is self:
            visual.act_unbind_relation_base("owner")
        self.act_unbind_relation_base("visual")
        return None

    def act_cross_section(self, x_param, **kwargs):
        from ...sample.defect_section import DefectSectionGrid

        return DefectSectionGrid(self, u_percent=x_param, **kwargs)

    def act_calc_omega(self, *args, **kwargs):
        return calc_defect_section_omega(self, *args, **kwargs)


__all__ = ["DisclinationLineSmooth"]
