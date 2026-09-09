"""Local polar sampling sections attached to smoothed disclination lines."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping

import numpy as np

from ..analysis.disclination.smooth import DisclinationLineSmooth
from ..core.class_base import AttrDef
from ..core.host_base import HostBase, OptsBase
from ..datatypes import UNSET, Unset, Vect, as_bool, as_number, as_vector
from ..format import is_given_str
from ..grid import wrap_points_to_box
from ..logging_decorator import logging_and_warning_decorator
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar


@dataclass(slots=True, repr=False)
class OptsDefectSectionGrid(OptsBase):
    """Options controlling the position of a defect-line sampling section."""

    u_percent: float | Unset = UNSET
    is_wrap: bool | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "u_percent": "Spline-parameter percentage along the smoothed defect line.",
        "is_wrap": "Whether to wrap the resolved origin into the periodic box.",
    }
    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **OptsBase.impl_validators,
        "u_percent": lambda v, d: as_number(v, name=d, value_range=(0, 100)),
        "is_wrap": lambda v, d: as_bool(v, name=d),
    }
    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(getattr(OptsBase, "impl_defaults_frozen", {})),
            "tag": "defect section grid options",
            "u_percent": 50,
            "is_wrap": False,
        }
    )


class DefectSectionGrid(HostBase):
    """Pose-owning wrapper around a local :class:`PlaneGridPolar`.

    The section position and orientation are resolved from a smoothed
    disclination line.  The wrapped polar grid owns the actual sampling
    geometry; this object owns the line-relative pose.
    """

    __attr_defs__ = {
        "state_normal": AttrDef(
            doc="Normal selector: 'tangent', a registered name, or a unit vector.",
            kind="state",
        ),
        "calc_normal": AttrDef(
            doc="Resolved normal currently driving the wrapped polar grid.",
            kind="calc",
        ),
        "impl_normals": AttrDef(
            doc="User-registered named normal providers.", kind="impl"
        ),
        "owner": AttrDef(
            doc="Owning smoothed disclination line.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "wrapped": AttrDef(
            doc="Wrapped PlaneGridPolar used for sampling geometry.",
            kind="relation",
            is_weak_by_default=False,
        ),
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in HostBase.__slots__
    )

    def __init__(
        self,
        line: DisclinationLineSmooth,
        u_percent: float | None = None,
        name: str = "defect section grid",
        name_replace: str = "defect section grid",
        state_normal: str | Vect(3) = "tangent",
        normals: Mapping[str, Any] | None = None,
        opts: OptsDefectSectionGrid | None = None,
        opts_grid: OptsPlaneGridPolar | None = None,
        is_wrap: bool | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        opts_grid_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        if not isinstance(line, DisclinationLineSmooth):
            raise TypeError(
                "`line` must be a DisclinationLineSmooth instance; "
                f"got {type(line).__name__}."
            )

        self_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in OptsDefectSectionGrid.__attrs__
        }
        if u_percent is not None:
            self_kwargs["u_percent"] = u_percent
        if is_wrap is not None:
            self_kwargs["is_wrap"] = is_wrap

        opts_grid_keys = set() if opts_grid is None else set(opts_grid.act_asdict())
        opts_grid_default_keys = (
            set()
            if opts_grid_defaults_override is None
            else set(opts_grid_defaults_override)
        )
        section_pose_keys = {"origin", "normal"} & (
            set(kwargs) | opts_grid_keys | opts_grid_default_keys
        )
        section_pose_keys |= {"grid_offset", "grid_transform"} & set(kwargs)
        if section_pose_keys:
            keys = ", ".join(sorted(section_pose_keys))
            raise ValueError(
                "DefectSectionGrid owns the section pose; do not pass "
                f"{keys}. Configure only polar sampling properties on opts_grid."
            )

        super().__init__(
            opts_type=OptsDefectSectionGrid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **self_kwargs,
        )
        self.act_bind_relation_base("owner", line, is_weak=True)
        object.__setattr__(self, "impl_normals", {})
        object.__setattr__(self, "calc_normal", None)
        self.opts.act_finalize(defaults=self.opts_defaults)

        if normals is not None:
            for key, value in normals.items():
                self.act_register_normal(key, value)

        object.__setattr__(
            self,
            "state_normal",
            self._helper_check_state_normal(
                state_normal, self.show_attr_doc("state_normal")
            ),
        )
        pose = self._helper_resolve_pose()
        grid = PlaneGridPolar(
            normal=pose["normal"],
            origin=pose["origin"],
            opts=opts_grid,
            opts_defaults_override=opts_grid_defaults_override,
            **kwargs,
        )
        grid.act_bind_wrapper(self, protected_attrs=["origin", "normal"])
        self.act_attach_enrich_kwargs_wrapped_task(
            "section_pose", self._helper_enrich_kwargs_wrapped_section
        )
        line.sections.act_register(self)

    def _helper_check_state_normal(self, state_normal, desc):
        if isinstance(state_normal, str):
            state_normal = state_normal.strip()
            if not state_normal:
                raise ValueError("Defect-section normal selector cannot be empty.")
            if state_normal != "tangent" and state_normal not in self.impl_normals:
                raise ValueError(
                    f"{desc} Got unknown registered normal name {state_normal!r}."
                )
            return state_normal
        return as_vector(
            state_normal, name="Direct defect-section normal", is_normalized=True
        )

    def _helper_resolve_normal(self, tangent):
        if is_given_str(self.state_normal, "tangent"):
            normal = tangent
        elif isinstance(self.state_normal, str):
            normal = self.impl_normals[self.state_normal]
            if callable(normal):
                normal = normal()
        else:
            normal = self.state_normal
        normal = as_vector(
            normal, name="Resolved defect-section normal", is_normalized=True
        )
        object.__setattr__(self, "calc_normal", normal)
        return normal

    def _helper_resolve_pose(self):
        tangent, origin = self.owner.act_calc_tangent(
            self.opts.u_percent, is_return_coord=True
        )
        if self.opts.is_wrap:
            origin = wrap_points_to_box(
                origin,
                self.owner.owner.raw_box_size_periodic_index,
                transform=self.owner.owner.raw_grid_transform,
                offset=self.owner.owner.raw_grid_offset,
            )
        return {"origin": origin, "normal": self._helper_resolve_normal(tangent)}

    def _helper_enrich_kwargs_wrapped_section(self, host=None, kwargs=None):
        del host, kwargs
        return self._helper_resolve_pose()

    def _helper_commit_pre_opts(self, kwargs):
        kwargs_sync, is_reapply_opts = super()._helper_commit_pre_opts(kwargs)
        kwargs_applied_state, is_reapply_state = self._helper_commit_pop_raw(
            kwargs, "state_normal", validator=self._helper_check_state_normal
        )
        return kwargs_sync | kwargs_applied_state, is_reapply_opts or is_reapply_state

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        del is_reapply_opts
        for key, value in kwargs.items():
            object.__setattr__(self.opts, key, value)
        return self._helper_resolve_pose(), kwargs

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.name!r}), u_percent={self.opts.u_percent}, "
            f"state_normal={self.state_normal!r}"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    @logging_and_warning_decorator()
    def act_register_normal(self, key, value, logger=None):
        """Register a named normal provider; ``tangent`` remains built in."""
        try:
            if not isinstance(key, str) or not key.strip():
                raise ValueError
            key = key.strip()
        except (TypeError, ValueError):
            logger.warning(f"Skip invalid normal name {key!r}.")
            return
        if key == "tangent":
            logger.warning("'tangent' is built in and cannot be registered.")
            return
        if callable(value):
            self.impl_normals[key] = value
            return
        try:
            value = as_vector(
                value, name=f"Registered normal {key!r}", is_normalized=True
            )
        except (TypeError, ValueError):
            logger.warning(f"Skip invalid registered normal {key!r}.")
            return
        self.impl_normals[key] = value

    @logging_and_warning_decorator()
    def show_normals(self, is_return=False, logger=None):
        """Show the built-in tangent selector and user-registered normals."""
        is_return = as_bool(is_return, name="Whether to return normal summary")
        lines = [f"Normals of {self.name!r}:", "  - tangent: built-in tangent"]
        for key, value in self.impl_normals.items():
            desc = (
                "given function"
                if callable(value)
                else np.array2string(
                    np.asarray(value, dtype=float), precision=3, separator=", "
                )
            )
            lines.append(f"  - {key}: {desc}")
        output = "\n".join(lines)
        logger.info(output)
        return output if is_return else None


__all__ = ["OptsDefectSectionGrid", "DefectSectionGrid"]
