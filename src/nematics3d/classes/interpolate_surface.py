"""Q-tensor interpolation on sampled surfaces with director visualization."""

from copy import deepcopy
from typing import Any, Mapping

from nematics3d.field import Q_diagonalize, n_color_immerse

from .class_base import AttrDef, ClassBase
from .grid_field import GridInterpolator
from .surface_sampling import SurfaceSampling, OptsSurfaceSampling


class InterpolateSurface(ClassBase):
    """
    InterpolateSurface samples a GridInterpolator on SurfaceSampling points.
    """

    __attr_defs__ = {
        "calc_result": AttrDef(
            doc="The interpolated physics values sampled on the current surface.",
            kind="calc",
        ),
        "sampling": AttrDef(
            doc="The surface sampling object associated with this interpolated field.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "interpolator": AttrDef(
            doc="The grid interpolator object used to sample this surface.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "result": AttrDef(
            doc="Read-only: Alias of calc_result.",
            kind="property",
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property")
        and name not in ClassBase.__slots__
    )

    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "interpolate surface",
        sampling: SurfaceSampling | None = None,
        surface=None,
        opts: OptsSurfaceSampling | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(name=name, name_replace="interpolate surface")

        if sampling is None:
            if surface is None:
                raise ValueError(
                    "Either `sampling` or `surface` must be provided for InterpolateSurface."
                )
            sampling = SurfaceSampling(
                surface=surface,
                opts=opts,
                opts_defaults_override=opts_defaults_override,
                name=self.name + "-sampling",
                **kwargs,
            )
        elif opts is not None or kwargs:
            sampling.act_commit(opts=opts, **kwargs)

        self.act_bind_relation_base("sampling", sampling, is_weak=False)
        sampling.act_bind_relation_base("field", self, is_weak=True)

        if not isinstance(interpolator, GridInterpolator):
            raise TypeError(
                "Interpolator for InterpolateSurface must be an instance of "
                "nematics3d.classes.grid_field.GridInterpolator."
            )
        self.act_bind_relation_base("interpolator", interpolator, is_weak=True)

        self.act_refresh()

    def act_refresh(self):
        coords = self.sampling.result
        result = self.interpolator.interpolate(coords)
        object.__setattr__(self, "calc_result", result)

    @property
    def result(self):
        return self.calc_result


