"""Surface-based interpolation results built on :class:`SurfaceSampling`."""

from typing import Any, Mapping

from ..core.class_base import AttrDef, ClassBase
from ..grid.field import GridInterpolator
from .surface_sampling import OptsSurfaceSampling, SurfaceSampling


class InterpolateSurface(ClassBase):
    """Sample a :class:`GridInterpolator` on surface-sampling points."""

    __attr_defs__ = {
        "calc_result": AttrDef(doc="Interpolated values on the current surface.", kind="calc"),
        "sampling": AttrDef(doc="Associated surface sampling object.", kind="relation", is_weak_by_default=False),
        "interpolator": AttrDef(doc="GridInterpolator used to sample the surface.", kind="relation", is_weak_by_default=True),
        "result": AttrDef(doc="Read-only alias of calc_result.", kind="property"),
    }
    __slots__ = tuple(
        name for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property") and name not in ClassBase.__slots__
    )

    def __init__(self, interpolator: GridInterpolator, name: str = "interpolate surface", sampling: SurfaceSampling | None = None, surface=None, opts: OptsSurfaceSampling | None = None, opts_defaults_override: Mapping[str, Any] | None = None, **kwargs):
        super().__init__(name=name, name_replace="interpolate surface")
        if sampling is None:
            if surface is None:
                raise ValueError("Either `sampling` or `surface` must be provided for InterpolateSurface.")
            sampling = SurfaceSampling(surface=surface, opts=opts, opts_defaults_override=opts_defaults_override, name=self.name + "-sampling", **kwargs)
        elif opts is not None or kwargs:
            sampling.act_commit(opts=opts, **kwargs)
        if not isinstance(sampling, SurfaceSampling):
            raise TypeError("`sampling` must be a SurfaceSampling instance.")
        if not isinstance(interpolator, GridInterpolator):
            raise TypeError("`interpolator` must be a nematics3d.grid.field.GridInterpolator instance.")
        self.act_bind_relation_base("sampling", sampling, is_weak=False)
        sampling.act_bind_relation_base("field", self, is_weak=True)
        self.act_bind_relation_base("interpolator", interpolator, is_weak=True)
        self.act_refresh()

    def act_refresh(self):
        object.__setattr__(self, "calc_result", self.interpolator.interpolate(self.sampling.result))

    @property
    def result(self):
        return self.calc_result


__all__ = ["InterpolateSurface"]
