"""Q-tensor sampling and derived structure on sampled surfaces."""

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from ..analysis.disclination import defect_detect_surface
from ..core.class_base import AttrDef
from ..geometry import triangulate_surface_points
from ..grid.field import GridInterpolator
from ..q_field.diagonalization import q_diagonalize
from ..visual.color import n_color_immerse
from .interpolate_surface import InterpolateSurface
from .surface_sampling import OptsSurfaceSampling, SurfaceSampling


class QSurface(InterpolateSurface):
    """Sample a Q-tensor field on a surface and derive director/defect structure."""

    __attr_defs__ = {
        "calc_n": AttrDef(doc="Director field derived from Q-diagonalization.", kind="calc"),
        "calc_S": AttrDef(doc="Scalar order derived from Q-diagonalization.", kind="calc"),
        "calc_surface_mesh": AttrDef(doc="Triangulated mesh built from sampled points.", kind="calc"),
        "calc_defect_pos": AttrDef(doc="Detected defect positions.", kind="calc"),
        "calc_is_near_defect": AttrDef(doc="Mask of sampled points near defects.", kind="calc"),
        "state_is_interactable": AttrDef(doc="Whether visualization controls may be opened.", kind="state"),
        "default_visual_opts": AttrDef(doc="Default overrides used by Q-surface visuals.", kind="default"),
        "visual_nb": AttrDef(doc="Optional bulk-director visual.", kind="relation", is_weak_by_default=False),
        "visual_nd": AttrDef(doc="Optional near-defect director visual.", kind="relation", is_weak_by_default=False),
        "visual_defect": AttrDef(doc="Optional defect-position visual.", kind="relation", is_weak_by_default=False),
    }
    __slots__ = tuple(name for name, spec in __attr_defs__.items() if spec.kind not in ("relation", "property", "opts") and name not in InterpolateSurface.__slots__)
    _origin_default_visual_opts = {"nb": {"color": n_color_immerse, "opacity": 0.2}, "nd": {"color": n_color_immerse, "opacity": 1}}

    def __init__(self, interpolator: GridInterpolator, name: str = "Q-surface", sampling: SurfaceSampling | None = None, surface=None, opts: OptsSurfaceSampling | None = None, opts_defaults_override: Mapping[str, Any] | None = None, visual_default: Mapping[str, Any] | None = None, **kwargs):
        defaults = deepcopy(self._origin_default_visual_opts)
        visual_default = {} if visual_default is None else dict(visual_default)
        unexpected = set(visual_default) - set(defaults)
        if unexpected:
            raise ValueError(f"`visual_default` only accepts {sorted(defaults)!r}; got {sorted(unexpected)!r}.")
        for key, override in visual_default.items():
            if not isinstance(override, Mapping):
                raise TypeError(f"`visual_default[{key!r}]` must be a mapping.")
            defaults[key] |= dict(override)
        object.__setattr__(self, "default_visual_opts", defaults)
        object.__setattr__(self, "state_is_interactable", True)
        object.__setattr__(self, "calc_surface_mesh", None)
        object.__setattr__(self, "calc_defect_pos", np.empty((0, 3), dtype=float))
        object.__setattr__(self, "calc_is_near_defect", np.empty(0, dtype=bool))
        super().__init__(interpolator=interpolator, name=name, sampling=sampling, surface=surface, opts=opts, opts_defaults_override=opts_defaults_override, **kwargs)

    def act_refresh(self):
        coords = self.sampling.result
        mesh = triangulate_surface_points(coords)
        Q = self.interpolator.interpolate(coords)
        diagonalization = q_diagonalize(Q)
        defect_pos, near_mask = defect_detect_surface(mesh, diagonalization.n, is_return_mask=True)
        object.__setattr__(self, "calc_surface_mesh", mesh)
        object.__setattr__(self, "calc_result", Q)
        object.__setattr__(self, "calc_S", diagonalization.S)
        object.__setattr__(self, "calc_n", diagonalization.n)
        object.__setattr__(self, "calc_defect_pos", defect_pos)
        object.__setattr__(self, "calc_is_near_defect", near_mask)
        if any(getattr(self, name, None) is not None for name in ("visual_nb", "visual_nd", "visual_defect")):
            from ..visual.q_surface import update_q_surface_visuals
            update_q_surface_visuals(self)

    def act_visualize_n(self, *args, **kwargs):
        from ..visual.q_surface import visualize_q_surface_n
        return visualize_q_surface_n(self, *args, **kwargs)


__all__ = ["QSurface"]
