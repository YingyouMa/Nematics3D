import numpy as np
from dataclasses import dataclass
from typing import Literal, Any, Mapping
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.field import generate_fixed_step_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box
from .opts import merge_opts_all
from .host_base import OptsBase, HostBase
from Nematics3D.datatypes import Number, as_Number, Vect, as_Vect, Tensor, as_Tensor, as_str, UNSET, Unset, as_bool
from .opts import cover_value

from .visual.plot_extent import PlotExtent
from .visual.plot_tube import OptsTube
from .visual.plot_figure import PlotFigure, OptsFigure
from .visual.plot_sphere import PlotSphere, OptsSphere


@dataclass(slots=True, repr=False)
class OptsPlaneGridPolar(OptsBase):
    """
    Options for generating a polar (concentric-ring) point lattice on a plane.

    This option set targets the "ring + equal arc-length" strategy:
    - Rings are placed at radii r_i = (i + 0.5) * dr up to R_max
    - Points on each ring are spaced by approximately constant arc length
      (via N_theta(i) ≈ round(2π r_i / arc_dist))
    - Rings are angularly staggered using the golden angle for reduced aliasing
      (a deterministic, reproducible staggering scheme)
    """

    origin:                     Vect(3) | Unset                  = UNSET  # disk center in 3D
    normal:                     Vect(3) | Unset                  = UNSET  # plane normal (unit)
    theta0_axis:                Vect(3) | None | Unset           = UNSET  # in-plane reference axis for theta=0 (unit after projection)
    R_max:                      Number | Unset                   = UNSET  # maximum radius of disk
    dr:                         Number | Unset                   = UNSET  # radial ring spacing
    arc_dist:                   Number | Unset                   = UNSET  # target arc-length spacing along each ring
    start_layer:                int | Unset                      = UNSET  # 0 includes center; 1 starts from first ring; 2 starts from second ring; ...
    corners_limit:              Tensor((8, 3)) | None | UNSET    = UNSET
    grid_offset:                Vect(3) | Unset                  = UNSET
    grid_transform:             Tensor((3, 3)) | Unset           = UNSET

    __descriptions__ = {
        **(OptsBase.__descriptions__),
        "origin":               "center of the polar grid in index coordinates",
        "normal":               "normal of the plane (unit vector)",
        "theta0_axis":          "in-plane reference axis defining theta=0; will be projected onto the plane and normalized (None uses the default axis)",
        "R_max":                "maximum radius of the disk (sampling domain)",
        "dr":                   "radial spacing between rings; rings at r_i = (i + 0.5) * dr",
        "arc_dist":             "target arc-length spacing between adjacent points along each ring",
        "start_layer":          "starting layer index: 0 includes the center point; 1 starts from the first ring (r=0.5*dr); 2 starts from the second ring, etc.",
        "corners_limit":        "bounding box corners (8×3 array)",
        "grid_offset":          "grid translation offset to map lattice indices to real-space coordinates",
        "grid_transform":       "grid transform matrix to map lattice indices to real-space coordinates (3x3 orthogonal matrix)",
    }

    _validators = {
        **(OptsBase._validators),
        "origin":               lambda v, d: as_Vect(v, name=d),
        "normal":               lambda v, d: as_Vect(v, name=d, is_norm=True),
        "theta0_axis":          lambda v, d: None if v is None else as_Vect(v, name=d, is_norm=True),
        "R_max":                lambda v, d: as_Number(v, name=d, value_range=(1e-6, np.inf)),
        "dr":                   lambda v, d: as_Number(v, name=d, value_range=(1e-6, np.inf)),
        "arc_dist":             lambda v, d: None if v is None else as_Number(v, name=d, value_range=(1e-6, np.inf)),
        "start_layer":          lambda v, d: as_Number(v, name=d, value_range=(0, np.inf), is_int=True),
        #!!! corner limit
        "grid_offset":          lambda v, d: as_Vect(v, name=d),
        "grid_transform":       lambda v, d: as_Tensor(v, (3, 3), name=d),
    }

    _DEFAULTS_FROZEN = MappingProxyType({
        **(OptsBase._DEFAULTS_FROZEN),
        "tag":                  "polar plane grid options",
        "theta0_axis":          None,
        "R_max":                5,
        "dr":                   0.5,
        "arc_dist":             None,
        "start_layer":          2,
        "corners_limit":        None,
        "grid_offset":          (0, 0, 0),
        "grid_transform":       np.diag((1, 1, 1)),
    })
    

class PlaneGridPolar(HostBase):
    
    __descriptions__ = {
        **dict(HostBase.__descriptions__),
        "_entity_grid": "Selected 3D grid points after applying transforms and optional bounding-box filtering (array of shape N×3)",
        "_entity_grid_all": "Complete 3D grid points before filtering, reshaped as (num1 × num2 × 3)",
        "_entity_polar": "The polar coordinates of points",
        "_calc_ring_offsets": "Cumulative offsets defining the start/end indices of each polar ring",
        "_calc_box_mask": "the flag indicating whether point in self._entity_grid_all is inside the corners limit",
        "_impl_field_ref": (
            "Quantity field evaluated on the 2D plane grid."
            "To assess it, use .field or ._impl_field."
        ),
        "_entity_fig_demo": "Diagnostic plot showing the generated 2D polar grid points.",

    }
    
    __slots__ = tuple(
            k for k, v in __descriptions__.items() 
            if not v.startswith("Property:") and k not in HostBase.__slots__
        )
    
    def __init__(self,
                 name: str | None = None,
                 name_replace: str = "polar grid",
                 opts: OptsPlaneGridPolar | None = None,
                 opts_defaults_override: Mapping[str, Any] | None = None,
                 **kwargs):
        
        super().__init__(
            OptsPlaneGridPolar,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs
            )
        
        object.__setattr__(self, '_entity_fig_demo', None)
        object.__setattr__(self, '_impl_field_ref', None)
        
        for name, value in {
            "origin": self.opts.origin,
            "normal": self.opts.normal,
        }.items():
            if value is UNSET:
                raise ValueError(
                    f"Missing required variable {name!r} to generate polar plane grid"
                )
        self.opts.act_finalize(defaults=self._opts_defaults)

        self._helper_commit_apply_opts()
        
        
    @logging_and_warning_decorator()
    def _helper_commit_apply_opts(self, logger=None, **kwargs):

        with self.opts._helper_internal_update():
            cover_value(self.opts,
                        is_allow_cover_target_set=True,
                        is_allow_unset_source=False,
                        **kwargs
                        )
         
        if self.opts.arc_dist is not None:
            arc_dist = self.opts.arc_dist
        else:
            arc_dist = self.opts.dr

        origin = self.opts.origin
        dr = self.opts.dr
        R_max = self.opts.R_max
        normal = self.opts.normal
        theta0_axis = self.opts.theta0_axis
        start_layer = self.opts.start_layer  # 0: include center; 1: start from first ring; ...

        if theta0_axis is not None:
            dot_product = normal @ theta0_axis
            if not np.isclose(dot_product, 0, atol=1e-8):
                old_theta0_axis = theta0_axis.copy()
                theta0_axis = theta0_axis - dot_product * normal
                theta0_axis /= np.linalg.norm(theta0_axis)
                logger.warning(
                    f"Invalid geometry: theta0_axis is not perpendicular to normal (dot product: {dot_product:.4e}). "
                    f"Projecting original theta0_axis {old_theta0_axis} onto the plane defined by normal {normal}. "
                    f"New orthonormal theta0_axis: {theta0_axis}."
                )
        else:
            from Nematics3D.general import rotation_matrix_from_vectors
            _rotation_matrix = rotation_matrix_from_vectors((0, 0, 1), normal)
            theta0_axis = _rotation_matrix @ np.array([1, 0, 0])
            logger.debug(
                f"theta0_axis not provided. Automatically generated a reference theta0_axis {theta0_axis} "
                f"perpendicular to normal {normal}."
            )

        e1 = theta0_axis
        e2 = np.cross(normal, e1)

        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        # ---- Generate rings ----
        points_list = []
        polar_list  = []
        ring_sizes  = []   # n_theta for each appended "block" (including center block if start_layer==0)

        # layer 0: center point
        if start_layer == 0:
            points_list.append(origin.copy()[None, :])          # (1,3)
            polar_list.append(np.array([[0.0, 0.0]]))           # (1,2)
            ring_sizes.append(1)

        # layers k>=1: rings at r_k = (k - 0.5) * dr
        k = max(1, int(start_layer))
        while True:
            r = (k - 0.5) * dr
            if r > R_max:
                break

            # Points per ring: approx equal arc length
            n_theta = int(np.round(2.0 * np.pi * r / arc_dist))
            n_theta = max(1, n_theta)

            phi = ((k - 1) * golden_angle) % (2.0 * np.pi)

            # Angles on this ring
            thetas = (2.0 * np.pi * np.arange(n_theta) / n_theta + phi) % (2.0 * np.pi)

            # Convert to 3D: origin + r*cos(theta)*e1 + r*sin(theta)*e2
            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)
            ring_points = origin + (r * cos_t)[:, None] * e1[None, :] + (r * sin_t)[:, None] * e2[None, :]

            points_list.append(ring_points)                                      # (n_theta,3)
            polar_list.append(np.column_stack([np.full(n_theta, r), thetas]))    # (n_theta,2)
            ring_sizes.append(n_theta)

            k += 1

        # ---- Flatten + ring offsets ----
        points = np.vstack(points_list)   # (N,3)
        polar  = np.vstack(polar_list)    # (N,2)

        ring_offsets = np.empty(len(ring_sizes) + 1, dtype=np.int64)
        ring_offsets[0] = 0
        ring_offsets[1:] = np.cumsum(ring_sizes, dtype=np.int64)  # last one is N
        
        points = apply_linear_transform(
            points, transform=self.opts.grid_transform, offset=self.opts.grid_offset
        )
        
        points_select, mask = select_grid_in_box(points, self.opts.corners_limit, is_return_mask=True)
        
        object.__setattr__(self, "_entity_grid_all", points)
        object.__setattr__(self, "_entity_grid", points_select)
        object.__setattr__(self, "_entity_polar", polar)
        object.__setattr__(self, "_calc_ring_offsets", ring_offsets)
        object.__setattr__(self, '_calc_box_mask', mask)
        
        if self._entity_fig_demo:
            self._entity_fig_demo['grid'].raw_coords = self._entity_grid
            self._entity_fig_demo['origin'].raw_coords = self.opts.origin
        
        if self.field:
            self.field._helper_commit()
            
        self._helper_trigger_sync_batch(**kwargs)
        
        
    def act_debug_plot(self,
                       opts_extent: OptsTube | None = None,
                       opts_points: OptsSphere | None = None,
                       opts_figure: OptsFigure | None = None,
                       opts_origin: OptsSphere | None = None,
                       **kwargs
                       ):
    
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_points is None:
            opts_points = OptsSphere()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_origin is None:
            opts_origin = OptsSphere()
            
        merge = merge_opts_all(
            {
                "figure_": opts_figure, 
                "point_": opts_points,
                "extent_": opts_extent,
                "origin_": opts_origin
            },
            kwargs, type(self).__name__)
        
        opts_figure = merge["figure_"]
        opts_points = merge["point_"]
        opts_extent = merge["extent_"]
        opts_origin = merge["origin_"]
        
        figure = PlotFigure(
            opts=opts_figure, 
            name=f"Diagnostic plot of plane {self.name!r}"
        )
        bulk = PlotSphere(
            coords=self._entity_grid, 
            opts=opts_points, 
            figure=figure,
            category="plane_grid_test", 
            name="grid"
            )
        PlotSphere(
            coords=self.opts.origin, 
            opts=opts_origin, 
            figure=figure,
            opts_defaults_override={
                "color": (1,0,0),
                "radius": 1.2*bulk._calc_radius[0]
            },
            category="plane_grid_test", 
            name="origin"
        )
        if self.opts.corners_limit is not None:
            PlotExtent(
                corners=self.opts.corners_limit, 
                opts=opts_extent, 
                figure=figure,
                category="plane_grid_test", 
                name="grid_extent"
            )
            
        object.__setattr__(self, "_entity_fig_demo", figure)
            
        return figure
     
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}, with normal={self.opts.normal} and origin={self.opts.origin}"
        return msg
    
    def __iter__(self):
        return iter(self._entity_grid)
    
    def __getitem__(self, idx):
        return self._entity_grid[idx]
    
    def __array__(self, dtype=None):
        arr = self._entity_grid
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
    
    def __call__(self):
        return self._entity_grid
        
        
    @property
    def field(self):
        ref = self._impl_field_ref
        return ref() if ref is not None else None
    
    _impl_field = field


