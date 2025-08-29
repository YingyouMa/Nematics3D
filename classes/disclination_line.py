import numpy as np
from typing import Optional, Tuple
from dataclasses import replace, dataclass, field, asdict

from ..general import sort_line_indices  # , get_plane, get_tangent
from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import (
    Vect,
    as_Vect,
    Tensor,
    as_Tensor,
    DefectIndex,
    as_DefectIndex,
    DimensionPeriodicInput,
    as_dimension_info,
    boundary_periodic_size_to_flag,
    as_str,
    as_Number
)
from ..field import apply_linear_transform
from .visual_mayavi.plot_tube import PlotTube, OptsTube
from .opts import merge_opts_all
from .smoothed_line import OptsSmooth

@dataclass(slots=True)
class InputLine:
    defect_indices: Optional[DefectIndex] = None
    box_size_periodic_index: DimensionPeriodicInput = False
    grid_offset: Vect(3) = (0,0,0)
    grid_transform: Tensor((3, 3)) = field(default_factory=lambda: np.eye(3))
    name: Optional[str] = None

    __descriptions__ = {
        "defect_indices": "indices of defect points in the Q array",
        "box_size_periodic_index": "the maximum index of each index in the Q array (finite values for periodic boundary conditions and np.inf for non-periodic)",
        "grid_offset": "grid translation offset to map lattice indices of Q array to real-space coordinates",
        "grid_transform": "grid transform matrix to map lattice indices of Q array to real-space coordinates (3x3)",
        "name": "name identifier of this line",
    }

    _validators = {
        "defect_indices": lambda self, v: as_DefectIndex(v, is_return_row=True),
        "box_size_periodic_index": lambda self, v: as_dimension_info(v, name=self.__descriptions__["box_size_periodic_index"]),
        "grid_offset": lambda self, v: as_Vect(
            v, name=self.__descriptions__["grid_offset"]
        ),
        "grid_transform": lambda self, v: as_Tensor(
            v, (3, 3), name=self.__descriptions__["grid_transform"]
        ),
        "name": lambda self, v: as_str(v, name="Name of Q field")
    }



class DisclinationLine:
    """
    Internal representation of a single disclination line detected
    from the Q-tensor field. 

    This class is not intended for direct user interaction.
    Instances are created and managed internally by the Q-field object.
    All input data (defect indices, periodic box info, grid transform, etc.)
    are provided by the parent Q-field and should not be modified manually.

    Responsibilities
    ----------------
    - Store the raw lattice indices of a defect line and the corresponding
      grid/box information from the Q-field.
    - Classify the line as one of:
        * ``"loop"`` — closed loop (start and end coincide)
        * ``"cross"`` — passes across periodic boundaries
        * ``"seg"`` — open segment
    - Compute derived quantities such as:
        * number of defect points
        * real-space coordinates of the defect line
        * transformed box size in real space
    - Provide smoothing utilities via :meth:`act_smooth`
      (Savitzky–Golay filtering + spline interpolation).
    - Provide visualization hooks via :meth:`act_visualize`,
      which returns a :class:`PlotTube` object. 

    Attributes
    ----------
    See :attr:`DisclinationLine.__descriptions__` for a complete list of
    raw inputs (``_raw_*``) and derived results (``_calc_*``).

    Methods
    -------
    act_smooth(opts=OptsSmooth(), **kwargs)
        Smooth the defect line trajectory with optional padding/unwrapping.
        Returns the smoothed coordinates and stores the associated
        :class:`SmoothedLine` object internally.

    act_visualize(is_wrap=True, is_smooth=True, scalars=None, opts=OptsTube())
        Visualize the defect line as tubes via Mayavi.
        Returns a :class:`PlotTube` object (not stored internally).

    Notes
    -----
    - This class is considered an **internal helper**; do not construct
      or modify it directly.
    """

    __descriptions__ = {
        # ========== user-facing ==========
        "name": "Name identifier of this disclination line",

        # ========== raw (copied directly from InputLine or computed) ==========
        "_raw_defect_indices": "Lattice indices of defect points forming the line (array of shape N×3)",
        "_raw_box_size_periodic_index": "Box size along each dimension in index space (finite for periodic boundaries, np.inf for non-periodic)",
        "_raw_grid_offset": "Grid translation offset mapping lattice indices to real-space coordinates (3-vector)",
        "_raw_grid_transform": "Grid transformation matrix (3×3) mapping lattice indices to real-space coordinates",

        # ========== calc (derived quantities) ==========
        "_calc_end2end_category": "Category of line ends: 'loop' (closed loop), 'cross' (wraps across boundary), or 'seg' (open segment)",
        "_calc_defect_num": "Number of defect points forming this line (integer)",
        "_calc_defect_coords": "Real-space coordinates of defect line (array of shape N×3)",
        "_calc_box_size_periodic_coord": "Box size expressed in real-space coordinates (3-vector, transformed from indices)",
        "_calc_defect_coords_smooth_obj": "SmoothedLine object generated by act_smooth (stores smoothing details)",
        "_calc_defect_coords_smooth": "Real-space coordinates of smoothed defect line (array of shape N×3)",
    }

    def __init__(
        self,
        inputValue = InputLine(),
        is_sorted: bool = True,
        **kwargs
    ):
        inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[""]
        if inputValue.defect_indices is None:
            raise ValueError("No defects are input into disclination line")
        for k, v in asdict(inputValue).items():
            if k == "name":
                setattr(self, "name", v)
            else:
                setattr(self, f"_raw_{k}", v)

        if is_sorted == False:
            self._raw_defect_indices = sort_line_indices(self._raw_defect_indices)

        self._raw_box_size_periodic_index = as_dimension_info(self._raw_box_size_periodic_index)

        if np.linalg.norm(self._raw_defect_indices[0] - self._raw_defect_indices[-1]) == 0:
            self._calc_end2end_category = "loop"
            self._raw_defect_indices = self._raw_defect_indices[:-1]
        else:
            defect1 = self._raw_defect_indices[0].copy()
            defect2 = self._raw_defect_indices[-1].copy()
            defect1 = np.where(
                self._raw_box_size_periodic_index == np.inf,
                defect1,
                defect1 % self._raw_box_size_periodic_index,
            )
            defect2 = np.where(
                self._raw_box_size_periodic_index == np.inf,
                defect2,
                defect2 % self._raw_box_size_periodic_index,
            )
            if np.linalg.norm(defect1 - defect2) == 0:
                self._calc_end2end_category = "cross"
                self._raw_defect_indices = self._raw_defect_indices[:-1]
            else:
                self._calc_end2end_category = "seg"
                self._raw_defect_indices = self._raw_defect_indices

        self._calc_defect_num = np.shape(self._raw_defect_indices)[0]

        self._calc_defect_coords = apply_linear_transform(
            self._raw_defect_indices,
            transform=self._raw_grid_transform,
            offset=self._raw_grid_offset,
        )
        self._calc_box_size_periodic_coord = apply_linear_transform(
            self._raw_box_size_periodic_index,
            transform=self._raw_grid_transform,
            offset=self._raw_grid_offset,
        )

    def act_smooth(
        self,
        opts: OptsSmooth = OptsSmooth(),
        padding_length: int = 50,
        head_padding_extra: int = 25,
        **kwargs
    ) -> np.ndarray:
        """
        Smooth the defect line trajectory using Savitzky–Golay filtering and spline interpolation.

        This method performs trajectory unwrapping (for periodic boundary conditions),
        optional head/tail padding to reduce boundary artifacts, and curve smoothing.

        Parameters
        ----------
        opts : Optssmooth, optional
            Options controlling the smoothing procedure.

        tail_length : int, default=50
            Number of lattice points to duplicate from the **head and tail** of the
            trajectory as padding. These duplicated points provide extra context
            for the smoothing filter to reduce boundary artifacts. The padding
            is trimmed away after smoothing.

        head_extra_unwrap_length : int, default=25
            Extra number of points at the **head** to unwrap before smoothing.
            This helps stabilize unwrapping when the trajectory crosses periodic
            boundaries and reduces artifacts near the trajectory start.

        **kwargs :
            Additional keyword arguments to override fields in `opts`.

        Returns
        -------
        np.ndarray
            smoothed trajectory coordinates, cropped to exclude the
            head/tail padding.
        
        Notes
        -----
        - If the defect line category is `"loop"`, smoothing is performed in wrap mode
        without padding.
        - If the category is `"cross"`, the trajectory is unwrapped across the box
        boundaries, padded with head/tail segments, smoothed, and then cropped back.
        - Otherwise, smoothing is performed in simple interpolation mode.
        """

        from ..field import unwrap_trajectory, shift_to_box
        from .smoothed_line import SmoothedLine

        coords = self._calc_defect_coords.copy()

        if self._calc_end2end_category == "loop":
            smooth_mode = "wrap"
            padding_length = 0
        elif self._calc_end2end_category == "cross":
            padding_length = as_Number(padding_length, is_int=True, value_range=(0, self._calc_defect_num))
            head_padding_extra = as_Number(head_padding_extra, is_int=True, value_range=(0, self._calc_defect_num))
            indices_origin = self._raw_defect_indices.copy()
            tail = indices_origin[:padding_length].copy()
            head = indices_origin[-padding_length - 1 :]
            indices = np.concatenate([head, indices_origin, tail])

            indices[:padding_length+head_padding_extra] = unwrap_trajectory(
                indices[:padding_length+head_padding_extra],
                box_size_periodic=self._raw_box_size_periodic_index,
                is_reverse=True,
            )
            indices = unwrap_trajectory(
                indices,
                box_size_periodic=self._raw_box_size_periodic_index,
                is_start_in_box=True,
            )

            coords = apply_linear_transform(
                indices,
                transform=self._raw_grid_transform,
                offset=self._raw_grid_offset,
            )

            smooth_mode = "interp"
        else:
            smooth_mode = "interp"
            padding_length = 0

        opts = merge_opts_all({"": opts}, kwargs, "SmoothedLine")[""]
        opts.mode = smooth_mode
        output = SmoothedLine(coords, opts=opts)

        result = output._entities[0][
            int(padding_length * output.opts_N_out_ratio) : int(
                (-padding_length - 1) * output.opts_N_out_ratio
            )
        ]
        result = shift_to_box(result, self._raw_box_size_periodic_index)

        self._calc_defect_coords_smooth_obj = output
        self._calc_defect_coords_smooth = result

        return output._entities[0]

    @logging_and_warning_decorator()
    def act_visualize(
        self,
        is_wrap: bool = True,
        is_smooth: bool = True,
        scalars: Optional[np.ndarray] = None,
        opts=OptsTube(),
        logger=None,
    ) -> None:
        """
        Visualize the defect line.

        Parameters
        ----------
        is_wrap : bool, optional
            Whether to apply periodic boundary wrapping to the defect line.
            Default is True.

        is_smooth : bool, optional
            Whether to use the smoothed version of the defect line.
            Default is True.

        scalars : np.ndarray, optional
            Optional scalar values for each vertex.
            (enables gradient coloring). If provided, overrides 'color'.

        opts : OptsTube, optional
            Options controlling properties of visualized tubes.
            See :attr:`OptsTube.__descriptions__` for definitions.
        """

        if not isinstance(is_smooth, bool):
            raise TypeError(
                f"is_smooth must be a boolean value. Got {is_smooth} instead."
            )

        self.opts = opts
        logger.debug(f"Start to visualize line: {self.opts.name}")

        if is_smooth:
            if hasattr(self, "_calc_defect_coords_smooth"):
                line_coords = self._calc_defect_coords_smooth
            else:
                msg = ">>> The line has not been smoothed\n"
                msg += ">>> Use original data instead"
                logger.warning(msg)
                line_coords = self._calc_defect_coords.copy()
        else:
            line_coords = self._calc_defect_coords.copy()

        if self._calc_end2end_category == "loop":
            line_coords = np.concatenate((line_coords, [line_coords[0]]))
            if scalars is not None:
                scalars = np.concatenate((scalars, [scalars[0]]))

        line_coords_all = [line_coords]

        if not is_wrap:
            scalars_all = [scalars]
            line_plot = PlotTube(
                line_coords_all,
                scalars_all=scalars_all,
                opts=opts,
                logger=logger,
            )
        else:
            boundary_flag = boundary_periodic_size_to_flag(
                self._raw_box_size_periodic_index
            )
            line_coords_origin = apply_linear_transform(
                line_coords,
                transform=np.linalg.inv(self._raw_grid_transform),
                offset=-self._raw_grid_offset,
            )

            line_coords_origin = np.where(
                boundary_flag,
                line_coords_origin % self._raw_box_size_periodic_index,
                line_coords_origin,
            )
            diff = line_coords_origin[1:] - line_coords_origin[:-1]
            diff = np.linalg.norm(diff, axis=-1)
            end_list = np.where(diff > 1)[0] + 1
            end_list = np.concatenate([[0], end_list, [len(line_coords_origin)]])

            line_coords = apply_linear_transform(
                line_coords_origin,
                transform=self._raw_grid_transform,
                offset=self._raw_grid_offset,
            )

            coords_all = []
            scalars_all = []

            for i in range(len(end_list) - 1):
                coords_all.append(line_coords[end_list[i] : end_list[i + 1]])
                if scalars is not None:
                    scalars_all.append(scalars[end_list[i] : end_list[i + 1]])
                else:
                    scalars_all.append(None)

            line_plot = PlotTube(
                coords_all,
                scalars_all=scalars_all,
                opts=opts,
                logger=logger,
            )

        return line_plot

    # def update_norm(self):
    #     self._norm = get_plane(self._calc_defect_coords)
    #     return self._norm

    # def update_center(self):
    #     self._center = np.average(self._raw_defect_indices, axis=0)
    #     return self._center

    # def update_rotation(self, n, num_shell=1, method='plane'):
    #     self._Omega = defect_rotation(self._raw_defect_indices, n,
    #                                   num_shell=num_shell, method=method, box_size_periodic=self._box_size_periodic)
    #     return self._Omega

    # def update_gamma(self, n=0, num_shell=1):

    #     if hasattr(self, '_Omega'):
    #         Omega = self._Omega
    #     else:
    #         Omega = self.update_rotation(n, num_shell=num_shell)

    #     if hasattr(self, '_norm'):
    #         norm = self._norm
    #     else:
    #         norm = self.update_norm()

    #     norm = np.broadcast_to(norm, (self._calc_defect_num,3))
    #     self._gamma = np.arccos(np.abs(np.einsum('ia, ia -> i', norm, Omega))) / np.pi * 180

    #     return self._gamma

    # def update_geometry(self, is_smooth=True):

    #     if is_smooth:
    #         if hasattr(self, '_defect_coords_smooth'):
    #             if self._calc_defect_coords_smooth_obj._N_out_ratio == 1:
    #                 points = self._calc_defect_coords_smooth
    #             else:
    #                 print('There are more points in the smooth line')
    #                 print('Start to re-smooth it with N_out_ratio=1')
    #                 print(f'window_length={self._calc_defect_coords_smooth_obj._window_length}')
    #                 print(f'order={self._calc_defect_coords_smooth_obj._order}')
    #                 print(f'mode={self._calc_defect_coords_smooth_obj._mode}')

    #                 points = smoothedLine(self._calc_defect_coords,
    #                                         window_length=self._calc_defect_coords_smooth_obj._window_length,
    #                                         order=self._calc_defect_coords_smooth_obj._order,
    #                                         N_out_ratio=1,
    #                                         mode=self._calc_defect_coords_smooth_obj._mode,
    #                                         is_keep_origin=False)._output
    #                 print('Done!')

    #         else:
    #             print('The line has not been smoothed')
    #             print('Use original data instead')
    #             points = self._calc_defect_coords
    #     else:
    #         points = self._calc_defect_coords

    #     is_periodic = self._calc_end2end_category == 'loop'

    #     tangents = get_tangent(points, is_periodic=is_periodic, is_norm=False)
    #     tangents_size = np.linalg.norm(tangents, axis=1, keepdims=True)
    #     tangents = tangents / tangents_size

    #     dT_ds = get_tangent(tangents, is_periodic=is_periodic, is_norm=False)
    #     dT_ds_size = np.linalg.norm(dT_ds, axis=1, keepdims=False)
    #     curvatures = dT_ds_size / tangents_size[:,0]

    #     length = np.sum(tangents_size, axis=0)[0]

    #     self._tangent = tangents
    #     self._curvature = curvatures
    #     self._length = length

    # def update_beta(self, n=0):

    #     if hasattr(self, '_Omega'):
    #         Omega = self._Omega
    #     else:
    #         Omega = self.update_rotation(n)

    #     if not hasattr(self, '_tangent'):
    #         self.update_geometry()
    #     tangent = self._tangent

    #     self._beta = np.arccos(np.einsum('ia, ia -> i', tangent, Omega)) / np.pi * 180

    #     return self._beta
