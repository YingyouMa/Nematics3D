import numpy as np
from typing import Optional
from dataclasses import dataclass, field, asdict
import os
import json

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
    as_Number,
    as_bool
)
from ..field import apply_linear_transform
from .visual.plot_figure import PlotFigure
from .visual.plot_tube import PlotTube, OptsTube
from .opts import merge_opts_all
from .smoothed_line import OptsSmooth, SmoothedLine

# extra attr

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
        
        # --- reserved for user extensions ---
        "extra": "Dictionary for user-defined extra attributes. "
                  "This is the only safe place for attaching arbitrary data "
                  "without modifying internal slots.",
    }
    
    __slots__ = tuple(__descriptions__.keys()) + ("_descriptions",)

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        inputValue = InputLine(),
        is_sorted: bool = False,
        logger=None,
        **kwargs
    ):
        
        self._descriptions = self.__class__.__descriptions__.copy()        
        
        inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[""]
        if inputValue.defect_indices is None:
            raise ValueError("No defects are input into disclination line")
        for k, v in asdict(inputValue).items():
            if k == "name":
                setattr(self, "name", v)
            else:
                setattr(self, f"_raw_{k}", v)
                
        logger.detail(f"Initializing the disclination line {self.name!r}")

        if self.name is None:
            self.name = "disclination_line"

        if is_sorted == False:
            logger.detail("Sorting defects by closest neighboring pairs.")
            self._raw_defect_indices = sort_line_indices(self._raw_defect_indices)

        self._raw_box_size_periodic_index = as_dimension_info(self._raw_box_size_periodic_index)

        logger.debug("Classifying line type by the distance between head and tail.")
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
        logger.debug(f"Disclination line {self.name!r} is of type {self._calc_end2end_category!r}")
        logger.detail(f"The first and end point are {self._raw_defect_indices[0]} and {self._raw_defect_indices[-1]}")

        self._calc_defect_num = np.shape(self._raw_defect_indices)[0]
        
        logger.detail('Calculateing the defects positions in real-space units.')
        self._calc_defect_coords = apply_linear_transform(
            self._raw_defect_indices,
            transform=self._raw_grid_transform,
            offset=self._raw_grid_offset,
        )
    
    @logging_and_warning_decorator()
    def act_smooth(
        self,
        opts: OptsSmooth = OptsSmooth(),
        padding_num: int = 50,
        head_padding_extra: int = 5,
        logger=None,
        **kwargs
    ) -> np.ndarray:
        """
        Smooth the defect line trajectory using Savitzky–Golay filtering and spline interpolation.
        
        For cross-type defect lines, periodic unwrapping is performed in two stages.
        
        Padding segments are first attached to both ends of the trajectory
        (`padding_num`) to avoid endpoint distortions during smoothing.
        
        A short prefix consisting of the padded head and an extra segment from
        the original trajectory (`head_padding_extra`) is then unwrapped in
        reverse direction. This step fixes the global periodic-image branch by
        anchoring the unwrapping on the original trajectory rather than on padding.
        
        Finally, the entire padded trajectory is unwrapped in forward direction
        to obtain a globally continuous curve for smoothing. The padded segments
        are removed after smoothing, and the result is shifted back into the box.


        Parameters
        ----------
        opts : Optssmooth, optional
            Options controlling the smoothing procedure.

        max_padding_num : int, default=50
            [Adaptive Smoothing Buffer] The maximum number of points to duplicate 
            from the trajectory ends for padding. 
            
            Logic:
            - If the trajectory is long: It uses `max_padding_num` to provide 
              sufficient context for the smoothing filter while maintaining 
              computational efficiency.
            - If the trajectory is short: It automatically caps the padding at 
              the total number of points available (N) to prevent out-of-bounds 
              errors and avoid biased smoothing caused by over-extrapolation.

        head_extra_unwrap_length : int, default=5
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
        
        opts.name = self.name
        logger.debug(f"Smoothing line ``{self.name}``, with type ``{self._calc_end2end_category}``")

        if self._calc_end2end_category == "loop":
            smooth_mode = "wrap"
            padding_num = 0
        elif self._calc_end2end_category == "cross":
            
            max_padding_num = as_Number(padding_num, is_int=True, value_range=(0, self._calc_defect_num))
            padding_num = min(max_padding_num, self._calc_defect_num)
            head_padding_extra = as_Number(head_padding_extra, is_int=True, value_range=(0, self._calc_defect_num))
            
            msg = f"line {self.name} is cross-type. It has to deal with periodic boundary condition to smooth this line. \n"
            msg += f"padding_num={padding_num}, head_padding_extra={head_padding_extra}. \n"
            msg += "You could change the value of padding_num by `padding_num` keywords in `act_smooth()`. Note that it will be automatically truncated to line length if it is longer."
            logger.debug(msg)
            
            indices_origin = self._raw_defect_indices.copy()
            tail = indices_origin[:padding_num].copy()
            head = indices_origin[-padding_num - 1 :]
            indices = np.concatenate([head, indices_origin, tail])

            logger.detail("Start the reverse unwrap for the beginning of this line.")
            indices[:padding_num+head_padding_extra] = unwrap_trajectory(
                indices[:padding_num+head_padding_extra],
                box_size_periodic=self._raw_box_size_periodic_index,
                is_reverse=True,
            )
            logger.detail("Start the whole unwrap.")
            indices = unwrap_trajectory(
                indices,
                box_size_periodic=self._raw_box_size_periodic_index,
                is_start_in_box=True,
            )
            
            logger.detail("Generating the points coordinates in real-space units.")
            coords = apply_linear_transform(
                indices,
                transform=self._raw_grid_transform,
                offset=self._raw_grid_offset,
            )

            smooth_mode = "interp"
        else:
            smooth_mode = "interp"
            padding_num = 0

        logger.debug("Start to smooth line")
        opts = merge_opts_all({"": opts}, kwargs, "SmoothedLine")[""]
        opts.mode = smooth_mode
        output = SmoothedLine(coords, opts=opts)
        
        result = output._entities[
            int(padding_num * output.opts.N_out_ratio) : int(
                (-padding_num - 1) * output.opts.N_out_ratio
            )
        ]
        logger.detail("Checking: shifting the entire trajectory so that the first point is inside the periodic box.")
        result = shift_to_box(result, self._raw_box_size_periodic_index)

        self._calc_defect_coords_smooth_obj = output
        self._calc_defect_coords_smooth = result

        return output._entities

    @logging_and_warning_decorator()
    def act_visualize(
        self,
        figure: PlotFigure | None = None,
        is_wrap: bool = True,
        is_smooth: bool = True,
        scalars_attr: str | None = None,
        opts: OptsTube | None = None,
        logger=None,
        **kwargs
    ) -> None:
        
        is_smooth = as_bool(
            is_smooth, 
            name="Whether visualize the smoothed line instead of original points",
            replace=False)

        logger.debug(f"Start to visualize line: {opts.name!r} with type ``{self._calc_end2end_category}``")

        if is_smooth:
            if hasattr(self, "_calc_defect_coords_smooth"):
                line_coords = self._calc_defect_coords_smooth
            else:
                msg = "The line has not been smoothed.\n"
                msg += "Use original data instead"
                logger.warning(msg)
                line_coords = self._calc_defect_coords.copy()
                is_smooth = False
        else:
            line_coords = self._calc_defect_coords.copy()

        if self._calc_end2end_category == "loop":
            logger.debug('Line {opts.name!r} is a loop. Closing the loop by appending the start point to the end.')
            line_coords = np.concatenate((line_coords, [line_coords[0]]))

        if not is_wrap:
            line_plot = PlotTube(
                line_coords,
                figure = figure,
                opts=opts,
                **kwargs
            )
        else:
            logger.debug('Start to deal with the periodic boundary condition')
            boundary_flag = boundary_periodic_size_to_flag(
                self._raw_box_size_periodic_index
            )
            logger.detail("Swtiching the point positions in real-space units back to lattice-grid indices and wrapping them.")
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
            
            logger.detail("Extracting the points at periodic boundaries.")
            diff = line_coords_origin[1:] - line_coords_origin[:-1]
            diff = np.linalg.norm(diff, axis=-1)
            end_list = np.where(diff > 1)[0] + 1
            end_list = np.concatenate([[0], end_list, [len(line_coords_origin)]])

            logger.detail("Switching to real-space units.")
            line_coords = apply_linear_transform(
                line_coords_origin,
                transform=self._raw_grid_transform,
                offset=self._raw_grid_offset,
            )

            logger.detail("Classifying the line into different segements due to periodic boundary conditions.")
            line_index = np.ones(len(line_coords))

            for i in range(len(end_list) - 1):
                line_index[end_list[i] : end_list[i + 1]] = i

            logger.debug('Done!')
            
            line_plot = PlotTube(
                line_coords,
                line_index=line_index,
                figure=figure,
                opts=opts,
                name=self.name,
                **kwargs
            )

        return line_plot
    
    def act_extra_description(self, attr: str, desc: str):
        desc = as_str(desc, name="The description of ``extra`` attribute for DisclinationLine")
        self._descriptions["extra"] = desc
    
    @logging_and_warning_decorator()
    def act_log_parameters(self, is_return: bool = False, logger=None) -> None:
        """
        Log parameters for inspection.

        This is the standard logging interface used in this library, which
        can be redirected to console or to a file depending on the logger
        configuration and the behavior of ``logging_and_warning_decorator``.

        All attributes listed in ``__descriptions__`` are included,
        formatted in a single log entry with a clear separator.
        """
        lines = []
        lines.append("-------------- DisclinationLine Parameters --------------")
        
        lines.append(f"[{self.name}] parameters:")
        for attr in self.__slots__:
            if attr == "_descriptions":
                continue
            if attr == "extra" and not hasattr(self, attr):
                continue
            desc = self._descriptions.get(attr, "(no description)")
            value = getattr(self, attr, None)
            lines.append(f"  {attr}: {value!r}  # {desc}")
        lines.append("-----------------------------------------------------")

        msg = "\n".join(lines)

        if is_return:
            return msg
        else:
            logger.info(msg)
            
    @logging_and_warning_decorator()
    def act_save(
        self,
        dirpath: Optional[str] = None,
        logger=None,
    ) -> None:
        
        # ---------- sanitize inputs ----------

        if dirpath is None:
             dirpath = os.path.join("save", "disclination_line", str(self.name))

        dirpath = as_str(
            dirpath,
            name=f"the folder to store disclination line ``{getattr(self, 'name', None)}``"
        )

        logger.debug(f"Start to save disclination line ``{self.name}`` into {dirpath}")

        # ---------- ensure folder ----------
        os.makedirs(dirpath, exist_ok=True)

        # ---------- compose JSON payload ----------
        json_payload = {
            "raw": {
                "box_size_periodic_index": np.asarray(getattr(self, "_raw_box_size_periodic_index", None)).tolist(),
                "grid_offset": np.asarray(getattr(self, "_raw_grid_offset", None)).tolist(),
                "grid_transform": np.asarray(getattr(self, "_raw_grid_transform", None)).tolist()
            },
            "metadata": {
                "name": getattr(self, "name", None),
                "category": getattr(self, "_calc_end2end_category", None),
                "defect_num": int(getattr(self, "_calc_defect_num", 0)),
                "box_size_periodic_coord": getattr(self, "_calc_box_size_periodic_coord", None).tolist()
            },
            "smooth": {
                "exists": hasattr(self, "_calc_defect_coords_smooth"),
            },
        }
        
        if hasattr(self, "_calc_defect_coords_smooth"):
            logger.debug("Start to store smoothed coordinates")
            smooth_dirpath = dirpath + "/smoothed_line/"
            self._calc_defect_coords_smooth_obj.act_save(dirpath=smooth_dirpath)

        json_path = os.path.join(dirpath, "info.json")
        logger.debug(f"Dtart to write JSON to {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=4)

        # ---------- compose NPZ arrays ----------
        npz_path = os.path.join(dirpath, "data.npz")
        logger.debug(f"Start to write NPZ to {npz_path}")
        arrays = {}
        arrays["defect_indices"] = np.asarray(self._raw_defect_indices)
        arrays["defect_coords"] = np.asarray(self._calc_defect_coords)
        if hasattr(self, "_calc_defect_coords_smooth"):
            arrays["defect_coords_smooth"] = np.asarray(self._calc_defect_coords_smooth)
        
        np.savez_compressed(npz_path, **arrays)

    @classmethod
    @logging_and_warning_decorator()
    def act_load(
        cls,
        dirpath: str,
        logger=None,
    ) -> "DisclinationLine":

        dirpath = as_str(dirpath, name="the folder to load disclination line")

        json_path = os.path.join(dirpath, "info.json")
        npz_path = os.path.join(dirpath, "data.npz")
        logger.debug(f"Start to load DisclinationLine from {json_path} and {npz_path}")

        if not os.path.exists(json_path) or not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing required files: {json_path} / {npz_path}")

        # ---------- read JSON ----------
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        name = meta.get("metadata", {}).get("name", None)
        category = meta.get("metadata", {}).get("category")
        raw = meta.get("raw", {}) or {}

        # ---------- read NPZ ----------
        data = np.load(npz_path, allow_pickle=True)
        if "defect_indices" not in data.files:
            raise ValueError("NPZ missing required array 'defect_indices'.")

        defect_indices = np.asarray(data["defect_indices"])
        defect_coords_smooth = (
            np.asarray(data["defect_coords_smooth"]) if "defect_coords_smooth" in data.files else None
        )

        # ---------- reconstruct InputLine ----------
        box_size_periodic_index = np.asarray(raw.get("box_size_periodic_index"))
        grid_offset = np.asarray(raw.get("grid_offset"))
        grid_transform = np.asarray(raw.get("grid_transform"))

        input_value = InputLine(
            defect_indices=defect_indices,
            box_size_periodic_index=box_size_periodic_index,
            grid_offset=grid_offset,
            grid_transform=grid_transform,
            name=name,
        )

        # ---------- construct object (will compute derived fields) ----------
        obj = cls(inputValue=input_value, is_sorted=True)
        object.__setattr__(obj, "_calc_end2end_category", category)

        if defect_coords_smooth is not None:
            logger.debug("Start to load smoothed coordinates")
            smooth_dirpath = dirpath + "/smoothed_line/"
            object.__setattr__(obj, "_calc_defect_coords_smooth", defect_coords_smooth)
            smoothObj = SmoothedLine.act_load(smooth_dirpath, logger=logger)
            object.__setattr__(obj, "_calc_defect_coords_smooth_obj", smoothObj)

        return obj
            
    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.act_log_parameters(is_return=True)    

    def __len__(self) -> int:
        return self._calc_defect_num        
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}(name={self.name!r}), has {self._calc_defect_num} defect points with type {self._calc_end2end_category}"
        return msg
    
    def __iter__(self):
        return iter(self._raw_defect_indices)
    
    def __getitem__(self, idx):
        return self._raw_defect_indices[idx]
    
    def __array__(self, dtype=None):
        arr = self._raw_defect_indices
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
    
    @logging_and_warning_decorator()
    def act_copy(self, logger=None) -> "DisclinationLine":
        """
        Create a deep copy of this DisclinationLine.

        - Rebuilds a new instance from InputLine with copied raw inputs.
        - Overwrites derived arrays with value-equal copies to avoid floating diffs.
        - If smoothed results exist, copies both the array and the SmoothedLine object.

        Returns
        -------
        DisclinationLine
            An independent copy of the current object.
        """
        logger.debug(f"Start to copy DisclinationLine(name={self.name!r}")
        
        defect_indices = np.array(self._raw_defect_indices, copy=True)
        box_size_periodic_index = np.array(self._raw_box_size_periodic_index, copy=True)
        grid_offset = np.array(self._raw_grid_offset, copy=True)
        grid_transform = np.array(self._raw_grid_transform, copy=True)
        name = getattr(self, "name", None)

        input_value = InputLine(
            defect_indices=defect_indices,
            box_size_periodic_index=box_size_periodic_index,
            grid_offset=grid_offset,
            grid_transform=grid_transform,
            name=name,
        )

        new_obj = type(self)(inputValue=input_value, is_sorted=True)
        object.__setattr__(new_obj, "_calc_end2end_category", self._calc_end2end_category)

        logger.debug("Start to copy smooth line")
        if hasattr(self, "_calc_defect_coords_smooth"):
            object.__setattr__(
                new_obj, "_calc_defect_coords_smooth",
                np.array(self._calc_defect_coords_smooth, copy=True)
            )
        if hasattr(self, "_calc_defect_coords_smooth_obj"):
            smooth_copy = self._calc_defect_coords_smooth_obj.act_copy()
            object.__setattr__(new_obj, "_calc_defect_coords_smooth_obj", smooth_copy)

        return new_obj
    
    @logging_and_warning_decorator()
    def __eq__(self, other, logger=None) -> bool:
        """
        Compare equality with another DisclinationLine object.

        Two DisclinationLine objects are considered equal iff all attributes
        in __slots__ are equal (deep equality for numpy arrays; for the
        smoothed-line object, delegate to its __eq__ when available).

        Parameters
        ----------
        other : object
            Another object to compare against.

        logger : logging.Logger, optional
            Logger instance to record differences when not equal.

        Returns
        -------
        bool
            True if all attributes match, False otherwise.
        """
        if not isinstance(other, DisclinationLine):
            logger.info("The other variable is not class DisclinationLine")
            return False

        diffs = []

        for attr in self.__slots__:
            has1 = hasattr(self, attr)
            has2 = hasattr(other, attr)

            if has1 != has2:
                diffs.append(f"{attr}: presence mismatch (self={has1}, other={has2})")
                continue

            if not has1 and not has2:
                continue

            v1 = getattr(self, attr)
            v2 = getattr(other, attr)

            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not (isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray) and np.array_equal(v1, v2)):
                    s1 = None if not isinstance(v1, np.ndarray) else np.shape(v1)
                    s2 = None if not isinstance(v2, np.ndarray) else np.shape(v2)
                    diffs.append(f"{attr}: arrays differ (self.shape={s1}, other.shape={s2})")
                continue

            if attr == "_calc_defect_coords_smooth_obj":
                equal = (v1 == v2)
                if not equal:
                    diffs.append(f"{attr}: SmoothedLine objects differ")
                continue

            if v1 != v2:
                diffs.append(f"{attr}: self={v1!r}, other={v2!r}")

        if diffs:
            logger.info("DisclinationLine objects are not equal.\nDifferences:\n" + "\n".join(diffs))
            return False
        else:
            logger.info("DisclinationLine objects are equal.")
            return True
    
    @property
    def smooth(self):
        return self._calc_defect_coords_smooth
    
    @property
    def smooth_obj(self):
        return self._calc_defect_coords_smooth_obj
    
    @property
    def category(self):
        return self._calc_end2end_category

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
