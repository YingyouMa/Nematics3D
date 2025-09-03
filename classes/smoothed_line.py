import numpy as np
from typing import Optional, Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from dataclasses import dataclass, asdict      
import os
import json

from ..logging_decorator import logging_and_warning_decorator
from .opts import merge_opts_all
from ..datatypes import Number, as_Number, as_str, ColorRGB, as_ColorRGB, Vect, as_Vect, as_bool


@dataclass(slots=True)
class OptsSmooth:
    window_ratio: Optional[Number] = None
    window_length: Optional[Number] = 41
    order: Number = 3
    N_out_ratio: Number = 3.0
    mode: Literal["interp", "wrap"] = "interp"
    min_line_length: int = 50
    name: str = "smoothed_line"
    is_window_warning: bool = True

    __descriptions__ = {
        "window_ratio": "window ratio for smoothing: line_length / window_length",
        "window_length": "explicit window length for smoothing",
        "order": "smoothing polynomial order",
        "N_out_ratio": "ratio between output and input #points in smoothing",
        "mode": "smoothing mode (interp or wrap)",
        "min_line_length": "minimum line length to be smoothed",
        "name": "name identifier of smooth options",
        "is_window_warning" : "whether present the warning when window_length and window_ratio are both input"
    }

    _validators = {
        "window_ratio": lambda self, v: (
            None
            if v is None
            else as_Number(v, name=self.__descriptions__["window_ratio"])
        ),
        "window_length": lambda self, v: (
            None
            if v is None
            else as_Number(v, name=self.__descriptions__["window_length"])
        ),
        "order": lambda self, v: as_Number(v, name=self.__descriptions__["order"]),
        "N_out_ratio": lambda self, v: as_Number(
            v, name=self.__descriptions__["N_out_ratio"]
        ),
        "mode": lambda self, v: (
            v
            if v in ("interp", "wrap")
            else (_ for _ in ()).throw(
                ValueError(
                    f"{self.__descriptions__['mode']} must be 'interp' or 'wrap', got {v!r}"
                )
            )
        ),
        "min_line_length": lambda self, v: as_Number(
            v, name=self.__descriptions__['min_line_length'], is_int = True
            ),
        "name": lambda self, v: as_str(v, name=self.__descriptions__["name"]),
        "is_window_warning": lambda self, v: v if isinstance(v, bool) else (_ for _ in ()).throw(
            TypeError(f"{self.__descriptions__['is_window_warning']} must be a bool, got {v}")
        )
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)



class SmoothedLine:
    """
    Smooth and resample a polyline using Savitzky–Golay filtering
    and parametric B-spline interpolation.

    Workflow
    --------
    1. Apply **Savitzky–Golay filter** to locally smooth the input coordinates.
    2. Perform **parametric B-spline interpolation** (`scipy.interpolate.splprep`
       with ``s=0``) on the smoothed points.
    3. Evaluate the spline at a uniformly spaced parameter grid (`splev`)
       to produce a resampled output line with higher or lower resolution.

    Parameters
    ----------
    line_coord_input : np.ndarray
        Input line coordinates of shape (N, D), where N is the number of points
        and D is the dimension (2D or 3D typically).

    opts : OptsSmooth, optional
        Options controlling the smoothing and resampling procedure.
        See :attr:`OptsSmooth.__descriptions__` for definitions.

    logger : logging.Logger, optional
        Logger instance for warnings and information messages.
        If None, falls back to global logging configuration.

    **kwargs
        Extra keyword arguments to override fields in `opts`.
        Keys must match attributes of :class:`OptsSmooth`.

    Attributes
    ----------
    See :attr:`smoothedLine.__descriptions__` for a full list and explanation
    of attributes (including both internal state such as ``_raw_coord`` and
    mirrored options such as ``opts_window_length``).

    Methods
    -------
    _helper_apply_smooth(opts, logger=None)
        Internal method that performs smoothing and resampling.
        Not intended for direct user calls. Use :meth:`act_commit` or re-initialize instead.

    act_commit()
        Commit changes to options and reapply smoothing.

    act_log_parameters()
        Log or return a formatted summary of parameters and results.

    act_visualize()
        Visualize smoothed lines by points    

    Python Special Methods
    ----------------------
    - ``len(line)`` → number of output points
    - ``iter(line)`` → iterate over smoothed points
    - ``line[i]`` → get the i-th point
    - ``np.array(line)`` → convert to NumPy array of points
    - ``str(line)`` → formatted summary of parameters. (e.g., ``print(line)``)
    - ``repr(line)`` → short identifier for debugging. (e.g., just type ``line`` in an interactive shell)
    - ``with line: ...`` → context manager for safe temporary option changes

    Notes
    -----
    - If both ``window_length`` and ``window_ratio`` are provided, ``window_ratio``
      will be IGNORED. Warnings depend on the `is_window_warning` flag in :class:`OptsSmooth`.
    - Option attributes (prefixed with ``opts_``) are implemented as properties
      and automatically update final result when changed.
    - For convenience, during user assignment the ``opts_`` prefix is optional:
      e.g. ``tube.order = 4`` is automatically redirected to
      ``tube.opts_order = 4``.  
    """

    __descriptions__ = {
        "name": "Name identifier of this line object",

        # --- internal states ---
        "_raw_coord": "Raw input line coordinates (shape: N x D)",
        "_calc_N_init": "Number of input points (before smoothing)",
        "_calc_N_out": "Number of output points (after smoothing)",
        "_entities": "Whose first element is smoothed output coordinates (shape: M x D)",
        "_state_is_smoothed": "Boolean flag indicating whether smoothing was applied",
        "_initializing": (        
            "Temporary flag used only during __init__ to bypass __setattr__ checks. "
            "Prevents infinite recursion when setting attributes in constructor. "
            "Should be set True at the start of __init__, and deleted (del self._initializing) "
            "at the end. Not intended for user access."
        ),
        "_backup_opts": "only used in __enter__ and __exit__, which helps users modify options" ,

        # ==== options mirrored onto the instance ====
        "opts_window_ratio": "Ratio used to compute window_length if not explicitly provided",
        "opts_window_length": "Explicit smoothing window length (overrides window_ratio if set)",
        "opts_order": "Polynomial order of Savitzky–Golay filter",
        "opts_N_out_ratio": "Ratio between output and input number of points",
        "opts_mode": "Smoothing mode (either 'interp' or 'wrap')",
        "opts_min_line_length": "Minimum line length required to apply smoothing",
        "opts_is_window_warning": "Whether present the warning when window_length and window_ratio are both input",
        "_opts_all": "The dataclass project to store all options values"
    }

    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator()
    def __init__(
        self,
        line_coord_input: np.ndarray,
        opts: OptsSmooth = OptsSmooth(),
        logger=None,
        **kwargs,
    ):
        
        line_coord_input = np.asarray(line_coord_input)
        if line_coord_input.ndim != 2:
            raise ValueError("line_coord_input for smoothing must be a 2D array of shape (N, D)")

        # We deliberately use object.__setattr__ here to bypass the custom __setattr__.
        # This ensures that internal state variables (e.g., _initializing, _entities,
        # _state_is_smoothed, etc.) can be assigned without triggering the validation
        # or auto-commit logic of __setattr__. (same below)
        object.__setattr__(self, "_initializing", True)

        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(self, "_opts_all", opts)

        object.__setattr__(self, "_raw_coord", line_coord_input)
        object.__setattr__(self, "_calc_N_init", len(self._raw_coord))

        self._helper_apply_smooth(self._opts_all, logger=logger)

        del self._initializing

    @logging_and_warning_decorator()
    def _helper_apply_smooth(self, opts, logger=None):

        for k, v in asdict(opts).items():
            if k == "name":
                object.__setattr__(self, "name", v)
            else:
                object.__setattr__(self, f"opts_{k}", v)
                
        if len(self._raw_coord) < self.opts_min_line_length:
            object.__setattr__(self, "_state_is_smoothed", False)
            logger.warning(
                f"{self.name!r} is not smoothed, because its length {len(self._raw_coord)} is shorter than the minum length {self.opts_min_line_length}."
            )
            object.__setattr__(self, "_entities", [self._raw_coord])
        else:
            object.__setattr__(self, "_state_is_smoothed", True)

            if self.opts_window_length is None:
                if self.opts_window_ratio is None:
                    raise ValueError("No input for smoothing window length!")
                self.opts_window_length = (
                    int(self._calc_N_init / self.opts_window_ratio / 2) * 2 + 1
                )
                self.opts_window_ratio = self._calc_N_init / self.opts_window_length
            else:
                if self.opts_window_ratio is not None and self.opts_is_window_warning == True:
                    logger.warning(
                        f"Window_length is manual input as {self.opts_window_length}. window_ratio would be ignored."
                    )     
                self.opts_window_length = self.opts_window_length
                self.opts_window_ratio = self._calc_N_init / self.opts_window_length

            object.__setattr__(self, "_calc_N_out", int(self._calc_N_init * self.opts_N_out_ratio))

            # Step 1: Apply Savitzky-Golay filter to smooth the curve
            line_length = self._calc_N_init
            if self.opts_window_length >= line_length:
                raise ValueError(
                    f"Filter window size {len(self.opts_window_length)} must be smaller than line length {line_length}"
                )
            line_points = savgol_filter(
                self._raw_coord,
                self.opts_window_length,
                self.opts_order,
                axis=0,
                mode=self.opts_mode,
            )

            # Step 2: Define spline parameter u
            uspline = np.arange(self._calc_N_init) / self._calc_N_init

            # Step 3: Fit and evaluate spline
            tck = splprep(line_points.T, u=uspline, s=0)[0]
            object.__setattr__(self, 
                               "_entities", 
                               [
                                np.array(splev(np.linspace(0, 1, self._calc_N_out), tck)).T
                                ]
                                )

    @logging_and_warning_decorator()
    def act_commit(self, logger=None, **changes):

        if not changes:
            return
        
        for k, v in changes.items():
            if k.startswith("opts_"):
                pre = len("opts_")
            else: 
                pre = 0
                setattr(self._opts_all, k[pre:], v)

        self._helper_apply_smooth(self._opts_all, logger=logger)

    @logging_and_warning_decorator()
    def act_visualize(self, 
                      color: ColorRGB = (1,1,1), 
                      scale_factor: Number = 1, 
                      move: Vect(3) = (0,0,0),
                      is_new: bool = True,
                      logger=None):

        color = as_ColorRGB(color, name="color to visualize smooth line")
        try:
            scale_factor = as_Number(scale_factor, name="scale_factor to visualize smooth line ")
        except:
            scale_factor = 1
            logger.recovery("Set scale_factor=1 in the following")
        try:
            move = as_Vect(move, name="The replacement to move smooth line")
        except:
            move = np.array([0,0,0])
        is_new = as_bool(is_new, name="whether to create a new figure to visualize smooth line")

        from mayavi import mlab

        if is_new:
            mlab.figure()

        pts = np.array(self)
        pts = pts[:, :3] + move
        mlab.points3d(*(pts.T), color=color, scale_factor=scale_factor)



    @logging_and_warning_decorator()
    def act_log_parameters(self, is_return: bool = False, logger=None) -> None:
        """
        Log internal filter and output parameters for inspection.

        This is the standard logging interface used in this library, which
        can be redirected to console or to a file depending on the logger
        configuration and the behavior of ``logging_and_warning_decorator``.

        All attributes listed in ``__descriptions__`` are included,
        formatted in a single log entry with a clear separator.
        """
        lines = []
        lines.append("-------------- SmoothLine Parameters --------------")

        if self._state_is_smoothed:
            lines.append(f"[{self.name}] smoothing parameters and results:")
            for attr in self.__slots__:
                desc = self.__descriptions__.get(attr, "(no description)")
                value = getattr(self, attr, None)

                if attr in ("opts_window_length", "opts_window_ratio"):
                    lines.append(f"  {attr}: {value!r}  # {desc} (derived final value)")
                elif attr in {"_is_window_warning", "_initializing", "_back_up_opts"}:
                    pass
                else:
                    lines.append(f"  {attr}: {value!r}  # {desc}")
        else:
            lines.append(
                f"[{self.name}] is not smoothed, because its length "
                f"{len(self._raw_coord)} < minimum required {self.opts_min_line_length}."
            )

        lines.append("-----------------------------------------------------")

        msg = "\n".join(lines)

        if is_return:
            return msg
        else:
            logger.info(msg)

    def __setattr__(self, key, value):

        if getattr(self, "_initializing", False):
            object.__setattr__(self, key, value)
            return
        
        if key == "name":
            object.__setattr__(self, key, value)
            return

        pre = 0
        if key.startswith("opts_"):
            pre = len("opts_")
        object.__setattr__(self, "_initializing", True)
        self.act_commit(**{key[pre:]: value})
        del self._initializing
        return


    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.act_log_parameters(is_return=True)
    
    def __len__(self) -> int:
        return self._calc_N_out
    

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if not self._state_is_smoothed:
            msg = f"{cls_name}(name={self.name!r}). This is not smoothed because its length {len(self._raw_coord)} is shorter than the minum length {self.opts_min_line_length}."
        else:
            msg = f"{cls_name}(name={self.name!r}), # input points is {self._calc_N_init}, window_length={self.opts_window_length}"

        return msg
    
    def __iter__(self):
        return iter(self._entities[0])
    
    def __getitem__(self, idx):
        return self._entities[0][idx]
    
    def __array__(self, dtype=None):
        arr = self._entities[0]
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
 
    def __enter__(self):
        object.__setattr__(self, "_backup_opts", asdict(self._opts_all))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self._backup_opts.items():
            if k in {"_is_window_warning", "_initializing", "_back_up_opts", '_state_is_smoothed'}:
                pass
            setattr(self._opts_all, k, v)
        self._helper_apply_smooth(self._opts_all)
        if hasattr(self, "_initializing"):
            del self._initializing
        del self._backup_opts
        return False  
    
    def __bool__(self):
        return self._state_is_smoothed
    
    def act_copy(self):
        return SmoothedLine(self._raw_coord.copy(), opts=OptsSmooth(**asdict(self._opts_all)))
    
    @logging_and_warning_decorator()
    def act_save(self, dirpath: Optional[str]=None, logger=None):

        if dirpath is None:
            dirpath = f"save/smoothed_line/{self.name}"
        dirpath = as_str(dirpath, name=f"the folder to store smoothed line ``{self.name}``")
            
        logger.debug(f"Start to save smoothed line ``{self.name}`` into {dirpath}")

        # ---------- ensure dirpath ----------
        os.makedirs(dirpath, exist_ok=True)

        # ---------- save JSON ----------
        json_path = os.path.join(dirpath, "info.json")
        param_dict = {
            "opts": asdict(self._opts_all),
            "metadata": {
                "name": self.name,
                "calc_N_init": self._calc_N_init,
                "calc_N_out": getattr(self, "_calc_N_out", None),
                "state_is_smoothed": self._state_is_smoothed,
            },
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(param_dict, f, indent=4)

        # ---------- save NumPy ----------
        npz_path = os.path.join(dirpath, "data.npz")
        np.savez_compressed(
            npz_path,
            raw_coord=self._raw_coord,
            smooth_out=self._entities[0]
        )

        
    @classmethod
    @logging_and_warning_decorator()
    def act_load(cls, dirpath: Optional[str]=None, logger=None):
        
        dirpath = as_str(dirpath, name="the folder to load smoothed line")

        json_path = os.path.join(dirpath, "info.json")
        npz_path = os.path.join(dirpath, f"data.npz")
        logger.debug(f"Start to load SmoothedLine from {json_path} and {npz_path}")

        if not os.path.exists(json_path) or not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Missing required files: {json_path} / {npz_path}"
            )

        # ---------- load JSON ----------
        with open(json_path, "r", encoding="utf-8") as f:
            param_dict = json.load(f)

        opts = OptsSmooth(**param_dict["opts"])

        # ---------- load NPZ ----------
        data = np.load(npz_path, allow_pickle=True)
        raw_coord = data["raw_coord"]
        smooth_out = data["smooth_out"]

        # ---------- reconstruct object ----------
        obj = cls(raw_coord, opts=opts)
        object.__setattr__(obj, "_entities", [smooth_out])
        object.__setattr__(obj, "_calc_N_init", param_dict["metadata"]["calc_N_init"])
        object.__setattr__(obj, "_calc_N_out", param_dict["metadata"]["calc_N_out"])
        object.__setattr__(obj, "_state_is_smoothed", param_dict["metadata"]["state_is_smoothed"])
        object.__setattr__(obj, "name", param_dict["metadata"]["name"])

        return obj
    
    @logging_and_warning_decorator()
    def __eq__(self, other, logger=None) -> bool:
        """
        Compare equality with another SmoothedLine object.

        Two SmoothedLine objects are considered equal iff **all attributes**
        in __slots__ are equal (deep equality for numpy arrays, shallow for scalars).

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
        if not isinstance(other, SmoothedLine):
            logger.info("The other variable is not class SmoothedLine")
            return False

        all_equal = True
        diffs = []

        for attr in self.__slots__:
            v1 = getattr(self, attr, None)
            v2 = getattr(other, attr, None)

            if attr == "_entities":
                v1, v2 = v1[0], v2[0]

            if attr == "_state_is_smoothed" and v1 != v2:
                if v1:
                    logger.info(f"{self.name} is smoothed while {other.name} is not")
                if v2:
                    logger.info(f"{other.name} is smoothed while {self.name} is not")
                all_equal = False

                pass

            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not (isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray) and np.array_equal(v1, v2)):
                    all_equal = False
                    diffs.append(f"{attr}: self={np.shape(v1)}, other={np.shape(v2)} (arrays differ)")
            else:
                if v1 != v2:
                    all_equal = False
                    diffs.append(f"{attr}: self={v1!r}, other={v2!r}")

        if not all_equal:
            if len(diffs)>0:
                logger.info(
                    "SmoothedLine objects are not equal.\nDifferences:\n" + "\n".join(diffs)
                )
        else:
            logger.info("SmoothedLine objects are equal.")

        return all_equal

