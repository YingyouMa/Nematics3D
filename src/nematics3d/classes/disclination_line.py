"""Disclination-line domain objects and their plot/section wrappers."""

import weakref
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping

import numpy as np
from scipy.interpolate import splprep

from ..general import sort_line_indices
from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import (
    Vect,
    as_Vect,
    Tensor,
    DefectIndex,
    as_DefectIndex,
    DimensionPeriodicInput,
    as_dimension_info,
    boundary_periodic_size_to_flag,
    as_str,
    as_Number,
    as_bool,
    UNSET,
    Unset,
)
from ..grid import (
    GRID_TRANSFORM_IDENTITY,
    apply_linear_transform,
    as_grid_transform,
    unwrap_trajectory,
    wrap_points_to_box,
)
from .visual.plot_figure import PlotFigure
from .visual.plot_tube import PlotTube, OptsTube
from .opts import merge_opts_all, cover_value
from .smoothed_line import OptsSmooth, SmoothedLine
from ..format import is_given_str
from ..general import find_plane_normal
from .class_base import ClassBase
from .host_base import OptsBase, HostBase
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar
from .q_plane import QPlanePolar
from .registry_base import RegistryBase
from .visual.qt.interact_disclination_line import InteractDisclinationLine


# extra attr


@dataclass(slots=True)
class InputLine:
    """
    Lightweight validated input bundle for constructing a DisclinationLine.

    InputLine stores the raw defect-index trajectory together with the lattice
    periodicity and the linear transform used to map lattice indices into
    real-space coordinates.

    Important readable attributes:

    - `defect_indices`: ordered lattice indices forming the defect line.
    - `box_size_periodic_index`: per-axis periodic size in index space.
    - `grid_offset`: translation applied after the lattice-to-real transform.
    - `grid_transform`: linear transform mapping index-space coordinates into
      real-space coordinates.

    Behavior:

    - field assignment is validated immediately through the class validators.
    - `repr(input_line)` uses the dataclass default field summary.
    """

    defect_indices: DefectIndex | None = None
    box_size_periodic_index: DimensionPeriodicInput = False
    grid_offset: Vect(3) = (0, 0, 0)
    grid_transform: Tensor((3, 3)) = GRID_TRANSFORM_IDENTITY

    __attrs__: ClassVar[Mapping[str, str]] = {
        "defect_indices": "indices of defect points in the Q array",
        "box_size_periodic_index": (
            "the maximum index of each index in the Q array "
            "(finite values for periodic boundary conditions and np.inf for "
            "non-periodic)"
        ),
        "grid_offset": (
            "grid translation offset to map lattice indices of Q array to "
            "real-space coordinates"
        ),
        "grid_transform": (
            "grid transform matrix to map lattice indices of Q array to "
            "real-space coordinates (3x3)"
        ),
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        "defect_indices": lambda v, d: (
            None if v is None else as_DefectIndex(v, is_return_row=True)
        ),
        "box_size_periodic_index": lambda v, d: as_dimension_info(v, name=d),
        "grid_offset": lambda v, d: as_Vect(v, name=d),
        "grid_transform": lambda v, d: as_grid_transform(v, name=d),
    }

    # -------------------------------
    # Validated assignment
    # -------------------------------

    # ==================== OVERRIDE ====================
    # InputLine overrides dataclass-style plain assignment so each supported
    # field is validated immediately when users construct or mutate the bundle.
    # ==================================================
    def __setattr__(self, key, value):
        validators = type(self).impl_validators
        if key in validators:
            desc = f"{key!r}: {type(self).__attrs__[key]}"
            value = validators[key](value, desc)
        object.__setattr__(self, key, value)


# DisclinationLine is the base wrapper for a traced defect trajectory in
# lattice and real-space coordinates.
#
# Subclasses should preserve the relationship among raw defect indices,
# transformed coordinates, line kind classification, and any generated smooth
# or visualization objects. If initialization is changed, keep the raw/cache
# fields synchronized before smoothing or plotting helpers are used.
class DisclinationLine(ClassBase):
    """
    DisclinationLine stores a defect line as ordered defect indices plus the
    corresponding transformed coordinates.

    This class is the main raw disclination-line wrapper. It keeps the traced
    lattice indices, the corresponding transformed coordinates, the line-kind
    classification, and any derived smoothed or visualized versions.

    Important readable attributes:

    - `raw_name`: the readable identity of this defect line.
    - `raw_defect_indices`: the ordered lattice indices of defect points.
    - `calc_defect_coords`: the corresponding real-space coordinates.
    - `calc_end2end_kind`: topology classification of the line, one of
      `"loop"`, `"cross"`, or `"seg"`.
    - `calc_defect_num`: the number of defect points currently stored.
    - `calc_norm`, `calc_norm_metric`: the latest estimated average plane
      normal and its confidence metrics, if computed.
    - `smooths`: all generated `DisclinationLineSmooth` versions.
    - `smooth`: the latest generated smoothed version, if any.
    - `kind`: shorthand property exposing `calc_end2end_kind`.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable line attributes.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations inherited from ClassBase.

    Common user actions:

    - `act_calc_norm()`: estimate an average plane normal for the line.
    - `act_smooth(...)`: create and store a smoothed version of the line.
    - `act_visualize(...)`: visualize either the raw line or one smoothed
      version through the plotting stack.

    Representation and array behavior:

    - `str(line)` returns the short ClassBase-style identity.
    - `repr(line)` returns a compact summary including topology kind and point count.
    - iteration, indexing, and `np.asarray(line)` operate on `raw_defect_indices`.
    """

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this disclination line.",
        },
        "raw_defect_indices": {
            "doc": "Lattice indices of defect points forming the line (array of shape Nx3).",
        },
        "raw_box_size_periodic_index": {
            "doc": (
                "Box size along each dimension in index space "
                "(finite for periodic boundaries, np.inf for non-periodic)."
            ),
        },
        "raw_grid_offset": {
            "doc": (
                "Grid translation offset mapping lattice indices to "
                "real-space coordinates (3-vector)."
            ),
        },
        "raw_grid_transform": {
            "doc": (
                "Grid transformation matrix (3x3) mapping lattice indices "
                "to real-space coordinates."
            ),
        },
        "calc_end2end_kind": {
            "doc": (
                "Kind of line ends: 'loop' (closed loop), "
                "'cross' (wraps across boundary), or 'seg' (open segment)."
            ),
            "kind": "calc",
        },
        "calc_defect_num": {
            "doc": "Number of defect points forming this line (integer).",
            "kind": "calc",
        },
        "calc_defect_coords": {
            "doc": "Real-space coordinates of the defect line (array of shape Nx3).",
            "kind": "calc",
        },
        "calc_norm": {
            "doc": "Estimated average plane normal vector of the disclination line.",
            "kind": "calc",
        },
        "calc_norm_metric": {
            "doc": "Collection of confidence scores for the plane-fitting result.",
            "kind": "calc",
        },
        "entity_smooth_objs": {
            "doc": (
                "Generated DisclinationLineSmooth objects produced by act_smooth()."
            ),
            "kind": "entity",
        },
        "smooths": {
            "doc": (
                "Read-only: All generated smoothed versions of this "
                "disclination line."
            ),
            "kind": "property",
        },
        "smooth": {
            "doc": "Read-only: The latest generated smoothed version, if any.",
            "kind": "property",
        },
        "kind": {
            "doc": "Read-only: Shorthand for the end-to-end topology kind of this line.",
            "kind": "property",
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in ClassBase.__slots__
    )

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DisclinationLine overrides ClassBase.__init__ because it must validate
    # defect-line input, classify the end-to-end topology, and initialize the
    # transformed coordinate cache before any smoothing helpers are used.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        inputValue: InputLine | None = None,
        is_sorted: bool = False,
        name: str | None = None,
        logger=None,
        **kwargs,
    ):

        if inputValue is None:
            inputValue = InputLine()
        if name is None:
            name = "disclination line"

        super().__init__(name=name, name_replace="disclination line", is_fixed=True)

        inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[""]
        if inputValue.defect_indices is None:
            raise ValueError("No defects are input into disclination line")
        for k, v in asdict(inputValue).items():
            object.__setattr__(self, f"raw_{k}", v)

        if not is_sorted:
            object.__setattr__(
                self, "raw_defect_indices", sort_line_indices(self.raw_defect_indices)
            )

        object.__setattr__(
            self,
            "raw_box_size_periodic_index",
            as_dimension_info(self.raw_box_size_periodic_index),
        )

        logger.debug("Classifying line kind by the distance between head and tail.")
        if (
            np.linalg.norm(self.raw_defect_indices[0] - self.raw_defect_indices[-1])
            == 0
        ):
            object.__setattr__(self, "calc_end2end_kind", "loop")
            object.__setattr__(self, "raw_defect_indices", self.raw_defect_indices[:-1])
        else:
            defect1 = self.raw_defect_indices[0].copy()
            defect2 = self.raw_defect_indices[-1].copy()
            defect1 = np.where(
                self.raw_box_size_periodic_index == np.inf,
                defect1,
                defect1 % self.raw_box_size_periodic_index,
            )
            defect2 = np.where(
                self.raw_box_size_periodic_index == np.inf,
                defect2,
                defect2 % self.raw_box_size_periodic_index,
            )
            if np.linalg.norm(defect1 - defect2) == 0:
                object.__setattr__(self, "calc_end2end_kind", "cross")
                object.__setattr__(
                    self, "raw_defect_indices", self.raw_defect_indices[:-1]
                )
            else:
                object.__setattr__(self, "calc_end2end_kind", "seg")
                object.__setattr__(self, "raw_defect_indices", self.raw_defect_indices)
        logger.debug(
            f"Disclination line {self.name!r} is of kind {self.calc_end2end_kind!r}"
        )

        object.__setattr__(
            self, "calc_defect_num", np.shape(self.raw_defect_indices)[0]
        )

        defect_coords = apply_linear_transform(
            self.raw_defect_indices,
            transform=self.raw_grid_transform,
            offset=self.raw_grid_offset,
        )
        object.__setattr__(self, "calc_defect_coords", defect_coords)
        object.__setattr__(self, "calc_norm", None)
        object.__setattr__(self, "calc_norm_metric", None)

        object.__setattr__(self, "entity_smooth_objs", [])

    # -------------------------------
    # Geometry analysis
    # -------------------------------

    @logging_and_warning_decorator()
    def act_calc_norm(self, logger=None) -> np.ndarray:
        """Estimate and cache one average plane normal for this defect line."""
        normal, metric = find_plane_normal(
            self.calc_defect_coords, is_return_metric=True
        )

        if metric["linearity_risk"] > 0.5:
            logger.warning(
                f"Low confidence in normal for {self.name!r}: "
                f"The disclination line is nearly straight. The calculated normal "
                f"may rotate arbitrarily around the line axis, "
                f"with linearity_risk as {metric['linearity_risk']:.2f}"
            )

        elif metric["planarity_score"] < 0.7:
            logger.warning(
                f"Low confidence in normal for {self.name!r}: "
                f"The line is highly non-planar (planarity_score={metric['planarity_score']:.2f}). "
                f"The result is only an 'average' plane normal."
            )

        object.__setattr__(self, "calc_norm", normal)
        object.__setattr__(self, "calc_norm_metric", metric)

        return normal

    # -------------------------------
    # Smoothing and visualization
    # -------------------------------

    def act_smooth(
        self,
        is_new=True,
        opts: OptsSmooth | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_window_warning: bool = True,
        **kwargs,
    ):
        """Create one smoothed version of this line and cache it on the instance."""
        if not is_new and len(self.smooths) > 0:
            return self.smooths[-1]

        name = self.name + " smooth_version " + str(len(self.smooths))

        item = DisclinationLineSmooth(
            self,
            name=name,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            is_window_warning=is_window_warning,
            **kwargs,
        )
        self.smooths.append(item)

        return item

    @logging_and_warning_decorator()
    # -------------------------------
    # Visualization and local analysis
    # -------------------------------

    def act_visualize(
        self,
        smooth_index: int = -1,
        figure: PlotFigure | None = None,
        is_wrap: bool = True,
        is_smooth: bool = True,
        opts: OptsTube | None = None,
        logger=None,
        **kwargs,
    ) -> None:
        """Visualize this line through one cached smoothed object or a new fallback."""
        if len(self.smooths) == 0:
            self.act_smooth(window_length=5, min_line_length=6)
            if is_smooth:
                logger.warning(
                    f"No cached smoothed version exists yet for disclination line {self.name!r}. "
                    "A smoothed object has been prepared for later interaction, "
                    "but this call will plot the original points instead."
                )
                is_smooth = False

        try:
            smooth_obj = self.smooths[smooth_index]
        except IndexError:
            logger.exception(
                f"Invalid smooth_index={smooth_index!r} for available smooth versions."
            )
            logger.recovery("Use the latest version instead.")
            smooth_obj = self.smooths[-1]

        if getattr(smooth_obj, "visual", None):
            smooth_obj = self.act_smooth(
                is_new=True,
                opts=smooth_obj.opts,
                is_window_warning=False,
            )

        line_plot = smooth_obj.act_visualize(
            figure=figure,
            is_wrap=is_wrap,
            is_smooth=is_smooth,
            opts_tube=opts,
            **kwargs,
        )

        return line_plot

    # -------------------------------
    # Array-style access and representation
    # -------------------------------

    def __len__(self) -> int:
        """Return the number of defect points currently stored in this line."""
        return self.calc_defect_num

    # ==================== OVERRIDE ====================
    # DisclinationLine overrides ClassBase.__repr__ because defect-line
    # objects are more useful when summarized by topology kind and point
    # count than by name alone.
    # ==================================================
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = (
            f"{cls_name}({self.name!r}), type {self.calc_end2end_kind}, "
            f"{self.calc_defect_num} defect points"
        )
        return msg

    # ==================== OVERRIDE ====================
    # DisclinationLine overrides ClassBase.__str__ to keep the plain string form
    # short and aligned with the repository-wide default identity style.
    # ==================================================
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def __iter__(self):
        """Iterate over the stored raw defect indices."""
        return iter(self.raw_defect_indices)

    def __getitem__(self, idx):
        """Return one raw defect-index entry or slice."""
        return self.raw_defect_indices[idx]

    def __array__(self, dtype=None):
        """Expose the raw defect indices as a NumPy array."""
        arr = self.raw_defect_indices
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr

    # -------------------------------
    # Readable properties
    # -------------------------------

    @property
    def smooths(self):
        """Return all cached smoothed versions of this disclination line."""
        return getattr(self, "entity_smooth_objs", None)

    @property
    def smooth(self):
        """Return the latest cached smoothed version, if one exists."""
        try:
            return self.smooths[-1]
        except (TypeError, IndexError):
            return None

    @property
    def kind(self):
        """Return the end-to-end topology classification of this line."""
        return self.calc_end2end_kind


# DisclinationLineSmooth extends SmoothedLine with defect-line-specific
# coordinate handling, visualization wrappers, and cross-section helpers.
#
# Subclasses should preserve the coupling between the owning disclination
# line, the smoothed index trajectory, transformed coordinates, and any
# derived section or visualization registries.
class DisclinationLineSmooth(SmoothedLine):
    """
    DisclinationLineSmooth is the smoothed version of a disclination line.

    This class extends SmoothedLine with defect-line-specific preprocessing,
    periodic padding/trimming, transformed real-space output coordinates, and
    one-to-one links to visualization or cross-section helpers.

    Important readable attributes:

    - `owner`: the raw DisclinationLine that this smoothed version belongs to.
    - `calc_coords_index`: the index-space trajectory after any periodic
      padding/unwrap and before physical-space smoothing.
    - `calc_coords`: the physical-space coordinates entering the smoothing
      pipeline.
    - `calc_result_index`: the final smoothed defect-line indices in lattice space.
    - `calc_result`: the final smoothed defect-line coordinates in real space.
    - `calc_result_coords`: compatibility alias of `calc_result`.
    - `calc_padding_num`: temporary padding length used for cross-boundary lines.
    - `visual`: the current one-to-one visualization wrapper, if any.
    - `sections`: the registry of derived cross-section grids.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable smoothed-line attributes.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations such as the owner, visual, and sections registry.

    Common user actions:

    - `act_commit(...)`: update smoothing parameters and rebuild the smoothed result.
    - `act_visualize(...)`: create a visualization wrapper for this smoothed line.
    - `act_cross_section(...)`: build a local polar cross-section along the line.
    - `act_calc_omega(...)`: evaluate local omega on a polar section along the line.
    - `act_calc_tangent(...)`: evaluate the tangent of the smoothed line spline.

    Representation and array behavior:

    - `str(obj)` returns the short HostBase-style identity.
    - `repr(obj)` follows the SmoothedLine summary style.
    - `np.asarray(obj)`, indexing, iteration, and `len(obj)` operate on the
      current smoothed real-space result.
    """

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(SmoothedLine.__attr_defs__),
        "calc_coords_index": {
            "doc": (
                "Index-space trajectory entering smoothing before conversion "
                "to real-space coordinates."
            ),
            "kind": "calc",
        },
        "calc_result_index": {
            "doc": "The smoothed disclination coordinates in lattice-index space.",
            "kind": "calc",
        },
        "calc_result_coords": {
            "doc": "Compatibility alias of calc_result for real-space coordinates.",
            "kind": "calc",
        },
        "calc_padding_num": {
            "doc": "Temporary padding length used when smoothing a cross-boundary line.",
            "kind": "calc",
        },
        "impl_owner_init_ref": {
            "doc": "Temporary owner weakref used before the managed owner relation is bound.",
            "kind": "impl",
        },
        "owner": {
            "doc": "The raw disclination line that owns this smoothed version.",
            "kind": "relation",
        },
        "visual": {
            "doc": (
                "The one-to-one visualization wrapper currently associated with "
                "this smoothed disclination line."
            ),
            "kind": "relation",
        },
        "visual_tube": {
            "doc": (
                "Read-only: PlotTube wrapped by the current visualization wrapper, "
                "or None when no visualization exists."
            ),
            "kind": "property",
        },
        "sections": {
            "doc": (
                "RegistryBase object managing cross-section grids created from "
                "this smoothed disclination line."
            ),
            "kind": "relation",
        },
        "linefunc_mode": {
            "doc": (
                "Read-only: Interpolation mode used by functions sampled along "
                "this smoothed disclination line. Loop and cross-boundary lines "
                "are periodic for line functions."
            ),
            "kind": "property",
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in SmoothedLine.__slots__
    )

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DisclinationLineSmooth overrides SmoothedLine.__init__ because
    # it must bind a DisclinationLine owner and initialize the
    # defect-line-specific entities before the generic smoothing setup.
    # ==================================================
    def __init__(
        self,
        line,
        opts: OptsSmooth | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        name: str | None = None,
        **kwargs,
    ):

        if not isinstance(line, DisclinationLine):
            raise TypeError(
                "The `line` input of DisclinationLineSmooth must be an "
                "instance of `DisclinationLine`. "
                f"Got type={type(line).__name__} instead."
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
            info=(
                "registry of cross-section grids for the smoothed "
                f"disclination line {self.name!r}"
            ),
        )
        self.act_bind_relation_base("sections", sections, is_weak=False)
        sections.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "calc_padding_num", 0)
        self.act_register_protected_attr(["coords", "mode"])

    # -------------------------------
    # Defect-line coordinate preprocessing
    # -------------------------------

    def _helper_get_owner_during_init(self) -> DisclinationLine:
        """Resolve the owner during bootstrap before the relation is fully bound."""
        owner = getattr(self, "owner", None)
        if owner is not None:
            return owner

        owner_ref = getattr(self, "impl_owner_init_ref", None)
        if isinstance(owner_ref, weakref.ReferenceType):
            owner = owner_ref()

        if owner is None:
            raise RuntimeError(
                "DisclinationLineSmooth could not resolve its owning line during initialization."
            )
        return owner

    @property
    def linefunc_mode(self):
        owner = self.owner
        if owner is None:
            return self.opts.mode
        if owner.calc_end2end_kind in ("loop", "cross"):
            return "wrap"
        return "interp"

    @property
    def visual_tube(self):
        visual = self.visual
        if visual is None:
            return None
        return visual.wrapped

    # ==================== OVERRIDE ====================
    # DisclinationLineSmooth overrides SmoothedLine._helper_resolve_coords
    # because defect-line smoothing must handle loop and cross-boundary
    # trajectories in index space before passing physical coordinates into the
    # generic smooth pipeline.
    # ==================================================
    def _helper_resolve_coords(self):
        owner = self._helper_get_owner_during_init()
        indices = owner.raw_defect_indices.copy()
        padding_num = 0
        smooth_mode = "interp"

        if owner.calc_end2end_kind == "loop":
            smooth_mode = "wrap"

        elif owner.calc_end2end_kind == "cross":
            box_size = owner.raw_box_size_periodic_index

            if self.opts.window_ratio is not None:
                padding_num = int(len(indices) / self.opts.window_ratio / 2)
            else:
                padding_num = int((self.opts.window_length or len(indices)) / 2)

            indices_origin = owner.raw_defect_indices.copy()
            tail = indices_origin[:padding_num].copy()
            head = indices_origin[-padding_num:].copy()
            indices = np.concatenate([head, indices_origin, tail])

            indices = unwrap_trajectory(indices, box_size_periodic=box_size)

            start_origin = owner.raw_defect_indices[0]
            start_now = indices[padding_num]
            mask = np.isfinite(box_size)
            shift = np.zeros(3, dtype=float)
            shift[mask] = np.round(
                (start_origin[mask] - start_now[mask]) / box_size[mask]
            )
            indices += shift * box_size

        object.__setattr__(self, "calc_padding_num", padding_num)
        object.__setattr__(self, "calc_coords_index", indices)
        coords = apply_linear_transform(
            indices,
            transform=owner.raw_grid_transform,
            offset=owner.raw_grid_offset,
        )
        object.__setattr__(self, "calc_coords", coords)
        object.__setattr__(self.opts, "mode", smooth_mode)

    # -------------------------------
    # Smoothing commit pipeline
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DisclinationLineSmooth overrides SmoothedLine._helper_commit_apply_opts_main
    # because smoothing results may need periodic trimming, raw-result
    # fallback, and transformed-coordinate updates specific to defect lines.
    # ==================================================
    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):
        owner = self._helper_get_owner_during_init()

        if "mode" in kwargs:
            kwargs.pop("mode")
            logger.warning(
                "'mode' is ignored and removed from kwargs because the smooth "
                "mode is determined by the kind of disclination line."
            )

        super()._helper_commit_apply_opts_main(
            is_reapply_opts=is_reapply_opts,
            **kwargs,
        )

        padding_num = int(getattr(self, "calc_padding_num", 0))
        if self.calc_is_smoothed and padding_num > 0 and len(self.calc_result) > 0:
            trim = int(round(padding_num * float(self.opts.num_out_ratio)))
            if trim > 0 and (2 * trim) < len(self.calc_result):
                result = self.calc_result[trim:-trim]
                object.__setattr__(self, "calc_result", result)
                tck = splprep(
                    result.T.copy(),
                    u=np.linspace(0.0, 1.0, len(result)),
                    s=0,
                    per=0,
                )[0]
                object.__setattr__(self, "entity_tck", tck)

        if not self.calc_is_smoothed:
            result_index = owner.raw_defect_indices.copy()
            result = apply_linear_transform(
                result_index,
                transform=owner.raw_grid_transform,
                offset=owner.raw_grid_offset,
            )
            object.__setattr__(
                self,
                "calc_result",
                result,
            )
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

        tube_wrapper = getattr(self, "visual", None)
        if tube_wrapper:
            tube_wrapper.act_commit()

    # -------------------------------
    # Public actions
    # -------------------------------

    def act_visualize(
        self,
        figure: PlotFigure | None = None,
        is_wrap: bool = True,
        is_smooth: bool = True,
        opts_tube: OptsTube | None = None,
        opts_tube_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        """Create and bind one PlotTube-based visualization wrapper."""
        tube = DisclinationLineSmoothPlot(
            self,
            is_smooth=is_smooth,
            is_wrap=is_wrap,
            figure=figure,
            opts_tube=opts_tube,
            opts_tube_defaults_override=opts_tube_defaults_override,
            **kwargs,
        )

        self.act_bind_relation_base("visual", tube, is_weak=False)

        return tube

    def act_cross_section(self, x_param, **kwargs):
        """Build one local polar cross-section along this smoothed line."""
        plane = DefectSectionGrid(self, u_percent=x_param, **kwargs)
        return plane

    @logging_and_warning_decorator()
    def act_calc_omega(
        self,
        u_percent,
        opts_grid: OptsPlaneGridPolar | None = None,
        opts_grid_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):
        """
        Evaluate local omega on one polar section along the smoothed line.

        The section pose is resolved from ``u_percent`` in real space. Keep
        pose and global grid-transform fields out of ``opts_grid`` and keyword
        arguments so the resolved physical sampling plane is never ambiguous.
        """
        if opts_grid is not None and not isinstance(opts_grid, OptsPlaneGridPolar):
            raise TypeError(
                "`opts_grid` must be an OptsPlaneGridPolar instance or None. "
                f"Got {type(opts_grid).__name__!r} instead."
            )

        ignored_clip_keys = []
        if "bounds" in kwargs:
            kwargs.pop("bounds")
            ignored_clip_keys.append("bounds")
        if "is_clip_inside" in kwargs:
            kwargs.pop("is_clip_inside")
            ignored_clip_keys.append("is_clip_inside")
        if (
            opts_grid is not None
            and opts_grid.act_asdict().get("is_clip_inside", UNSET) is not UNSET
        ):
            opts_grid = replace(opts_grid, is_clip_inside=UNSET)
            ignored_clip_keys.append("opts_grid.is_clip_inside")
        if (
            opts_grid_defaults_override is not None
            and "is_clip_inside" in opts_grid_defaults_override
        ):
            opts_grid_defaults_override = {
                key: value
                for key, value in opts_grid_defaults_override.items()
                if key != "is_clip_inside"
            }
            ignored_clip_keys.append("opts_grid_defaults_override.is_clip_inside")
        if ignored_clip_keys:
            logger.warning(
                "Ignoring omega clipping settings "
                f"{sorted(set(ignored_clip_keys))}. Omega is evaluated on the "
                "complete polar ring via `entity_grid_all`; clipping only "
                "changes the selected `entity_grid`, which is not used for "
                "the ring-wise omega calculation."
            )

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
                "act_calc_omega resolves the section pose from `u_percent`; "
                f"do not pass {keys} through `opts_grid`, "
                "`opts_grid_defaults_override`, or keyword arguments. Use "
                "`opts_grid` only for polar-grid sampling options such as "
                "layers, dr, arc_dist, theta0_axis, and clipping settings."
            )

        q_host = getattr(getattr(self.owner, "registry", None), "owner", None)
        if q_host is None:
            raise RuntimeError(
                "Cannot resolve the owning Q object needed to build a QPlanePolar section."
            )
        if q_host.interpolator is None:
            q_host.act_add_interpolator()

        tangent, origin = self.act_calc_tangent(u_percent, is_return_coord=True)
        origin = wrap_points_to_box(
            origin,
            self.owner.raw_box_size_periodic_index,
            transform=self.owner.raw_grid_transform,
            offset=self.owner.raw_grid_offset,
        )

        grid = PlaneGridPolar(
            normal=tangent,
            origin=origin,
            opts=opts_grid,
            opts_defaults_override=opts_grid_defaults_override,
            grid_offset=(0, 0, 0),
            grid_transform=GRID_TRANSFORM_IDENTITY,
            **kwargs,
        )
        q_plane = QPlanePolar(
            interpolator=q_host.act_add_interpolator(),
            grid=grid,
            name=f"omega plane of {self.name!r}",
        )

        layer = int(q_plane.grid.calc_ring_offsets.shape[0] - 2)
        if layer < 0:
            raise ValueError(
                "The local polar section does not contain a valid ring layer."
            )

        result = q_plane.act_calc_omega(layer)
        omega = np.asarray(result["omega"], dtype=float)
        cos_beta = abs(float(np.dot(tangent, omega)))
        if not np.isfinite(cos_beta):
            beta = np.nan
        else:
            cos_beta = float(np.clip(cos_beta, -1.0, 1.0))
            beta = float(np.degrees(np.arccos(cos_beta)))

        result["beta"] = beta
        result["u_percent"] = float(u_percent)
        result["position"] = origin

        return result


@dataclass(slots=True, repr=False)
class OptsDefectLinePlot(OptsBase):
    """
    Options object controlling high-level visualization choices for a smoothed
    disclination line wrapper.

    This opts object does not control the underlying tube style directly.
    Instead, it controls whether the visualization uses the smoothed geometry
    and whether periodic wrapping is applied before the wrapped PlotTube is
    updated.

    Important readable attributes:

    - `host`: the DisclinationLineSmoothPlot currently using this opts object, if any.
    - `is_smooth`: whether the displayed geometry should use the smoothed line.
    - `is_wrap`: whether periodic wrapping should be applied before plotting.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Representation:

    - `str(opts)` returns a short one-line identity.
    - `repr(opts)` returns the full current opts summary.
    """

    is_smooth: bool | Unset = UNSET
    is_wrap: bool | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "is_smooth": "Whether to apply geometric smoothing to defect lines during visualization.",
        "is_wrap": "Whether to apply periodic-boundary wrapping when visualizing defect lines.",
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **OptsBase.impl_validators,
        "is_smooth": lambda v, d: as_bool(v, name=d),
        "is_wrap": lambda v, d: as_bool(v, name=d),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(getattr(OptsBase, "impl_defaults_frozen", {})),
            "is_smooth": True,
            "is_wrap": True,
        }
    )


class DisclinationLineSmoothPlot(HostBase):
    """
    DisclinationLineSmoothPlot visualizes a smoothed disclination line.

    This wrapper owns a high-level visualization state and forwards the actual
    rendered geometry into an internal PlotTube. It lets users toggle whether
    the smoothed or raw defect trajectory should be displayed, and whether
    periodic wrapping should be applied before plotting.

    Important readable attributes:

    - `opts`: the paired OptsDefectLinePlot controlling smoothing/wrapping display state.
    - `owner`: the DisclinationLineSmooth currently being visualized.
    - `wrapped`: the internal PlotTube used for actual rendering.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable visualization-wrapper attributes.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations such as the owner and wrapped PlotTube.

    Common user actions:

    - `act_commit(...)`: update whether the wrapped plot should use smoothing or wrapping.
    - `act_set_name(name)`: rename the visualization wrapper.

    Representation:

    - `str(obj)` returns the short HostBase-style identity.
    - `repr(obj)` returns a compact summary including the two display toggles.
    """

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(HostBase.__attr_defs__),
        "raw_name": {
            **dict(HostBase.__attr_defs__["raw_name"]),
            "doc": "The name of this visualization wrapper for a smoothed disclination line.",
        },
        "owner": {
            "doc": "The smoothed disclination line currently visualized by this wrapper.",
            "kind": "relation",
        },
        "wrapped": {
            "doc": "The internal PlotTube used for actual rendering.",
            "kind": "relation",
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in HostBase.__slots__
    )

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DisclinationLineSmoothPlot overrides HostBase.__init__ because
    # it must create and bind an internal PlotTube wrapped by this
    # visualization wrapper during initialization.
    # ==================================================
    def __init__(
        self,
        line: DisclinationLineSmooth,
        figure: PlotFigure | None = None,
        opts: OptsDefectLinePlot | None = None,
        opts_tube: OptsTube | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        opts_tube_defaults_override: Mapping[str, Any] | None = None,
        name: str | None = None,
        **kwargs,
    ):

        if not isinstance(line, DisclinationLineSmooth):
            raise TypeError(
                "The `line` input should be DisclinationLineSmooth instance. "
                f"Got {type(line).__name__!r} instead"
            )

        self_descriptions = OptsDefectLinePlot.__attrs__
        self_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key in self_descriptions
        }

        name_replace = line.name
        name = name_replace if name is None else name
        super().__init__(
            opts_type=OptsDefectLinePlot,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **self_kwargs,
        )

        self.act_bind_relation_base("owner", line, is_weak=True)

        self.opts.act_finalize(defaults=self.opts_defaults)

        line_coords, line_index = self._helper_get_coords()
        tube = PlotTube(
            coords=line_coords,
            name=self.owner.name,
            category="disclination line",
            figure=figure,
            opts=opts_tube,
            line_index=line_index,
            opts_defaults_override=opts_tube_defaults_override,
            **kwargs,
        )

        tube.act_bind_wrapper(self, protected_attrs=["coords", "line_index"])
        self.act_attach_enrich_kwargs_wrapped_task(
            "visual_coords", self._helper_enrich_kwargs_wrapped_visual
        )
        tube.act_set_interact_func(lambda: InteractDisclinationLine(tube).show())

    # -------------------------------
    # Wrapped-geometry resolution
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DisclinationLineSmoothPlot adds `_helper_get_coords()` as the internal
    # coordinate resolver for wrapped visualization because the displayed
    # geometry depends on both smoothing and wrapping choices.
    # ==================================================
    @logging_and_warning_decorator()
    def _helper_get_coords(self, logger=None):

        is_smooth = bool(self.opts.is_smooth)
        is_wrap = bool(self.opts.is_wrap)
        owner = self.owner

        logger.debug(
            f"Start to visualize line: {owner.name!r} with kind ``{owner.owner.kind}``"
        )

        if not is_wrap:
            line_coords = owner.result if is_smooth else owner.owner.calc_defect_coords
            if owner.owner.kind == "loop":
                line_coords = np.concatenate((line_coords, [line_coords[0]]))
            line_index = None

        else:
            logger.debug("Start to deal with the periodic boundary condition")

            boundary_flag = boundary_periodic_size_to_flag(
                owner.owner.raw_box_size_periodic_index
            )

            line_coords_origin = (
                owner.calc_result_index if is_smooth else owner.owner.raw_defect_indices
            )
            if owner.owner.kind == "loop":
                line_coords_origin = np.concatenate(
                    (line_coords_origin, [line_coords_origin[0]])
                )

            line_coords_origin = np.where(
                boundary_flag,
                line_coords_origin % owner.owner.raw_box_size_periodic_index,
                line_coords_origin,
            )

            diff = line_coords_origin[1:] - line_coords_origin[:-1]
            if np.any(boundary_flag):
                # Split the rendered polyline only at actual periodic seams.
                # Normal spacing between neighboring smoothed samples should
                # remain connected inside the same periodic image.
                jump = np.any(
                    np.abs(diff[:, boundary_flag])
                    > owner.owner.raw_box_size_periodic_index[boundary_flag] / 2,
                    axis=-1,
                )
            else:
                jump = np.zeros(len(diff), dtype=bool)
            end_list = np.where(jump)[0] + 1
            end_list = np.concatenate([[0], end_list, [len(line_coords_origin)]])

            line_index = np.ones(len(line_coords_origin))
            for i in range(len(end_list) - 1):
                line_index[end_list[i] : end_list[i + 1]] = i

            line_coords = apply_linear_transform(
                line_coords_origin,
                transform=owner.owner.raw_grid_transform,
                offset=owner.owner.raw_grid_offset,
            )

        return line_coords, line_index

    def _helper_enrich_kwargs_wrapped_visual(self, host=None, kwargs=None):
        del host, kwargs
        line_coords, line_index = self._helper_get_coords()
        return {"coords": line_coords, "line_index": line_index}

    # -------------------------------
    # Representation
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DisclinationLineSmoothPlot overrides ClassBase.__repr__ because the
    # visualization wrapper is most useful when summarized by its two display
    # toggles in addition to its identity.
    # ==================================================
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}({self.name!r}), is_smooth={self.opts.is_smooth}, "
            f"is_wrap={self.opts.is_wrap}"
        )

    # ==================== OVERRIDE ====================
    # DisclinationLineSmoothPlot overrides ClassBase.__str__ to keep the plain
    # string form short and aligned with the repository-wide default identity
    # style.
    # ==================================================
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    # -------------------------------
    # Wrapped-state commit pipeline
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DisclinationLineSmoothPlot overrides HostBase._helper_commit_apply_opts_main
    # because its own opts only control wrapped-visualization state and should
    # be applied without redefining the wrapped PlotTube opts interface.
    # ==================================================
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        if not is_reapply_opts and not kwargs:
            return
        with self.opts.act_internal_update():
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )
        wrapped = self.wrapped
        if wrapped is not None:
            kwargs_wrapped = self._helper_commit_enrich_kwargs_wrapped({})
            with wrapped.act_wrapped_update():
                wrapped.act_commit(**kwargs_wrapped)


@dataclass(slots=True, repr=False)
class OptsDefectSectionGrid(OptsBase):
    """
    Options object controlling the sampling position of a defect-line section grid.

    This opts object is paired with DefectSectionGrid and currently exposes the
    normalized spline position used to place the local polar section plane.

    Important readable attributes:

    - `host`: the DefectSectionGrid currently using this opts object, if any.
    - `u_percent`: normalized spline position along the smoothed defect line.
    - `is_wrap`: whether the resolved section origin should be wrapped into the
      principal periodic box before building the local polar plane.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Representation:

    - `str(opts)` returns a short one-line identity.
    - `repr(opts)` returns the full current opts summary.
    """

    u_percent: float | Unset = UNSET
    is_wrap: bool | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "u_percent": (
            "Normalized spline parameter percentage along the smoothed defect line. "
            "0 means the start of the spline parameter domain and 100 means the end."
        ),
        "is_wrap": (
            "Whether the resolved section origin should be wrapped into the "
            "principal periodic box before constructing the local polar plane."
        ),
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **OptsBase.impl_validators,
        "u_percent": lambda v, d: as_Number(v, name=d, value_range=(0, 100)),
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


# DefectSectionGrid is the HostBase wrapper for a local polar section
# constructed along a smoothed disclination line.
# constructed along a smoothed disclination line.
#
# Subclasses should preserve the coupling among the owner line, the current
# section position (`u_percent`), the resolved section normal, and the wrapped
# `PlaneGridPolar`. If pose resolution changes, keep `calc_normal` and the
# wrapped-section enrichment task synchronized.
class DefectSectionGrid(HostBase):
    """
    DefectSectionGrid builds a local polar sampling plane along a smoothed
    disclination line.

    This wrapper resolves a local section pose from a smoothed defect line,
    chooses a normal from either the tangent, a registered named normal, or a
    direct vector, and then forwards the resulting pose into a wrapped
    PlaneGridPolar.

    Important readable attributes:

    - `opts`: the paired OptsDefectSectionGrid controlling the section position.
    - `owner`: the DisclinationLineSmooth that this section belongs to.
    - `wrapped`: the internal PlaneGridPolar used for actual sampling geometry.
    - `state_normal`: current normal selector, either `"tangent"`, a registered
      normal name, or a direct vector.
    - `calc_normal`: the resolved normal currently driving the wrapped section.
    - `impl_normals`: registered named normal providers for this section.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable section-grid attributes.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations such as the owner and wrapped grid.
    - `show_normals()`: show the currently registered named normals.

    Common user actions:

    - `act_commit(...)`: update section position or normal selection.
    - `act_set_name(name)`: rename the section wrapper.
    - `act_register_normal(name, value)`: register a named normal provider.

    Representation:

    - `str(obj)` returns the short HostBase-style identity.
    - `repr(obj)` returns a compact summary including `u_percent` and the current
      `state_normal`.
    """

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(HostBase.__attr_defs__),
        "raw_name": {
            **dict(HostBase.__attr_defs__["raw_name"]),
            "doc": "The name identifier of this local defect section grid wrapper.",
        },
        "state_normal": {
            "doc": (
                "Current normal selector for the section; either tangent, "
                "a registered name, or a direct vector."
            ),
        },
        "calc_normal": {
            "doc": "Resolved normal currently used by this defect section grid.",
            "kind": "calc",
        },
        "impl_normals": {
            "doc": "Named normal providers used to resolve section normals.",
            "kind": "impl",
        },
        "owner": {
            "doc": "The smoothed disclination line that owns this section grid.",
            "kind": "relation",
        },
        "wrapped": {
            "doc": "The internal PlaneGridPolar used for actual sampling geometry.",
            "kind": "relation",
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in HostBase.__slots__
    )

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DefectSectionGrid overrides HostBase.__init__ because it must resolve an
    # initial section pose, create a wrapped PlaneGridPolar, and bind the
    # section back into the owning smoothed line registry.
    # ==================================================
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
                "The `line` input should be DisclinationLineSmooth instance. "
                f"Got {type(line).__name__!r} instead"
            )

        self_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
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
                "DefectSectionGrid resolves the section pose in real space; "
                f"do not pass {keys} through `opts_grid`, "
                "`opts_grid_defaults_override`, or keyword arguments. Use "
                "`opts_grid` only for polar-grid sampling options such as "
                "layers, dr, arc_dist, theta0_axis, and clipping settings."
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
        object.__setattr__(self, "impl_normals", {"tangent": None})
        object.__setattr__(self, "calc_normal", None)
        self.opts.act_finalize(defaults=self.opts_defaults)

        if normals is not None:
            for key, value in normals.items():
                self.act_register_normal(key, value)

        state_normal = self._helper_check_state_normal(
            state_normal,
            self.show_attr_desc("state_normal"),
        )
        object.__setattr__(self, "state_normal", state_normal)

        self.impl_attrs["state_normal"]["validator"] = self._helper_check_state_normal
        pose = self._helper_resolve_pose()

        grid = PlaneGridPolar(
            normal=pose["normal"],
            origin=pose["origin"],
            opts=opts_grid,
            opts_defaults_override=opts_grid_defaults_override,
            grid_offset=(0, 0, 0),
            grid_transform=GRID_TRANSFORM_IDENTITY,
            **kwargs,
        )
        grid.act_bind_wrapper(self, protected_attrs=["origin", "normal"])
        self.act_attach_enrich_kwargs_wrapped_task(
            "section_pose", self._helper_enrich_kwargs_wrapped_section
        )
        line.sections.act_register(self)

    # -------------------------------
    # Normal validation and pose resolution
    # -------------------------------

    def _helper_check_state_normal(self, state_normal, desc):
        if isinstance(state_normal, str):
            state_normal = as_str(
                state_normal,
                name="The normal selector of defect section grid",
            )
            if state_normal != "tangent" and state_normal not in self.impl_normals:
                raise ValueError(
                    f"{desc} Got unknown registered normal name {state_normal!r}."
                )
            return state_normal
        return as_Vect(
            state_normal,
            name="The direct normal of defect section grid",
            is_norm=True,
        )

    def _helper_resolve_normal(self, tangent):
        normal = (
            tangent if is_given_str(self.state_normal, "tangent") else self.state_normal
        )
        if isinstance(normal, str):
            normal = self.impl_normals[normal]
            if callable(normal):
                normal = normal()
        normal = as_Vect(
            normal,
            name="The resolved normal of defect section grid",
            is_norm=True,
        )
        object.__setattr__(self, "calc_normal", normal)
        return normal

    def _helper_resolve_pose(self):
        tangent, origin = self.owner.act_calc_tangent(
            self.opts.u_percent,
            is_return_coord=True,
        )
        if self.opts.is_wrap:
            origin = wrap_points_to_box(
                origin,
                self.owner.owner.raw_box_size_periodic_index,
                transform=self.owner.owner.raw_grid_transform,
                offset=self.owner.owner.raw_grid_offset,
            )
        normal = self._helper_resolve_normal(tangent)
        return {"origin": origin, "normal": normal}

    def _helper_enrich_kwargs_wrapped_section(self, host=None, kwargs=None):
        del host, kwargs
        return self._helper_resolve_pose()

    # -------------------------------
    # Commit pipeline
    # -------------------------------

    # `state_normal` needs an instance-aware validator because valid string names
    # depend on this section's currently registered normals.
    # ==================== OVERRIDE ====================
    # DefectSectionGrid overrides HostBase._helper_commit_pre_opts because
    # `state_normal` uses instance-aware validation against the currently
    # registered normal names.
    # ==================================================
    def _helper_commit_pre_opts(self, kwargs):
        kwargs_sync, is_reapply_opts = super()._helper_commit_pre_opts(kwargs)
        kwargs_applied_state, is_reapply_state = self._helper_commit_pop_raw(
            kwargs,
            "state_normal",
            validator=self._helper_check_state_normal,
        )
        return kwargs_sync | kwargs_applied_state, (is_reapply_opts or is_reapply_state)

    # ==================== OVERRIDE ====================
    # DefectSectionGrid overrides HostBase._helper_commit_apply_opts_main
    # because section opts only update the local section pose and then forward
    # the resolved origin/normal to the wrapped polar grid.
    # ==================================================
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        for key, value in kwargs.items():
            object.__setattr__(self.opts, key, value)
        return self._helper_resolve_pose(), kwargs

    # -------------------------------
    # Representation
    # -------------------------------

    # ==================== OVERRIDE ====================
    # DefectSectionGrid overrides ClassBase.__repr__ because a section grid is
    # most useful when summarized by its current section position and normal
    # selector in addition to its identity.
    # ==================================================
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}({self.name!r}), u_percent={self.opts.u_percent}, "
            f"state_normal={self.state_normal!r}"
        )

    # ==================== OVERRIDE ====================
    # DefectSectionGrid overrides ClassBase.__str__ to keep the plain string
    # form short and aligned with the repository-wide default identity style.
    # ==================================================
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    # -------------------------------
    # Normal registry helpers
    # -------------------------------

    @logging_and_warning_decorator()
    def act_register_normal(self, key, value, logger=None):
        """Register one named normal provider for this defect section."""
        try:
            key = as_str(key, name="The name of a registered defect-section normal")
        except (TypeError, ValueError):
            logger.warning(
                "Skip registering a defect-section normal because "
                f"key={key!r} is not a valid string."
            )
            return

        if key == "tangent":
            logger.warning(
                "The reserved normal name 'tangent' is built in and cannot be "
                "overwritten. Skip this registration."
            )
            return

        if callable(value):
            self.impl_normals[key] = value
            return

        try:
            value = as_Vect(
                value,
                name=f"The normal {key!r} of defect section grid",
                is_norm=True,
            )
        except (TypeError, ValueError):
            logger.warning(
                f"Skip registering defect-section normal {key!r} because the "
                "value is neither a callable nor a valid 3-vector."
            )
            return

        self.impl_normals[key] = value

    @logging_and_warning_decorator()
    def show_normals(self, is_return=False, logger=None):
        """Show all currently registered named normals for this section grid."""
        is_return = as_bool(
            is_return,
            name="Whether to return the normal summary",
            replace=False,
        )

        lines = [f"Registered normals of {self.name!r}:"]
        for key, value in self.impl_normals.items():
            if key == "tangent":
                desc = "built-in tangent function"
            elif callable(value):
                desc = "given function"
            else:
                desc = np.array2string(
                    np.asarray(value, dtype=float),
                    precision=3,
                    separator=", ",
                )
            lines.append(f"  - {key}: {desc}")

        output = "\n".join(lines)

        logger.info(output)
        if is_return:
            return output
        return None
