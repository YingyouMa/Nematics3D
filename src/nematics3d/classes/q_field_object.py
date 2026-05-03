"""Q-field object model, defect analysis, and visualization helpers."""

# ---------------------------------------------------------------------------
# Planned architecture direction
# ---------------------------------------------------------------------------
#
# The long-term structure is expected to move toward a shared-grid dataset
# model:
#
#     GridFieldDataset
#         -> FieldData("Q")
#             -> QFieldObject
#
# In that design:
#
# - GridFieldDataset defines the common grid, boundary conditions, and later
#   shared interpolation / differential operators for all physical fields.
# - Q becomes one FieldData entry inside the dataset, alongside future fields
#   such as velocity, concentration, or active-force-related fields.
# - QFieldObject becomes the Q-specific analysis layer attached to that Q
#   field, rather than the owner of the whole grid/data system.
#
# The intended initialization paths for QFieldObject are:
#
# 1. Standalone / legacy-style initialization
#    - User passes raw Q-related input data, as in the current API.
#    - QFieldObject will create the needed GridFieldDataset and Q FieldData
#      automatically, then attach itself to that FieldData.
#
# 2. Attached analysis initialization
#    - User passes an existing parent Q FieldData (or an equivalent direct
#      reference to the dataset-owned Q field).
#    - QFieldObject does not rebuild the dataset or raw field storage, and only
#      performs Q-specific analysis on top of the existing field.
#
# Under that future structure, the canonical raw Q array should live on the
# dataset-owned FieldData, while QFieldObject serves as the analysis/view layer
# for defect detection, line classification, smoothing, visualization, and
# other Q-specific workflows.
#
# ---------------------------------------------------------------------------
# Staged migration plan for this legacy implementation
# ---------------------------------------------------------------------------
#
# This file is kept as the pre-migration reference implementation. The intended
# migration should be incremental rather than a full rewrite, so existing Q
# workflows remain usable while data ownership moves into the shared-grid model.
#
# Phase 1. Strengthen GridFieldDataset as the shared grid owner
# - Add shared geometry/cache state to GridFieldDataset:
#   calc_grid_index, calc_grid, calc_corners_index, calc_corners,
#   calc_box_size_periodic_index.
# - Move the grid/bounds construction logic currently implemented here into the
#   dataset layer, but keep QFieldObject behavior unchanged for callers.
# - Keep FieldData thin; it only needs enough surface to expose its values and
#   owning dataset cleanly.
#
# Phase 2. Move canonical raw Q ownership into dataset-owned FieldData
# - When QFieldObject is created from legacy raw inputs, create a
#   GridFieldDataset and bind FieldData("Q") internally.
# - Treat that dataset-owned FieldData as the source of truth for the raw Q
#   array, even if QFieldObject still exposes compatibility attributes such as
#   raw_Q during the transition.
# - Keep raw_S/raw_n compatibility on QFieldObject initially, but stop treating
#   QFieldObject itself as the long-term owner of raw field storage.
#
# Phase 3. Support two real initialization paths for QFieldObject
# - Legacy path: user passes raw Q/S/n input and QFieldObject builds the
#   dataset + Q field automatically.
# - Attached-analysis path: user passes an existing Q FieldData (or equivalent
#   dataset-owned Q reference), and QFieldObject attaches analysis behavior
#   without rebuilding the dataset or duplicating raw storage.
# - This phase makes the architecture comment above operational rather than
#   aspirational.
#
# Phase 4. Migrate shared-grid-dependent tools to dataset/field ownership
# - Update QInterpolator, plane helpers, and other grid-aware utilities to read
#   grid geometry, bounds, and periodic information from GridFieldDataset.
# - Keep QFieldObject convenience properties as compatibility facades, e.g.
#   calc_grid -> self.dataset.calc_grid and calc_corners -> self.dataset.calc_corners.
# - After this phase, QFieldObject should mainly provide Q-specific analysis,
#   not generic grid infrastructure.
#
# Phase 5. Clean the API and open the door for multi-field datasets
# - Deprecate or narrow QFieldObject attributes that really belong to the
#   dataset layer.
# - Let GridFieldDataset own multiple physical fields beyond Q, such as
#   velocity, concentration, or active-force-related fields.
# - Re-evaluate whether a dedicated QFieldData subclass is needed; until there
#   is a strong reason, prefer attaching QFieldObject to generic FieldData.
#
# Migration rule of thumb
# - First migrate ownership.
# - Then migrate shared geometry/cache computation.
# - Then migrate analysis attachment paths.
# - Finally clean compatibility shims and public API wording.

import time
from dataclasses import replace, dataclass, fields
from typing import Any, ClassVar, Mapping, Union

import numpy as np
from pyvistaqt import BackgroundPlotter
import pyvista as pv

from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import (
    Vect,
    as_Vect,
    Tensor,
    QField5,
    QField9,
    as_qfield5,
    SField,
    nField,
    check_Sn,
    Number,
    as_Number,
    DimensionFlagInput,
    as_dimension_info,
    check_bool_flags,
    UNSET,
    Unset,
    as_bool,
)
from ..field import (
    Q_diagonalize,
    getQ,
)
from ..grid import (
    GRID_TRANSFORM_IDENTITY,
    as_grid_transform,
    apply_linear_transform,
)
from ..disclination import defect_detect, defect_classify_into_lines
from .visual.plot_tube import OptsTube
from .visual.plot_rod import OptsRod
from .visual.plot_sphere import OptsSphere
from .visual.plot_surface import OptsSurface
from .visual.plot_figure import PlotFigure, OptsFigure
from .q_plane import QPlane, QPlanePolar
from .visual.figure_manager import FigureManager
from .plane_grid import OptsPlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar
from .bounds import as_bounds
from .grid_field import FieldData, GridFieldDataset, GridInterpolator, InputGridField
from .opts import merge_opts_all, cover_value
from ..general import blue_red_in_white_bg, sample_far
from .smoothed_line import OptsSmooth
from .registry_base import RegistryBase
from .disclination_line import DisclinationLine
from .class_base import ClassBase


@dataclass(slots=True)
class InputQ:
    """
    Validated input bundle for initializing a `QFieldObject`.

    At least one field description must be provided:

    - provide `Q`, or
    - provide `n`, optionally together with `S`.

    If `n` is provided while `S` is omitted, `S=1` is used everywhere.
    If both `Q` and `n` are provided, `n`/`S` take priority and `Q` is ignored.

    Parameters
    ----------
    Q
        Q-tensor field on the lattice. Compatible input representations are
        accepted and normalized to the internal `QField5` representation.
    S
        Scalar order parameter field with shape matching `n.shape[:3]`.
        Used together with `n` to reconstruct `Q`.
    n
        Director field with shape `(..., 3)`. Used to reconstruct `Q` when a
        raw Q-tensor field is not supplied or should be overridden.
    box_periodic_flag
        Periodic-boundary-condition flags for the three lattice directions.
    grid_offset
        Translation offset that maps lattice indices to real-space coordinates.
    grid_transform
        3x3 linear transform that maps lattice indices to real-space
        coordinates.
    default_miminum_line_length_smooth
        Default minimum disclination-line length required for smoothing.
    default_smooth_window_length
        Default smoothing window length used for line smoothing.
    default_miminum_line_length_visual
        Default minimum disclination-line length required for visualization.
    """

    Q: Union[QField5, QField9] | Unset = UNSET
    S: SField | Unset = UNSET
    n: nField | Unset = UNSET
    box_periodic_flag: DimensionFlagInput = False
    grid_offset: Vect(3) | None = None
    grid_transform: Tensor((3, 3)) = GRID_TRANSFORM_IDENTITY
    default_miminum_line_length_smooth: Number = 61
    default_smooth_window_length: Number = 41
    default_miminum_line_length_visual: Number = 75

    __attrs__ = {
        "Q": "Q field (tensor order parameter)",
        "S": "S field (scalar order parameter)",
        "n": "director field",
        "box_periodic_flag": (
            "flag indicating whether periodic boundary condition is applied "
            "along each dimension"
        ),
        "grid_offset": (
            "grid translation offset to map lattice indices to real-space "
            "coordinates"
        ),
        "grid_transform": (
            "grid transform matrix to map lattice indices to real-space "
            "coordinates (3x3)"
        ),
        "default_miminum_line_length_smooth": (
            "the minimum length (#points) of disclination lines to be smoothed"
        ),
        "default_smooth_window_length": (
            "the default window length (#points) of disclination lines to be "
            "smoothed"
        ),
        "default_miminum_line_length_visual": (
            "the minimum length (#points) of disclination lines to be visualized"
        ),
    }

    _validators = {
        "Q": lambda v, d: as_qfield5(v, name=d),
        "n": lambda v, d: check_Sn(v, "n"),
        "S": lambda v, d: check_Sn(v, "S"),
        "box_periodic_flag": lambda v, d: as_dimension_info(v, name=d, is_bool=True),
        "grid_offset": lambda v, d: None if v is None else as_Vect(v, name=d),
        "grid_transform": lambda v, d: as_grid_transform(v, name=d),
        "default_miminum_line_length_smooth": lambda v, d: as_Number(
            v, name=d, value_range=(1, np.inf)
        ),
        "default_smooth_window_length": lambda v, d: as_Number(
            v, name=d, value_range=(2, np.inf)
        ),
        "default_miminum_line_length_visual": lambda v, d: as_Number(
            v, name=d, value_range=(2, np.inf)
        ),
    }

    # ==================== OVERRIDE ====================
    # InputQ overrides dataclass assignment so every field stays validated both
    # during initialization and during later interactive edits.
    # ==================================================
    def __setattr__(self, key, value):
        if key in self._validators:
            if value is not UNSET:
                desc = f"{key!r}: {self.__class__.__attrs__[key]}"
                value = self._validators[key](value, desc)
        object.__setattr__(self, key, value)


class QFieldObject(ClassBase):
    """
    QFieldObject stores a Q-tensor field together with derived geometry,
    detected defects, disclination lines, and common visualization helpers.

    Important readable attributes:

    - `name`: identity of this Q field object.
    - `S`: scalar-order field derived from or paired with the Q data.
    - `n`: director field derived from or paired with the Q data.
    - `lines`: classified disclination lines registered under this Q field.
    - `figures` / `figs`: FigureManager storing figures created from this Q field.
    - `objects` / `objs`: RegistryBase storing physical objects derived from this Q field.
    - `interpolator`: GridInterpolator used for off-grid sampling.
    - `calc_grid`: full real-space lattice coordinates of the Q field.
    - `calc_corners`: Bounds object describing the Q-field box.
    - `calc_defect_indices` / `calc_defect_grid`: detected defect positions
      in index and world coordinates.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable Q-field attributes.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show bound figures, object registry, and interpolator.
    - `show_relation_tree()`: show how this Q field connects to derived objects.

    Common user actions:

    - `act_defect_detect()`: detect defect points from the director field.
    - `act_lines_classify()`: classify detected defects into disclination lines.
    - `act_lines_smooth(...)`: smooth eligible classified lines.
    - `act_add_interpolator()`: create and bind a GridInterpolator if absent.
    - `act_interpolate(points, ...)`: interpolate the Q field at arbitrary points.
    - `act_visualize_disclination_lines(...)`: draw disclination lines on a figure.
    - `act_visualize_n_plane(...)`: create a Cartesian director analysis plane.
    - `act_visualize_S_plane(...)`: create a Cartesian scalar-order analysis plane.
    - `act_visualize_n_near_defect(...)`: create a polar director analysis
      plane around a smoothed line.

    Representation:

    - `str(obj)` returns the short ClassBase-style identity.
    - `repr(obj)` returns the compact ClassBase summary.
    """

    # fmt: off
    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this Q tensor object.",
        },
        "raw_Q": {
            "doc": (
                "Raw Q-tensor field on lattice. Typically QField5 or QField9 "
                "(shape: (Nx, Ny, Nz, ...))."
            ),
        },
        "raw_S": {
            "doc": "Raw scalar order parameter field S on lattice (shape: (Nx, Ny, Nz)).",
        },
        "raw_n": {
            "doc": "Raw director field n on lattice (shape: (Nx, Ny, Nz, 3)).",
        },
        "raw_box_periodic_flag": {
            "doc": "Per-dimension periodic boundary condition flags (bool array-like of length 3).",
        },
        "raw_grid_offset": {
            "doc": (
                "A 3D vector, as the grid translation offset mapping lattice "
                "indices -> real-space coordinates."
            ),
        },
        "raw_grid_transform": {
            "doc": (
                "A 3x3 tensor, as the linear transform mapping lattice "
                "indices -> real-space coordinates"
            ),
        },
        "default_miminum_line_length_smooth": {
            "doc": "Default minimum line length (#points) required to apply smoothing.",
            "kind": "default",
        },
        "default_smooth_window_length": {
            "doc": "Default smoothing window length (#points) used when not specified.",
            "kind": "default",
        },
        "default_miminum_line_length_visual": {
            "doc": "Default minimum line length (#points) required for visualization.",
            "kind": "default",
        },
        "calc_grid_index": {
            "doc": "Lattice coordinate grid in index space (before applying transform/offset).",
            "kind": "property",
        },
        "calc_grid": {
            "doc": "Coordinate grid in real space after applying grid_transform and grid_offset.",
            "kind": "property",
        },
        "calc_corners_index": {
            "doc": "Box corners in lattice-index space.",
            "kind": "property",
        },
        "calc_corners": {
            "doc": "Bounds object describing the Q-field box in real-space coordinates.",
            "kind": "property",
        },
        "calc_box_size_periodic_index": {
            "doc": (
                "Effective periodic box size in index units. "
                "For periodic dims equals grid size, otherwise inf."
            ),
            "kind": "property",
        },
        "calc_defect_indices": {
            "doc": "Indices (lattice coordinates) of detected defect points.",
            "kind": "calc",
        },
        "calc_defect_grid": {
            "doc": "Real-space coordinates of detected defect points.",
            "kind": "calc",
        },
        "dataset": {
            "doc": "Shared-grid dataset that owns the canonical raw Q field.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "field": {
            "doc": "Dataset-owned FieldData entry storing the canonical raw Q values.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "figures": {
            "doc": "FigureManager object that manages PlotFigure objects created for this Q field.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "objects": {
            "doc": "RegistryBase object that manages physical objects related to this Q field.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "interpolator": {
            "doc": "The grid interpolator object associated with this Q field.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "lines": {
            "doc": "Read-only: Classified disclination lines.",
            "kind": "property",
        },
        "figs": {
            "doc": "Read-only: Visualization figures. Alias of `figures`.",
            "kind": "property",
        },
        "objs": {
            "doc": "Read-only: Physical objects. Alias of `objects`.",
            "kind": "property",
        },
    }
    # fmt: on

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
    # QFieldObject overrides ClassBase.__init__ because this class is not just a
    # passive data container. Initialization must:
    # - accept either raw Q/S/n input or an already prepared Q field,
    # - derive the companion S/n views when needed,
    # - attach the shared grid/dataset context used by later operations,
    # - create the object registry used to hold bounds, lines, planes, and
    #   other physical objects derived from this Q field,
    # - optionally run defect detection / line classification immediately.
    # ==================================================
    @logging_and_warning_decorator()
    def __init__(
        self,
        is_detect_defects: bool = True,
        is_classify_lines: bool = True,
        inputValue: InputQ | None = None,
        field: FieldData | None = None,
        name: str = "Q",
        logger=None,
        **kwargs,
    ) -> None:

        # Initialize the base identity/protection layer first so every later
        # relation and managed attribute attaches to a fully formed ClassBase.
        super().__init__(name=name, name_replace="Q", is_fixed=True)

        # `objects` is the catch-all registry for physical or geometric objects
        # derived from this Q field, for example bounds, disclination lines, and
        # analysis planes created later by user actions.
        objects = RegistryBase(
            "objects manager",
            info=f"physical objects attached to Q field {self.name!r}",
        )
        # Bind the registry both ways so derived objects can find their owning
        # QFieldObject and the QFieldObject can expose them through `self.objects`.
        self.act_bind_relation_base("objects", objects, is_weak=False)
        objects.act_bind_relation_base("owner", self, is_weak=True)

        logger.progress(f"Start to initialize Q tensor `{self.name}`.")
        if field is not None:
            # If a prepared field is provided, treat it as the source Q data and
            # build only the QFieldObject analysis layer around it.
            invalid_kwargs = [key for key in kwargs if not key.startswith("default_")]
            if invalid_kwargs:
                raise ValueError(
                    "Attached-analysis initialization via `field=...` only accepts "
                    "default_* override kwargs. Invalid key(s): "
                    f"{invalid_kwargs!r}."
                )

            # Attached initialization may still customize analysis defaults such
            # as smoothing/visual thresholds, but the raw Q data and grid model
            # must come entirely from the attached field + dataset pair.
            attached_defaults = InputQ() if inputValue is None else inputValue
            attached_defaults = merge_opts_all(
                {"": attached_defaults},
                kwargs,
                type(self).__name__,
            )[""]

            # Raw-Q and grid-related values always come from the attached
            # field/dataset pair. If the caller also passes these values through
            # InputQ, they are ignored here. To change them, create a new field
            # or a new QFieldObject from raw inputs instead of mixing both
            # initialization styles in one call.
            ignored_attached_inputs = [
                attr_name
                for attr_name in ("Q", "S", "n")
                if getattr(attached_defaults, attr_name) is not UNSET
            ]
            if ignored_attached_inputs:
                logger.warning(
                    "Attached-analysis initialization received extra raw field "
                    f"input(s) {ignored_attached_inputs!r}. These values are "
                    "ignored because the attached `field` already defines the "
                    "Q/S/n data used by this object. If you want to change the "
                    "underlying field data, please create a new QFieldObject "
                    "from raw inputs instead."
                )

            for attr_name in (
                "default_miminum_line_length_smooth",
                "default_smooth_window_length",
                "default_miminum_line_length_visual",
            ):
                object.__setattr__(
                    self,
                    attr_name,
                    getattr(attached_defaults, attr_name),
                )

            # The attached-analysis path is only meaningful when `field` is the
            # repository FieldData wrapper, because later code expects the field
            # to expose both its raw values and its owning dataset relation.
            if not isinstance(field, FieldData):
                raise TypeError(
                    "`field` must be a dataset-owned FieldData instance for "
                    "attached-analysis initialization."
                )
            dataset = field.owner
            if not isinstance(dataset, GridFieldDataset):
                raise TypeError(
                    "Attached-analysis initialization requires a FieldData whose "
                    "owner is a GridFieldDataset."
                )

            field.raw_values = as_qfield5(
                field.raw_values,
                name="attached Q field values",
            )

            # Reconstruct S and n from the provided Q values so the rest of the
            # class can keep using the same readable surfaces regardless of how
            # this object was initialized.
            object.__setattr__(self, "raw_Q", field.raw_values)
            temp_S, temp_n = Q_diagonalize(self.raw_Q)
            object.__setattr__(self, "raw_S", temp_S)
            object.__setattr__(self, "raw_n", temp_n)
            object.__setattr__(
                self,
                "raw_box_periodic_flag",
                dataset.raw_box_periodic_flag,
            )
            object.__setattr__(self, "raw_grid_offset", dataset.raw_grid_offset)
            object.__setattr__(self, "raw_grid_transform", dataset.raw_grid_transform)
        else:
            # Standalone path: accept raw Q/S/n-style input, normalize it into a
            # canonical Q field, and then continue exactly as if that field had
            # already been prepared elsewhere.
            if inputValue is None:
                inputValue = InputQ()

            inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[
                ""
            ]
            for f in fields(inputValue):
                k = f.name
                v = getattr(inputValue, k)
                if k.startswith("default"):
                    object.__setattr__(self, k, v)
                else:
                    object.__setattr__(self, f"raw_{k}", v)

            # Standalone initialization accepts either n/S or raw Q:
            # - if n is given, rebuild Q from n and S;
            # - otherwise, diagonalize the provided Q to recover S and n.
            # This keeps the three views synchronized before later analysis.
            if self.raw_n is not UNSET:
                logger.debug("Initialize Q field with S and n")
                if self.raw_S is UNSET:
                    logger.warning("No S input. Set to 1 everywhere.")
                    object.__setattr__(
                        self, "raw_S", np.zeros(np.shape(self.raw_n)[:-1]) + 1.0
                    )
                if self.raw_Q is not UNSET:
                    logger.warning(
                        "Both Q and n are provided to initialize Q field. Q will be IGNORED."
                    )
                if np.shape(self.raw_S) != np.shape(self.raw_n)[:3]:
                    raise ValueError(
                        "Shape mismatch between director field `n` and scalar field `S`: "
                        f"expected n.shape[:3] == S.shape, "
                        f"but got n.shape = {self.raw_n.shape}, S.shape = {self.raw_S.shape}."
                    )
                object.__setattr__(
                    self, "raw_Q", as_qfield5(getQ(self.raw_n, S=self.raw_S))
                )
            else:
                if self.raw_Q is not UNSET:
                    temp_S, temp_n = Q_diagonalize(self.raw_Q)
                    object.__setattr__(self, "raw_S", temp_S)
                    object.__setattr__(self, "raw_n", temp_n)
                else:
                    raise NameError("No data is input to initialize Q field.")

            # Build the shared grid container even for standalone construction,
            # so this Q field and any future sibling fields can live on the
            # same grid model.
            dataset = GridFieldDataset(
                inputValue=InputGridField(
                    shape=np.shape(self.raw_Q)[:3],
                    box_periodic_flag=self.raw_box_periodic_flag,
                    grid_offset=self.raw_grid_offset,
                    grid_transform=self.raw_grid_transform,
                ),
                name=f"{self.name} dataset",
            )
            field = dataset.act_add_field("Q", self.raw_Q)

        self.act_bind_relation_base("dataset", dataset, is_weak=False)
        self.act_bind_relation_base("field", field, is_weak=False)

        # After both relations are bound, sanity-check that they are consistent:
        # the canonical field used by this QFieldObject must actually belong to
        # the canonical dataset bound above.
        if field.owner is not dataset:
            raise RuntimeError(
                "QFieldObject internal binding error: the attached field is not "
                "owned by the attached dataset."
            )

        if field.name != "Q":
            logger.warning(
                f"Attached field name is {field.name!r}, not 'Q'. "
                "This is allowed for now, but QFieldObject will still treat it "
                "as the canonical Q field."
            )

        # From here on, `field.raw_values` is the Q data actually used by this
        # object. We mirror it onto `raw_Q` so existing methods can keep reading
        # `self.raw_Q` without caring how the field was supplied.
        object.__setattr__(self, "raw_Q", field.raw_values)
        # Register the box bounds as a normal derived object so they show up in
        # the same object registry as other geometry derived from this Q field.
        bounds = self.calc_corners
        self.objs.act_register(bounds, is_contain_ok=True)
        logger.debug(
            f"Box corners in lattice-index units is {self.calc_corners_index}."
            f"Box bounds in real-space coordinates is {self.calc_corners}."
        )

        if (not is_detect_defects) and is_classify_lines:
            is_classify_lines = False
            msg = (
                f"Invalid combination: is_detect_defects={is_detect_defects} "
                f"and is_classify_lines={is_classify_lines}.\n"
                "Line classification depends on defect detection. "
                "Automatically disabling line classification."
            )
            logger.warning(msg)

        if is_detect_defects:

            start = time.time()

            msg = "Start defect analysis as detecting defects"
            if is_classify_lines:
                msg += " and classifying them into distinct lines"
            msg += f" for Q tensor `{self.name}` \n"
            msg += "This operation might take a while.\n"
            msg += (
                "You can disable this automatic operation by setting "
                "is_detect_defects=False and is_classify_lines=False when "
                "initializing the Q tensor."
            )
            logger.progress(msg)

            self.act_defect_detect()

            if is_classify_lines:
                self.act_lines_classify()

            logger.progress(
                f"Defect analysis is finished, with {time.time()-start:.2f} s"
            )
        # Create the interpolator eagerly so later sampling/plane-visualization
        # actions can assume `self.interpolator` already exists.
        self.act_add_interpolator()
        # `figures` manages all PlotFigure windows created from this Q field and
        # tracks which one is currently active for later visualization calls.
        figures = FigureManager()
        self.act_bind_relation_base("figures", figures, is_weak=False)
        figures.act_bind_relation_base("owner", self, is_weak=True)

    # -------------------------------
    # Defect and line analysis
    # -------------------------------
    @logging_and_warning_decorator(start_finish_level=5)
    def act_defect_detect(self, logger=None):
        """
        Detect defect points from the current director field.

        This updates both `calc_defect_indices` in lattice-index coordinates
        and `calc_defect_grid` in real-space coordinates using the current
        grid transform and offset.
        """
        object.__setattr__(
            self,
            "calc_defect_indices",
            defect_detect(
                self.raw_n,
                is_boundary_periodic=self.raw_box_periodic_flag,
            ),
        )
        logger.info(f"{len(self.calc_defect_indices)} defects are found.")

        object.__setattr__(
            self,
            "calc_defect_grid",
            apply_linear_transform(
                self.calc_defect_indices,
                transform=self.raw_grid_transform,
                offset=self.raw_grid_offset,
            ),
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def act_lines_classify(self, logger=None):
        """
        Classify detected defect points into disclination lines.

        The classified lines are sorted by defect count, renamed in display
        order, registered into `self.objects`, and returned as a list.
        """
        lines = defect_classify_into_lines(
            self.calc_defect_indices,
            box_size_periodic=self.calc_box_size_periodic_index,
            grid_offset=self.raw_grid_offset,
            grid_transform=self.raw_grid_transform,
        )
        lines = sorted(lines, key=lambda line: line.calc_defect_num, reverse=True)
        for i, line in enumerate(lines):
            line.name = f"disclination line {i}"
            self.objects.act_register(line)

        logger.info(f"{len(lines)} lines are found.")

        return lines

    @logging_and_warning_decorator()
    def act_lines_smooth(
        self,
        opts: OptsSmooth | None = None,
        logger=None,
        **kwargs,
    ):
        """
        Smooth eligible disclination lines using shared smoothing options.

        Lines shorter than the configured minimum length are skipped. Missing
        smoothing defaults are filled from the Q-field object before
        delegating the actual smoothing to each line.

        Parameters
        ----------
        opts
            Base `OptsSmooth` configuration applied to all candidate lines.
        **kwargs
            Keyword overrides merged into `opts` before smoothing. Supported
            keys are the fields of `OptsSmooth`, including commonly used
            options such as `window_length`, `window_ratio`,
            `min_line_length`, and `order`.

        Notes
        -----
        If `min_line_length` is not provided, the method uses
        `self.default_miminum_line_length_smooth`.

        If both `window_length` and `window_ratio` are omitted, the method
        uses `self.default_smooth_window_length` as the default window length.

        If both `window_length` and `window_ratio` are provided,
        `window_length` takes priority and `window_ratio` is ignored.

        Examples
        --------
        Smooth all eligible lines with the object defaults::

            q.act_lines_smooth()

        Smooth using an explicit window length::

            q.act_lines_smooth(window_length=31)

        Smooth only sufficiently long lines::

            q.act_lines_smooth(min_line_length=100, window_ratio=8)

        See Also
        --------
        OptsSmooth
            Full smoothing-option container used by each line.
        """
        if opts is None:
            opts = OptsSmooth()

        opts = merge_opts_all({"": opts}, kwargs, "SmoothedLine")[""]

        if opts.min_line_length is UNSET:
            opts.min_line_length = self.default_miminum_line_length_smooth
            msg = "No input value provided for minimum smoothed line length. \n"
            msg += (
                "Using the default value "
                "self.default_miminum_line_length_smooth="
                f"{self.default_miminum_line_length_smooth}."
            )
            logger.info(msg)

        opts.act_finalize()

        if opts.window_length is not None and opts.window_ratio is not None:
            msg = (
                "``window_length`` of smoothing disclination lines is manual "
                f"input as {opts.window_length}.\n"
            )
            msg += f"``window_ratio`` as {opts.window_ratio} would be ignored."
            logger.warning(msg)
            opts.window_ratio = None

        if opts.window_length is None and opts.window_ratio is None:
            opts.window_length = self.default_smooth_window_length
            msg = "No input value provided for smooth window length of disclination lines. \n"
            msg += (
                "Using the default value self.default_smooth_window_length="
                f"{self.default_smooth_window_length}."
            )
            logger.info(msg)

        msg = (
            f"Start to smooth disclination lines in Q tensor {self.name!r} "
            "With paramaters: \n"
        )
        msg += f"window length = {opts.window_length}\n"
        msg += f"window ratio = {opts.window_ratio}\n"
        msg += f"minimum smoothed line length = {opts.min_line_length}"
        logger.debug(msg)

        num_smooth = 0
        window_list = {}
        for line in self.lines:
            if line.calc_defect_num >= opts.min_line_length:
                line.act_smooth(opts=opts, is_window_warning=False)
                num_smooth += 1
                window_list[line.name] = line.smooth.opts.window_length
            else:
                logger.debug(
                    f"Line `{line.name}` is not smoothed because it is too "
                    f"short, with only {line.calc_defect_num} defects. "
                )

        msg = (
            f"There are {len(self.lines)} disclination lines in total, with "
            f"{num_smooth} lines are smoothed.\n"
        )
        msg += "The smoothing window length is: "
        if opts.window_length is not None:
            msg += str(opts.window_length)
        else:
            msg += "\n"
            for k, v in window_list.items():
                msg += f"{k}: {v} \n"
        logger.info(msg)

    def act_add_interpolator(self):
        """Create and bind a `GridInterpolator` if one is not already present."""
        interpolator_old = self.interpolator
        if isinstance(interpolator_old, GridInterpolator):
            return interpolator_old

        interpolator = self.field.act_add_interpolator()
        self.act_bind_relation_base("interpolator", interpolator, is_weak=False)

        return self.interpolator

    def act_interpolate(
        self,
        points: np.ndarray,
        is_index=False,
        is_out_warning=False,
    ):
        """
        Interpolate the Q field at arbitrary sample points.

        Parameters
        ----------
        points
            Sample positions where the Q field should be evaluated.
        is_index
            If False, `points` are interpreted in real-space coordinates.
            If True, `points` are interpreted in lattice-index coordinates
            before interpolation.
        is_out_warning
            If True, warn when any sample point falls outside non-periodic
            dimensions and return those out-of-domain input points with the
            interpolated values.
        """
        if self.interpolator is None:
            self.act_add_interpolator()
        return self.field.act_interpolate(
            points,
            is_index=is_index,
            is_out_warning=is_out_warning,
        )

    # -------------------------------
    # Visualization helpers
    # -------------------------------

    @logging_and_warning_decorator()
    def _helper_set_figure(
        self,
        is_new: bool,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None,
        opts_figure: OptsFigure,
        title: str,
        logger=None,
    ):

        is_new = as_bool(is_new, name="Whether to create a new figure", replace=True)

        if is_new:
            if figure is not None:
                logger.warning(
                    "is_new=True was specified while figure is not None."
                    "The figure argument will be ignored and a new figure will be created."
                )
            figure = PlotFigure(opts=opts_figure, name=title)
        else:
            try:
                if isinstance(figure, (str, int)):
                    figure = self.figs[figure]
                    figure.act_commit(opts_figure)
                elif figure is None:
                    active_name = self.figs.active_name
                    if active_name is not None:
                        figure_active = self.figs[active_name]
                        if figure_active.is_alive:
                            figure = figure_active
                            figure.act_commit(opts_figure)
                        else:
                            figure = PlotFigure(opts=opts_figure, name=title)
                    elif len(self.figs) == 1 and self.figs[0].is_alive:
                        figure = self.figs[0]
                        figure.act_commit(opts_figure)
                    else:
                        figure = PlotFigure(opts=opts_figure, name=title)
                elif isinstance(figure, PlotFigure):
                    figure.act_commit(opts_figure)
                elif isinstance(figure, (BackgroundPlotter, pv.Plotter)):
                    figure = PlotFigure(plotter=figure, opts=opts_figure, name=title)
                else:
                    raise ValueError(
                        "`figure` input must be either index in FigureManager (str or int) "
                        "or a valid PlotFigure object, or a valid pyvistaqt "
                        "BackgroundPlotter object, "
                        "or None (creating a new figure) "
                        f"Got type {type(figure)!r} instead."
                    )
            except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                logger.exception("Could not find figure in FigureManager.")
                logger.recovery("Create a new figure instead.")
                figure = PlotFigure(opts=opts_figure, name=title)

        if figure.name.startswith(figure._DEFAULT_NAME):
            figure.act_set_name(title)
        self.figs.act_register(figure, is_contain_ok=True)
        self.figs.act_set_active(figure.name)

        return figure

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolve_visual_bounds(
        self, bounds=None, *, label: str = "plot", logger=None
    ):
        if bounds is None:
            return self.calc_corners

        try:
            bounds_obj = as_bounds(bounds, name=f"{label} bounds")
        except (TypeError, ValueError):
            logger.exception("Check input.")
            logger.recovery("Use the default Q bounds instead.")
            return self.calc_corners

        return bounds_obj

    @logging_and_warning_decorator()
    def act_visualize_disclination_lines(
        self,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
        is_new: bool = False,
        is_wrap: bool = True,
        is_smooth: bool = True,
        is_extent: bool = True,
        min_line_length: int | None = None,
        opts_figure: OptsFigure | None = None,
        opts_line: OptsTube | None = None,
        opts_extent: OptsTube | None = None,
        bounds=None,
        title: str = "disclination lines",
        logger=None,
        **kwargs,
    ):
        """
        Visualize classified disclination lines on a figure.

        Parameters
        ----------
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_wrap
            Whether periodic wrapping should be applied before plotting each
            line.
        is_smooth
            Whether smoothed line geometry should be used when available.
        is_extent
            Whether to also draw the bounding extent.
        is_wrap
            Whether the selected cross-section origin should be wrapped into
            the principal periodic box before the local polar grid is built.
        min_line_length
            Minimum defect count required for a line to be plotted. If not
            provided, `self.default_miminum_line_length_visual` is used.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_line
            Base `OptsTube` configuration for the plotted lines.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        bounds
            Bounds used for visualization and optional clipping. If omitted,
            the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        **kwargs
            Keyword overrides merged into `opts_figure`, `opts_line`, and
            `opts_extent` using the prefixes `figure_`, `line_`, and
            `extent_`.

        Examples
        --------
        Plot lines with the default visualization settings::

            q.act_visualize_disclination_lines()

        Plot only longer lines on a new figure::

            q.act_visualize_disclination_lines(is_new=True, min_line_length=100)

        Override line and extent options through keyword prefixes::

            q.act_visualize_disclination_lines(
                line_radius=0.8,
                extent_color=(0, 0, 0),
            )
        """
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_line is None:
            opts_line = OptsTube(color="sample_far")

        merge = merge_opts_all(
            {"figure_": opts_figure, "line_": opts_line, "extent_": opts_extent},
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_line = merge["line_"]
        opts_extent = merge["extent_"]

        check_bool_flags(locals())

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)
        bounds_input = bounds
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)
        line_bounds = None if not is_wrap and bounds_input is None else bounds

        if min_line_length is None:
            logger.info(
                "No minimum line length has been provided for the plotted lines. "
                f"Use the default value {self.default_miminum_line_length_visual}"
            )
            min_line_length = self.default_miminum_line_length_visual

        logger.debug(f"min_line_length = {min_line_length}")

        lines_plot = [
            line for line in self.lines if line.calc_defect_num >= min_line_length
        ]

        if opts_line.color == "sample_far":
            color_map = blue_red_in_white_bg()
            color_map_length = np.shape(color_map)[0] - 1
            lines_colors = color_map[
                (sample_far(len(lines_plot)) * color_map_length).astype(int)
            ]
        else:
            lines_colors = [opts_line.color for line in lines_plot]

        logger.debug("Start to draw disclination lines")
        for line, line_color in zip(lines_plot, lines_colors):
            opts_line = replace(opts_line, color=line_color)
            line.act_visualize(
                figure=figure,
                is_wrap=is_wrap,
                is_smooth=is_smooth,
                bounds=line_bounds,
                opts=opts_line,
            )

        if is_extent:
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    def act_visualize_n_plane(
        self,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
        is_new: bool = False,
        is_extent: bool = True,
        is_defect: bool = False,
        opts_grid: OptsPlaneGrid | None = None,
        opts_n: OptsRod | None = None,
        opts_nb: OptsRod | None = None,
        opts_nd: OptsRod | None = None,
        opts_figure: OptsFigure | None = None,
        opts_extent: OptsTube | None = None,
        opts_defect: OptsSphere | None = None,
        bounds=None,
        title: str = "visualization of n plane",
        plane_name: str = "n-plane",
        **kwargs,
    ):
        """
        Visualize the director field on a Cartesian analysis plane.

        This creates a `QPlane` from the current Q-field interpolator, then
        renders directors in bulk and near-defect regions on the target plane.

        Parameters
        ----------
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_extent
            Whether to also draw the bounding extent.
        is_defect
            Whether detected defect points on the plane should be visible.
        opts_grid
            Base `OptsPlaneGrid` configuration for constructing the analysis
            plane.
        opts_n
            Shared base `OptsRod` configuration copied into both bulk and
            near-defect director visuals unless those visuals override it.
        opts_nb
            `OptsRod` overrides for directors in bulk regions.
        opts_nd
            `OptsRod` overrides for directors near detected defects.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        opts_defect
            `OptsSphere` configuration for defect-point markers.
        bounds
            Bounds used for the plane construction and optional extent drawing.
            If omitted, the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        plane_name
            Name assigned to the generated `QPlane` object.
        **kwargs
            Keyword overrides merged into the option objects using the prefixes
            `figure_`, `grid_`, `extent_`, `n_`, `nb_`, `nd_`, and `defect_`.

        Examples
        --------
        Visualize the director plane with default settings::

            q.act_visualize_n_plane()

        Show defect markers on a new figure::

            q.act_visualize_n_plane(is_new=True, is_defect=True)

        Override plane-grid and director options through keyword prefixes::

            q.act_visualize_n_plane(
                grid_spacing=2,
                n_radius=0.4,
                defect_radius=1.5,
            )
        """
        if opts_grid is None:
            opts_grid = OptsPlaneGrid()
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_n is None:
            opts_n = OptsRod()
        if opts_nb is None:
            opts_nb = OptsRod()
        if opts_nd is None:
            opts_nd = OptsRod()
        if opts_defect is None:
            opts_defect = OptsSphere()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "grid_": opts_grid,
                "extent_": opts_extent,
                "n_": opts_n,
                "nb_": opts_nb,
                "nd_": opts_nd,
                "defect_": opts_defect,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_grid = merge["grid_"]
        opts_extent = merge["extent_"]
        opts_n = merge["n_"]
        opts_nb = merge["nb_"]
        opts_nd = merge["nd_"]
        opts_defect = merge["defect_"]

        cover_value(opts_nb, is_allow_cover_target_set=False, **(opts_n.act_asdict()))
        cover_value(opts_nd, is_allow_cover_target_set=False, **(opts_n.act_asdict()))

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)

        if self.interpolator is None:
            self.act_add_interpolator()

        n_plane = QPlane(
            self.interpolator,
            name=plane_name,
            opts=opts_grid,
            bounds=bounds,
            opts_defaults_override={
                "size": 1.8 * np.max(self.S.shape),
                "spacing": 1,
                "grid_offset": self.raw_grid_offset,
                "grid_transform": self.raw_grid_transform,
            },
        )
        self.objs.act_register(n_plane)

        n_plane.act_visualize_n(
            figure=figure,
            is_defect=is_defect,
            opts_nb=opts_nb,
            opts_nd=opts_nd,
            opts_defect=opts_defect,
        )

        if is_extent:
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    def act_visualize_S_plane(
        self,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
        is_new: bool = False,
        is_extent: bool = True,
        opts_grid: OptsPlaneGrid | None = None,
        opts_S: OptsSurface | None = None,
        opts_figure: OptsFigure | None = None,
        opts_extent: OptsTube | None = None,
        bounds=None,
        title: str = "visualization of S plane",
        plane_name: str = "S-plane",
        **kwargs,
    ):
        """
        Visualize the scalar order parameter on a Cartesian analysis plane.

        This creates a `QPlane` from the current Q-field interpolator, then
        renders the plane as an `S` surface on the target figure.

        Parameters
        ----------
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_extent
            Whether to also draw the bounding extent.
        opts_grid
            Base `OptsPlaneGrid` configuration for constructing the analysis
            plane.
        opts_S
            Base `OptsSurface` configuration for the rendered scalar-order
            surface.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        bounds
            Bounds used for the plane construction and optional extent drawing.
            If omitted, the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        plane_name
            Name assigned to the generated `QPlane` object.
        **kwargs
            Keyword overrides merged into the option objects using the prefixes
            `figure_`, `grid_`, `extent_`, and `S_`.

        Examples
        --------
        Visualize the scalar-order plane with default settings::

            q.act_visualize_S_plane()

        Create the plane on a new figure without the extent box::

            q.act_visualize_S_plane(is_new=True, is_extent=False)

        Override plane-grid and surface options through keyword prefixes::

            q.act_visualize_S_plane(
                grid_spacing=2,
                S_opacity=0.8,
            )
        """
        if opts_grid is None:
            opts_grid = OptsPlaneGrid()
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_S is None:
            opts_S = OptsSurface()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "grid_": opts_grid,
                "extent_": opts_extent,
                "S_": opts_S,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_grid = merge["grid_"]
        opts_extent = merge["extent_"]
        opts_S = merge["S_"]

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)

        if self.interpolator is None:
            self.act_add_interpolator()

        S_plane = QPlane(
            self.interpolator,
            name=plane_name,
            opts=opts_grid,
            bounds=bounds,
            opts_defaults_override={
                "size": 1.8 * np.max(self.S.shape),
                "spacing": 1,
                "grid_offset": self.raw_grid_offset,
                "grid_transform": self.raw_grid_transform,
            },
        )
        self.objs.act_register(S_plane)

        S_plane.act_visualize_S(
            figure=figure,
            opts_S=opts_S,
        )

        if is_extent:
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    def act_visualize_n_near_defect(
        self,
        u_percent: float,
        index_line: int = 0,
        index_smooth: int = -1,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
        is_new: bool = False,
        is_extent: bool = False,
        is_wrap: bool = True,
        opts_grid: OptsPlaneGridPolar | None = None,
        opts_n: OptsRod | None = None,
        opts_nb: OptsRod | None = None,
        opts_nd: OptsRod | None = None,
        opts_figure: OptsFigure | None = None,
        opts_extent: OptsTube | None = None,
        opts_defect: OptsSphere | None = None,
        bounds=None,
        title: str = "visualization of n near defect",
        plane_name: str | None = None,
        **kwargs,
    ):
        """
        Visualize the director field on a polar cross-section around a defect.

        This selects one smoothed disclination line by index, creates a
        `DefectSectionGrid`, wraps it as a `QPlanePolar`, and renders the
        director field near the selected cross-section.

        Parameters
        ----------
        u_percent
            Parametric position along the smoothed disclination line used to
            choose the cross-section.
        index_line
            Index of the classified disclination line in `self.lines`.
        index_smooth
            Index of the smoothed version under that line. Defaults to `-1`,
            the latest smoothed version.
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_extent
            Whether to also draw the bounding extent.
        opts_grid
            Base `OptsPlaneGridPolar` configuration for constructing the polar
            cross-section grid.
        opts_n
            Shared base `OptsRod` configuration copied into both bulk and
            near-defect director visuals unless those visuals override it.
        opts_nb
            `OptsRod` overrides for directors in bulk regions.
        opts_nd
            `OptsRod` overrides for directors near detected defects.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        opts_defect
            Reserved `OptsSphere` configuration for defect-point markers.
        bounds
            Bounds used for cross-section construction and optional extent
            drawing. If omitted, the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        plane_name
            Name assigned to the generated polar plane object.
        **kwargs
            Keyword overrides merged into the option objects using the prefixes
            `figure_`, `grid_`, `extent_`, `n_`, `nb_`, `nd_`, and `defect_`.

        Examples
        --------
        Visualize the director field near the middle of a smoothed line::

            q.act_visualize_n_near_defect(50, index_line=0)

        Create a new figure and override polar-grid settings::

            q.act_visualize_n_near_defect(
                25,
                index_line=1,
                index_smooth=-1,
                is_new=True,
                grid_layers=40,
                grid_arc_dist=0.4,
            )
        """

        if opts_grid is None:
            opts_grid = OptsPlaneGridPolar()
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_n is None:
            opts_n = OptsRod()
        if opts_nb is None:
            opts_nb = OptsRod()
        if opts_nd is None:
            opts_nd = OptsRod()
        if opts_defect is None:
            opts_defect = OptsSphere()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "grid_": opts_grid,
                "extent_": opts_extent,
                "n_": opts_n,
                "nb_": opts_nb,
                "nd_": opts_nd,
                "defect_": opts_defect,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_grid = merge["grid_"]
        opts_extent = merge["extent_"]
        opts_n = merge["n_"]
        opts_nb = merge["nb_"]
        opts_nd = merge["nd_"]
        opts_defect = merge["defect_"]

        cover_value(opts_nb, is_allow_cover_target_set=False, **(opts_n.act_asdict()))
        cover_value(opts_nd, is_allow_cover_target_set=False, **(opts_n.act_asdict()))

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)

        if self.interpolator is None:
            self.act_add_interpolator()

        try:
            line = self.lines[index_line]
        except IndexError as exc:
            raise IndexError(
                f"Invalid index_line={index_line!r}; "
                f"there are {len(self.lines)} disclination lines."
            ) from exc

        smooths = line.smooths
        if not smooths:
            raise ValueError(
                f"Disclination line {index_line!r} has no smoothed versions. "
                "Call `act_smooth()` on the line or `act_lines_smooth()` on "
                "the Q-field object first."
            )
        try:
            smooth = smooths[index_smooth]
        except IndexError as exc:
            raise IndexError(
                f"Invalid index_smooth={index_smooth!r} for line "
                f"{index_line!r}; there are {len(smooths)} smoothed versions."
            ) from exc

        section = smooth.act_cross_section(
            u_percent,
            opts_grid=opts_grid,
            name=plane_name,
            bounds=bounds,
            is_wrap=is_wrap,
        )
        plane_grid = section.wrapped
        n_plane_name = section.name + " of " + smooth.name
        n_plane = QPlanePolar(
            self.interpolator,
            name=n_plane_name,
            grid=plane_grid,
        )
        if plane_grid.wrapper is section:
            plane_grid.act_unbind_wrapper()
            n_plane.grid.act_bind_wrapper(section, protected_attrs=["origin", "normal"])
        self.objs.act_register(n_plane)

        n_plane.act_visualize_n(
            figure=figure,
            opts_nb=opts_nb,
            opts_nd=opts_nd,
        )

        if is_extent:
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    # -------------------------------
    # Readable properties and array-style access
    # -------------------------------

    @property
    def calc_grid_index(self):
        """Return the dataset-owned lattice coordinate grid in index space."""
        return self.dataset.calc_grid_index

    @property
    def calc_grid(self):
        """Return the dataset-owned coordinate grid in real space."""
        return self.dataset.calc_grid

    @property
    def calc_corners_index(self):
        """Return the dataset-owned box corners in lattice-index space."""
        return self.dataset.calc_corners_index

    @property
    def calc_corners(self):
        """Return the dataset-owned bounds object for this Q field."""
        return self.dataset.calc_corners

    @property
    def calc_box_size_periodic_index(self):
        """Return the dataset-owned periodic box size in index units."""
        return self.dataset.calc_box_size_periodic_index

    @property
    def lines(self):
        """Return registered disclination-line objects."""
        result = [item for item in self.objects if isinstance(item, DisclinationLine)]
        return result

    @property
    def figs(self):
        """Return the figure manager bound to this Q field."""
        return self.figures

    @property
    def objs(self):
        """Return the object registry bound to this Q field."""
        return self.objects

    def __call__(self) -> np.ndarray:
        return self.raw_Q
