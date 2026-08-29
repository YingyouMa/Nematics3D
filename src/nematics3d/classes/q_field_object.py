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
# - Add shared lightweight geometry state to GridFieldDataset:
#   calc_corners_index, calc_corners, calc_bounds, and
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
# - Keep lightweight QFieldObject convenience properties as facades, e.g.
#   calc_bounds -> self.dataset.calc_bounds. Full coordinate grids are explicit
#   dataset allocations through act_generate_grid().
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
    QField5,
    QField9,
    as_qfield5,
    SField,
    nField,
    MaskField,
    as_director_field,
    as_scalar_field,
    Number,
    as_number,
    DimensionInfo,
    as_dimension_info,
    UNSET,
    Unset,
    as_bool,
)
from ..field import get_q
from ..analysis.q_diagonalization import q_diagonalize
from ..analysis.sampling import sample_van_der_corput
from ..grid import (
    GRID_TRANSFORM_IDENTITY,
    GridTransform,
    as_grid_offset,
    as_grid_transform,
    apply_linear_transform,
)
from ..analysis.disclination import (
    defect_detect,
    defect_classify_into_lines,
    defect_validity_from_mask,
)
from .visual.plot_tube import OptsTube
from .visual.plot_rod import OptsRod
from .visual.plot_sphere import OptsSphere
from .visual.plot_delaunay import OptsDelaunay
from .visual.plot_figure import PlotFigure, OptsFigure
from .visual.color import blue_green_red_colors
from .q_plane import QPlane, QPlanePolar
from .visual.figure_manager import FigureManager
from .plane_grid import OptsPlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar
from .bounds import as_bounds
from .grid_field import FieldData, GridFieldDataset, GridInterpolator, InputGridField
from .opts import merge_opts_all, cover_value
from .smoothed_line import OptsSmooth
from .registry_base import RegistryBase
from .disclination_line import DisclinationLine
from .class_base import AttrDef, ClassBase


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
    mask
        Boolean validity field with shape matching the lattice grid
        `(Nx, Ny, Nz)`. True marks voxels where the Q data is physically
        meaningful; False marks voxels whose values must not enter any
        derived analysis. For example, defects supported by invalid voxels
        are excluded from defect analysis. If omitted, every voxel is
        treated as valid.
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
    mask: MaskField | Unset = UNSET
    box_periodic_flag: DimensionInfo = False
    grid_offset: Vect(3) | None = None
    grid_transform: GridTransform = GRID_TRANSFORM_IDENTITY
    default_miminum_line_length_smooth: Number = 61
    default_smooth_window_length: Number = 41
    default_miminum_line_length_visual: Number = 75

    __attrs__ = {
        "Q": "Q field (tensor order parameter)",
        "S": "S field (scalar order parameter)",
        "n": "director field",
        "mask": (
            "validity mask marking which voxels carry physically meaningful " "Q data"
        ),
        "box_periodic_flag": (
            "flag indicating whether periodic boundary condition is applied "
            "along each dimension"
        ),
        "grid_offset": (
            "grid translation offset to map lattice indices to real-space "
            "coordinates"
        ),
        "grid_transform": (
            "grid transform matrix that maps lattice indices to real-space "
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

    # `mask` is intentionally absent here: unlike Q/n/S, it is not used by
    # QFieldObject directly but forwarded as-is to the shared dataset, where
    # InputGridField validates it. Keeping a validator here would only duplicate
    # that check.
    _validators = {
        "Q": lambda v, d: as_qfield5(v, name=d),
        "n": lambda v, d: as_director_field(v, name=d, is_spatial_3d_required=True),
        "S": lambda v, d: as_scalar_field(v, name=d, is_spatial_3d_required=True),
        "box_periodic_flag": lambda v, d: as_dimension_info(
            v,
            name=d,
            is_bool=True,
        ),
        "grid_offset": lambda v, d: as_grid_offset(v, name=d),
        "grid_transform": lambda v, d: as_grid_transform(v, name=d),
        "default_miminum_line_length_smooth": lambda v, d: as_number(
            v, name=d, value_range=(1, np.inf)
        ),
        "default_smooth_window_length": lambda v, d: as_number(
            v, name=d, value_range=(2, np.inf)
        ),
        "default_miminum_line_length_visual": lambda v, d: as_number(
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
    - `calc_corners`: real-space box corner coordinates.
    - `calc_bounds`: Bounds object describing the Q-field box.
    - `calc_defect_indices` / `calc_defect_grid`: detected defect positions
      in index and world coordinates.
    - `mask`: boolean validity mask of the lattice, or None when the dataset
      has no mask. A read-only view of the dataset "mask" field.
    - `calc_defect_indices_masked`: detected defects excluded because their
      supporting plaquette touches invalid voxels of the validity mask.

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
    __attr_defs__: ClassVar = {
        "raw_Q": AttrDef(
            doc=(
                "Raw Q-tensor field on lattice. Typically QField5 or QField9 "
                "(shape: (Nx, Ny, Nz, ...))."
            ),
            kind="raw",
        ),
        "raw_S": AttrDef(
            doc="Raw scalar order parameter field S on lattice (shape: (Nx, Ny, Nz)).",
            kind="raw",
        ),
        "raw_n": AttrDef(
            doc="Raw director field n on lattice (shape: (Nx, Ny, Nz, 3)).",
            kind="raw",
        ),
        "raw_box_periodic_flag": AttrDef(
            doc="Per-dimension periodic boundary condition flags (bool array-like of length 3).",
            kind="raw",
        ),
        "raw_grid_offset": AttrDef(
            doc=(
                "A 3D vector, as the grid translation offset mapping lattice "
                "indices -> real-space coordinates."
            ),
            kind="raw",
        ),
        "raw_grid_transform": AttrDef(
            doc=(
                "A 3x3 tensor, as the linear transform mapping lattice "
                "indices -> real-space coordinates"
            ),
            kind="raw",
        ),
        "default_miminum_line_length_smooth": AttrDef(
            doc="Default minimum line length (#points) required to apply smoothing.",
            kind="default",
        ),
        "default_smooth_window_length": AttrDef(
            doc="Default smoothing window length (#points) used when not specified.",
            kind="default",
        ),
        "default_miminum_line_length_visual": AttrDef(
            doc="Default minimum line length (#points) required for visualization.",
            kind="default",
        ),
        "calc_corners_index": AttrDef(
            doc="Box corners in lattice-index space.",
            kind="property",
        ),
        "calc_corners": AttrDef(
            doc="Box corners in real-space coordinates.",
            kind="property",
        ),
        "calc_bounds": AttrDef(
            doc="Bounds object describing the Q-field box in real-space coordinates.",
            kind="property",
        ),
        "calc_box_size_periodic_index": AttrDef(
            doc=(
                "Effective periodic box size in index units. "
                "For periodic dims equals grid size, otherwise inf."
            ),
            kind="property",
        ),
        "calc_defect_indices": AttrDef(
            doc="Indices (lattice coordinates) of detected defect points.",
            kind="calc",
        ),
        "calc_defect_grid": AttrDef(
            doc="Real-space coordinates of detected defect points.",
            kind="calc",
        ),
        "calc_defect_indices_masked": AttrDef(
            doc=(
                "Indices (lattice coordinates) of detected defect points that "
                "were discarded because their supporting plaquette touches "
                "invalid voxels of the validity mask. Kept for inspection of "
                "the mask boundary; excluded from all downstream analysis."
            ),
            kind="calc",
        ),
        "dataset": AttrDef(
            doc="Shared-grid dataset that owns the canonical raw Q field.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "field": AttrDef(
            doc="Dataset-owned FieldData entry storing the canonical raw Q values.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "figures": AttrDef(
            doc="FigureManager object that manages PlotFigure objects created for this Q field.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "objects": AttrDef(
            doc="RegistryBase object that manages physical objects related to this Q field.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "interpolator": AttrDef(
            doc="The grid interpolator object associated with this Q field.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "mask": AttrDef(
            doc=(
                "Read-only: validity mask on lattice (bool, shape (Nx, Ny, Nz)) "
                "viewing the dataset 'mask' field, or None when the dataset has "
                "no mask. False marks voxels whose Q data is physically "
                "meaningless and must not enter derived analysis."
            ),
            kind="property",
        ),
        "lines": AttrDef(
            doc="Read-only: Classified disclination lines.",
            kind="property",
        ),
        "figs": AttrDef(
            doc="Read-only: Visualization figures. Alias of `figures`.",
            kind="property",
        ),
        "objs": AttrDef(
            doc="Read-only: Physical objects. Alias of `objects`.",
            kind="property",
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
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

        super().__init__(name=name, name_replace="Q", is_fixed=True)

        objects = RegistryBase(
            "objects manager",
            info=f"physical objects attached to Q field {self.name!r}",
        )
        self.act_bind_relation_base("objects", objects, is_weak=False)
        objects.act_bind_relation_base("owner", self, is_weak=True)

        logger.progress(f"Start to initialize Q tensor `{self.name}`.")
        if field is not None:
            invalid_kwargs = [key for key in kwargs if not key.startswith("default_")]
            if invalid_kwargs:
                raise ValueError(
                    "Attached-analysis initialization via `field=...` only accepts "
                    "default_* override kwargs. Invalid key(s): "
                    f"{invalid_kwargs!r}."
                )

            attached_defaults = InputQ() if inputValue is None else inputValue
            attached_defaults = merge_opts_all(
                {"": attached_defaults},
                kwargs,
                type(self).__name__,
            )[""]

            ignored_attached_inputs = [
                attr_name
                for attr_name in ("Q", "S", "n", "mask")
                if getattr(attached_defaults, attr_name) is not UNSET
            ]
            if ignored_attached_inputs:
                logger.warning(
                    "Attached-analysis initialization received extra raw field "
                    f"input(s) {ignored_attached_inputs!r}. These values are "
                    "ignored because the attached `field` and its dataset "
                    "already define the Q/S/n/mask data used by this object. "
                    "If you want to change the underlying field data, please "
                    "create a new QFieldObject from raw inputs instead. To "
                    "attach a validity mask, add it to the dataset as a field "
                    "named 'mask' before initializing this object."
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

            q_values = as_qfield5(
                field.raw_values,
                name="attached Q field values",
            )

            object.__setattr__(self, "raw_Q", q_values)
            diagonalization = q_diagonalize(self.raw_Q)
            object.__setattr__(self, "raw_S", diagonalization.S)
            object.__setattr__(self, "raw_n", diagonalization.n)
            object.__setattr__(
                self,
                "raw_box_periodic_flag",
                dataset.raw_box_periodic_flag,
            )
            object.__setattr__(self, "raw_grid_offset", dataset.raw_grid_offset)
            object.__setattr__(self, "raw_grid_transform", dataset.raw_grid_transform)
        else:
            if inputValue is None:
                inputValue = InputQ()

            inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[
                ""
            ]
            mask_input = inputValue.mask
            for f in fields(inputValue):
                k = f.name
                if k == "mask":
                    continue
                v = getattr(inputValue, k)
                if k.startswith("default"):
                    object.__setattr__(self, k, v)
                else:
                    object.__setattr__(self, f"raw_{k}", v)

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
                    self, "raw_Q", as_qfield5(get_q(self.raw_n, S=self.raw_S))
                )
            else:
                if self.raw_Q is not UNSET:
                    diagonalization = q_diagonalize(self.raw_Q)
                    object.__setattr__(self, "raw_S", diagonalization.S)
                    object.__setattr__(self, "raw_n", diagonalization.n)
                else:
                    raise NameError("No data is input to initialize Q field.")

            dataset = GridFieldDataset(
                inputValue=InputGridField(
                    shape=np.shape(self.raw_Q)[:3],
                    box_periodic_flag=self.raw_box_periodic_flag,
                    grid_offset=self.raw_grid_offset,
                    grid_transform=self.raw_grid_transform,
                    mask=mask_input,
                ),
                name=f"{self.name} dataset",
            )
            field = dataset.act_add_field("Q", self.raw_Q)

        self.act_bind_relation_base("dataset", dataset, is_weak=False)
        self.act_bind_relation_base("field", field, is_weak=False)

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

        object.__setattr__(self, "raw_Q", field.raw_values)

        mask = self.mask
        if mask is not None:
            logger.info(
                "Validity mask is active: "
                f"{int(np.count_nonzero(~mask))} of "
                f"{mask.size} voxels are invalid and will be "
                "excluded from defect analysis."
            )
        bounds = self.calc_bounds
        self.objs.act_register(bounds, is_contain_ok=True)
        logger.debug(
            f"Box corners in lattice-index units is {self.calc_corners_index}."
            f"Box bounds in real-space coordinates is {self.calc_bounds}."
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
        self.act_add_interpolator()
        figures = FigureManager()
        self.act_bind_relation_base("figures", figures, is_weak=False)
        figures.act_bind_relation_base("owner", self, is_weak=True)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_defect_detect(self, logger=None):
        """Detect defect points from the current director field."""
        defect_indices = defect_detect(
            self.raw_n,
            is_boundary_periodic=self.raw_box_periodic_flag,
            is_input_validated=True,
        )
        logger.info(f"{len(defect_indices)} defects are found.")

        mask = self.mask
        if mask is None:
            defect_indices_masked = np.empty((0, 3), dtype=float)
        else:
            validity = defect_validity_from_mask(
                defect_indices,
                mask,
                is_boundary_periodic=self.raw_box_periodic_flag,
            )
            defect_indices_masked = defect_indices[~validity]
            defect_indices = defect_indices[validity]
            logger.info(
                f"{len(defect_indices_masked)} defects are supported by "
                "invalid voxels of the validity mask. They are physically "
                "meaningless and excluded from defect analysis; the excluded "
                "points are kept in `calc_defect_indices_masked`. "
                f"{len(defect_indices)} valid defects remain."
            )

        object.__setattr__(self, "calc_defect_indices", defect_indices)
        object.__setattr__(self, "calc_defect_indices_masked", defect_indices_masked)

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
        """Classify detected defect points into disclination lines."""
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
        is_return_validity=False,
    ):
        if self.interpolator is None:
            self.act_add_interpolator()
        return self.field.act_interpolate(
            points,
            is_index=is_index,
            is_out_warning=is_out_warning,
            is_return_validity=is_return_validity,
        )

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
            return self.calc_bounds

        try:
            bounds_obj = as_bounds(bounds, name=f"{label} bounds")
        except (TypeError, ValueError):
            logger.exception("Check input.")
            logger.recovery("Use the default Q bounds instead.")
            return self.calc_bounds

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
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_line is None:
            opts_line = OptsTube(color="sample_van_der_corput")

        merge = merge_opts_all(
            {"figure_": opts_figure, "line_": opts_line, "extent_": opts_extent},
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_line = merge["line_"]
        opts_extent = merge["extent_"]

        is_new = as_bool(is_new, name="is_new")
        is_wrap = as_bool(is_wrap, name="is_wrap")
        is_smooth = as_bool(is_smooth, name="is_smooth")
        is_extent = as_bool(is_extent, name="is_extent")

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

        if opts_line.color == "sample_van_der_corput":
            color_map = blue_green_red_colors()
            color_map_length = np.shape(color_map)[0] - 1
            lines_colors = color_map[
                (sample_van_der_corput(len(lines_plot)) * color_map_length).astype(int)
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

    @property
    def calc_corners_index(self):
        return self.dataset.calc_corners_index

    @property
    def calc_corners(self):
        return self.dataset.calc_corners

    @property
    def calc_bounds(self):
        return self.dataset.calc_bounds

    @property
    def calc_box_size_periodic_index(self):
        return self.dataset.calc_box_size_periodic_index

    @property
    def mask(self):
        return self.dataset.mask

    @property
    def lines(self):
        return [item for item in self.objects if isinstance(item, DisclinationLine)]

    @property
    def figs(self):
        return self.figures

    @property
    def objs(self):
        return self.objects

    def __call__(self) -> np.ndarray:
        return self.raw_Q
