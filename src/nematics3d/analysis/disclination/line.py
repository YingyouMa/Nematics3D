"""Domain objects for ordered disclination trajectories."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Mapping

import numpy as np

from ...core.class_base import AttrDef, ClassBase
from ...datatypes import (
    BoxSizePeriodic,
    DefectIndex,
    Vect,
    as_box_size_periodic,
    as_defect_index,
)
from ...geometry import find_plane_normal
from ...grid import (
    GRID_TRANSFORM_IDENTITY,
    GridTransform,
    apply_linear_transform,
    as_grid_offset,
    as_grid_transform,
)
from ...logging_decorator import logging_and_warning_decorator


@dataclass(slots=True, frozen=True)
class DisclinationLineInput:
    """Validated construction data for :class:`DisclinationLine`."""

    defect_indices: DefectIndex | None = None
    box_size_periodic_index: BoxSizePeriodic = np.inf
    grid_offset: Vect(3) | None = None
    grid_transform: GridTransform = GRID_TRANSFORM_IDENTITY

    __attrs__: ClassVar[Mapping[str, str]] = {
        "defect_indices": "ordered defect indices forming one line",
        "box_size_periodic_index": "periodic box size in lattice-index space",
        "grid_offset": "translation from lattice-index to real space",
        "grid_transform": "linear transform from lattice-index to real space",
    }

    def __post_init__(self):
        object.__setattr__(
            self,
            "defect_indices",
            (
                None
                if self.defect_indices is None
                else as_defect_index(self.defect_indices, name="defect_indices")
            ),
        )
        object.__setattr__(
            self,
            "box_size_periodic_index",
            as_box_size_periodic(
                self.box_size_periodic_index, name="box_size_periodic_index"
            ),
        )
        object.__setattr__(
            self, "grid_offset", as_grid_offset(self.grid_offset, name="grid_offset")
        )
        object.__setattr__(
            self,
            "grid_transform",
            as_grid_transform(self.grid_transform, name="grid_transform"),
        )


class DisclinationLine(ClassBase):
    """One ordered discrete disclination line in lattice and real space."""

    __attr_defs__ = {
        "raw_defect_indices": AttrDef(
            doc="Ordered lattice defect indices.", kind="raw"
        ),
        "raw_box_size_periodic_index": AttrDef(
            doc="Periodic box size in index space.", kind="raw"
        ),
        "raw_grid_offset": AttrDef(doc="Grid-to-real translation.", kind="raw"),
        "raw_grid_transform": AttrDef(doc="Grid-to-real linear transform.", kind="raw"),
        "calc_end2end_kind": AttrDef(
            doc="Line kind: loop, cross, or seg.", kind="calc"
        ),
        "calc_defect_num": AttrDef(doc="Number of stored defect points.", kind="calc"),
        "calc_defect_coords": AttrDef(
            doc="Real-space defect coordinates.", kind="calc"
        ),
        "calc_norm": AttrDef(doc="Latest fitted average plane normal.", kind="calc"),
        "calc_norm_metric": AttrDef(
            doc="Metrics for the latest plane fit.", kind="calc"
        ),
        "entity_smooth_objs": AttrDef(
            doc="Generated smoothed line objects.", kind="entity"
        ),
        "smooths": AttrDef(
            doc="Generated smoothed line objects.",
            kind="property",
            is_public_settable=False,
        ),
        "smooth": AttrDef(
            doc="Latest smoothed line object.",
            kind="property",
            is_public_settable=False,
        ),
        "kind": AttrDef(
            doc="Shorthand for calc_end2end_kind.",
            kind="property",
            is_public_settable=False,
        ),
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in ClassBase.__slots__
    )

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        input_value: DisclinationLineInput | None = None,
        name: str | None = None,
        logger=None,
        **kwargs,
    ):
        if "inputValue" in kwargs:
            if input_value is not None:
                raise TypeError("Pass only one of `input_value` and `inputValue`.")
            input_value = kwargs.pop("inputValue")
        input_value = self._resolve_input(input_value, kwargs)
        if input_value.defect_indices is None:
            raise ValueError("`defect_indices` must be provided.")
        if len(input_value.defect_indices) < 2:
            raise ValueError(
                "A disclination line must contain at least two defect points."
            )

        super().__init__(
            name="disclination line" if name is None else name,
            name_replace="disclination line",
            is_fixed=True,
        )
        indices, kind = self._canonicalize_endpoints(
            input_value.defect_indices, input_value.box_size_periodic_index
        )
        object.__setattr__(self, "raw_defect_indices", indices)
        object.__setattr__(
            self, "raw_box_size_periodic_index", input_value.box_size_periodic_index
        )
        object.__setattr__(self, "raw_grid_offset", input_value.grid_offset)
        object.__setattr__(self, "raw_grid_transform", input_value.grid_transform)
        object.__setattr__(self, "calc_end2end_kind", kind)
        object.__setattr__(self, "calc_defect_num", len(indices))
        object.__setattr__(
            self,
            "calc_defect_coords",
            apply_linear_transform(
                indices,
                transform=input_value.grid_transform,
                offset=input_value.grid_offset,
            ),
        )
        object.__setattr__(self, "calc_norm", None)
        object.__setattr__(self, "calc_norm_metric", None)
        object.__setattr__(self, "entity_smooth_objs", [])
        logger.debug(f"Disclination line {self.name!r} is of kind {kind!r}.")

    @staticmethod
    def _resolve_input(input_value, kwargs: dict[str, Any]) -> DisclinationLineInput:
        if input_value is not None and not isinstance(
            input_value, DisclinationLineInput
        ):
            raise TypeError(
                "`input_value` must be a DisclinationLineInput instance or None."
            )
        valid = {field.name for field in fields(DisclinationLineInput)}
        unknown = set(kwargs) - valid
        if unknown:
            raise TypeError(
                f"Unknown DisclinationLine input keyword(s): {', '.join(sorted(unknown))}."
            )
        if input_value is None:
            return DisclinationLineInput(**kwargs)
        if not kwargs:
            return input_value
        values = {
            field.name: getattr(input_value, field.name)
            for field in fields(DisclinationLineInput)
        }
        values.update(kwargs)
        return DisclinationLineInput(**values)

    @staticmethod
    def _canonicalize_endpoints(defect_indices, box_size):
        indices = np.asarray(defect_indices, dtype=float).copy()
        first, last = indices[0], indices[-1]
        if np.array_equal(first, last):
            return indices[:-1], "loop"
        periodic = np.isfinite(box_size)
        first_wrapped, last_wrapped = first.copy(), last.copy()
        first_wrapped[periodic] %= box_size[periodic]
        last_wrapped[periodic] %= box_size[periodic]
        if np.array_equal(first_wrapped, last_wrapped):
            return indices[:-1], "cross"
        return indices, "seg"

    @logging_and_warning_decorator()
    def act_calc_norm(self, logger=None):
        result = find_plane_normal(self.calc_defect_coords)
        if result.linearity_risk > 0.5:
            logger.warning(
                f"Low confidence in normal for {self.name!r}: line is nearly straight (linearity_risk={result.linearity_risk:.2f})."
            )
        elif result.planarity_score < 0.7:
            logger.warning(
                f"Low confidence in normal for {self.name!r}: line is non-planar (planarity_score={result.planarity_score:.2f})."
            )
        object.__setattr__(self, "calc_norm", result.normal)
        object.__setattr__(self, "calc_norm_metric", result.metric)
        return result

    def act_smooth(self, is_new=True, **kwargs):
        from .smooth import DisclinationLineSmooth

        if not is_new and self.smooths:
            return self.smooths[-1]
        name = kwargs.pop("name", f"{self.name} smooth_version {len(self.smooths)}")
        item = DisclinationLineSmooth(self, name=name, **kwargs)
        self.smooths.append(item)
        return item

    def act_visualize(self, smooth_index=-1, **kwargs):
        if not self.smooths:
            self.act_smooth(window_length=5, min_line_length=6)
            kwargs.setdefault("is_smooth", False)
        smooth_obj = (
            self.smooths[smooth_index]
            if -len(self.smooths) <= smooth_index < len(self.smooths)
            else self.smooths[-1]
        )
        if getattr(smooth_obj, "visual", None):
            smooth_obj = self.act_smooth(opts=smooth_obj.opts, is_window_warning=False)
        return smooth_obj.act_visualize(**kwargs)

    def __len__(self):
        return self.calc_defect_num

    def __iter__(self):
        return iter(self.raw_defect_indices)

    def __getitem__(self, idx):
        return self.raw_defect_indices[idx]

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray(self.raw_defect_indices, dtype=dtype)

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}), type {self.kind}, {len(self)} defect points"

    def __str__(self):
        return f"{type(self).__name__}({self.name!r})"

    @property
    def smooths(self):
        return self.entity_smooth_objs

    @property
    def smooth(self):
        return self.smooths[-1] if self.smooths else None

    @property
    def kind(self):
        return self.calc_end2end_kind


InputLine = DisclinationLineInput

__all__ = ["DisclinationLine", "DisclinationLineInput", "InputLine"]
