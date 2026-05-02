"""Lightweight base class for inspectable dataclass result objects."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, ClassVar

from nematics3d.format import repr_field_line


class ResultBase:
    """Mixin for stable algorithm results represented as dataclasses.

    ResultBase gives result objects a small dict-like inspection surface while
    preserving normal attribute access. Subclasses should be dataclasses using
    ``repr=False`` so this base class can provide the aligned representation.
    """

    __result_name__: ClassVar[str | None] = "result"

    # -------------------------------
    # Field inspection
    # -------------------------------

    def _helper_fields(self):
        """Return dataclass fields, or fail clearly for invalid subclasses."""
        if not is_dataclass(self):
            raise TypeError("ResultBase subclasses must be dataclass instances.")
        return fields(self)

    def keys(self) -> tuple[str, ...]:
        """Return result field names in dataclass declaration order."""
        return tuple(field_info.name for field_info in self._helper_fields())

    def values(self) -> tuple[Any, ...]:
        """Return result values in dataclass declaration order."""
        return tuple(getattr(self, key) for key in self.keys())

    def items(self) -> tuple[tuple[str, Any], ...]:
        """Return ``(field_name, value)`` pairs in declaration order."""
        return tuple((key, getattr(self, key)) for key in self.keys())

    def asdict(self) -> dict[str, Any]:
        """Return a shallow dictionary view of this result."""
        return {key: getattr(self, key) for key in self.keys()}

    # -------------------------------
    # Dict-like conveniences
    # -------------------------------

    def __getitem__(self, key: str):
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default=None):
        """Return one result field, or ``default`` when absent."""
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    # -------------------------------
    # Representation
    # -------------------------------

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        result_name = type(self).__result_name__
        if result_name:
            lines = [f"{cls_name}: {result_name}"]
        else:
            lines = [cls_name]
        keys = list(self.keys())
        if not keys:
            return "\n".join(lines)

        width = max(len(key) for key in keys)
        for key in keys:
            value = getattr(self, key, "<missing>")
            lines.append(repr_field_line(key, value, width))

        return "\n".join(lines)
