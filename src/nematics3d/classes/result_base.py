"""Lightweight base class for inspectable dataclass result objects."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, ClassVar

from nematics3d.format import repr_field_line
from nematics3d.logging_decorator import logging_and_warning_decorator


class ResultBase:
    """Mixin for stable algorithm results represented as dataclasses.

    ResultBase gives result objects a small dict-like inspection surface while
    preserving normal attribute access. Subclasses should be dataclasses using
    ``repr=False`` so this base class can provide the aligned representation.
    """

    __result_name__: ClassVar[str | None] = "result"
    __field_docs__: ClassVar[dict[str, str]] = {}

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

    @logging_and_warning_decorator(start_finish_level=5)
    def show_readable_attrs(self, is_return=False, is_desc=True, logger=None):
        """Show readable result fields and optional field descriptions."""
        docs = type(self).__field_docs__
        keys = self.keys()
        lines = []

        if not keys:
            lines.append("- <none>")
        else:
            for key in keys:
                lines.append(f"- {key}")
                if is_desc:
                    lines.append(f"    {docs.get(key, '')}")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_attr_doc(self, name: str, is_return=False, logger=None):
        """Show the description for one readable result field."""
        if name not in self:
            raise KeyError(name)

        doc = type(self).__field_docs__.get(name, "")
        logger.info(doc)
        if is_return:
            return doc
        return None

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
