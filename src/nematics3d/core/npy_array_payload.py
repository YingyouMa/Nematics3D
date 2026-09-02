"""Reusable ``.npy``-backed array payload container for result objects."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Generic, Iterator, TypeVar

import numpy as np


TInfo = TypeVar("TInfo")
TPayload = TypeVar("TPayload", bound="NpyArrayPayload")


@dataclass(slots=True, frozen=True, repr=False)
class NpyArrayPayload(Generic[TInfo]):
    """
    Lightweight container for result payloads stored in memory or on disk.

    This class owns a NumPy array payload together with arbitrary metadata
    (`raw_info`) and an optional local `.npy` path. It does not depend on
    `ResultBase`; callers that want richer inspection or formatting behavior
    should compose or inherit that separately.
    """

    raw_values: np.ndarray | None
    raw_info: TInfo
    raw_path: str | None = None

    def keys(self) -> tuple[str, ...]:
        """Return dataclass field names in declaration order."""
        return tuple(field_info.name for field_info in fields(self))

    def values(self) -> tuple[Any, ...]:
        """Return field values in dataclass declaration order."""
        return tuple(getattr(self, key) for key in self.keys())

    def items(self) -> tuple[tuple[str, Any], ...]:
        """Return ``(field_name, value)`` pairs in declaration order."""
        return tuple((key, getattr(self, key)) for key in self.keys())

    def asdict(self) -> dict[str, Any]:
        """Return a shallow dictionary view of this payload container."""
        return {key: getattr(self, key) for key in self.keys()}

    def __getitem__(self, key: str):
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def _helper_load_values_from_path(self) -> np.ndarray:
        """Load array values from ``raw_path`` when no in-memory payload exists."""
        if self.raw_path is None:
            raise ValueError("No in-memory values or saved path are available.")
        return np.load(self.raw_path, allow_pickle=False)

    def act_save_values(
        self: TPayload,
        path,
        *,
        is_release: bool = False,
        is_overwrite: bool = False,
    ) -> TPayload:
        """
        Save payload values to a local ``.npy`` file.

        When ``is_release`` is true, the returned copy keeps only the saved path
        and releases the in-memory array reference.
        """
        save_path = Path(path)
        if save_path.suffix != ".npy":
            save_path = Path(f"{save_path}.npy")
        if save_path.exists() and not is_overwrite:
            raise FileExistsError(
                f"{type(self).__name__} path already exists: {save_path}"
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        values = self.raw_values
        if values is None:
            values = self._helper_load_values_from_path()
        np.save(save_path, values)

        raw_values = None if is_release else self.raw_values
        return replace(self, raw_values=raw_values, raw_path=str(save_path))

    def act_release_values(self: TPayload) -> TPayload:
        """Return a copy without the in-memory array reference."""
        if self.raw_path is None:
            raise ValueError("Cannot release values before saving them.")
        return replace(self, raw_values=None)

    def act_load_values(self: TPayload) -> TPayload:
        """Return a copy with values loaded into memory."""
        if self.raw_values is not None:
            return self
        return replace(self, raw_values=self._helper_load_values_from_path())

    @contextmanager
    def act_with_values(self) -> Iterator[np.ndarray]:
        """
        Temporarily expose the payload values.

        If values are already in memory, the existing array is yielded. If only
        ``raw_path`` is available, values are loaded for the ``with`` block and
        the temporary reference is dropped when the block exits.
        """
        if self.raw_values is not None:
            yield self.raw_values
            return

        values = self._helper_load_values_from_path()
        try:
            yield values
        finally:
            del values
