"""JSON persistence helpers for Opts-style dictionaries."""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..datatypes import UNSET


def _callable_note(value: Callable) -> str:
    module = getattr(value, "__module__", None) or "<unknown module>"
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if qualname is None:
        qualname = type(value).__name__
    return (
        f"<callable {module}.{qualname}; stored as note only and not restored "
        "automatically>"
    )


def _array_file_stem(name: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return stem.strip("_") or "array"


def _encode_value(
    value: Any,
    *,
    parent_dir: Path,
    array_stem: str,
    max_inline_array_size: int,
) -> Any:
    if value is UNSET:
        return {"__unset__": True}
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return {"__callable__": _callable_note(value)}

    if isinstance(value, np.ndarray):
        if value.size <= max_inline_array_size:
            return {
                "__ndarray__": "inline",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        filename = f"{_array_file_stem(array_stem)}.npy"
        np.save(parent_dir / filename, value)
        return {
            "__ndarray__": "file",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "path": filename,
        }

    if isinstance(value, dict):
        return {
            str(key): _encode_value(
                item,
                parent_dir=parent_dir,
                array_stem=f"{array_stem}_{key}",
                max_inline_array_size=max_inline_array_size,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _encode_value(
                item,
                parent_dir=parent_dir,
                array_stem=f"{array_stem}_{index}",
                max_inline_array_size=max_inline_array_size,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return {
            "__tuple__": [
                _encode_value(
                    item,
                    parent_dir=parent_dir,
                    array_stem=f"{array_stem}_{index}",
                    max_inline_array_size=max_inline_array_size,
                )
                for index, item in enumerate(value)
            ]
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Value of type {type(value).__name__} is not supported for Opts JSON export."
    )


def _decode_value(value: Any, *, parent_dir: Path) -> Any:
    if isinstance(value, list):
        return [_decode_value(item, parent_dir=parent_dir) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("__unset__") is True:
        return UNSET
    if "__callable__" in value:
        return value["__callable__"]
    if "__tuple__" in value:
        return tuple(
            _decode_value(item, parent_dir=parent_dir) for item in value["__tuple__"]
        )

    ndarray_mode = value.get("__ndarray__")
    if ndarray_mode == "inline":
        return np.asarray(value["data"], dtype=value.get("dtype"))
    if ndarray_mode == "file":
        array_path = parent_dir / value["path"]
        try:
            return np.load(array_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing external ndarray file while loading opts JSON: {array_path}"
            ) from exc
    return {
        key: _decode_value(item, parent_dir=parent_dir) for key, item in value.items()
    }


def save_opts_json(
    opts_dict: dict[str, Any],
    path: str | Path,
    *,
    opts_class_name: str,
    max_inline_array_size: int = 64,
) -> Path:
    """Save an opts mapping, externalizing large ndarrays to sidecar NPY files."""
    if isinstance(max_inline_array_size, bool) or not isinstance(
        max_inline_array_size, int
    ):
        raise TypeError("`max_inline_array_size` must be a non-negative integer.")
    if max_inline_array_size < 0:
        raise ValueError("`max_inline_array_size` must be non-negative.")

    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "__opts_class__": opts_class_name,
        "opts": {
            key: _encode_value(
                value,
                parent_dir=path.parent,
                array_stem=f"{path.stem}_{key}",
                max_inline_array_size=max_inline_array_size,
            )
            for key, value in opts_dict.items()
        },
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)
    return path


def load_opts_json(path: str | Path) -> tuple[Path, str | None, dict[str, Any]]:
    """Load an opts JSON payload and restore inline or sidecar arrays."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or "opts" not in payload:
        raise ValueError(
            f"Invalid opts JSON file {path}: expected a top-level 'opts' mapping."
        )
    data = payload["opts"]
    if not isinstance(data, dict):
        raise TypeError(
            f"Invalid opts JSON file {path}: 'opts' must be a mapping, got {type(data).__name__}."
        )
    opts_class_name = payload.get("__opts_class__")
    if opts_class_name is not None and not isinstance(opts_class_name, str):
        raise TypeError(
            f"Invalid opts JSON file {path}: '__opts_class__' must be a string when present."
        )
    return (
        path,
        opts_class_name,
        {
            key: _decode_value(value, parent_dir=path.parent)
            for key, value in data.items()
        },
    )


__all__ = ["load_opts_json", "save_opts_json"]
