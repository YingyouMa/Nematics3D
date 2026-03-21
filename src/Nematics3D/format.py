import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .datatypes import UNSET


def fmt_value(v, ndigits=2):
    """Format scalar or ndarray with fixed decimals (zero-padded)."""
    if isinstance(v, np.ndarray):
        fmt = np.vectorize(lambda x: f"{float(x):.{ndigits}f}", otypes=[str])
        return (
            "[" + ", ".join(fmt(v).ravel()) + "]"
            if v.ndim == 1
            else np.array2string(fmt(v), separator=", ")
        )
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return str(v)


def is_equal_array(v1, v2):
    """Compare two array-like values as ndarrays, treating NaNs as equal."""
    try:
        arr1 = np.asarray(v1)
        arr2 = np.asarray(v2)
    except Exception as e:
        raise TypeError(f"Input value cannot be converted to numpy array: {e}")

    if (not isinstance(arr1, np.ndarray)) or (not isinstance(arr2, np.ndarray)):
        raise TypeError("Both inputs must be numpy arrays or array-like values.")

    return np.array_equal(arr1, arr2, equal_nan=True)


def is_equal(v1, v2):
    """Safely compare scalars or array-like values with ndarray-aware fallback logic."""
    try:
        return is_equal_array(v1, v2)
    except TypeError:
        try:
            return v1 == v2
        except Exception:
            return False


def is_given_str(a, b):
    """Return True only when ``a`` is exactly the given string ``b``."""
    return isinstance(a, str) and a == b


def json_callable_note(value: Callable) -> str:
    module = getattr(value, "__module__", None) or "<unknown module>"
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if qualname is None:
        qualname = type(value).__name__
    return f"<callable {module}.{qualname}; stored as note only and not restored automatically>"


def json_array_file_stem(name: str) -> str:
    chars = []
    for ch in name:
        chars.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    stem = "".join(chars).strip("_")
    return stem or "array"


def json_encode_value(
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
        return {
            "__callable__": json_callable_note(value),
        }

    if isinstance(value, np.ndarray):
        if value.size <= max_inline_array_size:
            return {
                "__ndarray__": "inline",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }

        filename = f"{json_array_file_stem(array_stem)}.npy"
        np.save(parent_dir / filename, value)
        return {
            "__ndarray__": "file",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "path": filename,
        }

    if isinstance(value, dict):
        return {
            str(k): json_encode_value(
                v,
                parent_dir=parent_dir,
                array_stem=f"{array_stem}_{k}",
                max_inline_array_size=max_inline_array_size,
            )
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_encode_value(
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
                json_encode_value(
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


def json_decode_value(value: Any, *, parent_dir: Path) -> Any:
    if isinstance(value, list):
        return [json_decode_value(item, parent_dir=parent_dir) for item in value]

    if not isinstance(value, dict):
        return value

    if value.get("__unset__") is True:
        return UNSET

    if "__callable__" in value:
        return value["__callable__"]

    if "__tuple__" in value:
        return tuple(
            json_decode_value(item, parent_dir=parent_dir)
            for item in value["__tuple__"]
        )

    ndarray_mode = value.get("__ndarray__", None)
    if ndarray_mode == "inline":
        return np.asarray(value["data"], dtype=value.get("dtype", None))
    if ndarray_mode == "file":
        array_path = parent_dir / value["path"]
        try:
            return np.load(array_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing external ndarray file while loading opts JSON: {array_path}"
            ) from exc

    return {k: json_decode_value(v, parent_dir=parent_dir) for k, v in value.items()}


def save_opts_json(
    opts_dict: dict[str, Any],
    path: str | Path,
    *,
    opts_class_name: str,
    max_inline_array_size: int = 64,
) -> Path:
    """Save an opts dictionary to JSON, externalizing large ndarrays into sidecar `.npy` files."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "__opts_class__": opts_class_name,
        "opts": {
            key: json_encode_value(
                value,
                parent_dir=path.parent,
                array_stem=f"{path.stem}_{key}",
                max_inline_array_size=max_inline_array_size,
            )
            for key, value in opts_dict.items()
        },
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    return path


def load_opts_json(path: str | Path) -> tuple[Path, str | None, dict[str, Any]]:
    """Load an opts JSON payload, restoring inline arrays and sidecar `.npy` arrays."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "opts" not in payload:
        raise ValueError(
            f"Invalid opts JSON file {path}: expected a top-level 'opts' mapping."
        )

    data = payload["opts"]
    if not isinstance(data, dict):
        raise TypeError(
            f"Invalid opts JSON file {path}: 'opts' must be a mapping, got {type(data).__name__}."
        )

    opts_class_name = payload.get("__opts_class__", None)
    if opts_class_name is not None and not isinstance(opts_class_name, str):
        raise TypeError(
            f"Invalid opts JSON file {path}: '__opts_class__' must be a string when present."
        )

    return (
        path,
        opts_class_name,
        {k: json_decode_value(v, parent_dir=path.parent) for k, v in data.items()},
    )


def repr_format(v):
    """Format a value for compact repository-style repr output."""
    if isinstance(v, np.generic):
        v = v.item()

    if isinstance(v, float):
        return f"{v:.2g}"

    if isinstance(v, np.ndarray):
        if v.size > 6:
            return f"<ndarray shape={v.shape}, too many elements to display>"
        return repr(v)

    return repr(v)
