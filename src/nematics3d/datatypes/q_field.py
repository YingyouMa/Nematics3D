"""Q-field semantic aliases and runtime converters."""

from typing import Union

import numpy as np

from .number import as_number


# Tensor order parameter in 5-component representation, shape: (Nx, Ny, Nz, 5)
# Subtype of GeneralField
# Components: [Q_xx, Q_xy, Q_xz, Q_yy, Q_yz]
QField5 = np.ndarray

# Tensor order parameter in full 3x3 matrix form, shape: (Nx, Ny, Nz, 3, 3)
# Subtype of GeneralField
# Symmetric traceless tensor Q_ij with:
# Q[..., 0,0] = Q_xx, Q[..., 0,1] = Q_xy, Q[..., 1,0] = Q_xy, etc.
QField9 = np.ndarray


def _validate_qfield_single_shape(
    shape: tuple[int, ...],
    *,
    name: str,
    expected_ndim: int,
    expected_label: str,
) -> None:
    if len(shape) != expected_ndim:
        raise ValueError(
            f"{name!r} must be a single 3D Q field in one of the supported "
            "representations: (Nx, Ny, Nz, 5) or (Nx, Ny, Nz, 3, 3). "
            f"This input matches the {expected_label} representation by its "
            f"trailing dimensions, but has shape {shape}."
        )
    if any(axis_size == 0 for axis_size in shape[:3]):
        raise ValueError(
            f"{name!r} must have nonzero spatial dimensions when "
            f"is_strict_3d_field=True, but has shape {shape}."
        )


def _validate_qfield9_tensor(
    qtensor: QField9,
    *,
    name: str,
    symmetry_tolerance: float | None,
    trace_tolerance: float | None,
) -> None:
    """Validate the numerical and defining tensor properties of QField9."""
    if not np.all(np.isfinite(qtensor)):
        invalid_indices = np.argwhere(~np.all(np.isfinite(qtensor), axis=(-2, -1)))
        raise ValueError(
            f"{name!r} must be finite everywhere. Non-finite tensor indices "
            f"include {invalid_indices[:5].tolist()}."
        )

    machine_epsilon = np.finfo(qtensor.dtype).eps
    tensor_scale = np.maximum(1.0, np.max(np.abs(qtensor), axis=(-2, -1)))
    default_tolerance = 32 * machine_epsilon * tensor_scale

    effective_symmetry_tolerance = (
        default_tolerance if symmetry_tolerance is None else float(symmetry_tolerance)
    )
    effective_trace_tolerance = (
        default_tolerance if trace_tolerance is None else float(trace_tolerance)
    )

    # Compare only the three independent off-diagonal pairs. Reuse two
    # leading-shape arrays instead of materializing a full (..., 3, 3)
    # difference and absolute-value array for large fields.
    symmetry_error = np.empty(qtensor.shape[:-2], dtype=qtensor.dtype)
    symmetry_scratch = np.empty_like(symmetry_error)

    np.subtract(qtensor[..., 0, 1], qtensor[..., 1, 0], out=symmetry_error)
    np.abs(symmetry_error, out=symmetry_error)

    np.subtract(qtensor[..., 0, 2], qtensor[..., 2, 0], out=symmetry_scratch)
    np.abs(symmetry_scratch, out=symmetry_scratch)
    np.maximum(symmetry_error, symmetry_scratch, out=symmetry_error)

    np.subtract(qtensor[..., 1, 2], qtensor[..., 2, 1], out=symmetry_scratch)
    np.abs(symmetry_scratch, out=symmetry_scratch)
    np.maximum(symmetry_error, symmetry_scratch, out=symmetry_error)
    is_asymmetric = symmetry_error > effective_symmetry_tolerance
    if np.any(is_asymmetric):
        invalid_indices = np.argwhere(is_asymmetric)
        maximum_error = float(np.max(symmetry_error[is_asymmetric]))
        raise ValueError(
            f"{name!r} must be symmetric. Detected "
            f"{invalid_indices.shape[0]} asymmetric tensor(s); maximum asymmetry "
            f"is {maximum_error:.6g}. Invalid indices include "
            f"{invalid_indices[:5].tolist()}."
        )

    trace_error = np.abs(np.trace(qtensor, axis1=-2, axis2=-1))
    is_not_traceless = trace_error > effective_trace_tolerance
    if np.any(is_not_traceless):
        invalid_indices = np.argwhere(is_not_traceless)
        maximum_error = float(np.max(trace_error[is_not_traceless]))
        raise ValueError(
            f"{name!r} must be traceless. Detected "
            f"{invalid_indices.shape[0]} tensor(s) with nonzero trace; maximum "
            f"absolute trace is {maximum_error:.6g}. Invalid indices include "
            f"{invalid_indices[:5].tolist()}."
        )


def as_qfield9(
    qtensor: Union[QField5, QField9],
    name="QField",
    is_strict_3d_field: bool = True,
    *,
    is_validate_tensor: bool = True,
    symmetry_tolerance: float | None = None,
    trace_tolerance: float | None = None,
) -> QField9:
    """
    Convert a Q-tensor field into full 3×3 matrix form (QField9).

    Accepts either:
    - a 5-component representation (QField5), shape (Nx, Ny, Nz, 5), or
    - a full matrix representation (QField9), shape (Nx, Ny, Nz, 3, 3)

    Set ``is_strict_3d_field=False`` to allow the more general shapes
    ``(..., 5)`` and ``(..., 3, 3)`` for point sets, slices, batched tensors, or
    single Q tensors. Strict 3D fields must have nonzero spatial dimensions;
    empty arrays with a supported trailing shape remain valid in non-strict mode.

    Parameters
    ----------
    qtensor : QField5 or QField9
        Input Q-tensor field, either in 5-component or 3×3 form.
    name : str, optional
        Human-readable input name included in validation errors.
    is_strict_3d_field : bool, optional
        If True, require exactly three nonzero spatial axes, giving shape
        ``(Nx, Ny, Nz, 5)`` or ``(Nx, Ny, Nz, 3, 3)``. If False, preserve any
        leading dimensions, including empty dimensions.
    is_validate_tensor : bool, optional
        If True, require finite values and validate that a supplied 3×3
        representation is symmetric and traceless. If False, skip these
        numerical checks; dtype and shape validation still apply. The
        five-component representation guarantees symmetry and zero trace by
        construction, so only its finite values require numerical validation.
    symmetry_tolerance : float, optional
        Absolute tolerance for ``max(abs(Q - Q.T))``. It must be finite and
        non-negative. By default, each full tensor uses
        ``32 * eps(dtype) * max(1, max(abs(Q)))``.
    trace_tolerance : float, optional
        Absolute tolerance for ``abs(trace(Q))``. It must be finite and
        non-negative. By default, use the same per-tensor scale-aware rule as
        ``symmetry_tolerance``.

    Returns
    -------
    QField9
        Full 3×3 matrix form. Five-component input produces a new array. A
        full NumPy array is returned unchanged, preserving zero-copy behavior.

    Raises
    ------
    TypeError
        If the input dtype is not floating-point.
    ValueError
        If the shape is unsupported, a strict spatial axis is empty, a checked
        value is non-finite, a full tensor is not symmetric or traceless within
        tolerance, or a supplied tolerance is invalid.
    """
    qtensor = np.asarray(qtensor)

    if not np.issubdtype(qtensor.dtype, np.floating):
        raise TypeError(
            "QField must be a float-type NumPy array, got dtype "
            f"{qtensor.dtype}. Name of QField: {name}"
        )

    tolerance_inputs = {
        "symmetry_tolerance": symmetry_tolerance,
        "trace_tolerance": trace_tolerance,
    }
    for tolerance_name, tolerance in tolerance_inputs.items():
        if tolerance is not None:
            tolerance_inputs[tolerance_name] = as_number(
                tolerance,
                name=tolerance_name,
                value_range=(0.0, np.inf),
            )
    symmetry_tolerance = tolerance_inputs["symmetry_tolerance"]
    trace_tolerance = tolerance_inputs["trace_tolerance"]

    shape = qtensor.shape

    if len(shape) >= 1 and shape[-1] == 5:
        if is_strict_3d_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=4,
                expected_label="(Nx, Ny, Nz, 5)",
            )
        # Convert from 5-component representation to full 3x3 tensor
        full_tensor = np.zeros((*shape[:-1], 3, 3), dtype=qtensor.dtype)
        full_tensor[..., 0, 0] = qtensor[..., 0]  # Q_xx
        full_tensor[..., 0, 1] = qtensor[..., 1]  # Q_xy
        full_tensor[..., 0, 2] = qtensor[..., 2]  # Q_xz
        full_tensor[..., 1, 0] = qtensor[..., 1]  # Q_yx = Q_xy
        full_tensor[..., 1, 1] = qtensor[..., 3]  # Q_yy
        full_tensor[..., 1, 2] = qtensor[..., 4]  # Q_yz
        full_tensor[..., 2, 0] = qtensor[..., 2]  # Q_zx = Q_xz
        full_tensor[..., 2, 1] = qtensor[..., 4]  # Q_zy = Q_yz
        full_tensor[..., 2, 2] = -full_tensor[..., 0, 0] - full_tensor[..., 1, 1]
        if is_validate_tensor and not np.all(np.isfinite(full_tensor)):
            invalid_indices = np.argwhere(
                ~np.all(np.isfinite(full_tensor), axis=(-2, -1))
            )
            raise ValueError(
                f"{name!r} must be finite everywhere. Non-finite tensor indices "
                f"include {invalid_indices[:5].tolist()}."
            )
        return full_tensor

    if len(shape) >= 2 and shape[-2:] == (3, 3):
        if is_strict_3d_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=5,
                expected_label="(Nx, Ny, Nz, 3, 3)",
            )
        full_tensor = qtensor
        if is_validate_tensor:
            _validate_qfield9_tensor(
                full_tensor,
                name=name,
                symmetry_tolerance=symmetry_tolerance,
                trace_tolerance=trace_tolerance,
            )
        return full_tensor  # Already in QField9 form

    raise ValueError(
        "Invalid QField shape: expected (Nx, Ny, Nz, 5) or "
        f"(Nx, Ny, Nz, 3, 3), but got shape {shape}. "
        f"Name of QField: {name}"
    )


def as_qfield5(
    qtensor: Union[QField5, QField9],
    name="QField",
    is_strict_3d_field: bool = True,
    *,
    is_validate_tensor: bool = True,
) -> QField5:
    """
    Convert a Q-tensor field into compact five-component form (QField5).

    Accepts either:
    - a 5-component representation (QField5), shape (Nx, Ny, Nz, 5), or
    - a full matrix representation (QField9), shape (Nx, Ny, Nz, 3, 3)

    Set ``is_strict_3d_field=False`` to allow the more general shapes
    ``(..., 5)`` and ``(..., 3, 3)`` for point sets, slices, batched tensors, or
    single Q tensors. Strict 3D fields must have nonzero spatial dimensions;
    empty arrays with a supported trailing shape remain valid in non-strict mode.

    Full input is assumed to follow the symmetric, traceless Q-tensor contract,
    but this converter does not verify symmetry or trace. It extracts
    ``(Q_xx, Q_xy, Q_xz, Q_yy, Q_yz)`` directly.

    Parameters
    ----------
    qtensor : QField5 or QField9
        Input Q-tensor field, either in five-component or 3×3 form.
    name : str, optional
        Human-readable input name included in validation errors.
    is_strict_3d_field : bool, optional
        If True, require exactly three nonzero spatial axes, giving shape
        ``(Nx, Ny, Nz, 5)`` or ``(Nx, Ny, Nz, 3, 3)``. If False, preserve any
        leading dimensions, including empty dimensions.
    is_validate_tensor : bool, optional
        If True, require every input value to be finite. If False, skip this
        numerical check; dtype and shape validation still apply.

    Returns
    -------
    QField5
        Compact five-component form with shape ``(..., 5)``. Full input
        produces a new array; five-component NumPy input is returned unchanged.

    Raises
    ------
    TypeError
        If the input dtype is not floating-point.
    ValueError
        If the shape is unsupported, a strict spatial axis is empty, or a
        checked value is non-finite.
    """
    qtensor = np.asarray(qtensor)

    if not np.issubdtype(qtensor.dtype, np.floating):
        raise TypeError(
            f"QField must be a float-type NumPy array, got dtype {qtensor.dtype}. Name of QField: {name}"
        )

    shape = qtensor.shape

    if len(shape) >= 2 and shape[-2:] == (3, 3):
        if is_strict_3d_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=5,
                expected_label="(Nx, Ny, Nz, 3, 3)",
            )
        if is_validate_tensor and not np.all(np.isfinite(qtensor)):
            invalid_indices = np.argwhere(~np.all(np.isfinite(qtensor), axis=(-2, -1)))
            raise ValueError(
                f"{name!r} must be finite everywhere. Non-finite tensor indices "
                f"include {invalid_indices[:5].tolist()}."
            )

        Q5 = np.empty(shape[:-2] + (5,), dtype=qtensor.dtype)

        Q5[..., 0] = qtensor[..., 0, 0]  # Q_xx
        Q5[..., 1] = qtensor[..., 0, 1]  # Q_xy
        Q5[..., 2] = qtensor[..., 0, 2]  # Q_xz
        Q5[..., 3] = qtensor[..., 1, 1]  # Q_yy
        Q5[..., 4] = qtensor[..., 1, 2]  # Q_yz

        return Q5

    if len(shape) >= 1 and shape[-1] == 5:
        if is_strict_3d_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=4,
                expected_label="(Nx, Ny, Nz, 5)",
            )
        Q5 = qtensor
        if is_validate_tensor and not np.all(np.isfinite(Q5)):
            invalid_indices = np.argwhere(~np.all(np.isfinite(Q5), axis=-1))
            raise ValueError(
                f"{name!r} must be finite everywhere. Non-finite tensor indices "
                f"include {invalid_indices[:5].tolist()}."
            )
        return Q5

    raise ValueError(
        "Invalid QField shape: expected (Nx, Ny, Nz, 5) or "
        f"(Nx, Ny, Nz, 3, 3), but got shape {shape}. "
        f"Name of QField: {name}"
    )


# -------------------------
