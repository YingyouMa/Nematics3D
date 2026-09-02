"""Scalar-field semantic aliases and runtime converter."""

import numbers

import numpy as np

from ..logging_decorator import logging_and_warning_decorator


# Generic scalar field with arbitrary leading shape.
ScalarField = np.ndarray

# Scalar order parameter field, commonly shape (Nx, Ny, Nz).
# This domain-specific semantic subtype is validated structurally through
# as_scalar_field(); physical restrictions on S belong to the calling function.
# In the perfect ordered state, S is defined to be 1.
SField = ScalarField


@logging_and_warning_decorator(start_finish_level=5)
def as_scalar_field(
    input_data,
    name="scalar field",
    is_spatial_3d_required=False,
    replace=None,
    logger=None,
) -> ScalarField:
    """Validate a real, finite scalar field with arbitrary leading shape.

    Scalars and arrays of any rank are accepted unless
    ``is_spatial_3d_required=True``, which requires shape ``(Nx, Ny, Nz)``.
    No physical value range is imposed.
    """

    def validate(value):
        raw_value = np.asarray(value)
        if is_spatial_3d_required and raw_value.ndim != 3:
            raise ValueError(
                f"{name!r} must have shape (Nx, Ny, Nz). "
                f"Got shape {raw_value.shape}."
            )
        if raw_value.dtype.kind == "O":
            if not all(
                isinstance(component, numbers.Real) for component in raw_value.flat
            ):
                raise TypeError(
                    f"{name!r} must contain only real numbers. Got {value!r}."
                )
            scalar_field = np.asarray(raw_value, dtype=float)
        elif np.issubdtype(raw_value.dtype, np.floating):
            scalar_field = raw_value
        elif np.issubdtype(raw_value.dtype, np.integer) or np.issubdtype(
            raw_value.dtype,
            np.bool_,
        ):
            scalar_field = raw_value.astype(float)
        else:
            raise TypeError(
                f"{name!r} must contain only real numbers. Got dtype "
                f"{raw_value.dtype}."
            )

        if not np.isfinite(scalar_field).all():
            raise ValueError(
                f"{name!r} must contain only finite values. Got {value!r}."
            )
        return scalar_field

    try:
        return validate(input_data)
    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception(f"Invalid {name!r}; attempting the configured replacement.")
        logger.recovery(f"Use {replace!r} as {name!r} in the following.")
        return validate(replace)
