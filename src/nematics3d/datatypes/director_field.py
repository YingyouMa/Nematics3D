"""Director-field semantic alias and runtime converter."""

import numbers

import numpy as np

from ..logging_decorator import logging_and_warning_decorator


# Director field (unit vector), shape: (Nx, Ny, Nz, 3)
# Subtype of GeneralField
# It may relax to shape (..., 3)
#
# `nField` intentionally does not follow the usual PEP 8 CapWords convention
# for type aliases. The short scientific notation is retained to keep the name
# compact and visually aligned with the existing `SField` alias.
nField = np.ndarray


@logging_and_warning_decorator(start_finish_level=5)
def as_director_field(
    input_data,
    name="director field",
    is_spatial_3d_required=False,
    is_normalized=True,
    is_zero_allowed=True,
    replace=None,
    logger=None,
) -> nField:
    """Validate a real, finite director field with trailing shape ``(3,)``.

    Arbitrary leading dimensions are accepted unless
    ``is_spatial_3d_required=True``, which requires shape ``(Nx, Ny, Nz, 3)``.
    Normalization is applied independently at every field point. Allowed zero
    directors remain zero during normalization.
    """

    def validate(value):
        raw_value = np.asarray(value)
        if raw_value.ndim == 0 or raw_value.shape[-1] != 3:
            raise ValueError(
                f"{name!r} must have trailing shape (..., 3). "
                f"Got shape {raw_value.shape}."
            )
        if is_spatial_3d_required and raw_value.ndim != 4:
            raise ValueError(
                f"{name!r} must have shape (Nx, Ny, Nz, 3). "
                f"Got shape {raw_value.shape}."
            )
        if raw_value.dtype.kind == "O":
            if not all(
                isinstance(component, numbers.Real) for component in raw_value.flat
            ):
                raise TypeError(
                    f"{name!r} must contain only real numbers. Got {value!r}."
                )
            director = np.asarray(raw_value, dtype=float)
        elif np.issubdtype(raw_value.dtype, np.floating):
            director = raw_value
        elif np.issubdtype(raw_value.dtype, np.integer) or np.issubdtype(
            raw_value.dtype, np.bool_
        ):
            director = raw_value.astype(float)
        else:
            raise TypeError(
                f"{name!r} must contain only real numbers. Got dtype "
                f"{raw_value.dtype}."
            )

        if not np.isfinite(director).all():
            raise ValueError(
                f"{name!r} must contain only finite values. Got {value!r}."
            )

        if is_normalized or not is_zero_allowed:
            norms = np.linalg.norm(director, axis=-1, keepdims=True)
            is_zero = norms <= 1e-12
            if not is_zero_allowed and np.any(is_zero):
                raise ValueError(f"{name!r} must not contain zero directors.")

        if is_normalized:
            normalized = np.zeros_like(director)
            np.divide(director, norms, out=normalized, where=~is_zero)
            director = normalized
        return director

    try:
        return validate(input_data)
    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception(f"Invalid {name!r}; attempting the configured replacement.")
        logger.recovery(f"Use {replace!r} as {name!r} in the following.")
        return validate(replace)
