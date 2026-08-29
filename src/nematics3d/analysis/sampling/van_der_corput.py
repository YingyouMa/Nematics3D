"""Low-discrepancy sampling sequences."""

import numpy as np

from ...datatypes import as_number


def sample_van_der_corput(num: int) -> np.ndarray:
    """Return the first ``num`` points of the base-2 van der Corput sequence.

    The sequence starts as ``0, 1/2, 1/4, 3/4, 1/8, 5/8, ...`` and lies in
    the half-open unit interval ``[0, 1)``.
    """
    num = as_number(
        num,
        name="num",
        is_integer=True,
        value_range=(0, np.inf),
    )

    result = np.empty(num, dtype=float)
    if num == 0:
        return result

    result[0] = 0.0

    start = 1
    while start < num:
        stop = min(2 * start, num)
        indices = np.arange(start, stop)
        result[start:stop] = (
            0.5 * result[indices >> 1] + 0.5 * (indices & 1)
        )
        start = stop

    return result
