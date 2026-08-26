import numpy as np
import pytest

from nematics3d.datatypes import as_bool


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.bool_(False), False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        (np.int32(1), True),
        (np.float32(0), False),
    ],
)
def test_as_bool_accepts_boolean_like_scalars(value, expected):
    result = as_bool(value)

    assert result is expected
    assert type(result) is bool


@pytest.mark.parametrize("value", [2, -1, 0.5, np.nan, np.inf])
def test_as_bool_rejects_other_real_values(value):
    with pytest.raises(ValueError, match="numerically equal to 0 or 1"):
        as_bool(value)


@pytest.mark.parametrize("value", ["true", None, [1], np.array(1)])
def test_as_bool_rejects_non_scalar_or_non_real_inputs(value):
    with pytest.raises(TypeError, match="must be a boolean"):
        as_bool(value)


def test_as_bool_revalidates_replacement():
    assert as_bool("invalid", replace=np.int32(1), log_mode="none") is True

    with pytest.raises(TypeError, match="must be a boolean"):
        as_bool("invalid", replace="also invalid", log_mode="none")
