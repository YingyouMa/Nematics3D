"""Focused tests for the UNSET sentinel contract."""

import pytest

from nematics3d.datatypes import UNSET, Unset


def test_unset_identity_and_type():
    assert isinstance(UNSET, Unset)
    assert UNSET is UNSET


def test_unset_repr():
    assert repr(UNSET) == "UNSET"


def test_unset_has_no_instance_state():
    with pytest.raises(AttributeError):
        UNSET.value = 1


def test_additional_unset_instance_is_not_the_singleton():
    other = Unset()
    assert isinstance(other, Unset)
    assert other is not UNSET
