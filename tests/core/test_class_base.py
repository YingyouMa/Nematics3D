"""Focused tests for the canonical ClassBase object protocol."""

import pytest

from nematics3d.core.class_base import AttrDef, ClassBase
from nematics3d.core.host_base import HostBase, OptsBase


def _as_positive_int(value, doc):
    del doc
    value = int(value)
    if value <= 0:
        raise ValueError("value must be positive")
    return value


class DemoBase(ClassBase):
    __attr_defs__ = {
        "raw_value": AttrDef(
            doc="Primary integer input.",
            kind="raw",
            validator=_as_positive_int,
        ),
        "calc_double": AttrDef(
            doc="Twice the primary input.",
            kind="calc",
        ),
    }

    __slots__ = ("raw_value", "calc_double")

    def __init__(self, value=2):
        super().__init__(name="demo", name_replace="demo")
        object.__setattr__(self, "raw_value", _as_positive_int(value, "raw_value"))
        object.__setattr__(self, "calc_double", 2 * self.raw_value)


class DemoHost(HostBase):
    __attr_defs__ = {
        "raw_value": AttrDef(
            doc="Primary host input.",
            kind="raw",
            validator=_as_positive_int,
        ),
    }

    __slots__ = ("raw_value",)

    def __init__(self, value=2):
        super().__init__(OptsBase, name="host")
        object.__setattr__(self, "raw_value", _as_positive_int(value, "raw_value"))

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        del is_reapply_opts
        return kwargs, {}


def test_show_attr_doc_resolves_raw_alias():
    obj = DemoBase()
    assert obj.show_attr_doc("value", is_return=True) == "Primary integer input."
    assert obj.show_attr_doc("raw_value", is_return=True) == "Primary integer input."


def test_show_attr_desc_historical_duplicate_is_removed():
    obj = DemoBase()
    assert not hasattr(obj, "show_attr_desc")


def test_extra_default_is_validated_before_storage():
    obj = DemoBase()
    obj.act_add_attr(
        "sample_count",
        "User-side sample count.",
        default="3",
        validator=_as_positive_int,
    )
    assert obj.sample_count == 3
    assert isinstance(obj.sample_count, int)


def test_extra_default_validation_failure_does_not_register_attr():
    obj = DemoBase()
    with pytest.raises(ValueError):
        obj.act_add_attr(
            "sample_count",
            "User-side sample count.",
            default=0,
            validator=_as_positive_int,
        )
    assert "sample_count" not in obj.impl_extra
    assert "sample_count" not in obj.impl_assign_state


def test_extra_attr_cannot_use_semantic_prefix_or_shadow_method():
    obj = DemoBase()
    with pytest.raises(ValueError):
        obj.act_add_attr("raw_note", "Invalid semantic-looking side data.")
    with pytest.raises(AttributeError):
        obj.act_add_attr("show_doc", "Would shadow a public method.")


def test_remove_extra_attr_cleans_value_and_assignment_state():
    obj = DemoBase()
    obj.act_add_attr("note", "Temporary note.", default="hello")
    obj.act_register_protected_attr("note")

    removed = obj.act_remove_attr("note")

    assert removed == "hello"
    assert "note" not in obj.impl_extra
    assert "note" not in obj.impl_assign_state
    with pytest.raises(AttributeError):
        _ = obj.note


def test_remove_attr_rejects_static_fields():
    obj = DemoBase()
    with pytest.raises(AttributeError):
        obj.act_remove_attr("raw_value")


def test_show_attr_info_reports_role_alias_and_mutability():
    obj = DemoBase()
    output = obj.show_attr_info("value", is_return=True)
    assert "name: raw_value" in output
    assert "kind: raw" in output
    assert "alias: value" in output
    assert "modifiable: yes" in output
    assert "doc: Primary integer input." in output


def test_calc_output_is_not_modifiable():
    obj = DemoBase()
    output = obj.show_attr_info("calc_double", is_return=True)
    assert "kind: calc" in output
    assert "modifiable: no" in output
    with pytest.raises(AttributeError):
        obj.calc_double = 10


def test_host_show_attr_doc_resolves_host_and_opts_surfaces():
    host = DemoHost()
    assert host.show_attr_doc("value", is_return=True) == "Primary host input."
    assert (
        host.show_attr_doc("tag", is_return=True)
        == "name identifier of the option settings"
    )


def test_host_readable_attrs_uses_unified_doc_path():
    host = DemoHost()
    output = host.show_readable_attrs(is_return=True)
    assert "'value': Alias of 'raw_value'. Primary host input." in output
    assert "'tag': name identifier of the option settings" in output
    assert not hasattr(host, "show_attr_desc")
