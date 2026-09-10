"""Focused tests for the canonical ClassBase object protocol."""

import pytest

from nematics3d.core.class_base import AttrDef, ClassBase
from nematics3d.core.host_base import HostBase, OptsBase
from nematics3d.datatypes import as_str


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


class DocumentedBase(ClassBase):
    """A documented ClassBase test object.

    Used to verify concrete-class documentation inspection.
    """

    __slots__ = ()

    def __init__(self, name="documented"):
        super().__init__(name=name, name_replace="documented")


def _counting_label_validator(value, name="input_data", replace=None):
    CountingFieldBase.call_count += 1
    return as_str(value, name=name, replace=replace)


class CountingFieldBase(ClassBase):
    __slots__ = ("raw_label",)
    call_count = 0

    __attr_defs__ = {
        "raw_label": AttrDef(
            doc="The label string for this instance.",
            kind="raw",
            validator=_counting_label_validator,
        ),
    }

    def __init__(self, name="field", label="label"):
        super().__init__(name=name, name_replace="field")
        object.__setattr__(self, "raw_label", label)


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


def test_act_add_attr_can_overwrite_existing_extra_value():
    obj = DemoBase()
    obj.act_add_attr("tag", "Original tag doc.", default=1)

    obj.act_add_attr(
        "tag",
        "Updated tag doc.",
        default=2,
        is_overwrite=True,
    )

    assert obj.tag == 2
    assert obj.impl_extra["tag"].doc == "Updated tag doc."
    assert obj.impl_extra["tag"].value == 2


def test_public_raw_attr_assignment_runs_validator():
    CountingFieldBase.call_count = 0
    obj = CountingFieldBase()
    assert CountingFieldBase.call_count == 0

    obj.label = "alias-update"
    assert obj.raw_label == "alias-update"
    assert CountingFieldBase.call_count == 1

    obj.raw_label = "direct-update"
    assert obj.raw_label == "direct-update"
    assert CountingFieldBase.call_count == 2


def test_show_doc_returns_concrete_class_docstring():
    obj = DocumentedBase()

    assert obj.show_doc(is_return=True) == (
        "A documented ClassBase test object.\n\n"
        "Used to verify concrete-class documentation inspection."
    )
