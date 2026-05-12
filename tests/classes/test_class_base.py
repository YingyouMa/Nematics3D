import sys
from pathlib import Path
import types
import unittest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.class_base import ClassBase
from nematics3d.datatypes import as_str


def _counting_name_validator(value, name="input_data", pool=None, replace=None):
    del pool
    CountingNameBase.call_count += 1
    return as_str(value, name=name, replace=replace)


def _counting_label_validator(value, name="input_data", replace=None):
    CountingFieldBase.call_count += 1
    return as_str(value, name=name, replace=replace)


class DummyBase(ClassBase):
    __slots__ = ()

    def __init__(self, name="dummy"):
        super().__init__(name=name, name_replace="dummy")


class CountingNameBase(ClassBase):
    __slots__ = ()
    call_count = 0

    __attr_defs__ = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "validator": _counting_name_validator,
        },
    }

    def __init__(self, name="counter"):
        super().__init__(name=name, name_replace="counter")


class CountingFieldBase(ClassBase):
    __slots__ = ("raw_label",)
    call_count = 0

    __attr_defs__ = {
        **dict(ClassBase.__attr_defs__),
        "raw_label": {
            "doc": "The label string for this instance.",
            "validator": _counting_label_validator,
        },
    }

    def __init__(self, name="field", label="label"):
        super().__init__(name=name, name_replace="field")
        object.__setattr__(self, "raw_label", label)


class TestClassBase(unittest.TestCase):
    def test_act_add_attr_stores_extra_value_in_impl_attrs(self):
        obj = DummyBase()
        obj.act_add_attr("tag", "Original tag doc.", default=1)

        self.assertEqual(obj.tag, 1)
        self.assertEqual(obj.impl_attrs["tag"]["kind"], "extra")
        self.assertEqual(obj.impl_attrs["tag"]["value"], 1)

        obj.act_add_attr(
            "tag",
            "Updated tag doc.",
            default=2,
            is_overwrite=True,
        )

        self.assertEqual(obj.tag, 2)
        self.assertEqual(obj.impl_attrs["tag"]["doc"], "Updated tag doc.")
        self.assertEqual(obj.impl_attrs["tag"]["kind"], "extra")
        self.assertEqual(obj.impl_attrs["tag"]["value"], 2)

    def test_act_add_attr_cannot_overwrite_managed_attr(self):
        obj = DummyBase()

        with self.assertRaisesRegex(AttributeError, "managed attribute"):
            obj.act_add_attr("owner", "Should fail.", default=1, is_overwrite=True)

    def test_name_validator_runs_once_per_name_assignment_path(self):
        CountingNameBase.call_count = 0
        obj = CountingNameBase(name="init")
        self.assertEqual(CountingNameBase.call_count, 1)

        obj.raw_name = "next"
        self.assertEqual(CountingNameBase.call_count, 2)

        obj.name = "final"
        self.assertEqual(CountingNameBase.call_count, 3)

    def test_public_raw_attr_assignment_runs_validator(self):
        CountingFieldBase.call_count = 0
        obj = CountingFieldBase()
        self.assertEqual(CountingFieldBase.call_count, 0)

        obj.label = "alias-update"
        self.assertEqual(obj.raw_label, "alias-update")
        self.assertEqual(CountingFieldBase.call_count, 1)

        obj.raw_label = "direct-update"
        self.assertEqual(obj.raw_label, "direct-update")
        self.assertEqual(CountingFieldBase.call_count, 2)

    def test_show_attr_doc_returns_registered_doc(self):
        obj = CountingFieldBase()

        self.assertEqual(
            obj.show_attr_doc("label", is_return=True),
            "The label string for this instance.",
        )


if __name__ == "__main__":
    unittest.main()
