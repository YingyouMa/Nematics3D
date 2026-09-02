import sys
from dataclasses import dataclass
from pathlib import Path
import types
import unittest
from types import MappingProxyType

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.core.class_base import AttrDef
from nematics3d.core.host_base import HostBase, OptsBase


@dataclass(slots=True, repr=False)
class DemoOpts(OptsBase):
    width: int = 1

    __attrs__ = {
        **OptsBase.__attrs__,
        "width": "The width of the demo object.",
    }

    impl_validators = {
        **OptsBase.impl_validators,
        "width": lambda value, desc: int(value),
    }

    impl_defaults_frozen = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "width": 1,
        }
    )


class DemoHost(HostBase):
    __attr_defs__ = {
        "raw_level": AttrDef(
            doc="The level of the demo host.",
            kind="raw",
            validator=lambda value, desc: int(value),
        ),
        "theme": AttrDef(
            doc="Writable display theme.",
            kind="property",
            is_public_settable=True,
        ),
    }

    __slots__ = ("raw_level", "_theme")

    def __init__(self):
        super().__init__(
            opts_type=DemoOpts,
            name="demo",
            name_replace="demo",
        )
        object.__setattr__(self, "raw_level", 1)
        object.__setattr__(self, "_theme", "light")

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        object.__setattr__(self, "_theme", value)

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        del is_reapply_opts
        kwargs_left = {}
        for key, value in kwargs.items():
            object.__setattr__(self.opts, key, value)
        return kwargs_left, dict(kwargs)


class BadReturnHost(DemoHost):
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        del is_reapply_opts, kwargs
        return "bad-return"


@dataclass(slots=True, repr=False)
class MissingAttrFieldOpts(OptsBase):
    __attrs__ = {
        **OptsBase.__attrs__,
        "ghost": "Missing dataclass field.",
    }


@dataclass(slots=True, repr=False)
class InvalidValidatorKeyOpts(OptsBase):
    width: int = 1

    __attrs__ = {
        **OptsBase.__attrs__,
        "width": "Width field.",
    }

    impl_validators = {
        **OptsBase.impl_validators,
        "ghost": int,
    }


@dataclass(slots=True, repr=False)
class InvalidDefaultKeyOpts(OptsBase):
    width: int = 1

    __attrs__ = {
        **OptsBase.__attrs__,
        "width": "Width field.",
    }

    impl_defaults_frozen = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "ghost": 3,
        }
    )


class TestHostBase(unittest.TestCase):
    def test_opts_attrs_keys_must_be_dataclass_fields(self):
        with self.assertRaisesRegex(ValueError, "__attrs__ keys"):
            MissingAttrFieldOpts()

    def test_opts_validators_keys_must_belong_to_attrs(self):
        with self.assertRaisesRegex(ValueError, "impl_validators keys"):
            InvalidValidatorKeyOpts()

    def test_opts_defaults_keys_must_belong_to_attrs(self):
        with self.assertRaisesRegex(ValueError, "impl_defaults_frozen keys"):
            InvalidDefaultKeyOpts()

    def test_show_modifiable_attrs_excludes_forbidden_properties(self):
        host = DemoHost()
        host.act_register_protected_attr("theme")

        output = host.show_modifiable_attrs(is_return=True)

        self.assertNotIn("'theme': Writable display theme.", output)
        self.assertIn("Protected or wrapped fields are excluded", output)

    def test_internal_impl_assignment_rejects_unknown_field(self):
        opts = DemoOpts()

        with self.assertRaisesRegex(AttributeError, "Invalid internal option field"):
            opts.impl_missing = 1

    def test_commit_apply_opts_main_must_return_two_tuple(self):
        host = BadReturnHost()

        with self.assertRaisesRegex(TypeError, "must return None or a 2-tuple"):
            host.act_commit(width=3)

    def test_commit_extra_ignores_protected_extra_attr(self):
        host = DemoHost()
        host.act_add_attr("note", "Extra note.", default="a")
        host.act_register_protected_attr("note")

        host.act_commit(note="b")

        self.assertEqual(host.note, "a")
        self.assertIn("note", host.attrs_protected)
        self.assertIn("note", host.attrs_forbidden)

    def test_fixed_opts_protects_and_unprotects_all_opts_fields(self):
        host = DemoHost()
        host.act_register_protected_opts_all()

        host.act_commit(width=5)
        host.width = 6
        host.opts.width = 7

        self.assertEqual(host.opts.width, 1)
        self.assertIn("tag", host.attrs_protected)
        self.assertIn("width", host.attrs_protected)

        host.act_unregister_protected_opts_all()
        host.act_commit(width=5)

        self.assertEqual(host.opts.width, 5)
        self.assertNotIn("width", host.attrs_protected)

    def test_show_modifiable_attrs_excludes_forbidden_extra_attr(self):
        host = DemoHost()
        host.act_add_attr("note", "Extra note.", default="a")
        host.act_register_protected_attr("note")

        output = host.show_modifiable_attrs(is_return=True)

        self.assertNotIn("'note': Extra note.", output)
        self.assertIn("note", host.attrs_forbidden)

    def test_register_wrapped_attr_accepts_extra_attr(self):
        host = DemoHost()
        host.act_add_attr("note", "Extra note.", default="a")

        host.act_register_wrapped_attr("note")

        self.assertIn("note", host.attrs_wrapped)
        self.assertIn("note", host.attrs_forbidden)


if __name__ == "__main__":
    unittest.main()
