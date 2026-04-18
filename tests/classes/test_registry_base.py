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
from nematics3d.classes.registry_base import RegistryBase


class DemoRegistered(ClassBase):
    __slots__ = ()

    def __init__(self, name):
        super().__init__(name=name, name_replace="demo")


class TestRegistryBase(unittest.TestCase):
    def test_act_clear_unregisters_all_and_unbinds_registry(self):
        registry = RegistryBase("demo registry")
        first = DemoRegistered("first")
        second = DemoRegistered("second")
        registry.act_register(first)
        registry.act_register(second)

        result = registry.act_clear(is_show_existing=False)

        self.assertIsNone(result)
        self.assertEqual(len(registry), 0)
        self.assertIsNone(first.registry)
        self.assertIsNone(second.registry)

    def test_act_clear_can_return_removed_objects(self):
        registry = RegistryBase("demo registry")
        first = DemoRegistered("first")
        second = DemoRegistered("second")
        registry.act_register(first)
        registry.act_register(second)

        removed = registry.act_clear(
            is_return_removed=True,
            is_show_existing=False,
        )

        self.assertEqual(removed, (first, second))
        self.assertEqual(len(registry), 0)


if __name__ == "__main__":
    unittest.main()
