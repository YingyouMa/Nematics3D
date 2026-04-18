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
from nematics3d.classes.visual.figure_manager import FigureManager


class FakePlotter:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class FakeFigure(ClassBase):
    __slots__ = (
        "entity_plotter",
        "close_interacts_count",
        "state_is_alive",
        "act_close_count",
        "last_is_remove_glyphs",
    )

    def __init__(self, name):
        super().__init__(name=name, name_replace="figure")
        object.__setattr__(self, "entity_plotter", FakePlotter())
        object.__setattr__(self, "close_interacts_count", 0)
        object.__setattr__(self, "state_is_alive", True)
        object.__setattr__(self, "act_close_count", 0)
        object.__setattr__(self, "last_is_remove_glyphs", None)

    @property
    def is_alive(self):
        return self.state_is_alive

    @property
    def pl(self):
        return self.entity_plotter

    def _helper_close_interacts(self):
        object.__setattr__(
            self,
            "close_interacts_count",
            self.close_interacts_count + 1,
        )

    def act_close(self, *, is_remove_glyphs=True):
        object.__setattr__(self, "act_close_count", self.act_close_count + 1)
        object.__setattr__(self, "last_is_remove_glyphs", is_remove_glyphs)
        self._helper_close_interacts()
        self.pl.close()


class TestFigureManager(unittest.TestCase):
    def test_act_clear_resets_active_name_without_closing_by_default(self):
        manager = FigureManager()
        first = FakeFigure("first")
        second = FakeFigure("second")
        manager.act_register(first)
        manager.act_register(second)
        manager.act_set_active("second")

        removed = manager.act_clear(
            is_return_removed=True,
            is_show_existing=False,
        )

        self.assertEqual(removed, (first, second))
        self.assertEqual(len(manager), 0)
        self.assertIsNone(manager.active_name)
        self.assertEqual(first.pl.close_count, 0)
        self.assertEqual(second.pl.close_count, 0)

    def test_act_clear_can_close_figures(self):
        manager = FigureManager()
        first = FakeFigure("first")
        second = FakeFigure("second")
        manager.act_register(first)
        manager.act_register(second)

        result = manager.act_clear(is_close=True, is_show_existing=False)

        self.assertIsNone(result)
        self.assertEqual(len(manager), 0)
        self.assertIsNone(manager.active_name)
        self.assertEqual(first.close_interacts_count, 1)
        self.assertEqual(second.close_interacts_count, 1)
        self.assertEqual(first.pl.close_count, 1)
        self.assertEqual(second.pl.close_count, 1)
        self.assertEqual(first.act_close_count, 1)
        self.assertEqual(second.act_close_count, 1)
        self.assertTrue(first.last_is_remove_glyphs)
        self.assertTrue(second.last_is_remove_glyphs)

    def test_act_close_all_closes_and_can_return_removed_figures(self):
        manager = FigureManager()
        figure = FakeFigure("figure")
        manager.act_register(figure)
        manager.act_set_active("figure")

        removed = manager.act_close_all(
            is_return_removed=True,
            is_show_existing=False,
        )

        self.assertEqual(removed, (figure,))
        self.assertEqual(len(manager), 0)
        self.assertIsNone(manager.active_name)
        self.assertEqual(figure.close_interacts_count, 1)
        self.assertEqual(figure.pl.close_count, 1)
        self.assertEqual(figure.act_close_count, 1)
        self.assertTrue(figure.last_is_remove_glyphs)


if __name__ == "__main__":
    unittest.main()
