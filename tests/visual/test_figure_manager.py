import unittest

from nematics3d.core.class_base import ClassBase
from nematics3d.visual.figure_manager import FigureManager


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
    def test_single_registered_figure_becomes_active_on_access(self):
        manager = FigureManager()
        figure = FakeFigure("figure")
        manager.act_register(figure)

        self.assertIs(manager.active_fig, figure)
        self.assertEqual(manager.active_name, "figure")

    def test_multiple_figures_require_explicit_active_selection(self):
        manager = FigureManager()
        manager.act_register(FakeFigure("first"))
        manager.act_register(FakeFigure("second"))

        self.assertIsNone(manager.active_name)
        with self.assertRaises(KeyError):
            _ = manager.active_fig

    def test_act_set_active_accepts_name_index_and_registered_object(self):
        manager = FigureManager()
        first = FakeFigure("first")
        second = FakeFigure("second")
        manager.act_register(first)
        manager.act_register(second)

        self.assertIs(manager.act_set_active("second"), second)
        self.assertIs(manager.active_fig, second)
        self.assertIs(manager.act_set_active(0), first)
        self.assertIs(manager.active_fig, first)
        self.assertIs(manager.act_set_active(second), second)
        self.assertIs(manager.active_fig, second)

    def test_active_identity_survives_rename(self):
        manager = FigureManager()
        first = FakeFigure("first")
        second = FakeFigure("second")
        manager.act_register(first)
        manager.act_register(second)
        manager.act_set_active(first)

        first.act_set_name("renamed")

        self.assertIs(manager.active_fig, first)
        self.assertEqual(manager.active_name, "renamed")

    def test_unregister_active_figure_falls_back_to_only_remaining_figure(self):
        manager = FigureManager()
        first = FakeFigure("first")
        second = FakeFigure("second")
        manager.act_register(first)
        manager.act_register(second)
        manager.act_set_active(first)

        manager.act_unregister(first)

        self.assertIs(manager.active_fig, second)
        self.assertEqual(manager.active_name, "second")

    def test_dead_active_figure_is_not_returned(self):
        manager = FigureManager()
        first = FakeFigure("first")
        second = FakeFigure("second")
        manager.act_register(first)
        manager.act_register(second)
        manager.act_set_active(first)
        object.__setattr__(first, "state_is_alive", False)

        self.assertIsNone(manager.active_name)
        with self.assertRaises(KeyError):
            _ = manager.active_fig

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
