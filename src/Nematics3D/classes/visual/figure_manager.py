from Nematics3D.datatypes import as_str
from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import PlotFigure
from ..registry_base import RegistryBase


class FigureManager(RegistryBase):

    __attrs__ = {
        **(RegistryBase.__attrs__),
        "_state_active_name": "The name of current active figure",
    }
    __properties__ = {
        "active_name": "Read-only: The name of the current active figure.",
        "active_fig": "Read-only: The current active PlotFigure instance.",
    }

    __slots__ = tuple(__attrs__.keys())

    def __init__(self, name: str = "figures"):
        super().__init__(name)
        object.__setattr__(self, "_state_active_name", None)

    @property
    def active_name(self):
        return self._state_active_name

    @property
    def active_fig(self):
        return self._helper_get_active_fig()

    @logging_and_warning_decorator()
    def _helper_get_active_fig(self, logger=None):
        active_name = self._state_active_name
        if active_name is None:
            if len(self) == 1:
                figure = self[0]
                object.__setattr__(self, "_state_active_name", figure.name)
                active_name = figure.name
            elif len(self) == 0:
                raise KeyError("There is no figure in FigureManager, so no active figure can be returned.")
            else:
                raise KeyError("There are multiple figures in FigureManager but no active figure has been set.")

        figure = self[active_name]
        if not figure.is_alive:
            logger.warning(
                f"The active figure {figure.name!r} is not alive anymore."
            )
        return figure

    def act_set_active(self, id_fig: str):
        figure = self[id_fig]
        if figure.is_alive:
            self._state_active_name = figure.name
        else:
            raise KeyError("This figure is deleted and could not be set to active figure.")

    def __repr__(self):
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})\n"
        return msg + self._helper_repr_by_order()
