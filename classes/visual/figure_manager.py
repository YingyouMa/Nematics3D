from Nematics3D.datatypes import as_str
from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import PlotFigure
from ..registry_base import RegistryBase

class FigureManager(RegistryBase):

    __descriptions__ = {
        **(RegistryBase.__descriptions__)
    }

    __slots__ = tuple(__descriptions__.keys()) # + ('__weakref__' ,)
    
    def __init__(self, name: str = "figures"):
        super().__init__(name)
    
    @property
    def active_name(self):
        return self._state_active_name
    
    def act_set_active(self, id_fig: str):
        figure = self[id_fig]
        if figure:
            self._state_active_name = figure.name
        else:
            raise KeyError("This figure is deleted and could not be set to active figure.")

        