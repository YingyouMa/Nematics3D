from Nematics3D.datatypes import as_str
from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import PlotFigure
from ..registry_base import RegistryBase

class FigureManager(RegistryBase):

    __descriptions__ = {
        **(RegistryBase.__descriptions__),
        "_state_active_name": "The name of current active figure",
    }

    __slots__ = tuple(__descriptions__.keys()) # + ('__weakref__' ,)
    
    def __init__(self, name: str = "figures"):
        super().__init__(name)
    
    @property
    def active_name(self):
        return self._state_active_name
    
    @property
    def active_fig(self):
        return self[self.active_name]
    
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

        