from collections import OrderedDict
from typing import Union

from Nematics3D.datatypes import as_str
from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import PlotFigure

class FigureManager:
    
    def __init__(self,
                 name: str = "figures"):
        
        self.name = name
        self._entity = OrderedDict()
        self._state_active_name = None
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_figure_name(self, name: str, logger=None):
        
        name_set = set(self._entity.keys())
        name_input = name
        if name_input in name_set:
            new_name = name_input
            index = 1
            while new_name in name_set:
                new_name = f"{name_input}_{index}"
                index += 1
            logger.warning(f"{name_input!r} already exists in FigureManager! Renamed to {new_name!r}.")
            name = new_name
        return name
    
    def act_add_figure(self, figure: PlotFigure):
        if any(figure is x for x in self._entity.values()):
            return
        
        name = self._helper_check_figure_name(figure.name)
        figure.name = name
        self._entity[name] = figure
        
    def act_set_active(self, id_fig: str):
        figure = self[id_fig]
        if figure:
            self._state_active_name = figure.name
        else:
            raise KeyError("This figure is deleted and could not be set to active figure.")
    
    @property
    def active_name(self):
        return self._state_active_name
    
    
    def __len__(self) -> int:
        return len(self._entity)

    def __iter__(self):
        return iter(self._entity.values())

    def __contains__(self, name: str):
        return name in self._entity

    def __getitem__(self, key: Union[str, int, None]):
        if key is None:
            return None
        elif isinstance(key, str):
            return self._entity[key]
        elif isinstance(key, int):
            names = list(self._entity.keys())
            try:
                name = names[key]
            except IndexError:
                raise KeyError(
                    f"figure index {key} out of range for FigureManager "
                    f"(size={len(names)})"
                ) from None
            return self._entity[name]
        else:
            raise TypeError(
                f"`key` must be str or int for FigureManager indexing, "
                f"got {type(key).__name__} instead."
            )
                
    
    
    
    def act_ensure_alive(self, name: str):
        fig = self._entity[name]
        