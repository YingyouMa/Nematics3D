from collections import OrderedDict
from typing import Union

from Nematics3D.datatypes import as_str
from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import PlotFigure

class FigureManager:
    
    def __init__(self,
                 name: str = "figures"):
        
        self.name = name
        self._entities = OrderedDict()
        self._state_active_name = None
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_figure_name(self, name: str, logger=None):
        
        name_set = set(self._entities.keys())
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
        if any(figure is x for x in self._entities.values()):
            return
        
        name = self._helper_check_figure_name(figure.opts.name)
        figure.opts.name = name
        self._entities[name] = figure
        figure.opts._state_is_name_locked = True
        
    def act_set_active(self, id_fig: str):
        if isinstance(id_fig, str):
            name = id_fig
            if name in self._entities:
                if self._entities[name].act_check_is_alive():
                    self._state_active_name = name
                else:
                    raise RuntimeError(f"figure {name!r} is not alive.")
            else:
                raise KeyError(f'{name} does not exist in FigureManager {self.name}')
        elif isinstance(id_fig, int):
            self._state_active_name = list(self._entities.keys())[id_fig]
        else:
            raise TypeError("`id_fig` is used to identify the figure. It must be either the name or the index of a figure. Got {type(id_fig!r)} instead.")
    
    @property
    def active_name(self):
        return self._state_active_name
    
    
    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self):
        return iter(self._entities.values())

    def __contains__(self, name: str):
        return name in self._entities

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, str):
            return self._entities[key]
        elif isinstance(key, int):
            names = list(self._entities.keys())
            try:
                name = names[key]
            except IndexError:
                raise KeyError(
                    f"figure index {key} out of range for FigureManager "
                    f"(size={len(names)})"
                ) from None
            return self._entities[name]
        else:
            raise TypeError(
                f"`key` must be str or int for FigureManager indexing, "
                f"got {type(key).__name__} instead."
            )
                
    
    
    
    def act_ensure_alive(self, name: str):
        fig = self._entities[name]
        