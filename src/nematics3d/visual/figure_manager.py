import weakref

from nematics3d.datatypes import as_bool
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..core.class_base import AttrDef
from ..core.registry_base import RegistryBase


# FigureManager developer conventions:
# - FigureManager extends RegistryBase with one extra managed state: a weak
#   reference to the current active figure.
# - Preserve object identity across figure renames. `active_name` is always
#   derived from the active figure instead of being stored independently.
# - Preserve the expectation that `active_fig` is a convenience view over the
#   registry, not ownership of a second strong figure reference.


class FigureManager(RegistryBase):
    """
    Registry for PlotFigure objects with a command-line-friendly active figure.

    For most users, FigureManager is used through a host object such as
    `QFieldObject`, but it can also be used directly.

    Typical usage:

    - register figures through `act_register(fig)`
    - access a figure through `manager[name]` or `manager[index]`
    - set the current active figure through `act_set_active(figure_or_name_or_index)`
    - use `active_fig` or `active_name` to access the current active figure
    - use `repr(manager)` to inspect the figures stored in display order

    If there is exactly one live figure and no active figure has been set yet,
    `active_fig` will automatically fall back to that only figure. The active
    selection follows the figure object across renames.
    """

    # fmt: off
    __attr_defs__ = {
        "impl_active_ref": AttrDef(
            doc="Weak reference to the current active figure, or None.",
            kind="impl",
        ),
        "active_name": AttrDef(
            doc="Read-only: The name of the current active figure.",
            kind="property",
        ),
        "active_fig": AttrDef(
            doc="Read-only: The current active PlotFigure instance.",
            kind="property",
        ),
    }
    # fmt: on

    __slots__ = ("impl_active_ref",)

    def __init__(self, name: str = "figures"):
        super().__init__(name)
        object.__setattr__(self, "impl_active_ref", None)

    @property
    def active_name(self):
        """Return the active figure name, or None when no figure is selected."""
        figure = self._helper_resolve_active_fig(is_required=False)
        return None if figure is None else figure.name

    @property
    def active_fig(self):
        """Return the current live active figure, resolving the single-figure fallback."""
        return self._helper_get_active_fig()

    def _helper_resolve_active_fig(self, *, is_required: bool, logger=None):
        active_ref = self.impl_active_ref
        if active_ref is not None:
            figure = active_ref()
            if figure is None or figure not in self:
                if logger is not None:
                    logger.warning(
                        "The active figure is no longer registered. Reset it."
                    )
                object.__setattr__(self, "impl_active_ref", None)
            elif not figure.is_alive:
                if logger is not None:
                    logger.warning(
                        f"The active figure {figure.name!r} is not alive anymore. Reset it."
                    )
                object.__setattr__(self, "impl_active_ref", None)
            else:
                return figure

        if len(self) == 1:
            figure = self[0]
            if not figure.is_alive:
                raise KeyError(
                    "The only registered figure is not alive, so no active figure can be returned."
                )
            object.__setattr__(self, "impl_active_ref", weakref.ref(figure))
            return figure
        if len(self) == 0:
            if not is_required:
                return None
            raise KeyError(
                "There is no figure in FigureManager, so no active figure can be returned."
            )
        if not is_required:
            return None
        raise KeyError(
            "There are multiple figures in FigureManager but no active figure has been set."
        )

    @logging_and_warning_decorator()
    def _helper_get_active_fig(self, logger=None):
        return self._helper_resolve_active_fig(is_required=True, logger=logger)

    def act_set_active(self, id_fig):
        """Set and return the active figure by registered object, name, or index."""
        is_registered_object = any(id_fig is figure for figure in self.impl_entity)
        figure = id_fig if is_registered_object else self[id_fig]
        if figure.is_alive:
            object.__setattr__(self, "impl_active_ref", weakref.ref(figure))
            return figure
        raise KeyError("This figure is deleted and could not be set to active figure.")

    @logging_and_warning_decorator(start_finish_level=5)
    def act_unregister(self, term, is_missing_ok=False, logger=None):
        active_ref = self.impl_active_ref
        was_active = active_ref is not None and active_ref() is term
        super().act_unregister(term, is_missing_ok=is_missing_ok)
        if not was_active:
            return

        object.__setattr__(self, "impl_active_ref", None)
        if len(self) == 1:
            figure = self[0]
            if figure.is_alive:
                object.__setattr__(self, "impl_active_ref", weakref.ref(figure))
        elif len(self) > 1:
            logger.warning(
                "The active figure was removed. Active figure has been reset to None."
            )

    def _helper_close_figure(self, figure):
        """Best-effort close for a PlotFigure-like registered object."""
        close = getattr(figure, "act_close", None)
        try:
            if callable(close):
                close()
        except (AttributeError, RuntimeError, ReferenceError):
            pass

    def act_clear(
        self,
        *,
        is_close: bool = False,
        is_return_removed: bool = False,
        is_show_existing: bool = True,
    ):
        is_close = as_bool(is_close, name="Whether to close figures before clearing")

        removed = tuple(self.impl_entity)
        if is_close:
            for figure in removed:
                self._helper_close_figure(figure)

        super().act_clear(
            is_return_removed=False,
            is_show_existing=is_show_existing,
        )
        object.__setattr__(self, "impl_active_ref", None)

        if is_return_removed:
            return removed
        return None

    def act_close_all(
        self,
        *,
        is_return_removed: bool = False,
        is_show_existing: bool = True,
    ):
        """Close and unregister all figures currently managed by this registry."""
        return self.act_clear(
            is_close=True,
            is_return_removed=is_return_removed,
            is_show_existing=is_show_existing,
        )

    def __repr__(self):
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})\n"
        return msg + self.act_repr_by_order()


__all__ = ["FigureManager"]
