from nematics3d.datatypes import as_bool, as_str
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..class_base import AttrDef
from ..registry_base import RegistryBase


# FigureManager developer conventions:
# - FigureManager extends RegistryBase with one extra managed state: the current
#   active figure name.
# - Keep that state synchronized with the actual registry contents when changing
#   registration or activation behavior.
# - Preserve the expectation that `active_fig` is a convenience view over the
#   registry, not a second storage slot for a figure object.


class FigureManager(RegistryBase):
    """
    Registry for managing PlotFigure objects and tracking which one is active.

    For most users, FigureManager is used through a host object such as
    `QFieldObject`, but it can also be used directly.

    Typical usage:

    - register figures through `act_register(fig)`
    - access a figure through `manager[name]` or `manager[index]`
    - set the current active figure through `act_set_active(name_or_index)`
    - use `active_fig` or `active_name` to access the current active figure
    - use `repr(manager)` to inspect the figures stored in display order

    If there is exactly one figure and no active figure has been set yet,
    `active_fig` will automatically fall back to that only figure.
    """

    # fmt: off
    __attr_defs__ = {
        "state_active_name": AttrDef(
            doc="Current active figure name.",
            kind="state",
            validator=lambda v, d: None if v is None else as_str(v, name=d),
            is_public_settable=True,
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

    __slots__ = ("state_active_name",)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, name: str = "figures"):
        super().__init__(name)
        object.__setattr__(self, "state_active_name", None)

    # ------------------------------------------------------------------
    # Readable properties
    # ------------------------------------------------------------------

    @property
    def active_name(self):
        return self.state_active_name

    @property
    def active_fig(self):
        return self._helper_get_active_fig()

    # ------------------------------------------------------------------
    # Active-figure helpers / actions
    # ------------------------------------------------------------------

    @logging_and_warning_decorator()
    def _helper_get_active_fig(self, logger=None):
        active_name = self.state_active_name
        if active_name is not None:
            try:
                figure = self[active_name]
            except KeyError:
                logger.warning(
                    f"The stored active figure name {active_name!r} is no longer registered. Reset it."
                )
                object.__setattr__(self, "state_active_name", None)
                active_name = None
            else:
                if not figure.is_alive:
                    logger.warning(
                        f"The active figure {figure.name!r} is not alive anymore."
                    )
                return figure

        if len(self) == 1:
            figure = self[0]
            object.__setattr__(self, "state_active_name", figure.name)
            return figure
        if len(self) == 0:
            raise KeyError(
                "There is no figure in FigureManager, so no active figure can be returned."
            )
        raise KeyError(
            "There are multiple figures in FigureManager but no active figure has been set."
        )

    def act_set_active(self, id_fig: str):
        figure = self[id_fig]
        if figure.is_alive:
            self.state_active_name = figure.name
        else:
            raise KeyError(
                "This figure is deleted and could not be set to active figure."
            )

    # ==================== OVERRIDE ====================
    # FigureManager overrides RegistryBase.act_unregister so the managed active
    # figure name cannot dangle after the active figure is removed.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def act_unregister(self, term, is_missing_ok=False, logger=None):
        was_active = getattr(term, "name", None) == self.state_active_name
        super().act_unregister(term, is_missing_ok=is_missing_ok)
        if not was_active:
            return

        object.__setattr__(self, "state_active_name", None)
        if len(self) == 1:
            object.__setattr__(self, "state_active_name", self[0].name)
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

    # ==================== OVERRIDE ====================
    # FigureManager overrides RegistryBase.act_clear because it owns the
    # active-figure state and may optionally close live figure windows.
    # ==================================================
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
        object.__setattr__(self, "state_active_name", None)

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

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})\n"
        return msg + self.act_repr_by_order()
