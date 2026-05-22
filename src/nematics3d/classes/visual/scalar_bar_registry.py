"""Scalar-bar-specific registry and backend synchronization helpers."""

from __future__ import annotations

import pyvista as pv
from pyvista.plotting.tools import parse_font_family

from ..registry_base import RegistryBase
from .scalar_bar import ScalarBar


class ScalarBarRegistry(RegistryBase):
    """Figure-owned registry that manages scalar-bar declarations and backends."""

    _WIDGET_GEOMETRY_ROUND_DIGITS = 6

    def _helper_resolve_scalar_bar(self, key):
        """Resolve one registered scalar bar from a registry-compatible key."""
        if isinstance(key, ScalarBar):
            scalar_bar = key
        else:
            scalar_bar = self[key]
        if scalar_bar is None:
            return None
        if not isinstance(scalar_bar, ScalarBar):
            raise TypeError(
                "Resolved scalar-bar registry item must be a ScalarBar instance. "
                f"Got {type(scalar_bar).__name__!r} instead."
            )
        return scalar_bar

    def _helper_resolve_scalar_bar_mapper(self, scalar_bar):
        """Resolve the live VTK mapper that should drive one scalar bar."""
        source = getattr(scalar_bar, "source", None)
        if source is None:
            raise ValueError(
                f"Scalar bar {scalar_bar!r} has no bound source relation, so no "
                "PyVista mapper can be resolved yet."
            )

        actor = getattr(source, "entity_actor", None)
        mapper = getattr(actor, "mapper", None)
        if mapper is not None:
            return mapper

        mapper = getattr(source, "mapper", None)
        if mapper is not None:
            return mapper

        raise ValueError(
            f"Failed to resolve a mapper from source {source!r} for scalar bar "
            f"{scalar_bar!r}."
        )

    def _helper_find_scalar_bar_backend_key(self, scalar_bar):
        """Return the Plotter.scalar_bars key currently bound to one backend."""
        plotter_scalar_bars = getattr(self.owner.pl, "scalar_bars", None)
        if plotter_scalar_bars is None:
            return None

        backend = getattr(scalar_bar, "backend", None)
        backend_name = scalar_bar.impl_name_pv
        if backend_name in plotter_scalar_bars:
            actor = plotter_scalar_bars[backend_name]
            if backend is None or actor is backend:
                return backend_name

        if backend is None:
            return None

        for title, actor in plotter_scalar_bars.items():
            if actor is backend:
                return title
        return None

    def _helper_resolve_scalar_bar_widget(self, scalar_bar):
        """Resolve the live PyVista scalar-bar widget corresponding to one bar."""
        backend_key = self._helper_find_scalar_bar_backend_key(scalar_bar)
        if backend_key is None:
            return None

        widgets = getattr(self.owner.pl.scalar_bars, "_scalar_bar_widgets", None)
        if widgets is None:
            return None
        return widgets.get(backend_key)

    def _helper_configure_scalar_bar_widget(self, scalar_bar):
        """Hide widget decoration and bind geometry write-back for one bar."""
        widget = self._helper_resolve_scalar_bar_widget(scalar_bar)
        scalar_bar.act_set_backend_widget(widget)
        if widget is None:
            scalar_bar.act_set_backend_widget_observer_tag(None)
            return None

        rep = widget.GetRepresentation()
        rep.SetShowBorderToOff()
        rep.SetShowPolygonToOff()
        rep.SetShowHorizontalBorder(0)
        rep.SetShowVerticalBorder(0)

        observer_tag_old = getattr(scalar_bar, "impl_backend_widget_observer_tag", None)
        if observer_tag_old is not None:
            try:
                widget.RemoveObserver(observer_tag_old)
            except (AttributeError, RuntimeError, ReferenceError):
                pass

        def _on_end_interaction(_obj, _event):
            self._helper_pull_scalar_bar_widget_geometry(scalar_bar)

        observer_tag = widget.AddObserver("EndInteractionEvent", _on_end_interaction)
        scalar_bar.act_set_backend_widget_observer_tag(observer_tag)
        return widget

    def _helper_apply_scalar_bar_visibility(self, scalar_bar):
        """Apply scalar-bar visibility without deleting the live backend."""
        backend = getattr(scalar_bar, "backend", None)
        if backend is None:
            return None

        is_visible = bool(scalar_bar.opts.is_visible)
        backend.SetVisibility(is_visible)

        widget = getattr(scalar_bar, "backend_widget", None)
        if widget is not None:
            widget.SetEnabled(1 if is_visible else 0)
        return is_visible

    def _helper_pull_scalar_bar_widget_geometry(self, scalar_bar):
        """Pull the current widget geometry back into scalar-bar opts."""
        widget = getattr(scalar_bar, "backend_widget", None)
        if widget is None:
            return None

        rep = widget.GetRepresentation()
        position = tuple(
            round(float(value), self._WIDGET_GEOMETRY_ROUND_DIGITS)
            for value in rep.GetPosition()[:2]
        )
        size = tuple(
            round(float(value), self._WIDGET_GEOMETRY_ROUND_DIGITS)
            for value in rep.GetPosition2()[:2]
        )
        payload = {
            "position": position,
            "width": size[0],
            "height": size[1],
        }

        opts_now = scalar_bar.opts
        if (
            (
                tuple(
                    round(float(v), self._WIDGET_GEOMETRY_ROUND_DIGITS)
                    for v in opts_now.position
                )
                if opts_now.position is not None
                else None
            )
            == position
            and opts_now.width == size[0]
            and opts_now.height == size[1]
        ):
            return payload

        scalar_bar.act_set_backend_sync_guard(True)
        try:
            scalar_bar.act_commit(**payload)
        finally:
            scalar_bar.act_set_backend_sync_guard(False)
        return payload

    def _helper_create_scalar_bar_backend(self, scalar_bar):
        """Create one live PyVista scalar-bar backend from the current declaration."""
        kwargs_pyvista = dict(getattr(scalar_bar, "calc_pyvista_kwargs", {}))
        mapper = self._helper_resolve_scalar_bar_mapper(scalar_bar)
        if not isinstance(mapper.lookup_table, pv.LookupTable):
            mapper.lookup_table = pv.LookupTable()
        if scalar_bar.opts.cmap is not None:
            n_values = (
                scalar_bar.opts.n_colors
                if scalar_bar.opts.n_colors is not None
                else mapper.lookup_table.n_values
            )
            mapper.lookup_table.apply_cmap(scalar_bar.opts.cmap, n_values=n_values)
        if scalar_bar.opts.clim is not None:
            mapper.scalar_range = tuple(float(x) for x in scalar_bar.opts.clim)
            mapper.lookup_table.scalar_range = tuple(
                float(x) for x in scalar_bar.opts.clim
            )
        display_title = kwargs_pyvista.pop("title", scalar_bar.opts.title)
        kwargs_create = dict(kwargs_pyvista)
        kwargs_create["title"] = scalar_bar.impl_name_pv
        kwargs_create["mapper"] = mapper
        kwargs_create["render"] = False

        backend = self.owner.pl.add_scalar_bar(**kwargs_create)
        if backend is None:
            backend_key = self._helper_find_scalar_bar_backend_key(scalar_bar)
            if backend_key is not None:
                backend = self.owner.pl.scalar_bars[backend_key]
        if backend is None:
            raise RuntimeError(
                f"Failed to create or resolve PyVista scalar-bar backend for "
                f"{scalar_bar!r}."
            )

        if display_title is not None:
            backend.SetTitle(display_title)

        scalar_bar.act_set_backend(backend)
        self._helper_configure_scalar_bar_widget(scalar_bar)
        self._helper_apply_scalar_bar_visibility(scalar_bar)
        return backend

    def _helper_update_scalar_bar_backend(self, scalar_bar):
        """
        Update one existing scalar-bar backend from the current declaration.

        This applies the current scalar-bar opts onto the existing mapper, actor,
        and interactive widget representation without recreating the backend.
        """
        backend = getattr(scalar_bar, "backend", None)
        if backend is None:
            return None

        mapper = self._helper_resolve_scalar_bar_mapper(scalar_bar)
        widget = self._helper_resolve_scalar_bar_widget(scalar_bar)
        scalar_bar.act_set_backend_widget(widget)
        opts = scalar_bar.opts

        if not isinstance(mapper.lookup_table, pv.LookupTable):
            mapper.lookup_table = pv.LookupTable()

        lut = mapper.lookup_table
        if opts.cmap is not None:
            n_values = opts.n_colors if opts.n_colors is not None else lut.n_values
            lut.apply_cmap(opts.cmap, n_values=n_values)

        if opts.clim is not None:
            mapper.scalar_range = tuple(float(x) for x in opts.clim)
            lut.scalar_range = tuple(float(x) for x in opts.clim)

        if opts.n_colors is not None:
            backend.SetMaximumNumberOfColors(opts.n_colors)

        backend.SetTitle(opts.title)
        if opts.n_labels is not None and opts.n_labels < 1:
            backend.SetDrawTickLabels(False)
        elif opts.n_labels is not None:
            backend.SetDrawTickLabels(True)
            backend.SetNumberOfLabels(opts.n_labels)

        if opts.fmt is not None:
            backend.SetLabelFormat(opts.fmt)

        if opts.width is not None:
            backend.SetWidth(opts.width)
        if opts.height is not None:
            backend.SetHeight(opts.height)
        if opts.position is not None:
            backend.SetPosition(*opts.position)

        if opts.is_vertical:
            backend.SetOrientationToVertical()
        else:
            backend.SetOrientationToHorizontal()

        if opts.is_use_opacity is not None:
            backend.SetUseOpacity(opts.is_use_opacity)
        if opts.is_outline is not None:
            backend.SetDrawFrame(opts.is_outline)
        if opts.is_unconstrained_font_size is not None:
            backend.SetUnconstrainedFontSize(opts.is_unconstrained_font_size)
        if opts.is_nan_annotation:
            backend.DrawNanAnnotationOn()
        else:
            backend.DrawNanAnnotationOff()

        if opts.is_fill:
            backend.DrawBackgroundOn()
        else:
            backend.DrawBackgroundOff()

        if opts.below_label is not None:
            backend.DrawBelowRangeSwatchOn()
            backend.SetBelowRangeAnnotation(opts.below_label)
        else:
            backend.DrawBelowRangeSwatchOff()
        if opts.above_label is not None:
            backend.DrawAboveRangeSwatchOn()
            backend.SetAboveRangeAnnotation(opts.above_label)
        else:
            backend.DrawAboveRangeSwatchOff()

        label_text = backend.GetLabelTextProperty()
        anno_text = backend.GetAnnotationTextProperty()
        title_text = backend.GetTitleTextProperty()

        if opts.font_family is not None:
            font_family = parse_font_family(opts.font_family)
            label_text.SetFontFamily(font_family)
            anno_text.SetFontFamily(font_family)
            title_text.SetFontFamily(font_family)

        if opts.is_italic is not None:
            label_text.SetItalic(opts.is_italic)
            anno_text.SetItalic(opts.is_italic)
            title_text.SetItalic(opts.is_italic)
        if opts.is_bold is not None:
            label_text.SetBold(opts.is_bold)
            anno_text.SetBold(opts.is_bold)
            title_text.SetBold(opts.is_bold)
        if opts.is_shadow is not None:
            label_text.SetShadow(opts.is_shadow)
            anno_text.SetShadow(opts.is_shadow)
            title_text.SetShadow(opts.is_shadow)

        if opts.label_font_size is not None:
            label_text.SetFontSize(opts.label_font_size)
            anno_text.SetFontSize(opts.label_font_size)
        if opts.title_font_size is not None:
            title_text.SetFontSize(opts.title_font_size)

        if opts.color is not None:
            label_text.SetColor(opts.color)
            anno_text.SetColor(opts.color)
            title_text.SetColor(opts.color)
            backend.GetFrameProperty().SetColor(opts.color)

        if opts.background_color is not None:
            backend.GetBackgroundProperty().SetColor(opts.background_color)

        if widget is not None:
            rep = widget.GetRepresentation()
            if opts.position is not None:
                rep.SetPosition(*opts.position)
            if opts.width is not None or opts.height is not None:
                width = opts.width if opts.width is not None else backend.GetWidth()
                height = opts.height if opts.height is not None else backend.GetHeight()
                rep.SetPosition2(width, height)
            rep.SetOrientation(1 if opts.is_vertical else 0)
            self._helper_configure_scalar_bar_widget(scalar_bar)

        self._helper_apply_scalar_bar_visibility(scalar_bar)
        return backend

    def _helper_render_after_scalar_bar_sync(self):
        """Render the figure plotter after a scalar-bar synchronization pass."""
        try:
            self.owner.pl.render()
        except (AttributeError, RuntimeError, ReferenceError):
            pass

    def act_register(
        self,
        scalar_bar,
        is_contain_ok=False,
        is_bind_registry_relation=True,
        logger=None,
    ):
        """Register one ScalarBar object in this figure-owned registry."""
        if not isinstance(scalar_bar, ScalarBar):
            raise TypeError(
                "scalar_bar must be a ScalarBar instance. "
                f"Got {type(scalar_bar).__name__!r} instead."
            )

        bind_relation = getattr(scalar_bar, "act_bind_relation_base", None)
        if callable(bind_relation):
            bind_relation("owner", self.owner, is_weak=True)

        super().act_register(
            scalar_bar,
            is_contain_ok=is_contain_ok,
            is_bind_registry_relation=is_bind_registry_relation,
            logger=logger,
        )
        return scalar_bar

    def act_unregister(self, key, is_missing_ok=False, logger=None):
        """Unregister one scalar bar by registry index, name, or direct object."""
        try:
            scalar_bar = self._helper_resolve_scalar_bar(key)
        except (KeyError, IndexError):
            if is_missing_ok:
                return
            raise
        if scalar_bar is None:
            if is_missing_ok:
                return
            raise KeyError("Scalar-bar key cannot be None when unregistering.")

        backend_key = self._helper_find_scalar_bar_backend_key(scalar_bar)
        if backend_key is not None:
            try:
                self.owner.pl.remove_scalar_bar(title=backend_key, render=False)
            except (AttributeError, RuntimeError, ReferenceError, KeyError, ValueError):
                pass
        scalar_bar.act_set_backend_sync_guard(True)
        scalar_bar.act_clear_backend()
        scalar_bar.act_set_backend_sync_guard(False)

        super().act_unregister(scalar_bar, is_missing_ok=is_missing_ok, logger=logger)
        if getattr(scalar_bar, "owner", None) is self.owner:
            scalar_bar.act_unbind_relation_base("owner")

    def act_sync(self, key):
        """
        Synchronize one scalar-bar declaration into the live PyVista backend.

        The current framework establishes the create/update/render workflow and
        synchronization guard. Concrete per-opts backend updates can be filled in
        later inside `_helper_update_scalar_bar_backend()`.
        """
        scalar_bar = self._helper_resolve_scalar_bar(key)
        if scalar_bar is None:
            raise KeyError("Scalar-bar key cannot be None when syncing.")

        kwargs_return = dict(getattr(scalar_bar, "calc_pyvista_kwargs", {}))
        scalar_bar.act_set_backend_sync_guard(True)
        try:
            backend_key = self._helper_find_scalar_bar_backend_key(scalar_bar)
            backend = getattr(scalar_bar, "backend", None)
            if backend_key is None or backend is None:
                self._helper_create_scalar_bar_backend(scalar_bar)
            else:
                self._helper_update_scalar_bar_backend(scalar_bar)
        finally:
            scalar_bar.act_set_backend_sync_guard(False)

        self._helper_render_after_scalar_bar_sync()
        return kwargs_return
