from types import SimpleNamespace

from nematics3d.classes.visual.pick_manager import (
    OptsPickManager,
    PickManager,
    _ClickTracker,
    _Marker,
)


def _manager_without_qt_init():
    manager = PickManager.__new__(PickManager)
    object.__setattr__(manager, "opts", OptsPickManager())
    object.__setattr__(manager, "_impl_registry", {})
    return manager


def test_click_tracker_requires_same_actor_and_resets_after_double_click():
    tracker = _ClickTracker()
    actor_a = object()
    actor_b = object()

    assert tracker.consume(actor_a, 1.0, 0.3) is False
    assert tracker.consume(actor_b, 1.1, 0.3) is False
    assert tracker.consume(actor_b, 1.2, 0.3) is True
    assert tracker.consume(actor_b, 1.25, 0.3) is False


def test_click_tracker_does_not_double_click_after_threshold():
    tracker = _ClickTracker()
    actor = object()

    assert tracker.consume(actor, 1.0, 0.3) is False
    assert tracker.consume(actor, 1.31, 0.3) is False


def test_actor_registry_replaces_owner_and_missing_unregister_is_safe():
    manager = _manager_without_qt_init()
    actor = object()
    owner_a = object()
    owner_b = object()

    manager.act_register(actor, owner_a)
    assert manager._impl_registry[actor] is owner_a
    manager.act_register(actor, owner_b)
    assert manager._impl_registry[actor] is owner_b
    manager.act_unregister(actor)
    assert actor not in manager._impl_registry
    manager.act_unregister(actor)


def test_highlight_request_restores_temporary_silhouette_suppression():
    manager = _manager_without_qt_init()
    calls = []

    class Owner:
        state_is_silhouette = False
        entity_silhouette = None

        def act_highlight(self, **kwargs):
            calls.append(kwargs)

    owner = Owner()
    manager._helper_toggle_highlight(owner)

    assert owner.state_is_silhouette is True
    assert calls == [
        {
            "color": manager.opts.sil_color,
            "opacity": manager.opts.sil_opacity,
            "width": manager.opts.sil_width,
        }
    ]


def test_highlight_request_dehighlights_visible_silhouette():
    manager = _manager_without_qt_init()
    calls = []
    owner = SimpleNamespace(
        entity_silhouette=SimpleNamespace(visibility=True),
        act_dehighlight=lambda: calls.append("dehighlight"),
    )

    manager._helper_toggle_highlight(owner)

    assert calls == ["dehighlight"]


def test_clear_markers_releases_all_overlay_resources_and_click_state():
    calls = []

    class Overlay:
        def RemoveActor(self, actor):
            calls.append(("actor", actor))

        def RemoveActor2D(self, actor):
            calls.append(("text", actor))

    overlay = Overlay()
    marker = _Marker(overlay, object(), object(), "normal", "normal_text")
    helper = _Marker(overlay, object(), object(), "helper", "helper_text")

    manager = _manager_without_qt_init()
    object.__setattr__(manager, "_entity_markers", [marker])
    object.__setattr__(manager, "_entity_helper_markers", {"helper": helper})
    object.__setattr__(manager, "_state_left_click", _ClickTracker(1.0, object()))
    object.__setattr__(manager, "_state_right_click", _ClickTracker(2.0, object()))
    object.__setattr__(manager, "_impl_owner_ref", lambda: None)

    manager.act_clear_markers()

    assert calls == [
        ("actor", "normal"),
        ("text", "normal_text"),
        ("actor", "helper"),
        ("text", "helper_text"),
    ]
    assert manager._entity_markers == []
    assert manager._entity_helper_markers == {}
    assert manager._state_left_click.last_time is None
    assert manager._state_right_click.last_time is None
