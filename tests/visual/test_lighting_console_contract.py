from types import SimpleNamespace

import pytest
from qtpy import QtWidgets

from nematics3d.visual.qt.lighting_console import LightingConsole


@pytest.fixture(scope="module", autouse=True)
def qapplication():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class FakeHost:
    def __init__(self):
        self.name = "glyph"
        self.opts = SimpleNamespace(
            shading_type="phong",
            ambient=0.2,
            diffuse=0.7,
            specular=0.2,
            specular_power=20.0,
            specular_color=(1.0, 1.0, 1.0),
            metallic=0.0,
            roughness=0.5,
        )
        self.sync = {}
        self.commits = []

    def act_attach_sync_task(self, name, func):
        self.sync[name] = func

    def act_detach_sync_task(self, name):
        self.sync.pop(name, None)

    def act_commit(self, **kwargs):
        self.commits.append(kwargs)
        for key, value in kwargs.items():
            setattr(self.opts, key, value)


class FakeParent(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.str_now = "panel_test"
        self.slider_throttle_ms = 20
        self._lighting_console = None


def test_requires_lighting_compatible_host():
    parent = FakeParent()
    with pytest.raises(TypeError, match="lighting-compatible host"):
        LightingConsole(object(), parent)


def test_commit_collects_current_lighting_state():
    host = FakeHost()
    parent = FakeParent()
    console = LightingConsole(host, parent)
    console.state["ambient"] = 0.4
    console.state["metallic"] = 0.6

    console.commit()

    assert host.commits[-1]["ambient"] == pytest.approx(0.4)
    assert host.commits[-1]["metallic"] == pytest.approx(0.6)
    assert host.commits[-1]["specular_color"] == (1.0, 1.0, 1.0)
    console.close()


def test_sync_updates_widget_state_without_committing_back():
    host = FakeHost()
    parent = FakeParent()
    console = LightingConsole(host, parent)
    commit_count = len(host.commits)
    host.opts.roughness = 0.8

    console._sync_func(roughness=0.8)

    assert console.state["roughness"] == pytest.approx(0.8)
    assert len(host.commits) == commit_count
    console.close()
