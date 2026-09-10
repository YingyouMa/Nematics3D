"""High-level convenience workflows for common Nematics3D tasks."""

from importlib import import_module

__all__ = ["quick_visualize_q"]


def __getattr__(name):
    if name == "quick_visualize_q":
        value = getattr(import_module(".q", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
