"""Checks for the domain-independent core package."""

from nematics3d import core


def test_core_objects_report_canonical_module_paths():
    assert core.ClassBase.__module__ == "nematics3d.core.class_base"
    assert core.HostBase.__module__ == "nematics3d.core.host_base"
    assert core.OptsBase.__module__ == "nematics3d.core.host_base"
    assert core.RegistryBase.__module__ == "nematics3d.core.registry_base"
    assert core.ResultBase.__module__ == "nematics3d.core.result_base"
    assert core.NpyArrayPayload.__module__ == "nematics3d.core.npy_array_payload"
