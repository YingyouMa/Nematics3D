"""Compatibility checks for the domain-independent core package migration."""

from nematics3d import core
from nematics3d.classes.class_base import ClassBase as LegacyClassBase
from nematics3d.classes.host_base import HostBase as LegacyHostBase
from nematics3d.classes.host_base import OptsBase as LegacyOptsBase
from nematics3d.classes.npy_array_payload import (
    NpyArrayPayload as LegacyNpyArrayPayload,
)
from nematics3d.classes.opts import merge_opts_all as legacy_merge_opts_all
from nematics3d.classes.registry_base import RegistryBase as LegacyRegistryBase
from nematics3d.classes.result_base import ResultBase as LegacyResultBase


def test_legacy_core_imports_resolve_to_canonical_objects():
    assert LegacyClassBase is core.ClassBase
    assert LegacyHostBase is core.HostBase
    assert LegacyOptsBase is core.OptsBase
    assert LegacyRegistryBase is core.RegistryBase
    assert LegacyResultBase is core.ResultBase
    assert LegacyNpyArrayPayload is core.NpyArrayPayload
    assert legacy_merge_opts_all is core.merge_opts_all


def test_core_objects_report_canonical_module_paths():
    assert core.ClassBase.__module__ == "nematics3d.core.class_base"
    assert core.HostBase.__module__ == "nematics3d.core.host_base"
    assert core.OptsBase.__module__ == "nematics3d.core.host_base"
    assert core.RegistryBase.__module__ == "nematics3d.core.registry_base"
    assert core.ResultBase.__module__ == "nematics3d.core.result_base"
    assert core.NpyArrayPayload.__module__ == "nematics3d.core.npy_array_payload"
