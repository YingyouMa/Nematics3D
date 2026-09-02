# Core Package Migration

## Direction

Nematics3D is moving away from organizing domain objects by the Python
implementation category "class". New and migrated code should be grouped by
its functional domain. Only domain-independent object-model infrastructure
belongs in `nematics3d.core`.

The long-term goal is to remove `nematics3d.classes` after its domain modules
have moved incrementally to functional packages. This must not be attempted as
one repository-wide rename.

## First migration batch

The following modules are now canonical under `src/nematics3d/core/`:

| Canonical module | Main public objects |
| --- | --- |
| `core.class_base` | `AttrDef`, `ClassBase`, relation and assignment state dataclasses |
| `core.host_base` | `OptsBase`, `HostBase` |
| `core.registry_base` | `RegistryBase` |
| `core.result_base` | `ResultBase` |
| `core.opts` | generic opts merge, override, diff, and load helpers |
| `core.npy_array_payload` | `NpyArrayPayload` |

These modules are independent of concrete nematic, grid, surface, defect,
sampling, or visualization concepts.

## Compatibility policy

The old modules under `src/nematics3d/classes/` remain as thin compatibility
imports during the migration. For example, both imports currently resolve to
the same class object:

```python
from nematics3d.core.class_base import ClassBase
from nematics3d.classes.class_base import ClassBase as LegacyClassBase

assert ClassBase is LegacyClassBase
```

Package source code should use `nematics3d.core` directly. Tests may retain
selected legacy imports to verify compatibility until the old package is
formally deprecated and removed.

No runtime deprecation warning is emitted yet. Removing legacy paths requires
a separate compatibility decision and release note.

## Migration rules for later batches

1. Move one coherent functional area at a time.
2. Establish the new canonical package before changing internal imports.
3. Keep a thin old-path compatibility module while external users may depend
   on the previous import.
4. Confirm old and new imports resolve to the same objects.
5. Preserve top-level public imports unless an explicit API change is approved.
6. Run focused tests for the migrated area and the full suite when feasible.
7. Update repository instructions that identify authoritative source files.

Potential later functional packages include sampling, surface geometry,
disclination analysis, grid-field objects, and visualization. Their exact
boundaries should be decided when each area is migrated rather than fixed in
advance.
