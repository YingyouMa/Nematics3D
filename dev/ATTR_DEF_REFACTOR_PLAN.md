# AttrDef Refactoring Plan

## Problem: What Is Wrong with the Current Design

`impl_attrs` currently serves two unrelated purposes in a single dict:

1. **Static schema** — what an attribute *is*: its doc, validator, kind, and flags
   declared at class definition time and never changed.
2. **Runtime state** — what an attribute's *current state* is: whether it is
   protected, the current relation target, the current extra-attr value, etc.

This mixture causes several concrete problems:

- Every instance creation calls `deepcopy(__attr_defs__)`, copying long doc
  strings, nested dicts, and function references — even though the static schema
  never changes across instances.
- There is no type safety. A typo like `.get("validtor")` silently returns
  `None` instead of raising an error.
- The static schema can be accidentally mutated at runtime with no protection.
- Subclasses must manually write `{**Parent.__attr_defs__, ...}` to merge parent
  fields. Forgetting this silently drops parent fields at runtime.
- Relation fields require six boilerplate lines per declaration, three of which
  are runtime-state placeholders that leak internal implementation details into
  the class definition.
- The naming conventions (`raw_`, `calc_`, etc.) are enforced only by comments,
  not by any runtime or static check.

---

## Proposed Design: Three Separate Structures

### 1. `AttrDef` — Static Schema (class-level, shared, never copied)

```python
@dataclass(frozen=True, slots=True)
class AttrDef:
    doc: str
    kind: Literal[
        "raw", "state", "default",   # input layer
        "calc", "entity",            # output layer
        "impl",                      # internal implementation
        "relation", "property",      # special types
    ]
    validator: Callable | None = None
    is_reapply_opts_after_raw: bool = False
    is_public_settable: bool | None = None
    is_weak_by_default: bool = True
```

`frozen=True` means the static schema cannot be mutated at runtime.
`slots=True` means attribute access is faster than a dict lookup.

### 2. `AssignState` — Per-instance Assignment Control State

```python
# Defined in ClassBase
@dataclass(slots=True)
class AssignState:
    is_protected: bool = False
```

`AssignState` tracks assignment-control flags for public settable fields.
`ClassBase` only defines `is_protected`. Subclasses that introduce additional
assignment-control concerns should subclass `AssignState` and add their own
fields:

```python
# Defined in HostBase
@dataclass(slots=True)
class HostAssignState(AssignState):
    is_wrapped: bool = False
```

`HostBase.__init__` creates `HostAssignState` instances instead of plain
`AssignState`. `ClassBase` never needs to know about `is_wrapped`.

Only public settable fields (`raw_*`, `state_*`, writable properties, extra
attrs) get an `AssignState` entry. Read-only outputs (`calc_`, `entity_`) and
internal fields (`impl_`) do not.

### 3. `RelationState` — Per-instance Relation Binding State

```python
# Defined in ClassBase
@dataclass(slots=True)
class RelationState:
    is_weak: bool | None = None
    relation_value: object = None
    doc_runtime: str | None = None
```

`RelationState` tracks the live binding of a declared relation. Every field
declared with `kind="relation"` gets one `RelationState` entry per instance,
initialized with all fields set to their defaults (unbound state).

### 3. `ExtraAttrEntry` — Dynamically Registered Extra Attributes

```python
@dataclass(slots=True)
class ExtraAttrEntry:
    doc: str
    value: object = None
    validator: Callable | None = None
    is_protected: bool = False
```

Extra attributes are registered at runtime via `act_add_attr()` and stored in a
separate per-instance dict `impl_extra`. They are intentionally kept out of the
static `__attr_defs__` system so that dynamic registration does not invalidate
the static-schema assumptions.

---

## Extensible `kind` Validation

The set of valid `kind` values is not hardcoded in `ClassBase`. Each layer of
the class hierarchy can extend it:

```python
class ClassBase:
    _VALID_KINDS: ClassVar[frozenset[str]] = frozenset({
        "raw", "state", "default", "calc", "entity",
        "impl", "relation", "property",
    })

class HostBase(ClassBase):
    _VALID_KINDS = ClassBase._VALID_KINDS | {"bridge"}
    # "bridge" covers: opts, opts_defaults, opts_backup
```

A subclass that needs a new field category must explicitly extend `_VALID_KINDS`.
Undeclared kinds are rejected at class definition time.

---

## `__init_subclass__`: Auto-merge and Validation

`ClassBase.__init_subclass__` handles two responsibilities at class creation
time, before any instance is constructed:

```python
class ClassBase:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. Auto-merge __attr_defs__ from the full MRO
        merged = {}
        for base in reversed(cls.__mro__[1:]):
            merged.update(getattr(base, "__attr_defs__", {}))
        merged.update(vars(cls).get("__attr_defs__", {}))
        cls.__attr_defs__ = merged

        # 2. Validate name–kind consistency for every declared field
        for name, defn in cls.__attr_defs__.items():
            _validate_name_kind(name, defn.kind, cls._VALID_KINDS)
```

`_validate_name_kind` checks bidirectionally:

- A field with `kind="raw"` must start with `raw_`.
- A field whose name starts with `raw_` must have `kind="raw"`.
- Analogously for `state_`, `calc_`, `entity_`, `impl_`.
- Relations, properties, and bridge fields use their direct public names and are
  validated by kind alone.

Errors are raised at **import time**, not at runtime.

---

## Relation Declaration Policy

All relations must be declared in `__attr_defs__` at class definition time.
Dynamic creation of undeclared relations at runtime is not permitted.

`act_bind_relation_base` enforces this:

```python
def act_bind_relation_base(self, name, target, ...):
    if name not in type(self).__attr_defs__:
        raise AttributeError(
            f"Cannot bind undeclared relation {name!r}. "
            f"Add it to {type(self).__name__}.__attr_defs__ first."
        )
    if not self._helper_is_relation_attr(name):
        raise AttributeError(
            f"{name!r} is not declared as a relation."
        )
    ...
```

### Rationale

Relations are structural connections between objects, not user data. They are
determined by the class design, not by runtime inputs. Every relation a class
can participate in is knowable at class definition time.

Locking down dynamic creation means the complete relation interface of any
object is visible by reading its `__attr_defs__` alone. No hidden runtime state
can be silently attached by external code.

Any relation found to be used dynamically in the current codebase (such as
`bounds_visual_source` in `PlotTube`) should be treated as a missing declaration
and added to the appropriate `__attr_defs__` during migration.

---

## How Subclass Declarations Change

### Before (PlotGlyph, current)

```python
__attr_defs__ = {
    **dict(HostBase.__attr_defs__),      # manual merge — forgetting this silently drops fields
    "fig": {
        "doc": "The PlotFigure.",
        "kind": "relation",
        "is_weak_by_default": True,
        "is_weak": None,                 # runtime state mixed into the declaration
        "relation_value": None,
        "doc_runtime": None,
    },
    "raw_coords": {
        "doc": "The N x 3 coordinates.",
        "validator": lambda v, d: as_points(v, name=d),
        "is_reapply_opts_after_raw": True,
    },
    "calc_color": {"doc": "Resolved per-point RGB color."},
}
```

### After

```python
__attr_defs__ = {                        # no manual merge needed
    "fig": AttrDef(
        doc="The PlotFigure.",
        kind="relation",
    ),
    "raw_coords": AttrDef(
        doc="The N x 3 coordinates.",
        kind="raw",
        validator=lambda v, d: as_points(v, name=d),
        is_reapply_opts_after_raw=True,
    ),
    "calc_color": AttrDef(doc="Resolved per-point RGB color.", kind="calc"),
}
```

Runtime state fields (`is_weak`, `relation_value`, `doc_runtime`) disappear from
the declaration entirely — they are created automatically in `AttrState` during
instance initialization.

---

## Expected Benefits

### Safety

| Issue | Before | After |
|---|---|---|
| Static schema mutated at runtime | Silent success | `FrozenInstanceError` immediately |
| Typo in `.get("key")` | Returns `None` silently | `AttributeError` on access |
| Subclass forgets to merge parent | Fields silently dropped | Auto-merged by `__init_subclass__` |
| Name/kind inconsistency | Caught only by code review | Caught at import time |
| Extra attrs polluting static schema | Mixed into `impl_attrs` | Isolated in `impl_extra` |

### Code Simplicity

- Subclasses no longer write `{**Parent.__attr_defs__, ...}`.
- Relation declarations shrink from six lines to one.
- `_helper_is_raw_attr`, `_helper_is_impl_attr`, and similar helpers replace
  scattered `startswith()` string checks with a single `kind` lookup.
- The ~40-line naming-convention comment block in `class_base.py` can be removed
  — the convention becomes executable code.

### Encapsulation

- "What this attribute is" (`AttrDef`, frozen, class-shared) is cleanly
  separated from "what this attribute's current state is" (`AttrState`, mutable,
  per-instance).
- Runtime state no longer leaks into class declarations.
- Dynamic extra attributes are isolated in `impl_extra` and do not affect the
  static-schema assumptions.
- New field categories can be introduced by subclasses through `_VALID_KINDS`
  extension, without requiring `ClassBase` to anticipate them in advance.

### Efficiency

- Instance creation no longer calls `deepcopy` on the full nested dict
  structure. The static schema is shared at the class level and never copied.
- `AttrState` is a `slots=True` dataclass. Creating N small slots instances is
  significantly cheaper than deepcopying N nested dicts with long doc strings.
- Attribute access changes from `dict.get(key)` (hash lookup with default) to
  slots attribute access, which is approximately 2× faster.
- Only attributes that need runtime state get an `AttrState` entry. Read-only
  outputs (`calc_`, `entity_`) consume no per-instance state at all.

---

## Migration Strategy

The refactoring can be split into two independent steps to reduce risk:

**Step 1 — Auto-merge only (low risk, standalone)**

Add `__init_subclass__` with the MRO merge logic. Remove all
`{**Parent.__attr_defs__, ...}` patterns from every subclass. Add
`_validate_name_kind` with name/kind consistency checks. This step does not
change any runtime data structures and can be tested independently.

**Step 2 — AttrDef + AttrState split (high impact, requires Step 1)**

Introduce `AttrDef`, `AttrState`, and `ExtraAttrEntry`. Update all
`impl_attrs[x].get("y")` call sites to the new access pattern
(`attr_def.y` for static fields, `impl_state[x].y` for runtime state,
`impl_extra[x]` for extra attrs). This step touches `ClassBase`,
`HostBase`, and all access paths in `PlotGlyph` and its subclasses.
---

## Recommended Storage Model

The `AttrDef` / `AttrState` split answers two questions:

- what an attribute *is*
- what mutable runtime flags it currently has

It does **not** yet answer a third equally important question:

- where the attribute's **current value** is physically stored

That storage model should be made explicit before the refactor begins.

### Recommendation: Keep Real Field Values in Slots

For this repository, the recommended design is:

- keep normal field values in real instance attributes / `__slots__`
- move only runtime control state into dedicated side containers
- keep dynamic extra attrs in a separate dict

This preserves the current fast path for ordinary field access while still
eliminating the dangerous schema/state mixture in `impl_attrs`.

### Concrete Storage Rules

#### 1. Normal managed fields stay slot-backed

The current value of these categories should continue to live in real instance
attributes:

- `raw_*`
- `state_*`
- `default_*`
- `calc_*`
- `entity_*`
- `impl_*`
- `bridge` fields such as `opts`, `opts_defaults`, `opts_backup`

Examples:

- `self.raw_coords`
- `self.calc_result`
- `self.impl_sync_func`
- `self.opts`

These are the main working values of the object and should remain on the fast
attribute path rather than being moved into a generic `_store[name]` dict.

#### 2. Relation runtime state lives in a dedicated relation-state container

Relation declarations remain in `__attr_defs__` as `AttrDef(kind="relation")`,
but the live relation binding should not be stored in the schema.

Instead, store relation runtime state in a dedicated per-instance mapping such
as:

```python
impl_relation_state: dict[str, RelationState]
```

where `RelationState` contains:

- `is_weak`
- `relation_value`
- `doc_runtime`

So:

- the schema says `"owner" is a relation`
- the instance relation state says `"owner" currently points to X`

#### 3. Assignment-control flags live in a separate assign-state container

Protection flags should not live inside relation state and should not live on
the shared schema.

Store them in a dedicated per-instance mapping:

```python
impl_assign_state: dict[str, AssignState]
```

`ClassBase` populates this with plain `AssignState(is_protected=False)` entries.
`HostBase` overrides the initialization to use `HostAssignState` entries, which
also carry `is_wrapped`. This keeps assignment control orthogonal to value
storage and relation storage, and keeps `HostBase`-specific concerns out of
`ClassBase`.

#### 4. Extra attrs live in their own container

Dynamic extra attrs should be stored separately:

```python
impl_extra: dict[str, ExtraAttrEntry]
```

where `ExtraAttrEntry` contains:

- `doc`
- `value`
- `validator`
- `is_protected`

Extra attrs are intentionally not slot-backed and do not participate in the
compiled static schema.

#### 5. Properties do not get independent value storage

Property definitions belong to the static schema, but their values should still
be computed or routed through the normal Python property getter/setter logic.

So for `kind="property"`:

- `AttrDef` describes the property
- no standalone field-value storage is introduced automatically
- only assignment/runtime flags are tracked when needed

### Why Not Use One Generic `_store` Dict for Everything?

A single generic storage dict would simplify some implementation details, but it
is not recommended as the primary design for this repository.

Reasons:

- it would likely slow down the hot path for ordinary field reads/writes
- it would weaken the value of the current `__slots__`-based layout
- it would force high-frequency host fields and low-frequency dynamic state into
  the same generic storage mechanism

In other words, a generic `_store[name]` model is cleaner in the abstract, but
for Nematics3D it would likely trade too much practical performance for that
uniformity.

### Performance Expectation

With the recommended storage model above, the expected performance profile is:

- instance creation should improve because `deepcopy(__attr_defs__)` disappears
- ordinary field access should stay approximately as fast as today, because
  slot-backed values remain slot-backed
- relation / protection / extra-attr logic should become simpler and more
  targeted because each concern has its own container
- memory usage should be more predictable because long doc strings and validator
  references are no longer duplicated per instance

### Summary

The recommended storage model is therefore:

| Container | Defined in | Stores |
|---|---|---|
| `__attr_defs__` | class level | frozen `AttrDef` schema entries |
| `__slots__` instance attrs | class level | actual field values (`raw_*`, `calc_*`, `opts`, …) |
| `impl_relation_state` | instance level | `RelationState` per declared relation |
| `impl_assign_state` | instance level | `AssignState` / `HostAssignState` per public settable field |
| `impl_extra` | instance level | `ExtraAttrEntry` per dynamically registered extra attr |

This keeps the fast path fast, while still achieving the main architectural
goal of separating schema, runtime flags, relation bindings, and dynamic extra
attribute storage.
