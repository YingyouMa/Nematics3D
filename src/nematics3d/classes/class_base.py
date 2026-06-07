"""
Base object model for structured Nematics3D classes.

This module defines ``ClassBase``, the shared foundation for repository objects
that expose:

- a stable readable identity through ``raw_name`` / ``name``
- a frozen class-level attribute schema in ``__attr_defs__``
- per-instance relation binding state in ``impl_relation_state``
- per-instance assignment control state in ``impl_assign_state``
- dynamically registered extra attributes in ``impl_extra``
- semantic one-to-one object relations such as ``owner`` and ``registry``
- inspection helpers for readable, modifiable, and relational surfaces

Design overview
---------------
``__attr_defs__`` is a class-level dict of frozen ``AttrDef`` instances.  It
is auto-merged from the full MRO by ``__init_subclass__`` so subclasses never
write ``{**Parent.__attr_defs__, ...}`` manually.  The static schema is shared
across all instances and is never copied at instance creation time.

Per-instance mutable state is split into three purpose-specific containers:

- ``impl_relation_state`` — one ``RelationState`` per declared relation field,
  tracking the live binding (target, weak-ref flag, runtime doc override).
- ``impl_assign_state`` — one ``AssignState`` per public-settable field,
  tracking the ``is_protected`` flag.
- ``impl_extra`` — one ``ExtraAttrEntry`` per dynamically registered extra
  attribute (registered at runtime via ``act_add_attr()``).

Normal managed field values (``raw_*``, ``calc_*``, ``opts``, …) remain in
real slot-backed instance attributes on the fast path.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, ClassVar

from ..datatypes import as_list, as_str
from ..logging_decorator import logging_and_warning_decorator


# ---------------------------------------------------------------------------
# Static schema dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AttrDef:
    """Frozen static schema entry for one managed attribute.

    Instances live in ``__attr_defs__`` at the class level and are shared
    across all instances of that class.  Because the dataclass is frozen they
    cannot be mutated at runtime.

    Parameters
    ----------
    doc:
        Human-readable description of the attribute.
    kind:
        Semantic category — see the naming-convention table in ``ClassBase``.
    validator:
        Optional callable ``(value, doc) -> value`` called on public
        assignment.  Not auto-called for ``kind="property"`` — property
        setters must invoke it explicitly.
    is_reapply_opts_after_raw:
        If ``True``, opts are re-applied after this raw field is set.
        Interpreted by ``HostBase``; ignored by ``ClassBase``.
        ClassBase subclasses that do not pair an opts object should leave
        this ``False`` (the default).
    is_public_settable:
        Explicit override for whether the public surface is writable.
        Required for ``kind="property"``; inferred automatically for
        ``raw_``, ``state_``, ``default_``, and extra attrs.
    is_weak_by_default:
        For ``kind="relation"`` only — whether the initial binding is a
        weak reference when ``is_weak`` is not supplied explicitly.
    """

    doc: str
    kind: str
    validator: Callable | None = None
    is_reapply_opts_after_raw: bool = False
    is_public_settable: bool | None = None
    is_weak_by_default: bool = True


# ---------------------------------------------------------------------------
# Per-instance state dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RelationState:
    """Mutable per-instance binding state for one declared relation."""

    is_weak: bool | None = None
    relation_value: object = None
    doc_runtime: str | None = None  # runtime override for AttrDef.doc


@dataclass(slots=True)
class AssignState:
    """Mutable per-instance assignment-control flags for one public-settable field.

    ``is_protected`` blocks direct public assignment unconditionally.
    ``is_wrapped`` blocks assignment when this object is controlled by a
    wrapper host — it is only meaningful for ``HostBase`` instances and is
    ignored by plain ``ClassBase`` objects.
    """

    is_protected: bool = False
    is_wrapped: bool = False  # HostBase only; ignored by ClassBase


@dataclass(slots=True)
class ExtraAttrEntry:
    """Container for one dynamically registered extra attribute.

    Protection state (``is_protected``) is intentionally absent here.  It
    lives exclusively in ``impl_assign_state[name]``, which is the single
    authoritative source for protection across *all* public-settable fields —
    both statically declared and dynamically registered.  Keeping it here too
    would create a second, diverging copy that could silently go out of sync.
    """

    doc: str
    value: object = None
    validator: Callable | None = None


# ---------------------------------------------------------------------------
# Name / kind validation
# ---------------------------------------------------------------------------

# Prefixed kinds: every kind that enforces a name prefix.
# The prefix is always kind + "_", so the kind can be recovered by stripping
# the trailing underscore — no explicit mapping needed.
# Any kind NOT in this set is "free" (no prefix required), so subclasses can
# introduce new free kinds (e.g. "opts") via _VALID_KINDS without touching
# this constant.
_PREFIXED_KINDS: frozenset[str] = frozenset({"raw", "state", "default", "calc", "entity", "impl"})


def _validate_name_kind(
    name: str,
    kind: str,
    valid_kinds: frozenset[str],
    *,
    cls_name: str = "",
) -> None:
    """Raise ``ValueError`` if *name* and *kind* are inconsistent.

    Checks are bidirectional:

    - A field whose name starts with ``raw_`` must have ``kind="raw"``, and
      vice versa.  The same rule applies for ``state_``, ``default_``,
      ``calc_``, ``entity_``, and ``impl_``.
    - Fields with ``kind`` in ``_FREE_KINDS`` (``"relation"``, ``"property"``)
      may use any non-prefixed name.
    - Unknown ``kind`` values are rejected outright.
    """
    ctx = f" (class {cls_name!r})" if cls_name else ""

    if kind not in valid_kinds:
        raise ValueError(
            f"AttrDef {name!r}{ctx}: unknown kind {kind!r}. "
            f"Valid kinds: {sorted(valid_kinds)}."
        )

    # Determine which prefix (if any) this name carries.
    # Prefix is always kind + "_", so stripping "_" recovers the kind directly.
    matched_prefix: str | None = None
    for prefixed_kind in _PREFIXED_KINDS:
        prefix = prefixed_kind + "_"
        if name.startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is not None:
        required_kind = matched_prefix[:-1]  # strip trailing "_"
        if kind != required_kind:
            # calc_ fields may be explicitly declared as kind="property" to
            # indicate a read-only computed value backed by a Python property
            # rather than a stored slot.  The calc_ prefix still signals
            # "computed output" to readers; only the implementation differs.
            if matched_prefix == "calc_" and kind == "property":
                pass  # allowed explicitly
            else:
                raise ValueError(
                    f"AttrDef {name!r}{ctx}: name prefix {matched_prefix!r} "
                    f"requires kind={required_kind!r}, but got kind={kind!r}."
                )
    else:
        # No prefix — only reject if the kind is one that *requires* a prefix.
        if kind in _PREFIXED_KINDS:
            raise ValueError(
                f"AttrDef {name!r}{ctx}: kind={kind!r} requires the name "
                f"prefix '{kind}_', but the name has no such prefix."
            )


# ---------------------------------------------------------------------------
# ClassBase
# ---------------------------------------------------------------------------

class ClassBase:
    """
    Minimal structured base class for Nematics3D domain objects.

    ``ClassBase`` provides a lightweight object protocol centred around a small
    set of core ideas:

    - ``raw_name`` stores the underlying object identity.
    - ``name`` is the public readable alias of ``raw_name``.
    - ``__attr_defs__`` (class level, frozen) holds the static attribute
      schema, auto-merged from the full MRO.
    - ``impl_relation_state`` (per instance) tracks the live binding of every
      declared relation.
    - ``impl_assign_state`` (per instance) tracks protection flags for every
      public-settable field.
    - ``impl_extra`` (per instance) holds dynamically registered extra attrs.

    Naming conventions for subclass ``__attr_defs__`` declarations
    -------------------------------------------------------------
    =========  ==========  ==================================================
    Prefix     kind        Meaning
    =========  ==========  ==================================================
    ``raw_``   ``raw``     Canonical stored public input field.  Exposes a
                           readable alias without the prefix.
    ``state_`` ``state``   Writable runtime state input.  No shortened alias.
    ``default_``  ``default`` Optional managed default-layer input.
    ``calc_``  ``calc``    Computed readable output (read-only).
    ``entity_`` ``entity`` Computed object output (read-only).
    ``impl_``  ``impl``    Internal implementation field (not user-facing).
    (none)     ``relation`` One-to-one object link (e.g. ``owner``).
    (none)     ``property`` Python ``@property`` backed managed attribute.
    =========  ==========  ==================================================

    Only public assignment surfaces (``raw_``, ``state_``, ``default_``,
    writable properties, and extra attrs) get an ``AssignState`` entry.
    Read-only outputs and ``impl_`` fields do not.
    """

    # Subclasses may extend this frozenset to introduce new field categories.
    _VALID_KINDS: ClassVar[frozenset[str]] = frozenset({
        "raw", "state", "default",
        "calc", "entity",
        "impl",
        "relation", "property",
    })

    __attr_defs__: ClassVar[MappingProxyType] = MappingProxyType({
        "raw_name": AttrDef(
            doc="The underlying string identifier for this instance.",
            kind="raw",
            validator=as_str,
        ),
        "owner": AttrDef(
            doc="The object that owns this instance.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "registry": AttrDef(
            doc="The Registry object where this instance is registered.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "impl_is_fixed": AttrDef(
            doc=(
                "Whether the core raw/state data of this instance is frozen "
                "after initialization."
            ),
            kind="impl",
        ),
        "impl_relation_state": AttrDef(
            doc="Per-instance relation binding state dict.",
            kind="impl",
        ),
        "impl_assign_state": AttrDef(
            doc="Per-instance assignment control state dict.",
            kind="impl",
        ),
        "impl_extra": AttrDef(
            doc="Dynamically registered extra attributes dict.",
            kind="impl",
        ),
    })

    __slots__ = (
        "raw_name",
        "impl_is_fixed",
        "impl_relation_state",
        "impl_assign_state",
        "impl_extra",
        "__weakref__",
    )

    # ------------------------------------------------------------------
    # Class creation hook — MRO merge + validation
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. Auto-merge __attr_defs__ from the full MRO (most-base first).
        merged: dict[str, AttrDef] = {}
        for base in reversed(cls.__mro__[1:]):
            merged.update(getattr(base, "__attr_defs__", {}))
        # The class's own definitions override ancestors.
        merged.update(vars(cls).get("__attr_defs__", {}))
        # Wrap in MappingProxyType so the table is read-only at runtime.
        # AttrDef entries are already frozen; this seals the outer mapping too,
        # preventing runtime insertion or deletion of schema entries.
        cls.__attr_defs__ = MappingProxyType(merged)

        # 2. Validate name–kind consistency for every declared field.
        for name, defn in cls.__attr_defs__.items():
            _validate_name_kind(
                name, defn.kind, cls._VALID_KINDS, cls_name=cls.__name__
            )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        name: str | None,
        name_replace: str,
        is_fixed: bool = False,
    ):
        cls = type(self)
        attr_defs = cls.__attr_defs__

        # --- impl_relation_state: one entry per declared relation ---
        relation_state: dict[str, RelationState] = {
            n: RelationState()
            for n, d in attr_defs.items()
            if d.kind == "relation"
        }
        object.__setattr__(self, "impl_relation_state", relation_state)

        # --- impl_assign_state: one entry per public-settable field ---
        assign_state: dict[str, AssignState] = {}
        for n, d in attr_defs.items():
            if self._helper_is_public_settable_from_def(n, d):
                assign_state[n] = self._helper_make_assign_state()
        object.__setattr__(self, "impl_assign_state", assign_state)

        # --- impl_extra: empty; populated by act_add_attr() at runtime ---
        object.__setattr__(self, "impl_extra", {})

        # --- impl_is_fixed ---
        object.__setattr__(self, "impl_is_fixed", bool(is_fixed))

        # Normalize the initial name and route through registry-aware path.
        self._helper_assign_name(
            self._helper_validate_name(name, replace=name_replace),
        )

    def _helper_make_assign_state(self) -> AssignState:
        """Return a fresh ``AssignState`` for one public-settable field."""
        return AssignState()

    @classmethod
    def _helper_is_public_settable_from_def(cls, name: str, defn: AttrDef) -> bool:
        """Return whether one field is public-settable, given its ``AttrDef``."""
        if defn.kind in ("raw", "state", "default"):
            return True
        if defn.kind == "property":
            return bool(defn.is_public_settable)
        return False

    # ------------------------------------------------------------------
    # Name handling
    # ------------------------------------------------------------------

    def _helper_get_name_validator(self, defn: AttrDef):
        """Return the name validator, defaulting to ``as_str`` when omitted."""
        return defn.validator if defn.validator is not None else as_str

    def _helper_validate_name(self, name, *, replace=None):
        """Validate one name value through the registered raw_name validator."""
        if name is None:
            if replace is not None:
                return replace
            raise NameError(
                "`name` and `replace` are both None. A valid str name is needed."
            )
        defn = type(self).__attr_defs__["raw_name"]
        validator = self._helper_get_name_validator(defn)
        return validator(name, name=defn.doc, replace=replace)

    def act_set_name(self, name):
        """Validate and assign one public name for this instance."""
        name = self._helper_validate_name(name)
        return self._helper_assign_name(name)

    def _helper_assign_name(self, name):
        """Store one normalised name after registry-level uniqueness checks."""
        check_name = getattr(
            getattr(self, "registry", None), "_helper_check_name", None
        )
        if callable(check_name):
            name = check_name(name)
        object.__setattr__(self, "raw_name", name)
        return name

    # ------------------------------------------------------------------
    # Attribute classification helpers
    # ------------------------------------------------------------------

    def _helper_is_impl_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute belongs to impl_* storage."""
        defn = type(self).__attr_defs__.get(attr_name)
        if defn is not None:
            return defn.kind == "impl"
        return attr_name.startswith("impl_")

    def _helper_is_relation_attr(self, attr_name: str) -> bool:
        """Return whether one declared attribute has ``kind="relation"``."""
        defn = type(self).__attr_defs__.get(attr_name)
        return defn is not None and defn.kind == "relation"

    def _helper_is_property_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute is backed by a Python property."""
        return isinstance(getattr(type(self), attr_name, None), property)

    def _helper_is_extra_attr(self, attr_name: str) -> bool:
        """Return whether one attribute was dynamically registered as an extra attr."""
        return attr_name in self.impl_extra

    def _helper_is_fixed_blocked_attr(self, attr_name: str) -> bool:
        """Return whether one attr belongs to the fixed raw/state core surface."""
        return (attr_name != "raw_name") and attr_name.startswith(("raw_", "state_"))

    def _helper_raise_fixed_assignment_error(self, target_key: str) -> None:
        """Raise the standard assignment error for fixed raw/state attrs."""
        cls_name = type(self).__name__
        obj_name = getattr(self, "raw_name", "Uninitialized")
        raise AttributeError(
            f"[{cls_name}: {obj_name!r}] Assignment blocked: "
            f"{target_key!r} belongs to the fixed core data of this object. "
            "This class is designed so that its core raw/state data should not "
            "be modified after initialization, because too many dependent "
            "results may need synchronised updates. Please create a new "
            "instance instead."
        )

    def _helper_is_public_settable_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute is writable from the public surface."""
        # Extra attrs are always public-settable.
        if attr_name in self.impl_extra:
            return True
        defn = type(self).__attr_defs__.get(attr_name)
        if defn is None:
            return False
        return self._helper_is_public_settable_from_def(attr_name, defn)

    # ------------------------------------------------------------------
    # Extra attribute registration
    # ------------------------------------------------------------------

    def _helper_collect_readable_names(
        self,
        *,
        is_exclude_name: str | None = None,
        is_exclude_impl: bool = False,
    ) -> set[str]:
        """Collect the currently occupied readable attribute surface names."""
        readable_names: set[str] = set()

        # Declared attributes from the class schema.
        for attr_name, defn in type(self).__attr_defs__.items():
            if attr_name == is_exclude_name:
                continue
            if is_exclude_impl and defn.kind == "impl":
                continue
            readable_names.add(attr_name)
            if attr_name.startswith("raw_"):
                readable_names.add(attr_name[4:])

        # Dynamically registered extra attrs.
        for attr_name in self.impl_extra:
            if attr_name == is_exclude_name:
                continue
            readable_names.add(attr_name)

        return readable_names

    def _helper_check_readable_name_conflict(
        self,
        name: str,
        *,
        is_overwrite: bool,
    ) -> None:
        """Reject new registrations whose readable names collide with existing ones."""
        readable_names = {name}
        if name.startswith("raw_"):
            readable_names.add(name[4:])

        existing_names = self._helper_collect_readable_names(
            is_exclude_name=name if is_overwrite else None,
        )
        conflict_names = readable_names & existing_names
        if conflict_names:
            raise AttributeError(
                "Cannot register readable name(s) "
                f"{sorted(conflict_names)!r}: they conflict with an existing "
                f"readable surface of {type(self).__name__}."
            )

    def act_add_attr(
        self,
        name: str,
        doc: str,
        default=None,
        validator=None,
        is_overwrite: bool = False,
    ):
        """Register a dynamic extra attribute with documentation and a default value.

        Extra attributes are stored in ``impl_extra`` and are intentionally
        kept out of the static ``__attr_defs__`` schema.
        """
        name = as_str(name, name="Extra attribute name")
        if not name.isidentifier():
            raise ValueError(
                f"Invalid attribute name {name!r}: must be a valid Python identifier."
            )

        # Guard against clobbering a statically declared field.
        if name in type(self).__attr_defs__ and name not in self.impl_extra:
            raise AttributeError(
                f"Cannot register extra attribute {name!r}: it is already a "
                f"statically declared field of {type(self).__name__}."
            )

        if name in self.impl_extra and not is_overwrite:
            raise KeyError(
                f"Extra attribute {name!r} is already registered in "
                f"{type(self).__name__}.impl_extra."
            )

        self._helper_check_readable_name_conflict(name, is_overwrite=is_overwrite)

        entry = ExtraAttrEntry(
            doc=as_str(doc, name=f"Extra attr doc for {name!r}"),
            value=default,
            validator=validator,
        )
        self.impl_extra[name] = entry

        # Ensure an assign-state entry exists for protection tracking.
        if name not in self.impl_assign_state:
            self.impl_assign_state[name] = self._helper_make_assign_state()

    # ------------------------------------------------------------------
    # Protection
    # ------------------------------------------------------------------

    def _helper_set_protected_attr(self, attrs, is_protected: bool):
        """Set the protected flag for one or more registered attributes."""
        for attr_name in as_list(attrs, name="attrs"):
            target_key = attr_name
            if target_key not in self.impl_assign_state:
                raw_key = f"raw_{attr_name}"
                if raw_key in self.impl_assign_state:
                    target_key = raw_key
                else:
                    raise AttributeError(
                        f"Cannot update protection for {attr_name!r}: "
                        "it is not a public-settable attribute of "
                        f"{type(self).__name__}."
                    )
            self.impl_assign_state[target_key].is_protected = is_protected

    def act_register_protected_attr(self, attrs):
        """Mark registered attributes as protected from public assignment."""
        self._helper_set_protected_attr(attrs, True)

    def act_unregister_protected_attr(self, attrs):
        """Remove the protected flag from one or more registered attributes."""
        self._helper_set_protected_attr(attrs, False)

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def _helper_resolve_relation_value(self, name: str):
        """Return the current relation target, resolving weak references when needed."""
        rel = self.impl_relation_state[name]
        if isinstance(rel.relation_value, weakref.ReferenceType):
            return rel.relation_value()
        return rel.relation_value

    def _helper_get_relation_doc(self, name: str) -> str:
        """Return the runtime doc override for a relation, or its declared doc."""
        rel = self.impl_relation_state[name]
        if rel.doc_runtime is not None:
            return rel.doc_runtime
        return type(self).__attr_defs__[name].doc

    def _helper_relation_tree_node_label(self):
        """Return the display label used for this node in relation trees."""
        return str(self)

    def _helper_relation_tree_walk(
        self,
        *,
        depth: int,
        is_include_none: bool,
        _prefix: str = "",
        _visited: set[int] | None = None,
    ) -> list[str]:
        """Walk the current relation graph and return a tree-formatted line list."""
        if _visited is None:
            _visited = set()

        node_id = id(self)
        lines = [f"{_prefix}{self._helper_relation_tree_node_label()}"]
        if node_id in _visited:
            lines[-1] += " [visited]"
            return lines
        _visited.add(node_id)

        if depth <= 0:
            return lines

        entries = []
        for attr_name in self.impl_relation_state:
            target = self._helper_resolve_relation_value(attr_name)
            if target is None and (not is_include_none):
                continue
            entries.append((attr_name, target))

        if not entries:
            lines.append(f"{_prefix}  <none>")
            return lines

        last_index = len(entries) - 1
        for index, (attr_name, target) in enumerate(entries):
            is_last = index == last_index
            branch = "`- " if is_last else "|- "
            child_prefix = _prefix + ("   " if is_last else "|  ")

            if target is None:
                lines.append(f"{_prefix}{branch}{attr_name}: <none>")
                continue

            lines.append(f"{_prefix}{branch}{attr_name}:")
            walk = getattr(target, "_helper_relation_tree_walk", None)
            if callable(walk):
                lines.extend(
                    walk(
                        depth=depth - 1,
                        is_include_none=is_include_none,
                        _prefix=child_prefix,
                        _visited=_visited,
                    )
                )
            else:
                lines.append(f"{child_prefix}{target}")

        return lines

    def act_bind_relation_base(
        self,
        name: str,
        target,
        *,
        doc: str | None = None,
        is_weak: bool | None = None,
        is_replace: bool = True,
    ):
        """Bind or update a named relation on this instance.

        The relation must be declared in ``__attr_defs__`` before calling
        this method.  Dynamic creation of undeclared relations is not
        permitted.
        """
        name = as_str(name, name=f"Relation name for instance {self.raw_name!r}")

        if name not in type(self).__attr_defs__:
            raise AttributeError(
                f"Cannot bind undeclared relation {name!r}. "
                f"Add it to {type(self).__name__}.__attr_defs__ first."
            )
        if not self._helper_is_relation_attr(name):
            raise AttributeError(
                f"Cannot bind relation {name!r}: it is not declared as a relation "
                f"in {type(self).__name__}.__attr_defs__."
            )

        rel = self.impl_relation_state[name]
        defn = type(self).__attr_defs__[name]

        if doc is not None:
            rel.doc_runtime = as_str(
                doc, name=f"Relation doc for instance {self.raw_name!r}"
            )

        old_target = self._helper_resolve_relation_value(name)
        if old_target is not None and old_target is not target and (not is_replace):
            raise RuntimeError(
                f"Relation {name!r} of {type(self).__name__} is already bound."
            )

        if is_weak is None:
            is_weak = bool(defn.is_weak_by_default)

        rel.is_weak = bool(is_weak)
        rel.relation_value = (
            weakref.ref(target) if (is_weak and target is not None) else target
        )
        return target

    def act_unbind_relation_base(self, name: str):
        """Clear the current target of a named relation."""
        name = as_str(name, name=f"Relation name for instance {self.raw_name!r}")
        if name not in self.impl_relation_state:
            raise AttributeError(
                f"Cannot unbind relation {name!r}: it is not declared as a "
                f"relation in {type(self).__name__}.__attr_defs__."
            )
        rel = self.impl_relation_state[name]
        rel.relation_value = None
        rel.is_weak = None

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @logging_and_warning_decorator(start_finish_level=5)
    def show_readable_attrs(self, is_return=False, is_desc=True, logger=None):
        """Show the registered readable attributes for this instance."""
        lines = [
            "When reading, the raw_ prefix may be omitted where a public alias exists."
        ]

        def _sort_key(attr_name: str) -> tuple[int, str]:
            defn = type(self).__attr_defs__.get(attr_name)
            if defn is None:  # extra attr
                return 6, attr_name
            if defn.kind == "raw":
                return 0, attr_name
            if defn.kind == "state":
                return 1, attr_name
            if defn.kind == "default":
                return 2, attr_name
            if defn.kind == "relation":
                return 3, attr_name
            if defn.kind == "calc":
                return 4, attr_name
            if defn.kind == "property":
                return 5, attr_name
            return 6, attr_name

        attr_names = sorted(
            (
                name
                for name, defn in type(self).__attr_defs__.items()
                if defn.kind != "impl"
            ),
            key=_sort_key,
        )
        # Append extra attrs at the end.
        attr_names += sorted(self.impl_extra)

        if not attr_names:
            lines.append("- <none>")
        else:
            for attr_name in attr_names:
                if self._helper_is_relation_attr(attr_name):
                    desc = self._helper_get_relation_doc(attr_name)
                elif attr_name in self.impl_extra:
                    desc = self.impl_extra[attr_name].doc
                else:
                    desc = type(self).__attr_defs__[attr_name].doc
                lines.append(f"- {attr_name}")
                if is_desc:
                    lines.append(f"    {desc}")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_attr_doc(self, name: str, is_return=False, logger=None):
        """Show the description for one registered readable attribute."""
        name = as_str(name, name="Readable attribute name")
        attr_defs = type(self).__attr_defs__

        if name not in attr_defs:
            raw_name = f"raw_{name}"
            if raw_name in attr_defs:
                name = raw_name
            elif name not in self.impl_extra:
                raise AttributeError(
                    f"Readable attribute {name!r} is not registered in "
                    f"{type(self).__name__}."
                )

        if name in self.impl_extra:
            doc = self.impl_extra[name].doc
        elif self._helper_is_impl_attr(name):
            raise AttributeError(
                f"Attribute {name!r} is internal implementation metadata, "
                "not a readable public attribute."
            )
        elif self._helper_is_relation_attr(name):
            doc = self._helper_get_relation_doc(name)
        else:
            doc = attr_defs[name].doc

        logger.info(doc)
        if is_return:
            return doc
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_relations(self, is_return=False, logger=None):
        """Show currently bound relations and their descriptions."""
        lines = []

        for attr_name, rel in self.impl_relation_state.items():
            target = self._helper_resolve_relation_value(attr_name)
            if target is None:
                continue
            lines.append(f"- {attr_name}")
            lines.append(f"      {self._helper_get_relation_doc(attr_name)}")
            lines.append(f"      current: {target}")

        if not lines:
            lines.append("- <none>")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_relation_tree(
        self,
        depth: int = 2,
        is_return=False,
        is_include_none: bool = False,
        logger=None,
    ):
        """Show the current relation graph as a tree."""
        depth = int(depth)
        if depth < 0:
            raise ValueError("depth must be >= 0.")

        output = "\n".join(
            self._helper_relation_tree_walk(
                depth=depth,
                is_include_none=is_include_none,
            )
        )
        logger.info(output)
        if is_return:
            return output
        return None

    def show_attr_desc(self, attr_name: str) -> str:
        """Return the description of a registered attribute or its public alias."""
        attr_defs = type(self).__attr_defs__

        if attr_name in attr_defs:
            if self._helper_is_relation_attr(attr_name):
                return f"{attr_name!r}: {self._helper_get_relation_doc(attr_name)}"
            return f"{attr_name!r}: {attr_defs[attr_name].doc}"

        if attr_name in self.impl_extra:
            return f"{attr_name!r}: {self.impl_extra[attr_name].doc}"

        raw_attr_name = f"raw_{attr_name}"
        if raw_attr_name in attr_defs:
            return (
                f"{attr_name!r}: Alias of {raw_attr_name!r}. "
                f"{attr_defs[raw_attr_name].doc}"
            )

        raise KeyError(
            f"Attribute {attr_name!r} was not found in "
            f"{type(self).__name__}.__attr_defs__."
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def show_modifiable_attrs(self, is_return=False, is_desc=True, logger=None):
        """Show public attributes and properties intended for assignment."""
        lines = [
            "When assigning, the raw_ prefix may be omitted where a public alias exists."
        ]

        def _sort_key(attr_name: str) -> tuple[int, str]:
            defn = type(self).__attr_defs__.get(attr_name)
            if defn is None:  # extra attr
                return 4, attr_name
            if defn.kind == "raw":
                return 0, attr_name
            if defn.kind == "state":
                return 1, attr_name
            if defn.kind == "default":
                return 2, attr_name
            if defn.kind == "property":
                return 3, attr_name
            return 4, attr_name

        attr_names = []
        # Static declared fields.
        for attr_name, state in self.impl_assign_state.items():
            if attr_name in self.impl_extra:
                continue  # handled below
            if state.is_protected:
                continue
            if self.impl_is_fixed and self._helper_is_fixed_blocked_attr(attr_name):
                continue
            attr_names.append(attr_name)
        # Extra attrs — protection is always read from impl_assign_state,
        # never from ExtraAttrEntry (which carries no is_protected field).
        for attr_name in self.impl_extra:
            state = self.impl_assign_state.get(attr_name)
            if state is not None and state.is_protected:
                continue
            attr_names.append(attr_name)

        attr_names = sorted(attr_names, key=_sort_key)

        if not attr_names:
            lines.append("- <none>")
        else:
            for attr_name in attr_names:
                if attr_name in self.impl_extra:
                    desc = self.impl_extra[attr_name].doc
                else:
                    desc = type(self).__attr_defs__[attr_name].doc
                lines.append(f"- {attr_name}")
                if is_desc:
                    lines.append(f"    {desc}")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    # ------------------------------------------------------------------
    # Attribute access / assignment
    # ------------------------------------------------------------------

    def __getattr__(self, key):
        attr_defs = type(self).__attr_defs__

        # Public alias: ``name`` → ``raw_name``.
        raw_key = f"raw_{key}"
        if raw_key in attr_defs:
            return object.__getattribute__(self, raw_key)

        if key in attr_defs:
            if self._helper_is_relation_attr(key):
                return self._helper_resolve_relation_value(key)

        # Extra attr.
        impl_extra = object.__getattribute__(self, "impl_extra")
        if key in impl_extra:
            return impl_extra[key].value

        cls_name = type(self).__name__
        try:
            obj_name = object.__getattribute__(self, "raw_name")
        except AttributeError:
            obj_name = "Uninitialized"
        raise AttributeError(f"[{cls_name}: {obj_name!r}] has no attribute {key!r}.")

    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)

    def _helper_setattr_basic(self, key, value):
        """Resolve a public assignment target and apply validation/protection rules."""
        attr_defs = type(self).__attr_defs__
        target_key = key

        # Resolve public alias (e.g. ``obj.name = ...`` → ``raw_name``).
        if target_key not in attr_defs and target_key not in self.impl_extra:
            raw_key = f"raw_{key}"
            if raw_key in attr_defs:
                target_key = raw_key
            else:
                cls_name = type(self).__name__
                obj_name = getattr(self, "raw_name", "Uninitialized")
                raise AttributeError(
                    f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                    f"{key!r} is not a valid or registered attribute."
                )

        if not self._helper_is_public_settable_attr(target_key):
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{key!r} resolves to internal attribute {target_key!r}, "
                "which cannot be assigned through the public setattr path."
            )

        if self.impl_is_fixed and self._helper_is_fixed_blocked_attr(target_key):
            self._helper_raise_fixed_assignment_error(target_key)

        # Protection check.
        assign_state = self.impl_assign_state.get(target_key)
        if assign_state is not None and assign_state.is_protected:
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{target_key!r} is protected."
            )

        # Validation.
        if target_key in self.impl_extra:
            entry = self.impl_extra[target_key]
            if entry.validator is not None:
                value = entry.validator(value, entry.doc)
        elif target_key != "raw_name":
            defn = attr_defs.get(target_key)
            if defn is not None and defn.validator is not None:
                value = defn.validator(value, defn.doc)

        self._helper_setattr_final(key, value, target_key=target_key)

    def _helper_setattr_final(self, key, value, *, target_key=None):
        """Apply one validated public assignment to final storage."""
        target_key = key if target_key is None else target_key
        if target_key == "raw_name":
            self.act_set_name(value)
            return
        if target_key in self.impl_extra:
            self.impl_extra[target_key].value = value
            return
        object.__setattr__(self, target_key, value)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}({self.name!r})"
