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
<<<<<<< HEAD
``__attr_defs__`` is a class-level dict of frozen ``AttrDef`` instances.  It
is auto-merged from the full method resolution order (MRO) by
``__init_subclass__`` so subclasses never write
``{**Parent.__attr_defs__, ...}`` manually.  The static schema is shared
=======
``__attr_defs__`` is a class-level dict of frozen ``AttrDef`` instances. It is
auto-merged from the full MRO by ``__init_subclass__`` so subclasses never
write ``{**Parent.__attr_defs__, ...}`` manually. The static schema is shared
>>>>>>> fdb4802b4e74ce440c2cd4840102974610d91f0c
across all instances and is never copied at instance creation time.

Per-instance mutable state is split into three purpose-specific containers:

- ``impl_relation_state`` — one ``RelationState`` per declared relation field,
  tracking the live binding (target, weak-ref flag, runtime doc override).
- ``impl_assign_state`` — one ``AssignState`` per public-settable field,
  tracking assignment-control flags.
- ``impl_extra`` — one ``ExtraAttrEntry`` per dynamically registered extra
  attribute (registered at runtime via ``act_add_attr()``).

Normal managed field values (``raw_*``, ``calc_*``, ``opts``, …) remain in
real slot-backed instance attributes on the fast path.
"""

from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass
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
    across all instances of that class. Because the dataclass is frozen they
    cannot be mutated at runtime.

    Parameters
    ----------
    doc:
        Human-readable description of the attribute.
    kind:
        Semantic category — see the naming-convention table in ``ClassBase``.
    validator:
        Optional callable ``(value, doc) -> value`` called on public
        assignment. Not auto-called for ``kind="property"`` — property
        setters must invoke it explicitly.
    is_reapply_opts_after_raw:
        If ``True``, opts are re-applied after this raw field is set.
        Interpreted by ``HostBase``; ignored by ``ClassBase``.
        ClassBase subclasses that do not pair an opts object should leave
        this ``False`` (the default).
    is_public_settable:
        Explicit override for whether the public surface is writable.
        Required for ``kind="property"``; inferred automatically for
        ``raw_``, ``state_``, and ``default_`` fields.
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
    doc_runtime: str | None = None


@dataclass(slots=True)
class AssignState:
    """Mutable per-instance assignment-control flags for one public-settable field.

    ``is_protected`` blocks direct public assignment unconditionally.
    ``is_wrapped`` blocks assignment when this object is controlled by a
    wrapper host — it is only meaningful for ``HostBase`` instances and is
    ignored by plain ``ClassBase`` objects.
    """

    is_protected: bool = False
    is_wrapped: bool = False


@dataclass(slots=True)
class ExtraAttrEntry:
    """Container for one dynamically registered extra attribute.

    Protection state intentionally lives only in
    ``impl_assign_state[name]``. This keeps one authoritative protection state
    for every public-settable field, whether statically declared or dynamic.
    """

    doc: str
    value: object = None
    validator: Callable | None = None


# ---------------------------------------------------------------------------
# Name / kind validation
# ---------------------------------------------------------------------------


_PREFIXED_KINDS: frozenset[str] = frozenset(
    {"raw", "state", "default", "calc", "entity", "impl"}
)

_ATTR_KIND_ORDER: MappingProxyType = MappingProxyType(
    {
        "raw": 0,
        "state": 1,
        "default": 2,
        "relation": 3,
        "calc": 4,
        "entity": 5,
        "property": 6,
    }
)


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
      vice versa. The same rule applies for ``state_``, ``default_``,
      ``calc_``, ``entity_``, and ``impl_``.
    - Kinds not listed in ``_PREFIXED_KINDS`` (for example ``relation`` and
      ``property``) may use any non-prefixed name.
    - Unknown ``kind`` values are rejected outright.
    """
    ctx = f" (class {cls_name!r})" if cls_name else ""

    if kind not in valid_kinds:
        raise ValueError(
            f"AttrDef {name!r}{ctx}: unknown kind {kind!r}. "
            f"Valid kinds: {sorted(valid_kinds)}."
        )

    matched_prefix: str | None = None
    for prefixed_kind in _PREFIXED_KINDS:
        prefix = prefixed_kind + "_"
        if name.startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is not None:
        required_kind = matched_prefix[:-1]
        if kind != required_kind:
            # A calc_-prefixed Python property is still a computed public
            # output; only its storage mechanism differs.
            if matched_prefix == "calc_" and kind == "property":
                return
            raise ValueError(
                f"AttrDef {name!r}{ctx}: name prefix {matched_prefix!r} "
                f"requires kind={required_kind!r}, but got kind={kind!r}."
            )
    elif kind in _PREFIXED_KINDS:
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
    - ``impl_assign_state`` (per instance) tracks assignment-control flags for
      every public-settable field.
    - ``impl_extra`` (per instance) holds dynamically registered extra attrs.

    Naming conventions for subclass ``__attr_defs__`` declarations
    -------------------------------------------------------------
    =========  ==========  ==================================================
    Prefix     kind        Meaning
    =========  ==========  ==================================================
    ``raw_``   ``raw``     Canonical stored public input field. Exposes a
                           readable alias without the prefix.
    ``state_`` ``state``   Writable runtime state input. No shortened alias.
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

    _VALID_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "raw",
            "state",
            "default",
            "calc",
            "entity",
            "impl",
            "relation",
            "property",
        }
    )

    __attr_defs__: ClassVar[MappingProxyType] = MappingProxyType(
        {
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
        }
    )

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

        merged: dict[str, AttrDef] = {}
        for base in reversed(cls.__mro__[1:]):
            merged.update(getattr(base, "__attr_defs__", {}))
        merged.update(vars(cls).get("__attr_defs__", {}))
        cls.__attr_defs__ = MappingProxyType(merged)

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
        attr_defs = type(self).__attr_defs__

        relation_state: dict[str, RelationState] = {
            n: RelationState() for n, d in attr_defs.items() if d.kind == "relation"
        }
        object.__setattr__(self, "impl_relation_state", relation_state)

        assign_state: dict[str, AssignState] = {}
        for n, d in attr_defs.items():
            if self._helper_is_public_settable_from_def(n, d):
                assign_state[n] = self._helper_make_assign_state()
        object.__setattr__(self, "impl_assign_state", assign_state)

        object.__setattr__(self, "impl_extra", {})
        object.__setattr__(self, "impl_is_fixed", bool(is_fixed))

        self._helper_assign_name(
            self._helper_validate_name(name, replace=name_replace),
        )

    def _helper_make_assign_state(self) -> AssignState:
        """Return a fresh ``AssignState`` for one public-settable field."""
        return AssignState()

    @classmethod
    def _helper_is_public_settable_from_def(cls, name: str, defn: AttrDef) -> bool:
        """Return whether one field is public-settable, given its ``AttrDef``."""
        del name
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
    # Attribute classification / resolution helpers
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
        if attr_name in self.impl_extra:
            return False
        defn = type(self).__attr_defs__.get(attr_name)
        return (
            attr_name != "raw_name"
            and defn is not None
            and defn.kind in ("raw", "state")
        )

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
        if attr_name in self.impl_extra:
            return True
        defn = type(self).__attr_defs__.get(attr_name)
        if defn is None:
            return False
        return self._helper_is_public_settable_from_def(attr_name, defn)

    def _helper_resolve_attr_name(
        self,
        name: str,
        *,
        is_allow_impl: bool = False,
    ) -> str:
        """Resolve a public attribute name or raw alias to its canonical name."""
        name = as_str(name, name="Attribute name")
        attr_defs = type(self).__attr_defs__

        if name in self.impl_extra:
            return name

        if name in attr_defs:
            if (not is_allow_impl) and self._helper_is_impl_attr(name):
                raise AttributeError(
                    f"Attribute {name!r} is internal implementation metadata, "
                    "not a readable public attribute."
                )
            return name

        raw_name = f"raw_{name}"
        if raw_name in attr_defs:
            return raw_name

        raise AttributeError(
            f"Readable attribute {name!r} is not registered in "
            f"{type(self).__name__}."
        )

    def _helper_get_attr_doc(self, name: str) -> str:
        """Return the public documentation string for one attribute or alias."""
        canonical_name = self._helper_resolve_attr_name(name)
        if canonical_name in self.impl_extra:
            return self.impl_extra[canonical_name].doc
        if self._helper_is_relation_attr(canonical_name):
            return self._helper_get_relation_doc(canonical_name)
        return type(self).__attr_defs__[canonical_name].doc

    @staticmethod
    def _helper_truncate_repr(value, max_length: int = 240) -> str:
        """Return a bounded one-line repr suitable for inspection output."""
        text = repr(value).replace("\n", " ")
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @classmethod
    def _helper_attr_sort_key(cls, attr_name: str) -> tuple[int, str]:
        """Return the standard display sort key for one declared attribute."""
        defn = cls.__attr_defs__.get(attr_name)
        if defn is None:
            return len(_ATTR_KIND_ORDER), attr_name
        return _ATTR_KIND_ORDER.get(defn.kind, len(_ATTR_KIND_ORDER)), attr_name

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

        for attr_name, defn in type(self).__attr_defs__.items():
            if attr_name == is_exclude_name:
                continue
            if is_exclude_impl and defn.kind == "impl":
                continue
            readable_names.add(attr_name)
            if attr_name.startswith("raw_"):
                readable_names.add(attr_name[4:])

        for attr_name in self.impl_extra:
            if attr_name != is_exclude_name:
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
        existing_names = self._helper_collect_readable_names(
            is_exclude_name=name if is_overwrite else None,
        )
        conflict_names = readable_names & existing_names

        # Extra attrs must also stay clear of methods/properties and other
        # class-level public surfaces that are not represented in __attr_defs__.
        class_conflicts = {candidate for candidate in readable_names if hasattr(type(self), candidate)}
        conflict_names |= class_conflicts

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
        """Register a documented dynamic side-data attribute on this instance.

        Extra attrs deliberately do not participate in the ``raw_`` / ``calc_``
        semantic data model. Their values are instance-local user side data.
        """
        name = as_str(name, name="Extra attribute name")
        if not name.isidentifier():
            raise ValueError(
                f"Invalid attribute name {name!r}: must be a valid Python identifier."
            )

        for prefixed_kind in _PREFIXED_KINDS:
            if name.startswith(prefixed_kind + "_"):
                raise ValueError(
                    f"Extra attribute name {name!r} uses the reserved semantic "
                    f"prefix {prefixed_kind + '_'!r}. Choose an unprefixed side-data name."
                )

        if validator is not None and not callable(validator):
            raise TypeError("validator must be callable or None.")

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

        doc = as_str(doc, name=f"Extra attr doc for {name!r}")
        if validator is not None:
            default = validator(default, doc)

        entry = ExtraAttrEntry(
            doc=doc,
            value=default,
            validator=validator,
        )
        self.impl_extra[name] = entry

        if name not in self.impl_assign_state:
            self.impl_assign_state[name] = self._helper_make_assign_state()

    def act_remove_attr(self, name: str):
        """Remove one dynamic extra attribute and return its previous value."""
        name = as_str(name, name="Extra attribute name")
        if name not in self.impl_extra:
            raise AttributeError(
                f"Cannot remove {name!r}: it is not a dynamic extra attribute of "
                f"{type(self).__name__}."
            )
        value = self.impl_extra.pop(name).value
        self.impl_assign_state.pop(name, None)
        return value

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
        this method. Dynamic creation of undeclared relations is not permitted.
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
    def show_doc(self, is_return=False, logger=None):
        """Show the class docstring for this instance's concrete type."""
        cls = type(self)
        doc = cls.__dict__.get("__doc__")
        if doc is None:
            output = f"{cls.__name__} has no class docstring."
        else:
            output = inspect.cleandoc(doc)

        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_readable_attrs(self, is_return=False, is_desc=True, logger=None):
        """Show the registered readable attributes for this instance."""
        lines = [
            "When reading, the raw_ prefix may be omitted where a public alias exists."
        ]

        attr_names = sorted(
            (
                name
                for name, defn in type(self).__attr_defs__.items()
                if defn.kind != "impl"
            ),
            key=type(self)._helper_attr_sort_key,
        )
        attr_names += sorted(self.impl_extra)

        if not attr_names:
            lines.append("- <none>")
        else:
            for attr_name in attr_names:
                lines.append(f"- {attr_name}")
                if is_desc:
                    lines.append(f"    {self._helper_get_attr_doc(attr_name)}")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_attr_doc(self, name: str, is_return=False, logger=None):
        """Show the documentation for one registered readable attribute."""
        doc = self._helper_get_attr_doc(name)
        logger.info(doc)
        if is_return:
            return doc
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_attr_info(self, name: str, is_return=False, logger=None):
        """Show one attribute's role, mutability, current value, and documentation."""
        requested_name = as_str(name, name="Readable attribute name")
        canonical_name = self._helper_resolve_attr_name(requested_name)
        is_extra = canonical_name in self.impl_extra

        if is_extra:
            kind = "extra"
        else:
            kind = type(self).__attr_defs__[canonical_name].kind

        is_modifiable = self._helper_is_public_settable_attr(canonical_name)
        state = self.impl_assign_state.get(canonical_name)
        is_protected = bool(state is not None and state.is_protected)
        if self.impl_is_fixed and self._helper_is_fixed_blocked_attr(canonical_name):
            is_modifiable = False
        if is_protected:
            is_modifiable = False

        if self._helper_is_relation_attr(canonical_name):
            value = self._helper_resolve_relation_value(canonical_name)
        else:
            value = getattr(self, requested_name)

        lines = [
            f"name: {canonical_name}",
            f"kind: {kind}",
        ]
        if canonical_name.startswith("raw_"):
            lines.append(f"alias: {canonical_name[4:]}")
        lines.extend(
            [
                f"modifiable: {'yes' if is_modifiable else 'no'}",
                f"protected: {'yes' if is_protected else 'no'}",
                f"value: {self._helper_truncate_repr(value)}",
                f"doc: {self._helper_get_attr_doc(requested_name)}",
            ]
        )

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_relations(self, is_return=False, logger=None):
        """Show currently bound relations and their descriptions."""
        lines = []

        for attr_name in self.impl_relation_state:
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

    @logging_and_warning_decorator(start_finish_level=5)
    def show_modifiable_attrs(self, is_return=False, is_desc=True, logger=None):
        """Show public attributes and properties intended for assignment."""
        lines = [
            "When assigning, the raw_ prefix may be omitted where a public alias exists."
        ]

        attr_names = []
        for attr_name, state in self.impl_assign_state.items():
            if state.is_protected:
                continue
            if self.impl_is_fixed and self._helper_is_fixed_blocked_attr(attr_name):
                continue
            attr_names.append(attr_name)

        attr_names = sorted(attr_names, key=type(self)._helper_attr_sort_key)

        if not attr_names:
            lines.append("- <none>")
        else:
            for attr_name in attr_names:
                lines.append(f"- {attr_name}")
                if is_desc:
                    lines.append(f"    {self._helper_get_attr_doc(attr_name)}")

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

        raw_key = f"raw_{key}"
        if raw_key in attr_defs:
            return object.__getattribute__(self, raw_key)

        if key in attr_defs and self._helper_is_relation_attr(key):
            return self._helper_resolve_relation_value(key)

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

        assign_state = self.impl_assign_state.get(target_key)
        if assign_state is not None and assign_state.is_protected:
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{target_key!r} is protected."
            )

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
