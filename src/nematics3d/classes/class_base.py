"""
Base object model for structured Nematics3D classes.

This module defines ``ClassBase``, the shared foundation for repository objects
that expose:

- a stable readable identity through ``raw_name`` / ``name``
- per-instance attribute metadata in ``impl_attrs``
- semantic one-to-one object relations such as ``owner`` and ``registry``
- inspection helpers for readable, modifiable, and relational surfaces

In the current implementation, class-level ``__attr_defs__`` acts as a template.
Each instance copies that template into ``impl_attrs`` during initialization,
then keeps all later definition and runtime updates in that single per-instance
dictionary.
"""

from copy import deepcopy
import weakref

from ..datatypes import as_list, as_str
from ..logging_decorator import logging_and_warning_decorator


# ClassBase declaration conventions for subclasses:
# - every managed field must be declared in `__attr_defs__`; do not rely on
#   ad hoc instance attributes for normal public fields, relations, or
#   properties.
# - `raw_` fields are the canonical stored public input fields. A `raw_xxx`
#   field automatically exposes the readable public alias `xxx`.
# - `state_` fields are writable runtime state inputs that are part of the
#   managed attribute system and may affect later computation. They do not
#   create a shortened public alias.
# - `default_` fields are optional managed default-layer inputs for subclasses
#   that need them.
# - `calc_` fields are computed readable outputs.
# - `entity_` fields are computed object outputs such as created runtime
#   entities, actors, meshes, or other attached result objects.
# - `impl_` fields are internal implementation metadata or runtime containers
#   and should not be treated as a user-facing readable surface.
# - relation names use their direct public names, such as `owner` or
#   `registry`.
# - relations in the current ClassBase protocol are one-to-one links only;
#   do not use relations to represent one-to-many or collection-style data.
# - property metadata should also be registered in `__attr_defs__`, while the
#   actual getter/setter behavior remains a normal Python `@property` on the
#   class.
# - only public assignment surfaces need assignment-related flags in
#   `__attr_defs__`: `raw_...`, `state_...`, writable properties, and extra
#   attrs. For properties and extra attrs, register `is_public_settable`
#   explicitly to declare whether the public surface is writable.
# - read-only outputs such as `calc_...` / `entity_...`, internal storage such
#   as `impl_...`, and non-public relations should not register no-op
#   `validator` or `is_protected` entries in the static schema.
# - writable property validators are not auto-called by `ClassBase`; if a
#   subclass registers one, its property setter should call that validator
#   explicitly.
# - extra attrs are runtime-registered public fields and should still enter the
#   managed schema through the provided registration helpers. Their current
#   runtime values also live in the same ``impl_attrs`` entry via ``value``.
# - for semantic clarity, do not introduce other non-underscore public field
#   categories beyond these conventions: `raw_`, `state_`, `default_`, `calc_`,
#   `entity_`, `impl_`, direct-named relations, direct-named properties, and
#   extra attrs.
# - when registering new fields into `impl_attrs`, choose names that respect
#   these categories and do not collide with an existing readable surface.
class ClassBase:
    """
    Minimal structured base class for Nematics3D domain objects.

    ``ClassBase`` provides a lightweight object protocol centered around a small
    set of core ideas:

    - ``raw_name`` stores the underlying object identity
    - ``name`` remains the public readable alias of ``raw_name``
    - ``impl_attrs`` stores the live attribute metadata for this instance

    Each entry in ``impl_attrs`` combines relatively stable definition data
    such as ``doc`` and ``validator`` with runtime state such as protection
    flags, current relation bindings, or current extra-attribute values.

    This class supports ordinary raw/public attribute access, runtime protection
    of registered attributes, dynamic registration of extra attributes, semantic
    relation binding for object links such as ``owner`` and ``registry``, and
    inspection helpers such as ``show_attr_desc()``, ``show_readable_attrs()``,
    ``show_modifiable_attrs()``, ``show_relations()``, and
    ``show_relation_tree()``.
    """

    __attr_defs__ = {
        "raw_name": {
            "doc": "The underlying string identifier for this instance.",
            "validator": as_str,
        },
        "owner": {
            "doc": "The object that owns this instance.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "registry": {
            "doc": "The Registry object where this instance is registered.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "impl_attrs": {
            "doc": "Runtime attribute metadata copied from the class template.",
        },
        "impl_is_fixed": {
            "doc": (
                "Whether the core raw/state data of this instance is frozen "
                "after initialization."
            ),
        },
    }

    __slots__ = ("raw_name", "impl_attrs", "impl_is_fixed", "__weakref__")

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
        # Each instance starts from a private copy of the class-level
        # attribute template.
        impl_attrs = deepcopy(type(self).__attr_defs__)
        object.__setattr__(self, "impl_attrs", impl_attrs)
        object.__setattr__(self, "impl_is_fixed", bool(is_fixed))

        # Normalize the initial name once, then route it through the shared
        # registry-aware assignment path.
        self._helper_assign_name(
            self._helper_validate_name(name, replace=name_replace),
        )

    # ------------------------------------------------------------------
    # Name handling
    # ------------------------------------------------------------------

    def _helper_get_name_validator(self, attr_info):
        """Return the name validator, defaulting to as_str when omitted."""
        validator = attr_info.get("validator")
        if validator is None:
            return as_str
        return validator

    def _helper_validate_name(self, name, *, replace=None):
        """Validate one name value through the registered raw_name validator."""
        if name is None:
            if replace is not None:
                return replace
            else:
                raise NameError(
                    "`name` and `replace` are both None." "A valid str name is needed"
                )
        attr_info = self.impl_attrs["raw_name"]
        validator = self._helper_get_name_validator(attr_info)
        name = validator(name, name=attr_info["doc"], replace=replace)
        return name

    def act_set_name(self, name):
        """Validate and assign one public name for this instance."""
        name = self._helper_validate_name(name)
        return self._helper_assign_name(name)

    def _helper_assign_name(self, name):
        """Store one normalized name after registry-level uniqueness checks."""
        check_name = getattr(
            getattr(self, "registry", None), "_helper_check_name", None
        )
        if callable(check_name):
            name = check_name(name)

        object.__setattr__(self, "raw_name", name)
        return name

    # ------------------------------------------------------------------
    # Attribute classification
    # ------------------------------------------------------------------

    def _helper_is_impl_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute belongs to impl_* storage."""
        return attr_name.startswith("impl_")

    def _helper_is_relation_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute is a relation."""
        return self.impl_attrs[attr_name].get("kind") == "relation"

    def _helper_is_property_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute is backed by a Python property."""
        return isinstance(getattr(type(self), attr_name, None), property)

    def _helper_is_extra_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute was registered as an extra attr."""
        return self.impl_attrs[attr_name].get("kind") == "extra"

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
            "results may need synchronized updates. Please create a new "
            "instance instead."
        )

    def _helper_is_public_settable_attr(self, attr_name: str) -> bool:
        """Return whether one managed attribute is writable from the public surface."""
        if attr_name == "raw_name" or attr_name.startswith(("raw_", "state_")):
            return True
        if self._helper_is_property_attr(attr_name) or self._helper_is_extra_attr(
            attr_name
        ):
            return bool(self.impl_attrs[attr_name].get("is_public_settable", False))
        return False

    # ------------------------------------------------------------------
    # Attribute definition / registration
    # ------------------------------------------------------------------

    def _helper_collect_readable_names(
        self,
        *,
        is_exclude_name: str | None = None,
        is_exclude_impl: bool = False,
    ):
        """Collect the currently occupied readable attribute surface names."""
        readable_names: set[str] = set()

        for attr_name in self.impl_attrs:
            if attr_name == is_exclude_name:
                continue
            if is_exclude_impl and self._helper_is_impl_attr(attr_name):
                continue

            readable_names.add(attr_name)
            if attr_name.startswith("raw_"):
                readable_names.add(attr_name[4:])

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

    def _helper_register_attr_def(
        self,
        name: str,
        *,
        doc: str,
        kind: str | None = None,
        validator=None,
        is_public_settable: bool | None = None,
        is_overwrite: bool = False,
        **extra_def,
    ):
        """Register or update one runtime attribute metadata entry."""
        name = as_str(name, name="Attribute name")
        if not name.isidentifier():
            raise ValueError(
                f"Invalid attribute name {name!r}: must be a valid Python identifier."
            )

        self._helper_check_readable_name_conflict(name, is_overwrite=is_overwrite)

        if (name in self.impl_attrs) and (not is_overwrite):
            raise KeyError(
                f"Attribute {name!r} is already registered in "
                f"{type(self).__name__}.impl_attrs."
            )

        attr_info = {
            "doc": as_str(doc, name=f"Definition doc for {name!r}"),
        }
        if validator is not None:
            attr_info["validator"] = validator
        if is_public_settable is not None:
            attr_info["is_public_settable"] = bool(is_public_settable)

        is_public_assignable = (
            name == "raw_name"
            or name.startswith(("raw_", "state_"))
            or bool(attr_info.get("is_public_settable", False))
        )
        if is_public_assignable:
            attr_info["is_protected"] = False
        if kind is not None:
            attr_info["kind"] = as_str(kind, name=f"Definition kind for {name!r}")
        attr_info.update(extra_def)
        self.impl_attrs[name] = attr_info
        return attr_info

    # ------------------------------------------------------------------
    # Protection
    # ------------------------------------------------------------------

    def _helper_set_protected_attr(self, attrs, is_protected: bool):
        """Set the protected flag for one or more registered attributes."""
        for attr_name in as_list(attrs, name="attrs"):
            target_key = attr_name
            if target_key not in self.impl_attrs:
                raw_key = f"raw_{attr_name}"
                if raw_key in self.impl_attrs:
                    target_key = raw_key
                else:
                    raise AttributeError(
                        f"Cannot update protection for {attr_name!r}: "
                        "it is not registered in "
                        f"{type(self).__name__}.impl_attrs."
                    )
            if not self._helper_is_public_settable_attr(target_key):
                raise AttributeError(
                    f"Cannot update protection for {attr_name!r}: "
                    f"{target_key!r} is not a public assignment surface."
                )
            self.impl_attrs[target_key]["is_protected"] = is_protected

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
        relation_value = self.impl_attrs[name].get("relation_value", None)
        if isinstance(relation_value, weakref.ReferenceType):
            return relation_value()
        return relation_value

    def _helper_get_relation_doc(self, name: str) -> str:
        """Return the runtime doc override for a relation, or its declared doc."""
        doc_runtime = self.impl_attrs[name].get("doc_runtime", None)
        if doc_runtime is not None:
            return doc_runtime
        return self.impl_attrs[name]["doc"]

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
        for attr_name in self.impl_attrs:
            if not self._helper_is_relation_attr(attr_name):
                continue
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
        """Bind or update a named relation on this instance."""
        name = as_str(name, name=f"Relation name for instance {self.raw_name!r}")

        if name not in self.impl_attrs:
            self._helper_register_attr_def(
                name,
                doc="Newly added relation." if doc is None else doc,
                kind="relation",
                is_overwrite=False,
                is_weak_by_default=True,
                is_weak=None,
                relation_value=None,
                doc_runtime=None,
            )

        attr_info = self.impl_attrs[name]
        if not self._helper_is_relation_attr(name):
            raise AttributeError(
                f"Cannot bind relation {name!r}: it is not registered as a relation."
            )

        if doc is not None:
            self.impl_attrs[name]["doc_runtime"] = as_str(
                doc, name=f"Relation doc for instance {self.raw_name!r}"
            )

        old_target = self._helper_resolve_relation_value(name)
        if old_target is not None and old_target is not target and (not is_replace):
            raise RuntimeError(
                f"Relation {name!r} of {type(self).__name__} is already bound."
            )

        if is_weak is None:
            is_weak = bool(attr_info.get("is_weak_by_default", True))

        self.impl_attrs[name]["is_weak"] = bool(is_weak)
        self.impl_attrs[name]["relation_value"] = (
            weakref.ref(target) if (is_weak and target is not None) else target
        )
        return target

    def act_unbind_relation_base(self, name: str):
        """Clear the current target of a named relation."""
        name = as_str(name, name=f"Relation name for instance {self.raw_name!r}")
        if name not in self.impl_attrs:
            raise AttributeError(
                f"Cannot unbind relation {name!r}: it is not registered in "
                f"{type(self).__name__}.impl_attrs."
            )
        if not self._helper_is_relation_attr(name):
            raise AttributeError(
                f"Cannot unbind relation {name!r}: it is not registered as a relation."
            )

        self.impl_attrs[name]["relation_value"] = None
        self.impl_attrs[name]["is_weak"] = None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_readable_attrs(self, is_return=False, is_desc=True, logger=None):
        """Show the registered readable attributes for this instance."""
        lines = [
            "When reading, the raw_ prefix may be omitted where a public alias exists."
        ]

        def _readable_attr_sort_key(attr_name: str) -> tuple[int, str]:
            if attr_name.startswith("raw_"):
                group = 0
            elif attr_name.startswith("state_"):
                group = 1
            elif attr_name.startswith("default_"):
                group = 2
            elif self._helper_is_relation_attr(attr_name):
                group = 3
            elif attr_name.startswith("calc_"):
                group = 4
            elif self._helper_is_property_attr(attr_name):
                group = 5
            else:
                group = 6
            return group, attr_name

        attr_names = sorted(
            (
                attr_name
                for attr_name in self.impl_attrs
                if not self._helper_is_impl_attr(attr_name)
            ),
            key=_readable_attr_sort_key,
        )

        if not attr_names:
            lines.append("- <none>")
        else:
            for attr_name in attr_names:
                if self._helper_is_relation_attr(attr_name):
                    desc = self._helper_get_relation_doc(attr_name)
                else:
                    desc = self.impl_attrs[attr_name]["doc"]
                lines.append(f"- {attr_name}")
                if is_desc:
                    lines.append(f"    {desc}")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_relations(self, is_return=False, logger=None):
        """Show currently bound relations and their descriptions."""
        lines = []

        for attr_name in self.impl_attrs:
            if not self._helper_is_relation_attr(attr_name):
                continue

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

    # ------------------------------------------------------------------
    # Extra attributes
    # ------------------------------------------------------------------

    def _helper_resolve_extra_attr_value(self, name: str):
        """Return the current runtime value of one registered extra attribute."""
        return self.impl_attrs[name].get("value", None)

    def _helper_store_extra_attr(self, name: str, value) -> None:
        """Store one dynamic extra attribute value in impl_attrs metadata."""
        self.impl_attrs[name]["value"] = value

    def act_add_attr(
        self,
        name: str,
        doc: str,
        default=None,
        is_overwrite: bool = False,
    ):
        """Register a dynamic extra attribute with documentation and a default value."""
        if name in self.impl_attrs and not self._helper_is_extra_attr(name):
            raise AttributeError(
                f"Cannot register extra attribute {name!r}: it is already a managed "
                f"attribute of {type(self).__name__} and cannot be overwritten "
                "through act_add_attr()."
            )

        self._helper_register_attr_def(
            name,
            doc=doc,
            kind="extra",
            is_public_settable=True,
            is_overwrite=is_overwrite,
            value=default,
        )

    # ------------------------------------------------------------------
    # Attribute inspection
    # ------------------------------------------------------------------

    def show_attr_desc(self, attr_name: str) -> str:
        """Return the description of a registered attribute or its public alias."""
        if attr_name in self.impl_attrs:
            if self._helper_is_relation_attr(attr_name):
                return f"{attr_name!r}: {self._helper_get_relation_doc(attr_name)}"
            return f"{attr_name!r}: {self.impl_attrs[attr_name]['doc']}"

        raw_attr_name = f"raw_{attr_name}"
        if raw_attr_name in self.impl_attrs:
            return (
                f"{attr_name!r}: Alias of {raw_attr_name!r}. "
                f"{self.impl_attrs[raw_attr_name]['doc']}"
            )

        raise KeyError(
            f"Attribute {attr_name!r} was not found in "
            f"{type(self).__name__}.impl_attrs."
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def show_modifiable_attrs(self, is_return=False, is_desc=True, logger=None):
        """Show public attributes and properties intended for assignment."""
        lines = [
            "When assigning, the raw_ prefix may be omitted where a public alias exists."
        ]

        def _modifiable_attr_sort_key(attr_name: str) -> tuple[int, str]:
            if attr_name.startswith("raw_"):
                group = 0
            elif attr_name.startswith("state_"):
                group = 1
            elif self._helper_is_property_attr(attr_name):
                group = 2
            elif self._helper_is_extra_attr(attr_name):
                group = 3
            else:
                group = 4
            return group, attr_name

        attr_names = []
        for attr_name, attr_info in self.impl_attrs.items():
            if not self._helper_is_public_settable_attr(attr_name):
                continue
            if attr_info.get("is_protected", False):
                continue
            if self.impl_is_fixed and self._helper_is_fixed_blocked_attr(attr_name):
                continue
            attr_names.append(attr_name)

        attr_names = sorted(attr_names, key=_modifiable_attr_sort_key)

        if not attr_names:
            lines.append("- <none>")
        else:
            for attr_name in attr_names:
                desc = self.impl_attrs[attr_name]["doc"]
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
        raw_key = f"raw_{key}"
        if raw_key in self.impl_attrs:
            return object.__getattribute__(self, raw_key)

        if key in self.impl_attrs:
            if self._helper_is_relation_attr(key):
                return self._helper_resolve_relation_value(key)
            if self._helper_is_extra_attr(key):
                return self._helper_resolve_extra_attr_value(key)

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
        attr_info_map = self.impl_attrs
        target_key = key

        if target_key not in attr_info_map:
            raw_key = f"raw_{key}"
            if raw_key in attr_info_map:
                target_key = raw_key
            else:
                cls_name = type(self).__name__
                obj_name = getattr(self, "raw_name", "Uninitialized")
                raise AttributeError(
                    f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                    f"{key!r} is not a valid or registered attribute."
                )

        attr_info = attr_info_map[target_key]
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

        if attr_info.get("is_protected", False):
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{target_key!r} is protected."
            )

        validator = attr_info.get("validator")
        if validator is not None and target_key != "raw_name":
            value = validator(value, attr_info["doc"])

        self._helper_setattr_final(key, value, target_key=target_key)

    def _helper_setattr_final(self, key, value, *, target_key=None):
        """Apply one validated public assignment to final storage."""
        target_key = key if target_key is None else target_key
        if target_key == "raw_name":
            self.act_set_name(value)
            return
        if self._helper_is_extra_attr(target_key):
            self._helper_store_extra_attr(target_key, value)
            return

        object.__setattr__(self, target_key, value)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        msg = f"{cls_name}({self.name!r})"
        return msg
