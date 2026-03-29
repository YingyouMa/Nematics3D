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
# - `raw_` fields are the canonical stored public data fields. A `raw_xxx`
#   field automatically exposes the readable public alias `xxx`.
# - `state_` fields represent writable runtime state inputs that are part of
#   the managed attribute system and may affect later computation. They do not
#   create a shortened public alias.
# - `default_` fields represent default-parameter style managed fields. The
#   current `ClassBase` does not declare one itself, but subclasses may use
#   this prefix when they need default-layer state in the managed schema.
# - `calc_` fields represent read-only values derived by computation.
# - `impl_` fields represent internal implementation metadata or runtime
#   containers and should not be treated as a user-facing readable surface.
# - relation names use their direct public names, such as `owner` or
#   `registry`.
# - relations in the current ClassBase protocol are one-to-one links only;
#   do not use relations to represent one-to-many or collection-style data.
# - property metadata should also be registered in `__attr_defs__` with
#   `kind="property"`, while the actual getter/setter behavior remains a
#   normal Python `@property` on the class.
# - extra attrs are runtime-registered public fields and should still enter the
#   managed schema through the provided registration helpers.
# - for semantic clarity, do not introduce other non-underscore public field
#   categories beyond these conventions: `raw_`, `state_`, `default_`, `calc_`,
#   `impl_`, direct-named relations, direct-named properties, and extra attrs.
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

    Each entry in ``impl_attrs`` combines both relatively stable definition data
    such as ``kind``, ``doc``, and ``validator``, and runtime state such as
    protection flags or current relation bindings.

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
            "kind": "raw",
            "validator": as_str,
            "is_public_settable": True,
            "is_protected": False,
        },
        "owner": {
            "doc": "The object that owns this instance.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "registry": {
            "doc": "The Registry object where this instance is registered.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "impl_attrs": {
            "doc": "Runtime attribute metadata copied from the class template.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
    }

    __slots__ = ("raw_name", "impl_attrs", "__weakref__")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, *, name: str | None, name_replace: str):
        # Each instance starts from a private copy of the class-level
        # attribute template.
        impl_attrs = deepcopy(type(self).__attr_defs__)
        object.__setattr__(self, "impl_attrs", impl_attrs)

        # Normalize the initial name before routing it through the shared
        # name assignment path.
        if name is None:
            name_final = name_replace
        else:
            name_final = self.impl_attrs["raw_name"]["validator"](
                name,
                name=self.impl_attrs["raw_name"]["doc"],
                replace=name_replace,
            )
            if not name_final:
                name_final = name_replace

        self._helper_assign_name(name_final)

    # ------------------------------------------------------------------
    # Name handling
    # ------------------------------------------------------------------

    def _helper_assign_name(self, name):
        """Validate registry-level naming constraints and store ``raw_name``."""
        check_name = getattr(
            getattr(self, "registry", None), "_helper_check_name", None
        )
        if callable(check_name):
            name = check_name(name)

        object.__setattr__(self, "raw_name", name)
        return name

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

        for attr_name, attr_info in self.impl_attrs.items():
            if attr_name == is_exclude_name:
                continue
            if is_exclude_impl and attr_info["kind"] == "impl":
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
        kind: str,
        validator=None,
        is_public_settable: bool,
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
            "kind": as_str(kind, name=f"Definition kind for {name!r}"),
            "validator": validator,
            "is_public_settable": bool(is_public_settable),
            "is_protected": False,
        }
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
        for attr_name, attr_info in self.impl_attrs.items():
            if attr_info["kind"] != "relation":
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
            branch = "â””â”€ " if is_last else "â”œâ”€ "
            child_prefix = _prefix + ("   " if is_last else "â”‚  ")

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
                validator=None,
                is_public_settable=False,
                is_overwrite=False,
                is_weak_by_default=True,
                is_weak=None,
                relation_value=None,
                doc_runtime=None,
            )

        attr_info = self.impl_attrs[name]
        if attr_info["kind"] != "relation":
            raise AttributeError(
                f"Cannot bind relation {name!r}: it is registered as kind "
                f"{attr_info['kind']!r}, not 'relation'."
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
        if self.impl_attrs[name]["kind"] != "relation":
            raise AttributeError(
                f"Cannot unbind relation {name!r}: it is registered as kind "
                f"{self.impl_attrs[name]['kind']!r}, not 'relation'."
            )

        self.impl_attrs[name]["relation_value"] = None
        self.impl_attrs[name]["is_weak"] = None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_relations(self, is_return=False, logger=None):
        """Show currently bound relations and their descriptions."""
        lines = []

        for attr_name, attr_info in self.impl_attrs.items():
            if attr_info["kind"] != "relation":
                continue

            target = self._helper_resolve_relation_value(attr_name)
            if target is None:
                continue

            lines.append(f"{attr_name}: {self._helper_get_relation_doc(attr_name)}")
            lines.append(f"  current: {target}")

        if not lines:
            lines.append("<none>")

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

    def act_add_attr(
        self,
        name: str,
        doc: str,
        default=None,
        is_overwrite: bool = False,
    ):
        """Register a dynamic extra attribute with documentation and a default value."""
        self._helper_register_attr_def(
            name,
            doc=doc,
            kind="extra",
            validator=None,
            is_public_settable=True,
            is_overwrite=is_overwrite,
        )

        if is_overwrite or (not hasattr(self, name)):
            object.__setattr__(self, name, default)

    # ------------------------------------------------------------------
    # Attribute inspection
    # ------------------------------------------------------------------

    def show_attr_desc(self, attr_name: str) -> str:
        """Return the description of a registered attribute or its public alias."""
        if attr_name in self.impl_attrs:
            if self.impl_attrs[attr_name]["kind"] == "relation":
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
    def show_readable_attrs(self, is_return=False, logger=None):
        """Show the public readable attribute surface for this instance."""
        lines = [
            "When reading or assigning, the 'raw_' prefix may be omitted "
            "where a public alias exists."
        ]

        for attr_name in sorted(
            self._helper_collect_readable_names(is_exclude_impl=True)
        ):
            lines.append(self.show_attr_desc(attr_name))

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_modifiable_attrs(self, is_return=False, logger=None):
        """Show public attributes and properties intended for assignment."""
        lines = [
            "When assigning, the 'raw_' prefix may be omitted.",
        ]

        attr_names = []
        property_names = []
        for attr_name, attr_info in self.impl_attrs.items():
            if attr_info["kind"] == "property":
                if attr_info.get("is_public_settable", False):
                    property_names.append(attr_name)
                continue

            if not attr_info["is_public_settable"]:
                continue
            if attr_info["is_protected"]:
                continue

            attr_names.append(attr_name)
            if attr_name.startswith("raw_"):
                attr_names.append(attr_name[4:])

        if attr_names:
            lines.append("[Attributes]")
            for attr_name in sorted(attr_names):
                lines.append(f"  - {attr_name}")
        else:
            lines.append("[Attributes]")
            lines.append("  - <none>")

        if property_names:
            lines.append("[Writable properties]")
            for prop_name in sorted(property_names):
                lines.append(f"  - {prop_name}")
        else:
            lines.append("[Writable properties]")
            lines.append("  - <none>")

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

        if key in self.impl_attrs and self.impl_attrs[key]["kind"] == "relation":
            return self._helper_resolve_relation_value(key)

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
        if not attr_info["is_public_settable"]:
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{key!r} resolves to internal attribute {target_key!r}, "
                "which cannot be assigned through the public setattr path."
            )

        if attr_info["is_protected"]:
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{target_key!r} is protected."
            )

        if target_key == "raw_name":
            value = attr_info["validator"](value, name=attr_info["doc"])

        self._helper_setattr_final(key, value, target_key=target_key)

    def _helper_setattr_final(self, key, value, *, target_key=None):
        """Apply one validated public assignment to final storage."""
        target_key = key if target_key is None else target_key
        if target_key == "raw_name":
            self._helper_assign_name(value)
            return

        object.__setattr__(self, target_key, value)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        msg = f"{cls_name}({self.name!r})"
        return msg





