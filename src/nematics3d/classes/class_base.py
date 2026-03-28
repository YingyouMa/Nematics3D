from copy import deepcopy
import weakref

from ..datatypes import as_list, as_str
from ..logging_decorator import logging_and_warning_decorator


class ClassBase:
    __attr_defs__ = {
        "raw_name": {
            "doc": "The underlying string identifier for this instance.",
            "kind": "raw",
            "validator": as_str,
            "is_public_settable": True,
        },
        "owner": {
            "doc": "The object that owns this instance.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_weak_by_default": True,
        },
        "registry": {
            "doc": "The Registry object where this instance is registered.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_weak_by_default": True,
        },
        "impl_attr_defs": {
            "doc": "Runtime attribute-definition metadata copied from the class template.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
        },
        "impl_attr_state": {
            "doc": "Runtime state metadata for each registered attribute.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
        },
    }

    __slots__ = ("raw_name", "impl_attr_defs", "impl_attr_state", "__weakref__")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, *, name: str | None, name_replace: str):
        impl_attr_defs = deepcopy(type(self).__attr_defs__)
        object.__setattr__(self, "impl_attr_defs", impl_attr_defs)

        impl_attr_state = {
            attr_name: {
                "is_protected": False,
            }
            for attr_name in self.impl_attr_defs
        }
        object.__setattr__(self, "impl_attr_state", impl_attr_state)

        if name is None:
            name_final = name_replace
        else:
            name_final = self.impl_attr_defs["raw_name"]["validator"](
                name,
                name=self.impl_attr_defs["raw_name"]["doc"],
                replace=name_replace,
            )
            if not name_final:
                name_final = name_replace

        self._helper_assign_name(name_final)
        return None

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
        """Register or update one runtime attribute definition and its base state."""
        name = as_str(name, name="Attribute name")
        if not name.isidentifier():
            raise ValueError(
                f"Invalid attribute name {name!r}: must be a valid Python identifier."
            )

        if (name in self.impl_attr_defs) and (not is_overwrite):
            raise KeyError(
                f"Attribute {name!r} is already registered in {type(self).__name__}.impl_attr_defs."
            )

        attr_def = {
            "doc": as_str(doc, name=f"Definition doc for {name!r}"),
            "kind": as_str(kind, name=f"Definition kind for {name!r}"),
            "validator": validator,
            "is_public_settable": bool(is_public_settable),
        }
        attr_def.update(extra_def)
        self.impl_attr_defs[name] = attr_def

        if name not in self.impl_attr_state:
            self.impl_attr_state[name] = {
                "is_protected": False,
            }

        return attr_def

    # ------------------------------------------------------------------
    # Protection
    # ------------------------------------------------------------------

    def _helper_set_protected_attr(self, attrs, is_protected: bool):
        """Set the protected flag for one or more registered attributes."""
        for attr_name in as_list(attrs, name="attrs"):
            target_key = attr_name
            if target_key not in self.impl_attr_defs:
                raw_key = f"raw_{attr_name}"
                if raw_key in self.impl_attr_defs:
                    target_key = raw_key
                else:
                    raise AttributeError(
                        f"Cannot update protection for {attr_name!r}: it is not registered in "
                        f"{type(self).__name__}.impl_attr_defs."
                    )
            self.impl_attr_state[target_key]["is_protected"] = is_protected
        return None

    def act_register_protected_attr(self, attrs):
        """Mark one or more registered attributes as protected from public assignment."""
        self._helper_set_protected_attr(attrs, True)
        return None

    def act_unregister_protected_attr(self, attrs):
        """Remove the protected flag from one or more registered attributes."""
        self._helper_set_protected_attr(attrs, False)
        return None

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def _helper_resolve_relation_value(self, name: str):
        """Return the current relation target, resolving weak references when needed."""
        relation_value = self.impl_attr_state[name].get("relation_value", None)
        if isinstance(relation_value, weakref.ReferenceType):
            return relation_value()
        return relation_value

    def _helper_get_relation_doc(self, name: str) -> str:
        """Return the runtime doc override for a relation, or its declared doc."""
        doc_runtime = self.impl_attr_state[name].get("doc_runtime", None)
        if doc_runtime is not None:
            return doc_runtime
        return self.impl_attr_defs[name]["doc"]

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

        if name not in self.impl_attr_defs:
            self._helper_register_attr_def(
                name,
                doc="Newly added relation." if doc is None else doc,
                kind="relation",
                validator=None,
                is_public_settable=False,
                is_overwrite=False,
                is_weak_by_default=True,
            )

        attr_def = self.impl_attr_defs[name]
        if attr_def["kind"] != "relation":
            raise AttributeError(
                f"Cannot bind relation {name!r}: it is registered as kind {attr_def['kind']!r}, not 'relation'."
            )

        if doc is not None:
            self.impl_attr_state[name]["doc_runtime"] = as_str(
                doc, name=f"Relation doc for instance {self.raw_name!r}"
            )

        old_target = self._helper_resolve_relation_value(name)
        if old_target is not None and old_target is not target and (not is_replace):
            raise RuntimeError(
                f"Relation {name!r} of {type(self).__name__} is already bound."
            )

        if is_weak is None:
            is_weak = bool(attr_def.get("is_weak_by_default", True))

        self.impl_attr_state[name]["is_weak"] = bool(is_weak)
        self.impl_attr_state[name]["relation_value"] = (
            weakref.ref(target) if (is_weak and target is not None) else target
        )
        return target

    def act_unbind_relation_base(self, name: str):
        """Clear the current target of a named relation."""
        name = as_str(name, name=f"Relation name for instance {self.raw_name!r}")
        if name not in self.impl_attr_defs:
            raise AttributeError(
                f"Cannot unbind relation {name!r}: it is not registered in "
                f"{type(self).__name__}.impl_attr_defs."
            )
        if self.impl_attr_defs[name]["kind"] != "relation":
            raise AttributeError(
                f"Cannot unbind relation {name!r}: it is registered as kind "
                f"{self.impl_attr_defs[name]['kind']!r}, not 'relation'."
            )

        self.impl_attr_state[name]["relation_value"] = None
        self.impl_attr_state[name]["is_weak"] = bool(
            self.impl_attr_defs[name].get("is_weak_by_default", True)
        )
        return None

    # ------------------------------------------------------------------
    # Attribute inspection
    # ------------------------------------------------------------------

    @classmethod
    def _helper_is_writable_property(cls, name: str) -> bool:
        """Return whether a declared property is marked as writable."""
        attr_def = cls.__attr_defs__.get(name, {})
        return attr_def.get("kind") == "property" and attr_def.get(
            "is_public_settable", False
        )

    def show_attr_desc(self, attr_name: str) -> str:
        """Return the description of a registered attribute or its public alias."""
        if attr_name in self.impl_attr_defs:
            if self.impl_attr_defs[attr_name]["kind"] == "relation":
                return f"{attr_name!r}: {self._helper_get_relation_doc(attr_name)}"
            return f"{attr_name!r}: {self.impl_attr_defs[attr_name]['doc']}"

        raw_attr_name = f"raw_{attr_name}"
        if raw_attr_name in self.impl_attr_defs:
            return (
                f"{attr_name!r}: Alias of {raw_attr_name!r}. "
                f"{self.impl_attr_defs[raw_attr_name]['doc']}"
            )

        raise KeyError(
            f"Attribute {attr_name!r} was not found in "
            f"{type(self).__name__}.impl_attr_defs."
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def show_getattrs(self, is_return=False, logger=None):
        """Show readable attributes, aliases, and declared properties."""
        lines = [
            "When reading or assigning, the 'raw_' prefix may be omitted where a public alias exists."
        ]

        for attr_name in self.impl_attr_defs:
            lines.append(self.show_attr_desc(attr_name))
            if attr_name.startswith("raw_"):
                lines.append(self.show_attr_desc(attr_name[4:]))

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
        for attr_name, attr_def in self.impl_attr_defs.items():
            if attr_def["kind"] == "property":
                if attr_def.get("is_public_settable", False):
                    property_names.append(attr_name)
                continue

            if not attr_def["is_public_settable"]:
                continue
            if self.impl_attr_state[attr_name]["is_protected"]:
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

    @logging_and_warning_decorator(start_finish_level=5)
    def show_relations(self, is_return=False, logger=None):
        """Show currently bound relations and their descriptions."""
        lines = []

        for attr_name, attr_def in self.impl_attr_defs.items():
            if attr_def["kind"] != "relation":
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

    # ------------------------------------------------------------------
    # Attribute access / assignment
    # ------------------------------------------------------------------

    def __getattr__(self, key):
        raw_key = f"raw_{key}"
        if raw_key in self.impl_attr_defs:
            return object.__getattribute__(self, raw_key)

        if (
            key in self.impl_attr_defs
            and self.impl_attr_defs[key]["kind"] == "relation"
        ):
            return self._helper_resolve_relation_value(key)

        cls_name = type(self).__name__
        try:
            obj_name = object.__getattribute__(self, "raw_name")
        except AttributeError:
            obj_name = "Uninitialized"
        raise AttributeError(f"[{cls_name}: {obj_name!r}] has no attribute {key!r}.")

    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)
        return None

    def _helper_setattr_basic(self, key, value):
        """Resolve a public assignment target and apply validation/protection rules."""
        attr_defs = self.impl_attr_defs
        target_key = key

        if target_key not in attr_defs:
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

        attr_def = attr_defs[target_key]
        if not attr_def["is_public_settable"]:
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{key!r} resolves to internal attribute {target_key!r}, "
                "which cannot be assigned through the public setattr path."
            )

        if self.impl_attr_state[target_key]["is_protected"]:
            cls_name = type(self).__name__
            obj_name = getattr(self, "raw_name", "Uninitialized")
            raise AttributeError(
                f"[{cls_name}: {obj_name!r}] Assignment blocked: "
                f"{target_key!r} is protected."
            )

        if target_key == "raw_name":
            value = attr_def["validator"](value, name=attr_def["doc"])
            self._helper_assign_name(value)
            return None

        object.__setattr__(self, target_key, value)
        return None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        msg = f"{cls_name}({self.name!r})"
        return msg
