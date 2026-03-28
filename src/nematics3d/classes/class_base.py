from copy import deepcopy

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

    # ------------------------------------------------------------------
    # Attribute access / assignment
    # ------------------------------------------------------------------

    def __getattr__(self, key):
        raw_key = f"raw_{key}"
        if raw_key in self.impl_attr_defs:
            return object.__getattribute__(self, raw_key)

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
