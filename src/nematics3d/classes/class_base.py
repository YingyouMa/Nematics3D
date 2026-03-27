from ..datatypes import as_str


class ClassBase:
    __attr_defs__ = {
        "raw_name": {
            "doc": "The underlying string identifier for this instance.",
            "kind": "raw",
            "validator": as_str,
            "is_public_settable": True,
        },
        "impl_attr_state": {
            "doc": "Runtime state metadata for each registered attribute.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
        },
    }

    __slots__ = ("raw_name", "impl_attr_state", "__weakref__")

    def __init__(self, *, name: str | None, name_replace: str):
        impl_attr_state = {
            attr_name: {
                "is_protected": False,
            }
            for attr_name in type(self).__attr_defs__
        }
        object.__setattr__(self, "impl_attr_state", impl_attr_state)

        if name is None:
            name_final = name_replace
        else:
            name_final = type(self).__attr_defs__["raw_name"]["validator"](
                name,
                name=type(self).__attr_defs__["raw_name"]["doc"],
                replace=name_replace,
            )
            if not name_final:
                name_final = name_replace

        self._helper_assign_name(name_final)

    def _helper_assign_name(self, name):
        check_name = getattr(
            getattr(self, "registry", None), "_helper_check_name", None
        )
        if callable(check_name):
            name = check_name(name)

        object.__setattr__(self, "raw_name", name)
        return name

    def __getattr__(self, key):
        raw_key = f"raw_{key}"
        if raw_key in type(self).__attr_defs__:
            return object.__getattribute__(self, raw_key)

        cls_name = type(self).__name__
        try:
            obj_name = object.__getattribute__(self, "raw_name")
        except AttributeError:
            obj_name = "Uninitialized"
        raise AttributeError(f"[{cls_name}: {obj_name!r}] has no attribute {key!r}.")

    def show_attr_desc(self, attr_name: str) -> str:
        if attr_name in type(self).__attr_defs__:
            return f"{attr_name!r}: {type(self).__attr_defs__[attr_name]['doc']}"

        raw_attr_name = f"raw_{attr_name}"
        if raw_attr_name in type(self).__attr_defs__:
            return (
                f"{attr_name!r}: Alias of {raw_attr_name!r}. "
                f"{type(self).__attr_defs__[raw_attr_name]['doc']}"
            )

        raise KeyError(
            f"Attribute {attr_name!r} was not found in "
            f"{type(self).__name__}.__attr_defs__."
        )

    def _helper_setattr_basic(self, key, value):
        attr_defs = type(self).__attr_defs__
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

        if target_key == "raw_name":
            value = attr_def["validator"](value, name=attr_def["doc"])
            value = self._helper_assign_name(value)
            return

        object.__setattr__(self, target_key, value)

    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)
