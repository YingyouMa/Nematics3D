from nematics3d.datatypes import as_str
from nematics3d.logging_decorator import logging_and_warning_decorator
from .class_base import ClassBase


# RegistryBase developer conventions:
# - RegistryBase is intentionally a lightweight registry helper, not a
#   ClassBase-style entity base. Keep it ordinary enough that HostBase classes
#   can safely inherit from it without multiple-layout conflicts.
# - `__attr_defs__` is kept as lightweight metadata so nearby code can still use
#   ClassBase-like docs and declarations when helpful.
# - Registered objects should still be renamed through the registry when needed
#   and should be bound/unbound through `act_bind_relation_base()` /
#   `act_unbind_relation_base()` when available.
# - Keep `__repr__` as the detailed registry view. `str(registry)` should remain
#   the compact identity-style display used in relation trees and short logs.


class RegistryBase(ClassBase):
    """
    Lightweight registry for storing and looking up named objects.

    For most users, RegistryBase is meant to be used directly rather than
    subclassed.

    Typical usage:

    - use `act_register(obj)` to add an object
    - use `registry[name]` or `registry[index]` to retrieve an object
    - use `for obj in registry` or `len(registry)` to work with the collection
    - use `repr(registry)` to inspect the registered contents in detail

    When an object with a duplicate name is registered, the registry will
    automatically rename it so names stay unique inside that registry.
    """

    # fmt: off
    __attr_defs__ = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            "doc": "The name of the Registry.",
            "validator": as_str,
            "is_protected": False,
        },
        "raw_info": {
            "doc":       "The extra introduction for this instance for clarity.",
            "validator": lambda v, d: None if v is None else as_str(v, name=d, replace=None),
            "is_protected": False,
        },
        "_entity": {
            "doc": "Internal container storing the registered objects.",
        },
    }
    # fmt: on

    __slots__ = ("raw_info", "_entity")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, name, info=None):
        super().__init__(name=name, name_replace="registry")
        info = self.impl_attrs["raw_info"]["validator"](
            info,
            self.impl_attrs["raw_info"]["doc"],
        )
        object.__setattr__(self, "raw_info", info)
        object.__setattr__(self, "_entity", [])

    # ------------------------------------------------------------------
    # Readable properties / compatibility helpers
    # ------------------------------------------------------------------

    @property
    def _impl_owner(self):
        """Compatibility alias for older code that expects `_impl_owner`."""
        return self.owner

    @property
    def entities(self):
        """Return the registered objects as a tuple in current registry order."""
        return tuple(self._entity)

    def act_set_name(self, value):
        self.name = value
        return self.name

    # ------------------------------------------------------------------
    # Naming / display helpers
    # ------------------------------------------------------------------

    def _helper_show_name_info(self) -> str:
        info = getattr(self, "raw_info", None)
        if info:
            return f"Registry {self.name!r} ({info})"
        return f"Registry {self.name!r}"

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_name(self, name: str, logger=None):
        name_set = {item.name for item in self._entity}
        name = as_str(name, name="The name of the term to register")
        name_input = name
        if name_input in name_set:
            new_name = name_input
            index = 1
            while new_name in name_set:
                new_name = f"{name_input}_{index}"
                index += 1
            logger.warning(
                f"{name_input!r} already exists in {self._helper_show_name_info()}! Renamed to {new_name!r}."
            )
            name = new_name
        return name

    # ------------------------------------------------------------------
    # Registration actions
    # ------------------------------------------------------------------

    @logging_and_warning_decorator(start_finish_level=5)
    def act_register(
        self,
        term,
        is_contain_ok=False,
        is_bind_registry_relation=True,
        logger=None,
    ):
        if term in self._entity:
            if not is_contain_ok:
                try:
                    raise ValueError(
                        f"term {term!r} is already registered in {self._helper_show_name_info()}"
                    )
                except ValueError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this process.")
            return

        if not hasattr(term, "name"):
            raise TypeError("term must have attribute `name`.")

        old_registry = getattr(term, "registry", None)
        if old_registry is not None and old_registry is not self:
            logger.warning(
                f"{term!r} already has a registry {old_registry!r}. Move it to {self._helper_show_name_info()}."
            )
            old_registry.act_unregister(term, is_missing_ok=True)

        name = self._helper_check_name(term.name)
        set_name = getattr(term, "act_set_name", None)
        if callable(set_name):
            set_name(name)
        else:
            term.name = name
        self._entity.append(term)

        if not is_bind_registry_relation:
            return

        bind_relation = getattr(term, "act_bind_relation_base", None)
        if callable(bind_relation):
            bind_relation("registry", self, is_weak=True)
        else:
            logger.warning(
                f"Failed to assign registry relation for {term!r}. "
                "This registration is one-way only because the object does not expose act_bind_relation_base()."
            )

    @logging_and_warning_decorator(start_finish_level=5)
    def act_unregister(self, term, is_missing_ok=False, logger=None):
        if term not in self._entity:
            if not is_missing_ok:
                try:
                    raise KeyError(
                        f"term {term!r} is not registered in {self._helper_show_name_info()}"
                    )
                except KeyError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this process.")
            return

        self._entity.remove(term)

        registry = getattr(term, "registry", None)
        if registry is self:
            unbind_relation = getattr(term, "act_unbind_relation_base", None)
            if callable(unbind_relation):
                unbind_relation("registry")

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __call__(self):
        return tuple(self._entity)

    def __len__(self) -> int:
        return len(self._entity)

    def __iter__(self):
        return iter(self._entity)

    def __contains__(self, item):
        return item in self._entity

    def __getitem__(self, key: str | int | None):
        if key is None:
            return None
        if isinstance(key, int):
            return self._entity[key]
        if isinstance(key, str):
            for obj in self._entity:
                if obj.name == key:
                    return obj
            raise KeyError(
                f"No object with name '{key}' found in {self._helper_show_name_info()}."
            )
        raise TypeError(
            f"`key` must be str or int for {self._helper_show_name_info()} indexing, "
            f"got {type(key).__name__} instead."
        )

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def _helper_repr_by_order(self, is_name=True) -> str:
        if not self._entity:
            return "<empty registry>"

        if is_name:
            names = [obj.name for obj in self._entity]
        else:
            names = [str(obj) for obj in self._entity]

        idx_width = len(str(len(names) - 1))
        name_width = max(len(name) for name in names)

        lines = []
        for i, name in enumerate(names):
            lines.append(f"{i:>{idx_width}d}:       {name:<{name_width}}")

        return "\n".join(lines)

    def _helper_repr_by_category(self, is_name=False) -> str:
        if not self._entity:
            return "<empty registry>"

        records = []
        for obj in self._entity:
            name = obj.name if is_name else str(obj)
            category = getattr(obj, "raw_category", type(obj).__name__)
            records.append((category, name))

        grouped: dict[str, list[str]] = {}
        for category, name in records:
            grouped.setdefault(category, []).append(name)

        cat_width = max(len(cat) for cat in grouped.keys())

        lines: list[str] = []
        for category, names in grouped.items():
            joined_names = ", ".join(names)
            lines.append(f"{category:<{cat_width}} : {joined_names}")

        return "\n".join(lines)

    def __str__(self):
        return f"{type(self).__name__}({self.name!r})"

    def __repr__(self):
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})\n"
        return msg + self._helper_repr_by_order()
