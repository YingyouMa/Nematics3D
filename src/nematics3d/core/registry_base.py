"""Lightweight registry helpers for named repository objects."""

from nematics3d.datatypes import as_str
from nematics3d.logging_decorator import logging_and_warning_decorator
from .class_base import AttrDef, ClassBase


# RegistryBase developer conventions:
# - RegistryBase is a lightweight `ClassBase` subclass for ordered named-object
#   registration. Keep the managed schema minimal and avoid turning it into a
#   HostBase-style commit pipeline container.
# - `__attr_defs__` here should stay small: normal public inputs such as
#   `raw_name` / `raw_info`, inherited direct-named relations, and only the
#   runtime storage that RegistryBase itself actually owns.
# - Do not register no-op schema metadata such as static `is_protected` flags
#   when they add no real behavior.
# - Unless there is a strong readability or compatibility reason, internal
#   runtime variables should also follow the normal naming scheme instead of
#   defaulting to a leading underscore.
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
        "raw_name": AttrDef(
            doc="The name of the Registry.",
            kind="raw",
            validator=as_str,
        ),
        "raw_info": AttrDef(
            doc="The extra introduction for this instance for clarity.",
            kind="raw",
            validator=lambda v, d: None if v is None else as_str(v, name=d, replace=None),
        ),
        "impl_entity": AttrDef(
            doc="Internal mutable storage for registered objects in insertion order.",
            kind="impl",
        ),
        "entity": AttrDef(
            doc="Read-only: Registered objects in insertion order.",
            kind="property",
        ),
    }
    # fmt: on

    __slots__ = ("raw_info", "impl_entity")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, name, info=None):
        super().__init__(name=name, name_replace="registry")
        defn = type(self).__attr_defs__["raw_info"]
        info = defn.validator(info, defn.doc)
        object.__setattr__(self, "raw_info", info)
        object.__setattr__(self, "impl_entity", [])

    # ------------------------------------------------------------------
    # Readable properties
    # ------------------------------------------------------------------

    @property
    def entity(self):
        """Return the registered objects as an immutable tuple view."""
        return tuple(self.impl_entity)

    # ------------------------------------------------------------------
    # Naming / display helpers
    # ------------------------------------------------------------------

    def _helper_show_name_info(self) -> str:
        info = getattr(self, "raw_info", None)
        if info:
            return f"Registry {self.name!r} ({info})"
        return f"Registry {self.name!r}"

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_name(self, name: str, *, exclude=None, logger=None):
        """Return a unique registry name, optionally ignoring one registered object."""
        name_set = {
            item.name
            for item in self.impl_entity
            if exclude is None or item is not exclude
        }
        name = as_str(name, name="The name of the term to register")
        name_input = name
        if name_input in name_set:
            new_name = name_input
            index = 1
            while new_name in name_set:
                new_name = f"{name_input}_{index}"
                index += 1
            logger.warning(
                f"{name_input!r} already exists in "
                f"{self._helper_show_name_info()}! "
                f"Renamed to {new_name!r}."
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
        """Register one object, keep its name unique, and return that object."""
        if term in self.impl_entity:
            if not is_contain_ok:
                try:
                    raise ValueError(
                        f"term {term!r} is already registered in {self._helper_show_name_info()}"
                    )
                except ValueError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this process.")
            return term

        if not hasattr(term, "name"):
            raise TypeError("term must have attribute `name`.")

        old_registry = getattr(term, "registry", None)
        if old_registry is not None and old_registry is not self:
            logger.warning(
                f"{term!r} already has a registry "
                f"{old_registry!r}. Move it to "
                f"{self._helper_show_name_info()}."
            )
            old_registry.act_unregister(term, is_missing_ok=True)

        name = self._helper_check_name(term.name)
        set_name = getattr(term, "act_set_name", None)
        if callable(set_name):
            set_name(name)
        else:
            term.name = name
        self.impl_entity.append(term)

        if not is_bind_registry_relation:
            return term

        bind_relation = getattr(term, "act_bind_relation_base", None)
        if callable(bind_relation):
            bind_relation("registry", self, is_weak=True)
        else:
            logger.warning(
                f"Failed to assign registry relation for {term!r}. "
                "This registration is one-way only because the object "
                "does not expose act_bind_relation_base()."
            )
        return term

    @logging_and_warning_decorator(start_finish_level=5)
    def act_unregister(self, term, is_missing_ok=False, logger=None):
        """Unregister one object and unbind its registry relation when possible."""
        if term not in self.impl_entity:
            if not is_missing_ok:
                try:
                    raise KeyError(
                        f"term {term!r} is not registered in {self._helper_show_name_info()}"
                    )
                except KeyError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this process.")
            return

        self.impl_entity.remove(term)

        registry = getattr(term, "registry", None)
        if registry is self:
            unbind_relation = getattr(term, "act_unbind_relation_base", None)
            if callable(unbind_relation):
                unbind_relation("registry")

    @logging_and_warning_decorator(start_finish_level=5)
    def act_clear(
        self,
        *,
        is_return_removed: bool = False,
        is_show_existing: bool = True,
        logger=None,
    ):
        """Unregister all objects currently stored in this registry."""
        removed = tuple(self.impl_entity)

        if is_show_existing:
            logger.info(
                "Clear registered objects from "
                f"{self._helper_show_name_info()}:\n{self.act_repr_by_order()}"
            )

        for term in removed:
            self.act_unregister(term, is_missing_ok=True)

        if is_return_removed:
            return removed
        return None

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __call__(self):
        """Return the registered objects as a tuple in current registry order."""
        return tuple(self.entity)

    def __len__(self) -> int:
        """Return the number of registered objects."""
        return len(self.impl_entity)

    def __iter__(self):
        """Iterate over the registered objects in insertion order."""
        return iter(self.impl_entity)

    def __contains__(self, item):
        """Return whether one object is currently registered."""
        return item in self.impl_entity

    def __getitem__(self, key: str | int | None):
        """Lookup one registered object by index, name, or None passthrough."""
        if key is None:
            return None
        if isinstance(key, int):
            return self.impl_entity[key]
        if isinstance(key, str):
            for obj in self.impl_entity:
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

    def act_repr_by_order(self, is_name=True) -> str:
        """Return an order-preserving multi-line summary of registered objects."""
        if not self.impl_entity:
            return "<empty registry>"

        if is_name:
            names = [obj.name for obj in self.impl_entity]
        else:
            names = [str(obj) for obj in self.impl_entity]

        idx_width = len(str(len(names) - 1))
        name_width = max(len(name) for name in names)

        lines = []
        for i, name in enumerate(names):
            lines.append(f"{i:>{idx_width}d}:       {name:<{name_width}}")

        return "\n".join(lines)

    def act_repr_by_category(self, is_name=False) -> str:
        """Return a category-grouped multi-line summary of registered objects."""
        if not self.impl_entity:
            return "<empty registry>"

        records = []
        for obj in self.impl_entity:
            name = obj.name if is_name else str(obj)
            category = getattr(obj, "raw_category", type(obj).__name__)
            records.append((category, name))

        grouped: dict[str, list[str]] = {}
        for category, name in records:
            grouped.setdefault(category, []).append(name)

        cat_width = max(len(cat) for cat in grouped)

        lines: list[str] = []
        for category, names in grouped.items():
            joined_names = ", ".join(names)
            lines.append(f"{category:<{cat_width}} : {joined_names}")

        return "\n".join(lines)

    def __str__(self):
        """Return the compact identity-style string form of this registry."""
        return f"{type(self).__name__}({self.name!r})"

    def __repr__(self):
        """Return the detailed registry summary with registered object order."""
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})\n"
        return msg + self.act_repr_by_order()
