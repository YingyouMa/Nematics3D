import weakref

from Nematics3D.logging_decorator import (
    log_caught_exception,
    logging_and_warning_decorator,
)
from Nematics3D.datatypes import as_str


class RegistryBase:

    # fmt: off
    __attrs__ = {
        "raw_name":         "The name of the Registry.",
        "raw_info":        "The extra introduction for this instance for clarity",
        "_entity":          "The container storing the objects.",
    }
    __relations__ = {
        "owner": (
            "The owner object associated with this registry. "
            "To access it, use .owner or ._impl_owner."
        ),
    }
    # fmt: on
    
    def __init__(self, name, info=None):
        name = as_str(name, name="The name of the Registry")
        object.__setattr__(self, "raw_name", name)
        object.__setattr__(self, "_impl_owner_ref", None)
        object.__setattr__(self, "_entity", [])

        info = (
            None
            if info is None
            else as_str(
                info,
                name="extra information of the RegistryBase instance",
                replace=None,
            )
        )
        object.__setattr__(self, "raw_info", info)

    def act_bind_relation_base(
        self,
        name: str,
        target,
        *,
        is_weak: bool = True,
        is_replace: bool = True,
    ):
        name = as_str(name, name="Relation name for RegistryBase")
        if name != "owner":
            raise AttributeError(
                f"RegistryBase only supports relation {name!r} through its lightweight relation interface."
            )

        current_owner = self.owner
        if current_owner is not None and current_owner is not target and (not is_replace):
            raise RuntimeError(f"Relation {name!r} of RegistryBase is already bound.")

        object.__setattr__(
            self,
            "_impl_owner_ref",
            weakref.ref(target) if (is_weak and target is not None) else target,
        )
        return target

    def act_unbind_relation_base(self, name: str):
        name = as_str(name, name="Relation name for RegistryBase")
        if name == "owner":
            object.__setattr__(self, "_impl_owner_ref", None)

    def _helper_show_name_info(self) -> str:
        info = getattr(self, "raw_info", None)
        if info:
            return f"Registry {self.name!r} ({info})"
        return f"Registry {self.name!r}"

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_name(self, name: str, logger=None):

        name_set = set([item.name for item in self._entity])
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

    @logging_and_warning_decorator(start_finish_level=5)
    def act_register(self, term, is_contain_ok=False, logger=None):
        if term in self._entity:
            if not is_contain_ok:
                log_caught_exception(
                    logger,
                    ValueError(
                        f"term {term!r} is already registered in {self._helper_show_name_info()}"
                    ),
                    exception_msg="Check input.",
                    recovery_msg="Ignore this process.",
                )
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

        bind_relation = getattr(term, "act_bind_relation_base", None)
        if callable(bind_relation):
            bind_relation("registry", self, is_weak=True)
        else:
            try:
                object.__setattr__(term, "_impl_registry_ref", weakref.ref(self))
            except Exception as e:
                logger.warning(
                    f"Failed to assign registry relation for {term!r}: {e}. "
                    "This registration is one-way only."
                )

    @logging_and_warning_decorator(start_finish_level=5)
    def act_unregister(self, term, is_missing_ok=False, logger=None):
        if term not in self._entity:
            if not is_missing_ok:
                log_caught_exception(
                    logger,
                    KeyError(
                        f"term {term!r} is not registered in {self._helper_show_name_info()}"
                    ),
                    exception_msg="Check input.",
                    recovery_msg="Ignore this process.",
                )
            return

        self._entity.remove(term)

        registry = getattr(term, "registry", None)
        if registry is self:
            unbind_relation = getattr(term, "act_unbind_relation_base", None)
            if callable(unbind_relation):
                unbind_relation("registry")
            else:
                ref = getattr(term, "_impl_registry_ref", None)
                registry_ref = ref() if callable(ref) else None
                if registry_ref is self:
                    object.__setattr__(term, "_impl_registry_ref", None)

    @property
    def name(self):
        return self.raw_name

    @name.setter
    def name(self, value):
        name = as_str(value, name="The name of the Registry")
        object.__setattr__(self, "raw_name", name)

    @property
    def owner(self):
        ref = self._impl_owner_ref
        return ref() if callable(ref) else ref

    @property
    def _impl_owner(self):
        return self.owner

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
        elif isinstance(key, int):
            return self._entity[key]
        elif isinstance(key, str):
            for obj in self._entity:
                if obj.name == key:
                    return obj
            raise KeyError(
                f"No object with name '{key}' found in {self._helper_show_name_info()}."
            )
        else:
            raise TypeError(
                f"`key` must be str or int for {self._helper_show_name_info()} indexing, "
                f"got {type(key).__name__} instead."
            )

    def _helper_repr_by_order(self, is_name=True) -> str:

        if not self._entity:
            return "<empty registry>"

        if is_name:
            names = [obj.name for obj in self._entity]
        else:
            names = [str(obj) for obj in self._entity]

        # Width control
        idx_width = len(str(len(names) - 1))
        name_width = max(len(n) for n in names)

        lines = []
        for i, name in enumerate(names):
            lines.append(f"{i:>{idx_width}d}:       {name:<{name_width}}")

        return "\n".join(lines)

    def _helper_repr_by_category(self, is_name=False) -> str:

        if not self._entity:
            return "<empty registry>"

        # --- collect (category, name) pairs, preserving order ---
        records = []
        for obj in self._entity:
            if is_name:
                name = obj.name
            else:
                name = str(obj)
            category = getattr(obj, "raw_category", type(obj).__name__)
            records.append((category, name))

        # --- group while preserving category order ---
        grouped: dict[str, list[str]] = {}
        for category, name in records:
            grouped.setdefault(category, []).append(name)

        # --- width control ---
        cat_width = max(len(cat) for cat in grouped.keys())

        lines: list[str] = []
        for category, names in grouped.items():
            for i, name in enumerate(names):
                if i == 0:
                    lines.append(f"{category:<{cat_width}}:    {name}")
                else:
                    lines.append(f"{'':<{cat_width}}     {name}")

        return "\n".join(lines)

    def __repr__(self):
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})\n"
        return msg + self._helper_repr_by_category()
