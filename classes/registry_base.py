import weakref

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import as_str


class RegistryBase:

    # fmt: off
    __descriptions__ = {
        "raw_name":         "The name of the Registry.",
        "_raw_info":        "The extra introduction for this instance for clarity",
        "_entity":          "The container storing the objects.",

        "_impl_owner_ref":  (
            "A weak reference to the owner object associated with this instance. "
            "To access it, use .owner or ._impl_owner."
        ),
    }
    # fmt: on

    def __init__(self, name, info=None):

        name = as_str(name, name="The name of the Registry")
        object.__setattr__(self, "raw_name", name)
        object.__setattr__(self, "_impl_owner_ref", None)
        object.__setattr__(self, "_entity", [])
        
        if info is not None:
            info = as_str(info, name="extra information of the RegistryBase instance", replace=None)
        object.__setattr__(self, "_raw_info", None)

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
                f"{name_input!r} already exists in Registry {self.name!r}! Renamed to {new_name!r}."
            )
            name = new_name
        return name

    @logging_and_warning_decorator(start_finish_level=5)
    def act_register(self, term, is_contain_ok=False, logger=None):

        logger.detail(f"Register term into Registry {self.name!r}: term={term!r}")

        if term in self._entity:
            if not is_contain_ok:
                try:
                    raise ValueError(
                        f"term {term!r} is already registered in Registry {self.name!r}"
                    )
                except ValueError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this process.")
            return

        if not hasattr(term, "name"):
            raise TypeError("term must have attribute `name`.")
        name = self._helper_check_name(term.name)
        term.name = name
        self._entity.append(term)

        old_ref = getattr(term, "_impl_registry_ref", None)
        old_registry = old_ref() if callable(old_ref) else None
        if old_registry is not None and old_registry is not self:
            logger.warning(
                f"{term!r} already has a registry {old_registry!r}. Overwrite registry to {self!r}."
            )
        object.__setattr__(term, "_impl_registry_ref", weakref.ref(self))

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
        return ref() if ref is not None else None

    _impl_owner = owner

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
                f"No object with name '{key}' found in Registry {self.name!r}."
            )
        else:
            raise TypeError(
                f"`key` must be str or int for Registry {self.name!r} indexing, "
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
