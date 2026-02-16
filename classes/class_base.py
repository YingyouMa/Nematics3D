from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import as_str


class ClassBase:
    """
    A foundational base class providing structured identity, hierarchy management,
    and strict attribute control.

    ### Main Features:
    * **Identity & Hierarchy**: Manages the object's name and its relationships
        with an 'Owner' (parent object) and a 'Registry' (container). It supports
        automatic name conflict resolution via the Registry's helper methods.
    * **Attribute Control**: Uses ``__slots__`` to optimize memory and prevent
        accidental assignment of undefined variables.
    * **Dynamic Extension**: Supports 'Human-added' attributes via ``act_add_attr``.

    ### Variables & Descriptions:
    All attributes, properties, and internal implementations are documented
    in the ``__descriptions__`` dictionary. Please refer to it for detailed
    per-variable metadata.

    ### Inheritance Guidelines:
    1.  **Unique Relationships**: An instance can have at most **one owner** and
        be registered in **one registry** at any given time.
    2.  **Metadata Expansion**: When inheriting, subclasses should update or
        extend the ``__descriptions__`` dictionary to reflect new attributes.
        It is highly recommended to re-include ``raw_name`` in the subclass's 
        descriptions to ensure consistent behavior for string representation.
    3.  **Slot Maintenance**: To preserve memory efficiency and the strict
        assignment policy, subclasses **MUST** define ``__slots__`` (or include
        new fields) to prevent the accidental creation of a ``__dict__``.
    """

    __descriptions__ = {
        "name": (
            "Property: The display name of the instance. "
            "Returns 'raw_name' and can be updated via the setter."
        ),
        "owner": (
            "Property: The object that owns this instance. "
            "An instance can belong to at most one owner at a time."
        ),
        "registry": (
            "Property: The Registry object where this instance is registered. "
            "An instance can belong to at most one registry at a time."
        ),
        # Internal Attributes
        "raw_name": "The underlying string identifier for this instance.",
        "_impl_owner_ref": (
            "A weak reference to the owner object. "
            "Use the 'owner' property for safe access."
        ),
        "_impl_registry_ref": (
            "A weak reference to the associated Registry. "
            "Use the 'registry' property for safe access."
        ),
        "_impl_extra_attrs": (
            "A dictionary storing dynamic user-defined attributes. "
            "Managed via 'act_add_attr'."
        ),
        "_impl_extra_attrs_docs": "Documentation strings for user-defined extra attributes.",
    }

    __slots__ = tuple(
        k for k, v in __descriptions__.items() if not v.startswith("Property:")
    ) + ("__weakref__",)

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(self, *, name: str, name_replace: str, logger=None):

        logger.detail("Dealing with basic attributes and input")
        if not hasattr(self, "_impl_extra_attrs"):
            object.__setattr__(self, "_impl_extra_attrs", {})
        if not hasattr(self, "_impl_extra_attrs_docs"):
            object.__setattr__(self, "_impl_extra_attrs_docs", {})
        if not hasattr(self, "_impl_owner_ref"):
            object.__setattr__(self, "_impl_owner_ref", None)
        if not hasattr(self, "_impl_registry_ref"):
            object.__setattr__(self, "_impl_registry_ref", None)

        name = (
            as_str(name, name=self.__descriptions__["raw_name"], replace=name_replace)
            if name
            else name_replace
        )
        self.act_set_name(name)

    @property
    def owner(self):
        ref = self._impl_owner_ref
        return ref() if ref is not None else None

    _impl_owner = owner

    @property
    def registry(self):
        ref = self._impl_registry_ref
        return ref() if ref is not None else None

    _impl_registry = registry

    @property
    def name(self):
        return self.raw_name

    @name.setter
    def name(self, value: str):
        self.act_set_name(value)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_set_name(self, name, logger=None):

        logger.detail(f"Set name requested: {name!r}")

        try:
            name = as_str(name, name=self.__descriptions__["raw_name"])
        except (TypeError, ValueError):
            logger.exception("Invalid name.")
            logger.recovery("Ignore this modification.")
            return

        check_name = (
            getattr(self.registry, "_helper_check_name", None)
            if self.registry
            else None
        )
        if callable(check_name):
            logger.detail(
                "The registry provides _helper_check_name; resolving name conflict."
            )
            name = check_name(name)
        object.__setattr__(self, "raw_name", name)

        return name

    def __getattr__(self, key):
        extra = object.__getattribute__(self, "_impl_extra_attrs")
        if key in extra:
            return extra[key]
        else:
            raise AttributeError(
                f"{type(self).__name__!s} object has no attribute {key!r}."
            )

    def act_add_attr(
        self,
        name: str,
        doc: str,
        default=None,
        overwrite: bool = False,
    ):

        name = as_str(name, name=f"Extra attribute name for instance {self.name!r}")
        doc = as_str(doc, name=f"Extra attribute doc for instance {self.name!r}")

        if not name.isidentifier():
            raise ValueError(
                f"Invalid extra attribute name {name!r}: must be a valid Python identifier."
            )

        if hasattr(type(self), name) or (
            name in getattr(type(self), "__descriptions__", ())
        ):
            raise AttributeError(
                f"Cannot register extra attribute {name!r}: it conflicts with an existing attribute of {type(self).__name__}."
            )

        docs = self._impl_extra_attrs_docs
        data = self._impl_extra_attrs

        if (name in docs) and (not overwrite):
            raise KeyError(
                f"Extra attribute {name!r} is already registered. Use overwrite=True to override."
            )

        docs[name] = doc
        if overwrite or (name not in data):
            data[name] = default

    def _helper_setattr_basic(self, key, value, allowed_extra=None):

        if allowed_extra is None:
            allowed_extra = []
        allowed_core = list(allowed_extra) + ["name", "raw_name"]

        extra = object.__getattribute__(self, "_impl_extra_attrs")
        docs = object.__getattribute__(self, "_impl_extra_attrs_docs")
        if key in docs:
            extra[key] = value
            return

        if key not in allowed_core:
            raise AttributeError(
                f"Invalid attribute assignment: {key!r}. "
                f"Only attributes in {allowed_core} can be modified directly, "
                f"or a registered extra attribute."
            )

        object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})"
        return msg
