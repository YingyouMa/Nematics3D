from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import as_str

class ClassBase:
    
    __descriptions__ = {
        "raw_name":                 "The name of the ClassBase instance.",
        
        "_impl_owner_ref":      (
            "A weak reference to the owner object associated with this instance. "
            "To access it, use .owner or ._impl_owner."
        ),
        "_impl_registry_ref":   (
            "A weak reference to the Registry that this object is currently registered in. "
            "Each object is expected to be associated with at most one registry at a time."
        ),
        
        "_impl_extra_attrs":    (
            "A dict storing user-registered extra attributes. "
            "These are accessed via `glyph.<name>` after calling `act_add_attr(name, doc)`."
        ),
        "_impl_extra_attrs_docs": "A dict storing docstrings for user-registered extra attributes.",
        }
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        *,
        name: str,
        name_replace: str,
        logger=None
            ):
        
        logger.detail("Dealing with basic attributes and input")
        if not hasattr(self, "_impl_extra_attrs"):
            object.__setattr__(self, "_impl_extra_attrs", {})
        if not hasattr(self, "_impl_extra_attrs_docs"):
            object.__setattr__(self, "_impl_extra_attrs_docs", {})
        if not hasattr(self, "_impl_owner_ref"):
            object.__setattr__(self, "_impl_owner_ref", None)
        if not hasattr(self, "_impl_registry_ref"):
            object.__setattr__(self, "_impl_registry_ref", None)
        if not hasattr(self, "_impl_opts_backup"):
            object.__setattr__(self, "_impl_opts_backup", {})
            
        name = as_str(name, name=self.__descriptions__["raw_name"], replace=name_replace) if name else name_replace
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
        except:
            logger.exception("Invalid name.")
            logger.recovery("Ignore this modification.")
            return
        
        check_name = getattr(self.owner, "_helper_check_name", None) if self.owner else None
        if callable(check_name):
            logger.detail("Owner provides _helper_check_name; resolving name conflict.")
            name = check_name(name)
        object.__setattr__(self, "raw_name", name)

        return name
    
    
    def __getattr__(self, key):
        extra = object.__getattribute__(self, "_impl_extra_attrs")
        if key in extra:
            return extra[key]
        else:
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {key!r}.")
            
            
    def act_add_attr(
        self,
        name: str,
        doc: str,
        default=None,
        overwrite: bool = False,
    ):

        name = as_str(name, name='Extra attribute name for PlotGlyph')
        doc = as_str(doc, name='Extra attribute doc for PlotGlyph')


        if not name.isidentifier():
            raise ValueError(f"Invalid extra attribute name {name!r}: must be a valid Python identifier.")

        if hasattr(type(self), name) or (name in getattr(type(self), "__slots__", ())):
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
        
        
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})"
        return msg 