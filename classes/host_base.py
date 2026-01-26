from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Type, Sequence
import weakref
from contextlib import contextmanager
import numpy as np

from ..logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import Unset, UNSET, as_str
from Nematics3D.general import pop_exclusive
from .opts import merge_opts_all, build_dict_override


@dataclass(slots=True, repr=False)
class OptsBase:
    tag: str | Unset = UNSET

    _internal_owner_ref: weakref.ReferenceType | None = field(
        default=None, repr=False, init=False
    )
    _state_is_functioning: bool = field(default=False, init=False, repr=False)
    _internal_sync_func: dict[str, Sequence[Callable]] = field(default_factory=dict, init=False, repr=False)

    __descriptions__: ClassVar[Mapping[str, str]] = {
        "tag":                  "name identifier of the option settings",
        }
    
    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        "tag":                  lambda v, d: as_str(v, name=d)
        }
    
    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        "tag":                  "options"
        })
    
    
    def __post_init__(self):
        self._internal_sync_func = {k: [] for k in self.__descriptions__.keys()}

    # ---------------------------------------------------------------------
    # Basic core: assignment with validation + lifecycle rule + owner commit
    # ---------------------------------------------------------------------
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_setattr_basic(self, key: str, value: Any, *, logger=None) -> Any:

        is_final = bool(getattr(self, "_state_is_functioning", False)) and getattr(self, "_internal_owner_ref", None) is not None

        # --- setting UNSET after functioning is forbidden ---
        if value is UNSET:
            if is_final:
                try:
                    raise TypeError(
                        "Attribute could not be set as UNSET after first functioning!"
                    )
                except TypeError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this modification")
                return value  # ignored
            object.__setattr__(self, key, value)
            return value

        # --- validate if needed ---
        if key in self.__class__._validators:
            desc = f"{key!r}: {self.__class__.__descriptions__[key]}"
            try:
                value2 = self.__class__._validators[key](value, desc)
                value = value2
            except Exception:
                logger.exception(f"Assignment to {key!r} failed")
                if is_final:
                    logger.recovery("Automatically ignore this modification")
                    return value  # ignored
                else:
                    logger.recovery("Reset this assignment to UNSET.")
                    object.__setattr__(self, key, UNSET)
                    value = UNSET
        else:
            if key.startswith("_") or not is_final:
                object.__setattr__(self, key, value)
                return value
                

        # --- owner commit (only after functioning) ---
        if not key.startswith("_") and is_final:
            self._helper_sync(key, value)
            return value
                
        object.__setattr__(self, key, value)
        return value


    def _helper_sync(self, key, value):
        self._helper_owner_apply(key, value)
        sync_func = self._internal_sync_func.get(key, None)
        for func in sync_func:
            func()
        
    def _helper_owner_apply(self, key, value):
        owner = self._internal_owner_ref()
        if owner is not None:
            owner._helper_commit_apply(**{key: value})
            return value

    # ---------------------------------------------------------------------
    # Basic core: finalize (fill UNSET by defaults then freeze state)
    # ---------------------------------------------------------------------
    def _helper_finalize_basic(self, 
                               defaults: Mapping[str, Any] | None = None,
                               is_allow_UNSET=False) -> None:

        if getattr(self, "_state_is_functioning", False):
            raise RuntimeError("This Opts has already been finalized.")

        defaults_dict = {} if defaults is None else dict(defaults)

        for k in self.__descriptions__.keys():
            if getattr(self, k) is UNSET:
                v = defaults_dict.get(k, self.__class__._DEFAULTS_FROZEN.get(k, UNSET))
                if v is UNSET and not is_allow_UNSET:
                    raise KeyError(f"Missing default for field {k!r}.")
                setattr(self, k, v)

        object.__setattr__(self, "_state_is_functioning", True)

    # ---------------------------------------------------------------------
    # Basic core: export to dict
    # ---------------------------------------------------------------------
    def _helper_asdict_basic(self, *, is_include_UNSET: bool = False) -> dict[str, Any]:

        result: dict[str, Any] = {}
        for k in self.__class__.__descriptions__.keys():
            v = getattr(self, k)
            if (not is_include_UNSET) and (v is UNSET):
                continue
            result[k] = v
        return result

    
    @contextmanager
    def _helper_internal_update(self):
        state_current = getattr(self, '_state_is_functioning', False)
        object.__setattr__(self, "_state_is_functioning", False)
        try:
            yield
        finally:
            object.__setattr__(self, "_state_is_functioning", state_current)
            
            

    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)
            
    def act_finalize(self, 
                     defaults: Mapping[str, Any] | None = None,
                     is_allow_UNSET=False):
        self._helper_finalize_basic(defaults, is_allow_UNSET=is_allow_UNSET)
        
    def act_asdict(self, is_include_UNSET=False):
        return self._helper_asdict_basic(is_include_UNSET=is_include_UNSET)
    

    def __repr__(self) -> str:
        cls = type(self)
        cls_name = cls.__name__

        # --- header line ---
        owner = getattr(self, "_internal_owner_ref", None)
        owner = owner() if owner else None
        if owner:
            lines = [f"{cls_name}: the options of {owner!r}"]
        else:
            lines = [f"{cls_name}"]
            

        # --- collect fields ---
        keys = list(cls.__descriptions__.keys())
        if not keys:
            return "\n".join(lines)

        width = max(len(k) for k in keys)

        for k in keys:
            try:
                v = getattr(self, k)
            except AttributeError:
                v = "<missing>"
            lines.append(f"  {k:<{width}} = {self._repr_format(v)}")

        return "\n".join(lines)
    
    @staticmethod
    def _repr_format(v):
        if isinstance(v, np.generic):
            v = v.item()

        if isinstance(v, float):
            return f"{v:.2g}"

        if isinstance(v, np.ndarray):
            if v.size > 6:
                return f"<ndarray shape={v.shape}, too many elements to display>"
            else:
                return repr(v)

        return repr(v)

            

class HostBase:
    
    __descriptions__: ClassVar[Mapping[str, str]] = {
        "raw_name":                 "The name identifier of the host object",
        
        "opts":                     "The Opts instance controlling options.",
        "opts_defaults":            "The default option settings.",
        
        "_internal_owner_ref":      ("A weak reference to the owner object associated with this instance."
                                     "To access it, use .owner or ._internal_owner."),
        "_internal_extra_attrs":    (
            "A dict storing user-registered extra attributes. "
            "These are accessed via `glyph.<name>` after calling `act_add_attr(name, doc)`."
        ),
        "_internal_extra_attrs_docs": (
            "A dict storing docstrings for user-registered extra attributes."
        ),
        }
    
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        opts_type: Type[OptsBase],
        opts: OptsBase | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        name : str | None = None,
        name_replace: str = "unnamed",
        logger = None,
        **kwargs
            ):    
        
        self.__descriptions__["raw_name"] = f"The name identifier of the {type(self).__name__} instance"
        
        logger.detail("Dealing with basic attributes and input")
        object.__setattr__(self, "_internal_extra_attrs", {})
        object.__setattr__(self, "_internal_extra_attrs_docs", {})
        object.__setattr__(self, "_internal_owner_ref", None)
        
        opts = self._helper_check_opts(opts, opts_type=opts_type)
                
        logger.detail('Handling explicit kwargs overrides ...')
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_internal_owner_ref", weakref.ref(self))
        object.__setattr__(self, "opts", opts)

        
        logger.detail("Building default option values ...")
        opts_defaults = build_dict_override(
                            opts._DEFAULTS_FROZEN,
                            opts_defaults_override,
                            name=type(opts).__name__,
                        )
        object.__setattr__(self, "opts_defaults", opts_defaults)
        
        name = as_str(name, name=self.__descriptions__["raw_name"], replace=name_replace) if name else name_replace
        self.act_set_name(name)
        
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_opts(self, opts, opts_type=None, logger=None):
        
        if not opts_type:
            opts_type = self.opts.__class__
        
        if opts is None:
            opts = opts_type()
        elif not isinstance(opts, opts_type):
            try:
                raise TypeError(
                        f"opts must be an instance of {opts_type.__name__}, "
                        f"got {type(opts).__name__}"
                    )
            except TypeError:
                logger.exception("Check input.")
                logger.recovery(f"Create a default instance of {opts_type.__name__} instead.")
                opts = opts_type()
                
        return opts
        

    @property
    def _internal_owner(self):
        ref = self._internal_owner_ref
        return ref() if ref is not None else None
    
    @property
    def owner(self):
        ref = self._internal_owner_ref
        return ref() if ref is not None else None
        
    @property
    def name(self):
        return self.raw_name
    
    @name.setter
    def name(self, value: str):
        self.act_set_name(value)
        
    @logging_and_warning_decorator(start_finish_level=5)
    def act_set_name(self, name, logger=None):
        
        try:
            name = as_str(name, name=self.__descriptions__["raw_name"])
        except:
            logger.exception("Invalid name.")
            logger.recovery("Ignore this modification.")
            return
        
        check_name = getattr(self.owner, "_helper_check_name", None) if self.owner else None
        if callable(check_name):
            name = check_name(name)
        object.__setattr__(self, "raw_name", name)
            
            
            
    def __getattr__(self, key):
        extra = object.__getattribute__(self, "_internal_extra_attrs")
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

        docs = self._internal_extra_attrs_docs
        data = self._internal_extra_attrs

        if (name in docs) and (not overwrite):
            raise KeyError(
                f"Extra attribute {name!r} is already registered. Use overwrite=True to override."
            )

        docs[name] = doc
        if overwrite or (name not in data):
            data[name] = default
            

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_setattr_basic(self, key, value, allowed_extra=[], logger=None):
    
        allowed_core = list(allowed_extra) + ["name", "raw_name"]
        
        extra = object.__getattribute__(self, "_internal_extra_attrs")
        docs = object.__getattribute__(self, "_internal_extra_attrs_docs")
        if key in docs:
            extra[key] = value
            return
    
        if key not in allowed_core:
            raise AttributeError(
                f"Invalid attribute assignment: {key!r}. "
                "Only attributes in {allowed_core} can be modified directly, "
                f"or a registered extra attribute."
            )

        self.act_commit(**{key: value})
            
    
    def _helper_merge_opts_kwargs(self, opts=None, **kwargs):
        opts = self._helper_check_opts(opts)
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        kwargs = opts.act_asdict()
        if 'tag' in kwargs.keys():
            object.__setattr__(self.opts, 'tag', kwargs['tag'])
            kwargs.pop('tag')
        return kwargs
    

    def _helper_commit_pre_opts(self, **kwargs):
        found, name, kwargs = pop_exclusive(kwargs, "name", "raw_name")
        if found:
            self.act_set_name(name)
        return kwargs
    
    
    def act_commit(self, opts=None, **kwargs):
        kwargs = self._helper_commit_pre_opts(**kwargs)
        kwargs = self._helper_merge_opts_kwargs(opts=opts, **kwargs)
        self._helper_commit_apply(**kwargs)
    
    def _helper_commit_apply(self, **kwargs):
        raise NotImplementedError(...)
        
        
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})"
        return msg 

