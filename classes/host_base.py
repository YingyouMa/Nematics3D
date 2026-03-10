from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Type, Sequence
import weakref
from contextlib import contextmanager
import numpy as np
import datetime
from copy import deepcopy


from ..logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import Unset, UNSET, as_str
from Nematics3D.general import pop_exclusive
from .opts import merge_opts_all, build_dict_override, diff_dict_values
from .class_base import ClassBase


@dataclass(slots=True, repr=False)
class OptsBase:
    """
    A reactive configuration base class designed for pre-processing,
    validation, and synchronized state management.

    ### Configuration Workflow:
    1.  **Validation Layer**: Each attribute assignment is pre-checked by
        ``_validators``. This ensures that only data meeting specific type or
        value constraints enters the system.
    2.  **UNSET & Finalization**: Attributes are initialized as ``UNSET`` by
        default if no input is provided. During the ``act_finalize`` phase,
        all ``UNSET`` fields are automatically populated using a hierarchy of
        default values (instance-level overrides -> class-level defaults).
    3.  **Lifecycle & Commitment**:
        * Once finalized, the instance enters the ``is_functioning`` state as
          self._state_is_functioning = True
        * Setting an attribute to ``UNSET`` is strictly forbidden after finalization.
        * Any subsequent modification to public attributes will be treated as a
          request and forwarded to the associated Host via the commit pipeline.
    4.  **Data Export**: The current state of all non-hidden attributes can
        be retrieved as a standard dictionary via ``act_asdict()``.
    """

    tag: str | Unset = UNSET

    _impl_host_ref: weakref.ReferenceType | None = field(
        default=None, repr=False, init=False
    )
    _state_is_functioning: bool = field(default=False, init=False, repr=False)

    __descriptions__: ClassVar[Mapping[str, str]] = {
        "tag": "name identifier of the option settings",
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        "tag": lambda v, d: as_str(v, name=d)
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({"tag": "options"})

    @property
    def host(self):
        ref = self._impl_host_ref
        return ref() if ref is not None else None

    # ---------------------------------------------------------------------
    # Basic core: assignment with validation + lifecycle rule + host commit
    # ---------------------------------------------------------------------
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_setattr_basic(self, key: str, value: Any, *, logger=None) -> Any:

        is_final = bool(getattr(self, "_state_is_functioning", False)) and (
            self.host is not None
        )

        if not key.startswith("_") and key not in self.__class__.__descriptions__:
            raise AttributeError(
                f"Invalid option field {key!r}. "
                f"Valid fields are: {list(self.__class__.__descriptions__.keys())}"
            )

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
                return value
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

        # --- host commit (only after functioning) ---
        if not key.startswith("_") and is_final and key in self.__class__.__descriptions__:
            self._helper_host_apply(key, value)
            return value

        object.__setattr__(self, key, value)
        return value

    def _helper_host_apply(self, key, value):
        if self.host is not None:
            self.host._helper_commit_apply_opts(**{key: value})
            return value

    # ---------------------------------------------------------------------
    # Basic core: finalize (fill UNSET by defaults then freeze state)
    # ---------------------------------------------------------------------
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_finalize_basic(
        self,
        defaults: Mapping[str, Any] | None = None,
        is_allow_UNSET=False,
        logger=None,
    ) -> None:

        if getattr(self, "_state_is_functioning", False):
            raise RuntimeError("This Opts has already been finalized.")

        logger.detail("finalize: fill UNSET fields with defaults")

        defaults_dict = {} if defaults is None else dict(defaults)

        for k in self.__descriptions__.keys():
            if getattr(self, k) is UNSET:
                v = defaults_dict.get(k, self.__class__._DEFAULTS_FROZEN.get(k, UNSET))
                if v is UNSET and not is_allow_UNSET:
                    raise KeyError(f"Missing default for field {k!r}.")
                setattr(self, k, v)
                logger.detail(f"finalize: set default {k!r}={v!r}")

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
        state_current = getattr(self, "_state_is_functioning", False)
        object.__setattr__(self, "_state_is_functioning", False)
        try:
            yield
        finally:
            object.__setattr__(self, "_state_is_functioning", state_current)

    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)

    def act_finalize(
        self, defaults: Mapping[str, Any] | None = None, is_allow_UNSET=False
    ):
        self._helper_finalize_basic(defaults, is_allow_UNSET=is_allow_UNSET)

    def act_asdict(self, is_include_UNSET=False):
        return self._helper_asdict_basic(is_include_UNSET=is_include_UNSET)

    def __repr__(self) -> str:
        cls = type(self)
        cls_name = cls.__name__

        # --- header line ---
        host = self.host
        if host:
            lines = [f"{cls_name}: the options of {str(host)}"]
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


class HostBase(ClassBase):
    """
    A high-level controller class that manages complex state through a
    'Opts' configuration layer and a strict commit-based update pipeline.

    ### 1. Centralized Data Storage (.opts)
    All critical parameters and functional settings are stored exclusively within
    the ``.opts`` attribute, which is an instance of ``OptsBase`` (or its subclass).
    The Host instance itself does not hold primary state variables; instead, it
    acts as the logic engine that governs and applies the configuration held
    by the Opts instance.

    ### 2. Host-Opts Interaction Semantics
    This class operates on a "Request-Commit-Apply" model. Instead of
    direct mutation, the Host delegates its public configuration to an
    associated ``OptsBase`` instance.
    * **State Isolation**: Before 'finalization', Opts acts as a buffer.
        Once finalized (functioning state), any change to Opts triggers a
        request back to the Host.
    * **The Commit Pipeline**: All public attribute assignments on the Host
        are intercepted and routed through ``act_commit()``. This ensures that
        changes undergo validation, preprocessing, and side-effect management
        (e.g., hardware updates, cache invalidation) before state realization.
    * **Write-Back Policy**: When a commit is accepted, the Host is responsible
        for updating its internal state and writing resolved values back to
        the Opts instance via bypass methods to avoid recursive loops.

    ### 2. Core Functional Modules
    * **Identity Management**: Inherits robust naming and conflict resolution
        from ``ClassBase``.
    * **Option Lifecycle**: Manages the binding, override merging, and
        finalization of configuration options (Opts).
    * **State Snapshots**: Provides a timestamped backup mechanism
        (``_opts_backup``) to archive configuration history.

    ### 3. Variables & Metadata
    Refer to the ``__descriptions__`` dictionary for granular details on
    internal implementation slots and public properties.
    Key internal stores include:
    * ``opts``: The primary configuration engine.
    * ``_opts_defaults``: The baseline configuration used during finalization.
    * ``_opts_backup``: Historical archive of previous option states.

    ### 4. Inheritance Guidelines
    * HostBase.__init__ only performs minimal wiring (opts binding,
        defaults construction, and name initialization). Concrete host
        subclasses are responsible for:
          - finalizing opts at the appropriate lifecycle stage, and
          - defining how finalized opts are consumed and applied.
    * The kwargs received by HostBase._helper_commit_apply_opts(...) are
        guaranteed to have passed all opts-level preprocessing and basic
        validation. Host implementations may assume that input values are
        already sanitized, and therefore should not repeat opts-level
        validation. Host-side logic should focus on state-dependent or
        cross-field constraints and side effects.
    * When a host accepts an update in _helper_commit_apply_opts(...), it
        MUST write the resolved value back to opts. This write-back must bypass
        the normal opts assignment path (e.g. via object.__setattr__ or
        OptsBase._helper_internal_update) to avoid recursive commit loops.
        The host is also responsible for calling self._impl_sync_func() to update
        all downstream listeners by the resolved value.
    * Other inheritance guidelines of ClassBase class.
    """

    __descriptions__ = {
        **(ClassBase.__descriptions__),
        "raw_name": "The name identifier of the host object",
        "opts": "The Opts instance controlling options.",
        "_opts_defaults": "The default option settings.",
        "_opts_backup": (
            "A dictionary storing potentially useful options, indexed by timestamp."
            "Key: Current time, or manualy set value; Value: A dictionary of options (opts)."
        ),
        "_impl_sync_func": (
                    "A dictionary of callback functions for post-commit synchronization. "
                    "Key: unique identifier (str); Value: callable task(host, **kwargs)."
        ),
        "_impl_attrs_wrapped": (
            "Protected attributes under wrapping. When wrapped, these attributes cannot be modified "
            "unless within _helper_wrapped_update() context."
        ),
        "_impl_wrapper_ref": "A weak reference to the wrapper object that controls this host. ",
        "_entity_wrapped": "The host object being wrapped and controlled by this wrapper.",
        "_impl_enrich_kwargs_wrapped_func": (
                    "A dictionary of callback functions to enrich kwargs before forwarding to wrapped host. "
                    "Key: unique identifier (str); Value: callable task(host, kwargs, kwargs_sync)."
        ),
        "_impl_enrich_kwargs_sync_func": (
                    "A dictionary of callback functions to enrich kwargs_sync before sync task execution. "
                    "Key: unique identifier (str); Value: callable task(host, kwargs_sync)."
        ),
    }

    __slots__ = tuple(
        k
        for k, v in __descriptions__.items()
        if not v.startswith("Property:") and k not in ClassBase.__slots__
    )
    
    _impl_validators = {}
    # Validator keys correspond to the public name (without the ``raw_`` prefix).
    # For example, ``raw_coords`` will use the validator registered under ``coords``.
    # The validator must accept two arguments: (value, description).
    _impl_attrs_reapply_opts_after_raw: set()
    # Public attribute names (without "raw_") that should force an opts re-apply
    # after raw/public assignment in a commit even if no explicit opts update is provided.

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        opts_type: Type[OptsBase],
        opts: OptsBase | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        name: str | None = None,
        name_replace: str = "unnamed",
        logger=None,
        **kwargs,
    ):

        super().__init__(name=name, name_replace=name_replace)

        logger.detail("Handling explicit kwargs overrides ...")
        opts = self._helper_check_opts(opts, opts_type=opts_type)
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_impl_host_ref", weakref.ref(self))
        object.__setattr__(self, "opts", opts)

        logger.detail("Building default option values ...")
        opts_defaults = {
            **{k: UNSET for k in opts.__descriptions__},
            **dict(opts._DEFAULTS_FROZEN),
        }
        opts_defaults = build_dict_override(
            opts_defaults,
            opts_defaults_override,
            name=type(opts).__name__,
        )
        object.__setattr__(self, "_opts_defaults", opts_defaults)
        object.__setattr__(self, "_opts_backup", {})
        object.__setattr__(self, "_impl_sync_func", {})
        object.__setattr__(self, "_impl_enrich_kwargs_wrapped_func", {})
        object.__setattr__(self, "_impl_enrich_kwargs_sync_func", {})
        object.__setattr__(self, "_impl_attrs_wrapped", set())
        object.__setattr__(self, "_impl_wrapper_ref", None)
        object.__setattr__(self, "_entity_wrapped", None)
        
        # remaining tasks for __init__():
        # - finalizing opts at the appropriate lifecycle stage, and
        # - defining how finalized opts are consumed and applied.

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
                logger.recovery(
                    f"Create a default instance of {opts_type.__name__} instead."
                )
                opts = opts_type()

        return opts
    
    # -------------------------------
    # -------------------------------
    # The functions to commit update
    # -------------------------------
    # -------------------------------
    
    @logging_and_warning_decorator()
    def act_commit(self, opts=None, opts_wrapped=None, logger=None, **kwargs):
        
        # _helper_commit_pre_opts:  Preprocess kwargs for this host prior to opts-level processing,
        #                           especially handle mutable attributes that are managed outside the opts system.
        #       _helper_check_wrapped_attr:  remove attributes protected by the wrapper, so they cannot be modified
        #                                    directly from this level.
        #       _helper_commit_name: extract ``name`` / ``raw_name`` from ``kwargs``
        #                            and apply host naming updates immediately.
        #       _helper_commit_raw:  extract host-side raw/public attributes from
        #                            ``kwargs``, validate them if needed, and write
        #                            them directly to the host instead of routing
        #                            them through opts.
        #
        # _helper_commit_self: handle all updates that belong to this host itself.
        #       _helper_merge_opts_kwargs:   merge explicit ``opts`` input with the
        #                                   relevant entries in ``kwargs`` and convert
        #                                   them into a normalized dictionary of
        #                                   opts-level updates for this host.
        #       _helper_commit_apply_opts: apply the normalized updates to this host.
        #       _helper_check_wrapped_attr: perform one more protection check before
        #                                   the actual application step.
        #       _helper_commit_apply_opts_main: perform the real host-side state
        #                                       update, write resolved values back
        #                                       to ``self.opts`` by bypassing the
        #                                       normal opts assignment path, and
        #                                       remove any keys that fail or are
        #                                       intentionally consumed so they are
        #                                       not forwarded further.
        #       _helper_trigger_sync_batch: notify all registered downstream sync
        #                                   callbacks using the final successfully
        #                                   applied updates.
        # Remaining ``kwargs`` after the self-handling stage are treated as
        # updates intended for ``self.wrapped``. If a wrapped host exists, they
        # are forwarded by calling ``self.wrapped.act_commit(...)``. If no wrapped
        # host exists, they are treated as invalid leftover arguments.
        
        opts_keys = self.opts.__class__.__descriptions__
        is_opts_request = (opts is not None) or any(k in opts_keys for k in kwargs)
        
        kwargs_applied_raw = self._helper_commit_pre_opts(kwargs)
        
        is_reapply = kwargs["is_reapply"]
        if is_reapply or is_opts_request:
            kwargs, kwargs_applied_opts = self._helper_commit_self(
                opts=opts, 
                is_reapply=is_reapply,
                **kwargs
            )
            kwargs_sync = kwargs_applied_raw | kwargs_applied_opts
            
        kwargs_sync = self._helper_commit_enrich_kwargs_sync(kwargs_sync)
        if kwargs_sync:
            self._helper_trigger_sync_batch(**kwargs_sync)
        
        kwargs = self._helper_commit_enrich_kwargs_wrapped(kwargs, kwargs_sync=kwargs_sync)
        if kwargs or opts_wrapped:
            if self.wrapped is not None:
                self.wrapped.act_commit(opts=opts_wrapped, **kwargs)
            else:
                cls_name = self.__class__.__name__
                obj_name = getattr(self, "raw_name", "Uninitialized")
                logger.warning(f"[{cls_name}: {obj_name!r}] Invalid arguments: {list(kwargs.keys())}")
        
                
    # -----------------------           
    # _helper_commit_pre_opts
    # -----------------------
    def _helper_commit_pre_opts(self, kwargs):
        
        self._helper_check_wrapped_attr(kwargs)
        kwargs_applied_name = self._helper_commit_name(kwargs)
        kwargs_applied_raw = self._helper_commit_raw(kwargs)
        return kwargs_applied_raw | kwargs_applied_name
        # Any value modified by the wrapper must be written back to ``kwargs``
        # so the updated parameters are forwarded to the wrapped object.
        # For example, the wrapped object only accepts ``radius``, while the wrapper
        # expose a convenience parameter ``radius_scale``. If the wrapper receives 
        # ``radius_scale=2``, it should delete ``radius_scale`` and replace
        # ``radius`` with the doubled value (radius=2*radius) before forwarding
        # ``kwargs`` to the wrapped object.

    @logging_and_warning_decorator()
    def _helper_check_wrapped_attr(self, kwargs, logger=None):
        if not kwargs:
            return
        blocked = [k for k in kwargs.keys() if k in self._impl_attrs_wrapped]
        for key in blocked:
            kwargs.pop(key)
            try:
                raise AttributeError(
                    f"{key!r} is protected by self.wrapper and could not be directly modified"
                )
            except AttributeError:
                logger.exception("Invalid attr")
                logger.recovery("Automatically ignore this attr")
                
    def _helper_commit_name(self, kwargs):
        if not kwargs:
            return {}
        found, name = pop_exclusive(kwargs, "name", "raw_name")
        if found:
            self.act_set_name(name)
            return {"name": self.name}
        else:
            return {}
                
    def _helper_commit_raw(self, kwargs):
        if not kwargs:
            return {}
        kwargs_applied_raw = {}
        is_reapply_opts = False
        for key in list(kwargs.keys()):
            if key in self.__descriptions__ or ("raw_" + key) in self.__descriptions__:
                kwargs_applied_here = self._helper_commit_pop_raw(kwargs, key)
                kwargs_applied_raw, is_reapply_opts_here = kwargs_applied_raw | kwargs_applied_here
                is_reapply_opts = is_reapply_opts or is_reapply_opts_here
        if not kwargs.get("is_reapply_opts", False):
            kwargs["is_reapply_opts"] = is_reapply_opts
        return kwargs_applied_raw

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_pop_raw(
        self,
        kwargs: dict[str, Any],
        attr_name_origin: str,
        validator: Callable | None = None,
        exception_msg: str | None = None,
        recovery_msg: str | None = None,
        logger=None,
    ):
        
        if attr_name_origin.startswith('raw_'):
            raw_attr_name = attr_name_origin
            attr_name = attr_name_origin[4:]
        else:
            raw_attr_name = "raw_" + attr_name_origin
            attr_name = attr_name_origin
    
        found, attr_value = pop_exclusive(kwargs, attr_name, raw_attr_name)
        if not found:
            return {}, False
    
        if exception_msg is None:
            exception_msg = (
                f"Validation failed for attribute {attr_name!r}. "
                f"This may be due to an invalid value or an incorrectly implemented validator. "
                f"The validator must accept two arguments: (value, description)."
            )
        if recovery_msg is None:
            recovery_msg = f"Ignore this modification of {attr_name!r}."
        
        if validator is None:
            if attr_name in self._impl_validators:
                validator = self._impl_validators[attr_name]
            
        if validator is not None:
            try:
                value_valid = validator(
                    attr_value, 
                    self.__descriptions__[raw_attr_name]
                    
                )
                object.__setattr__(self, raw_attr_name, value_valid)
                return {attr_name: attr_value}, (
                    attr_name in self.__class__._impl_attrs_reapply_opts_after_raw
                )
        
            except Exception:
                logger.exception(exception_msg)
                logger.recovery(recovery_msg)
                return {}
        else:
            object.__setattr__(self, raw_attr_name, attr_value)
            return {attr_name: attr_value}, (
                attr_name in self.__class__._impl_attrs_reapply_opts_after_raw
            )
            
            
    # -----------------------           
    # _helper_commit_self
    # -----------------------
    def _helper_commit_self(self, opts=None, **kwargs):
        if kwargs or opts:
            self_keys = self.opts.__class__.__descriptions__
            kwargs_self = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in self_keys}
            kwargs_self = self._helper_merge_opts_kwargs(opts=opts, **kwargs_self)
            kwargs_self["is_reapply"] = kwargs.pop("is_reapply")
            kwargs_left, kwargs_applied_opts = self._helper_commit_apply_opts(**kwargs_self)
            kwargs = kwargs | kwargs_left
            return kwargs, kwargs_applied_opts

    def _helper_merge_opts_kwargs(self, opts=None, **kwargs):
        if kwargs or opts:
            opts = self._helper_check_opts(opts)
            opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
            return opts.act_asdict()
        else:
            return {}

        
    def _helper_commit_apply_opts(self, **kwargs):
        self._helper_check_wrapped_attr(kwargs)
        opts_before = self.opts.act_asdict()
        kwargs_applied_opts = {}
        if "tag" in kwargs:
            object.__setattr__(self.opts, "tag", kwargs["tag"])
            kwargs_applied_opts["tag"] =  kwargs["tag"]
            kwargs.pop("tag")
        return_main = self._helper_commit_apply_opts_main(**kwargs)
        if return_main is None:
            kwargs_left = {}
            opts_after = self.opts.act_asdict()
            _, kwargs_applied_opts_main = diff_dict_values(opts_before, opts_after)
        else:
            kwargs_left, kwargs_applied_opts_main = return_main
        kwargs_applied_opts = kwargs_applied_opts | kwargs_applied_opts_main
        return kwargs_left, kwargs_applied_opts
        # the input kwargs should only include the attributes in options
        
    def _helper_commit_apply_opts_main(self, **kwargs):
        is_reapply = kwargs.pop("is_reapply")
        raise NotImplementedError(...)
        # Any value modified by the wrapper must be written back to ``kwargs``
        # so the updated parameters are forwarded to the wrapped object.
        # For example, the wrapped object only accepts ``radius``, while the wrapper
        # expose a convenience parameter ``radius_scale``. If the wrapper receives 
        # ``radius_scale=2``, it should delete ``radius_scale`` and replace
        # ``radius`` with the doubled value (radius=2*radius) before forwarding
        # ``kwargs`` to the wrapped object.
        # This is the same with _helper_commit_pre_opts()
        # Additionally, if assigning a value from ``kwargs`` fails at this stage,
        # the corresponding key should also be removed from ``kwargs`` so it is not
        # forwarded to the wrapped object or the sync func.
        
    @logging_and_warning_decorator()
    def _helper_trigger_sync_batch(self, logger=None, **kwargs):
        for name, func in self._impl_sync_func.items():
            try:
                func(host=self, **kwargs)
            except Exception as e:
                logger.exception(f"Sync task '{name}' failed: {e}")
                logger.recovery("Automatically skip this function.")
                
    @logging_and_warning_decorator()
    def _helper_commit_enrich_kwargs_sync(self, kwargs_sync: dict[str, Any], logger=None):
        kwargs_sync_out = dict(kwargs_sync)
        for name, func in self._impl_enrich_kwargs_sync_func.items():
            try:
                output = func(host=self, kwargs_sync=kwargs_sync_out)
                if output is not None:
                    kwargs_sync_out = output
            except Exception as e:
                logger.exception(f"Sync kwargs task {name!r} failed: {e}")
                logger.recovery("Automatically skip this function.")
        return kwargs_sync_out
                
    @logging_and_warning_decorator()
    def _helper_commit_enrich_kwargs_wrapped(self, kwargs: dict[str, Any], kwargs_sync=None, logger=None):
        if kwargs_sync is None:
            kwargs_sync = {}

        kwargs_wrapped = dict(kwargs)
        for name, func in self._impl_enrich_kwargs_wrapped_func.items():
            try:
                output = func(host=self, kwargs=kwargs_wrapped, kwargs_sync=kwargs_sync)
                if output is not None:
                    kwargs_wrapped = output
            except Exception as e:
                logger.exception(f"Wrapped kwargs task {name!r} failed: {e}")
                logger.recovery("Automatically skip this function.")
        return kwargs_wrapped
    
    
    
    
    def act_attach_enrich_kwargs_sync_task(self, name: str, func: Callable):
        if not callable(func):
            raise TypeError(f"The sync kwargs task {name!r} must be callable.")
        self._impl_enrich_kwargs_sync_func[name] = func

    def act_detach_enrich_kwargs_sync_task(self, name: str):
        self._impl_enrich_kwargs_sync_func.pop(name, None) 
    
    def act_attach_enrich_kwargs_wrapped_task(self, name: str, func: Callable):
        if not callable(func):
            raise TypeError(f"The wrapped kwargs task {name!r} must be callable.")
        self._impl_enrich_kwargs_wrapped_func[name] = func

    def act_detach_enrich_kwargs_wrapped_task(self, name: str):
        self._impl_enrich_kwargs_wrapped_func.pop(name, None)


        


    def act_save_opts(self, name=None):
        if not name:
            name = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        self._opts_backup[name] = self.opts.act_asdict()
        
    def act_attach_sync_task(self, name: str, func: Callable):
        if not callable(func):
            raise TypeError(f"The sync task '{name}' must be callable.")
        self._impl_sync_func[name] = func
        
    def act_detach_sync_task(self, name: str):
        self._impl_sync_func.pop(name, None)
       

    @logging_and_warning_decorator()
    def act_register_wrapped_attr(self, attrs: Sequence[str] | str, logger=None) -> None:
        """Register a group of public attribute as protected under wrapping."""
        
        if isinstance(attrs, str):
            attrs = [attrs]
        elif not isinstance(attrs, (list, tuple, set)):
            raise TypeError(
                "attrs must be a string or a sequence of strings, "
                f"got {type(attrs).__name__}."
            )
        
        for attr in attrs:
            try:
                attr = as_str(attr, name="The name of attr to be wrapped")
                if attr.startswith("raw_"):
                    if attr in self.__descriptions__:
                        self._impl_attrs_wrapped.update([attr, attr[4:]])
                    else:
                        raise AttributeError(
                            f"Attribute {attr!r} is not a valid public attribute of {type(self).__name__}."
                        )
                else:
                    if attr in self.opts.__class__.__descriptions__:
                        self._impl_attrs_wrapped.add(attr)
                    else:
                        raw_attr = "raw_" + attr
                        if raw_attr in self.__descriptions__:
                            self._impl_attrs_wrapped.update([raw_attr, raw_attr[4:]])
                        else:
                            raise AttributeError(
                                f"Attribute {attr!r} is not a valid public attribute of {type(self).__name__} or its opts."
                        )
            except Exception:
                logger.exception("Invalid attr name.")
                logger.recovery("Automatically ignore this attr.")
                
    @contextmanager
    def _helper_wrapped_update(self):
        protected = self._impl_attrs_wrapped
        backup = set(protected)
        protected.clear()
        try:
            yield
        finally:
            protected.update(backup)
            
    @property
    def wrapper(self):
        ref = self._impl_wrapper_ref
        return ref() if ref is not None else None
    
    @property
    def wrapped(self):
        return self._entity_wrapped
    
    def act_bind_wrapper(
        self,
        wrapper: HostBase,
        protected_attrs: Sequence[str] | str | None = None,
    ):
        
        old = self.wrapper
        if old is not None and (old is not wrapper):
            raise RuntimeError(
                f"{type(self).__name__} is already wrapped by {type(old).__name__}."
            )

        object.__setattr__(wrapper, "_entity_wrapped", self)
        object.__setattr__(self, "_impl_wrapper_ref", weakref.ref(wrapper))

        if protected_attrs:
            self.act_register_wrapped_attr(set(protected_attrs))
    
    
    # Rewrite from ClassBase. To handle opts.
    def _helper_setattr_final(self, key, value):
        self.act_commit(**{key: value})
    
    

