from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Type, Sequence
import weakref
from contextlib import contextmanager
import numpy as np
import datetime

from ..logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import Unset, UNSET, as_str
from Nematics3D.general import pop_exclusive
from .opts import merge_opts_all, build_dict_override
from .class_base import ClassBase


# ---------------------------------------------------------------------
# Design Contract: Opts / Host Interaction Semantics
#
# The following assumptions are fundamental to the correct usage of
# OptsBase and HostBase in this framework:
#
# 1. After an Opts instance has been finalized (i.e. enters the
#    "functioning" state), assignments to public (non-underscore) fields
#    of opts do NOT directly mutate the opts object. Such assignments are
#    treated as *requests* and are forwarded to the host via commit.
#
#    Consequently, opts primarily serve as a lightweight preprocessing
#    and validation layer. The host is responsible for deciding whether
#    an update is accepted and, if so, for writing the final value back.
#
# 2. The kwargs received by HostBase._helper_commit_apply(...) are
#    guaranteed to have passed all opts-level preprocessing and basic
#    validation. Host implementations may assume that input values are
#    already sanitized, and therefore should not repeat opts-level
#    validation. Host-side logic should focus on state-dependent or
#    cross-field constraints and side effects.
#
# 3. HostBase.__init__ only performs minimal wiring (opts binding,
#    defaults construction, and name initialization). Concrete host
#    subclasses are responsible for:
#       - finalizing opts at the appropriate lifecycle stage, and
#       - defining how finalized opts are consumed and applied.
#
# 4. When a host accepts an update in _helper_commit_apply(...), it MUST
#    write the resolved value back to opts. This write-back must bypass
#    the normal opts assignment path (e.g. via object.__setattr__ or
#    OptsBase._helper_impl_update) to avoid recursive commit loops.
#
# Violating any of these assumptions may result in silent state
# inconsistencies or hard-to-debug behavior.
# ---------------------------------------------------------------------


@dataclass(slots=True, repr=False)
class OptsBase:
    tag: str | Unset = UNSET

    _impl_host_ref: weakref.ReferenceType | None = field(
        default=None, repr=False, init=False
    )
    _state_is_functioning: bool = field(default=False, init=False, repr=False)
    _impl_sync_func: dict[str, dict[str, Callable[[], Any]]] = field(
        default_factory=dict, init=False, repr=False
    )

    __descriptions__: ClassVar[Mapping[str, str]] = {
        "tag": "name identifier of the option settings",
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        "tag": lambda v, d: as_str(v, name=d)
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({"tag": "options"})

    def __post_init__(self):
        object.__setattr__(
            self, "_impl_sync_func", {k: {} for k in self.__descriptions__.keys()}
        )

    @property
    def host(self):
        ref = self._impl_host_ref
        return ref() if ref is not None else None

    # ---------------------------------------------------------------------
    # Basic core: assignment with validation + lifecycle rule + host commit
    # ---------------------------------------------------------------------
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_setattr_basic(self, key: str, value: Any, *, logger=None) -> Any:

        is_final = bool(getattr(self, "_state_is_functioning", False)) and (self.host is not None)

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

        # --- host commit (only after functioning) ---
        if not key.startswith("_") and is_final:
            self._helper_sync(key, value)
            return value

        object.__setattr__(self, key, value)
        return value

    def _helper_sync(self, key, value):
        self._helper_host_apply(key, value)
        sync_func = self._impl_sync_func.get(key, {})
        for func in sync_func.values():
            func()

    def _helper_host_apply(self, key, value):
        if self.host:
            self.host._helper_commit_apply(**{key: value})
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
    def _helper_impl_update(self):
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
            lines = [f"{cls_name}: the options of {host!r}"]
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

    # fmt: off
    __descriptions__: ClassVar[Mapping[str, str]] = {
        **(ClassBase.__descriptions__),
        
        "raw_name":             "The name identifier of the host object",
        
        "opts":                 "The Opts instance controlling options.",
        "opts_defaults":        "The default option settings.",
        
        "_impl_opts_backup": (
            "A dictionary storing potentially useful options, indexed by timestamp."
            "Key: Current time, or manualy set value; Value: A dictionary of options (opts)."
        ),
    }
    # fmt: off

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
        opts_defaults = build_dict_override(
            opts._DEFAULTS_FROZEN,
            opts_defaults_override,
            name=type(opts).__name__,
        )
        object.__setattr__(self, "opts_defaults", opts_defaults)

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

    def _helper_merge_opts_kwargs(self, opts=None, **kwargs):
        opts = self._helper_check_opts(opts)
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        kwargs = opts.act_asdict()
        if "tag" in kwargs.keys():
            object.__setattr__(self.opts, "tag", kwargs["tag"])
            kwargs.pop("tag")
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
        
        

    def act_save_opts(self, name=None):
        if not name:
            name = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        self._impl_opts_backup[name] = self.opts.act_asdict()
        


    # -----------------------------------------------------------------
    # OVERRIDE:
    #
    # This method intentionally overrides ClassBase._helper_setattr_basic.
    #
    # For Host objects, direct assignment to public (non-underscore)
    # attributes does NOT mutate the host instance immediately.
    # Instead, such assignments are forwarded to act_commit(...) and
    # handled by the commit pipeline.
    #
    # This enforces a strict "commit-driven" update model for hosts:
    # all externally visible state changes must pass through
    # _helper_commit_apply(...), where consistency checks and side
    # effects are centrally managed.
    # -----------------------------------------------------------------

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

        self.act_commit(**{key: value})

