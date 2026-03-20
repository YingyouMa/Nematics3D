from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Type, Sequence
import weakref
from contextlib import contextmanager
import numpy as np
import datetime


from ..format import repr_format, save_opts_json
from ..logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import Unset, UNSET, as_str
from Nematics3D.general import pop_exclusive
from .opts import (
    merge_opts_all,
    build_dict_override,
    diff_dict_values,
    load_json_into_opts,
)
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

    Important readable attributes on ``OptsBase`` include:
    - ``host`` to access the owning host object, if one is currently attached

    User-facing convenience methods on ``OptsBase`` are:
    - ``act_finalize()`` to fill defaults and enter the functioning state
    - ``act_asdict()`` to export the current option payload
    - ``act_save_json()`` to serialize the current opts to disk
    - ``act_load_json()`` to load a saved JSON payload back into this instance

    Representation behavior is split intentionally:
    - ``str(opts)`` gives a compact one-line identity like ``OptsFigure``
    - ``repr(opts)`` prints the full field-by-field summary that is meant for
      interactive inspection
    """

    tag: str | Unset = UNSET

    _impl_host_ref: weakref.ReferenceType | None = field(
        default=None, repr=False, init=False
    )
    _state_is_functioning: bool = field(default=False, init=False, repr=False)

    __attrs__: ClassVar[Mapping[str, str]] = {
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

        is_functioning = bool(getattr(self, "_state_is_functioning", False))
        is_has_host = getattr(self, "host", None) is not None

        if not key.startswith("_") and key not in self.__class__.__attrs__:
            raise AttributeError(
                f"Invalid option field {key!r}. "
                f"Valid fields are: {list(self.__class__.__attrs__.keys())}"
            )

        # --- setting UNSET after functioning is forbidden ---
        if value is UNSET:
            if is_functioning:
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
            desc = f"{key!r}: {self.__class__.__attrs__[key]}"
            try:
                value2 = self.__class__._validators[key](value, desc)
                value = value2
            except Exception:
                logger.exception(f"Assignment to {key!r} failed")
                if is_functioning:
                    logger.recovery("Automatically ignore this modification")
                    return value  # ignored
                else:
                    logger.recovery("Reset this assignment to UNSET.")
                    object.__setattr__(self, key, UNSET)
                    value = UNSET
        else:
            if key.startswith("_") or not is_functioning:
                object.__setattr__(self, key, value)
                return value

        # --- host commit (only after functioning) ---
        if (
            not key.startswith("_")
            and is_functioning
            and is_has_host
            and key in self.__class__.__attrs__
        ):
            self._helper_host_apply(key, value)
            return value

        object.__setattr__(self, key, value)
        return value

    def _helper_host_apply(self, key, value):
        if self.host is not None:
            self.host.act_commit(**{key: value})
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

        defaults_dict = {} if defaults is None else dict(defaults)

        for k in self.__attrs__.keys():
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
        for k in self.__class__.__attrs__.keys():
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

    @logging_and_warning_decorator(start_finish_level=5)
    def act_save_json(
        self,
        path: str | Path,
        *,
        max_inline_array_size: int = 64,
        is_include_UNSET: bool = False,
        logger=None,
    ) -> Path:
        path = save_opts_json(
            self.act_asdict(is_include_UNSET=is_include_UNSET),
            path,
            opts_class_name=type(self).__name__,
            max_inline_array_size=max_inline_array_size,
        )
        logger.info(f"Saved opts JSON to {path}.")
        return path

    @logging_and_warning_decorator(start_finish_level=5)
    def act_load_json(
        self,
        path: str | Path,
        *,
        is_finalize: bool = False,
        logger=None,
    ):
        return load_json_into_opts(
            self,
            path,
            is_finalize=is_finalize,
        )

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        cls = type(self)
        cls_name = cls.__name__

        # --- header line ---
        host = self.host
        if host is not None:
            lines = [f"{cls_name}: the options of {str(host)}"]
        else:
            lines = [f"{cls_name}"]

        # --- collect fields ---
        keys = list(cls.__attrs__.keys())
        if not keys:
            return "\n".join(lines)

        width = max(len(k) for k in keys)

        for k in keys:
            try:
                v = getattr(self, k)
            except AttributeError:
                v = "<missing>"
            lines.append(f"  {k:<{width}} = {repr_format(v)}")

        return "\n".join(lines)


# Subclassing rules:
# - Keep host-side stored fields in `__attrs__`, and keep options in the paired
#   `OptsBase` subclass rather than spreading configurable state across both
#   layers without documentation.
# - Extend `__attrs__`, `__relations__`, `__properties__`, and `_impl_validators`
#   deliberately. These declarations define both behavior and user-facing
#   inspection output.
# - `__init__()` should only do minimal wiring. Concrete subclasses remain
#   responsible for finalizing opts and for defining how accepted option updates
#   are realized in host state.
# - `_helper_commit_apply_opts_main()` is the main subclass hook. If a subclass
#   accepts an update there, it must write the resolved value back to
#   `self.opts` through a non-recursive internal path.
# - Wrapper hosts that transform inputs must write transformed values back into
#   the forwarded kwargs so downstream wrapped hosts receive the resolved
#   parameters.
class HostBase(ClassBase):
    """
    Shared host controller for objects that manage state through an associated
    `OptsBase` configuration object.

    For typical users of this package, a HostBase-style object provides:

    - a normal object identity and relation interface inherited from `ClassBase`
    - a paired `.opts` object that stores configurable parameters
    - a commit-style update path instead of ad hoc direct mutation
    - inspection helpers such as `show_getattrs()`, `show_modifiable_attrs()`,
      and `show_relations()`

    The `show_*` helpers are especially useful when exploring an unfamiliar
    host object. In particular, `show_modifiable_attrs()` helps distinguish
    host-side fields from opts-managed fields before making updates.

    Important readable attributes on `HostBase` include:
    - `opts` to access the paired options object that controls host behavior

    User-facing `show_*` methods on `HostBase` include both the inherited
    inspection helpers and the host-specific saved-opts view:

    - `show_getattrs()` to list readable host and visible saved-state surfaces
    - `show_attr_desc()` to explain a host attr, relation, alias, or opts attr
    - `show_modifiable_attrs()` to separate host attrs, opts attrs, extra attrs,
      and writable properties
    - `show_relations()` / `show_relation_tree()` to inspect current object links
    - `show_saved_opts()` to list named snapshots stored in `_opts_backup`

    User-facing `act_*` methods on `HostBase` include both the inherited
    `ClassBase` actions and the host-specific commit utilities. Common ones are:

    - `act_set_name()` to rename the host through the validated identity path
    - `act_bind_relation_base()` / `act_unbind_relation_base()` to manage
      semantic host relations such as wrapper or owner links
    - `act_add_attr()` to register user-defined runtime attributes with docs
    - `act_register_protected_attr()` / `act_unregister_protected_attr()` to
      protect or unprotect host and opts-facing public attributes
    - `act_commit()` to apply host and opts updates through the managed commit
      pipeline
    - `act_save_opts()` to snapshot current opts into `_opts_backup`
    - `act_attach_sync_task()` / `act_detach_sync_task()` and the related
      enrich-kwargs task registration helpers
    - `act_register_wrapped_attr()` / `act_unregister_wrapped_attr()` and
      `act_bind_wrapper()` / `act_unbind_wrapper()` for wrapper forwarding

    Most package users should work with concrete host subclasses rather than
    subclassing HostBase directly.
    """

    __attrs__ = {
        **(ClassBase.__attrs__),
        "raw_name": "The name identifier of the host object",
        "_opts": "The Opts instance controlling options.",
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
        "_impl_attrs_protected": (
            "Additional protected attributes declared directly by this host. "
            "These attrs cannot be modified through act_commit() by external callers."
        ),
        "_impl_enrich_kwargs_wrapped_func": (
            "A dictionary of callback functions to enrich kwargs before forwarding to wrapped host. "
            "Key: unique identifier (str); Value: callable task(host, kwargs, kwargs_sync)."
        ),
        "_impl_enrich_kwargs_sync_func": (
            "A dictionary of callback functions to enrich kwargs_sync before sync task execution. "
            "Key: unique identifier (str); Value: callable task(host, kwargs_sync)."
        ),
    }
    __relations__ = {
        **(ClassBase.__relations__),
        "wrapper": (
            "The wrapper host that controls this host. "
            "An instance can be wrapped by at most one wrapper at a time."
        ),
        "wrapped": (
            "The wrapped host controlled by this host as a wrapper. "
            "An instance can wrap at most one wrapped host at a time."
        ),
    }
    __properties__ = {
        **(ClassBase.__properties__),
        "opts": "Read-only: The paired Opts object controlling this host.",
        "attrs_forbidden": (
            "Read-only: Union of wrapped-protected attrs and host-declared protected attrs."
        ),
    }

    __slots__ = tuple(k for k in __attrs__.keys() if k not in ClassBase.__slots__)

    _impl_validators = {}
    # Validator keys correspond to the public name (without the ``raw_`` prefix).
    # For example, ``raw_coords`` will use the validator registered under ``coords``.
    # The validator must accept two arguments: (value, description).

    _impl_attrs_reapply_opts_after_raw = set()
    # Public attribute names (without "raw_") that should force an opts re-apply
    # after raw/public assignment in a commit even if no explicit opts update is provided.

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.__init__ because a host must bind an
    # OptsBase instance, build opts defaults, and initialize commit-related
    # runtime stores in addition to the basic naming/relations setup.
    # ==================================================
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

        kwargs_host = {}
        for key in list(kwargs.keys()):
            if key in self.__attrs__ and (
                key.startswith("raw_") or key.startswith("state_")
            ):
                kwargs_host[key] = kwargs.pop(key)
            elif ("raw_" + key) in self.__attrs__:
                kwargs_host[key] = kwargs.pop(key)

        opts = self._helper_check_opts(opts, opts_type=opts_type)
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_impl_host_ref", weakref.ref(self))
        object.__setattr__(self, "_opts", opts)

        opts_defaults = {
            **{k: UNSET for k in opts.__attrs__},
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
        object.__setattr__(self, "_impl_attrs_protected", set())
        object.__setattr__(self, "_impl_enrich_kwargs_wrapped_func", {})
        object.__setattr__(self, "_impl_enrich_kwargs_sync_func", {})
        object.__setattr__(self, "_impl_attrs_wrapped", set())

        if kwargs_host:
            self._helper_commit_raw(kwargs_host)

        # remaining tasks for __init__():
        # - finalizing opts at the appropriate lifecycle stage, and
        # - defining how finalized opts are consumed and applied.

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_opts(self, opts, opts_type=None, logger=None):

        if opts_type is None:
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

    @property
    def opts(self):
        return self._opts

    # ------------------------------------------------------------------
    # Commit entrypoint
    # ------------------------------------------------------------------

    # -------------------------------
    # -------------------------------
    # The functions to commit update
    # -------------------------------
    # -------------------------------

    @logging_and_warning_decorator()
    def act_commit(
        self, opts=None, opts_wrapped=None, is_reapply_opts=False, logger=None, **kwargs
    ):

        # _helper_pop_private_key:
        #       remove keys starting with '_' from ``kwargs``.
        #       These keys are treated as non-public commit inputs and are ignored.
        #
        # _helper_commit_pre_opts:
        #       preprocess kwargs for this host before opts-level application.
        #       _helper_check_protected_attr: remove attrs protected by wrapper or by host declaration.
        #       _helper_commit_name: consume ``name`` / ``raw_name`` and update host name.
        #       _helper_commit_raw: consume host-side raw/state attrs, validate if configured,
        #                           then write directly to host (not through opts).
        #
        # _helper_commit_self:
        #       handle updates that belong to this host's opts domain.
        #       _helper_merge_opts_kwargs: merge explicit ``opts`` + opts-like kwargs,
        #                                  normalize to opts-dict payload.
        #       _helper_commit_apply_opts: perform wrapped-attr check + apply opts updates.
        #       _helper_commit_apply_opts_main: subclass-defined real apply logic;
        #                                       should write resolved values back to ``self.opts``
        #
        # sync stage:
        #       merge pre-opts sync kwargs and opts-applied kwargs,
        #       then run _helper_commit_enrich_kwargs_sync(...) to enrich/transform sync payload.
        #       if non-empty, call _helper_trigger_sync_batch(**kwargs_sync).
        #
        # forwarding stage:
        #       call _helper_kwargs_to_wrapped(kwargs, opts_wrapped=opts_wrapped).
        #       _helper_commit_enrich_kwargs_wrapped(...) can enrich kwargs before forwarding.
        #       if wrapped exists, forward via self.wrapped.act_commit(...);
        #       otherwise warn about unhandled remaining kwargs/opts_wrapped.

        self._helper_pop_private_key(kwargs)
        kwargs_sync, is_reapply_opts_from_raw = self._helper_commit_pre_opts(kwargs)
        is_reapply_opts = is_reapply_opts or is_reapply_opts_from_raw

        opts_keys = self.opts.__class__.__attrs__
        is_opts_request = (opts is not None) or any(k in opts_keys for k in kwargs)
        if is_reapply_opts or is_opts_request:
            kwargs, kwargs_applied_opts = self._helper_commit_self(
                opts=opts, is_reapply_opts=is_reapply_opts, **kwargs
            )
            kwargs_sync = kwargs_sync | kwargs_applied_opts

        kwargs_sync = self._helper_commit_enrich_kwargs_sync(kwargs_sync)
        if kwargs_sync:
            self._helper_trigger_sync_batch(**kwargs_sync)

        self._helper_kwargs_to_wrapped(kwargs, opts_wrapped=opts_wrapped)

    # ------------------------------------------------------------------
    # Commit preprocessing
    # ------------------------------------------------------------------

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_pop_private_key(self, kwargs, logger=None):
        private_keys = [k for k in list(kwargs.keys()) if k.startswith("_")]
        for key in private_keys:
            kwargs.pop(key)
            logger.warning(
                f"{key!r} is not a valid public commit key. "
                "Names starting with '_' are not accepted by act_commit(). "
                "If you intentionally need to modify an internal attribute, use object.__setattr__() directly."
            )

    # -----------------------
    # _helper_commit_pre_opts
    # -----------------------
    def _helper_commit_pre_opts(self, kwargs):
        if not kwargs:
            return {}, False
        self._helper_check_protected_attr(kwargs)
        kwargs_applied_name = self._helper_commit_name(kwargs)
        kwargs_applied_raw, is_reapply_opts = self._helper_commit_raw(kwargs)
        return kwargs_applied_raw | kwargs_applied_name, is_reapply_opts
        # Any value modified by the wrapper must be written back to ``kwargs``
        # so the updated parameters are forwarded to the wrapped object.
        # For example, the wrapped object only accepts ``radius``, while the wrapper
        # expose a convenience parameter ``radius_scale``. If the wrapper receives
        # ``radius_scale=2``, it should delete ``radius_scale`` and replace
        # ``radius`` with the doubled value (radius=2*radius) before forwarding
        # ``kwargs`` to the wrapped object.

    @logging_and_warning_decorator()
    def _helper_check_protected_attr(self, kwargs, logger=None):
        if not kwargs:
            return
        blocked = [k for k in kwargs.keys() if k in self.attrs_forbidden]
        for key in blocked:
            kwargs.pop(key)
            try:
                raise AttributeError(
                    f"{key!r} is protected and could not be directly modified"
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
            return {}, False
        kwargs_applied_raw = {}
        is_reapply_opts = False
        for key in list(kwargs.keys()):
            is_host_attr = (
                key in self.__attrs__
                and (key.startswith("raw_") or key.startswith("state_"))
            ) or (("raw_" + key) in self.__attrs__)
            if is_host_attr:
                kwargs_applied_here, is_reapply_opts_here = self._helper_commit_pop_raw(
                    kwargs, key
                )
                kwargs_applied_raw |= kwargs_applied_here
                is_reapply_opts = is_reapply_opts or is_reapply_opts_here
        return kwargs_applied_raw, is_reapply_opts

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

        is_state_attr = attr_name_origin.startswith("state_")
        if attr_name_origin.startswith("raw_"):
            host_attr_name = attr_name_origin
            public_attr_name = attr_name_origin[4:]
            found, attr_value = pop_exclusive(kwargs, public_attr_name, host_attr_name)
            validator_key = public_attr_name
            reapply_key = public_attr_name
            attr_name_return = public_attr_name
        elif is_state_attr:
            host_attr_name = attr_name_origin
            found = host_attr_name in kwargs
            attr_value = kwargs.pop(host_attr_name) if found else None
            validator_key = host_attr_name
            reapply_key = host_attr_name
            attr_name_return = host_attr_name
        else:
            host_attr_name = "raw_" + attr_name_origin
            public_attr_name = attr_name_origin
            found, attr_value = pop_exclusive(kwargs, public_attr_name, host_attr_name)
            validator_key = public_attr_name
            reapply_key = public_attr_name
            attr_name_return = public_attr_name
        if not found:
            return {}, False

        if exception_msg is None:
            exception_msg = (
                f"Validation failed for attribute {attr_name_return!r}. "
                f"This may be due to an invalid value or an incorrectly implemented validator. "
                f"The validator must accept two arguments: (value, description)."
            )
        if recovery_msg is None:
            recovery_msg = f"Ignore this modification of {attr_name_return!r}."

        if validator is None and validator_key in self._impl_validators:
            validator = self._impl_validators[validator_key]

        if validator is not None:
            try:
                value_valid = validator(attr_value, self.__attrs__[host_attr_name])
                object.__setattr__(self, host_attr_name, value_valid)
                return {attr_name_return: value_valid}, (
                    reapply_key in self.__class__._impl_attrs_reapply_opts_after_raw
                )

            except Exception:
                logger.exception(exception_msg)
                logger.recovery(recovery_msg)
                return {}, False
        else:
            object.__setattr__(self, host_attr_name, attr_value)
            return {attr_name_return: attr_value}, (
                reapply_key in self.__class__._impl_attrs_reapply_opts_after_raw
            )

    # ------------------------------------------------------------------
    # Opts application
    # ------------------------------------------------------------------

    # -----------------------
    # _helper_commit_self
    # -----------------------
    def _helper_commit_self(self, opts=None, is_reapply_opts=False, **kwargs):
        if kwargs or opts or is_reapply_opts:
            self_keys = self.opts.__class__.__attrs__
            kwargs_self = {
                k: kwargs.pop(k) for k in list(kwargs.keys()) if k in self_keys
            }
            kwargs_self = self._helper_merge_opts_kwargs(opts=opts, **kwargs_self)
            kwargs_left, kwargs_applied_opts = self._helper_commit_apply_opts(
                is_reapply_opts=is_reapply_opts, **kwargs_self
            )
            kwargs |= kwargs_left
            return kwargs, kwargs_applied_opts
        else:
            return kwargs, {}

    def _helper_merge_opts_kwargs(self, opts=None, **kwargs):
        if kwargs or opts:
            opts = self._helper_check_opts(opts)
            opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
            return opts.act_asdict()
        else:
            return {}

    def _helper_commit_apply_opts(self, is_reapply_opts=False, **kwargs):
        self._helper_check_protected_attr(kwargs)
        opts_before = self.opts.act_asdict()
        kwargs_applied_opts = {}
        if "tag" in kwargs:
            tag_value = kwargs["tag"]  # After initialization, tag could be any values
            object.__setattr__(self.opts, "tag", tag_value)
            kwargs_applied_opts["tag"] = tag_value
            kwargs.pop("tag")
        return_main = self._helper_commit_apply_opts_main(
            is_reapply_opts=is_reapply_opts, **kwargs
        )
        if return_main is None:
            kwargs_left = {}
            opts_after = self.opts.act_asdict()
            _, kwargs_applied_opts_main = diff_dict_values(opts_before, opts_after)
        else:
            kwargs_left, kwargs_applied_opts_main = return_main
        kwargs_applied_opts = kwargs_applied_opts | kwargs_applied_opts_main
        return kwargs_left, kwargs_applied_opts
        # the input kwargs should only include the attributes in options

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
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

    # ------------------------------------------------------------------
    # Sync and wrapped forwarding
    # ------------------------------------------------------------------

    @logging_and_warning_decorator()
    def _helper_trigger_sync_batch(self, logger=None, **kwargs):
        for name, func in self._impl_sync_func.items():
            try:
                func(**kwargs)
            except Exception as e:
                logger.exception(f"Sync task '{name}' failed: {e}")
                logger.recovery("Automatically skip this function.")

    @logging_and_warning_decorator()
    def _helper_commit_enrich_kwargs_sync(
        self, kwargs_sync: dict[str, Any], logger=None
    ):
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
    def _helper_commit_enrich_kwargs_wrapped(self, kwargs: dict[str, Any], logger=None):
        kwargs_wrapped = dict(kwargs)
        for name, func in self._impl_enrich_kwargs_wrapped_func.items():
            try:
                output = func(host=self, kwargs=kwargs_wrapped)
                if output is not None:
                    kwargs_wrapped |= output
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

    def act_attach_sync_task(self, name: str, func: Callable):
        if not callable(func):
            raise TypeError(f"The sync task '{name}' must be callable.")
        self._impl_sync_func[name] = func

    def act_detach_sync_task(self, name: str):
        self._impl_sync_func.pop(name, None)

    @logging_and_warning_decorator()
    def _helper_kwargs_to_wrapped(self, kwargs, opts_wrapped=None, logger=None):
        kwargs = self._helper_commit_enrich_kwargs_wrapped(kwargs)
        if kwargs or opts_wrapped:
            if self.wrapped is not None:
                with self.wrapped._helper_wrapped_update():
                    self.wrapped.act_commit(opts=opts_wrapped, **kwargs)
            else:
                cls_name = self.__class__.__name__
                obj_name = getattr(self, "raw_name", "Uninitialized")
                lines = [f"[{cls_name}: {obj_name!r}] Unhandled commit arguments."]
                if kwargs:
                    lines.append(f"  Remaining kwargs keys: {list(kwargs.keys())}")
                if opts_wrapped is not None:
                    lines.append(f"  opts_wrapped: {opts_wrapped!r}")
                logger.warning("\n".join(lines))

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.show_getattrs so host exploration focuses on
    # user-facing readable names while still exposing `_opts_backup`, which is a
    # useful saved-state surface for end users.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def show_getattrs(self, is_return=False, logger=None):
        names = sorted(
            name for name in self._impl_getattr_names if not name.startswith("_impl_")
        )

        hidden_names = {
            "_opts",
            "_opts_defaults",
        }
        lines = [
            "When reading or assigning, the 'raw_' prefix may be omitted where a public alias exists."
        ]
        for name in names:
            if name in hidden_names:
                continue
            try:
                lines.append(self.show_attr_desc(name))
            except KeyError:
                continue

        if len(lines) == 1:
            lines.append("<none>")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.show_attr_desc so descriptions can be
    # resolved from both the host layer and the paired opts layer.
    # ==================================================
    def show_attr_desc(self, attr_name: str) -> str:
        try:
            return super().show_attr_desc(attr_name)
        except KeyError:
            pass

        opts = getattr(self, "_opts", None)
        if opts is not None:
            descriptions_opts = opts.__class__.__attrs__
            if attr_name in descriptions_opts:
                return f"{attr_name!r}: {descriptions_opts[attr_name]}"
            properties_opts = getattr(opts.__class__, "__properties__", {})
            if attr_name in properties_opts:
                return f"{attr_name!r}: {properties_opts[attr_name]}"
            raise KeyError(
                f"Attribute {attr_name!r} was not found in {type(self).__name__}.__attrs__ / __properties__ / __relations__ / extra attrs "
                f"or {type(opts).__name__}.__attrs__ / __properties__."
            )

        raise KeyError(
            f"Attribute {attr_name!r} was not found in {type(self).__name__}.__attrs__ / __properties__ / __relations__ / extra attrs. "
            "The opts attrs are not available yet because self._opts has not been initialized; "
            "the attribute may belong to opts."
        )

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.show_modifiable_attrs to present modifiable
    # fields by category: host attrs, opts attrs, and writable properties.
    # ==================================================
    @logging_and_warning_decorator()
    def show_modifiable_attrs(self, is_return=False, logger=None):
        lines = [
            "When assigning host fields, the 'raw_' prefix may be omitted.",
        ]

        attrs_forbidden = self.attrs_forbidden
        attrs_host = sorted(
            k
            for k in self.__class__.__attrs__.keys()
            if (k.startswith("raw_") or k.startswith("state_"))
            and (k not in attrs_forbidden)
        )
        attrs_opts = sorted(
            k for k in self.opts.__class__.__attrs__.keys() if k not in attrs_forbidden
        )
        attrs_extra = sorted(
            k for k in self._impl_extra_attrs_docs.keys() if k not in attrs_forbidden
        )
        if "tag" in attrs_opts:
            attrs_opts.remove("tag")
            attrs_opts.insert(0, "tag")
        attrs_properties = sorted(
            k
            for k in self.__class__.__properties__.keys()
            if self.__class__._helper_is_writable_property(k)
            and (k not in attrs_forbidden)
        )
        attrs_opts_properties = sorted(
            k
            for k in getattr(self.opts.__class__, "__properties__", {}).keys()
            if self.opts.__class__._helper_is_writable_property(k)
            and (k not in attrs_forbidden)
        )

        if attrs_host:
            lines.append("[Host attributes]")
            for attr_name in attrs_host:
                lines.append(f"  - {self.show_attr_desc(attr_name)}")

        if attrs_opts:
            lines.append("[Opts attributes]")
            for attr_name in attrs_opts:
                lines.append(f"  - {self.show_attr_desc(attr_name)}")

        if attrs_extra:
            lines.append("[Extra attributes]")
            for attr_name in attrs_extra:
                lines.append(f"  - {self.show_attr_desc(attr_name)}")

        if attrs_properties:
            lines.append("[Host writable properties]")
            for attr_name in attrs_properties:
                lines.append(f"  - {self.show_attr_desc(attr_name)}")

        if attrs_opts_properties:
            lines.append("[Opts writable properties]")
            for attr_name in attrs_opts_properties:
                lines.append(f"  - {self.show_attr_desc(attr_name)}")

        if (
            (not attrs_host)
            and (not attrs_opts)
            and (not attrs_extra)
            and (not attrs_properties)
            and (not attrs_opts_properties)
        ):
            lines.append("  (None)")

        if attrs_forbidden:
            lines.append(
                "Protected or wrapped fields are excluded from the lists above and cannot be modified through normal commit/setattr paths."
            )

        output = "\n".join(lines)
        logger.info(output)

        if is_return:
            return output

    # ------------------------------------------------------------------
    # Saved opts
    # ------------------------------------------------------------------

    def act_save_opts(self, name=None):
        if not name:
            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._opts_backup[name] = self.opts.act_asdict()

    @logging_and_warning_decorator(start_finish_level=5)
    def show_saved_opts(self, is_return=False, logger=None):
        backups = self._opts_backup
        lines = ["Saved opts snapshots in '_opts_backup':"]

        if backups:
            for name in backups.keys():
                lines.append(f"  - {name}")
        else:
            lines.append("  - <none>")

        lines.append(
            "Use `self._opts_backup[name]` to inspect a full saved opts dictionary."
        )
        lines.append(
            "To restore one manually, call `self.act_commit(**self._opts_backup[name])`."
        )
        lines.append(
            "To compare two saved opts dictionaries, use `diff_dict_values(dict1, dict2)` from `Nematics3D.classes.opts`."
        )

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output

    # ------------------------------------------------------------------
    # Protection and wrapping
    # ------------------------------------------------------------------

    @logging_and_warning_decorator()
    def _helper_register_protected_attr(
        self,
        attrs: Sequence[str] | str,
        target_set: set[str],
        attr_name: str,
        logger=None,
    ) -> None:
        if isinstance(attrs, str):
            attrs = [attrs]
        elif not isinstance(attrs, (list, tuple, set)):
            raise TypeError(
                "attrs must be a string or a sequence of strings, "
                f"got {type(attrs).__name__}."
            )

        for attr in attrs:
            try:
                attr = as_str(attr, name=attr_name)
                if attr.startswith("raw_"):
                    if attr in self.__attrs__:
                        target_set.update([attr, attr[4:]])
                    else:
                        raise AttributeError(
                            f"Attribute {attr!r} is not a valid public attribute of {type(self).__name__}."
                        )
                elif attr.startswith("state_"):
                    if attr in self.__attrs__:
                        target_set.add(attr)
                    else:
                        raise AttributeError(
                            f"Attribute {attr!r} is not a valid public state attribute of {type(self).__name__}."
                        )
                else:
                    if attr in self.opts.__class__.__attrs__:
                        target_set.add(attr)
                    else:
                        raw_attr = "raw_" + attr
                        if raw_attr in self.__attrs__:
                            target_set.update([raw_attr, raw_attr[4:]])
                        else:
                            raise AttributeError(
                                f"Attribute {attr!r} is not a valid public attribute of {type(self).__name__} or its opts."
                            )
            except Exception:
                logger.exception("Invalid attr name.")
                logger.recovery("Automatically ignore this attr.")

    def act_register_wrapped_attr(self, attrs: Sequence[str] | str) -> None:
        """Register a group of public attributes as protected under wrapping."""
        self._helper_register_protected_attr(
            attrs,
            target_set=self._impl_attrs_wrapped,
            attr_name="The name of attr to be wrapped",
        )

    def act_unregister_wrapped_attr(
        self, attrs: Sequence[str] | str | None = None
    ) -> None:
        if attrs is None:
            self._impl_attrs_wrapped.clear()
            return

        if isinstance(attrs, str):
            attrs = [attrs]
        elif not isinstance(attrs, (list, tuple, set)):
            raise TypeError(
                "attrs must be a string or a sequence of strings, "
                f"got {type(attrs).__name__}."
            )

        for attr in attrs:
            attr = as_str(attr, name="The name of attr to be unwrapped")
            if attr.startswith("raw_"):
                self._impl_attrs_wrapped.discard(attr)
                self._impl_attrs_wrapped.discard(attr[4:])
            elif attr.startswith("state_"):
                self._impl_attrs_wrapped.discard(attr)
            else:
                self._impl_attrs_wrapped.discard(attr)
                self._impl_attrs_wrapped.discard("raw_" + attr)

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.act_register_protected_attr because protected
    # names may belong either to the host itself or to its paired opts object.
    # ==================================================
    def act_register_protected_attr(self, attrs: Sequence[str] | str) -> None:
        """Register a group of public attributes as directly protected by this host."""
        self._helper_register_protected_attr(
            attrs,
            target_set=self._impl_attrs_protected,
            attr_name="The name of attr to be protected",
        )

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.act_unregister_protected_attr because the
    # protected-name surface may include host aliases and paired opts attrs.
    # ==================================================
    def act_unregister_protected_attr(self, attrs: Sequence[str] | str) -> None:
        if isinstance(attrs, str):
            attrs = [attrs]
        elif not isinstance(attrs, (list, tuple, set)):
            raise TypeError(
                "attrs must be a string or a sequence of strings, "
                f"got {type(attrs).__name__}."
            )

        for attr in attrs:
            attr = as_str(attr, name="The name of attr to be unprotected")
            if attr.startswith("raw_"):
                self._impl_attrs_protected.discard(attr)
                self._impl_attrs_protected.discard(attr[4:])
            elif attr.startswith("state_"):
                self._impl_attrs_protected.discard(attr)
            else:
                self._impl_attrs_protected.discard(attr)
                self._impl_attrs_protected.discard("raw_" + attr)

    @property
    def attrs_forbidden(self):
        return set(self._impl_attrs_wrapped) | set(self._impl_attrs_protected)

    @contextmanager
    def _helper_wrapped_update(self):
        protected = self._impl_attrs_wrapped
        backup = set(protected)
        protected.clear()
        try:
            yield
        finally:
            protected.update(backup)

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

        old_wrapped = wrapper.wrapped
        if old_wrapped is not None and (old_wrapped is not self):
            raise RuntimeError(
                f"{type(wrapper).__name__} already wraps {type(old_wrapped).__name__}."
            )

        self.act_bind_relation_base("wrapper", wrapper, is_weak=True)
        wrapper.act_bind_relation_base("wrapped", self, is_weak=False)

        if protected_attrs:
            self.act_register_wrapped_attr(protected_attrs)

    def act_unbind_wrapper(self):
        wrapper = self.wrapper
        if wrapper is not None and wrapper.wrapped is self:
            wrapper.act_unbind_relation_base("wrapped")
        self.act_unbind_relation_base("wrapper")
        self.act_unregister_wrapped_attr()

    # ------------------------------------------------------------------
    # Public assignment
    # ------------------------------------------------------------------

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase._helper_setattr_final so public assignment is
    # routed through `act_commit()` instead of writing directly to host storage.
    # ==================================================
    def _helper_setattr_final(self, key, value):
        self.act_commit(**{key: value})
