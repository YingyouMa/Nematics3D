"""
Host-side foundations for objects that are driven by an associated opts object.

This module provides two closely related bases:

- ``OptsBase`` stores validated configuration values and manages the opts
  lifecycle from editable construction state into functioning runtime state.
- ``HostBase`` extends ``ClassBase`` with a paired ``opts`` object, a
  commit-style update pipeline, host/opts protection bookkeeping, wrapped-host
  forwarding, sync callback registries, and saved opts snapshots.

The design intentionally stays close to the original Nematics3D host model.
Public opts fields remain explicit dataclass slots, host-side runtime
containers keep ``impl_*`` names, and user-facing convenience access is
provided through readable properties such as ``host``, ``defaults_frozen``,
``opts``, and ``attrs_forbidden``.

At a high level the workflow is:

1. ``OptsBase`` validates public field assignment through
   ``_helper_setattr_basic()``.
2. ``act_finalize()`` fills ``UNSET`` values from defaults and marks opts as
   functioning.
3. Once functioning, public opts edits are forwarded back to the owning host
   through ``host.act_commit(...)``.
4. ``HostBase.act_commit()`` separates host-side raw/state changes, opts-side
   changes, synchronization callbacks, and wrapped-host forwarding.
5. Inspection helpers such as ``show_readable_attrs()``,
   ``show_modifiable_attrs()``, ``show_relations()``, and
   ``show_saved_opts()`` help users explore an unfamiliar host object before
   mutating it.

Concrete subclasses are expected to define their own opts class and to
implement ``_helper_commit_apply_opts_main()`` so finalized opts values can be
applied to actual host state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence, Type
import weakref

from ..datatypes import UNSET, Unset, as_list, as_str
from ..format import repr_format, save_opts_json
from ..general import pop_exclusive
from ..logging_decorator import logging_and_warning_decorator
from .class_base import ClassBase
from .opts import (
    build_dict_override,
    diff_dict_values,
    load_json_into_opts,
    merge_opts_all,
)


# HostBase declaration conventions for subclasses:
# - The strong rule is that host-side independent/input variables may only live
#   in `raw_...`, `state_...`, or the paired opts system (`opts`,
#   `opts_defaults`, `opts_backup`).
# - `raw_` fields are canonical stored public host data fields and may expose
#   a shortened public alias without the prefix.
# - `state_` fields represent writable host runtime state inputs that remain
#   part of the managed host schema and affect later computation.
# - Changing a `raw_`, `state_`, or opts field is not merely local assignment;
#   it may trigger validation, opts reapplication, recomputation, wrapped-host
#   forwarding, or sync behavior through the host commit pipeline.
# - Other host-side fields should normally be treated as dependent/derived or
#   internal values rather than as user-controlled inputs.
# - `calc_` fields represent derived host-side calculated data. They must be
#   read-only from the public surface. When a calculated view should behave as
#   a property, prefer registering it as `kind="property"` instead of exposing
#   the storage field itself as modifiable state.
# - `entity_` fields represent attached external/runtime entities such as
#   cached engine objects or UI/render handles. They must be treated as
#   read-only from the public surface.
# - if an `entity_` field is really a semantic one-to-one object link, prefer
#   expressing it as a relation instead of as an `entity_` field.
# - `impl_` fields represent internal host-side implementation storage and
#   should not be treated as the normal user-facing surface.
# - host-side default-value semantics should normally live in the paired opts
#   system (`opts`, `opts_defaults`, `opts_backup`) rather than in separate
#   `default_` fields on the host itself.
# - direct public names on HostBase should usually be reserved for relations,
#   properties, and the host/opts bridge fields such as `opts`.
# - relations in the current HostBase protocol are one-to-one links only;
#   do not use relations to represent one-to-many or collection-style data.
# - if a new managed host field is not a relation, property, or opts bridge
#   field, it should normally be expressed as `raw_...`, `state_...`,
#   `calc_...`, `entity_...`, or `impl_...` rather than inventing another
#   public field category.


@dataclass(slots=True, repr=False)
class OptsBase:
    """
    Reactive validated configuration base for Host-style objects.

    ``OptsBase`` is the light-weight configuration companion that sits beside a
    concrete ``HostBase`` object. Public option fields such as ``tag`` are
    stored directly on the instance, must all be registered in ``__attrs__``,
    while host wiring and lifecycle bookkeeping live in ``impl_*`` storage
    fields.

    The lifecycle is intentionally explicit:

    1. During normal construction, public fields may remain ``UNSET``.
    2. ``act_finalize()`` fills remaining ``UNSET`` fields from
       instance-provided defaults and then from ``defaults_frozen``.
    3. After finalization, the opts instance becomes functioning and public
       assignment may no longer set fields back to ``UNSET``.
    4. Once functioning and attached to a host, any public field update is
       forwarded through ``host.act_commit(...)`` rather than being treated as
       a purely local mutation.

    Important readable interfaces on ``OptsBase`` are:

    - ``host`` for the currently attached host object, if the weakref is alive
    - ``impl_is_functioning`` for the finalized runtime lifecycle state
    - ``defaults_frozen`` for the class-level frozen defaults mapping

    Common user-facing actions are:

    - ``act_finalize()`` to fill defaults and enter the functioning state
    - ``act_asdict()`` to export the current public opts payload
    - ``act_save_json()`` to serialize the current opts payload to disk
    - ``act_load_json()`` to restore a saved opts payload from disk

    Representation is split intentionally:

    - ``str(opts)`` gives a compact one-line identity such as ``OptsFigure``
    - ``repr(opts)`` prints the full field-by-field summary for inspection
    """

    tag: str | Unset = UNSET

    impl_host_ref: weakref.ReferenceType | None = field(
        default=None,
        init=False,
        repr=False,
    )
    impl_is_functioning: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    __attrs__: ClassVar[Mapping[str, str]] = {
        "tag": "name identifier of the option settings",
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        "tag": lambda v, d: as_str(v, name=d),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"tag": "options"}
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate dataclass field naming against the OptsBase declaration convention."""
        for field_name in type(self).__dataclass_fields__:
            if field_name in type(self).__attrs__:
                continue
            if not field_name.startswith("impl_"):
                raise ValueError(
                    "OptsBase dataclass fields that are not declared in __attrs__ "
                    "must use the impl_ prefix. "
                    f"Got invalid internal field name: {field_name!r}."
                )

    # ------------------------------------------------------------------
    # Readable properties
    # ------------------------------------------------------------------

    @property
    def host(self):
        """Return the attached host object, if the stored weakref is alive."""
        host_ref = getattr(self, "impl_host_ref", None)
        return host_ref() if host_ref is not None else None


    @property
    def defaults_frozen(self) -> Mapping[str, Any]:
        """Expose the class-level frozen defaults mapping."""
        return type(self).impl_defaults_frozen

    # ------------------------------------------------------------------
    # Basic core
    # ------------------------------------------------------------------

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_setattr_basic(self, key: str, value: Any, *, logger=None) -> None:
        """Validate one assignment and forward live option updates to the host."""
        is_functioning = bool(getattr(self, "impl_is_functioning", False))
        is_has_host = self.host is not None
        is_internal_key = key.startswith("impl_")

        if (not is_internal_key) and (key not in type(self).__attrs__):
            raise AttributeError(
                f"Invalid option field {key!r}. Valid fields are: "
                f"{list(type(self).__attrs__)}"
            )

        if value is UNSET:
            if is_functioning:
                try:
                    raise TypeError(
                        "Attribute could not be set as UNSET after first "
                        "functioning!"
                    )
                except TypeError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this modification")
                return

            object.__setattr__(self, key, value)
            return

        # If a validator exists, validate first and then let the later branches
        # decide whether this should remain a local assignment or be forwarded
        # through the host pipeline.
        if key in type(self).impl_validators:
            desc = f"{key!r}: {type(self).__attrs__[key]}"
            try:
                value = type(self).impl_validators[key](value, desc)
            except (TypeError, ValueError, KeyError, AttributeError):
                logger.exception(f"Assignment to {key!r} failed")
                if is_functioning:
                    logger.recovery("Automatically ignore this modification")
                    return

                logger.recovery("Reset this assignment to UNSET.")
                object.__setattr__(self, key, UNSET)
                return
        # If no validator exists, there are two intended cases:
        # 1. internal `impl_` fields: these are implementation storage, so they
        #    can be assigned directly.
        # 2. non-internal public fields without a local validator: before opts
        #    is functioning, store the draft value locally; after opts becomes
        #    functioning and is attached to a host, let the host own the update
        #    path because that host may perform more complex validation and
        #    application than OptsBase can know about locally.
        elif is_internal_key or (not is_functioning):
            object.__setattr__(self, key, value)
            return

        if (
            (not is_internal_key)
            and is_functioning
            and is_has_host
            and (key in type(self).__attrs__)
        ):
            self._helper_host_apply(key, value)
            return

        object.__setattr__(self, key, value)

    def _helper_host_apply(self, key: str, value: Any) -> None:
        """Forward one option update through the attached host commit pipeline."""
        if self.host is not None:
            self.host.act_commit(**{key: value})

    def _helper_finalize_basic(
        self,
        defaults: Mapping[str, Any] | None = None,
        is_allow_unset: bool = False,
    ) -> None:
        """Fill ``UNSET`` values by defaults, then enter the functioning state."""

        if getattr(self, "impl_is_functioning", False):
            raise RuntimeError("This Opts has already been finalized.")

        defaults_dict = {} if defaults is None else dict(defaults)

        for key in type(self).__attrs__:
            if getattr(self, key) is UNSET:
                value = defaults_dict.get(key, self.defaults_frozen.get(key, UNSET))
                if (value is UNSET) and (not is_allow_unset):
                    raise KeyError(f"Missing default for field {key!r}.")
                setattr(self, key, value)

        object.__setattr__(self, "impl_is_functioning", True)

    def _helper_asdict_basic(self, *, is_include_unset: bool = False) -> dict[str, Any]:
        """Return the current public option payload as a plain dictionary."""
        result: dict[str, Any] = {}
        for key in type(self).__attrs__:
            value = getattr(self, key)
            if (not is_include_unset) and (value is UNSET):
                continue
            result[key] = value
        return result

    @contextmanager
    def _helper_internal_update(self):
        """Temporarily suspend the functioning lifecycle state."""
        is_functioning_current = getattr(self, "impl_is_functioning", False)
        object.__setattr__(self, "impl_is_functioning", False)
        try:
            yield
        finally:
            object.__setattr__(
                self,
                "impl_is_functioning",
                is_functioning_current,
            )

    # ------------------------------------------------------------------
    # Public actions
    # ------------------------------------------------------------------

    def act_finalize(
        self,
        defaults: Mapping[str, Any] | None = None,
        is_allow_unset: bool = False,
    ) -> None:
        """Finalize this opts instance by filling defaults and freezing lifecycle."""
        self._helper_finalize_basic(defaults, is_allow_unset=is_allow_unset)

    def act_asdict(self, is_include_unset: bool = False) -> dict[str, Any]:
        """Return the current option payload as a plain dictionary."""
        return self._helper_asdict_basic(is_include_unset=is_include_unset)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_save_json(
        self,
        path: str | Path,
        *,
        max_inline_array_size: int = 64,
        is_include_unset: bool = False,
        logger=None,
    ) -> Path:
        """Serialize the current opts payload to a JSON file."""
        path = save_opts_json(
            self.act_asdict(is_include_unset=is_include_unset),
            path,
            opts_class_name=type(self).__name__,
            max_inline_array_size=max_inline_array_size,
        )
        logger.info(f"Saved opts JSON to {path}.")
        return path

    def act_load_json(
        self,
        path: str | Path,
        *,
        is_finalize: bool = False,
    ):
        """Load saved JSON data back into this opts instance."""
        return load_json_into_opts(
            self,
            path,
            is_finalize=is_finalize,
        )

    # ------------------------------------------------------------------
    # Object protocol
    # ------------------------------------------------------------------

    # ==================== OVERRIDE ====================
    # OptsBase overrides object.__setattr__ so every public opts assignment
    # runs through validation, lifecycle checks, and optional host forwarding.
    # ==================================================
    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)

    # ==================== OVERRIDE ====================
    # OptsBase overrides object.__str__ to keep opts instances readable as a
    # short one-line identity in logs and interactive inspection.
    # ==================================================
    def __str__(self) -> str:
        return type(self).__name__

    # ==================== OVERRIDE ====================
    # OptsBase overrides object.__repr__ so the full public opts payload is
    # visible during debugging and interactive exploration.
    # ==================================================
    def __repr__(self) -> str:
        cls_name = type(self).__name__

        host = self.host
        if host is not None:
            lines = [f"{cls_name}: the options of {str(host)}"]
        else:
            lines = [f"{cls_name}"]

        keys = list(type(self).__attrs__)
        if not keys:
            return "\n".join(lines)

        width = max(len(key) for key in keys)
        for key in keys:
            try:
                value = getattr(self, key)
            except AttributeError:
                value = "<missing>"
            lines.append(f"  {key:<{width}} = {repr_format(value)}")

        return "\n".join(lines)


class HostBase(ClassBase):
    """
    Shared host controller for objects driven by an associated ``OptsBase``.

    ``HostBase`` combines the object identity and relation model of
    ``ClassBase`` with a paired opts object and a managed commit pipeline.
    For most package users, a HostBase-style object provides:

    - a normal object identity and relation interface inherited from
      ``ClassBase``
    - a paired ``.opts`` object that stores configurable parameters
    - a commit-style update path instead of ad hoc direct mutation
    - inspection helpers for readable attrs, modifiable attrs, relations, and
      saved opts snapshots

    The host-side readable surface is centered on:

    - ``opts`` for the paired options object that controls host behavior
    - ``opts_defaults`` for the default opts payload used by initialization and
      later reapplication
    - ``opts_backup`` for named snapshots of opts dictionaries
    - ``attrs_forbidden`` for the current union of wrapped and directly
      protected public attrs

    Important user-facing inspection helpers are:

    - ``show_readable_attrs()`` to list readable host fields and opts attrs
    - ``show_attr_desc()`` to explain one host attr, relation, alias, extra
      attr, or opts attr
    - ``show_modifiable_attrs()`` to separate host attrs, opts attrs, extra
      attrs, and writable host properties
    - ``show_relations()`` / ``show_relation_tree()`` to inspect object links
    - ``show_saved_opts()`` to list named snapshots stored in ``opts_backup``

    Important user-facing actions include both the inherited ``ClassBase``
    actions and host-specific commit utilities:

    - ``act_commit()`` to apply host and opts updates through the managed
      commit pipeline
    - ``act_save_opts()`` to snapshot current opts into ``opts_backup``
    - ``act_attach_sync_task()`` / ``act_detach_sync_task()`` and the related
      enrich-kwargs registration helpers
    - ``act_register_wrapped_attr()`` / ``act_unregister_wrapped_attr()`` and
      ``act_bind_wrapper()`` / ``act_unbind_wrapper()`` for wrapper forwarding
    - ``act_register_protected_attr()`` /
      ``act_unregister_protected_attr()`` to protect or unprotect public host
      and opts attrs

    Concrete subclasses are expected to choose when opts are finalized and to
    implement ``_helper_commit_apply_opts_main()`` so finalized opts changes
    can be translated into actual host-side state updates.
    """

    __attr_defs__ = {
        **ClassBase.__attr_defs__,
        "opts": {
            "doc": "The Opts instance controlling options.",
            "kind": "opts",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "opts_defaults": {
            "doc": "The default option settings.",
            "kind": "opts",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "opts_backup": {
            "doc": (
                "A dictionary storing potentially useful options, indexed by "
                "timestamp or a manual key."
            ),
            "kind": "opts",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "impl_sync_func": {
            "doc": "A dictionary of callback functions for post-commit synchronization.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "impl_attrs_wrapped": {
            "doc": "Protected attributes under wrapping.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "impl_attrs_protected": {
            "doc": "Additional protected attributes declared directly by this host.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "impl_enrich_kwargs_wrapped_func": {
            "doc": "Callback functions that enrich forwarded kwargs for wrapped hosts.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "impl_enrich_kwargs_sync_func": {
            "doc": "Callback functions that enrich sync kwargs before sync execution.",
            "kind": "impl",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
        "wrapper": {
            "doc": "The wrapper host that controls this host.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "wrapped": {
            "doc": "The wrapped host controlled by this host as a wrapper.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "attrs_forbidden": {
            "doc": (
                "Read-only union of wrapped attrs and host-declared " "protected attrs."
            ),
            "kind": "property",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
    }

    __slots__ = (
        "opts",
        "opts_defaults",
        "opts_backup",
        "impl_sync_func",
        "impl_attrs_wrapped",
        "impl_attrs_protected",
        "impl_enrich_kwargs_wrapped_func",
        "impl_enrich_kwargs_sync_func",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.__init__ because a host must bind a paired
    # OptsBase instance and initialize host-side runtime stores in addition to
    # the base ClassBase identity and relation skeleton.
    # ==================================================
    def __init__(
        self,
        opts_type: Type[OptsBase],
        opts: OptsBase | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        name: str | None = None,
        name_replace: str = "unnamed",
        **kwargs,
    ):
        # Initialize the ClassBase identity and base relation skeleton first.
        super().__init__(name=name, name_replace=name_replace)

        # Split out host-side initialization kwargs so opt kwargs can be
        # merged into the paired opts object separately.
        kwargs_host = {}
        for key in list(kwargs):
            if key in self.impl_attrs and (
                key.startswith("raw_") or key.startswith("state_")
            ):
                kwargs_host[key] = kwargs.pop(key)
            elif f"raw_{key}" in self.impl_attrs:
                kwargs_host[key] = kwargs.pop(key)

        # Normalize or create the paired opts instance, then merge any
        # remaining option kwargs into it.
        opts = self._helper_check_opts(opts, opts_type=opts_type)
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "impl_host_ref", weakref.ref(self))
        object.__setattr__(self, "opts", opts)

        # Build the frozen-on-init opts default payload used by later host
        # commit/finalize steps.
        opts_defaults = {
            **{key: UNSET for key in type(opts).__attrs__},
            **dict(opts.defaults_frozen),
        }
        opts_defaults = build_dict_override(
            opts_defaults,
            opts_defaults_override,
            name=type(opts).__name__,
        )

        # Initialize host-side runtime stores for opts snapshots, sync hooks,
        # wrapped attr bookkeeping, and host-declared protected attrs.
        object.__setattr__(self, "opts_defaults", opts_defaults)
        object.__setattr__(self, "opts_backup", {})
        object.__setattr__(self, "impl_sync_func", {})
        object.__setattr__(self, "impl_attrs_protected", set())
        object.__setattr__(self, "impl_enrich_kwargs_wrapped_func", {})
        object.__setattr__(self, "impl_enrich_kwargs_sync_func", {})
        object.__setattr__(self, "impl_attrs_wrapped", set())

        # Apply any host-side raw/state initialization values that were
        # separated from the opts kwargs above.
        if kwargs_host:
            self._helper_commit_raw(kwargs_host)

        # HostBase intentionally stops here.
        # Concrete subclasses remain responsible for choosing when to finalize
        # opts and when to trigger the first opts->host application pass.

    # ------------------------------------------------------------------
    # Readable properties / basic helpers
    # ------------------------------------------------------------------

    @property
    def attrs_forbidden(self) -> set[str]:
        """Return the union of wrapped attrs and host-declared protected attrs."""
        return set(self.impl_attrs_wrapped) | set(self.impl_attrs_protected)

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_opts(
        self,
        opts: OptsBase | None,
        opts_type: Type[OptsBase] | None = None,
        logger=None,
    ) -> OptsBase:
        """Normalize one opts input against the required opts class."""
        if opts_type is None:
            opts_type = type(self.opts)

        if opts is None:
            return opts_type()

        if not isinstance(opts, opts_type):
            try:
                raise TypeError(
                    f"opts must be an instance of {opts_type.__name__}, "
                    f"got {type(opts).__name__}."
                )
            except TypeError:
                logger.exception("Check input.")
                logger.recovery(
                    f"Create a default instance of {opts_type.__name__} instead."
                )
            return opts_type()

        return opts

    # ------------------------------------------------------------------
    # Commit pipeline
    # ------------------------------------------------------------------

    def act_commit(
        self,
        opts: OptsBase | None = None,
        opts_wrapped: OptsBase | None = None,
        is_reapply_opts: bool = False,
        **kwargs,
    ) -> None:
        """Apply host and opts updates through the managed commit pipeline."""
        self._helper_pop_private_key(kwargs)
        kwargs_sync, is_reapply_opts_from_raw = self._helper_commit_pre_opts(kwargs)
        is_reapply_opts = is_reapply_opts or is_reapply_opts_from_raw

        opts_keys = type(self.opts).__attrs__
        is_opts_request = (opts is not None) or any(key in opts_keys for key in kwargs)
        if is_reapply_opts or is_opts_request:
            kwargs, kwargs_applied_opts = self._helper_commit_self(
                opts=opts,
                is_reapply_opts=is_reapply_opts,
                **kwargs,
            )
            kwargs_sync = kwargs_sync | kwargs_applied_opts

        kwargs_sync = self._helper_commit_enrich_kwargs_sync(kwargs_sync)
        if kwargs_sync:
            self._helper_trigger_sync_batch(**kwargs_sync)

        self._helper_kwargs_to_wrapped(kwargs, opts_wrapped=opts_wrapped)

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_pop_private_key(self, kwargs: dict[str, Any], logger=None) -> None:
        """Drop private commit keys that should never be accepted publicly."""
        private_keys = [
            key
            for key in list(kwargs)
            if key.startswith("_") or key.startswith("impl_")
        ]
        for key in private_keys:
            kwargs.pop(key)
            logger.warning(
                f"{key!r} is not a valid public commit key. "
                "Names starting with '_' or 'impl_' are not accepted by act_commit()."
            )

    def _helper_commit_pre_opts(
        self,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Consume host-side public updates before opts processing begins."""
        if not kwargs:
            return {}, False

        self._helper_check_protected_attr(kwargs)
        kwargs_applied_name = self._helper_commit_name(kwargs)
        kwargs_applied_raw, is_reapply_opts = self._helper_commit_raw(kwargs)
        return kwargs_applied_raw | kwargs_applied_name, is_reapply_opts

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_protected_attr(self, kwargs: dict[str, Any], logger=None) -> None:
        """Remove forbidden public commit keys before any assignment happens."""
        if not kwargs:
            return

        blocked = [key for key in list(kwargs) if key in self.attrs_forbidden]
        for key in blocked:
            kwargs.pop(key)
            try:
                raise AttributeError(
                    f"{key!r} is protected and could not be directly modified."
                )
            except AttributeError:
                logger.exception("Invalid attr")
                logger.recovery("Automatically ignore this attr")

    def _helper_commit_name(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Consume a name/raw_name update if present."""
        if not kwargs:
            return {}

        found, name = pop_exclusive(kwargs, "name", "raw_name")
        if not found:
            return {}

        attr_info = self.impl_attrs["raw_name"]
        name = attr_info["validator"](
            name,
            name=attr_info["doc"],
        )
        self._helper_assign_name(name)
        return {"name": self.name}

    def _helper_commit_raw(
        self,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Consume host-side raw/state updates and write them directly to the host."""
        if not kwargs:
            return {}, False

        kwargs_applied_raw: dict[str, Any] = {}
        is_reapply_opts = False
        for key in list(kwargs):
            is_host_attr = (
                (key in self.impl_attrs)
                and (self.impl_attrs[key]["kind"] in {"raw", "state"})
            ) or (f"raw_{key}" in self.impl_attrs)
            if is_host_attr:
                kwargs_applied_here, is_reapply_opts_here = self._helper_commit_pop_raw(
                    kwargs,
                    key,
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
    ) -> tuple[dict[str, Any], bool]:
        """Consume one host raw/state update and apply validation when needed."""
        is_state_attr = attr_name_origin.startswith("state_")

        if attr_name_origin.startswith("raw_"):
            host_attr_name = attr_name_origin
            public_attr_name = attr_name_origin[4:]
            found, attr_value = pop_exclusive(kwargs, public_attr_name, host_attr_name)
            attr_name_return = public_attr_name
        elif is_state_attr:
            host_attr_name = attr_name_origin
            found = host_attr_name in kwargs
            attr_value = kwargs.pop(host_attr_name) if found else None
            attr_name_return = host_attr_name
        else:
            host_attr_name = f"raw_{attr_name_origin}"
            public_attr_name = attr_name_origin
            found, attr_value = pop_exclusive(kwargs, public_attr_name, host_attr_name)
            attr_name_return = public_attr_name

        if not found:
            return {}, False

        if exception_msg is None:
            exception_msg = (
                f"Validation failed for attribute {attr_name_return!r}. "
                "The validator must accept two arguments: (value, description)."
            )
        if recovery_msg is None:
            recovery_msg = f"Ignore this modification of {attr_name_return!r}."

        if validator is None:
            validator = self.impl_attrs[host_attr_name].get("validator")

        try:
            if validator is not None:
                attr_value = validator(
                    attr_value, self.impl_attrs[host_attr_name]["doc"]
                )

            if host_attr_name == "raw_name":
                self._helper_assign_name(attr_value)
            else:
                object.__setattr__(self, host_attr_name, attr_value)
        except (TypeError, ValueError, KeyError, AttributeError):
            logger.exception(exception_msg)
            logger.recovery(recovery_msg)
            return {}, False

        return {attr_name_return: attr_value}, bool(
            self.impl_attrs[host_attr_name].get("is_reapply_opts_after_raw", False)
        )

    def _helper_commit_self(
        self,
        opts: OptsBase | None = None,
        is_reapply_opts: bool = False,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply updates that belong to this host's paired opts domain."""
        if not (kwargs or opts or is_reapply_opts):
            return kwargs, {}

        self_keys = type(self.opts).__attrs__
        kwargs_self = {key: kwargs.pop(key) for key in list(kwargs) if key in self_keys}
        kwargs_self = self._helper_merge_opts_kwargs(opts=opts, **kwargs_self)
        kwargs_left, kwargs_applied_opts = self._helper_commit_apply_opts(
            is_reapply_opts=is_reapply_opts,
            **kwargs_self,
        )
        kwargs |= kwargs_left
        return kwargs, kwargs_applied_opts

    def _helper_merge_opts_kwargs(
        self,
        opts: OptsBase | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Merge explicit opts plus opts-like kwargs into one plain opts payload."""
        if not (kwargs or opts):
            return {}

        opts = self._helper_check_opts(opts)
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        return opts.act_asdict()

    def _helper_commit_apply_opts(
        self,
        is_reapply_opts: bool = False,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply opts updates and return leftover kwargs plus applied opts changes."""
        self._helper_check_protected_attr(kwargs)
        opts_before = self.opts.act_asdict()
        kwargs_applied_opts: dict[str, Any] = {}

        if "tag" in kwargs:
            tag_value = kwargs.pop("tag")
            object.__setattr__(self.opts, "tag", tag_value)
            kwargs_applied_opts["tag"] = tag_value

        return_main = self._helper_commit_apply_opts_main(
            is_reapply_opts=is_reapply_opts,
            **kwargs,
        )
        if return_main is None:
            kwargs_left = {}
            opts_after = self.opts.act_asdict()
            _, kwargs_applied_opts_main = diff_dict_values(opts_before, opts_after)
        else:
            kwargs_left, kwargs_applied_opts_main = return_main

        kwargs_applied_opts |= kwargs_applied_opts_main
        return kwargs_left, kwargs_applied_opts

    def _helper_commit_apply_opts_main(
        self,
        is_reapply_opts: bool = False,
        **kwargs,
    ):
        """Subclass hook for applying opts updates to concrete host state."""
        del is_reapply_opts, kwargs
        raise NotImplementedError(
            f"{type(self).__name__} must implement _helper_commit_apply_opts_main()."
        )

    # ------------------------------------------------------------------
    # Sync and wrapped forwarding
    # ------------------------------------------------------------------

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_trigger_sync_batch(self, logger=None, **kwargs) -> None:
        """Run all registered sync callbacks with the merged sync payload."""
        for name, func in self.impl_sync_func.items():
            try:
                func(**kwargs)
            except (
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
                RuntimeError,
            ) as exc:
                logger.exception(f"Sync task {name!r} failed: {exc}")
                logger.recovery("Automatically skip this function.")

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_enrich_kwargs_sync(
        self,
        kwargs_sync: dict[str, Any],
        logger=None,
    ) -> dict[str, Any]:
        """Allow registered callbacks to enrich the sync payload before execution."""
        kwargs_sync_out = dict(kwargs_sync)
        for name, func in self.impl_enrich_kwargs_sync_func.items():
            try:
                output = func(host=self, kwargs_sync=kwargs_sync_out)
                if output is not None:
                    kwargs_sync_out = output
            except (
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
                RuntimeError,
            ) as exc:
                logger.exception(f"Sync kwargs task {name!r} failed: {exc}")
                logger.recovery("Automatically skip this function.")
        return kwargs_sync_out

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_enrich_kwargs_wrapped(
        self,
        kwargs: dict[str, Any],
        logger=None,
    ) -> dict[str, Any]:
        """Allow registered callbacks to enrich forwarded kwargs for the wrapped host."""
        kwargs_wrapped = dict(kwargs)
        for name, func in self.impl_enrich_kwargs_wrapped_func.items():
            try:
                output = func(host=self, kwargs=kwargs_wrapped)
                if output is not None:
                    kwargs_wrapped |= output
            except (
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
                RuntimeError,
            ) as exc:
                logger.exception(f"Wrapped kwargs task {name!r} failed: {exc}")
                logger.recovery("Automatically skip this function.")
        return kwargs_wrapped

    def act_attach_enrich_kwargs_sync_task(self, name: str, func: Callable) -> None:
        """Register one sync-kwargs enrichment callback."""
        if not callable(func):
            raise TypeError(f"The sync kwargs task {name!r} must be callable.")
        self.impl_enrich_kwargs_sync_func[name] = func

    def act_detach_enrich_kwargs_sync_task(self, name: str) -> None:
        """Detach one sync-kwargs enrichment callback."""
        self.impl_enrich_kwargs_sync_func.pop(name, None)

    def act_attach_enrich_kwargs_wrapped_task(self, name: str, func: Callable) -> None:
        """Register one wrapped-kwargs enrichment callback."""
        if not callable(func):
            raise TypeError(f"The wrapped kwargs task {name!r} must be callable.")
        self.impl_enrich_kwargs_wrapped_func[name] = func

    def act_detach_enrich_kwargs_wrapped_task(self, name: str) -> None:
        """Detach one wrapped-kwargs enrichment callback."""
        self.impl_enrich_kwargs_wrapped_func.pop(name, None)

    def act_attach_sync_task(self, name: str, func: Callable) -> None:
        """Register one post-commit sync callback."""
        if not callable(func):
            raise TypeError(f"The sync task {name!r} must be callable.")
        self.impl_sync_func[name] = func

    def act_detach_sync_task(self, name: str) -> None:
        """Detach one post-commit sync callback."""
        self.impl_sync_func.pop(name, None)

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_kwargs_to_wrapped(
        self,
        kwargs: dict[str, Any],
        opts_wrapped: OptsBase | None = None,
        logger=None,
    ) -> None:
        """Forward leftover kwargs and opts to the wrapped host when present."""
        kwargs_wrapped = self._helper_commit_enrich_kwargs_wrapped(kwargs)
        if not (kwargs_wrapped or opts_wrapped):
            return

        if self.wrapped is not None:
            with self.wrapped.act_wrapped_update():
                self.wrapped.act_commit(opts=opts_wrapped, **kwargs_wrapped)
            return

        cls_name = type(self).__name__
        obj_name = getattr(self, "raw_name", "Uninitialized")
        lines = [f"[{cls_name}: {obj_name!r}] Unhandled commit arguments."]
        if kwargs_wrapped:
            lines.append(f"  Remaining kwargs keys: {list(kwargs_wrapped)}")
        if opts_wrapped is not None:
            lines.append(f"  opts_wrapped: {opts_wrapped!r}")
        logger.warning("\n".join(lines))

    # ------------------------------------------------------------------
    # Saved opts
    # ------------------------------------------------------------------

    def act_save_opts(self, name: str | None = None) -> None:
        """Save the current opts payload into the host snapshot dictionary."""
        if not name:
            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.opts_backup[name] = self.opts.act_asdict()

    # ------------------------------------------------------------------
    # Attribute inspection
    # ------------------------------------------------------------------

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.show_readable_attrs so the readable surface also
    # includes the paired opts fields in addition to host-side attrs.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def show_readable_attrs(self, is_return=False, logger=None):
        """Show readable host and opts-facing surfaces."""
        lines = [
            "When reading host fields, the 'raw_' prefix may be omitted "
            "where a public alias exists."
        ]

        host_names = sorted(
            name
            for name in self._helper_collect_readable_names(is_exclude_impl=True)
            if name not in {"opts", "opts_defaults"}
        )
        for attr_name in host_names:
            lines.append(self.show_attr_desc(attr_name))

        if self.opts is not None:
            lines.append("[Opts attributes]")
            for attr_name in type(self.opts).__attrs__:
                lines.append(self.show_attr_desc(attr_name))

        if len(lines) == 1:
            lines.append("<none>")

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.show_attr_desc so descriptions can be
    # resolved from both the host layer and the paired opts layer.
    # ==================================================
    def show_attr_desc(self, attr_name: str) -> str:
        """Return a description from the host layer or the paired opts layer."""
        try:
            return super().show_attr_desc(attr_name)
        except KeyError:
            pass

        opts = getattr(self, "opts", None)
        if opts is not None:
            descriptions_opts = type(opts).__attrs__
            if attr_name in descriptions_opts:
                return f"{attr_name!r}: {descriptions_opts[attr_name]}"
            raise KeyError(
                f"Attribute {attr_name!r} was not found in "
                f"{type(self).__name__}.impl_attrs or {type(opts).__name__}.__attrs__."
            )

        raise KeyError(
            f"Attribute {attr_name!r} was not found in "
            f"{type(self).__name__}.impl_attrs. "
            "The opts attrs are not available yet because self.opts has not been "
            "initialized; the attribute may belong to opts."
        )

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.show_modifiable_attrs so writable surfaces
    # are presented by host-side attrs, opts attrs, and host properties.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def show_modifiable_attrs(self, is_return=False, logger=None):
        """Show modifiable host and opts attributes by category."""
        lines = [
            "When assigning host fields, the 'raw_' prefix may be omitted.",
        ]

        attrs_forbidden = self.attrs_forbidden
        attrs_host = []
        attrs_opts = []
        attrs_extra = []
        attrs_properties = []

        for attr_name, attr_info in self.impl_attrs.items():
            if attr_info["kind"] == "property":
                if attr_info.get("is_public_settable", False):
                    attrs_properties.append(attr_name)
                continue

            if attr_info["kind"] not in {"raw", "state", "extra"}:
                continue
            if not attr_info["is_public_settable"]:
                continue
            if attr_info["is_protected"]:
                continue
            if attr_name in attrs_forbidden:
                continue

            if attr_info["kind"] == "extra":
                attrs_extra.append(attr_name)
                continue

            attrs_host.append(attr_name)

        for attr_name in type(self.opts).__attrs__:
            if attr_name in attrs_forbidden:
                continue
            attrs_opts.append(attr_name)

        if "tag" in attrs_opts:
            attrs_opts.remove("tag")
            attrs_opts.insert(0, "tag")

        if attrs_host:
            lines.append("[Host attributes]")
            for attr_name in sorted(attrs_host):
                lines.append(f"  - {self.show_attr_desc(attr_name)}")
        else:
            lines.append("[Host attributes]")
            lines.append("  - <none>")

        if attrs_opts:
            lines.append("[Opts attributes]")
            for attr_name in sorted(attrs_opts):
                lines.append(f"  - {self.show_attr_desc(attr_name)}")
        else:
            lines.append("[Opts attributes]")
            lines.append("  - <none>")

        if attrs_extra:
            lines.append("[Extra attributes]")
            for attr_name in sorted(attrs_extra):
                lines.append(f"  - {self.show_attr_desc(attr_name)}")

        if attrs_properties:
            lines.append("[Host writable properties]")
            for attr_name in sorted(attrs_properties):
                lines.append(f"  - {self.show_attr_desc(attr_name)}")
        else:
            lines.append("[Host writable properties]")
            lines.append("  - <none>")

        if (
            (not attrs_host)
            and (not attrs_opts)
            and (not attrs_extra)
            and (not attrs_properties)
        ):
            lines.append("  (None)")

        if attrs_forbidden:
            lines.append(
                "Protected or wrapped fields are excluded from the lists above "
                "and cannot be modified through normal commit/setattr paths."
            )

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    @logging_and_warning_decorator(start_finish_level=5)
    def show_saved_opts(self, is_return=False, logger=None):
        """Show saved option snapshots currently stored on this host."""
        lines = ["Saved opts snapshots in 'opts_backup':"]
        if self.opts_backup:
            for key in self.opts_backup:
                lines.append(f"  - {key}")
        else:
            lines.append("  - <none>")

        lines.append("Use `self.opts_backup[name]` to inspect a full saved opts dict.")
        lines.append(
            "To restore one manually, call `self.act_commit(**self.opts_backup[name])`."
        )
        lines.append(
            "To compare two saved opts dictionaries, use `diff_dict_values(dict1, dict2)` "
            "from `nematics3d.classes.opts`."
        )

        output = "\n".join(lines)
        logger.info(output)
        if is_return:
            return output
        return None

    # ------------------------------------------------------------------
    # Protection and wrapping
    # ------------------------------------------------------------------

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_register_protected_attr(
        self,
        attrs: Sequence[str] | str,
        *,
        target_set: set[str],
        attr_name: str,
        logger=None,
    ) -> None:
        """Register host/opts public names into one protection target set."""
        for attr in as_list(attrs, name="attrs"):
            try:
                attr = as_str(attr, name=attr_name)
                if attr.startswith("raw_"):
                    if (
                        attr in self.impl_attrs
                        and self.impl_attrs[attr]["kind"] == "raw"
                    ):
                        target_set.update([attr, attr[4:]])
                    else:
                        raise AttributeError(
                            f"Attribute {attr!r} is not a valid public host attr."
                        )
                elif attr.startswith("state_"):
                    if (
                        attr in self.impl_attrs
                        and self.impl_attrs[attr]["kind"] == "state"
                    ):
                        target_set.add(attr)
                    else:
                        raise AttributeError(
                            f"Attribute {attr!r} is not a valid public host state attr."
                        )
                elif attr in type(self.opts).__attrs__:
                    target_set.add(attr)
                elif f"raw_{attr}" in self.impl_attrs:
                    target_set.update([attr, f"raw_{attr}"])
                else:
                    raise AttributeError(
                        f"Attribute {attr!r} is not a valid public host or opts attr."
                    )
            except (TypeError, ValueError, KeyError, AttributeError):
                logger.exception("Invalid attr name.")
                logger.recovery("Automatically ignore this attr.")

    def act_register_wrapped_attr(self, attrs: Sequence[str] | str) -> None:
        """Register a group of public attributes as protected under wrapping."""
        self._helper_register_protected_attr(
            attrs,
            target_set=self.impl_attrs_wrapped,
            attr_name="The name of attr to be wrapped",
        )

    def act_unregister_wrapped_attr(
        self,
        attrs: Sequence[str] | str | None = None,
    ) -> None:
        """Remove wrapped protection from one or more public attributes."""
        if attrs is None:
            self.impl_attrs_wrapped.clear()
            return

        for attr in as_list(attrs, name="attrs"):
            attr = as_str(attr, name="The name of attr to be unwrapped")
            if attr.startswith("raw_"):
                self.impl_attrs_wrapped.discard(attr)
                self.impl_attrs_wrapped.discard(attr[4:])
            elif attr.startswith("state_"):
                self.impl_attrs_wrapped.discard(attr)
            else:
                self.impl_attrs_wrapped.discard(attr)
                self.impl_attrs_wrapped.discard(f"raw_{attr}")

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.act_register_protected_attr because
    # protected names may belong either to the host itself or to its paired
    # opts object.
    # ==================================================
    def act_register_protected_attr(self, attrs: Sequence[str] | str) -> None:
        """Register a group of public attributes as directly protected."""
        self._helper_register_protected_attr(
            attrs,
            target_set=self.impl_attrs_protected,
            attr_name="The name of attr to be protected",
        )

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.act_unregister_protected_attr because the
    # protected-name surface may include host aliases and paired opts attrs.
    # ==================================================
    def act_unregister_protected_attr(self, attrs: Sequence[str] | str) -> None:
        """Remove direct protection from one or more host/opts public names."""
        for attr in as_list(attrs, name="attrs"):
            attr = as_str(attr, name="The name of attr to be unprotected")
            if attr.startswith("raw_"):
                self.impl_attrs_protected.discard(attr)
                self.impl_attrs_protected.discard(attr[4:])
            elif attr.startswith("state_"):
                self.impl_attrs_protected.discard(attr)
            else:
                self.impl_attrs_protected.discard(attr)
                self.impl_attrs_protected.discard(f"raw_{attr}")

    @contextmanager
    def act_wrapped_update(self):
        """Temporarily disable wrapped protection within a managed context."""
        protected_backup = set(self.impl_attrs_wrapped)
        self.impl_attrs_wrapped.clear()
        try:
            yield
        finally:
            self.impl_attrs_wrapped.update(protected_backup)

    def act_bind_wrapper(
        self,
        wrapper: "HostBase",
        protected_attrs: Sequence[str] | str | None = None,
    ) -> None:
        """Bind one wrapper host and optionally register wrapped protected attrs."""
        old_wrapper = self.wrapper
        if old_wrapper is not None and (old_wrapper is not wrapper):
            raise RuntimeError(
                f"{type(self).__name__} is already wrapped by {type(old_wrapper).__name__}."
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

    def act_unbind_wrapper(self) -> None:
        """Detach the current wrapper relation and clear wrapped protection."""
        wrapper = self.wrapper
        if wrapper is not None and wrapper.wrapped is self:
            wrapper.act_unbind_relation_base("wrapped")
        self.act_unbind_relation_base("wrapper")
        self.act_unregister_wrapped_attr()

    # ------------------------------------------------------------------
    # Object protocol
    # ------------------------------------------------------------------

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase._helper_setattr_final so validated public
    # host assignment is routed through the managed act_commit() pipeline.
    # ==================================================
    def _helper_setattr_final(self, key, value, *, target_key=None):
        """Route validated public host assignment through the commit pipeline."""
        del target_key
        self.act_commit(**{key: value})

    # ==================== OVERRIDE ====================
    # HostBase overrides ClassBase.__setattr__ so paired opts fields are also
    # routed through the managed act_commit() pipeline.
    # ==================================================
    def __setattr__(self, key, value):
        is_opts_key = hasattr(self, "opts") and (key in type(self.opts).__attrs__)

        if (not key.startswith("_")) and is_opts_key:
            self.act_commit(**{key: value})
            return

        super().__setattr__(key, value)



