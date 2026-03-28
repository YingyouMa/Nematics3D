"""
Host-side option foundations for Nematics3D objects.

This module currently provides ``OptsBase``, the validated options container
used by Host-style classes. The implementation stays close to the original
HostBase design: public option fields remain explicit dataclass slots, runtime
host wiring stays in ``impl_*`` fields, lifecycle state stays in
``state_*`` fields, and user-facing convenience access is exposed through
small readable properties.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Type
import weakref

from ..datatypes import UNSET, Unset, as_str
from ..format import repr_format, save_opts_json
from ..logging_decorator import logging_and_warning_decorator
from .class_base import ClassBase
from .opts import build_dict_override, load_json_into_opts, merge_opts_all


@dataclass(slots=True, repr=False)
class OptsBase:
    """
    Reactive validated configuration base for Host-style objects.

    ``OptsBase`` stays intentionally close to the original host-side design:
    public option fields such as ``tag`` are stored directly on the instance,
    host wiring uses ``impl_*`` storage, and lifecycle state uses
    ``state_*`` storage.

    Readable convenience properties include:
    - ``host`` for the attached host object, if any
    - ``is_functioning`` for the finalized lifecycle state
    - ``defaults_frozen`` for the class-level frozen defaults mapping
    """

    tag: str | Unset = UNSET

    impl_host_ref: weakref.ReferenceType | None = field(
        default=None,
        init=False,
        repr=False,
    )
    state_is_functioning: bool = field(
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

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({"tag": "options"})

    # ------------------------------------------------------------------
    # Readable properties
    # ------------------------------------------------------------------

    @property
    def host(self):
        """Return the attached host object, if the stored weakref is alive."""
        host_ref = self.impl_host_ref
        return host_ref() if host_ref is not None else None

    @property
    def is_functioning(self) -> bool:
        """Return whether this opts instance has already been finalized."""
        return bool(self.state_is_functioning)

    @property
    def defaults_frozen(self) -> Mapping[str, Any]:
        """Expose the frozen defaults mapping without requiring the private name."""
        return type(self)._DEFAULTS_FROZEN

    # ------------------------------------------------------------------
    # Basic core
    # ------------------------------------------------------------------

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_setattr_basic(self, key: str, value: Any, *, logger=None) -> None:
        """Validate one assignment and forward live option updates to the host."""
        is_functioning = self.is_functioning
        is_has_host = self.host is not None

        if (not key.startswith("_")) and (key not in type(self).__attrs__):
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
        elif key.startswith("_") or (not is_functioning):
            object.__setattr__(self, key, value)
            return

        if (
            (not key.startswith("_"))
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

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_finalize_basic(
        self,
        defaults: Mapping[str, Any] | None = None,
        is_allow_unset: bool = False,
        *,
        logger=None,
    ) -> None:
        """Fill ``UNSET`` values by defaults, then enter the functioning state."""
        del logger

        if self.state_is_functioning:
            raise RuntimeError("This Opts has already been finalized.")

        defaults_dict = {} if defaults is None else dict(defaults)

        for key in type(self).__attrs__:
            if getattr(self, key) is UNSET:
                value = defaults_dict.get(key, self.defaults_frozen.get(key, UNSET))
                if (value is UNSET) and (not is_allow_unset):
                    raise KeyError(f"Missing default for field {key!r}.")
                setattr(self, key, value)

        object.__setattr__(self, "state_is_functioning", True)

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
        is_functioning_current = self.state_is_functioning
        object.__setattr__(self, "state_is_functioning", False)
        try:
            yield
        finally:
            object.__setattr__(self, "state_is_functioning", is_functioning_current)

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

    @logging_and_warning_decorator(start_finish_level=5)
    def act_load_json(
        self,
        path: str | Path,
        *,
        is_finalize: bool = False,
        logger=None,
    ):
        """Load saved JSON data back into this opts instance."""
        del logger
        return load_json_into_opts(
            self,
            path,
            is_finalize=is_finalize,
        )

    # ------------------------------------------------------------------
    # Object protocol
    # ------------------------------------------------------------------

    def __setattr__(self, key, value):
        self._helper_setattr_basic(key, value)

    def __str__(self) -> str:
        return type(self).__name__

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
    Minimal host controller built on top of ``ClassBase`` and ``OptsBase``.

    ``HostBase`` extends the current ``ClassBase`` attribute-definition model
    with host-specific runtime storage for paired opts, opts defaults, saved
    opts backups, wrapped/protected attr bookkeeping, and sync callback
    registries.
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
        if opts is None:
            opts = opts_type()
        elif not isinstance(opts, opts_type):
            raise TypeError(
                f"opts must be an instance of {opts_type.__name__}, "
                f"got {type(opts).__name__}."
            )

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
            for key, value in kwargs_host.items():
                target_key = key if key in self.impl_attrs else f"raw_{key}"
                object.__setattr__(self, target_key, value)

        # Remaining work for HostBase.__init__:
        # - finalize opts at the appropriate lifecycle stage
        # - define how finalized opts are consumed and applied by the host
