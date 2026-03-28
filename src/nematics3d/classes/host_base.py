"""
Host-side option foundations for Nematics3D objects.

This module currently provides ``OptsBase``, the validated options container
used by Host-style classes. The implementation stays close to the original
HostBase design: public option fields remain explicit dataclass slots, runtime
host wiring stays in ``_impl_*`` fields, lifecycle state stays in
``_state_*`` fields, and user-facing convenience access is exposed through
small readable properties.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping
import weakref

from ..datatypes import UNSET, Unset, as_str
from ..format import repr_format, save_opts_json
from ..logging_decorator import logging_and_warning_decorator
from .opts import load_json_into_opts


@dataclass(slots=True, repr=False)
class OptsBase:
    """
    Reactive validated configuration base for Host-style objects.

    ``OptsBase`` stays intentionally close to the original host-side design:
    public option fields such as ``tag`` are stored directly on the instance,
    host wiring uses ``_impl_*`` storage, and lifecycle state uses
    ``_state_*`` storage.

    Readable convenience properties include:
    - ``host`` for the attached host object, if any
    - ``is_functioning`` for the finalized lifecycle state
    - ``defaults_frozen`` for the class-level frozen defaults mapping
    """

    tag: str | Unset = UNSET

    _impl_host_ref: weakref.ReferenceType | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _state_is_functioning: bool = field(
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
        host_ref = self._impl_host_ref
        return host_ref() if host_ref is not None else None

    @property
    def is_functioning(self) -> bool:
        """Return whether this opts instance has already been finalized."""
        return bool(self._state_is_functioning)

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

        if self._state_is_functioning:
            raise RuntimeError("This Opts has already been finalized.")

        defaults_dict = {} if defaults is None else dict(defaults)

        for key in type(self).__attrs__:
            if getattr(self, key) is UNSET:
                value = defaults_dict.get(key, self.defaults_frozen.get(key, UNSET))
                if (value is UNSET) and (not is_allow_unset):
                    raise KeyError(f"Missing default for field {key!r}.")
                setattr(self, key, value)

        object.__setattr__(self, "_state_is_functioning", True)

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
        is_functioning_current = self._state_is_functioning
        object.__setattr__(self, "_state_is_functioning", False)
        try:
            yield
        finally:
            object.__setattr__(self, "_state_is_functioning", is_functioning_current)

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
