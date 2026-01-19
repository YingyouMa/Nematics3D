from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping
import weakref

from ..logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET


@dataclass(slots=True)
class OptsBase:

    _internal_owner_ref: weakref.ReferenceType | None = field(
        default=None, repr=False, init=False
    )
    _state_is_functioning: bool = field(default=False, init=False, repr=False)

    __descriptions__: ClassVar[Mapping[str, str]] = MappingProxyType({})
    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = MappingProxyType(
        {}
    )
    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    # ---------------------------------------------------------------------
    # Basic core: assignment with validation + lifecycle rule + owner commit
    # ---------------------------------------------------------------------
    @logging_and_warning_decorator(start_finish_level=5)
    def setattr_basic(self, key: str, value: Any, *, logger=None) -> Any:
        """
        Core assignment routine.

        Parameters
        ----------
        key : str
            Attribute name to assign.
        value : Any
            Value to assign. May be UNSET.
        logger : Any, optional
            Logger compatible with your decorator; if provided, exceptions and
            recovery messages are emitted.

        Returns
        -------
        value_assigned : Any
            The value that was ultimately assigned (may become UNSET), or the
            original value if ignored (in which case the object is unchanged).

        Behavior
        --------
        - If value is UNSET:
            - allowed only before first functioning; after functioning -> ignored.
        - If key in validators and value is not UNSET:
            - validate then assign
            - on failure:
                - before functioning: reset that field to UNSET
                - after functioning: ignore the modification
        - After functioning, propagate to owner via owner.act_commit(..., is_setattr=False).
        """
        is_final = bool(getattr(self, "_state_is_functioning", False))

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
                object.__setattr__(self, key, value2)
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
            object.__setattr__(self, key, value)

        # --- owner commit (only after functioning) ---
        if (
            (not key.startswith("_"))
            and is_final
            and getattr(self, "_internal_owner_ref", None) is not None
        ):
            owner = self._internal_owner_ref()
            if owner is not None:
                owner.act_commit(**{key: value}, is_setattr=False)

        return value

    # ---------------------------------------------------------------------
    # Basic core: finalize (fill UNSET by defaults then freeze state)
    # ---------------------------------------------------------------------
    def finalize_basic(self, defaults: Mapping[str, Any] | None = None) -> None:
        """
        Fill all UNSET public fields and mark this opts as functioning.

        Parameters
        ----------
        defaults : Mapping[str, Any] | None
            Per-call defaults. Applied before `_DEFAULTS_FROZEN`.

        Raises
        ------
        RuntimeError
            If already finalized.
        KeyError
            If a public field is UNSET and missing from both defaults sources.
        """

        if getattr(self, "_state_is_functioning", False):
            raise RuntimeError("This Opts has already been finalized.")

        defaults_dict = {} if defaults is None else dict(defaults)

        for f in fields(self):
            k = f.name
            if k.startswith("_"):
                continue

            if getattr(self, k) is UNSET:
                v = defaults_dict.get(k, self.__class__._DEFAULTS_FROZEN.get(k, UNSET))
                if v is UNSET:
                    raise KeyError(f"Missing default for field {k!r}.")
                setattr(self, k, v)

        object.__setattr__(self, "_state_is_functioning", True)

    # ---------------------------------------------------------------------
    # Basic core: export to dict
    # ---------------------------------------------------------------------
    def asdict_basic(self, *, is_include_UNSET: bool = False) -> dict[str, Any]:
        """
        Export options to a dict, following `__descriptions__` key order.

        Parameters
        ----------
        is_include_UNSET : bool
            Whether to include keys whose value is UNSET.

        Returns
        -------
        result : dict[str, Any]
            Exported mapping.
        """

        result: dict[str, Any] = {}
        for k in self.__class__.__descriptions__.keys():
            v = getattr(self, k)
            if (not is_include_UNSET) and (v is UNSET):
                continue
            result[k] = v
        return result
