"""Singleton sentinel used to represent an option value that has not been provided."""


class Unset:
    """Type of the singleton :data:`UNSET` sentinel.

    ``Unset`` exists primarily for type annotations. Runtime code should use
    the singleton ``UNSET`` and test it by identity (``value is UNSET``), rather
    than constructing additional ``Unset`` instances.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = Unset()
