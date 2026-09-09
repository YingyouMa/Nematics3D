"""Q-tensor field construction, decomposition, and object workflows."""

from .diagonalization import QDiagonalizationResult, q_diagonalize
from .get_q import get_q

__all__ = [
    "InputQ",
    "QDiagonalizationResult",
    "QFieldObject",
    "get_q",
    "q_diagonalize",
]


def __getattr__(name: str):
    if name in {"InputQ", "QFieldObject"}:
        from .q_field_object import InputQ, QFieldObject

        return {"InputQ": InputQ, "QFieldObject": QFieldObject}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
