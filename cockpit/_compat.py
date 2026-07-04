from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:

    class StrEnum(str, Enum):
        """Python 3.10 fallback for enum.StrEnum."""

        def __str__(self) -> str:
            return self.value


__all__ = ["StrEnum"]
