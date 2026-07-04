from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def with_safe_directory(args: list[str], safe_directory: Path | str) -> list[str]:
    if not args or args[0] != "git" or _has_safe_directory(args):
        return args
    return ["git", "-c", f"safe.directory={safe_directory}", *args[1:]]


def _has_safe_directory(args: list[str]) -> bool:
    return any(arg.startswith("safe.directory=") for arg in args)
