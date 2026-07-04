from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

from cockpit.models import RunColor, WorkstreamStatus


def test_cockpit_status_enums_behave_as_strings() -> None:
    assert WorkstreamStatus.ACTIVE == "active"
    assert RunColor.GREEN == "green"
    assert str(WorkstreamStatus.NEEDS_USER_DECISION) == "needs-user-decision"


def test_strenum_compat_fallback_supports_python_310_import_path(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_stdlib_strenum(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "enum" and "StrEnum" in fromlist:
            raise ImportError("cannot import name 'StrEnum' from 'enum'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_stdlib_strenum)
    compat_path = Path("cockpit/_compat.py")
    spec = importlib.util.spec_from_file_location("cockpit_compat_py310_probe", compat_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Example(module.StrEnum):
        VALUE = "value"

    assert Example.VALUE == "value"
    assert isinstance(Example.VALUE, str)
    assert str(Example.VALUE) == "value"
