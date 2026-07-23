#!/usr/bin/env python
"""Run pytest with a unique temporary root and an isolated per-run cache.

On this Windows host, pytest temp and cache directories can carry
identity-specific ACLs. A fixed directory can therefore become inaccessible
when test runs alternate between a normal user and sandbox identities.

This launcher creates a fresh, inheriting parent for every invocation, routes
Python temporary files plus pytest's basetemp and cache beneath it, and removes
only that parent when the subprocess exits.
"""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType

RUN_ROOT_PREFIX = "mci-gru-pytest-"
CACHE_DIR_NAME = "cache"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PYTEST_CHILD_SCRIPT = Path(__file__).resolve().with_name("pytest_isolation_child.py")
TEMP_PARENT_ENV = "MCI_GRU_PYTEST_TEMP_ROOT"
BASETEMP_PATH_ENV = "MCI_GRU_PYTEST_BASETEMP"
CACHE_DIR_PATH_ENV = "MCI_GRU_PYTEST_CACHE_DIR"
PYTEST_ADDOPTS_ENV = "PYTEST_ADDOPTS"
PYTEST_PLUGINS_ENV = "PYTEST_PLUGINS"
KEEP_TEMP_OPTION = "--mci-keep-temp"
PYTEST_USAGE_ERROR = 4
CACHE_PROVIDER_ALIASES = frozenset(
    {"cacheprovider", "pytest_cacheprovider", "_pytest.cacheprovider"}
)


def default_temp_parent(environ: Mapping[str, str] | None = None) -> Path:
    """Return the parent under which isolated pytest roots should be created."""
    env = os.environ if environ is None else environ
    if configured := env.get(TEMP_PARENT_ENV):
        return Path(configured).expanduser()
    if os.name == "nt" and (local_app_data := env.get("LOCALAPPDATA")):
        return Path(local_app_data) / "Temp"
    return Path(tempfile.gettempdir())


def create_run_root(temp_parent: Path) -> Path:
    """Create one unique direct child of temp_parent with inherited ACLs."""
    resolved_parent = temp_parent.expanduser().resolve()
    resolved_parent.mkdir(mode=0o777, parents=True, exist_ok=True)
    for _ in range(10):
        candidate = resolved_parent / f"{RUN_ROOT_PREFIX}{uuid.uuid4().hex}"
        try:
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            continue
        return candidate.resolve()
    raise OSError(f"could not create a unique pytest root beneath {resolved_parent}")


def _retry_remove_readonly(
    function: Callable[[str], object],
    path: str,
    exc_info: tuple[type[BaseException], BaseException, TracebackType | None],
) -> None:
    """Clear a read-only attribute and retry one failed rmtree operation."""
    error = exc_info[1]
    if not isinstance(error, PermissionError):
        raise error
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _is_reparse_point(path: Path) -> bool:
    """Return whether path is a symlink or Windows filesystem reparse point."""
    if path.is_symlink():
        return True
    file_attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(file_attributes & reparse_flag)


def cleanup_run_root(run_root: Path, expected_parent: Path) -> None:
    """Remove run_root after proving it is the expected unique direct child."""
    resolved_parent = expected_parent.expanduser().resolve()
    lexical_root = Path(os.path.abspath(run_root.expanduser()))
    if lexical_root.parent.resolve() != resolved_parent or not lexical_root.name.startswith(
        RUN_ROOT_PREFIX
    ):
        raise ValueError(f"refusing to remove unexpected pytest temp root: {lexical_root}")
    if os.path.lexists(lexical_root) and _is_reparse_point(lexical_root):
        raise ValueError(f"refusing to remove pytest temp root reparse point: {lexical_root}")
    if lexical_root.exists():
        shutil.rmtree(lexical_root, onerror=_retry_remove_readonly)


def _normalise_plugin_spec(plugin_spec: str) -> str:
    """Return a canonical spelling for a pytest ``-p`` plugin specification."""
    normalised = plugin_spec.strip()
    if normalised.startswith("="):
        normalised = normalised[1:].lstrip()
    if normalised.startswith("no:"):
        normalised = normalised[3:].lstrip()
    return normalised.casefold()


def _is_cache_provider_spec(plugin_spec: str) -> bool:
    return _normalise_plugin_spec(plugin_spec) in CACHE_PROVIDER_ALIASES


def _managed_ini_override_key(override: str) -> str | None:
    normalised = override.strip()
    if normalised.startswith("="):
        normalised = normalised[1:].lstrip()
    key = normalised.split("=", maxsplit=1)[0].strip().casefold()
    return key if key in {"addopts", "cache_dir"} else None


def validate_pytest_args(pytest_args: Sequence[str]) -> None:
    """Reject options that would bypass the launcher's ACL isolation contract."""
    for index, argument in enumerate(pytest_args):
        if argument.startswith("@"):
            raise ValueError(
                "pytest response files are unsupported because isolation options are managed "
                "by the isolated runner"
            )
        if argument == "--basetemp" or argument.startswith("--basetemp="):
            raise ValueError("--basetemp is managed by the isolated runner")

        plugin_spec: str | None = None
        if argument == "-p" and index + 1 < len(pytest_args):
            plugin_spec = pytest_args[index + 1]
        elif argument.startswith("-p"):
            plugin_spec = argument[2:]
        if plugin_spec is not None and _is_cache_provider_spec(plugin_spec):
            raise ValueError("the pytest cache provider is managed by the isolated runner")

        ini_override: str | None = None
        if argument in {"-o", "--override-ini"} and index + 1 < len(pytest_args):
            ini_override = pytest_args[index + 1]
        elif argument.startswith("--override-ini="):
            ini_override = argument.partition("=")[2]
        elif argument.startswith("-o") and argument != "-o":
            ini_override = argument[2:]
        if ini_override is not None and (managed_key := _managed_ini_override_key(ini_override)):
            raise ValueError(f"pytest {managed_key} is managed by the isolated runner")


def validate_pytest_environment(environ: Mapping[str, str] | None = None) -> None:
    """Reject pytest environment settings that could escape run isolation."""
    env = os.environ if environ is None else environ
    if addopts := env.get(PYTEST_ADDOPTS_ENV):
        try:
            addopt_args = shlex.split(addopts)
        except ValueError as exc:
            raise ValueError(f"invalid {PYTEST_ADDOPTS_ENV}: {exc}") from exc
        if "--" in addopt_args:
            raise ValueError(
                f"{PYTEST_ADDOPTS_ENV} may not contain -- because isolation options are managed "
                "by the isolated runner"
            )
        validate_pytest_args(addopt_args)

    for plugin_spec in env.get(PYTEST_PLUGINS_ENV, "").split(","):
        if plugin_spec and _is_cache_provider_spec(plugin_spec):
            raise ValueError(
                f"the pytest cache provider in {PYTEST_PLUGINS_ENV} is managed by the "
                "isolated runner"
            )


def build_pytest_command(
    *,
    python_executable: Path,
    basetemp: Path,
    cache_dir: Path,
    pytest_args: Sequence[str],
) -> list[str]:
    """Build pytest argv with managed options last, but before a ``--`` marker."""
    forwarded = list(pytest_args)
    marker_index = forwarded.index("--") if "--" in forwarded else len(forwarded)
    managed_options = [
        "-p",
        "cacheprovider",
        "-p",
        "no:_pytest.cacheprovider",
        "--basetemp",
        str(basetemp),
        "-o",
        f"cache_dir={cache_dir}",
    ]
    return [
        str(python_executable),
        str(PYTEST_CHILD_SCRIPT),
        *forwarded[:marker_index],
        *managed_options,
        *forwarded[marker_index:],
    ]


def build_subprocess_environment(
    run_root: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Route Python and pytest temporary files into run_root."""
    child_env = dict(os.environ if environ is None else environ)
    temp_path = str(run_root)
    basetemp_path = str(run_root / "pytest")
    cache_dir_path = str(run_root / CACHE_DIR_NAME)
    existing_pythonpath = child_env.get("PYTHONPATH")
    pythonpath_parts = [str(REPOSITORY_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    child_env.update(
        {
            "TMP": temp_path,
            "TEMP": temp_path,
            "TMPDIR": temp_path,
            "PYTEST_DEBUG_TEMPROOT": temp_path,
            BASETEMP_PATH_ENV: basetemp_path,
            CACHE_DIR_PATH_ENV: cache_dir_path,
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        }
    )
    return child_env


def run_pytest(
    pytest_args: Sequence[str],
    *,
    temp_parent: Path | None = None,
    keep_temp: bool = False,
) -> int:
    """Run pytest once and return its exit code without reusing temp state."""
    parent_env = dict(os.environ)
    validate_pytest_args(pytest_args)
    validate_pytest_environment(parent_env)
    parent = default_temp_parent(parent_env) if temp_parent is None else temp_parent
    run_root = create_run_root(parent)
    command = build_pytest_command(
        python_executable=Path(sys.executable),
        basetemp=run_root / "pytest",
        cache_dir=run_root / CACHE_DIR_NAME,
        pytest_args=pytest_args,
    )
    child_env = build_subprocess_environment(run_root, parent_env)
    print(f"run_pytest_isolated: temp root {run_root}", flush=True)
    try:
        completed = subprocess.run(command, env=child_env, check=False)
        return completed.returncode
    finally:
        if keep_temp:
            print(f"run_pytest_isolated: retained temp root {run_root}", flush=True)
        else:
            try:
                cleanup_run_root(run_root, parent)
            except (OSError, ValueError) as exc:
                print(
                    f"run_pytest_isolated: warning: could not remove isolated temp root "
                    f"{run_root}: {exc}",
                    file=sys.stderr,
                )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse runner-only options and forward everything else to pytest."""
    incoming = list(sys.argv[1:] if argv is None else argv)
    keep_temp = False
    pytest_args: list[str] = []
    for argument in incoming:
        if argument == KEEP_TEMP_OPTION:
            keep_temp = True
        else:
            pytest_args.append(argument)
    try:
        return run_pytest(pytest_args, keep_temp=keep_temp)
    except ValueError as exc:
        print(f"run_pytest_isolated: error: {exc}", file=sys.stderr)
        return PYTEST_USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
