"""Regression tests for the isolated pytest launcher."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_pytest_isolated as runner

SAFE_INNER_TEST = (
    "tests/test_run_pytest_isolated.py::"
    "test_default_temp_parent_honors_explicit_environment_override"
)


def test_default_temp_parent_honors_explicit_environment_override(tmp_path: Path) -> None:
    environ = {runner.TEMP_PARENT_ENV: str(tmp_path)}

    assert runner.default_temp_parent(environ) == tmp_path


def test_create_run_root_is_unique_and_directly_below_parent(tmp_path: Path) -> None:
    first = runner.create_run_root(tmp_path)
    second = runner.create_run_root(tmp_path)

    try:
        assert first.parent == tmp_path.resolve()
        assert second.parent == tmp_path.resolve()
        assert first != second
        assert first.name.startswith(runner.RUN_ROOT_PREFIX)
        assert second.name.startswith(runner.RUN_ROOT_PREFIX)
    finally:
        runner.cleanup_run_root(first, tmp_path)
        runner.cleanup_run_root(second, tmp_path)


def test_cleanup_refuses_a_directory_outside_the_runner_namespace(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    with pytest.raises(ValueError, match="refusing to remove unexpected"):
        runner.cleanup_run_root(unrelated, tmp_path)

    assert unrelated.is_dir()


def test_cleanup_refuses_a_prefixed_directory_outside_expected_parent(tmp_path: Path) -> None:
    expected_parent = tmp_path / "expected"
    outside_parent = tmp_path / "outside"
    expected_parent.mkdir()
    outside_parent.mkdir()
    outside = outside_parent / f"{runner.RUN_ROOT_PREFIX}other"
    outside.mkdir()

    with pytest.raises(ValueError, match="refusing to remove unexpected"):
        runner.cleanup_run_root(outside, expected_parent)

    assert outside.is_dir()


def test_cleanup_refuses_a_reparse_point_instead_of_following_it(tmp_path: Path) -> None:
    target = tmp_path / f"{runner.RUN_ROOT_PREFIX}target"
    link = tmp_path / f"{runner.RUN_ROOT_PREFIX}link"
    target.mkdir()
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="reparse point"):
        runner.cleanup_run_root(link, tmp_path)

    assert target.is_dir()


def test_cleanup_removes_read_only_files_created_by_git(tmp_path: Path) -> None:
    run_root = runner.create_run_root(tmp_path)
    read_only = run_root / "objects" / "ab" / "object"
    read_only.parent.mkdir(parents=True)
    read_only.write_bytes(b"git object")
    read_only.chmod(stat.S_IREAD)

    runner.cleanup_run_root(run_root, tmp_path)

    assert not run_root.exists()


def test_build_pytest_command_injects_isolation_before_user_arguments(tmp_path: Path) -> None:
    basetemp = tmp_path / "pytest"
    cache_dir = tmp_path / "cache"

    command = runner.build_pytest_command(
        python_executable=Path(sys.executable),
        basetemp=basetemp,
        cache_dir=cache_dir,
        pytest_args=["tests/test_example.py", "-q"],
    )

    assert command == [
        sys.executable,
        str(runner.PYTEST_CHILD_SCRIPT),
        "tests/test_example.py",
        "-q",
        "-p",
        "cacheprovider",
        "-p",
        "no:_pytest.cacheprovider",
        "--basetemp",
        str(basetemp),
        "-o",
        f"cache_dir={cache_dir}",
    ]


def test_build_pytest_command_places_managed_options_before_double_dash(
    tmp_path: Path,
) -> None:
    command = runner.build_pytest_command(
        python_executable=Path(sys.executable),
        basetemp=tmp_path / "pytest",
        cache_dir=tmp_path / "cache",
        pytest_args=["tests", "--", "-odd_test_name"],
    )

    marker_index = command.index("--")
    assert command[marker_index - 2 : marker_index] == [
        "-o",
        f"cache_dir={tmp_path / 'cache'}",
    ]
    assert command[marker_index + 1 :] == ["-odd_test_name"]


def test_build_subprocess_environment_routes_all_temp_apis(tmp_path: Path) -> None:
    child_env = runner.build_subprocess_environment(
        tmp_path,
        {"PRESERVE": "yes", "PYTHONPATH": "existing-path"},
    )

    assert child_env["PRESERVE"] == "yes"
    assert child_env["TMP"] == str(tmp_path)
    assert child_env["TEMP"] == str(tmp_path)
    assert child_env["TMPDIR"] == str(tmp_path)
    assert child_env["PYTEST_DEBUG_TEMPROOT"] == str(tmp_path)
    assert child_env[runner.BASETEMP_PATH_ENV] == str(tmp_path / "pytest")
    assert child_env[runner.CACHE_DIR_PATH_ENV] == str(tmp_path / runner.CACHE_DIR_NAME)
    assert child_env["PYTHONPATH"] == os.pathsep.join(
        [str(runner.REPOSITORY_ROOT), "existing-path"]
    )


@pytest.mark.parametrize(
    "pytest_args",
    [
        ["--basetemp", "elsewhere"],
        ["--basetemp=elsewhere"],
        ["-p", "cacheprovider"],
        ["-pcacheprovider"],
        ["-p", "_pytest.cacheprovider"],
        ["-p", " _pytest.cacheprovider "],
        ["-p_pytest.cacheprovider"],
        ["-p=_pytest.cacheprovider"],
        ["-o", "cache_dir=elsewhere"],
        ["-ocache_dir=elsewhere"],
        ["-o=cache_dir=elsewhere"],
        ["--override-ini", "cache_dir=elsewhere"],
        ["--override-ini=cache_dir=elsewhere"],
        ["-o", "addopts=--basetemp=elsewhere --"],
        ["-oaddopts=-o cache_dir=elsewhere --"],
        ["--override-ini=addopts=@pytest-args.txt"],
        ["@pytest-args.txt"],
    ],
)
def test_validate_pytest_args_rejects_acl_unsafe_overrides(pytest_args: list[str]) -> None:
    with pytest.raises(ValueError, match="managed by the isolated runner"):
        runner.validate_pytest_args(pytest_args)


@pytest.mark.parametrize(
    "pytest_args",
    [
        ["-p", "no:randomly"],
        ["-pno:randomly"],
        ["-o", "python_files=check_*.py"],
        ["-opython_files=check_*.py"],
        ["--override-ini=python_files=check_*.py"],
        ["tests", "--", "-odd_test_name"],
    ],
)
def test_validate_pytest_args_preserves_unrelated_options(pytest_args: list[str]) -> None:
    runner.validate_pytest_args(pytest_args)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PYTEST_ADDOPTS", "-p _pytest.cacheprovider"),
        ("PYTEST_ADDOPTS", "-o cache_dir=elsewhere"),
        ("PYTEST_ADDOPTS", "--basetemp=elsewhere"),
        ("PYTEST_ADDOPTS", "-o addopts=--"),
        ("PYTEST_ADDOPTS", "@pytest-args.txt"),
        ("PYTEST_ADDOPTS", "--"),
        ("PYTEST_PLUGINS", "example_plugin,_pytest.cacheprovider"),
    ],
)
def test_validate_pytest_environment_rejects_acl_unsafe_overrides(
    name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="managed by the isolated runner"):
        runner.validate_pytest_environment({name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PYTEST_ADDOPTS", "-p _pytest.cacheprovider"),
        ("PYTEST_PLUGINS", "_pytest.cacheprovider"),
    ],
)
def test_main_rejects_unsafe_environment_before_creating_temp_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    temp_parent = tmp_path / "runner-temp"
    monkeypatch.setenv(runner.TEMP_PARENT_ENV, str(temp_parent))
    monkeypatch.setenv(name, value)

    assert runner.main([SAFE_INNER_TEST, "-q"]) == int(pytest.ExitCode.USAGE_ERROR)
    assert not temp_parent.exists()


def test_response_file_cannot_override_the_managed_basetemp(tmp_path: Path) -> None:
    external_basetemp = tmp_path / "external"
    response_file = tmp_path / "pytest-args.txt"
    response_file.write_text(f"--basetemp={external_basetemp}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="response files are unsupported"):
        runner.run_pytest([f"@{response_file}"], temp_parent=tmp_path / "runner-temp")

    assert not external_basetemp.exists()


def test_main_reports_runner_option_conflicts_as_pytest_usage_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = runner.main(["--basetemp=elsewhere"])

    assert exit_code == int(pytest.ExitCode.USAGE_ERROR)
    assert "--basetemp is managed by the isolated runner" in capsys.readouterr().err


def test_run_pytest_cleans_unique_root_after_success(tmp_path: Path) -> None:
    temp_parent = tmp_path / "runner-temp"

    exit_code = runner.run_pytest([SAFE_INNER_TEST, "-q"], temp_parent=temp_parent)

    assert exit_code == 0
    assert list(temp_parent.glob(f"{runner.RUN_ROOT_PREFIX}*")) == []


def test_run_pytest_preserves_failure_exit_and_cleans_root(tmp_path: Path) -> None:
    temp_parent = tmp_path / "runner-temp"

    exit_code = runner.run_pytest(
        [SAFE_INNER_TEST, "-q", "-k", "definitely_not_selected"],
        temp_parent=temp_parent,
    )

    assert exit_code == int(pytest.ExitCode.NO_TESTS_COLLECTED)
    assert list(temp_parent.glob(f"{runner.RUN_ROOT_PREFIX}*")) == []


def test_run_pytest_places_managed_paths_before_double_dash(tmp_path: Path) -> None:
    temp_parent = tmp_path / "runner-temp"

    exit_code = runner.run_pytest(
        ["-q", "--", SAFE_INNER_TEST],
        temp_parent=temp_parent,
    )

    assert exit_code == 0
    assert list(temp_parent.glob(f"{runner.RUN_ROOT_PREFIX}*")) == []


def test_run_pytest_can_keep_temp_for_explicit_diagnostics(tmp_path: Path) -> None:
    temp_parent = tmp_path / "runner-temp"

    exit_code = runner.run_pytest(
        [SAFE_INNER_TEST, "-q"],
        temp_parent=temp_parent,
        keep_temp=True,
    )

    retained = list(temp_parent.glob(f"{runner.RUN_ROOT_PREFIX}*"))
    assert exit_code == 0
    assert len(retained) == 1
    runner.cleanup_run_root(retained[0], temp_parent)


def test_run_pytest_contains_active_cache_and_removes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    cache_probe = project / "test_cache_probe.py"
    cache_probe.write_text(
        """from pathlib import Path
import os


def test_cache_is_inside_runner_root(pytestconfig):
    expected = Path(os.environ[\"MCI_GRU_PYTEST_CACHE_DIR\"]).resolve()
    created = pytestconfig.cache.mkdir(\"probe\").resolve()
    assert created.is_relative_to(expected)
""",
        encoding="utf-8",
    )
    temp_parent = tmp_path / "runner-temp"
    monkeypatch.chdir(project)

    exit_code = runner.run_pytest(
        [str(cache_probe), "-q"],
        temp_parent=temp_parent,
    )

    assert exit_code == 0
    assert not (project / ".pytest_cache").exists()
    assert list(temp_parent.glob(f"{runner.RUN_ROOT_PREFIX}*")) == []


@pytest.mark.parametrize("select_config_explicitly", [False, True])
def test_run_pytest_rejects_config_addopts_that_hide_managed_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    select_config_explicitly: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external_basetemp = tmp_path / "external-basetemp"
    external_cache = tmp_path / "external-cache"
    config = project / "pytest.ini"
    config.write_text(
        "[pytest]\n"
        f"addopts = --basetemp={external_basetemp.as_posix()} "
        f"-o cache_dir={external_cache.as_posix()} --\n",
        encoding="utf-8",
    )
    path_probe = project / "test_path_probe.py"
    path_probe.write_text(
        """from pathlib import Path
import os


def test_managed_paths_survive_config_addopts(tmp_path_factory, pytestconfig):
    expected_basetemp = Path(os.environ[\"MCI_GRU_PYTEST_BASETEMP\"]).resolve()
    expected_cache = Path(os.environ[\"MCI_GRU_PYTEST_CACHE_DIR\"]).resolve()
    assert tmp_path_factory.getbasetemp().resolve() == expected_basetemp
    created = pytestconfig.cache.mkdir(\"probe\").resolve()
    assert created.is_relative_to(expected_cache)
""",
        encoding="utf-8",
    )
    pytest_args = [str(path_probe), "-q"]
    if select_config_explicitly:
        pytest_args[:0] = ["-c", str(config)]
    temp_parent = tmp_path / "runner-temp"
    monkeypatch.chdir(project)

    exit_code = runner.run_pytest(pytest_args, temp_parent=temp_parent)

    assert exit_code == int(pytest.ExitCode.USAGE_ERROR)
    assert not external_basetemp.exists()
    assert not external_cache.exists()
    assert not (project / ".pytest_cache").exists()
    assert list(temp_parent.glob(f"{runner.RUN_ROOT_PREFIX}*")) == []


def test_run_pytest_preserves_safe_config_addopts_and_overrides_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external_cache = tmp_path / "external-cache"
    (project / "pytest.ini").write_text(
        f"[pytest]\naddopts = -q --tb=short\ncache_dir = {external_cache.as_posix()}\n",
        encoding="utf-8",
    )
    path_probe = project / "test_path_probe.py"
    path_probe.write_text(
        """from pathlib import Path
import os


def test_managed_paths_survive_safe_config(tmp_path_factory, pytestconfig):
    expected_basetemp = Path(os.environ[\"MCI_GRU_PYTEST_BASETEMP\"]).resolve()
    expected_cache = Path(os.environ[\"MCI_GRU_PYTEST_CACHE_DIR\"]).resolve()
    assert tmp_path_factory.getbasetemp().resolve() == expected_basetemp
    created = pytestconfig.cache.mkdir(\"probe\").resolve()
    assert created.is_relative_to(expected_cache)
""",
        encoding="utf-8",
    )
    temp_parent = tmp_path / "runner-temp"
    monkeypatch.chdir(project)

    exit_code = runner.run_pytest([str(path_probe)], temp_parent=temp_parent)

    assert exit_code == 0
    assert not external_cache.exists()
    assert not (project / ".pytest_cache").exists()
    assert list(temp_parent.glob(f"{runner.RUN_ROOT_PREFIX}*")) == []


@pytest.mark.parametrize(
    "cleanup_error",
    [PermissionError("ACL denied"), ValueError("safety refusal")],
)
def test_cleanup_failure_warns_without_masking_pytest_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cleanup_error: Exception,
) -> None:
    temp_parent = tmp_path / "runner-temp"

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=5)

    def fake_cleanup(run_root: Path, expected_parent: Path) -> None:
        raise cleanup_error

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "cleanup_run_root", fake_cleanup)

    exit_code = runner.run_pytest(["tests"], temp_parent=temp_parent)

    assert exit_code == 5
    assert "could not remove isolated temp root" in capsys.readouterr().err
