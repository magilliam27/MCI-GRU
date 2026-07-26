"""Contract tests for scripts/generate_test_registry.py.

The registry generator must list every test function in tests/, record which
first-party modules each file exercises, merge last-run status from a junit XML
file when one exists, and expose a deterministic inventory digest that ``--check``
can use to prove the committed registry is fresh.
"""

import textwrap
from pathlib import Path

from scripts.generate_test_registry import (
    build_registry,
    inventory_digest,
    load_junit_results,
    main,
    parse_test_module,
    recorded_inventory_digest,
    registry_is_current,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_fake_test_file(tmp_path: Path) -> Path:
    test_file = tmp_path / "test_fake_module.py"
    test_file.write_text(
        textwrap.dedent(
            '''
            """Fake test module covering the data manager."""

            import pytest

            from mci_gru.data.data_manager import combined_collate_fn


            def test_alpha():
                """Checks alpha behavior."""
                assert combined_collate_fn is not None


            @pytest.mark.slow
            def test_beta():
                assert True


            class TestGamma:
                def test_inside_class(self):
                    assert True

                class TestNested:
                    def test_deep(self):
                        assert True
            '''
        ),
        encoding="utf-8",
    )
    return test_file


def test_parse_test_module_extracts_tests_docs_markers_and_imports(tmp_path):
    module = parse_test_module(_write_fake_test_file(tmp_path))

    assert module.doc == "Fake test module covering the data manager."
    assert module.covers == ["mci_gru.data.data_manager"]
    names = [t.name for t in module.tests]
    assert names == [
        "test_alpha",
        "test_beta",
        "TestGamma.test_inside_class",
        "TestGamma.TestNested.test_deep",
    ]
    assert module.tests[0].doc == "Checks alpha behavior."
    assert module.tests[1].markers == ["slow"]


def test_load_junit_results_collapses_parametrized_cases_and_ranks_status(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        textwrap.dedent(
            """
            <testsuites>
              <testsuite>
                <testcase classname="tests.test_fake" name="test_p[a]" time="0.10"/>
                <testcase classname="tests.test_fake" name="test_p[b]" time="0.20">
                  <failure message="boom"/>
                </testcase>
                <testcase classname="tests.test_fake" name="test_ok" time="0.05"/>
                <testcase classname="tests.test_fake.TestGroup" name="test_in_class" time="0.02"/>
                <testcase classname="" name="tests.test_optional_dep" time="0.0">
                  <skipped message="collection skipped"/>
                </testcase>
              </testsuite>
            </testsuites>
            """
        ),
        encoding="utf-8",
    )

    results, module_skips = load_junit_results(junit)

    status, seconds = results[("test_fake", "test_p")]
    assert status == "FAILED"  # worst status across parametrized cases wins
    assert abs(seconds - 0.30) < 1e-9
    assert results[("test_fake", "test_ok")][0] == "PASSED"
    assert results[("test_fake", "TestGroup.test_in_class")][0] == "PASSED"
    # Module-level collection skips are reported separately, not as testcases.
    assert module_skips == {"test_optional_dep"}
    assert ("test_optional_dep", "") not in results


def test_build_registry_writes_markdown_with_and_without_junit(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"

    content = build_registry(tests_dir, junit_path=None, out_path=out)

    assert out.exists()
    assert "test_alpha" in content
    assert "mci_gru.data.data_manager" in content
    assert "## `tests/test_fake_module.py`" in content
    assert "No junit results found" in content
    assert content.endswith("\n")
    assert not content.endswith("\n\n")
    assert all(line == line.rstrip() for line in content.splitlines())

    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_fake_module" '
        'name="test_alpha" time="0.01"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    content = build_registry(tests_dir, junit_path=junit, out_path=out)
    assert "PASSED" in content
    assert "Last run" in content
    assert content.endswith("\n")
    assert not content.endswith("\n\n")
    assert all(line == line.rstrip() for line in content.splitlines())


def test_build_registry_preview_records_digest_without_writing(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"

    preview = build_registry(tests_dir, junit_path=None, out_path=out, write=False)

    assert not out.exists()
    assert recorded_inventory_digest(preview) is not None
    assert preview == build_registry(tests_dir, junit_path=None, out_path=out)
    assert out.read_text(encoding="utf-8") == preview
    # Line endings must not depend on the host platform.
    assert b"\r\n" not in out.read_bytes()


def test_registry_is_current_detects_inventory_drift(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"

    assert not registry_is_current(tests_dir, out)  # nothing generated yet

    build_registry(tests_dir, junit_path=None, out_path=out)
    assert registry_is_current(tests_dir, out)

    test_file.write_text(
        test_file.read_text(encoding="utf-8") + "\n\ndef test_added_later():\n    assert True\n",
        encoding="utf-8",
    )
    assert not registry_is_current(tests_dir, out)

    build_registry(tests_dir, junit_path=None, out_path=out)
    assert registry_is_current(tests_dir, out)


def test_check_mode_reports_staleness_through_exit_code(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"
    check_argv = ["--tests-dir", str(tests_dir), "--out", str(out), "--check"]

    assert main(check_argv) == 1  # registry missing

    assert main(["--tests-dir", str(tests_dir), "--out", str(out)]) == 0
    generated = out.read_text(encoding="utf-8")
    assert main(check_argv) == 0

    test_file.write_text(
        test_file.read_text(encoding="utf-8") + "\n\ndef test_unregistered():\n    assert True\n",
        encoding="utf-8",
    )
    assert main(check_argv) == 1
    # --check must never rewrite the registry it is judging.
    assert out.read_text(encoding="utf-8") == generated


def test_inventory_digest_ignores_run_metadata_and_location(tmp_path):
    first_dir = tmp_path / "first" / "tests"
    second_dir = tmp_path / "second" / "renamed_tests"
    for directory in (first_dir, second_dir):
        directory.mkdir(parents=True)
        _write_fake_test_file(directory)
    out = tmp_path / "TEST_REGISTRY.md"

    first_modules = [parse_test_module(p) for p in sorted(first_dir.glob("test_*.py"))]
    second_modules = [parse_test_module(p) for p in sorted(second_dir.glob("test_*.py"))]

    digest = inventory_digest(first_modules)
    assert digest == inventory_digest(first_modules)  # stable across repeated calls
    assert digest == inventory_digest(second_modules)  # independent of the parent directory

    # Durations and last-run status come from junit and vary per run and machine,
    # so they must not move the digest.
    def _junit(path: Path, seconds: str, failed: bool) -> Path:
        failure = "<failure message='boom'/>" if failed else ""
        path.write_text(
            '<testsuites><testsuite><testcase classname="tests.test_fake_module" '
            f'name="test_alpha" time="{seconds}">{failure}</testcase></testsuite></testsuites>',
            encoding="utf-8",
        )
        return path

    fast = build_registry(first_dir, _junit(tmp_path / "a.xml", "0.01", False), out, write=False)
    slow = build_registry(first_dir, _junit(tmp_path / "b.xml", "9.99", True), out, write=False)

    assert fast != slow  # the rendered rows do differ
    assert recorded_inventory_digest(fast) == digest
    assert recorded_inventory_digest(slow) == digest

    # A marker change is part of the inventory and must move the digest.
    marked = first_dir / "test_fake_module.py"
    marked.write_text(
        marked.read_text(encoding="utf-8").replace(
            "def test_alpha", "@pytest.mark.requires_fred\ndef test_alpha"
        ),
        encoding="utf-8",
    )
    changed = [parse_test_module(p) for p in sorted(first_dir.glob("test_*.py"))]
    assert inventory_digest(changed) != digest


def test_registry_covers_every_real_test_file():
    """The committed registry generator must see every test file in tests/."""
    real_tests = sorted(p.name for p in (REPO_ROOT / "tests").glob("test_*.py"))
    parsed = [parse_test_module(REPO_ROOT / "tests" / name) for name in real_tests]
    # Every test file must contribute at least one test function to the registry.
    empty = [m.path.name for m in parsed if not m.tests]
    assert not empty, f"Test files with no top-level test functions: {empty}"
