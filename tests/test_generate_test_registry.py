"""Contract tests for scripts/generate_test_registry.py.

The registry generator must list every test function in tests/, record which
first-party modules each file exercises, and merge last-run status from a
junit XML file when one exists.
"""

import textwrap
from pathlib import Path

from scripts.generate_test_registry import (
    build_registry,
    load_junit_results,
    parse_test_module,
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

    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_fake_module" '
        'name="test_alpha" time="0.01"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    content = build_registry(tests_dir, junit_path=junit, out_path=out)
    assert "PASSED" in content
    assert "Last run" in content


def test_registry_covers_every_real_test_file():
    """The committed registry generator must see every test file in tests/."""
    real_tests = sorted(p.name for p in (REPO_ROOT / "tests").glob("test_*.py"))
    parsed = [parse_test_module(REPO_ROOT / "tests" / name) for name in real_tests]
    # Every test file must contribute at least one test function to the registry.
    empty = [m.path.name for m in parsed if not m.tests]
    assert not empty, f"Test files with no top-level test functions: {empty}"
