"""Contract tests for scripts/generate_test_registry.py.

The registry generator must list every test function in tests/, record which
first-party modules each file exercises, and keep the committed registry to the
test inventory alone, so that every rendered line depends on one test file and
branches that add tests to different files merge without a registry conflict
(#149). ``--check`` regenerates the registry, parses the inventory back out of
both copies, and fails on any genuine difference, while ignoring order,
formatting, and the preamble. Last-run status from a junit XML file goes to a
separate report, never to the committed copy.
"""

import datetime as dt
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts.generate_test_registry import (
    build_registry,
    load_junit_results,
    main,
    parse_registry,
    parse_test_module,
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


def _write_module(tests_dir: Path, stem: str, test_names: list[str]) -> Path:
    body = "".join(
        f'\n\ndef {name}():\n    """{name} checks a {stem} behaviour."""\n    assert True\n'
        for name in test_names
    )
    path = tests_dir / f"{stem}.py"
    path.write_text(f'"""Module {stem}."""\n{body}', encoding="utf-8")
    return path


def _append_test(path: Path, name: str) -> None:
    path.write_text(
        path.read_text(encoding="utf-8") + f"\n\ndef {name}():\n    assert True\n",
        encoding="utf-8",
    )


def _junit_for_fake_module(path: Path, seconds: str, failed: bool) -> Path:
    failure = "<failure message='boom'/>" if failed else ""
    path.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_fake_module" '
        f'name="test_alpha" time="{seconds}">{failure}</testcase></testsuite></testsuites>',
        encoding="utf-8",
    )
    return path


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
    assert "Last run" not in content
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


def test_build_registry_preview_matches_the_written_file_without_writing(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"

    preview = build_registry(tests_dir, junit_path=None, out_path=out, write=False)

    assert not out.exists()
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


def test_parsed_inventory_ignores_run_status_but_not_markers(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"

    plain = build_registry(tests_dir, None, out, write=False)
    fast = build_registry(
        tests_dir, _junit_for_fake_module(tmp_path / "a.xml", "0.01", False), out, write=False
    )
    slow = build_registry(
        tests_dir, _junit_for_fake_module(tmp_path / "b.xml", "9.99", True), out, write=False
    )

    # Durations and last-run status vary per run and machine; the rows differ,
    # the inventory does not.
    assert fast != slow
    assert parse_registry(fast).modules == parse_registry(plain).modules
    assert parse_registry(slow).modules == parse_registry(plain).modules
    assert parse_registry(fast).has_status_columns
    assert not parse_registry(plain).has_status_columns

    # A marker is inventory: adding one must change what is parsed.
    marked = tests_dir / "test_fake_module.py"
    marked.write_text(
        marked.read_text(encoding="utf-8").replace(
            "def test_alpha", "@pytest.mark.requires_fred\ndef test_alpha"
        ),
        encoding="utf-8",
    )
    remarked = build_registry(tests_dir, None, out, write=False)
    assert parse_registry(remarked).modules != parse_registry(plain).modules


def test_parse_registry_round_trips_every_real_test_file():
    """Every real test, with its description and markers, survives render then parse.

    Real descriptions contain ``|``, so a parser that splits rows naively would
    lose or misread them, and ``--check`` would stop seeing changes to them.
    """
    tests_dir = REPO_ROOT / "tests"
    modules = [parse_test_module(p) for p in sorted(tests_dir.glob("test_*.py"))]
    parsed = parse_registry(
        build_registry(tests_dir, None, REPO_ROOT / "unused.md", write=False)
    ).modules

    assert set(parsed) == {f"tests/{m.path.name}" for m in modules}
    for module in modules:
        entry = parsed[f"tests/{module.path.name}"]
        assert entry.doc == module.doc
        assert entry.covers == tuple(module.covers)
        assert sorted(entry.tests) == sorted(
            (test.name, test.doc, tuple(test.markers)) for test in module.tests
        )


def test_committed_registry_carries_no_run_or_whole_inventory_lines(tmp_path):
    """No line may depend on more than one test file, or on when it was generated."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_fake_test_file(tests_dir)
    _write_module(tests_dir, "test_other", ["test_one"])
    out = tmp_path / "TEST_REGISTRY.md"

    before = build_registry(tests_dir, None, out, write=False)
    _write_module(tests_dir, "test_other", ["test_one", "test_two"])
    after = build_registry(tests_dir, None, out, write=False)

    # Adding a test to test_other.py changes only test_other.py's section.
    changed = set(after.splitlines()) ^ set(before.splitlines())
    assert changed == {"| `test_two` | test_two checks a test_other behaviour. |  |"}
    # A generation date changes on every regeneration on a new day.
    assert dt.date.today().isoformat() not in after


GIT = shutil.which("git")


@pytest.mark.skipif(GIT is None, reason="git is required to perform a real three-way merge")
@pytest.mark.parametrize(
    ("ours_change", "theirs_change"),
    [
        pytest.param(
            ("append", "test_alpha_mod", "test_added_on_ours"),
            ("append", "test_zeta_mod", "test_added_on_theirs"),
            id="tests-added-to-two-existing-files",
        ),
        pytest.param(
            ("create", "test_beta_new", "test_new_on_ours"),
            ("create", "test_omega_new", "test_new_on_theirs"),
            id="two-new-test-files",
        ),
    ],
)
def test_registries_from_concurrent_test_additions_merge_cleanly(
    tmp_path, ours_change, theirs_change
):
    """Two branches that each add a test must three-way merge with no registry conflict.

    ``git merge-file`` is the same line merge ``git merge`` applies to the file,
    so a header count, digest, or date line that both sides rewrite fails here.
    """

    def tree(changes: list[tuple[str, str, str]], name: str) -> Path:
        tests_dir = tmp_path / name / "tests"
        tests_dir.mkdir(parents=True)
        for stem in ("test_alpha_mod", "test_mid_mod", "test_zeta_mod"):
            _write_module(tests_dir, stem, ["test_first", "test_second"])
        for kind, stem, test_name in changes:
            if kind == "append":
                _append_test(tests_dir / f"{stem}.py", test_name)
            else:
                _write_module(tests_dir, stem, [test_name])
        return tests_dir

    def registry(tests_dir: Path) -> Path:
        out = tests_dir.parent / "TEST_REGISTRY.md"
        build_registry(tests_dir, None, out)
        return out

    base = registry(tree([], "base"))
    ours = registry(tree([ours_change], "ours"))
    theirs = registry(tree([theirs_change], "theirs"))
    merged_tests = tree([ours_change, theirs_change], "merged")

    result = subprocess.run(
        [GIT, "merge-file", "-p", str(ours), str(base), str(theirs)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, f"registry conflict:\n{result.stdout}"
    merged = tmp_path / "merged" / "TEST_REGISTRY.md"
    merged.write_text(result.stdout, encoding="utf-8", newline="\n")
    assert registry_is_current(merged_tests, merged)
    # Nothing is left to regenerate: the clean merge is what the generator writes.
    assert result.stdout == build_registry(merged_tests, None, merged, write=False)


def _fresh_fake_registry(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"
    assert main(["--tests-dir", str(tests_dir), "--out", str(out)]) == 0
    return tests_dir, out, ["--tests-dir", str(tests_dir), "--out", str(out), "--check"]


@pytest.mark.parametrize(
    ("corrupt", "expected_message"),
    [
        pytest.param(
            lambda text: text.replace("| `test_beta` |  | slow |\n", ""),
            "tests/test_fake_module.py::test_beta: missing from the registry",
            id="test-missing-from-registry",
        ),
        pytest.param(
            lambda text: text.replace(
                "| `test_beta` |  | slow |\n",
                "| `test_beta` |  | slow |\n| `test_removed_long_ago` |  |  |\n",
            ),
            "tests/test_fake_module.py::test_removed_long_ago: listed, but not in the test file",
            id="removed-test-still-listed",
        ),
        pytest.param(
            lambda text: text.replace("| `test_beta` |  | slow |", "| `test_beta` |  |  |"),
            "tests/test_fake_module.py::test_beta: description or markers differ",
            id="marker-dropped",
        ),
        pytest.param(
            lambda text: text.replace(
                "| `test_alpha` | Checks alpha behavior. |  |",
                "| `test_alpha` | Checks alpha behavior. | slow |",
            ),
            "tests/test_fake_module.py::test_alpha: description or markers differ",
            id="marker-invented",
        ),
        pytest.param(
            lambda text: text.replace("| Checks alpha behavior. |", "| Checks nothing now. |"),
            "tests/test_fake_module.py::test_alpha: description or markers differ",
            id="test-description-changed",
        ),
        pytest.param(
            lambda text: text.replace(
                "Fake test module covering the data manager.", "Fake test module, reworded."
            ),
            "tests/test_fake_module.py: module description differs",
            id="module-description-changed",
        ),
        pytest.param(
            lambda text: text.replace(
                "**Exercises:** `mci_gru.data.data_manager`", "**Exercises:** `mci_gru.pipeline`"
            ),
            "tests/test_fake_module.py: exercised modules differ",
            id="exercised-modules-changed",
        ),
        pytest.param(
            lambda text: text.replace("## `tests/test_fake_module.py`", "## `tests/test_gone.py`"),
            "tests/test_gone.py: listed, but not in the tests directory",
            id="section-for-a-file-that-does-not-exist",
        ),
        pytest.param(
            lambda text: text + "\n" + text[text.index("## `tests/test_fake_module.py`") :],
            "tests/test_fake_module.py: listed more than once",
            id="duplicated-section",
        ),
        pytest.param(
            lambda text: text.replace(
                "| Test | Description | Markers |\n|---|---|---|",
                "| Test | Description | Markers | Last run | Time (s) |\n|---|---|---|---|---|",
            ).replace(
                "| `test_beta` |  | slow |",
                "| `test_beta` |  | slow | PASSED | 0.01 |",
            ),
            "carries last-run status columns",
            id="status-columns-committed",
        ),
    ],
)
def test_check_rejects_known_bad_registries(tmp_path, capsys, corrupt, expected_message):
    tests_dir, out, check_argv = _fresh_fake_registry(tmp_path)
    good = out.read_text(encoding="utf-8")
    assert main(check_argv) == 0  # control: the untouched registry passes

    bad = corrupt(good)
    assert bad != good  # the corruption must actually change the file
    out.write_text(bad, encoding="utf-8", newline="\n")
    capsys.readouterr()

    assert main(check_argv) == 1
    assert expected_message in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == bad  # --check never rewrites


def test_check_ignores_order_formatting_and_preamble(tmp_path):
    tests_dir, out, check_argv = _fresh_fake_registry(tmp_path)
    _write_module(tests_dir, "test_second_module", ["test_one", "test_two"])
    assert main(["--tests-dir", str(tests_dir), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")

    # Swap the two sections, reverse one table's rows, pad cells, and rewrite
    # the preamble. None of it changes the inventory, so none of it may fail.
    first_heading = text.index("## `tests/test_fake_module.py`")
    second_heading = text.index("## `tests/test_second_module.py`")
    preamble = text[:first_heading]
    first = text[first_heading:second_heading]
    second = text[second_heading:]
    rows = [line for line in second.splitlines() if line.startswith("| `")]
    second = second.replace("\n".join(rows), "\n".join(reversed(rows)))
    reshuffled = (
        preamble.replace("do not edit by hand", "hand edits are overwritten")
        + second.rstrip("\n")
        + "\n\n"
        + first.replace("| `test_beta` |  | slow |", "|  `test_beta`  |   |  slow  |")
    )
    out.write_text(reshuffled, encoding="utf-8", newline="\n")

    assert main(check_argv) == 0


def test_junit_status_goes_to_a_separate_report_not_the_committed_registry(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write_fake_test_file(tests_dir)
    out = tmp_path / "TEST_REGISTRY.md"
    status_out = tmp_path / "reports" / "TEST_REGISTRY_STATUS.md"
    junit = _junit_for_fake_module(tmp_path / "junit.xml", "0.01", False)

    argv = ["--tests-dir", str(tests_dir), "--out", str(out)]
    assert main([*argv, "--junit", str(junit), "--status-out", str(status_out)]) == 0

    committed = out.read_text(encoding="utf-8")
    assert "Last run" not in committed
    assert "PASSED" not in committed
    assert not parse_registry(committed).has_status_columns
    report = status_out.read_text(encoding="utf-8")
    assert "| `test_alpha` | Checks alpha behavior. |  | PASSED | 0.01 |" in report
    assert main([*argv, "--check"]) == 0


def test_registry_covers_every_real_test_file():
    """The committed registry generator must see every test file in tests/."""
    real_tests = sorted(p.name for p in (REPO_ROOT / "tests").glob("test_*.py"))
    parsed = [parse_test_module(REPO_ROOT / "tests" / name) for name in real_tests]
    # Every test file must contribute at least one test function to the registry.
    empty = [m.path.name for m in parsed if not m.tests]
    assert not empty, f"Test files with no top-level test functions: {empty}"
