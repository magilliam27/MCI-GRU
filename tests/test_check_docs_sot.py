from pathlib import Path

from scripts import check_docs_sot

ALLOWLISTED_NAME = "REARCHITECTURE_TECHNICAL_SPEC_2026-07-01.md"
OFFENDER_NAME = "NEW_SHINY_RESULTS_REPORT_2026-07-04.md"


def _make_docs_tree(tmp_path: Path, names: list[str]) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for name in names:
        (docs_dir / name).write_text("stub", encoding="utf-8")
    return docs_dir


def test_new_offender_fails_but_allowlisted_file_passes(tmp_path: Path) -> None:
    docs_dir = _make_docs_tree(tmp_path, [ALLOWLISTED_NAME, OFFENDER_NAME])
    assert check_docs_sot.find_offenders(docs_dir) == [OFFENDER_NAME]
    assert check_docs_sot.main(docs_dir) == 1


def test_clean_tree_passes(tmp_path: Path) -> None:
    docs_dir = _make_docs_tree(
        tmp_path,
        [ALLOWLISTED_NAME, "ARCHITECTURE.md", "not_a_dated_report.md"],
    )
    assert check_docs_sot.find_offenders(docs_dir) == []
    assert check_docs_sot.main(docs_dir) == 0


def test_subdirectory_reports_are_not_flagged(tmp_path: Path) -> None:
    docs_dir = _make_docs_tree(tmp_path, [])
    research_dir = docs_dir / "research" / "current"
    research_dir.mkdir(parents=True)
    (research_dir / OFFENDER_NAME).write_text("stub", encoding="utf-8")
    assert check_docs_sot.find_offenders(docs_dir) == []
    assert check_docs_sot.main(docs_dir) == 0


def test_real_docs_tree_is_clean() -> None:
    repo_docs = Path(__file__).resolve().parents[1] / "docs"
    assert check_docs_sot.main(repo_docs) == 0
