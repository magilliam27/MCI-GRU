"""CI lint: block new root-level dated reports under docs/.

The docs source-of-truth policy (docs/agents/domain.md, docs/research/README.md)
requires dated research reports to live in docs/research/{current,archive}/.
This check fails when a markdown file matching the dated-report naming pattern
appears directly under docs/ and is not one of the grandfathered pre-existing
files below. It does not move existing files; it only stops new offenders.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DATED_REPORT_PATTERN = re.compile(r"^[A-Z0-9_]+_\d{4}-\d{2}-\d{2}\.md$")

# Pre-existing root-level files grandfathered in when this check was added.
# The two REARCHITECTURE_TECHNICAL_SPEC files are specs, not research reports,
# but match the naming pattern and are intentionally kept at docs/ root.
ALLOWED_EXISTING_FILES: frozenset[str] = frozenset(
    {
        "ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md",
        "LONG_HISTORY_PIT_EVAL_RESULTS_2026-05-18.md",
        "PIT_2022_2024_REPLAY_DIAGNOSTICS_2026-05-24.md",
        "PIT_LSEG_ALIAS_COVERAGE_AUDIT_2026-05-16.md",
        "PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16.md",
        "PIT_REPEATED_SEED_OPTION_A_RESULTS_2026-05-21.md",
        "REARCHITECTURE_TECHNICAL_SPEC_2026-07-01.md",
        "REARCHITECTURE_TECHNICAL_SPEC_ADDENDUM_2026-07-04.md",
    }
)


def find_offenders(docs_dir: Path) -> list[str]:
    """Return non-allowlisted dated-report filenames directly under docs_dir."""
    return sorted(
        path.name
        for path in docs_dir.glob("*.md")
        if DATED_REPORT_PATTERN.match(path.name) and path.name not in ALLOWED_EXISTING_FILES
    )


def main(docs_dir: Path | None = None) -> int:
    """Run the check; return 0 when clean, 1 when new offenders exist."""
    resolved_docs_dir = (
        docs_dir if docs_dir is not None else Path(__file__).resolve().parents[1] / "docs"
    )
    offenders = find_offenders(resolved_docs_dir)
    if not offenders:
        print("check_docs_sot: OK - no new root-level dated reports under docs/.")
        return 0
    print("check_docs_sot: FAIL - new dated report(s) added directly under docs/:")
    for name in offenders:
        print(f"  docs/{name}")
    print(
        "Dated research reports belong in docs/research/current/ or "
        "docs/research/archive/, not at the docs/ root "
        "(see docs/research/README.md)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
