"""CI lint: block new root-level dated reports under docs/.

The docs source-of-truth policy (docs/agents/domain.md, docs/research/README.md)
requires dated research reports to live in docs/research/{current,archive}/.
This check fails when a markdown file matching the dated-report naming pattern
appears directly under docs/. The root-level reports that were grandfathered
when the check was added all moved into the lifecycle in 2026-09, so there is no
allowlist: a dated report at the docs/ root is always an offender.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DATED_REPORT_PATTERN = re.compile(r"^[A-Z0-9_]+_\d{4}-\d{2}-\d{2}\.md$")


def find_offenders(docs_dir: Path) -> list[str]:
    """Return dated-report filenames directly under docs_dir."""
    return sorted(
        path.name for path in docs_dir.glob("*.md") if DATED_REPORT_PATTERN.match(path.name)
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
