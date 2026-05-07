from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "research-paper-to-mci-gru"
    / "scripts"
    / "extract_paper_text.py"
)


def _load_extractor():
    spec = importlib.util.spec_from_file_location("extract_paper_text", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_infers_title_authors_and_abstract_from_first_page_text():
    extractor = _load_extractor()
    text = """Critical Finance Review, 2023, 12: 355-366
Conditional Skewness in Asset Pricing:
25 Years of Out-of-Sample Evidence
Campbell R. Harvey1
Akhtar Siddique2*
1Duke University, USA
2Office of the Comptroller of the Currency, USA
ABSTRACT
Much attention is paid to portfolio variance, but skewness is also important.
Keywords: Risk premium, Downside risk
JEL Codes: G12
1 Introduction
"""

    metadata = extractor.infer_metadata(text, pdf_metadata={})

    assert metadata["title"] == (
        "Conditional Skewness in Asset Pricing: 25 Years of Out-of-Sample Evidence"
    )
    assert metadata["authors"] == ["Campbell R. Harvey", "Akhtar Siddique"]
    assert metadata["abstract"] == (
        "Much attention is paid to portfolio variance, but skewness is also important."
    )
    assert metadata["warnings"] == [
        "PDF metadata title was blank; inferred title from first-page text."
    ]


def test_builds_markdown_intake_artifact_with_warnings_and_full_text():
    extractor = _load_extractor()
    metadata = {
        "title": "Example Paper",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "abstract": "A compact abstract.",
        "keywords": ["Risk premium", "Skewness"],
        "warnings": ["Used inferred metadata."],
    }

    artifact = extractor.build_markdown_artifact(
        source_path=Path("paper.pdf"),
        page_count=2,
        chars_by_page=[100, 200],
        metadata=metadata,
        full_text="First page\n\nSecond page",
    )

    assert artifact.startswith("# Paper Intake\n")
    assert "- path: `paper.pdf`" in artifact
    assert "- pages: 2" in artifact
    assert "- title: Example Paper" in artifact
    assert "- authors: Ada Lovelace; Grace Hopper" in artifact
    assert "- Used inferred metadata." in artifact
    assert "## Full Text\n\nFirst page\n\nSecond page" in artifact
