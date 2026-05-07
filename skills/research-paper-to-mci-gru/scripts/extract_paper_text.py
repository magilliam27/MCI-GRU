from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

LIGATURES = str.maketrans(
    {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "∗": "*",
    }
)


def normalize_text(text: str) -> str:
    """Normalize common academic-PDF extraction quirks without rewriting content."""
    text = text.translate(LIGATURES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\b([A-Z])\s+([a-z]{2,})\b", r"\1\2", text)
    return text


def _clean_inline(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def _metadata_value(pdf_metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = pdf_metadata.get(key)
        if value is not None:
            cleaned = _clean_inline(str(value))
            if cleaned:
                return cleaned
    return ""


def _first_page_lines(text: str) -> list[str]:
    before_abstract = re.split(r"\bABSTRACT\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return [_clean_inline(line) for line in before_abstract.splitlines() if _clean_inline(line)]


def _looks_like_author(line: str) -> bool:
    if "@" in line:
        return False
    if re.search(r"\b(University|Institute|Office|Department|School|Bank|USA|UK)\b", line):
        return False
    if re.match(r"^\d", line):
        return False
    words = line.split()
    if not 2 <= len(words) <= 5:
        return False
    return bool(re.match(r"^[A-Z][A-Za-z.' -]+[0-9*†‡]*$", line))


def _clean_author(line: str) -> str:
    line = re.sub(r"[0-9*†‡]+$", "", line).strip()
    return _clean_inline(line)


def _infer_title_from_text(text: str) -> str:
    lines = _first_page_lines(text)
    title_lines: list[str] = []
    started = False

    for line in lines:
        if re.search(r"\b(ISSN|DOI|Copyright|©|Keywords|JEL Codes)\b", line, re.I):
            continue
        if re.search(r"\b(Review|Journal)\b.*\d{4}", line):
            continue
        if _looks_like_author(line):
            break
        if re.search(r"@\w+", line):
            break
        if started or re.search(r"[A-Za-z]{4,}", line):
            started = True
            title_lines.append(line)
        if len(title_lines) >= 4:
            break

    return _clean_inline(" ".join(title_lines))


def _infer_authors_from_text(text: str, title: str) -> list[str]:
    lines = _first_page_lines(text)
    title_tokens = set(title.split())
    authors: list[str] = []

    for line in lines:
        if not line or line in title:
            continue
        if title_tokens and set(line.split()).issubset(title_tokens):
            continue
        if re.match(r"^\d", line):
            if authors:
                break
            continue
        if re.search(r"@\w+|University|Institute|Office|Department|School|USA|NBER", line):
            if authors:
                break
            continue
        if _looks_like_author(line):
            author = _clean_author(line)
            if author and author not in authors:
                authors.append(author)
    return authors


def _infer_authors_from_metadata(author_value: str) -> list[str]:
    if not author_value:
        return []
    parts = re.split(r"\s+(?:and|&)\s+|;", author_value)
    return [_clean_author(part) for part in parts if _clean_author(part)]


def _infer_abstract(text: str) -> str:
    match = re.search(
        r"\bABSTRACT\b\s+(.*?)(?=\bKeywords?:|\bJEL Codes?:|\n\s*1\s+Introduction\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _clean_inline(match.group(1)) if match else ""


def _infer_keywords(text: str, pdf_metadata: dict[str, Any]) -> list[str]:
    metadata_keywords = _metadata_value(pdf_metadata, "/Keywords", "Keywords")
    if metadata_keywords:
        return [k.strip() for k in re.split(r"[,;]", metadata_keywords) if k.strip()]
    match = re.search(r"\bKeywords?:\s*(.*?)(?=\n\s*JEL Codes?:|\n\s*1\s+Introduction\b)", text, re.I | re.S)
    if not match:
        return []
    return [k.strip() for k in re.split(r"[,;]", _clean_inline(match.group(1))) if k.strip()]


def infer_metadata(text: str, pdf_metadata: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(text)
    warnings: list[str] = []

    title = _metadata_value(pdf_metadata, "/Title", "Title")
    if not title:
        title = _infer_title_from_text(text)
        if title:
            warnings.append("PDF metadata title was blank; inferred title from first-page text.")

    author_value = _metadata_value(pdf_metadata, "/Author", "Author")
    authors = _infer_authors_from_metadata(author_value)
    text_authors = _infer_authors_from_text(text, title)
    if text_authors:
        authors = text_authors

    abstract = _infer_abstract(text)
    keywords = _infer_keywords(text, pdf_metadata)

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "keywords": keywords,
        "warnings": warnings,
    }


def extract_pdf(path: Path, max_pages: int | None = None) -> dict[str, Any]:
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError("pypdf is required to extract PDFs. Install it or paste paper text.") from exc

    reader = pypdf.PdfReader(str(path))
    pages = reader.pages[:max_pages] if max_pages is not None else reader.pages
    page_text = [normalize_text(page.extract_text() or "") for page in pages]
    full_text = "\n\n".join(page_text)
    pdf_metadata = {str(k): str(v) for k, v in (reader.metadata or {}).items()}
    metadata = infer_metadata(full_text, pdf_metadata)
    page_count = len(reader.pages)
    if max_pages is not None and max_pages < page_count:
        metadata["warnings"].append(
            f"Extraction limited to first {max_pages} page(s) of {page_count} total pages."
        )
    return {
        "source_path": str(path),
        "page_count": page_count,
        "extracted_pages": len(page_text),
        "chars_by_page": [len(page) for page in page_text],
        "metadata": metadata,
        "full_text": full_text,
    }


def build_markdown_artifact(
    source_path: Path,
    page_count: int,
    chars_by_page: list[int],
    metadata: dict[str, Any],
    full_text: str,
) -> str:
    authors = "; ".join(metadata.get("authors") or []) or "unknown"
    keywords = "; ".join(metadata.get("keywords") or []) or "none detected"
    warnings = metadata.get("warnings") or []
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- none"

    return "\n".join(
        [
            "# Paper Intake",
            "",
            "## Source",
            f"- path: `{source_path}`",
            f"- pages: {page_count}",
            f"- extracted characters by page: {chars_by_page}",
            "",
            "## Inferred Metadata",
            f"- title: {metadata.get('title') or 'unknown'}",
            f"- authors: {authors}",
            f"- keywords: {keywords}",
            "",
            "## Abstract",
            "",
            metadata.get("abstract") or "No abstract detected.",
            "",
            "## Extraction Warnings",
            "",
            warning_lines,
            "",
            "## Full Text",
            "",
            full_text.rstrip(),
            "",
        ]
    )


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract academic-paper text into Markdown or JSON.")
    parser.add_argument("pdf_path", type=Path, help="Local PDF path to extract.")
    parser.add_argument("-o", "--output", type=Path, help="Optional output path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--max-pages", type=int, help="Limit extraction to the first N pages.")
    args = parser.parse_args(argv)

    if not args.pdf_path.exists():
        parser.error(f"PDF not found: {args.pdf_path}")

    result = extract_pdf(args.pdf_path, max_pages=args.max_pages)
    if args.json:
        output = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    else:
        output = build_markdown_artifact(
            source_path=args.pdf_path,
            page_count=result["page_count"],
            chars_by_page=result["chars_by_page"],
            metadata=result["metadata"],
            full_text=result["full_text"],
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
