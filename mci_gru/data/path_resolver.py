"""
Path resolution helpers for project data files.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_data_path(configured_path: str) -> Path:
    """
    Resolve a data file path using project-aware fallbacks.

    Resolution order:
      1. Exact configured path (absolute or relative to cwd)
      2. Relative to project root
      3. Common organized data directories (by basename)
    """
    candidate = Path(configured_path)
    if candidate.exists():
        return candidate.resolve()

    root_relative = PROJECT_ROOT / configured_path
    if root_relative.exists():
        return root_relative.resolve()

    basename = candidate.name
    fallback_dirs = [
        PROJECT_ROOT / "data" / "raw" / "market",
        PROJECT_ROOT / "data" / "raw" / "constituents",
        PROJECT_ROOT / "data" / "raw" / "reference",
        PROJECT_ROOT / "data" / "external",
        PROJECT_ROOT / "data" / "interim",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT,
    ]
    for directory in fallback_dirs:
        fallback = directory / basename
        if fallback.exists():
            return fallback.resolve()

    raise FileNotFoundError(
        f"Data file not found for configured path '{configured_path}'. "
        f"Tried explicit path and project fallbacks."
    )

