"""
Compatibility wrapper for relocated script.
"""

from runpy import run_path


if __name__ == "__main__":
    run_path("scripts/data/download_all_universes.py", run_name="__main__")
