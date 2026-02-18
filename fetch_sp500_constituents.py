"""
Compatibility wrapper for relocated script.
"""

from runpy import run_path


if __name__ == "__main__":
    run_path("scripts/data/fetch_sp500_constituents.py", run_name="__main__")
