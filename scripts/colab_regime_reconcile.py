"""Deprecated Colab-side regime CSV reconciliation entrypoint.

The live FRED/LSEG-backed regime loader is now the canonical workflow and
emits the full seven-variable regime input surface directly. This script is
kept only to fail loudly for older notebooks or commands that still reference
the retired CSV reconciliation path.
"""

DEPRECATION_MESSAGE = """
scripts/colab_regime_reconcile.py is deprecated.

Use the live regime input path instead:
- leave features.regime_inputs_csv unset/null
- set FRED_API_KEY
- enable include_global_regime in the selected feature config

If a legacy offline CSV override is absolutely required, build it outside this
script and include dt plus all seven regime variables documented in
docs/REGIME_DATA_CONTRACT.md.
""".strip()


def main() -> None:
    raise SystemExit(DEPRECATION_MESSAGE)


if __name__ == "__main__":
    main()
