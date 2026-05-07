"""Deprecated LSEG regime CSV export entrypoint.

The live FRED/LSEG-backed regime loader is now the canonical workflow. This
script previously produced partial CSV inputs for the retired Colab
reconciliation flow; those partial files no longer satisfy the regime CSV
contract.
"""

DEPRECATION_MESSAGE = """
scripts/export_lseg_regime.py is deprecated.

Use the live regime input path instead:
- leave features.regime_inputs_csv unset/null
- set FRED_API_KEY
- configure the regime LSEG RICs when data.source=lseg

Partial LSEG regime CSV exports are not part of the supported workflow.
""".strip()


def main() -> None:
    raise SystemExit(DEPRECATION_MESSAGE)


if __name__ == "__main__":
    main()
