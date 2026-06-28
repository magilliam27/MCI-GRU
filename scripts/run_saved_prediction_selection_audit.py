from __future__ import annotations

import argparse

from mci_gru.evaluation.selection_audit import build_selection_audit, write_selection_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit saved predictions without retraining.")
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--market-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-t", type=int, default=5)
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--trial-count", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_selection_audit(
        predictions_dir=args.predictions_dir,
        market_data_path=args.market_data_path,
        label_t=args.label_t,
        top_k_values=args.top_k or [10, 20, 50],
        trial_count=args.trial_count,
    )
    path = write_selection_audit(audit, args.output_dir, force=args.force)
    print(f"selection_audit_summary: {path}")


if __name__ == "__main__":
    main()
