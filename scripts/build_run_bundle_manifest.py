from __future__ import annotations

import argparse

from mci_gru.evaluation.run_bundle import write_run_manifest


def _optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write additive run manifest artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection-rule", default=None)
    parser.add_argument("--sibling-trial-id", action="append", default=None)
    parser.add_argument("--command", default=None)
    parser.add_argument("--feature-lag-policy", default=None)
    parser.add_argument("--normalization-reference", default=None)
    parser.add_argument("--graph-policy", default=None)
    parser.add_argument("--mlflow-run-id", default=None)
    parser.add_argument("--seed-policy", default=None)
    parser.add_argument("--paper-trade-eligible", type=_optional_bool, default=None)
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = write_run_manifest(
        args.run_dir,
        selection_rule=args.selection_rule,
        sibling_trial_ids=args.sibling_trial_id,
        command=args.command,
        feature_lag_policy=args.feature_lag_policy,
        normalization_reference=args.normalization_reference,
        graph_policy=args.graph_policy,
        mlflow_run_id=args.mlflow_run_id,
        seed_policy=args.seed_policy,
        paper_trade_eligible=args.paper_trade_eligible,
        repo_dir=args.repo_dir,
        force=args.force,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
