from __future__ import annotations

import argparse

from mci_gru.evaluation.selection_audit import (
    SelectionResearchProtocol,
    build_selection_audit,
    build_selection_research_evidence,
    write_selection_audit,
    write_selection_research_evidence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit saved predictions without retraining.")
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--market-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-t", type=int, default=5)
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--trial-count", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--research-evidence", action="store_true")
    parser.add_argument("--research-semantics-version", default="selection-research-v1")
    parser.add_argument("--study-name")
    parser.add_argument("--trial-family-id")
    parser.add_argument("--pit-universe-csv")
    parser.add_argument("--expected-scorable-csv")
    parser.add_argument("--calendar-csv")
    parser.add_argument("--calendar-source", default="DECLARED_SESSION_CSV")
    parser.add_argument("--exchange-timezone", default="America/New_York")
    parser.add_argument("--signal-close-local-time", default="16:00:00")
    parser.add_argument("--pit-known-from-timezone", default="UTC")
    parser.add_argument("--prediction-source-run-id")
    parser.add_argument("--prediction-ensemble-rule", default="SAVED_PREDICTION_SET")
    parser.add_argument("--prediction-ensemble-member-count", type=int)
    parser.add_argument("--prediction-seed-id")
    parser.add_argument("--prediction-source-code-commit")
    parser.add_argument("--prediction-label-contract", default="UNKNOWN")
    parser.add_argument("--prediction-label-horizon", type=int)
    parser.add_argument("--price-basis", default="UNKNOWN")
    parser.add_argument("--price-adjustment-provenance", default="UNKNOWN")
    parser.add_argument("--test-start")
    parser.add_argument("--test-end")
    parser.add_argument("--data-as-of")
    parser.add_argument("--null-draws", type=int, default=5000)
    parser.add_argument("--null-seed", type=int, default=73)
    parser.add_argument("--hac-lag", type=int)
    parser.add_argument("--bootstrap-block-length", type=int)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--trial-ledger-path")
    parser.add_argument("--trial-ledger-complete", action="store_true")
    parser.add_argument(
        "--expected-trial-id",
        action="append",
        default=None,
        help="Expected trial identifier; repeat once per declared family member.",
    )
    parser.add_argument(
        "--oos-not-previously-accessed",
        action="store_true",
        help="Declare that the study period was untouched before this frozen test.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.research_evidence:
        _run_research_evidence(args)
        return
    if args.trial_count is None:
        raise ValueError("--trial-count is required for legacy selection-audit mode")
    audit = build_selection_audit(
        predictions_dir=args.predictions_dir,
        market_data_path=args.market_data_path,
        label_t=args.label_t,
        top_k_values=args.top_k or [10, 20, 50],
        trial_count=args.trial_count,
    )
    path = write_selection_audit(audit, args.output_dir, force=args.force)
    print(f"selection_audit_summary: {path}")


def _run_research_evidence(args: argparse.Namespace) -> None:
    if args.force:
        raise ValueError("Canonical research evidence cannot be overwritten with --force")
    required = {
        "--study-name": args.study_name,
        "--trial-family-id": args.trial_family_id,
        "--calendar-csv": args.calendar_csv,
        "--test-start": args.test_start,
        "--test-end": args.test_end,
        "--data-as-of": args.data_as_of,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Research-evidence mode is missing required arguments: " + ", ".join(missing)
        )
    top_k_values = args.top_k or [10]
    if len(top_k_values) != 1:
        raise ValueError("Research-evidence mode accepts exactly one --top-k value")
    protocol = SelectionResearchProtocol(
        research_semantics_version=args.research_semantics_version,
        study_name=args.study_name,
        trial_family_id=args.trial_family_id,
        predictions_dir=args.predictions_dir,
        market_data_path=args.market_data_path,
        pit_universe_path=args.pit_universe_csv,
        expected_scorable_path=args.expected_scorable_csv,
        calendar_path=args.calendar_csv,
        label_horizon=args.label_t,
        test_start=args.test_start,
        test_end=args.test_end,
        data_as_of=args.data_as_of,
        top_k=top_k_values[0],
        price_basis=args.price_basis,
        price_adjustment_provenance=args.price_adjustment_provenance,
        null_draws=args.null_draws,
        null_seed=args.null_seed,
        hac_lag=(args.hac_lag if args.hac_lag is not None else args.label_t - 1),
        bootstrap_block_length=(
            args.bootstrap_block_length if args.bootstrap_block_length is not None else args.label_t
        ),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        ci_level=args.ci_level,
        alpha=args.alpha,
        trial_ledger_path=args.trial_ledger_path,
        trial_ledger_complete=args.trial_ledger_complete,
        expected_trial_ids=tuple(args.expected_trial_id or ()),
        oos_previously_accessed=not args.oos_not_previously_accessed,
        exchange_timezone=args.exchange_timezone,
        signal_close_local_time=args.signal_close_local_time,
        calendar_source=args.calendar_source,
        pit_known_from_timezone=args.pit_known_from_timezone,
        prediction_source_run_id=args.prediction_source_run_id,
        prediction_ensemble_rule=args.prediction_ensemble_rule,
        prediction_ensemble_member_count=args.prediction_ensemble_member_count,
        prediction_seed_id=args.prediction_seed_id,
        prediction_source_code_commit=args.prediction_source_code_commit,
        prediction_label_contract=args.prediction_label_contract,
        prediction_label_horizon=args.prediction_label_horizon,
    )
    evidence = build_selection_research_evidence(protocol)
    paths = write_selection_research_evidence(evidence, args.output_dir)
    print(f"selection_research_study_id: {paths['study_id']}")
    print(f"selection_research_bundle: {paths['bundle_dir']}")


if __name__ == "__main__":
    main()
