"""Generate the approval-gated 2026-YTD LambdaRankIC Colab notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from nb_lib import code, colab_setup_cell, md, write_notebook
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/lambdarankic_2026_ytd_colab.ipynb"
APPROVAL_OUT = ROOT / "configs/launch_manifests/lambdarankic_2026_ytd_110_name.json"
EXPERIMENT_PRESET = "lambdarankic_2026_ytd_110_name"
BRANCH = "codex/lambdarankic-2026-ytd-20260713"

MARKET_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20210104_20260713_lseg_20190101_20260713.csv"
PIT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20210104_20260713_pit_universe.csv"
SNAPSHOT_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20210104_20260713_snapshots.csv"
SELECTOR_META_FILENAME = "sp500_pit_gics_top10_mcap_monthly_20210104_20260713_meta.json"
PRICE_META_FILENAME = MARKET_FILENAME.replace(".csv", ".meta.json")

with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
    cfg = compose(config_name="config", overrides=[f"+experiment={EXPERIMENT_PRESET}"])
RESOLVED_CONFIG = OmegaConf.to_container(cfg, resolve=True)

CAMPAIGN = {
    "campaign_id": "lambdarankic_2026_ytd_110_name",
    "code_branch": BRANCH,
    "hydra_experiment": EXPERIMENT_PRESET,
    "objective_matrix": ["lambdarank_ic"],
    "base_seeds": [314159, 271828, 161803, 141421, 173205],
    "ensemble_models_per_seed": 20,
    "expected_training_jobs": 5,
    "expected_model_fits": 100,
    "universe": {
        "kind": "monthly PIT S&P 500 top-10 by market cap within each GICS sector",
        "expected_active_names": 110,
        "expected_sectors": 11,
        "top_n_per_sector": 10,
    },
    "data_export": {
        "source": "LSEG",
        "selector_start": "2021-01-04",
        "selector_end": "2026-07-13",
        "history_start": "2019-01-01",
        "history_end": "2026-07-13",
        "frequency": "monthly",
        "market_filename": MARKET_FILENAME,
        "pit_filename": PIT_FILENAME,
        "snapshot_filename": SNAPSHOT_FILENAME,
        "selector_meta_filename": SELECTOR_META_FILENAME,
        "price_meta_filename": PRICE_META_FILENAME,
    },
    "split_contract": {
        "train": ["2021-01-01", "2024-12-31"],
        "validation": ["2025-01-10", "2025-12-23"],
        "test": ["2026-01-01", "2026-07-13"],
        "label_sessions": 5,
        "boundary_policy": "actual-session embargo; predictions include the unlabeled YTD tail",
    },
    "runtime": {
        "platform": "Google Colab",
        "runtime_type": "G4 GPU",
        "allowed_gpu_markers": ["G4", "RTX PRO", "BLACKWELL"],
        "blocked_gpu_markers": ["T4", "L4"],
        "execution": "visible foreground Chrome session",
    },
    "evaluation_scope": {
        "built_in_test_metrics": True,
        "cross_seed_mean_and_sample_std": True,
        "strategy_backtest": False,
        "matched_pure_ic_control": False,
    },
    "artifact_contract": {
        "drive_root": "/content/drive/MyDrive/MCI-GRU-Ablations/lambdarank_ic_2026_ytd",
        "periodic_sync_seconds": 60,
        "persist_checkpoints": True,
        "persist_per_model_predictions": True,
        "persist_averaged_predictions": True,
        "persist_graph_data": True,
        "persist_logs_and_summaries": True,
        "resume_granularity": "no automatic resume; synced artifacts are salvaged for an explicit recovery plan",
    },
}

SOURCE_FILES = ["pyproject.toml", "requirements.txt", "run_experiment.py"]
SOURCE_FILES.extend(
    path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "mci_gru").rglob("*.py"))
)
SOURCE_FILES.extend(
    path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "configs").rglob("*.yaml"))
)


def normalized_text_sha256(path: Path) -> str:
    """Hash text reproducibly across Windows and Colab line endings."""
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


CAMPAIGN["source_files_sha256"] = {
    relative: normalized_text_sha256(ROOT / relative) for relative in SOURCE_FILES
}

APPROVAL_BUNDLE = {
    "schema_version": 1,
    "campaign": CAMPAIGN,
    "resolved_hydra_config": RESOLVED_CONFIG,
}
CANONICAL_APPROVAL_JSON = json.dumps(
    APPROVAL_BUNDLE,
    sort_keys=True,
    separators=(",", ":"),
)
APPROVAL_SHA256 = hashlib.sha256(CANONICAL_APPROVAL_JSON.encode("utf-8")).hexdigest()
EMBEDDED_BUNDLE = json.dumps(APPROVAL_BUNDLE, indent=2, sort_keys=True)

approval_cell = r"""
# This cell is read-only: it creates no Drive run folder and starts no training.
APPROVAL_BUNDLE = json.loads(r'''__APPROVAL_BUNDLE__''')
CANONICAL_APPROVAL_JSON = json.dumps(
    APPROVAL_BUNDLE,
    sort_keys=True,
    separators=(",", ":"),
)
CONFIG_SHA256 = hashlib.sha256(CANONICAL_APPROVAL_JSON.encode("utf-8")).hexdigest()
EXPECTED_CONFIG_SHA256 = "__APPROVAL_SHA256__"
if CONFIG_SHA256 != EXPECTED_CONFIG_SHA256:
    raise RuntimeError("Embedded approval bundle digest mismatch.")

CAMPAIGN = APPROVAL_BUNDLE["campaign"]
BASE_SEEDS = CAMPAIGN["base_seeds"]
print("Approval digest:", CONFIG_SHA256)
print("Campaign:", CAMPAIGN["campaign_id"])
print("Base seeds:", BASE_SEEDS)
print("Model fits:", CAMPAIGN["expected_model_fits"])
print(json.dumps(APPROVAL_BUNDLE["resolved_hydra_config"], indent=2, sort_keys=True))
""".replace("__APPROVAL_BUNDLE__", EMBEDDED_BUNDLE).replace("__APPROVAL_SHA256__", APPROVAL_SHA256)

data_audit_cell = r"""
import pandas as pd

if not IN_COLAB:
    raise RuntimeError("This campaign must be staged and trained in Google Colab.")

if not os.environ.get("FRED_API_KEY"):
    from google.colab import userdata

    fred_secret = userdata.get("FRED_API_KEY")
    if fred_secret:
        os.environ["FRED_API_KEY"] = fred_secret
if not os.environ.get("FRED_API_KEY"):
    raise RuntimeError("FRED_API_KEY is required for the strict global-regime recipe.")

export = CAMPAIGN["data_export"]
drive_data_dir = Path("/content/drive/MyDrive/MCI_GRU_shared/data")
staged_paths = {}
for key, target_dir in {
    "market_filename": REPO_DIR / "data/raw/market",
    "pit_filename": REPO_DIR / "data/raw/constituents",
    "snapshot_filename": REPO_DIR / "data/raw/constituents",
    "selector_meta_filename": REPO_DIR / "data/raw/constituents",
    "price_meta_filename": REPO_DIR / "data/raw/market",
}.items():
    filename = export[key]
    source = drive_data_dir / filename
    if not source.is_file():
        raise FileNotFoundError(f"Missing approved input in Drive: {source}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    shutil.copy2(source, target)
    staged_paths[key] = target

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

market = pd.read_csv(staged_paths["market_filename"])
pit = pd.read_csv(staged_paths["pit_filename"])
snapshots = pd.read_csv(staged_paths["snapshot_filename"])
selector_meta = json.loads(staged_paths["selector_meta_filename"].read_text(encoding="utf-8"))
price_meta = json.loads(staged_paths["price_meta_filename"].read_text(encoding="utf-8"))
if selector_meta.get("source") != "refinitiv.data":
    raise RuntimeError(f"Unexpected selector source: {selector_meta.get('source')!r}")
if price_meta.get("source") != "refinitiv.data.get_history":
    raise RuntimeError(f"Unexpected price source: {price_meta.get('source')!r}")
if selector_meta.get("frequency") != export["frequency"]:
    raise RuntimeError(f"Unexpected selector frequency: {selector_meta.get('frequency')!r}")
if selector_meta.get("top_n_per_sector") != CAMPAIGN["universe"]["top_n_per_sector"]:
    raise RuntimeError("Selector top-N metadata differs from the approved universe.")
if selector_meta.get("expected_sectors") != CAMPAIGN["universe"]["expected_sectors"]:
    raise RuntimeError("Selector sector-count metadata differs from the approved universe.")
for key in ("selector_start", "selector_end", "history_start", "history_end"):
    metadata_key = {
        "selector_start": "start",
        "selector_end": "end",
        "history_start": "history_start",
        "history_end": "history_end",
    }[key]
    if str(selector_meta.get(metadata_key)) != export[key]:
        raise RuntimeError(
            f"Selector metadata mismatch for {metadata_key}: "
            f"{selector_meta.get(metadata_key)!r} vs {export[key]!r}"
        )
if str(price_meta.get("start")) != export["history_start"]:
    raise RuntimeError("Price metadata history start differs from the approved export.")
if str(price_meta.get("end")) != export["history_end"]:
    raise RuntimeError("Price metadata history end differs from the approved export.")
if price_meta.get("missing_identifiers"):
    raise RuntimeError(f"Price export has missing identifiers: {price_meta['missing_identifiers']}")
if price_meta.get("resolved_identifiers_with_rows") != price_meta.get("requested_identifiers"):
    raise RuntimeError("Not every requested PIT-union identifier has price rows.")
required_market = {"kdcode", "dt", "open", "high", "low", "close", "volume"}
missing_market = sorted(required_market - set(market.columns))
if missing_market:
    raise RuntimeError(f"Market CSV is missing required columns: {missing_market}")
required_pit = {"kdcode", "valid_from", "valid_to"}
if not required_pit.issubset(pit.columns):
    raise RuntimeError(f"PIT CSV is missing columns: {sorted(required_pit - set(pit.columns))}")
required_snapshot = {"as_of_date", "kdcode", "gics_sector"}
if not required_snapshot.issubset(snapshots.columns):
    raise RuntimeError(
        f"Snapshot CSV is missing columns: {sorted(required_snapshot - set(snapshots.columns))}"
    )

market["dt"] = pd.to_datetime(market["dt"]).dt.strftime("%Y-%m-%d")
pit["valid_from"] = pd.to_datetime(pit["valid_from"]).dt.strftime("%Y-%m-%d")
pit["valid_to"] = pd.to_datetime(pit["valid_to"]).dt.strftime("%Y-%m-%d")
snapshots["as_of_date"] = pd.to_datetime(snapshots["as_of_date"]).dt.strftime("%Y-%m-%d")
if market.duplicated(["dt", "kdcode"]).any():
    raise RuntimeError("Market CSV contains duplicate (dt, kdcode) rows.")
market["kdcode"] = market["kdcode"].astype(str)
pit["kdcode"] = pit["kdcode"].astype(str)
snapshots["kdcode"] = snapshots["kdcode"].astype(str)
pit_codes = set(pit["kdcode"])
market_codes = set(market["kdcode"])
snapshot_codes = set(snapshots["kdcode"])
if pit_codes != market_codes or pit_codes != snapshot_codes:
    raise RuntimeError(
        "PIT/market/snapshot identifier mismatch: "
        f"pit_not_market={sorted(pit_codes - market_codes)[:10]} "
        f"market_not_pit={sorted(market_codes - pit_codes)[:10]} "
        f"pit_not_snapshot={sorted(pit_codes - snapshot_codes)[:10]} "
        f"snapshot_not_pit={sorted(snapshot_codes - pit_codes)[:10]}"
    )

split = CAMPAIGN["split_contract"]
train_start, train_end = split["train"]
val_start, val_end = split["validation"]
test_start, test_end = split["test"]
all_dates = sorted(market["dt"].unique().tolist())
if not all_dates or all_dates[-1] < test_end:
    raise RuntimeError(f"Market history ends at {all_dates[-1] if all_dates else None}, before {test_end}.")

active_end = set(
    pit.loc[(pit["valid_from"] <= test_end) & (pit["valid_to"] >= test_end), "kdcode"]
    .dropna()
    .astype(str)
)
end_rows = market[(market["dt"] == test_end) & market["kdcode"].astype(str).isin(active_end)]
complete_end = set(
    end_rows.dropna(subset=["open", "high", "low", "close", "volume"])["kdcode"].astype(str)
)
expected_names = CAMPAIGN["universe"]["expected_active_names"]
if len(active_end) != expected_names or complete_end != active_end:
    raise RuntimeError(
        "Final-date PIT coverage failed: "
        f"active={len(active_end)} complete_ohlc={len(complete_end)} "
        f"missing={sorted(active_end - complete_end)[:10]}"
    )

snapshot_counts = snapshots.groupby("as_of_date")["kdcode"].nunique()
sector_counts = snapshots.groupby(["as_of_date", "gics_sector"])["kdcode"].nunique()
if snapshot_counts.min() != expected_names or snapshot_counts.max() != expected_names:
    raise RuntimeError("Each monthly PIT snapshot must contain exactly 110 names.")
if snapshots["gics_sector"].nunique() != CAMPAIGN["universe"]["expected_sectors"]:
    raise RuntimeError("Expected exactly 11 GICS sectors in the selector output.")
if not (sector_counts == CAMPAIGN["universe"]["top_n_per_sector"]).all():
    raise RuntimeError("Every snapshot/sector cell must contain exactly 10 names.")

complete_market = market.dropna(subset=["open", "high", "low", "close", "volume"])
complete_by_date = {
    date: set(group["kdcode"])
    for date, group in complete_market.groupby("dt", sort=False)
}
experiment_dates = [date for date in all_dates if train_start <= date <= test_end]
daily_coverage = []
for date in experiment_dates:
    active = set(
        pit.loc[(pit["valid_from"] <= date) & (pit["valid_to"] >= date), "kdcode"]
    )
    complete = complete_by_date.get(date, set()) & active
    daily_coverage.append(
        {
            "date": date,
            "active": len(active),
            "complete_ohlcv": len(complete),
            "missing": sorted(active - complete),
        }
    )
bad_active = [row for row in daily_coverage if row["active"] != expected_names]
min_scoreable = APPROVAL_BUNDLE["resolved_hydra_config"]["data"]["pit_min_scoreable_stocks"]
bad_complete = [
    row for row in daily_coverage if row["complete_ohlcv"] < min_scoreable
]
if bad_active:
    raise RuntimeError(f"Daily PIT active breadth is not 110: {bad_active[:3]}")
if bad_complete:
    raise RuntimeError(
        f"Daily complete OHLCV breadth is below {min_scoreable}: {bad_complete[:3]}"
    )

pretrain_dates = [date for date in all_dates if date < train_start]
if len(pretrain_dates) < 252:
    raise RuntimeError(f"Need at least 252 pre-train sessions; found {len(pretrain_dates)}.")

def first_session_on_or_after(date: str) -> str:
    return next(item for item in all_dates if item >= date)

def global_label_target_session(date: str, label_t: int) -> str:
    index = all_dates.index(date)
    return all_dates[index + label_t]

train_target_end = global_label_target_session(train_end, split["label_sessions"])
val_target_end = global_label_target_session(val_end, split["label_sessions"])
first_val_session = first_session_on_or_after(val_start)
first_test_session = first_session_on_or_after(test_start)
if not train_target_end < first_val_session:
    raise RuntimeError(f"Train/validation label overlap: {train_target_end=} {first_val_session=}")
if not val_target_end < first_test_session:
    raise RuntimeError(f"Validation/test label overlap: {val_target_end=} {first_test_session=}")

label_t = split["label_sessions"]
stock_dates = {
    kdcode: sorted(group["dt"].unique().tolist())
    for kdcode, group in market.groupby("kdcode", sort=False)
}

def prove_per_stock_embargo(
    split_start: str,
    split_end: str,
    first_next_session: str,
) -> dict:
    checked_targets = []
    violations = []
    for interval in pit.itertuples(index=False):
        interval_start = max(split_start, interval.valid_from)
        interval_end = min(split_end, interval.valid_to)
        if interval_start > interval_end:
            continue
        dates = stock_dates.get(interval.kdcode, [])
        eligible = [date for date in dates if interval_start <= date <= interval_end]
        if not eligible:
            violations.append(
                {
                    "kdcode": interval.kdcode,
                    "reason": "no_market_row_in_active_interval",
                    "interval_end": interval_end,
                }
            )
            continue
        finite_target_candidates = [
            (label_date, dates.index(label_date) + label_t)
            for label_date in eligible
            if dates.index(label_date) + label_t < len(dates)
        ]
        if not finite_target_candidates:
            continue  # production label is NaN and excluded by the masked-panel loss mask
        label_date, target_index = finite_target_candidates[-1]
        target_date = dates[target_index]
        checked_targets.append(target_date)
        if target_date >= first_next_session:
            violations.append(
                {
                    "kdcode": interval.kdcode,
                    "label_date": label_date,
                    "target_date": target_date,
                    "first_next_session": first_next_session,
                }
            )
    if violations:
        raise RuntimeError(f"Per-stock label embargo failed: {violations[:5]}")
    return {
        "checked_target_count": len(checked_targets),
        "latest_target": max(checked_targets) if checked_targets else None,
    }

train_stock_embargo = prove_per_stock_embargo(train_start, train_end, first_val_session)
validation_stock_embargo = prove_per_stock_embargo(val_start, val_end, first_test_session)

from mci_gru.graph import GraphBuilder

graph_cfg = APPROVAL_BUNDLE["resolved_hydra_config"]["graph"]
graph_builder = GraphBuilder(
    judge_value=graph_cfg["judge_value"],
    update_frequency_months=graph_cfg["update_frequency_months"],
    corr_lookback_days=graph_cfg["corr_lookback_days"],
    top_k=graph_cfg["top_k"],
    top_k_metric=graph_cfg["top_k_metric"],
    use_multi_feature_edges=graph_cfg["use_multi_feature_edges"],
    use_lead_lag_features=graph_cfg["use_lead_lag_features"],
    lead_lag_days=graph_cfg["lead_lag_days"],
)
preflight_edge_index, preflight_edge_weight = graph_builder.build_graph(
    market,
    sorted(pit_codes),
    train_start,
    show_progress=False,
)
if preflight_edge_index.shape[1] == 0:
    raise RuntimeError("Approved static graph has zero edges on the refreshed input.")
if preflight_edge_weight.ndim != 2 or preflight_edge_weight.shape[1] != 4:
    raise RuntimeError(
        f"Expected four-channel edge weights; found shape {tuple(preflight_edge_weight.shape)}"
    )

label_complete_through = all_dates[-(label_t + 1)]
DATA_AUDIT = {
    "status": "OK",
    "input_sha256": {key: sha256_file(path) for key, path in staged_paths.items()},
    "market_date_min": all_dates[0],
    "market_date_max": all_dates[-1],
    "selector_created_at_utc": selector_meta.get("created_at_utc"),
    "price_export_created_at_utc": price_meta.get("created_at_utc"),
    "price_requested_identifiers": price_meta.get("requested_identifiers"),
    "price_resolved_identifiers": price_meta.get("resolved_identifiers_with_rows"),
    "price_metadata_date_max": price_meta.get("date_max"),
    "pit_union_names": int(pit["kdcode"].nunique()),
    "final_active_names": len(active_end),
    "final_complete_ohlc_names": len(complete_end),
    "snapshot_count": int(len(snapshot_counts)),
    "snapshot_min_names": int(snapshot_counts.min()),
    "snapshot_max_names": int(snapshot_counts.max()),
    "daily_active_min": min(row["active"] for row in daily_coverage),
    "daily_active_max": max(row["active"] for row in daily_coverage),
    "daily_complete_ohlcv_min": min(row["complete_ohlcv"] for row in daily_coverage),
    "daily_complete_ohlcv_max": max(row["complete_ohlcv"] for row in daily_coverage),
    "static_graph_edge_count": int(preflight_edge_index.shape[1]),
    "static_graph_edge_feature_dim": int(preflight_edge_weight.shape[1]),
    "pretrain_session_count": len(pretrain_dates),
    "train_session_count": len([d for d in all_dates if train_start <= d <= train_end]),
    "validation_session_count": len([d for d in all_dates if val_start <= d <= val_end]),
    "test_prediction_session_count": len([d for d in all_dates if test_start <= d <= test_end]),
    "train_label_target_end": train_target_end,
    "train_per_stock_embargo": train_stock_embargo,
    "first_validation_session": first_val_session,
    "validation_label_target_end": val_target_end,
    "validation_per_stock_embargo": validation_stock_embargo,
    "first_test_session": first_test_session,
    "test_prediction_end": test_end,
    "test_label_complete_through": label_complete_through,
}
print(json.dumps(DATA_AUDIT, indent=2, sort_keys=True))
"""

compose_audit_cell = r"""
from hydra import compose, initialize_config_dir
from mci_gru.config import create_config_from_dict
from omegaconf import OmegaConf

with initialize_config_dir(version_base=None, config_dir=str(REPO_DIR / "configs")):
    live_cfg = compose(
        config_name="config",
        overrides=[f"+experiment={CAMPAIGN['hydra_experiment']}"],
    )
live_resolved = OmegaConf.to_container(live_cfg, resolve=True)
if live_resolved != APPROVAL_BUNDLE["resolved_hydra_config"]:
    raise RuntimeError("Live resolved Hydra config differs from the approved config bundle.")
create_config_from_dict(live_resolved)

RESOLVED_CONFIGS = {}
for base_seed in BASE_SEEDS:
    job_name = f"lambdarankic_2026_ytd_110_name_seed{base_seed}"
    with initialize_config_dir(version_base=None, config_dir=str(REPO_DIR / "configs")):
        job_cfg = compose(
            config_name="config",
            overrides=[
                f"+experiment={CAMPAIGN['hydra_experiment']}",
                f"seed={base_seed}",
                f"experiment_name={job_name}",
            ],
        )
    resolved = OmegaConf.to_container(job_cfg, resolve=True)
    typed = create_config_from_dict(resolved)
    assert typed.training.loss_type == "lambdarank_ic"
    assert typed.training.selection_metric == "val_rank_ic"
    assert typed.training.num_models == CAMPAIGN["ensemble_models_per_seed"]
    RESOLVED_CONFIGS[base_seed] = OmegaConf.to_yaml(job_cfg, resolve=True)

max_all_pairs = (
    CAMPAIGN["universe"]["expected_active_names"]
    * (CAMPAIGN["universe"]["expected_active_names"] - 1)
    // 2
)
pair_cap = live_resolved["training"]["lambdarank_ic_max_pairs_per_day"]
if pair_cap < max_all_pairs:
    raise RuntimeError(f"Pair cap {pair_cap} is below the 110-name all-pairs count {max_all_pairs}.")
print("Resolved config verified for every base seed.")
print("110-name all-pairs count:", max_all_pairs)
print("Configured pair cap:", pair_cap)
"""

launch_cell = r"""
# EDIT THESE TWO VALUES ONLY AFTER THE USER APPROVES THE DISPLAYED DIGEST.
RUN_TRAINING = False
APPROVED_CONFIG_SHA256 = ""

if not RUN_TRAINING:
    raise RuntimeError("Approval gate closed: RUN_TRAINING is False. No run was started.")
launch_approval_json = json.dumps(
    APPROVAL_BUNDLE,
    sort_keys=True,
    separators=(",", ":"),
)
launch_config_sha256 = hashlib.sha256(launch_approval_json.encode("utf-8")).hexdigest()
if launch_config_sha256 != EXPECTED_CONFIG_SHA256:
    raise RuntimeError("Approval gate closed: the in-memory approval bundle changed.")
if APPROVED_CONFIG_SHA256 != launch_config_sha256:
    raise RuntimeError(
        "Approval gate closed: APPROVED_CONFIG_SHA256 does not match the displayed config digest."
    )
CONFIG_SHA256 = launch_config_sha256
if DATA_AUDIT.get("status") != "OK":
    raise RuntimeError("Approval gate closed: data audit is not OK.")
if set(RESOLVED_CONFIGS) != set(BASE_SEEDS):
    raise RuntimeError("Approval gate closed: not every base seed has a resolved config.")

def source_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()

def assert_approved_sources() -> None:
    mismatches = {}
    for relative, expected in CAMPAIGN["source_files_sha256"].items():
        observed = source_text_sha256(REPO_DIR / relative)
        if observed != expected:
            mismatches[relative] = {"expected": expected, "observed": observed}
    if mismatches:
        raise RuntimeError(f"Approval gate closed: approved source files changed: {mismatches}")

assert_approved_sources()

from datetime import timezone

RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOCAL_RUN_ROOT = Path("/content/lambdarankic_2026_ytd") / RUN_TAG
LOCAL_TRAINING_ROOT = LOCAL_RUN_ROOT / "training"
DRIVE_RUN_ROOT = Path(CAMPAIGN["artifact_contract"]["drive_root"]) / RUN_TAG
if DRIVE_RUN_ROOT.exists():
    raise RuntimeError(f"Refusing to reuse Drive run root: {DRIVE_RUN_ROOT}")
LOCAL_TRAINING_ROOT.mkdir(parents=True, exist_ok=False)
DRIVE_RUN_ROOT.mkdir(parents=True, exist_ok=False)

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)

def sync_tree(source: Path, destination: Path, *, force: bool = False) -> int:
    if not source.exists():
        return 0
    copied = 0
    now = time.time()
    for source_path in source.rglob("*"):
        if not source_path.is_file() or source_path.name.endswith(".tmp"):
            continue
        if not force and now - source_path.stat().st_mtime < 15:
            continue
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            force
            or not destination_path.exists()
            or destination_path.stat().st_size != source_path.stat().st_size
            or destination_path.stat().st_mtime < source_path.stat().st_mtime
        ):
            shutil.copy2(source_path, destination_path)
            copied += 1
    return copied

def heartbeat(status: str, phase: str, **extra) -> None:
    payload = {
        "status": status,
        "phase": phase,
        "campaign_id": CAMPAIGN["campaign_id"],
        "config_sha256": CONFIG_SHA256,
        "gpu_name": GPU_NAME,
        "run_tag": RUN_TAG,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_json(DRIVE_RUN_ROOT / "heartbeat.json", payload)

git_commit = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    text=True,
).strip()
approval_record = {
    "status": "APPROVED_FOR_LAUNCH",
    "config_sha256": CONFIG_SHA256,
    "approved_config_sha256": APPROVED_CONFIG_SHA256,
    "code_branch": BRANCH,
    "git_commit": git_commit,
    "gpu_name": GPU_NAME,
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "approval_bundle": APPROVAL_BUNDLE,
}
write_json(DRIVE_RUN_ROOT / "config_approval.json", approval_record)
write_json(DRIVE_RUN_ROOT / "data_audit.json", DATA_AUDIT)
gpu_evidence = subprocess.run(
    ["nvidia-smi"],
    text=True,
    capture_output=True,
    check=False,
)
(DRIVE_RUN_ROOT / "runtime_gpu.txt").write_text(
    gpu_evidence.stdout + "\n" + gpu_evidence.stderr,
    encoding="utf-8",
)
for base_seed, resolved_yaml in RESOLVED_CONFIGS.items():
    resolved_path = DRIVE_RUN_ROOT / "resolved_configs" / f"prelaunch_seed{base_seed}.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(resolved_yaml, encoding="utf-8")

def latest_run_dir(job_name: str) -> Path:
    candidates = sorted(
        path.parent
        for path in (LOCAL_TRAINING_ROOT / job_name).glob("*/training_summary.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No completed run directory found for {job_name}")
    return candidates[-1]

def write_result_tables(rows: list[dict]) -> None:
    write_json(DRIVE_RUN_ROOT / "training_results.json", rows)
    scalar_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "evaluation_metrics"}
        flat.update(row.get("evaluation_metrics", {}))
        scalar_rows.append(flat)
    if not scalar_rows:
        return
    fieldnames = sorted({key for row in scalar_rows for key in row})
    csv_path = DRIVE_RUN_ROOT / "training_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)

rows = []
heartbeat("RUNNING", "launch", completed_jobs=0, expected_jobs=len(BASE_SEEDS))
try:
    for base_seed in BASE_SEEDS:
        assert_approved_sources()
        job_name = f"lambdarankic_2026_ytd_110_name_seed{base_seed}"
        log_path = LOCAL_RUN_ROOT / "logs" / f"{job_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        launch_overrides = [
            f"+experiment={CAMPAIGN['hydra_experiment']}",
            f"seed={base_seed}",
            f"experiment_name={job_name}",
            f"output_dir={LOCAL_TRAINING_ROOT.as_posix()}",
        ]
        with initialize_config_dir(version_base=None, config_dir=str(REPO_DIR / "configs")):
            launch_job_cfg = compose(
                config_name="config",
                overrides=launch_overrides,
            )
        launch_resolved = OmegaConf.to_container(launch_job_cfg, resolve=True)
        expected_launch = json.loads(
            json.dumps(APPROVAL_BUNDLE["resolved_hydra_config"])
        )
        expected_launch["seed"] = base_seed
        expected_launch["experiment_name"] = job_name
        expected_launch["output_dir"] = LOCAL_TRAINING_ROOT.as_posix()
        if launch_resolved != expected_launch:
            raise RuntimeError(
                f"Launch config for seed {base_seed} differs from the approved config."
            )
        create_config_from_dict(launch_resolved)
        launch_config_path = (
            DRIVE_RUN_ROOT / "resolved_configs" / f"launch_seed{base_seed}.yaml"
        )
        launch_config_path.write_text(
            OmegaConf.to_yaml(launch_job_cfg, resolve=True),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-u",
            str(REPO_DIR / "run_experiment.py"),
            *launch_overrides,
        ]
        started = time.time()
        heartbeat(
            "RUNNING",
            "training",
            current_job=job_name,
            base_seed=base_seed,
            completed_jobs=len(rows),
            expected_jobs=len(BASE_SEEDS),
        )
        print("Starting", job_name, "on", GPU_NAME)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            while process.poll() is None:
                time.sleep(CAMPAIGN["artifact_contract"]["periodic_sync_seconds"])
                copied = sync_tree(LOCAL_RUN_ROOT, DRIVE_RUN_ROOT)
                heartbeat(
                    "RUNNING",
                    "training",
                    current_job=job_name,
                    base_seed=base_seed,
                    completed_jobs=len(rows),
                    expected_jobs=len(BASE_SEEDS),
                    elapsed_seconds=round(time.time() - started, 1),
                    files_synced=copied,
                )
        if process.returncode != 0:
            sync_tree(LOCAL_RUN_ROOT, DRIVE_RUN_ROOT, force=True)
            raise RuntimeError(f"Training failed for {job_name}; return code {process.returncode}")

        run_dir = latest_run_dir(job_name)
        training_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        evaluation_summary = json.loads(
            (run_dir / "evaluation_summary.json").read_text(encoding="utf-8")
        )
        checkpoint_count = len(list((run_dir / "checkpoints").glob("model_*_best.pth")))
        averaged_prediction_count = len(list((run_dir / "averaged_predictions").glob("*.csv")))
        per_model_prediction_dirs = len(list(run_dir.glob("predictions_model_*")))
        if checkpoint_count != CAMPAIGN["ensemble_models_per_seed"]:
            raise RuntimeError(f"Expected 20 checkpoints for {job_name}; found {checkpoint_count}")
        if per_model_prediction_dirs != CAMPAIGN["ensemble_models_per_seed"]:
            raise RuntimeError(
                f"Expected 20 per-model prediction directories for {job_name}; "
                f"found {per_model_prediction_dirs}"
            )
        if averaged_prediction_count != DATA_AUDIT["test_prediction_session_count"]:
            raise RuntimeError(
                f"Averaged-prediction count mismatch for {job_name}: "
                f"{averaged_prediction_count} vs {DATA_AUDIT['test_prediction_session_count']}"
            )
        if not (run_dir / "graph_data.pt").is_file():
            raise FileNotFoundError(f"Missing graph_data.pt for {job_name}")

        row = {
            "status": "OK",
            "job_name": job_name,
            "base_seed": base_seed,
            "ensemble_models": checkpoint_count,
            "averaged_prediction_count": averaged_prediction_count,
            "elapsed_seconds": round(time.time() - started, 1),
            "run_dir": str(run_dir),
            "mean_best_val_rank_ic": training_summary.get("mean_best_val_rank_ic"),
            "mean_best_val_ic": training_summary.get("mean_best_val_ic"),
            "mean_best_val_loss": training_summary.get("mean_best_val_loss"),
            "evaluation_metrics": evaluation_summary.get("metrics", {}),
        }
        rows.append(row)
        sync_tree(LOCAL_RUN_ROOT, DRIVE_RUN_ROOT, force=True)
        write_result_tables(rows)
        heartbeat(
            "RUNNING",
            "job_complete",
            current_job=job_name,
            completed_jobs=len(rows),
            expected_jobs=len(BASE_SEEDS),
        )

    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.get("evaluation_metrics", {}).items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        }
    )
    cross_seed = {}
    for key in numeric_keys:
        values = [
            float(row["evaluation_metrics"][key])
            for row in rows
            if isinstance(row.get("evaluation_metrics", {}).get(key), (int, float))
            and np.isfinite(row["evaluation_metrics"][key])
        ]
        if values:
            cross_seed[key] = {
                "n": len(values),
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "values": values,
            }
    write_json(DRIVE_RUN_ROOT / "cross_seed_evaluation_summary.json", cross_seed)
    write_json(
        DRIVE_RUN_ROOT / "run_summary.json",
        {
            "status": "OK",
            "campaign_id": CAMPAIGN["campaign_id"],
            "config_sha256": CONFIG_SHA256,
            "git_commit": git_commit,
            "gpu_name": GPU_NAME,
            "data_audit": DATA_AUDIT,
            "completed_jobs": len(rows),
            "expected_jobs": len(BASE_SEEDS),
            "training_rows": rows,
            "cross_seed_evaluation_summary": cross_seed,
        },
    )
    heartbeat("OK", "complete", completed_jobs=len(rows), expected_jobs=len(BASE_SEEDS))
    print("Complete. Durable run root:", DRIVE_RUN_ROOT)
except Exception as exc:
    sync_tree(LOCAL_RUN_ROOT, DRIVE_RUN_ROOT, force=True)
    heartbeat(
        "FAILED",
        "failed",
        error=repr(exc),
        completed_jobs=len(rows),
        expected_jobs=len(BASE_SEEDS),
    )
    raise
finally:
    try:
        from google.colab import runtime

        runtime.unassign()
    except Exception as exc:
        print("Manual Runtime > Disconnect and delete runtime may be needed:", exc)
"""

cells = [
    md(
        f"""
        # LambdaRankIC 2026-YTD Multi-Seed Confirmation

        This notebook is pinned to `{BRANCH}` and the 110-name monthly PIT
        sector-balanced universe. It trains five independent base-seed runs;
        each base seed contains a 20-member ensemble, for 100 model fits total.

        **Approval safety:** the source ships with `RUN_TRAINING = False` and an
        empty approval digest. The final foreground cell refuses to create a
        Drive run folder or launch `run_experiment.py` until the user-approved
        SHA-256 is entered exactly.

        Runtime contract: visible Colab **G4 GPU** only. T4, L4, and CPU are
        rejected. Durable checkpoints, per-model predictions, averaged
        predictions, graph data, logs, heartbeats, and summaries are synced to
        Drive. The final foreground cell unassigns the runtime in `finally`.

        Approval digest: `{APPROVAL_SHA256}`
        """
    ),
    md("## 1. Visible G4 setup (run only after conversational approval)"),
    colab_setup_cell(
        branch=BRANCH,
        blocked_gpu_names=("T4", "L4"),
        allowed_gpu_markers=("G4", "RTX PRO", "BLACKWELL"),
        strict_gpu_markers=("G4", "RTX PRO", "BLACKWELL"),
        extra_setup_source=r"""
        import hashlib
        import numpy as np

        from mci_gru.config import TrainingConfig
        from mci_gru.training.losses import build_training_loss

        probe_cfg = TrainingConfig(
            loss_type="lambdarank_ic",
            selection_metric="val_rank_ic",
            lambdarank_ic_max_pairs_per_day=8192,
            lambdarank_ic_temperature=1.0,
        )
        probe_loss, probe_name = build_training_loss(probe_cfg)
        print("LambdaRankIC probe:", probe_name, type(probe_loss).__name__)
        """,
    ),
    md("## 2. Display the immutable approval bundle"),
    code(approval_cell),
    md("## 3. Stage refreshed inputs and prove split/data coverage"),
    code(data_audit_cell),
    md("## 4. Recompose and validate every seed configuration"),
    code(compose_audit_cell),
    md("## 5. Foreground launcher — closed until explicit approval"),
    code(launch_cell),
]

write_notebook(cells, OUT, indent=1, trailing_newline=True)
APPROVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
APPROVAL_OUT.write_text(
    json.dumps(
        {"config_sha256": APPROVAL_SHA256, **APPROVAL_BUNDLE},
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Wrote {APPROVAL_OUT}")
print(f"Approval SHA-256: {APPROVAL_SHA256}")
