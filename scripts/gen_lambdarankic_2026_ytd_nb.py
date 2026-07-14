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
BRANCH = "codex/lambdarankic-ytd-drive-durability-20260714"
APPROVED_CODE_BRANCH = "codex/lambdarankic-2026-ytd-20260713"
APPROVED_CAMPAIGN_COMMIT = "9bd17d5b7ff14594681c7bdbee3bb17a9882b264"
CAMPAIGN_DRIVE_FOLDER_ID = "1iXHVKRwHBF3Jv_ruIMNWYIXFEyeYy4Po"

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
    "code_branch": APPROVED_CODE_BRANCH,
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

import io
import mimetypes
import zipfile
from datetime import timezone

from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from mci_gru.evaluation.run_bundle import write_run_manifest
from scripts.colab_recovery_upload_filter import iter_recovery_upload_files

CAMPAIGN_DRIVE_FOLDER_ID = "__CAMPAIGN_DRIVE_FOLDER_ID__"
APPROVED_CAMPAIGN_COMMIT = "__APPROVED_CAMPAIGN_COMMIT__"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOCAL_RUN_ROOT = Path("/content/lambdarankic_2026_ytd") / RUN_TAG
LOCAL_TRAINING_ROOT = LOCAL_RUN_ROOT / "training"
LOCAL_TRAINING_ROOT.mkdir(parents=True, exist_ok=False)

def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")

def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temp.replace(path)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def escape_drive_value(value: str) -> str:
    return value.replace("'", "\\'")

def execute_with_retries(request, *, label: str, attempts: int = 5):
    for attempt in range(1, attempts + 1):
        try:
            return request.execute()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(60, 2**attempt)
            print(f"{label} failed on attempt {attempt}/{attempts}: {exc!r}; retrying in {delay}s")
            time.sleep(delay)

auth.authenticate_user()
DRIVE_SERVICE = build("drive", "v3")
DRIVE_FOLDER_CACHE = {}
DRIVE_FILE_IDS = {}
PUBLISHED_FINGERPRINTS = {}
PUBLISHED_METADATA = {}

def find_drive_child(parent_id: str, name: str, *, mime_type: str | None = None):
    clauses = [
        f"'{parent_id}' in parents",
        f"name = '{escape_drive_value(name)}'",
        "trashed = false",
    ]
    if mime_type:
        clauses.append(f"mimeType = '{mime_type}'")
    response = execute_with_retries(
        DRIVE_SERVICE.files().list(
            q=" and ".join(clauses),
            fields="files(id,name,mimeType,size,md5Checksum)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        label=f"find Drive child {name}",
    )
    files = response.get("files", [])
    if len(files) > 1:
        raise RuntimeError(f"Ambiguous Drive children named {name!r} under {parent_id}")
    return files[0] if files else None

def create_drive_folder(parent_id: str, name: str) -> str:
    existing = find_drive_child(parent_id, name, mime_type=DRIVE_FOLDER_MIME)
    if existing:
        raise FileExistsError(f"Refusing to reuse Drive run folder {name!r}: {existing['id']}")
    created = execute_with_retries(
        DRIVE_SERVICE.files().create(
            body={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
            fields="id",
            supportsAllDrives=True,
        ),
        label=f"create Drive run folder {name}",
    )
    return created["id"]

def ensure_drive_folder(parent_id: str, name: str) -> str:
    key = (parent_id, name)
    if key in DRIVE_FOLDER_CACHE:
        return DRIVE_FOLDER_CACHE[key]
    existing = find_drive_child(parent_id, name, mime_type=DRIVE_FOLDER_MIME)
    if existing:
        folder_id = existing["id"]
    else:
        created = execute_with_retries(
            DRIVE_SERVICE.files().create(
                body={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
                fields="id",
                supportsAllDrives=True,
            ),
            label=f"create Drive folder {name}",
        )
        folder_id = created["id"]
    DRIVE_FOLDER_CACHE[key] = folder_id
    return folder_id

DRIVE_RUN_FOLDER_ID = create_drive_folder(CAMPAIGN_DRIVE_FOLDER_ID, RUN_TAG)
DRIVE_RUN_URL = f"https://drive.google.com/drive/folders/{DRIVE_RUN_FOLDER_ID}"

def ensure_remote_parent(relative_path: Path) -> str:
    parent_id = DRIVE_RUN_FOLDER_ID
    for part in relative_path.parts[:-1]:
        parent_id = ensure_drive_folder(parent_id, part)
    return parent_id

def resolve_remote_folder(parts: tuple[str, ...]) -> str:
    parent_id = DRIVE_RUN_FOLDER_ID
    for part in parts:
        child = find_drive_child(parent_id, part, mime_type=DRIVE_FOLDER_MIME)
        if not child:
            raise FileNotFoundError(f"Missing remote Drive folder: {'/'.join(parts)}")
        parent_id = child["id"]
    return parent_id

def verify_remote_file_metadata(local_path: Path, remote: dict, relative_value: str) -> dict:
    local_size = local_path.stat().st_size
    remote_size = int(remote.get("size", -1))
    if remote_size != local_size:
        raise RuntimeError(
            f"Remote size mismatch for {relative_value}: local={local_size} remote={remote_size}"
        )
    local_md5 = md5_file(local_path)
    remote_md5 = remote.get("md5Checksum")
    if not remote_md5 or remote_md5 != local_md5:
        raise RuntimeError(
            f"Remote MD5 mismatch for {relative_value}: local={local_md5} remote={remote_md5}"
        )
    return {
        "path": relative_value,
        "file_id": remote["id"],
        "size_bytes": local_size,
        "md5": local_md5,
        "sha256": sha256_file(local_path),
    }

def upload_relative_file(local_path: Path, *, force: bool = False) -> dict:
    relative_path = local_path.relative_to(LOCAL_RUN_ROOT)
    relative_value = relative_path.as_posix()
    stat = local_path.stat()
    fingerprint = (stat.st_size, stat.st_mtime_ns)
    if not force and PUBLISHED_FINGERPRINTS.get(relative_value) == fingerprint:
        return PUBLISHED_METADATA[relative_value]
    parent_id = ensure_remote_parent(relative_path)
    cache_key = (parent_id, relative_path.name)
    file_id = DRIVE_FILE_IDS.get(cache_key)
    if file_id is None:
        existing = find_drive_child(parent_id, relative_path.name)
        file_id = existing["id"] if existing else None
    mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    media = MediaFileUpload(
        str(local_path),
        mimetype=mime_type,
        chunksize=DRIVE_UPLOAD_CHUNK_BYTES,
        resumable=True,
    )
    if file_id:
        request = DRIVE_SERVICE.files().update(
            fileId=file_id,
            media_body=media,
            fields="id,name,size,md5Checksum",
            supportsAllDrives=True,
        )
    else:
        request = DRIVE_SERVICE.files().create(
            body={"name": relative_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id,name,size,md5Checksum",
            supportsAllDrives=True,
        )
    response = None
    while response is None:
        _, response = request.next_chunk(num_retries=5)
    file_id = response["id"]
    DRIVE_FILE_IDS[cache_key] = file_id
    remote = execute_with_retries(
        DRIVE_SERVICE.files().get(
            fileId=file_id,
            fields="id,name,size,md5Checksum",
            supportsAllDrives=True,
        ),
        label=f"read back Drive metadata for {relative_value}",
    )
    verified = verify_remote_file_metadata(local_path, remote, relative_value)
    PUBLISHED_FINGERPRINTS[relative_value] = fingerprint
    PUBLISHED_METADATA[relative_value] = verified
    return verified

def upload_json_verified(relative_value: str, payload) -> dict:
    local_path = LOCAL_RUN_ROOT / relative_value
    write_json(local_path, payload)
    return upload_relative_file(local_path, force=True)

def upload_text_verified(relative_value: str, text_value: str) -> dict:
    local_path = LOCAL_RUN_ROOT / relative_value
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp = local_path.with_name(local_path.name + ".tmp")
    temp.write_text(text_value, encoding="utf-8")
    temp.replace(local_path)
    return upload_relative_file(local_path, force=True)

def read_remote_json(relative_value: str) -> dict:
    relative_path = Path(relative_value)
    parent_id = resolve_remote_folder(tuple(relative_path.parts[:-1]))
    remote_file = find_drive_child(parent_id, relative_path.name)
    if not remote_file:
        raise FileNotFoundError(f"Missing remote JSON artifact: {relative_value}")
    request = DRIVE_SERVICE.files().get_media(
        fileId=remote_file["id"],
        supportsAllDrives=True,
    )
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk(num_retries=5)
    return json.loads(buffer.getvalue().decode("utf-8"))

def list_drive_children(parent_id: str) -> list[dict]:
    children = []
    page_token = None
    while True:
        response = execute_with_retries(
            DRIVE_SERVICE.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                fields="nextPageToken,files(id,name,mimeType,size,md5Checksum)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            label=f"list Drive folder {parent_id}",
        )
        children.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return children

def verify_remote_csv_directory(relative_directory: str, expected_count: int) -> dict:
    relative_path = Path(relative_directory)
    local_directory = LOCAL_RUN_ROOT / relative_path
    remote_folder_id = resolve_remote_folder(tuple(relative_path.parts))
    remote_csvs = {
        child["name"]: {
            "size": int(child.get("size", -1)),
            "md5": child.get("md5Checksum"),
        }
        for child in list_drive_children(remote_folder_id)
        if child["name"].lower().endswith(".csv")
    }
    local_csvs = {
        path.name: {"size": path.stat().st_size, "md5": md5_file(path)}
        for path in local_directory.glob("*.csv")
    }
    if len(local_csvs) != expected_count or len(remote_csvs) != expected_count:
        raise RuntimeError(
            f"Published CSV count mismatch for {relative_directory}: "
            f"expected={expected_count} local={len(local_csvs)} remote={len(remote_csvs)}"
        )
    if local_csvs != remote_csvs:
        raise RuntimeError(f"Published CSV names, sizes, or MD5 hashes differ for {relative_directory}")
    canonical = json.dumps(local_csvs, sort_keys=True, separators=(",", ":"))
    return {
        "path": relative_directory,
        "remote_folder_id": remote_folder_id,
        "csv_count": len(remote_csvs),
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }

def heartbeat(status: str, phase: str, **extra) -> dict:
    payload = {
        "status": status,
        "phase": phase,
        "campaign_id": CAMPAIGN["campaign_id"],
        "config_sha256": CONFIG_SHA256,
        "gpu_name": GPU_NAME,
        "run_tag": RUN_TAG,
        "drive_run_folder_id": DRIVE_RUN_FOLDER_ID,
        "drive_run_url": DRIVE_RUN_URL,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    upload_json_verified("heartbeat.json", payload)
    return payload

def latest_run_dir(job_name: str) -> Path:
    candidates = sorted(
        path.parent
        for path in (LOCAL_TRAINING_ROOT / job_name).glob("*/training_summary.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No completed run directory found for {job_name}")
    return candidates[-1]

def build_per_model_predictions_archive(
    run_dir: Path,
    expected_models: int,
    expected_csvs_per_model: int,
) -> tuple[Path, dict]:
    expected_names = {f"predictions_model_{index}" for index in range(expected_models)}
    observed_dirs = {path.name for path in run_dir.glob("predictions_model_*") if path.is_dir()}
    if observed_dirs != expected_names:
        raise RuntimeError(
            f"Per-model prediction directory mismatch: expected={sorted(expected_names)} "
            f"observed={sorted(observed_dirs)}"
        )
    archive_dir = run_dir / "model_artifacts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "per_model_predictions.zip"
    temp_path = archive_path.with_name(archive_path.name + ".tmp")
    entries = []
    with zipfile.ZipFile(
        temp_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for model_index in range(expected_models):
            prediction_dir = run_dir / f"predictions_model_{model_index}"
            csv_paths = sorted(prediction_dir.glob("*.csv"))
            if len(csv_paths) != expected_csvs_per_model:
                raise RuntimeError(
                    f"Expected {expected_csvs_per_model} CSVs in {prediction_dir.name}; "
                    f"found {len(csv_paths)}"
                )
            for csv_path in csv_paths:
                member_name = f"{prediction_dir.name}/{csv_path.name}"
                archive.write(csv_path, arcname=member_name)
                entries.append(
                    {
                        "path": member_name,
                        "size_bytes": csv_path.stat().st_size,
                        "sha256": sha256_file(csv_path),
                    }
                )
        manifest_payload = {
            "schema_version": 1,
            "model_count": expected_models,
            "csvs_per_model": expected_csvs_per_model,
            "member_count": len(entries),
            "members": entries,
        }
        manifest_json = json.dumps(
            manifest_payload,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        archive.writestr("per_model_predictions_manifest.json", manifest_json)
    temp_path.replace(archive_path)
    return archive_path, {
        "model_count": expected_models,
        "csvs_per_model": expected_csvs_per_model,
        "member_count": len(entries),
        "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    }

def publish_in_progress(job_name: str, log_path: Path) -> dict:
    published = []
    for checkpoint in sorted(
        (LOCAL_TRAINING_ROOT / job_name).glob("*/checkpoints/model_*_best.pth")
    ):
        published.append(upload_relative_file(checkpoint))
    if log_path.is_file():
        published.append(upload_relative_file(log_path))
    return {"published_or_verified": len(published)}

def publish_completed_seed(
    run_dir: Path,
    job_name: str,
    base_seed: int,
    command: list[str],
) -> dict:
    expected_models = CAMPAIGN["ensemble_models_per_seed"]
    expected_csvs = DATA_AUDIT["test_prediction_session_count"]
    bundle_paths = write_run_manifest(
        run_dir,
        selection_rule="mean of 20 members selected independently by val_rank_ic",
        command=" ".join(command),
        feature_lag_policy="strict PIT current-only features with actual-session label embargo",
        normalization_reference="train-period-only zscore recorded in run_metadata.json",
        graph_policy="static train-window graph frozen in graph_data.pt",
        seed_policy=f"base_seed={base_seed}; 20 deterministic ensemble-member seeds",
        paper_trade_eligible=False,
        repo_dir=REPO_DIR,
    )
    artifact_validation = json.loads(
        bundle_paths["validation"].read_text(encoding="utf-8")
    )
    if artifact_validation.get("status") != "OK":
        raise RuntimeError(
            f"Local run-bundle validation failed for {job_name}: {artifact_validation}"
        )
    archive_path, archive_summary = build_per_model_predictions_archive(
        run_dir,
        expected_models,
        expected_csvs,
    )
    selected = set(
        iter_recovery_upload_files(
            run_dir,
            upload_checkpoints=True,
            upload_per_model_predictions=False,
        )
    )
    for extra_path in (
        run_dir / "graph_data.pt",
        run_dir / "run_manifest.json",
        run_dir / "artifact_validation.json",
        archive_path,
    ):
        if extra_path.is_file():
            selected.add(extra_path)
    raw_prediction_files = [
        path
        for path in selected
        if any(part.startswith("predictions_model_") for part in path.relative_to(run_dir).parts)
    ]
    if raw_prediction_files:
        raise RuntimeError(f"Raw per-model predictions entered the Drive upload set: {raw_prediction_files[:3]}")
    uploaded = [upload_relative_file(path) for path in sorted(selected)]
    run_relative = run_dir.relative_to(LOCAL_RUN_ROOT).as_posix()
    averaged_verification = verify_remote_csv_directory(
        f"{run_relative}/averaged_predictions",
        expected_csvs,
    )
    checkpoint_names = {
        path.name for path in (run_dir / "checkpoints").glob("model_*_best.pth")
    }
    expected_checkpoint_names = {
        f"model_{index}_best.pth" for index in range(expected_models)
    }
    if checkpoint_names != expected_checkpoint_names:
        raise RuntimeError(
            f"Checkpoint IDs differ for {job_name}: "
            f"missing={sorted(expected_checkpoint_names - checkpoint_names)} "
            f"extra={sorted(checkpoint_names - expected_checkpoint_names)}"
        )
    seed_payload = {
        "schema_version": 1,
        "status": "VERIFIED",
        "job_name": job_name,
        "base_seed": base_seed,
        "config_sha256": CONFIG_SHA256,
        "run_relative_path": run_relative,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(checkpoint_names),
        "checkpoint_names": sorted(checkpoint_names),
        "averaged_predictions": averaged_verification,
        "per_model_predictions_archive": archive_summary,
        "uploaded_files": uploaded,
    }
    seed_manifest_relative = f"{run_relative}/seed_durability.json"
    upload_json_verified(seed_manifest_relative, seed_payload)
    readback = read_remote_json(seed_manifest_relative)
    if readback != seed_payload:
        raise RuntimeError(f"Remote seed durability readback differs for {job_name}")
    return {
        "status": "VERIFIED",
        "run_relative_path": run_relative,
        "seed_manifest_relative_path": seed_manifest_relative,
        "uploaded_file_count": len(uploaded),
        "averaged_predictions": averaged_verification,
        "per_model_predictions_archive": archive_summary,
    }

def write_result_tables(rows: list[dict]) -> None:
    upload_json_verified("training_results.json", rows)
    scalar_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "evaluation_metrics"}
        flat.update(row.get("evaluation_metrics", {}))
        scalar_rows.append(flat)
    if not scalar_rows:
        return
    fieldnames = sorted({key for row in scalar_rows for key in row})
    csv_path = LOCAL_RUN_ROOT / "training_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)
    upload_relative_file(csv_path, force=True)

def verify_remote_run(rows: list[dict]) -> bool:
    expected_seeds = set(BASE_SEEDS)
    verified_seeds = set()
    for row in rows:
        seed_payload = read_remote_json(row["remote_seed_manifest"])
        if seed_payload.get("status") != "VERIFIED":
            raise RuntimeError(f"Seed durability is not VERIFIED: {row['job_name']}")
        verified_seeds.add(int(seed_payload["base_seed"]))
    if verified_seeds != expected_seeds:
        raise RuntimeError(
            f"Remote seed set mismatch: expected={sorted(expected_seeds)} "
            f"observed={sorted(verified_seeds)}"
        )
    remote_summary = read_remote_json("run_summary.json")
    if remote_summary.get("status") != "OK":
        raise RuntimeError("Remote run_summary.json is not OK")
    if remote_summary.get("completed_jobs") != len(expected_seeds):
        raise RuntimeError("Remote run_summary.json does not prove all expected jobs")
    if remote_summary.get("config_sha256") != CONFIG_SHA256:
        raise RuntimeError("Remote run_summary.json config digest differs")
    remote_heartbeat = read_remote_json("heartbeat.json")
    if remote_heartbeat.get("status") != "OK" or remote_heartbeat.get("phase") != "complete":
        raise RuntimeError("Remote heartbeat is not terminal OK/complete")
    if not remote_heartbeat.get("remote_durability_verified"):
        raise RuntimeError("Remote heartbeat lacks the durability-verification marker")
    return True

launcher_git_commit = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    text=True,
).strip()
approval_record = {
    "status": "APPROVED_FOR_LAUNCH",
    "config_sha256": CONFIG_SHA256,
    "approved_config_sha256": APPROVED_CONFIG_SHA256,
    "approved_code_branch": CAMPAIGN["code_branch"],
    "approved_campaign_commit": APPROVED_CAMPAIGN_COMMIT,
    "launcher_branch": BRANCH,
    "launcher_commit": launcher_git_commit,
    "gpu_name": GPU_NAME,
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "approval_bundle": APPROVAL_BUNDLE,
}
upload_json_verified("config_approval.json", approval_record)
upload_json_verified("data_audit.json", DATA_AUDIT)
gpu_evidence = subprocess.run(
    ["nvidia-smi"],
    text=True,
    capture_output=True,
    check=False,
)
upload_text_verified(
    "runtime_gpu.txt",
    gpu_evidence.stdout + "\n" + gpu_evidence.stderr,
)
for base_seed, resolved_yaml in RESOLVED_CONFIGS.items():
    upload_text_verified(f"resolved_configs/prelaunch_seed{base_seed}.yaml", resolved_yaml)

rows = []
REMOTE_DURABILITY_VERIFIED = False
current_job = None
current_log_path = None
heartbeat("RUNNING", "launch", completed_jobs=0, expected_jobs=len(BASE_SEEDS))
try:
    for base_seed in BASE_SEEDS:
        assert_approved_sources()
        job_name = f"lambdarankic_2026_ytd_110_name_seed{base_seed}"
        current_job = job_name
        log_path = LOCAL_RUN_ROOT / "logs" / f"{job_name}.log"
        current_log_path = log_path
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
        expected_launch = json.loads(json.dumps(APPROVAL_BUNDLE["resolved_hydra_config"]))
        expected_launch["seed"] = base_seed
        expected_launch["experiment_name"] = job_name
        expected_launch["output_dir"] = LOCAL_TRAINING_ROOT.as_posix()
        if launch_resolved != expected_launch:
            raise RuntimeError(
                f"Launch config for seed {base_seed} differs from the approved config."
            )
        create_config_from_dict(launch_resolved)
        launch_config_relative = f"resolved_configs/launch_seed{base_seed}.yaml"
        upload_text_verified(
            launch_config_relative,
            OmegaConf.to_yaml(launch_job_cfg, resolve=True),
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
                try:
                    progress = publish_in_progress(job_name, log_path)
                    heartbeat(
                        "RUNNING",
                        "training",
                        current_job=job_name,
                        base_seed=base_seed,
                        completed_jobs=len(rows),
                        expected_jobs=len(BASE_SEEDS),
                        elapsed_seconds=round(time.time() - started, 1),
                        **progress,
                    )
                except Exception as publication_exc:
                    print("Non-terminal Drive publication warning:", repr(publication_exc))
        publish_in_progress(job_name, log_path)
        if process.returncode != 0:
            raise RuntimeError(f"Training failed for {job_name}; return code {process.returncode}")

        run_dir = latest_run_dir(job_name)
        training_summary = json.loads(
            (run_dir / "training_summary.json").read_text(encoding="utf-8")
        )
        evaluation_summary = json.loads(
            (run_dir / "evaluation_summary.json").read_text(encoding="utf-8")
        )
        checkpoints = sorted((run_dir / "checkpoints").glob("model_*_best.pth"))
        averaged_predictions = sorted((run_dir / "averaged_predictions").glob("*.csv"))
        prediction_dirs = sorted(path for path in run_dir.glob("predictions_model_*") if path.is_dir())
        if len(checkpoints) != CAMPAIGN["ensemble_models_per_seed"]:
            raise RuntimeError(f"Expected 20 checkpoints for {job_name}; found {len(checkpoints)}")
        if len(prediction_dirs) != CAMPAIGN["ensemble_models_per_seed"]:
            raise RuntimeError(
                f"Expected 20 per-model prediction directories for {job_name}; "
                f"found {len(prediction_dirs)}"
            )
        if len(averaged_predictions) != DATA_AUDIT["test_prediction_session_count"]:
            raise RuntimeError(
                f"Averaged-prediction count mismatch for {job_name}: "
                f"{len(averaged_predictions)} vs {DATA_AUDIT['test_prediction_session_count']}"
            )
        if not (run_dir / "graph_data.pt").is_file():
            raise FileNotFoundError(f"Missing graph_data.pt for {job_name}")

        heartbeat(
            "RUNNING",
            "publishing_seed",
            current_job=job_name,
            base_seed=base_seed,
            completed_jobs=len(rows),
            expected_jobs=len(BASE_SEEDS),
        )
        seed_remote = publish_completed_seed(
            run_dir,
            job_name,
            base_seed,
            command,
        )
        row = {
            "status": "OK",
            "job_name": job_name,
            "base_seed": base_seed,
            "ensemble_models": len(checkpoints),
            "averaged_prediction_count": len(averaged_predictions),
            "elapsed_seconds": round(time.time() - started, 1),
            "local_run_dir": str(run_dir),
            "remote_run_relative_path": seed_remote["run_relative_path"],
            "remote_seed_manifest": seed_remote["seed_manifest_relative_path"],
            "remote_durability_status": seed_remote["status"],
            "mean_best_val_rank_ic": training_summary.get("mean_best_val_rank_ic"),
            "mean_best_val_ic": training_summary.get("mean_best_val_ic"),
            "mean_best_val_loss": training_summary.get("mean_best_val_loss"),
            "evaluation_metrics": evaluation_summary.get("metrics", {}),
        }
        rows.append(row)
        write_result_tables(rows)
        heartbeat(
            "RUNNING",
            "job_complete",
            current_job=job_name,
            base_seed=base_seed,
            completed_jobs=len(rows),
            expected_jobs=len(BASE_SEEDS),
            remote_durability_status="VERIFIED",
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
    upload_json_verified("cross_seed_evaluation_summary.json", cross_seed)
    run_summary = {
        "status": "OK",
        "campaign_id": CAMPAIGN["campaign_id"],
        "config_sha256": CONFIG_SHA256,
        "approved_code_branch": CAMPAIGN["code_branch"],
        "approved_campaign_commit": APPROVED_CAMPAIGN_COMMIT,
        "launcher_branch": BRANCH,
        "launcher_commit": launcher_git_commit,
        "gpu_name": GPU_NAME,
        "drive_run_folder_id": DRIVE_RUN_FOLDER_ID,
        "drive_run_url": DRIVE_RUN_URL,
        "data_audit": DATA_AUDIT,
        "completed_jobs": len(rows),
        "expected_jobs": len(BASE_SEEDS),
        "training_rows": rows,
        "cross_seed_evaluation_summary": cross_seed,
    }
    upload_json_verified("run_summary.json", run_summary)
    heartbeat(
        "OK",
        "complete",
        completed_jobs=len(rows),
        expected_jobs=len(BASE_SEEDS),
        verified_seeds=[row["base_seed"] for row in rows],
        remote_durability_verified=True,
    )
    REMOTE_DURABILITY_VERIFIED = verify_remote_run(rows)
    print("Complete. Durable run root:", DRIVE_RUN_URL)
except Exception as exc:
    try:
        if current_job and current_log_path:
            publish_in_progress(current_job, current_log_path)
        heartbeat(
            "FAILED",
            "failed",
            error=repr(exc),
            completed_jobs=len(rows),
            expected_jobs=len(BASE_SEEDS),
            remote_durability_verified=False,
        )
    except Exception as failure_publish_exc:
        print("Unable to publish terminal failure heartbeat:", repr(failure_publish_exc))
    print("Runtime intentionally left assigned because remote durability was not proved.")
    raise

if REMOTE_DURABILITY_VERIFIED:
    try:
        from google.colab import runtime

        runtime.unassign()
    except Exception as exc:
        print("Manual Runtime > Disconnect and delete runtime may be needed:", exc)
else:
    print("Runtime intentionally left assigned because remote durability was not proved.")
""".replace("__CAMPAIGN_DRIVE_FOLDER_ID__", CAMPAIGN_DRIVE_FOLDER_ID).replace(
    "__APPROVED_CAMPAIGN_COMMIT__", APPROVED_CAMPAIGN_COMMIT
)

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
        rejected. Durable checkpoints, averaged predictions, graph data, logs,
        heartbeats, and summaries are uploaded through the Drive API and read
        back before success. Per-model predictions are stored as one verified
        archive per seed, never as thousands of individual Drive files. The
        runtime is unassigned only after terminal remote verification succeeds.

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
