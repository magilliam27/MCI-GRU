"""Generate the graph-specification ablation Colab notebook (ticket 166).

Runnable form of the issue-164 protocol: five screening arms (A0 zeroed
control, A1 shipped thr 0.8, A2 thr 0.5, A3 top-K 20, A4 sector-only) at
3 seeds x 20 epochs with frozen-recipe semantics preserved, then frozen-recipe
confirmation (20 x 100) for A0, A1, and any arm separating from A0. The
GOOG.OQ/GOOGL.OQ twin is excluded from every arm's adjacency via
``graph.exclude_edge_pairs``. Arbiter: test-span pooled daily IC with CI;
April's composite decision score is computed alongside and is not the arbiter;
per-span graph density/isolation and a twin-edge-count check are disclosed per
arm.

The protocol data below is module-level on purpose: the notebook cells are
rendered from it, and ``tests/test_graph_specification_ablation_notebook.py``
imports it to compose every arm through Hydra into a typed
``ExperimentConfig`` before any GPU minute is spent.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from pprint import pformat

from nb_lib import code, colab_setup_cell, md, write_notebook

OUT = Path("notebooks/graph_specification_ablation_colab.ipynb")

# ── protocol constants (issue-164 resolution; ticket-166 defaults confirmed
# with the maintainer) ─────────────────────────────────────────────────────

BASE_SEED = 1729
SCREEN_NUM_MODELS = 3
SCREEN_NUM_EPOCHS = 20
CONFIRM_NUM_MODELS = 20
CONFIRM_NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
BOOTSTRAP_RESAMPLES = 1000

TWIN_PAIR = ("GOOG.OQ", "GOOGL.OQ")
TWIN_OVERRIDE = '+graph.exclude_edge_pairs=[["GOOG.OQ","GOOGL.OQ"]]'

ALWAYS_CONFIRMED_ARMS = ["A0_zeroed", "A1_shipped"]

# The five arms. Adjacency-rule keys ride the arm (preset or inline override)
# and nothing else touches them; A1 pins the incumbent explicitly rather than
# inheriting a mutable base default.
ARMS: list[dict] = [
    {
        "key": "A0_zeroed",
        "label": "zeroed control",
        "overrides": ["+experiment=graph_zeroed"],
        "hypothesis": (
            "The anchor: empty correlation edges, GAT and parameters intact, "
            "self-loops only. An arm that cannot separate from this one is not "
            "earning its edges."
        ),
    },
    {
        "key": "A1_shipped",
        "label": "shipped threshold 0.8",
        "overrides": ["graph.judge_value=0.8", "graph.top_k=0"],
        "hypothesis": (
            "The incumbent, measured rather than assumed. Deployment-window "
            "edges are near-empty and 100% twin-or-intra-sector (Phase 0)."
        ),
    },
    {
        "key": "A2_thr05",
        "label": "recalibrated threshold 0.5",
        "overrides": ["+experiment=graph_thr05"],
        "hypothesis": (
            "Same selection rule at a sane calibration (~13% isolation vs ~76% "
            "at 0.8): separates calibration from rule."
        ),
    },
    {
        "key": "A3_topk20",
        "label": "top-K 20 (corr), static",
        "overrides": ["+experiment=graph_topk20_static"],
        "hypothesis": (
            "Best measured out-of-sample persistence, majority cross-sector; "
            "re-tests April 2026's anti-top-K result under the corrected PIT "
            "protocol with a populated rank-4 edge tensor."
        ),
    },
    {
        "key": "A4_sector_only",
        "label": "sector relation only",
        "overrides": ["+experiment=graph_sector_only"],
        "hypothesis": (
            "Exact, estimation-free version of the structure the shipped graph "
            "approximates. Extra dual-GAT parameters: specification-level, not "
            "parameter-matched."
        ),
    },
]

# Frozen-recipe semantics (docs/DEFAULT_EXPERIMENT_RECIPE.md), minus the graph
# block, the training budget, and the seed, which are handled separately.
RECIPE_OVERRIDES = [
    "data=gics_top10_110_2016",
    "features=with_momentum",
    "features.include_momentum=true",
    "features.include_weekly_momentum=true",
    "features.momentum_encoding=binary",
    "features.momentum_blend_mode=static",
    "features.momentum_blend_fast_weight=0.5",
    "features.include_global_regime=true",
    "features.regime_strict=true",
    "features.regime_enforce_lag_days=0",
    "features.regime_include_subsequent_returns=false",
    "features.regime_change_months=12",
    "features.regime_norm_months=120",
    "features.regime_exclusion_months=1",
    "features.regime_similarity_quantile=0.2",
    "features.regime_min_history_months=24",
    "training.learning_rate=5e-5",
    "training.lr_scheduler=cosine",
    "training.loss_type=ic",
    "training.label_type=returns",
    "training.selection_metric=val_ic",
    "training.shuffle_train=true",
    "model.label_t=5",
]

# Held fixed across arms by the protocol; deliberately excludes judge_value /
# top_k / top_k_metric / zero_edges / use_sector_relation, which are arm keys.
HELD_FIXED_OVERRIDES = [
    "graph.corr_lookback_days=252",
    "graph.update_frequency_months=0",
    "graph.use_multi_feature_edges=true",
    "graph.append_snapshot_age_days=false",
    "graph.use_lead_lag_features=false",
    "graph.drop_edge_p=0.1",
]

# Tracking off: the Drive export is the record for these runs.
RUN_CONTROL_OVERRIDES = [
    "tracking.enabled=false",
    "tracking.log_artifacts=false",
    "tracking.log_checkpoints=false",
    "tracking.log_predictions=false",
]


def arm_overrides(arm: dict, num_models: int, num_epochs: int) -> list[str]:
    """Full Hydra override list for one arm at one training budget."""
    return [
        *arm["overrides"],
        *RECIPE_OVERRIDES,
        *HELD_FIXED_OVERRIDES,
        TWIN_OVERRIDE,
        f"training.num_models={num_models}",
        f"training.num_epochs={num_epochs}",
        f"training.early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        f"evaluation.bootstrap_resamples={BOOTSTRAP_RESAMPLES}",
        *RUN_CONTROL_OVERRIDES,
        f"seed={BASE_SEED}",
    ]


def _embed(obj) -> str:
    """Render *obj* for interpolation into an 8-space-indented cell template.

    Continuation lines get the template's indent so ``textwrap.dedent`` inside
    ``code()`` strips a uniform prefix; ``lstrip`` leaves the first line to sit
    where the template places it.
    """
    return textwrap.indent(pformat(obj, width=88, sort_dicts=False), " " * 8).lstrip()


def _embed_source(func) -> str:
    return textwrap.indent(inspect.getsource(func), " " * 8).lstrip()


cells = [
    md(
        """
        # Graph-Specification Ablation (Wayfinder map 157, ticket 166)

        Runnable form of the ticket-164 protocol: five screening arms over the
        correlation graph's specification, a cheap screen (3 seeds x 20
        epochs), and frozen-recipe confirmation (20 x 100) for A0, A1, and any
        arm separating from A0 on the screen.

        - **Arbiter:** test-span pooled daily IC with a bootstrap CI
          (`avg_ic`, `avg_ic_ci_lower`, `avg_ic_ci_upper` from each run's
          `evaluation_summary.json`), reported per calendar year.
        - **Alongside, not the arbiter:** April 2026's composite decision
          score, with its exact historical weights.
        - **Disclosed per arm:** graph density and isolated-node fraction per
          span on the PIT-admissible axis, and a twin-edge-count check that
          must read zero everywhere.
        - **Hygiene rule:** the GOOG.OQ/GOOGL.OQ same-company twin is excluded
          from every constructed adjacency in every arm, sector included.

        Set `RUN_STAGE = "screen"` first. After reviewing the promotion cell's
        output, rerun with `RUN_STAGE = "confirm"` (same `RUN_TAG` root) to
        train the promoted arms at the frozen recipe. `SMOKE_MODE = True` is a
        mechanics smoke only: it proves wiring, never performance.
        """
    ),
    md("## 1. Setup"),
    colab_setup_cell(branch="main"),
    md("## 2. FRED Key And Data Staging"),
    code(
        r"""
        import hashlib
        import shutil

        if IN_COLAB and not os.environ.get("FRED_API_KEY"):
            try:
                from google.colab import userdata

                secret = userdata.get("FRED_API_KEY")
                if secret:
                    os.environ["FRED_API_KEY"] = secret
                    print("FRED_API_KEY loaded from Colab Secrets.")
            except Exception as exc:
                print("Could not read FRED_API_KEY from Colab Secrets:", exc)

        if not os.environ.get("FRED_API_KEY"):
            raise RuntimeError("FRED_API_KEY is required: the recipe enables strict global regime features.")

        def sha256_file(path: Path) -> str:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        drive_data_dir = Path("/content/drive/MyDrive/MCI_GRU_shared/data")

        STAGED_FILES = {
            "market_csv": (
                "sp500_pit_gics_top10_mcap_monthly_20160104_20260731_lseg_20150101_20260731.csv",
                "data/raw/market",
            ),
            "pit_universe_csv": (
                "sp500_pit_gics_top10_mcap_monthly_20160104_20260731_pit_universe.csv",
                "data/raw/constituents",
            ),
            "sector_map_csv": (
                "sp500_pit_gics_top10_mcap_monthly_20160104_20260731_all_metadata_snapshots.csv",
                "data/raw/constituents",
            ),
        }

        staged: dict[str, Path] = {}
        staged_sha256: dict[str, str] = {}
        for role, (name, repo_rel_dir) in STAGED_FILES.items():
            repo_path = REPO_DIR / repo_rel_dir / name
            if IN_COLAB:
                src = drive_data_dir / name
                if not src.exists():
                    raise FileNotFoundError(f"Missing {role} on Drive: {src}")
                repo_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, repo_path)
            elif not repo_path.exists():
                raise FileNotFoundError(
                    f"Missing {role} locally: {repo_path} (stage it or run in Colab)"
                )
            staged[role] = repo_path
            staged_sha256[role] = sha256_file(repo_path)
            print(role, "->", repo_path, staged_sha256[role][:16])

        # The data config and the A4 preset reference these repo-relative
        # paths, so staging them is all the wiring the runs need.
        """
    ),
    md("## 3. Protocol: Arms, Recipe Semantics, Budgets"),
    code(
        f"""
        RUN_STAGE = "screen"  # screen | confirm
        SMOKE_MODE = False  # True = 1 model x 2 epochs: a mechanics smoke, never evidence
        RUN_TAG_OVERRIDE = ""
        # Confirm stage: leave empty to read PROMOTED_ARMS from the screen's
        # promotion.json under SCREEN_SOURCE_TAG; set to force an arm list.
        CONFIRM_ARMS_OVERRIDE: list[str] = []
        SCREEN_SOURCE_TAG = ""

        BASE_SEED = {BASE_SEED}
        SCREEN_NUM_MODELS = {SCREEN_NUM_MODELS}
        SCREEN_NUM_EPOCHS = {SCREEN_NUM_EPOCHS}
        CONFIRM_NUM_MODELS = {CONFIRM_NUM_MODELS}
        CONFIRM_NUM_EPOCHS = {CONFIRM_NUM_EPOCHS}
        EARLY_STOPPING_PATIENCE = {EARLY_STOPPING_PATIENCE}
        BOOTSTRAP_RESAMPLES = {BOOTSTRAP_RESAMPLES}

        TWIN_PAIR = {TWIN_PAIR!r}
        TWIN_OVERRIDE = {TWIN_OVERRIDE!r}

        ALWAYS_CONFIRMED_ARMS = {ALWAYS_CONFIRMED_ARMS!r}

        ARMS = {_embed(ARMS)}

        RECIPE_OVERRIDES = {_embed(RECIPE_OVERRIDES)}

        HELD_FIXED_OVERRIDES = {_embed(HELD_FIXED_OVERRIDES)}

        RUN_CONTROL_OVERRIDES = {_embed(RUN_CONTROL_OVERRIDES)}


        {_embed_source(arm_overrides)}

        # Split boundaries of data=gics_top10_110_2016, for the per-span
        # disclosure cells. Embargo-safe for label_t=5 (see that config's note).
        SPANS = {{
            "train": ("2016-01-04", "2023-12-31"),
            "val": ("2024-01-22", "2024-12-31"),
            "test": ("2025-01-22", "2025-12-31"),
        }}
        """
    ),
    md("## 4. Jobs And Manifest"),
    code(
        r"""
        import json
        from datetime import datetime

        RUN_TAG = RUN_TAG_OVERRIDE.strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = Path("/content/drive/MyDrive/MCI-GRU-Ablations/graph_specification_ablation") / RUN_TAG
        if not IN_COLAB:
            RUN_ROOT = REPO_DIR / "results" / "graph_specification_ablation" / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / "training" / RUN_STAGE
        RUN_ROOT.mkdir(parents=True, exist_ok=True)

        if SMOKE_MODE:
            num_models, num_epochs = 1, 2
        elif RUN_STAGE == "screen":
            num_models, num_epochs = SCREEN_NUM_MODELS, SCREEN_NUM_EPOCHS
        elif RUN_STAGE == "confirm":
            num_models, num_epochs = CONFIRM_NUM_MODELS, CONFIRM_NUM_EPOCHS
        else:
            raise ValueError(f"Unknown RUN_STAGE {RUN_STAGE!r}")

        if RUN_STAGE == "confirm":
            if CONFIRM_ARMS_OVERRIDE:
                confirm_arm_keys = list(CONFIRM_ARMS_OVERRIDE)
            else:
                source_tag = SCREEN_SOURCE_TAG.strip() or RUN_TAG
                promotion_path = RUN_ROOT.parent / source_tag / "graph_specification_ablation_promotion.json"
                if not promotion_path.exists():
                    raise FileNotFoundError(
                        f"No promotion record at {promotion_path}; run the screen stage first "
                        "or set CONFIRM_ARMS_OVERRIDE."
                    )
                confirm_arm_keys = json.loads(promotion_path.read_text(encoding="utf-8"))["PROMOTED_ARMS"]
            arms_to_run = [arm for arm in ARMS if arm["key"] in confirm_arm_keys]
            missing = sorted(set(confirm_arm_keys) - {arm["key"] for arm in arms_to_run})
            if missing:
                raise ValueError(f"Unknown arm keys in confirmation list: {missing}")
        else:
            arms_to_run = list(ARMS)

        jobs = []
        for arm in arms_to_run:
            name = f"graphspec_{RUN_STAGE}_{arm['key']}_seed{BASE_SEED}"
            jobs.append(
                {
                    "arm": arm["key"],
                    "label": arm["label"],
                    "hypothesis": arm["hypothesis"],
                    "name": name,
                    "overrides": [
                        *arm_overrides(arm, num_models, num_epochs),
                        f"experiment_name={name}",
                        f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                    ],
                }
            )

        manifest = {
            "issue": "ticket 166 (Wayfinder map 157)",
            "protocol": "issue-164 resolution",
            "run_tag": RUN_TAG,
            "run_stage": RUN_STAGE,
            "smoke_mode": SMOKE_MODE,
            "num_models": num_models,
            "num_epochs": num_epochs,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "base_seed": BASE_SEED,
            "twin_pair": list(TWIN_PAIR),
            "promotion_rule": (
                "A0 and A1 always confirmed at the frozen 20x100 recipe; additionally any arm "
                "whose test-span pooled daily IC 95% bootstrap CI does not overlap A0's on the "
                "same screen."
            ),
            "arms": {arm["key"]: arm for arm in arms_to_run},
            "jobs": jobs,
            "staged_files": {k: str(v) for k, v in staged.items()},
            "staged_sha256": staged_sha256,
        }
        manifest_path = RUN_ROOT / f"graph_specification_ablation_manifest_{RUN_STAGE}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print("Run root:", RUN_ROOT)
        print("Stage:", RUN_STAGE, "| models:", num_models, "| epochs:", num_epochs)
        print("Jobs:", len(jobs))
        for job in jobs:
            print("-", job["name"], "|", job["label"])
        print("Manifest:", manifest_path)
        """
    ),
    md("## 5. Train The Arms"),
    code(
        r"""
        GPU_UTIL_PATH = RUN_ROOT / f"gpu_util_{RUN_STAGE}.csv"
        GPU_UTIL_STOP_PATH = RUN_ROOT / f"gpu_util_{RUN_STAGE}.stop"
        RESUME_COMPLETED_TRAINING = True

        def start_gpu_sampler():
            if not IN_COLAB:
                return None
            if GPU_UTIL_STOP_PATH.exists():
                GPU_UTIL_STOP_PATH.unlink()
            monitor_script = REPO_DIR / "scripts/monitor_gpu_util.py"
            if not monitor_script.exists():
                raise FileNotFoundError(f"Missing GPU monitor: {monitor_script}")
            return subprocess.Popen(
                [
                    sys.executable,
                    str(monitor_script),
                    "--output",
                    str(GPU_UTIL_PATH),
                    "--interval",
                    "1",
                    "--stop-file",
                    str(GPU_UTIL_STOP_PATH),
                ],
                cwd=str(REPO_DIR),
            )

        def stop_gpu_sampler(proc):
            if proc is None:
                return
            GPU_UTIL_STOP_PATH.write_text("stop", encoding="utf-8")
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)

        def latest_run_dir(experiment_name: str) -> Path | None:
            base = TRAINING_OUTPUT_DIR / experiment_name
            if not base.exists():
                return None
            candidates = sorted(path for path in base.iterdir() if path.is_dir())
            return candidates[-1] if candidates else None

        def tail(path: Path, n: int = 60) -> str:
            if not path.exists():
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])

        def run_training(job: dict) -> Path:
            existing = latest_run_dir(job["name"])
            if (
                RESUME_COMPLETED_TRAINING
                and existing is not None
                and (existing / "evaluation_summary.json").is_file()
            ):
                print("Skipping completed training:", existing)
                return existing

            print("=" * 100)
            print("Training:", job["name"], "|", job["label"])
            cmd = [sys.executable, "-u", str(REPO_DIR / "run_experiment.py"), *job["overrides"]]
            logs_dir = RUN_ROOT / "logs" / job["name"]
            logs_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = logs_dir / "stdout.log"
            stderr_path = logs_dir / "stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w",
                encoding="utf-8",
            ) as stderr:
                proc = subprocess.run(cmd, cwd=str(REPO_DIR), stdout=stdout, stderr=stderr, text=True)
            print("Training return code:", proc.returncode)
            if proc.returncode != 0:
                print(tail(stdout_path))
                print(tail(stderr_path))
                raise RuntimeError(f"Training failed for {job['name']}")

            run_dir = latest_run_dir(job["name"])
            if run_dir is None or not (run_dir / "evaluation_summary.json").is_file():
                raise FileNotFoundError(f"Missing evaluation_summary.json for {job['name']}")
            return run_dir

        run_dirs: dict[str, Path] = {}
        gpu_sampler_proc = start_gpu_sampler()
        try:
            for job in jobs:
                run_dirs[job["arm"]] = run_training(job)
        finally:
            stop_gpu_sampler(gpu_sampler_proc)

        print("Completed runs:")
        for arm_key, run_dir in run_dirs.items():
            print("-", arm_key, "->", run_dir)
        """
    ),
    md("## 6. Collect The Arbiter Metrics"),
    code(
        r"""
        import pandas as pd

        def load_json(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8"))

        rows = []
        for job in jobs:
            run_dir = run_dirs[job["arm"]]
            metrics = load_json(run_dir / "evaluation_summary.json")["metrics"]
            training = load_json(run_dir / "training_summary.json")
            rows.append(
                {
                    "arm": job["arm"],
                    "label": job["label"],
                    "run_dir": str(run_dir),
                    # Arbiter: test-span pooled daily IC with CI (production
                    # block bootstrap over days, block size label_t).
                    "avg_ic": metrics.get("avg_ic"),
                    "avg_ic_ci_lower": metrics.get("avg_ic_ci_lower"),
                    "avg_ic_ci_upper": metrics.get("avg_ic_ci_upper"),
                    # April-composite inputs and context metrics.
                    "avg_spearman_corr": metrics.get("avg_spearman_corr"),
                    "avg_rank_ic": metrics.get("avg_rank_ic"),
                    "sharpe_top_20_newey_west": metrics.get("sharpe_top_20_newey_west"),
                    "return_top_20": metrics.get("return_top_20"),
                    "top_20_return_ci_lower": metrics.get("top_20_return_ci_lower"),
                    "models_trained": training.get("models_trained"),
                    "mean_best_val_ic": training.get("mean_best_val_ic"),
                    "hypothesis": job["hypothesis"],
                }
            )

        results_df = pd.DataFrame(rows)
        results_path = RUN_ROOT / f"graph_specification_ablation_results_{RUN_STAGE}.csv"
        results_df.to_csv(results_path, index=False)
        print("Results:", results_path)
        display(results_df.drop(columns=["hypothesis"]))
        """
    ),
    md("## 7. Disclosure: Twin Check And Per-Span Density / Isolation"),
    code(
        r"""
        import numpy as np
        import torch

        from mci_gru.data.pit import active_membership_mask, load_pit_intervals

        pit_intervals = load_pit_intervals(str(staged["pit_universe_csv"]))
        market_sessions = sorted(
            pd.read_csv(staged["market_csv"], usecols=["dt"])["dt"].astype(str).unique()
        )

        def span_dates(span: str) -> list[str]:
            start, end = SPANS[span]
            return [d for d in market_sessions if start <= d <= end]

        def adjacency_disclosure(edge_index, kdcode_list: list[str]) -> dict:
            out: dict[str, float] = {}
            n = len(kdcode_list)
            edges = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
            for span in SPANS:
                dates = span_dates(span)
                active = active_membership_mask(kdcode_list, dates, pit_intervals)
                densities, isolated_fractions = [], []
                for day_active in active:
                    n_active = int(day_active.sum())
                    if n_active < 2:
                        continue
                    kept = day_active[edges[0]] & day_active[edges[1]] if edges.size else np.zeros(0, bool)
                    degree = np.zeros(n, dtype=np.int64)
                    if kept.any():
                        np.add.at(degree, edges[0][kept], 1)
                        np.add.at(degree, edges[1][kept], 1)
                    densities.append(float(kept.sum()) / (n_active * (n_active - 1)))
                    isolated_fractions.append(
                        float(((degree == 0) & day_active).sum()) / n_active
                    )
                out[f"{span}_density"] = float(np.mean(densities)) if densities else float("nan")
                out[f"{span}_isolated_fraction"] = (
                    float(np.mean(isolated_fractions)) if isolated_fractions else float("nan")
                )
            return out

        def twin_edge_count(edge_index, kdcode_list: list[str]) -> int:
            if edge_index is None:
                return 0
            index_by_code = {code: idx for idx, code in enumerate(kdcode_list)}
            if TWIN_PAIR[0] not in index_by_code or TWIN_PAIR[1] not in index_by_code:
                return 0
            a, b = index_by_code[TWIN_PAIR[0]], index_by_code[TWIN_PAIR[1]]
            edges = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
            if edges.size == 0:
                return 0
            forward = ((edges[0] == a) & (edges[1] == b)).sum()
            reverse = ((edges[0] == b) & (edges[1] == a)).sum()
            return int(forward + reverse)

        disclosure_rows = []
        for job in jobs:
            run_dir = run_dirs[job["arm"]]
            kdcode_list = load_json(run_dir / "run_metadata.json")["kdcode_list"]
            graph_data = torch.load(run_dir / "graph_data.pt", weights_only=False)
            row = {"arm": job["arm"], "n_names": len(kdcode_list)}

            row["twin_edge_count"] = twin_edge_count(graph_data["edge_index"], kdcode_list)
            row.update(
                {f"corr_{k}": v for k, v in adjacency_disclosure(graph_data["edge_index"], kdcode_list).items()}
            )

            sector_index = graph_data.get("edge_index_sector")
            if sector_index is not None:
                row["twin_edge_count"] += twin_edge_count(sector_index, kdcode_list)
                row.update(
                    {f"sector_{k}": v for k, v in adjacency_disclosure(sector_index, kdcode_list).items()}
                )
            disclosure_rows.append(row)

        disclosure_df = pd.DataFrame(disclosure_rows)
        disclosure_path = RUN_ROOT / f"graph_specification_ablation_density_disclosure_{RUN_STAGE}.csv"
        disclosure_df.to_csv(disclosure_path, index=False)
        print("Density/isolation disclosure:", disclosure_path)
        display(disclosure_df)

        # The hygiene rule is a hard invariant of every arm, sector included.
        bad = disclosure_df[disclosure_df["twin_edge_count"] != 0]
        if not bad.empty:
            raise AssertionError(f"Twin edge present in arms: {bad['arm'].tolist()}")
        print("Twin check passed: twin_edge_count == 0 in every arm.")
        """
    ),
    md("## 8. Disclosure: Pooled Daily IC Per Year"),
    code(
        r"""
        # Disclosure-grade per-year breakdown of the arbiter. The pooled
        # test-span number and its CI come from the production evaluation
        # summary (section 6); this cell recomputes the daily IC series from
        # each run's averaged predictions and panel-derived 5-session forward
        # returns to break it out per calendar year, with a moving-block
        # bootstrap CI (block 5) per year.
        close = (
            pd.read_csv(staged["market_csv"], usecols=["dt", "kdcode", "close"])
            .assign(dt=lambda df: df["dt"].astype(str))
            .pivot_table(index="dt", columns="kdcode", values="close")
            .sort_index()
        )
        forward_return = close.shift(-5) / close - 1

        def daily_ic_series(predictions_dir: Path) -> pd.Series:
            out = {}
            for csv_path in sorted(predictions_dir.glob("*.csv")):
                scores = pd.read_csv(csv_path)
                date = str(scores["dt"].iloc[0])
                if date not in forward_return.index:
                    continue
                merged = scores.set_index("kdcode")["score"].to_frame().join(
                    forward_return.loc[date].rename("fwd"), how="inner"
                ).dropna()
                if len(merged) >= 5 and merged["score"].std() > 0 and merged["fwd"].std() > 0:
                    out[date] = float(merged["score"].corr(merged["fwd"]))
            return pd.Series(out).sort_index()

        def block_bootstrap_ci(values: np.ndarray, block: int = 5, resamples: int = 1000, seed: int = 42):
            if len(values) < block:
                return float("nan"), float("nan")
            rng = np.random.default_rng(seed)
            n_blocks = int(np.ceil(len(values) / block))
            starts = np.arange(len(values) - block + 1)
            means = []
            for _ in range(resamples):
                idx = rng.choice(starts, size=n_blocks, replace=True)
                sample = np.concatenate([values[i : i + block] for i in idx])[: len(values)]
                means.append(float(np.mean(sample)))
            return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

        per_year_rows = []
        for job in jobs:
            series = daily_ic_series(run_dirs[job["arm"]] / "averaged_predictions")
            for year, year_series in series.groupby(series.index.str[:4]):
                lo, hi = block_bootstrap_ci(year_series.to_numpy())
                per_year_rows.append(
                    {
                        "arm": job["arm"],
                        "year": year,
                        "n_days": len(year_series),
                        "pooled_daily_ic": float(year_series.mean()),
                        "ci_lower": lo,
                        "ci_upper": hi,
                    }
                )

        per_year_df = pd.DataFrame(per_year_rows)
        per_year_path = RUN_ROOT / f"graph_specification_ablation_per_year_ic_{RUN_STAGE}.csv"
        per_year_df.to_csv(per_year_path, index=False)
        print("Per-year pooled daily IC:", per_year_path)
        display(per_year_df)
        """
    ),
    md("## 9. Promotion Decision And April Composite"),
    code(
        r"""
        # April 2026's composite decision score, computed alongside for
        # reconciliation with the historical ablation reports. It is not the
        # arbiter; the pooled daily IC CI above is.
        DECISION_SCORE_WEIGHTS = {
            "avg_ic": 0.35,
            "avg_spearman_corr": 0.25,
            "sharpe_top_20_newey_west": 0.25,
            "return_top_20": 0.15,
        }

        scored_df = results_df.copy()
        score = pd.Series(0.0, index=scored_df.index)
        for col, weight in DECISION_SCORE_WEIGHTS.items():
            vals = pd.to_numeric(scored_df[col], errors="coerce")
            denom = vals.std(skipna=True)
            if pd.notna(denom) and denom > 0:
                score = score + weight * ((vals - vals.mean(skipna=True)) / denom).fillna(0.0)
        scored_df["decision_score"] = score

        promoted = list(ALWAYS_CONFIRMED_ARMS)
        separation_notes = {}
        if RUN_STAGE == "screen":
            a0 = scored_df[scored_df["arm"] == "A0_zeroed"].iloc[0]
            for _, row in scored_df.iterrows():
                if row["arm"] in ALWAYS_CONFIRMED_ARMS:
                    continue
                separated = bool(
                    row["avg_ic_ci_lower"] > a0["avg_ic_ci_upper"]
                    or row["avg_ic_ci_upper"] < a0["avg_ic_ci_lower"]
                )
                separation_notes[row["arm"]] = {
                    "ci": [row["avg_ic_ci_lower"], row["avg_ic_ci_upper"]],
                    "a0_ci": [a0["avg_ic_ci_lower"], a0["avg_ic_ci_upper"]],
                    "separated_from_A0": separated,
                }
                if separated:
                    promoted.append(row["arm"])

        PROMOTED_ARMS = promoted
        promotion = {
            "run_tag": RUN_TAG,
            "run_stage": RUN_STAGE,
            "smoke_mode": SMOKE_MODE,
            "rule": "non-overlapping 95% bootstrap CIs on test-span pooled daily IC vs A0_zeroed",
            "ALWAYS_CONFIRMED_ARMS": ALWAYS_CONFIRMED_ARMS,
            "PROMOTED_ARMS": PROMOTED_ARMS,
            "separation": separation_notes,
        }
        promotion_path = RUN_ROOT / "graph_specification_ablation_promotion.json"
        if RUN_STAGE == "screen" and not SMOKE_MODE:
            promotion_path.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
            print("Promotion record:", promotion_path)
        else:
            print("Promotion record not written (stage:", RUN_STAGE, "| smoke:", SMOKE_MODE, ")")

        summary_lines = [
            "# Graph-Specification Ablation Summary",
            "",
            f"- Run root: `{RUN_ROOT}`",
            f"- Stage: `{RUN_STAGE}` | smoke: `{SMOKE_MODE}`",
            f"- Budget: {num_models} models x {num_epochs} epochs, patience {EARLY_STOPPING_PATIENCE}",
            f"- Twin exclusion: `{TWIN_OVERRIDE}` (twin_edge_count == 0 verified per arm)",
            "- Arbiter: test-span pooled daily IC with 95% CI; the composite below is not the arbiter.",
            "",
            "## Arms",
            "",
            scored_df[
                [
                    "arm",
                    "avg_ic",
                    "avg_ic_ci_lower",
                    "avg_ic_ci_upper",
                    "decision_score",
                ]
            ].to_markdown(index=False),
            "",
            "## Promotion",
            "",
            f"- ALWAYS_CONFIRMED_ARMS: {ALWAYS_CONFIRMED_ARMS}",
            f"- PROMOTED_ARMS: {PROMOTED_ARMS}",
        ]
        summary_path = RUN_ROOT / f"graph_specification_ablation_summary_{RUN_STAGE}.md"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print("Summary:", summary_path)
        display(scored_df[["arm", "avg_ic", "avg_ic_ci_lower", "avg_ic_ci_upper", "decision_score"]])
        print("PROMOTED_ARMS:", PROMOTED_ARMS)
        """
    ),
]


def main() -> None:
    write_notebook(cells, OUT, indent=1)


if __name__ == "__main__":
    main()
