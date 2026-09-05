"""Generate the graph-specification ablation Colab notebook (tickets 166, 183).

Runnable form of the multi-year protocol pre-registered on ticket 181. A job is
(fold, arm, stage): six arms over the correlation graph's specification, six
folds -- four core test years 2022-2025 plus the 2021 stress fold and the 2025
bridge fold that run alongside and are never pooled into the primary -- and
three stages, a 1x2 mechanics smoke, a 3x20 mechanics-and-sanity screen, and the
frozen-recipe 20x100 confirm that every arm reaches unconditionally.

Arbiter: the mean paired daily Pearson IC difference against the zeroed control
on the arbiter's label, pooled over the core folds, with lag-4 HAC, a 5-session
circular block bootstrap, mandatory per-fold disclosure, and BHY across the five
arm-versus-control contrasts. The GOOG.OQ/GOOGL.OQ twin is excluded from every
arm's adjacency via ``graph.exclude_edge_pairs``, and ``drop_edge`` runs on a
forked random stream in every arm so the non-graph draws coincide.

The protocol data below is module-level on purpose: the notebook cells are
rendered from it, and ``tests/test_graph_specification_ablation_notebook.py``
imports it to compose every (fold, arm) pair through Hydra into a typed
``ExperimentConfig`` before any GPU minute is spent.
"""

from __future__ import annotations

import inspect
import json
import math
import textwrap
from pathlib import Path
from pprint import pformat

from nb_lib import build_notebook, code, colab_setup_cell, md

OUT = Path("notebooks/graph_specification_ablation_colab.ipynb")

# -- protocol constants (ticket-181 resolution; ticket-164 defaults carried) --

BASE_SEED_ORIGIN = 1729
BASE_SEED_FOLD_STRIDE = 1000
SCREEN_NUM_MODELS = 3
SCREEN_NUM_EPOCHS = 20
CONFIRM_NUM_MODELS = 20
CONFIRM_NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
BOOTSTRAP_RESAMPLES = 1000

TWIN_PAIR = ("GOOG.OQ", "GOOGL.OQ")
TWIN_OVERRIDE = '+graph.exclude_edge_pairs=[["GOOG.OQ","GOOGL.OQ"]]'

# Paired arbiter settings, resolved exactly as
# ``mci_gru.evaluation.experiment_summary.resolved_evaluation_kwargs`` resolves
# them for the frozen recipe: block = label_t, Newey-West lags = label_t - 1.
CONTROL_ARM = "A0_zeroed"
LABEL_T = 5
BLOCK_SIZE = 5
HAC_LAGS = 4
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 1729
CI_LEVEL = 0.95
POWER = 0.8
ALPHA = 0.05
TOP_K_VALUES = [10, 20, 50, 100]

# The line the outcome map is drawn against (ticket 181 section 1): a mean daily
# IC difference over the control worth changing the recipe for.
EFFECT_WORTH_ACTING_ON = 0.005
SIGN_CONSISTENT_FOLDS_REQUIRED = 3

# The six arms. Adjacency-rule keys ride the arm (preset or inline override) and
# nothing else touches them; A1 pins the incumbent explicitly rather than
# inheriting a mutable base default. ``use_multi_feature_edges`` is per-arm
# because A3s is the arm that turns it off.
ARMS: list[dict] = [
    {
        "key": "A0_zeroed",
        "label": "zeroed control",
        "overrides": ["+experiment=graph_zeroed"],
        "use_multi_feature_edges": True,
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
        "use_multi_feature_edges": True,
        "hypothesis": (
            "The incumbent, measured rather than assumed. Deployment-window "
            "edges are near-empty and 100% twin-or-intra-sector (Phase 0)."
        ),
    },
    {
        "key": "A2_thr05",
        "label": "recalibrated threshold 0.5",
        "overrides": ["+experiment=graph_thr05"],
        "use_multi_feature_edges": True,
        "hypothesis": (
            "Same selection rule at a sane calibration (~13% isolation vs ~76% "
            "at 0.8): separates calibration from rule."
        ),
    },
    {
        "key": "A3_topk20",
        "label": "top-K 20 (corr), static",
        "overrides": ["+experiment=graph_topk20_static"],
        "use_multi_feature_edges": True,
        "hypothesis": (
            "Best measured out-of-sample persistence, majority cross-sector; "
            "re-tests April 2026's anti-top-K result under the corrected PIT "
            "protocol with a populated rank-4 edge tensor."
        ),
    },
    {
        "key": "A3s_topk20_scalar",
        "label": "top-K 20 (corr), scalar edge weight",
        "overrides": ["+experiment=graph_topk20_static"],
        "use_multi_feature_edges": False,
        "hypothesis": (
            "The neighbours-versus-channels disambiguation that could not "
            "trigger in 2025: the same top-K adjacency carrying one scalar edge "
            "weight instead of four channels. Disclosed asymmetry: edge width 1 "
            "means a smaller GAT edge projection than every other arm's."
        ),
    },
    {
        "key": "A4_sector_only",
        "label": "sector relation only",
        "overrides": ["+experiment=graph_sector_only"],
        "use_multi_feature_edges": True,
        "hypothesis": (
            "Exact, estimation-free version of the structure the shipped graph "
            "approximates. Extra dual-GAT parameters: specification-level, not "
            "parameter-matched."
        ),
    },
]

COMPARISON_ARMS = [arm["key"] for arm in ARMS if arm["key"] != CONTROL_ARM]
#: BHY multiplicity, fixed before any number is seen (ticket 181 section 6).
BHY_CONTRAST_COUNT = len(COMPARISON_ARMS)

# The folds. Rolling five-year train, validation Y-1, test Y, with the
# post-issue-115 22-day gaps; the first fold's train starts on the universe's
# opening session rather than 1 January. ``index`` drives the base seed and is
# data rather than list position so the seed of any fold is auditable from the
# table alone.
FOLDS: list[dict] = [
    {
        "key": "F2022",
        "index": 0,
        "pool": "core",
        "test_year": 2022,
        "train_start": "2016-01-04",
        "train_end": "2020-12-31",
        "val_start": "2021-01-22",
        "val_end": "2021-12-31",
        "test_start": "2022-01-22",
        "test_end": "2022-12-31",
        "note": "First core fold; train opens on the universe's first session.",
    },
    {
        "key": "F2023",
        "index": 1,
        "pool": "core",
        "test_year": 2023,
        "train_start": "2017-01-01",
        "train_end": "2021-12-31",
        "val_start": "2022-01-22",
        "val_end": "2022-12-31",
        "test_start": "2023-01-22",
        "test_end": "2023-12-31",
        "note": "Core fold.",
    },
    {
        "key": "F2024",
        "index": 2,
        "pool": "core",
        "test_year": 2024,
        "train_start": "2018-01-01",
        "train_end": "2022-12-31",
        "val_start": "2023-01-22",
        "val_end": "2023-12-31",
        "test_start": "2024-01-22",
        "test_end": "2024-12-31",
        "note": "Core fold.",
    },
    {
        "key": "F2025",
        "index": 3,
        "pool": "core",
        "test_year": 2025,
        "train_start": "2019-01-01",
        "train_end": "2023-12-31",
        "val_start": "2024-01-22",
        "val_end": "2024-12-31",
        "test_start": "2025-01-22",
        "test_end": "2025-12-31",
        "note": "Core fold; shares its test days with the bridge fold.",
    },
    {
        "key": "F2021_stress",
        "index": 4,
        "pool": "stress",
        "test_year": 2021,
        "train_start": "2016-01-04",
        "train_end": "2019-12-31",
        "val_start": "2020-01-22",
        "val_end": "2020-12-31",
        "test_start": "2021-01-22",
        "test_end": "2021-12-31",
        "note": (
            "Alongside: four training years and the COVID year as validation. "
            "A different train length and regime, so never pooled."
        ),
    },
    {
        "key": "F2025_bridge",
        "index": 5,
        "pool": "bridge",
        "test_year": 2025,
        "train_start": "2016-01-04",
        "train_end": "2023-12-31",
        "val_start": "2024-01-22",
        "val_end": "2024-12-31",
        "test_start": "2025-01-22",
        "test_end": "2025-12-31",
        "note": (
            "Alongside: configs/data/gics_top10_110_2016.yaml verbatim, the "
            "ticket-167 expanding window. The one check that this harness "
            "reproduces that fold. Its test days are the core 2025 fold's, so "
            "pooling it would double-count them."
        ),
    },
]

CORE_FOLD_KEYS = [fold["key"] for fold in FOLDS if fold["pool"] == "core"]
ALONGSIDE_FOLD_KEYS = [fold["key"] for fold in FOLDS if fold["pool"] != "core"]

# Frozen-recipe semantics (docs/DEFAULT_EXPERIMENT_RECIPE.md), minus the graph
# block, the training budget, the split boundaries, and the seed, which are
# handled separately.
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
# top_k / top_k_metric / zero_edges / use_sector_relation, which are arm keys,
# and use_multi_feature_edges, which became per-arm when A3s was admitted.
HELD_FIXED_OVERRIDES = [
    "graph.corr_lookback_days=252",
    "graph.update_frequency_months=0",
    "graph.append_snapshot_age_days=false",
    "graph.use_lead_lag_features=false",
    "graph.drop_edge_p=0.1",
    "graph.isolate_edge_dropout_rng=true",
]

# Tracking off: the Drive export is the record for these runs.
RUN_CONTROL_OVERRIDES = [
    "tracking.enabled=false",
    "tracking.log_artifacts=false",
    "tracking.log_checkpoints=false",
    "tracking.log_predictions=false",
]


def fold_seed(fold: dict) -> int:
    """One base seed per fold, shared by every arm in that fold.

    Member *i* of every arm therefore shares its initialisation with member *i*
    of the control, which is what makes the per-model secondary paired.
    """
    return BASE_SEED_ORIGIN + BASE_SEED_FOLD_STRIDE * fold["index"]


def fold_overrides(fold: dict) -> list[str]:
    """Per-fold split boundaries as Hydra ``data.*`` overrides.

    Never the walkforward path: it varies the training-set size and resolves
    its own windows, which is not the fold shape this protocol fixed.
    """
    return [
        f"data.train_start={fold['train_start']}",
        f"data.train_end={fold['train_end']}",
        f"data.val_start={fold['val_start']}",
        f"data.val_end={fold['val_end']}",
        f"data.test_start={fold['test_start']}",
        f"data.test_end={fold['test_end']}",
    ]


def held_fixed_overrides(arm: dict) -> list[str]:
    """Held-fixed graph keys plus this arm's edge-width pin.

    The ticket-166 harness pinned ``use_multi_feature_edges=true`` for every
    arm. A3s needs it false, so the pin rides the arm rather than the block.
    """
    multi = bool(arm.get("use_multi_feature_edges", True))
    return [*HELD_FIXED_OVERRIDES, f"graph.use_multi_feature_edges={str(multi).lower()}"]


def arm_overrides(arm: dict, fold: dict, num_models: int, num_epochs: int) -> list[str]:
    """Full Hydra override list for one (arm, fold) job at one training budget."""
    return [
        *arm["overrides"],
        *RECIPE_OVERRIDES,
        *held_fixed_overrides(arm),
        *fold_overrides(fold),
        TWIN_OVERRIDE,
        f"training.num_models={num_models}",
        f"training.num_epochs={num_epochs}",
        f"training.early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        f"evaluation.bootstrap_resamples={BOOTSTRAP_RESAMPLES}",
        *RUN_CONTROL_OVERRIDES,
        f"seed={fold_seed(fold)}",
    ]


def job_name(stage: str, fold: dict, arm: dict) -> str:
    """Resume key: one name per (fold, arm, stage) triple."""
    return f"graphspec_{stage}_{fold['key']}_{arm['key']}_seed{fold_seed(fold)}"


def confirm_arm_keys() -> list[str]:
    """Every arm confirms, in every fold (ticket 181 section 8)."""
    return [arm["key"] for arm in ARMS]


def promotion_rule_text(arms: list, stage: str, num_models: int, num_epochs: int) -> str:
    """The confirm rule, rendered from the arm list and the running stage.

    The ticket-166 harness carried this as prose, and it went stale the moment
    confirm was widened past the promoted pair (ticket 167's disclosed
    deviation). Rendering it from the data that decides it is what stops that
    recurring: add or drop an arm and the string follows.
    """
    keys = ", ".join(arm["key"] for arm in arms)
    return (
        f"no promotion gate (ticket 181 s8): all {len(arms)} arms ({keys}) confirm "
        f"unconditionally at {CONFIRM_NUM_MODELS}x{CONFIRM_NUM_EPOCHS} in every fold; "
        f"the running stage {stage!r} is {num_models}x{num_epochs} and promotes nothing."
    )


def outcome_branch(
    bhy_p: float,
    mean_delta: float,
    sign_consistent_folds: int,
    realised_mde: float,
    effect_worth_acting_on: float = EFFECT_WORTH_ACTING_ON,
    folds_required: int = SIGN_CONSISTENT_FOLDS_REQUIRED,
    alpha: float = ALPHA,
) -> str:
    """Ticket 181 section 9's three-branch outcome map, applied to one arm.

    Branches 2 and 3 are separated by the realised MDE alone. The sign of the
    point estimate does not route them: ticket 181 rejected reading a negative
    estimate as branch 2, so an arm that fails to clear is placed by what the
    run could have detected rather than by which way it happened to land.
    """
    cleared = (
        bhy_p is not None
        and math.isfinite(bhy_p)
        and bhy_p < alpha
        and mean_delta > 0.0
        and sign_consistent_folds >= folds_required
    )
    if cleared:
        return "cleared"
    if realised_mde <= effect_worth_acting_on:
        return "does_not_earn_its_edges"
    return "undecidable_at_this_universe_and_horizon"


def fold_spans(fold: dict) -> dict:
    """Split boundaries of one fold, for the per-span disclosure cells."""
    return {
        "train": (fold["train_start"], fold["train_end"]),
        "val": (fold["val_start"], fold["val_end"]),
        "test": (fold["test_start"], fold["test_end"]),
    }


def _embed(obj) -> str:
    """Render *obj* for interpolation into a 12-space-indented cell template.

    Continuation lines get the template's indent so ``textwrap.dedent`` inside
    ``code()`` strips a uniform prefix; ``lstrip`` leaves the first line to sit
    where the template places it.
    """
    return textwrap.indent(pformat(obj, width=88, sort_dicts=False), " " * 12).lstrip()


def _embed_source(func) -> str:
    return textwrap.indent(inspect.getsource(func), " " * 12).lstrip()


def build_cells() -> list[dict]:
    return [
        md(
            """
            # Graph-Specification Ablation (Wayfinder map 157, tickets 166 and 183)

            Runnable form of the multi-year protocol pre-registered on ticket
            181. A job is **(fold, arm, stage)**.

            - **Six arms:** A0 zeroed control, A1 shipped thr 0.8, A2 thr 0.5,
              A3 top-K 20, A3s top-K 20 with a single scalar edge weight, A4
              sector-only. All six confirm unconditionally at 20 x 100 in every
              fold: **there is no promotion gate.**
            - **Six folds:** rolling five-year train, validation Y-1, test Y,
              22-day gaps. Core pool 2022-2025; the 2021 stress fold and the
              2025 bridge fold run alongside, are disclosed per fold, and are
              **never pooled** into the primary.
            - **Arbiter:** the mean paired daily Pearson IC difference against
              A0 on the arbiter's label, pooled over the core folds, lag-4 HAC,
              5-session circular block bootstrap, per-fold disclosure mandatory,
              BHY across the five arm-versus-control contrasts.
            - **Secondaries:** paired Spearman, seed-paired per-model IC, median
              with a sign test. They qualify a verdict and never promote.
            - **Disclosure only:** per-span density and isolation, graph
              staleness, the ensemble-scale audit and rank-averaged ensemble,
              top-K Sharpe with block-bootstrap intervals, paired basket returns
              by K, and April 2026's composite for continuity.
            - **Hygiene:** the GOOG.OQ/GOOGL.OQ same-company twin is excluded
              from every constructed adjacency in every arm, sector included,
              and `graph.isolate_edge_dropout_rng=true` forks the random stream
              around edge dropout so the non-graph draws coincide across arms.

            Run `SMOKE_MODE = True` first (1 model x 2 epochs: mechanics only,
            never evidence), then `RUN_STAGE = "screen"` for the
            mechanics-and-sanity pass, then `RUN_STAGE = "confirm"`. Resume is
            keyed on (fold, arm, stage) and the manifest, not this notebook's
            display, is the record.
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
            # paths, so staging them is all the wiring the runs need. The panel
            # opens 2015-01-02, exactly 252 sessions before the 2016-01-04
            # universe opening, which is what the first fold's lookback needs.
            """
        ),
        md("## 3. Protocol: Arms, Folds, Recipe Semantics, Budgets"),
        code(
            f"""
            RUN_STAGE = "screen"  # screen | confirm
            SMOKE_MODE = False  # True = 1 model x 2 epochs: a mechanics smoke, never evidence
            RUN_TAG_OVERRIDE = ""
            # Empty = every fold / every arm. Narrow them to split a stage over
            # several Colab sessions; resume is keyed on (fold, arm, stage).
            FOLDS_TO_RUN: list[str] = []
            ARMS_TO_RUN: list[str] = []

            import math

            BASE_SEED_ORIGIN = {BASE_SEED_ORIGIN}
            BASE_SEED_FOLD_STRIDE = {BASE_SEED_FOLD_STRIDE}
            SCREEN_NUM_MODELS = {SCREEN_NUM_MODELS}
            SCREEN_NUM_EPOCHS = {SCREEN_NUM_EPOCHS}
            CONFIRM_NUM_MODELS = {CONFIRM_NUM_MODELS}
            CONFIRM_NUM_EPOCHS = {CONFIRM_NUM_EPOCHS}
            EARLY_STOPPING_PATIENCE = {EARLY_STOPPING_PATIENCE}
            BOOTSTRAP_RESAMPLES = {BOOTSTRAP_RESAMPLES}

            CONTROL_ARM = {CONTROL_ARM!r}
            COMPARISON_ARMS = {COMPARISON_ARMS!r}
            BHY_CONTRAST_COUNT = {BHY_CONTRAST_COUNT}
            LABEL_T = {LABEL_T}
            BLOCK_SIZE = {BLOCK_SIZE}
            HAC_LAGS = {HAC_LAGS}
            N_RESAMPLES = {N_RESAMPLES}
            BOOTSTRAP_SEED = {BOOTSTRAP_SEED}
            CI_LEVEL = {CI_LEVEL}
            POWER = {POWER}
            ALPHA = {ALPHA}
            TOP_K_VALUES = {TOP_K_VALUES!r}
            EFFECT_WORTH_ACTING_ON = {EFFECT_WORTH_ACTING_ON}
            SIGN_CONSISTENT_FOLDS_REQUIRED = {SIGN_CONSISTENT_FOLDS_REQUIRED}
            STAGES_FOR_ANALYSIS = ["confirm", "screen"]

            TWIN_PAIR = {TWIN_PAIR!r}
            TWIN_OVERRIDE = {TWIN_OVERRIDE!r}

            ARMS = {_embed(ARMS)}

            FOLDS = {_embed(FOLDS)}

            CORE_FOLD_KEYS = {CORE_FOLD_KEYS!r}
            ALONGSIDE_FOLD_KEYS = {ALONGSIDE_FOLD_KEYS!r}

            RECIPE_OVERRIDES = {_embed(RECIPE_OVERRIDES)}

            HELD_FIXED_OVERRIDES = {_embed(HELD_FIXED_OVERRIDES)}

            RUN_CONTROL_OVERRIDES = {_embed(RUN_CONTROL_OVERRIDES)}


            {_embed_source(fold_seed)}

            {_embed_source(fold_overrides)}

            {_embed_source(held_fixed_overrides)}

            {_embed_source(arm_overrides)}

            {_embed_source(job_name)}

            {_embed_source(confirm_arm_keys)}

            {_embed_source(promotion_rule_text)}

            {_embed_source(outcome_branch)}

            {_embed_source(fold_spans)}
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
                BOOTSTRAP_RESAMPLES = 25
            elif RUN_STAGE == "screen":
                num_models, num_epochs = SCREEN_NUM_MODELS, SCREEN_NUM_EPOCHS
            elif RUN_STAGE == "confirm":
                num_models, num_epochs = CONFIRM_NUM_MODELS, CONFIRM_NUM_EPOCHS
            else:
                raise ValueError(f"Unknown RUN_STAGE {RUN_STAGE!r}")

            # Every arm confirms in every fold. The only narrowing is the manual
            # one above, which exists to split a stage over Colab sessions.
            unknown_folds = sorted(set(FOLDS_TO_RUN) - {fold["key"] for fold in FOLDS})
            unknown_arms = sorted(set(ARMS_TO_RUN) - {arm["key"] for arm in ARMS})
            if unknown_folds or unknown_arms:
                raise ValueError(f"Unknown fold keys {unknown_folds}, arm keys {unknown_arms}")
            folds_to_run = [fold for fold in FOLDS if not FOLDS_TO_RUN or fold["key"] in FOLDS_TO_RUN]
            arms_to_run = [arm for arm in ARMS if not ARMS_TO_RUN or arm["key"] in ARMS_TO_RUN]

            jobs = []
            for fold in folds_to_run:
                for arm in arms_to_run:
                    name = job_name(RUN_STAGE, fold, arm)
                    jobs.append(
                        {
                            "fold": fold["key"],
                            "pool": fold["pool"],
                            "test_year": fold["test_year"],
                            "arm": arm["key"],
                            "stage": RUN_STAGE,
                            "seed": fold_seed(fold),
                            "label": arm["label"],
                            "hypothesis": arm["hypothesis"],
                            "name": name,
                            "overrides": [
                                *arm_overrides(arm, fold, num_models, num_epochs),
                                f"experiment_name={name}",
                                f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                            ],
                        }
                    )

            manifest = {
                "issue": "ticket 183 (Wayfinder map 157)",
                "protocol": "ticket-181 resolution (multi-year); ticket-164 defaults carried",
                "run_tag": RUN_TAG,
                "run_stage": RUN_STAGE,
                "smoke_mode": SMOKE_MODE,
                "num_models": num_models,
                "num_epochs": num_epochs,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "base_seed_origin": BASE_SEED_ORIGIN,
                "base_seed_fold_stride": BASE_SEED_FOLD_STRIDE,
                "fold_seeds": {fold["key"]: fold_seed(fold) for fold in FOLDS},
                "core_fold_keys": CORE_FOLD_KEYS,
                "alongside_fold_keys": ALONGSIDE_FOLD_KEYS,
                "twin_pair": list(TWIN_PAIR),
                "control_arm": CONTROL_ARM,
                "bhy_contrast_count": BHY_CONTRAST_COUNT,
                "effect_worth_acting_on": EFFECT_WORTH_ACTING_ON,
                # Rendered from the arm list and the stage, never written as prose.
                # ARMS, never arms_to_run: ARMS_TO_RUN narrows a session, not the
                # protocol, and a two-arm session must not record "all 2 arms
                # confirm unconditionally". That is the staleness this string was
                # made data-driven to end.
                "promotion_rule": promotion_rule_text(ARMS, RUN_STAGE, num_models, num_epochs),
                "confirm_arm_keys": confirm_arm_keys(),
                "arms": {arm["key"]: arm for arm in arms_to_run},
                "folds": {fold["key"]: fold for fold in folds_to_run},
                "jobs": jobs,
                "completed_jobs": [],
                "staged_files": {k: str(v) for k, v in staged.items()},
                "staged_sha256": staged_sha256,
            }
            MANIFEST_PATH = RUN_ROOT / f"graph_specification_ablation_manifest_{RUN_STAGE}.json"


            def write_manifest() -> None:
                MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


            write_manifest()

            print("Run root:", RUN_ROOT)
            print("Stage:", RUN_STAGE, "| models:", num_models, "| epochs:", num_epochs)
            print("Folds:", [fold["key"] for fold in folds_to_run])
            print("Jobs:", len(jobs), "=", len(folds_to_run), "folds x", len(arms_to_run), "arms")
            print("Rule:", manifest["promotion_rule"])
            print("Manifest:", MANIFEST_PATH)
            """
        ),
        md("## 5. Train The Jobs"),
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
                # Resume key = the job name, which carries (fold, arm, seed), under
                # TRAINING_OUTPUT_DIR, which carries the stage. A partial run
                # directory without a summary is not a completed job.
                existing = latest_run_dir(job["name"])
                if (
                    RESUME_COMPLETED_TRAINING
                    and existing is not None
                    and (existing / "evaluation_summary.json").is_file()
                ):
                    print("Skipping completed training:", existing)
                    return existing

                print("=" * 100)
                print("Training:", job["name"], "|", job["fold"], "|", job["label"])
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

            # (fold, arm) -> run directory. The manifest is rewritten as each job
            # lands, so a session that dies mid-stage leaves a truthful record.
            run_dirs: dict[tuple[str, str], Path] = {}
            gpu_sampler_proc = start_gpu_sampler()
            try:
                for job in jobs:
                    run_dir = run_training(job)
                    run_dirs[(job["fold"], job["arm"])] = run_dir
                    manifest["completed_jobs"].append(
                        {
                            "fold": job["fold"],
                            "arm": job["arm"],
                            "stage": job["stage"],
                            "name": job["name"],
                            "run_dir": str(run_dir),
                        }
                    )
                    write_manifest()
            finally:
                stop_gpu_sampler(gpu_sampler_proc)

            print("Completed runs:", len(run_dirs), "of", len(jobs))
            for (fold_key, arm_key), run_dir in run_dirs.items():
                print("-", fold_key, arm_key, "->", run_dir)
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
                run_dir = run_dirs[(job["fold"], job["arm"])]
                metrics = load_json(run_dir / "evaluation_summary.json")["metrics"]
                training = load_json(run_dir / "training_summary.json")
                rows.append(
                    {
                        "fold": job["fold"],
                        "pool": job["pool"],
                        "test_year": job["test_year"],
                        "arm": job["arm"],
                        "label": job["label"],
                        "seed": job["seed"],
                        "run_dir": str(run_dir),
                        # Production per-run pooled daily IC with CI. The arbiter
                        # is the paired difference in section 9; this is context.
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
        md("## 7. Disclosure: Twin Check, Per-Span Density / Isolation, Graph Staleness"),
        code(
            r"""
            import numpy as np
            import torch

            from mci_gru.data.pit import active_membership_mask, load_pit_intervals
            from mci_gru.graph.builder import GraphBuilder

            pit_intervals = load_pit_intervals(str(staged["pit_universe_csv"]))
            market_frame = pd.read_csv(staged["market_csv"], usecols=["dt", "kdcode", "close"])
            market_frame["dt"] = market_frame["dt"].astype(str)
            market_sessions = sorted(market_frame["dt"].unique())

            def span_dates(fold: dict, span: str) -> list[str]:
                start, end = fold_spans(fold)[span]
                return [d for d in market_sessions if start <= d <= end]

            def adjacency_disclosure(edge_index, kdcode_list: list[str], fold: dict) -> dict:
                out: dict[str, float] = {}
                n = len(kdcode_list)
                edges = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
                for span in fold_spans(fold):
                    dates = span_dates(fold, span)
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

            def edge_pair_set(edge_index) -> set:
                edges = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
                if edges.size == 0:
                    return set()
                return {(int(u), int(v)) for u, v in zip(edges[0], edges[1])}

            def staleness_jaccard(frozen_edge_index, kdcode_list: list[str], arm: dict, fold: dict) -> float:
                # Jaccard of the frozen graph against one realised on the test
                # year. The graph is built from the fold's training window and
                # frozen, so by test time it is one to two years old; this says
                # how much of it the test year would still have selected. NaN
                # for arms whose correlation adjacency is empty by construction
                # (A0, A4).
                frozen = edge_pair_set(frozen_edge_index)
                if not frozen:
                    return float("nan")
                overrides = {key.split("=", 1)[0]: key.split("=", 1)[1] for key in arm["overrides"] if "=" in key}
                builder = GraphBuilder(
                    judge_value=float(overrides.get("graph.judge_value", 0.8)),
                    update_frequency_months=0,
                    corr_lookback_days=252,
                    top_k=int(overrides.get("graph.top_k", 20 if "topk20" in arm["key"] else 0)),
                    top_k_metric="corr",
                    use_multi_feature_edges=bool(arm.get("use_multi_feature_edges", True)),
                    exclude_edge_pairs=[list(TWIN_PAIR)],
                )
                corr = builder.compute_correlation_matrix(market_frame, kdcode_list, fold["test_end"])
                realised_index, _ = builder.build_edges(corr, kdcode_list, show_progress=False)
                realised = edge_pair_set(realised_index)
                if not realised:
                    return float("nan")
                return float(len(frozen & realised) / len(frozen | realised))

            disclosure_rows = []
            for job in jobs:
                fold = next(f for f in FOLDS if f["key"] == job["fold"])
                arm = next(a for a in ARMS if a["key"] == job["arm"])
                run_dir = run_dirs[(job["fold"], job["arm"])]
                kdcode_list = load_json(run_dir / "run_metadata.json")["kdcode_list"]
                graph_data = torch.load(run_dir / "graph_data.pt", weights_only=False)
                row = {"fold": job["fold"], "arm": job["arm"], "n_names": len(kdcode_list)}

                row["twin_edge_count"] = twin_edge_count(graph_data["edge_index"], kdcode_list)
                row.update(
                    {
                        f"corr_{k}": v
                        for k, v in adjacency_disclosure(graph_data["edge_index"], kdcode_list, fold).items()
                    }
                )
                row["staleness_jaccard"] = staleness_jaccard(
                    graph_data["edge_index"], kdcode_list, arm, fold
                )

                sector_index = graph_data.get("edge_index_sector")
                if sector_index is not None:
                    row["twin_edge_count"] += twin_edge_count(sector_index, kdcode_list)
                    row.update(
                        {
                            f"sector_{k}": v
                            for k, v in adjacency_disclosure(sector_index, kdcode_list, fold).items()
                        }
                    )
                disclosure_rows.append(row)

            disclosure_df = pd.DataFrame(disclosure_rows)
            disclosure_path = RUN_ROOT / f"graph_specification_ablation_density_disclosure_{RUN_STAGE}.csv"
            disclosure_df.to_csv(disclosure_path, index=False)
            print("Density / isolation / staleness disclosure:", disclosure_path)
            display(disclosure_df)

            # The hygiene rule is a hard invariant of every arm, sector included.
            bad = disclosure_df[disclosure_df["twin_edge_count"] != 0]
            if not bad.empty:
                raise AssertionError(f"Twin edge present in: {bad[['fold', 'arm']].to_dict('records')}")
            print("Twin check passed: twin_edge_count == 0 in every fold and arm.")
            """
        ),
        md("## 8. Disclosure: Pooled Daily IC Per Year"),
        code(
            r"""
            # Disclosure-grade per-year breakdown. Section 9's paired arbiter uses
            # the production label `close[t+5]/close[t+1] - 1`; this cell keeps the
            # ticket-166 `close[t+5]/close[t] - 1` variant for continuity with the
            # earlier report, and the two sit 0.003-0.006 apart for that reason
            # alone (ticket 179, section 3).
            close = (
                market_frame.pivot_table(index="dt", columns="kdcode", values="close").sort_index()
            )
            forward_return = close.shift(-LABEL_T) / close - 1

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
                series = daily_ic_series(run_dirs[(job["fold"], job["arm"])] / "averaged_predictions")
                for year, year_series in series.groupby(series.index.str[:4]):
                    lo, hi = block_bootstrap_ci(year_series.to_numpy())
                    per_year_rows.append(
                        {
                            "fold": job["fold"],
                            "pool": job["pool"],
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
        md("## 9. Paired Inference Against The Control"),
        code(
            r"""
            # The arbiter (ticket 181 sections 5, 6, 7 and 9), on the merged
            # mci_gru.evaluation.paired_inference module. Per fold and pooled over
            # the core folds only; the alongside folds are reported per fold and
            # never pooled.
            from mci_gru.evaluation.paired_inference import (
                align_daily_series,
                bhy_adjusted_p_values,
                minimum_detectable_effect,
                paired_daily_differences,
                paired_mean_inference,
            )
            from mci_gru.evaluation.prediction_report import (
                load_prediction_files,
                realized_returns_from_market_data,
            )
            from mci_gru.evaluation.statistics import cross_sectional_ic
            from scipy import stats

            ALL_ARMS = [CONTROL_ARM, *COMPARISON_ARMS]

            # The mechanics smoke proves wiring and supports no claim (ticket
            # 181 section 8), so the paired arbiter refuses to run on one. This
            # is what keeps the smoke from becoming an input to anything.
            if SMOKE_MODE or RUN_STAGE not in STAGES_FOR_ANALYSIS:
                raise RuntimeError(
                    f"paired inference runs only on {STAGES_FOR_ANALYSIS} and never on a smoke "
                    f"run; got RUN_STAGE={RUN_STAGE!r}, SMOKE_MODE={SMOKE_MODE}"
                )

            # The alongside folds are reported per fold and never pooled into the
            # primary. Asserted rather than assumed: the 2025 bridge fold shares
            # its test days with the core 2025 fold, so pooling it would
            # double-count them.
            POOLED_FOLD_KEYS = list(CORE_FOLD_KEYS)
            if set(POOLED_FOLD_KEYS) & set(ALONGSIDE_FOLD_KEYS):
                raise AssertionError("alongside folds must never enter the pooled primary")

            # Arbiter label: close[t + LABEL_T] / close[t + 1] - 1, identical to
            # mci_gru.data.preprocessing.compute_labels and to what the production
            # evaluation received as `true_returns`.
            realized = realized_returns_from_market_data(market_frame, label_t=LABEL_T)
            LABEL_ARBITER = realized.pivot_table(
                index="dt", columns="kdcode", values="realized_return"
            ).sort_index()

            def score_pivot(predictions_dir: Path) -> pd.DataFrame:
                scores = load_prediction_files(predictions_dir)
                scores["dt"] = scores["dt"].astype(str)
                return scores.pivot_table(index="dt", columns="kdcode", values="score").sort_index()

            def dated_ic(scores: pd.DataFrame, label: pd.DataFrame, method: str) -> pd.Series:
                # Same per-day semantics as mci_gru.evaluation.statistics.daily_ic_series,
                # but dated so the arms can be paired on the calendar.
                dates = [d for d in scores.index if d in label.index]
                cols = scores.columns.intersection(label.columns)
                pred = scores.loc[dates, cols].to_numpy(dtype=float)
                rets = label.loc[dates, cols].to_numpy(dtype=float)
                out = {}
                for i, date in enumerate(dates):
                    mask = np.isfinite(pred[i]) & np.isfinite(rets[i])
                    if int(mask.sum()) < 2:
                        continue
                    value = cross_sectional_ic(pred[i][mask], rets[i][mask], method=method)
                    if np.isfinite(value):
                        out[date] = float(value)
                return pd.Series(out, dtype=float).sort_index()

            IC_METHODS = ["pearson", "spearman"]
            FOLD_KEYS_PRESENT = list(dict.fromkeys(fold_key for fold_key, _ in run_dirs))
            IC = {
                method: {
                    (fold_key, arm_key): dated_ic(score_pivot(run_dir / "averaged_predictions"), LABEL_ARBITER, method)
                    for (fold_key, arm_key), run_dir in run_dirs.items()
                }
                for method in IC_METHODS
            }

            def paired_rows(fold_key: str, method: str) -> list[dict]:
                present = [arm for arm in ALL_ARMS if (fold_key, arm) in IC[method]]
                if CONTROL_ARM not in present or len(present) < 2:
                    return []
                aligned = align_daily_series({arm: IC[method][(fold_key, arm)] for arm in present})
                deltas = paired_daily_differences(aligned, control=CONTROL_ARM)
                rows = []
                for arm in [a for a in COMPARISON_ARMS if a in present]:
                    result = paired_mean_inference(
                        deltas[arm],
                        arm=arm,
                        control=CONTROL_ARM,
                        label_horizon=LABEL_T,
                        block_size=BLOCK_SIZE,
                        hac_lags=HAC_LAGS,
                        n_resamples=N_RESAMPLES,
                        seed=BOOTSTRAP_SEED,
                        ci_level=CI_LEVEL,
                    )
                    rows.append({"fold": fold_key, "method": method, **result.__dict__})
                return rows

            # Per fold, every fold, core and alongside alike. Mandatory disclosure:
            # a pooled result carried by one year has to be visible as such.
            paired_per_fold = pd.DataFrame(
                [row for fold_key in FOLD_KEYS_PRESENT for method in IC_METHODS for row in paired_rows(fold_key, method)]
            )
            paired_per_fold["pool"] = paired_per_fold["fold"].map(
                {fold["key"]: fold["pool"] for fold in FOLDS}
            )
            paired_per_fold.to_csv(RUN_ROOT / f"paired_per_fold_{RUN_STAGE}.csv", index=False)

            def fold_deltas(arm: str, method: str, fold_key: str):
                # One fold's paired daily differences against the control.
                if (fold_key, arm) not in IC[method] or (fold_key, CONTROL_ARM) not in IC[method]:
                    return None
                aligned = align_daily_series(
                    {
                        CONTROL_ARM: IC[method][(fold_key, CONTROL_ARM)],
                        arm: IC[method][(fold_key, arm)],
                    }
                )
                return paired_daily_differences(aligned, control=CONTROL_ARM)[arm]

            def pooled_deltas(arm: str, method: str):
                # Pools POOLED_FOLD_KEYS and nothing else -- the guard above is
                # what makes that list the core folds.
                chunks = []
                for fold_key in POOLED_FOLD_KEYS:
                    chunk = fold_deltas(arm, method, fold_key)
                    if chunk is not None:
                        chunks.append(chunk)
                if not chunks:
                    return None
                pooled = pd.concat(chunks)
                pooled.attrs["n_core_folds"] = len(chunks)
                return pooled

            def pooled_rows(method: str) -> list[dict]:
                rows = []
                for arm in COMPARISON_ARMS:
                    pooled = pooled_deltas(arm, method)
                    if pooled is None:
                        continue
                    result = paired_mean_inference(
                        pooled,
                        arm=arm,
                        control=CONTROL_ARM,
                        label_horizon=LABEL_T,
                        block_size=BLOCK_SIZE,
                        hac_lags=HAC_LAGS,
                        n_resamples=N_RESAMPLES,
                        seed=BOOTSTRAP_SEED,
                        ci_level=CI_LEVEL,
                    )
                    rows.append(
                        {"method": method, "n_core_folds": pooled.attrs["n_core_folds"], **result.__dict__}
                    )
                return rows

            paired_pooled_core = pd.DataFrame([row for method in IC_METHODS for row in pooled_rows(method)])
            # BHY across the five arm-versus-control contrasts, fixed on ticket 181
            # before any number was seen and never extended afterwards.
            paired_pooled_core["bhy_p"] = paired_pooled_core.groupby("method")["hac_p"].transform(
                lambda s: bhy_adjusted_p_values(s.to_numpy())
            )
            paired_pooled_core["ci_excludes_zero"] = (paired_pooled_core["ci_lower"] > 0) | (
                paired_pooled_core["ci_upper"] < 0
            )

            # Realised MDE from the run's own sd, and the per-arm branch of the
            # pre-registered outcome map against the 0.005 line.
            branches = []
            for _, row in paired_pooled_core.iterrows():
                realised_mde = minimum_detectable_effect(
                    float(row["sd_delta"]), int(row["n_days"]), power=POWER, alpha=ALPHA
                )
                core = paired_per_fold[
                    (paired_per_fold["method"] == row["method"])
                    & (paired_per_fold["arm"] == row["arm"])
                    & (paired_per_fold["fold"].isin(POOLED_FOLD_KEYS))
                ]
                same_sign = int((np.sign(core["mean_delta"]) == np.sign(row["mean_delta"])).sum())
                branches.append(
                    {
                        "method": row["method"],
                        "arm": row["arm"],
                        "mean_delta": float(row["mean_delta"]),
                        "sd_delta": float(row["sd_delta"]),
                        "n_days": int(row["n_days"]),
                        "hac_p": float(row["hac_p"]),
                        "bhy_p": float(row["bhy_p"]),
                        "sign_consistent_core_folds": same_sign,
                        "realised_mde": realised_mde,
                        "effect_worth_acting_on": EFFECT_WORTH_ACTING_ON,
                        "mde_at_or_below_line": bool(realised_mde <= EFFECT_WORTH_ACTING_ON),
                        "outcome_branch": outcome_branch(
                            bhy_p=float(row["bhy_p"]),
                            mean_delta=float(row["mean_delta"]),
                            sign_consistent_folds=same_sign,
                            realised_mde=realised_mde,
                        ),
                    }
                )
            outcome_df = pd.DataFrame(branches)
            paired_pooled_core.to_csv(RUN_ROOT / f"paired_pooled_core_{RUN_STAGE}.csv", index=False)
            outcome_df.to_csv(RUN_ROOT / f"paired_outcome_map_{RUN_STAGE}.csv", index=False)

            # Secondary: seed-paired per-model IC, member i against member i,
            # twenty pairs per arm per fold, pooled over the core folds and
            # disclosed per fold; paired t with BHY (ticket 181 section 7).
            def per_model_ic(fold_key: str, arm_key: str) -> list[float]:
                run_dir = run_dirs[(fold_key, arm_key)]
                member_root = run_dir / "model_predictions"
                member_dirs = sorted(member_root.glob("model_*")) if member_root.exists() else []
                out = []
                for member_dir in member_dirs:
                    series = dated_ic(score_pivot(member_dir), LABEL_ARBITER, "pearson")
                    out.append(float(series.mean()) if len(series) else float("nan"))
                return out

            per_model_rows = []
            for fold_key in FOLD_KEYS_PRESENT:
                if (fold_key, CONTROL_ARM) not in run_dirs:
                    continue
                control_members = per_model_ic(fold_key, CONTROL_ARM)
                for arm in COMPARISON_ARMS:
                    if (fold_key, arm) not in run_dirs:
                        continue
                    arm_members = per_model_ic(fold_key, arm)
                    width = min(len(control_members), len(arm_members))
                    for member_id in range(width):
                        per_model_rows.append(
                            {
                                "fold": fold_key,
                                "arm": arm,
                                "member": member_id,
                                "delta": arm_members[member_id] - control_members[member_id],
                            }
                        )
            per_model_df = pd.DataFrame(per_model_rows)
            per_model_df.to_csv(RUN_ROOT / f"per_model_ic_paired_{RUN_STAGE}.csv", index=False)

            def per_model_inference(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
                # Paired t over the member deltas -- the deltas are already
                # member-paired, so a one-sample t on them is the paired t --
                # then BHY across the same five contrasts.
                rows = []
                for arm in COMPARISON_ARMS:
                    values = frame.loc[frame["arm"] == arm, "delta"].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size < 2:
                        continue
                    result = stats.ttest_1samp(values, 0.0)
                    rows.append(
                        {
                            "scope": scope,
                            "arm": arm,
                            "n_pairs": int(values.size),
                            "mean_delta": float(values.mean()),
                            "t": float(result.statistic),
                            "p": float(result.pvalue),
                        }
                    )
                table = pd.DataFrame(rows)
                if not table.empty:
                    table["bhy_p"] = bhy_adjusted_p_values(table["p"].to_numpy())
                return table

            per_model_stats = pd.DataFrame()
            if not per_model_df.empty:
                per_model_stats = pd.concat(
                    [
                        per_model_inference(
                            per_model_df[per_model_df["fold"].isin(POOLED_FOLD_KEYS)], "pooled_core"
                        ),
                        *[
                            per_model_inference(per_model_df[per_model_df["fold"] == fold_key], fold_key)
                            for fold_key in FOLD_KEYS_PRESENT
                        ],
                    ],
                    ignore_index=True,
                )
            per_model_stats.to_csv(RUN_ROOT / f"per_model_ic_inference_{RUN_STAGE}.csv", index=False)

            # Secondary: median difference with a sign test, per fold and pooled
            # (ticket 181 section 7).
            def sign_test_p(values: np.ndarray) -> float:
                x = values[np.isfinite(values) & (values != 0.0)]
                if x.size == 0:
                    return float("nan")
                return float(stats.binomtest(int((x > 0).sum()), x.size, 0.5).pvalue)

            def median_sign_rows(scope: str, deltas_by_arm: dict) -> list[dict]:
                rows = []
                for arm, values in deltas_by_arm.items():
                    if values is None or values.size == 0:
                        continue
                    rows.append(
                        {
                            "scope": scope,
                            "arm": arm,
                            "n_days": int(values.size),
                            "median_delta": float(np.median(values)),
                            "sign_test_p": sign_test_p(values),
                        }
                    )
                return rows

            median_frames = []
            pooled_by_arm = {}
            for arm in COMPARISON_ARMS:
                pooled = pooled_deltas(arm, "pearson")
                pooled_by_arm[arm] = None if pooled is None else pooled.to_numpy()
            median_frames.append(pd.DataFrame(median_sign_rows("pooled_core", pooled_by_arm)))
            for fold_key in FOLD_KEYS_PRESENT:
                fold_by_arm = {}
                for arm in COMPARISON_ARMS:
                    series = fold_deltas(arm, "pearson", fold_key)
                    fold_by_arm[arm] = None if series is None else series.to_numpy()
                median_frames.append(pd.DataFrame(median_sign_rows(fold_key, fold_by_arm)))

            median_df = pd.concat([frame for frame in median_frames if not frame.empty], ignore_index=True)
            median_df["bhy_p"] = median_df.groupby("scope")["sign_test_p"].transform(
                lambda s: bhy_adjusted_p_values(s.to_numpy())
            )
            median_df.to_csv(RUN_ROOT / f"median_sign_test_{RUN_STAGE}.csv", index=False)

            # Arm-versus-arm contrasts: reported paired with intervals, labelled
            # descriptive and unadjusted (ticket 181 section 6). They route D1 only
            # after some arm has cleared the bar against the control.
            DESCRIPTIVE_CONTRASTS = [
                ("A4_sector_only", "A2_thr05"),
                ("A4_sector_only", "A3_topk20"),
                ("A3_topk20", "A3s_topk20_scalar"),
            ]
            descriptive_rows = []
            for left, right in DESCRIPTIVE_CONTRASTS:
                chunks = [
                    paired_daily_differences(
                        align_daily_series(
                            {right: IC["pearson"][(fold_key, right)], left: IC["pearson"][(fold_key, left)]}
                        ),
                        control=right,
                    )[left]
                    for fold_key in POOLED_FOLD_KEYS
                    if (fold_key, left) in IC["pearson"] and (fold_key, right) in IC["pearson"]
                ]
                if not chunks:
                    continue
                result = paired_mean_inference(
                    pd.concat(chunks),
                    arm=left,
                    control=right,
                    label_horizon=LABEL_T,
                    block_size=BLOCK_SIZE,
                    hac_lags=HAC_LAGS,
                    n_resamples=N_RESAMPLES,
                    seed=BOOTSTRAP_SEED,
                    ci_level=CI_LEVEL,
                )
                descriptive_rows.append({"adjusted": False, "note": "descriptive", **result.__dict__})
            descriptive_df = pd.DataFrame(descriptive_rows)
            descriptive_df.to_csv(RUN_ROOT / f"arm_vs_arm_descriptive_{RUN_STAGE}.csv", index=False)

            print("=== primary: pooled over the core folds, Pearson on the arbiter label ===")
            display(paired_pooled_core[paired_pooled_core["method"] == "pearson"])
            print("=== outcome map, per arm, against", EFFECT_WORTH_ACTING_ON, "===")
            display(outcome_df[outcome_df["method"] == "pearson"])
            print("=== per fold (core and alongside; alongside is never pooled) ===")
            display(paired_per_fold[paired_per_fold["method"] == "pearson"])
            print("=== secondaries: median with a sign test; seed-paired per-model IC ===")
            display(median_df)
            display(per_model_stats)
            print("=== arm versus arm: descriptive, unadjusted ===")
            display(descriptive_df)
            """
        ),
        md("## 10. Disclosure: Ensemble Scale, Sharpe Intervals, Basket Returns, April Composite"),
        code(
            r"""
            from mci_gru.evaluation.paired_inference import sharpe_block_bootstrap_ci

            # Ensemble-scale audit and the rank-averaged ensemble. The IC-loss
            # ensemble is an implicit scale-weighted average, so the rank-average
            # is reported alongside to show whether the ordering survives it.
            ensemble_rows = []
            for (fold_key, arm_key), run_dir in run_dirs.items():
                member_root = run_dir / "model_predictions"
                member_dirs = sorted(member_root.glob("model_*")) if member_root.exists() else []
                if not member_dirs:
                    continue
                pivots = [score_pivot(member_dir) for member_dir in member_dirs]
                scales = [float(p.std(axis=1).mean()) for p in pivots]
                ranks = [p.rank(axis=1, pct=True) for p in pivots]
                rank_mean = sum(ranks) / len(ranks)
                ensemble_rows.append(
                    {
                        "fold": fold_key,
                        "arm": arm_key,
                        "n_members": len(member_dirs),
                        "ensemble_scale_mean": float(np.mean(scales)),
                        "ensemble_scale_min": float(np.min(scales)),
                        "ensemble_scale_max": float(np.max(scales)),
                        "ensemble_scale_cv": float(np.std(scales) / np.mean(scales)) if np.mean(scales) else float("nan"),
                        "raw_mean_ensemble_ic": float(dated_ic(score_pivot(run_dir / "averaged_predictions"), LABEL_ARBITER, "pearson").mean()),
                        "rank_average_ensemble_ic": float(dated_ic(rank_mean, LABEL_ARBITER, "pearson").mean()),
                    }
                )
            ensemble_df = pd.DataFrame(ensemble_rows)
            ensemble_df.to_csv(RUN_ROOT / f"ensemble_scale_audit_{RUN_STAGE}.csv", index=False)

            # Top-K Sharpe with block-bootstrap intervals, and the paired daily
            # basket returns by K. Disclosure only: at 238 days per fold the
            # intervals are about five units wide and cannot order arms.
            def basket_returns(fold_key: str, arm_key: str, top_k: int) -> pd.Series:
                scores = score_pivot(run_dirs[(fold_key, arm_key)] / "averaged_predictions")
                dates = [d for d in scores.index if d in LABEL_ARBITER.index]
                cols = scores.columns.intersection(LABEL_ARBITER.columns)
                out = {}
                for date in dates:
                    row = scores.loc[date, cols].dropna()
                    rets = LABEL_ARBITER.loc[date, cols]
                    picks = row.sort_values(ascending=False).head(top_k).index
                    values = rets.reindex(picks).dropna()
                    if len(values):
                        out[date] = float(values.mean())
                return pd.Series(out, dtype=float).sort_index()

            sharpe_rows, paired_basket_rows = [], []
            for fold_key in FOLD_KEYS_PRESENT:
                for top_k in TOP_K_VALUES:
                    baskets = {
                        arm: basket_returns(fold_key, arm, top_k)
                        for arm in ALL_ARMS
                        if (fold_key, arm) in run_dirs
                    }
                    for arm, series in baskets.items():
                        if not len(series):
                            continue
                        interval = sharpe_block_bootstrap_ci(
                            series.to_numpy(),
                            nw_lags=HAC_LAGS,
                            block_size=BLOCK_SIZE,
                            n_resamples=N_RESAMPLES,
                            seed=BOOTSTRAP_SEED,
                            ci_level=CI_LEVEL,
                        )
                        sharpe_rows.append({"fold": fold_key, "arm": arm, "top_k": top_k, **interval})
                    if CONTROL_ARM not in baskets:
                        continue
                    aligned = align_daily_series({arm: series for arm, series in baskets.items() if len(series)})
                    deltas = paired_daily_differences(aligned, control=CONTROL_ARM)
                    for arm in deltas.columns:
                        result = paired_mean_inference(
                            deltas[arm],
                            arm=arm,
                            control=CONTROL_ARM,
                            label_horizon=LABEL_T,
                            block_size=BLOCK_SIZE,
                            hac_lags=HAC_LAGS,
                            n_resamples=N_RESAMPLES,
                            seed=BOOTSTRAP_SEED,
                            ci_level=CI_LEVEL,
                        )
                        paired_basket_rows.append({"fold": fold_key, "top_k": top_k, **result.__dict__})

            sharpe_df = pd.DataFrame(sharpe_rows)
            sharpe_df.to_csv(RUN_ROOT / f"sharpe_block_bootstrap_{RUN_STAGE}.csv", index=False)
            paired_basket_return_df = pd.DataFrame(paired_basket_rows)
            paired_basket_return_df.to_csv(RUN_ROOT / f"paired_basket_returns_{RUN_STAGE}.csv", index=False)

            # April 2026's composite decision score, computed alongside for
            # reconciliation with the historical ablation reports. The composite
            # is not the arbiter; the paired difference in section 9 is.
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
                grouped = vals.groupby(scored_df["fold"])
                standardised = (vals - grouped.transform("mean")) / grouped.transform("std")
                score = score + weight * standardised.fillna(0.0)
            scored_df["decision_score"] = score
            scored_df.to_csv(RUN_ROOT / f"april_composite_{RUN_STAGE}.csv", index=False)

            display(ensemble_df)
            display(sharpe_df)
            display(paired_basket_return_df)
            display(scored_df[["fold", "arm", "avg_ic", "avg_ic_ci_lower", "avg_ic_ci_upper", "decision_score"]])
            """
        ),
        md("## 11. Sanity Summary"),
        code(
            r"""
            # There is no promotion gate, so this stage decides nothing. What it
            # does is say whether the mechanics held: every job trained, the twin
            # rule held everywhere, and each fold's arms shared one base seed.
            summary_lines = [
                "# Graph-Specification Ablation Summary",
                "",
                f"- Run root: `{RUN_ROOT}`",
                f"- Stage: `{RUN_STAGE}` | smoke: `{SMOKE_MODE}`",
                f"- Budget: {num_models} models x {num_epochs} epochs, patience {EARLY_STOPPING_PATIENCE}",
                f"- Jobs: {len(jobs)} = {len(folds_to_run)} folds x {len(arms_to_run)} arms",
                f"- Rule: {manifest['promotion_rule']}",
                f"- Twin exclusion: `{TWIN_OVERRIDE}` (twin_edge_count == 0 verified per fold and arm)",
                "- Edge-dropout RNG isolated: `graph.isolate_edge_dropout_rng=true` in every arm",
                f"- Arbiter: mean paired daily Pearson IC difference against {CONTROL_ARM}, pooled over "
                f"{POOLED_FOLD_KEYS}, HAC lag {HAC_LAGS}, block {BLOCK_SIZE}, BHY over {BHY_CONTRAST_COUNT} contrasts",
                f"- Alongside folds {ALONGSIDE_FOLD_KEYS} are disclosed per fold and never pooled",
                "",
                "## Per fold and arm",
                "",
                results_df[["fold", "arm", "seed", "avg_ic", "avg_ic_ci_lower", "avg_ic_ci_upper"]].to_markdown(index=False),
                "",
                "## Outcome map (pooled core folds, Pearson)",
                "",
                outcome_df[outcome_df["method"] == "pearson"][
                    ["arm", "mean_delta", "bhy_p", "sign_consistent_core_folds", "realised_mde", "outcome_branch"]
                ].to_markdown(index=False),
            ]
            summary_path = RUN_ROOT / f"graph_specification_ablation_summary_{RUN_STAGE}.md"
            summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
            print("Summary:", summary_path)

            sanity = {
                "jobs_expected": len(jobs),
                "jobs_completed": len(run_dirs),
                "twin_edges_total": int(disclosure_df["twin_edge_count"].sum()),
                "distinct_fold_seeds": int(results_df.groupby("fold")["seed"].nunique().max()),
                "folds_present": sorted({fold_key for fold_key, _ in run_dirs}),
                "arms_present": sorted({arm_key for _, arm_key in run_dirs}),
            }
            print(json.dumps(sanity, indent=2))
            if sanity["jobs_completed"] != sanity["jobs_expected"]:
                raise AssertionError("Not every job completed; the manifest records what landed.")
            if sanity["twin_edges_total"] != 0:
                raise AssertionError("Twin edge present; the hygiene rule failed.")
            if sanity["distinct_fold_seeds"] != 1:
                raise AssertionError("A fold's arms did not share one base seed.")
            print("Mechanics sanity passed.")
            """
        ),
    ]


def render() -> str:
    """Exact notebook payload; the contract test compares this to the committed file."""
    return json.dumps(build_notebook(build_cells()), indent=1)


def main(out: Path = OUT) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
