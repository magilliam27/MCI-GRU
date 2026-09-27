"""Generate the paired re-analysis Colab notebook for the graph ablation (ticket 179).

Phase 0 of the multi-year protocol proposal recorded on Wayfinder map 157: read
the ``20260901_015032`` screen and confirm run directories from Drive, recompute
each arm's daily cross-sectional IC on the arbiter's own label
(``close[t+5] / close[t+1] - 1``, per ``mci_gru.data.preprocessing.compute_labels``),
and analyse the arms as *paired* against the graph-zeroed control on shared test
days. Also computed: the power a multi-year protocol would have, the sensitivity
of the verdict to Spearman / winsorised / median variants, portfolio Sharpe with
error bars, seed-paired per-model IC, and the ensemble-scale audit.

Nothing here changes the ticket-164 arbiter or trains anything. The notebook runs
on a Colab CPU runtime. The protocol constants are module-level so
``tests/test_graph_paired_reanalysis_notebook.py`` can import them and check the
emitted notebook against them, including byte-identical regeneration.
"""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pformat

from nb_lib import build_notebook, code, colab_setup_cell, md

OUT = Path("notebooks/graph_paired_reanalysis_colab.ipynb")

# The notebook imports ``mci_gru.evaluation.paired_inference``, which lands on
# ``main`` only when ticket 179's pull request merges; until then the notebook
# must clone the ticket branch. Flip this to "main" after the merge and
# regenerate; the contract test pins the notebook to this constant.
NOTEBOOK_BRANCH = "claude/179-paired-reanalysis"

RUN_TAG = "20260901_015032"
CONTROL_ARM = "A0_zeroed"
COMPARISON_ARMS = ["A1_shipped", "A2_thr05", "A3_topk20", "A4_sector_only"]
STAGES = ["confirm", "screen"]
CONFIRM_NUM_MODELS = 20

# The arbiter's evaluation settings, resolved exactly as
# ``mci_gru.evaluation.experiment_summary.resolved_evaluation_kwargs`` resolves
# them for the frozen recipe: block = label_t, Newey-West lags = label_t - 1.
LABEL_T = 5
BLOCK_SIZE = 5
HAC_LAGS = 4
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 1729
CI_LEVEL = 0.95
TOP_K_VALUES = [10, 20, 50, 100]
WINSOR_QUANTILES = (0.01, 0.99)
TAIL_TOP_FRACTION = 0.1
POWER = 0.8
ALPHA = 0.05
POWER_DAY_GRID = [238, 476, 714, 952, 1190, 1428]
POWER_MDE_GRID = [0.002, 0.005, 0.010]

PANEL_NAME = "sp500_pit_gics_top10_mcap_monthly_20160104_20260731_lseg_20150101_20260731.csv"
PANEL_SHA256 = "d64c4d041ef4c1632ed76e1456885ffe8301a477c8e27c4a73805f94ff97aeb4"

# CPU notebook: no accelerator key, otherwise the canonical Colab metadata.
NOTEBOOK_METADATA: dict = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def _embed(obj) -> str:
    return pformat(obj, width=88, sort_dicts=False)


def build_cells() -> list[dict]:
    return [
        md(
            f"""
            # Paired Re-analysis Of The Graph-Specification Ablation (ticket 179)

            Phase 0 of the multi-year protocol proposal on Wayfinder map 157. Reads the
            `{RUN_TAG}` screen and confirm runs from Drive and re-scores every arm on the
            **same test days as the control**, so arms are compared through the daily
            difference `delta_k(d) = IC_k(d) - IC_{CONTROL_ARM}(d)` rather than through
            independent-looking intervals.

            - **Label:** the arbiter's own, `close[t+5] / close[t+1] - 1`
              (`mci_gru.data.preprocessing.compute_labels`). The ablation notebook's
              section 8 used `close[t+5] / close[t] - 1`; that variant is computed
              alongside so the discrepancy is quantified, not guessed.
            - **Inference:** Newey-West HAC (`lags = {HAC_LAGS}`) and a circular
              {BLOCK_SIZE}-session block bootstrap on the mean difference; BHY across the
              {len(COMPARISON_ARMS)} arm-versus-control comparisons.
            - **Power:** `sd(delta_k)` and the minimum detectable effect per pooled-day
              count, which decides whether a multi-year design can separate these arms.
            - **Sensitivity:** Spearman, winsorised Pearson, median, tail-day share,
              Sharpe with bootstrap intervals, seed-paired per-model IC, ensemble scale.

            This notebook does **not** change the ticket-164 arbiter and trains nothing.
            It runs on a CPU runtime. Smoke run tag `20260901_014022` is not an input.
            """
        ),
        md("## 1. Setup (CPU runtime)"),
        colab_setup_cell(branch=NOTEBOOK_BRANCH, require_gpu=False),
        md("## 2. Constants And Run Root"),
        code(
            f"""
            import hashlib

            import numpy as np
            import pandas as pd

            from mci_gru.evaluation.paired_inference import (
                align_daily_series,
                bhy_adjusted_p_values,
                minimum_detectable_effect,
                paired_daily_differences,
                paired_mean_inference,
                required_days,
                sharpe_block_bootstrap_ci,
                tail_share,
                winsorize_rows,
            )
            from mci_gru.evaluation.portfolio import top_k_returns
            from mci_gru.evaluation.prediction_report import (
                load_prediction_files,
                realized_returns_from_market_data,
            )
            from mci_gru.evaluation.statistics import cross_sectional_ic

            RUN_TAG = {RUN_TAG!r}
            CONTROL_ARM = {CONTROL_ARM!r}
            COMPARISON_ARMS = {_embed(COMPARISON_ARMS)}
            ALL_ARMS = [CONTROL_ARM, *COMPARISON_ARMS]
            STAGES = {_embed(STAGES)}
            CONFIRM_NUM_MODELS = {CONFIRM_NUM_MODELS}

            LABEL_T = {LABEL_T}
            BLOCK_SIZE = {BLOCK_SIZE}
            HAC_LAGS = {HAC_LAGS}
            N_RESAMPLES = {N_RESAMPLES}
            BOOTSTRAP_SEED = {BOOTSTRAP_SEED}
            CI_LEVEL = {CI_LEVEL}
            TOP_K_VALUES = {_embed(TOP_K_VALUES)}
            WINSOR_QUANTILES = {WINSOR_QUANTILES!r}
            TAIL_TOP_FRACTION = {TAIL_TOP_FRACTION}
            POWER = {POWER}
            ALPHA = {ALPHA}
            POWER_DAY_GRID = {_embed(POWER_DAY_GRID)}
            POWER_MDE_GRID = {_embed(POWER_MDE_GRID)}

            PANEL_NAME = {PANEL_NAME!r}
            PANEL_SHA256 = {PANEL_SHA256!r}

            DRIVE_DATA_DIR = Path("/content/drive/MyDrive/MCI_GRU_shared/data")
            RUN_ROOT = Path("/content/drive/MyDrive/MCI-GRU-Ablations/graph_specification_ablation") / RUN_TAG
            if not IN_COLAB:
                RUN_ROOT = REPO_DIR / "results" / "graph_specification_ablation" / RUN_TAG
            OUT_DIR = RUN_ROOT / "reanalysis"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            if not (RUN_ROOT / "training").is_dir():
                raise FileNotFoundError(f"Run root has no training directory: {{RUN_ROOT}}")


            def latest_complete_run_dir(stage: str, arm: str) -> Path:
                # Newest timestamp directory that actually finished (has an evaluation
                # summary). The interrupted two-arm confirm attempt left a partial
                # directory under A0 with no summary; this skips it by construction.
                base = RUN_ROOT / "training" / stage / f"graphspec_{{stage}}_{{arm}}_seed{{BOOTSTRAP_SEED}}"
                candidates = sorted(
                    path for path in base.iterdir()
                    if path.is_dir() and (path / "evaluation_summary.json").is_file()
                )
                if not candidates:
                    raise FileNotFoundError(f"No completed run under {{base}}")
                return candidates[-1]


            RUN_DIRS = {{
                stage: {{arm: latest_complete_run_dir(stage, arm) for arm in ALL_ARMS}}
                for stage in STAGES
            }}
            for stage in STAGES:
                for arm in ALL_ARMS:
                    print(stage, arm, "->", RUN_DIRS[stage][arm].name)
            print("Outputs ->", OUT_DIR)
            """
        ),
        md("## 3. The Panel And The Arbiter's Label"),
        code(
            r"""
            def sha256_file(path: Path) -> str:
                h = hashlib.sha256()
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()


            panel_path = (DRIVE_DATA_DIR if IN_COLAB else REPO_DIR / "data" / "raw" / "market") / PANEL_NAME
            if not panel_path.is_file():
                raise FileNotFoundError(f"Missing market panel: {panel_path}")
            panel_sha = sha256_file(panel_path)
            if panel_sha != PANEL_SHA256:
                raise RuntimeError(f"Panel digest {panel_sha} != expected {PANEL_SHA256}")
            print("Panel:", panel_path.name, panel_sha[:16])

            market = pd.read_csv(panel_path, usecols=["dt", "kdcode", "close"])
            market["dt"] = pd.to_datetime(market["dt"]).dt.strftime("%Y-%m-%d")

            # Arbiter label: close[t + LABEL_T] / close[t + 1] - 1, identical to
            # mci_gru.data.preprocessing.compute_labels and to what the production
            # evaluation received as ``true_returns``.
            realized = realized_returns_from_market_data(market, label_t=LABEL_T)
            LABEL_ARBITER = realized.pivot_table(index="dt", columns="kdcode", values="realized_return").sort_index()

            # Section-8 variant from the ablation notebook: close[t + LABEL_T] / close[t] - 1.
            close_pivot = market.pivot_table(index="dt", columns="kdcode", values="close").sort_index()
            LABEL_SECTION8 = close_pivot.shift(-LABEL_T) / close_pivot - 1

            LABELS = {"arbiter": LABEL_ARBITER, "section8": LABEL_SECTION8}
            print("Label frames:", {k: v.shape for k, v in LABELS.items()})
            """
        ),
        md("## 4. Daily IC Per Arm, Recomputed On The Arbiter's Label"),
        code(
            r"""
            def score_pivot(predictions_dir: Path) -> pd.DataFrame:
                scores = load_prediction_files(predictions_dir)
                scores["dt"] = scores["dt"].astype(str)
                return scores.pivot_table(index="dt", columns="kdcode", values="score").sort_index()


            def dated_ic(scores: pd.DataFrame, label: pd.DataFrame, method: str, winsor=None) -> pd.Series:
                # Same per-day semantics as mci_gru.evaluation.statistics.daily_ic_series
                # (finite pairs only, at least two names, non-degenerate), but dated so the
                # arms can be paired on the calendar.
                dates = [d for d in scores.index if d in label.index]
                cols = scores.columns.intersection(label.columns)
                pred = scores.loc[dates, cols].to_numpy(dtype=float)
                if winsor is not None:
                    pred = winsorize_rows(pred, *winsor)
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


            IC_VARIANTS = {
                "pearson": ("arbiter", "pearson", None),
                "spearman": ("arbiter", "spearman", None),
                "pearson_winsor": ("arbiter", "pearson", WINSOR_QUANTILES),
                "pearson_section8": ("section8", "pearson", None),
            }

            SCORES = {stage: {arm: score_pivot(RUN_DIRS[stage][arm] / "averaged_predictions") for arm in ALL_ARMS} for stage in STAGES}
            IC = {
                stage: {
                    variant: {arm: dated_ic(SCORES[stage][arm], LABELS[label_key], method, winsor) for arm in ALL_ARMS}
                    for variant, (label_key, method, winsor) in IC_VARIANTS.items()
                }
                for stage in STAGES
            }

            # Reconciliation against the stored arbiter: the pearson recomputation on the
            # arbiter label should reproduce evaluation_summary.json's avg_ic closely; the
            # section-8 variant shows how far the 167 per-year table sat from it.
            rows = []
            for stage in STAGES:
                for arm in ALL_ARMS:
                    stored = json.loads((RUN_DIRS[stage][arm] / "evaluation_summary.json").read_text(encoding="utf-8"))["metrics"]
                    rows.append(
                        {
                            "stage": stage,
                            "arm": arm,
                            "stored_avg_ic": stored.get("avg_ic"),
                            "recomputed_pearson_arbiter_label": float(IC[stage]["pearson"][arm].mean()),
                            "recomputed_pearson_section8_label": float(IC[stage]["pearson_section8"][arm].mean()),
                            "stored_avg_rank_ic": stored.get("avg_rank_ic"),
                            "recomputed_spearman_arbiter_label": float(IC[stage]["spearman"][arm].mean()),
                            "n_days_recomputed": int(len(IC[stage]["pearson"][arm])),
                        }
                    )
            reconciliation_df = pd.DataFrame(rows)
            reconciliation_df["abs_gap_arbiter"] = (reconciliation_df["recomputed_pearson_arbiter_label"] - reconciliation_df["stored_avg_ic"]).abs()
            reconciliation_df["abs_gap_section8"] = (reconciliation_df["recomputed_pearson_section8_label"] - reconciliation_df["stored_avg_ic"]).abs()
            reconciliation_df.to_csv(OUT_DIR / "arbiter_reconciliation.csv", index=False)
            display(reconciliation_df)
            """
        ),
        md("## 5. Paired Differences Against The Control"),
        code(
            r"""
            def paired_table(stage: str, variant: str) -> pd.DataFrame:
                aligned = align_daily_series({arm: IC[stage][variant][arm] for arm in ALL_ARMS})
                deltas = paired_daily_differences(aligned, control=CONTROL_ARM)
                rows = []
                for arm in COMPARISON_ARMS:
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
                    rows.append({"stage": stage, "variant": variant, **result.__dict__})
                table = pd.DataFrame(rows)
                table["bhy_p"] = bhy_adjusted_p_values(table["hac_p"].to_numpy())
                table["ci_excludes_zero"] = (table["ci_lower"] > 0) | (table["ci_upper"] < 0)
                return table


            PAIRED = {stage: {variant: paired_table(stage, variant) for variant in IC_VARIANTS} for stage in STAGES}
            paired_df = pd.concat([PAIRED[s][v] for s in STAGES for v in IC_VARIANTS], ignore_index=True)
            paired_df.to_csv(OUT_DIR / "paired_inference.csv", index=False)
            for stage in STAGES:
                print(f"=== {stage}: primary (pearson, arbiter label) ===")
                display(PAIRED[stage]["pearson"][["arm", "n_days", "mean_delta", "median_delta", "sd_delta", "win_rate", "hac_t", "hac_p", "bhy_p", "ci_lower", "ci_upper", "top_decile_share"]])
            """
        ),
        md("## 6. Power: What A Multi-Year Design Could Detect"),
        code(
            r"""
            rows = []
            for stage in STAGES:
                for _, row in PAIRED[stage]["pearson"].iterrows():
                    sd = float(row["sd_delta"])
                    entry = {"stage": stage, "arm": row["arm"], "sd_delta": sd, "n_days_observed": int(row["n_days"])}
                    for n in POWER_DAY_GRID:
                        entry[f"mde_at_{n}_days"] = minimum_detectable_effect(sd, n, power=POWER, alpha=ALPHA)
                    for mde in POWER_MDE_GRID:
                        entry[f"days_for_mde_{mde:g}"] = required_days(sd, mde, power=POWER, alpha=ALPHA)
                    entry["observed_mean_delta"] = float(row["mean_delta"])
                    entry["observed_abs_mean_over_mde_at_952"] = abs(float(row["mean_delta"])) / entry["mde_at_952_days"]
                    rows.append(entry)
            power_df = pd.DataFrame(rows)
            power_df.to_csv(OUT_DIR / "power.csv", index=False)
            display(power_df)
            """
        ),
        md("## 7. Shape Of The Daily Differences"),
        code(
            r"""
            rows = []
            largest = []
            for stage in STAGES:
                aligned = align_daily_series({arm: IC[stage]["pearson"][arm] for arm in ALL_ARMS})
                deltas = paired_daily_differences(aligned, control=CONTROL_ARM)
                for arm in COMPARISON_ARMS:
                    d = deltas[arm]
                    q = d.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
                    rows.append(
                        {
                            "stage": stage,
                            "arm": arm,
                            "n_days": int(d.size),
                            "mean": float(d.mean()),
                            "median": float(d.median()),
                            "win_rate": float((d > 0).mean()),
                            "q05": float(q.loc[0.05]),
                            "q25": float(q.loc[0.25]),
                            "q50": float(q.loc[0.50]),
                            "q75": float(q.loc[0.75]),
                            "q95": float(q.loc[0.95]),
                            "top_decile_share": tail_share(d.to_numpy(), TAIL_TOP_FRACTION),
                            "bottom_decile_share": tail_share(-d.to_numpy(), TAIL_TOP_FRACTION),
                        }
                    )
                    top = d.abs().sort_values(ascending=False).head(10)
                    for date, mag in top.items():
                        largest.append({"stage": stage, "arm": arm, "date": date, "delta": float(d.loc[date]), "abs_delta": float(mag)})
            delta_shape_df = pd.DataFrame(rows)
            delta_shape_df.to_csv(OUT_DIR / "delta_distribution.csv", index=False)
            pd.DataFrame(largest).to_csv(OUT_DIR / "largest_abs_delta_days.csv", index=False)
            display(delta_shape_df)
            """
        ),
        md("## 8. Portfolio Sharpe With Error Bars, And Paired Daily Portfolio Returns"),
        code(
            r"""
            def portfolio_daily_returns(stage: str, arm: str, k: int) -> pd.Series:
                scores = SCORES[stage][arm]
                label = LABEL_ARBITER
                dates = [d for d in scores.index if d in label.index]
                cols = scores.columns.intersection(label.columns)
                pred = scores.loc[dates, cols].to_numpy(dtype=float)
                rets = label.loc[dates, cols].to_numpy(dtype=float)
                # top_k_returns skips days with no finite pair; keep the date alignment by
                # computing per row.
                out = {}
                for i, date in enumerate(dates):
                    values = top_k_returns(pred[i : i + 1], rets[i : i + 1], top_k=k)
                    if values.size:
                        out[date] = float(values[0])
                return pd.Series(out, dtype=float).sort_index()


            sharpe_rows, paired_port_rows = [], []
            for stage in STAGES:
                for k in TOP_K_VALUES:
                    series = {arm: portfolio_daily_returns(stage, arm, k) for arm in ALL_ARMS}
                    for arm in ALL_ARMS:
                        ci = sharpe_block_bootstrap_ci(
                            series[arm].to_numpy(),
                            nw_lags=HAC_LAGS,
                            block_size=BLOCK_SIZE,
                            n_resamples=N_RESAMPLES,
                            seed=BOOTSTRAP_SEED + k,
                            ci_level=CI_LEVEL,
                        )
                        sharpe_rows.append({"stage": stage, "top_k": k, "arm": arm, "mean_daily_return": float(series[arm].mean()), **{f"sharpe_{key}": value for key, value in ci.items()}})
                    aligned = align_daily_series(series)
                    deltas = paired_daily_differences(aligned, control=CONTROL_ARM)
                    for arm in COMPARISON_ARMS:
                        result = paired_mean_inference(
                            deltas[arm],
                            arm=arm,
                            control=CONTROL_ARM,
                            label_horizon=LABEL_T,
                            block_size=BLOCK_SIZE,
                            hac_lags=HAC_LAGS,
                            n_resamples=N_RESAMPLES,
                            seed=BOOTSTRAP_SEED + k,
                            ci_level=CI_LEVEL,
                        )
                        paired_port_rows.append({"stage": stage, "top_k": k, **result.__dict__})
            sharpe_df = pd.DataFrame(sharpe_rows)
            sharpe_df.to_csv(OUT_DIR / "sharpe_intervals.csv", index=False)
            paired_port_df = pd.DataFrame(paired_port_rows)
            paired_port_df["bhy_p_within_k"] = paired_port_df.groupby(["stage", "top_k"])["hac_p"].transform(lambda s: bhy_adjusted_p_values(s.to_numpy()))
            paired_port_df.to_csv(OUT_DIR / "paired_portfolio_returns.csv", index=False)
            display(sharpe_df[sharpe_df["stage"] == "confirm"])
            display(paired_port_df[paired_port_df["stage"] == "confirm"][["top_k", "arm", "n_days", "mean_delta", "hac_t", "hac_p", "bhy_p_within_k", "ci_lower", "ci_upper"]])
            """
        ),
        md("## 9. Seed-Paired Per-Model IC And The Ensemble-Scale Audit (confirm)"),
        code(
            r"""
            # Within a stage every arm shares the base seed, so model i of arm k and model
            # i of the control share initialisation: twenty paired observations per arm.
            # Reads 20 x 5 prediction directories from Drive; this is the slow cell.
            stage = "confirm"
            per_model_ic = {arm: [] for arm in ALL_ARMS}
            per_model_scale = {arm: [] for arm in ALL_ARMS}
            rank_ensemble_scores = {arm: None for arm in ALL_ARMS}
            for arm in ALL_ARMS:
                rank_sum = None
                for i in range(CONFIRM_NUM_MODELS):
                    pivot = score_pivot(RUN_DIRS[stage][arm] / f"predictions_model_{i}")
                    ic = dated_ic(pivot, LABEL_ARBITER, "pearson")
                    per_model_ic[arm].append(float(ic.mean()))
                    per_model_scale[arm].append(float(pivot.std(axis=1, skipna=True).mean()))
                    ranks = pivot.rank(axis=1, method="average", na_option="keep")
                    rank_sum = ranks if rank_sum is None else rank_sum.add(ranks, fill_value=np.nan)
                rank_ensemble_scores[arm] = rank_sum / CONFIRM_NUM_MODELS
                print(arm, "per-model mean IC", np.round(np.mean(per_model_ic[arm]), 5), "| mean per-model score std", np.round(np.mean(per_model_scale[arm]), 4))

            seed_rows = []
            control_ic = np.asarray(per_model_ic[CONTROL_ARM])
            for arm in COMPARISON_ARMS:
                d = np.asarray(per_model_ic[arm]) - control_ic
                se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else float("nan")
                t = float(d.mean() / se) if se and np.isfinite(se) and se > 0 else float("nan")
                seed_rows.append(
                    {
                        "stage": stage,
                        "arm": arm,
                        "n_models": int(d.size),
                        "mean_model_ic_arm": float(np.mean(per_model_ic[arm])),
                        "mean_model_ic_control": float(control_ic.mean()),
                        "mean_paired_diff": float(d.mean()),
                        "sd_paired_diff": float(d.std(ddof=1)),
                        "paired_t_over_seeds": t,
                        "models_where_arm_beats_control": int((d > 0).sum()),
                    }
                )
            seed_df = pd.DataFrame(seed_rows)
            seed_df.to_csv(OUT_DIR / "seed_paired_per_model_ic.csv", index=False)
            display(seed_df)

            scale_rows = []
            for arm in ALL_ARMS:
                raw_ensemble_ic = float(IC[stage]["pearson"][arm].mean())
                rank_ic = dated_ic(rank_ensemble_scores[arm], LABEL_ARBITER, "pearson")
                scale_rows.append(
                    {
                        "stage": stage,
                        "arm": arm,
                        "mean_per_model_score_std": float(np.mean(per_model_scale[arm])),
                        "min_per_model_score_std": float(np.min(per_model_scale[arm])),
                        "max_per_model_score_std": float(np.max(per_model_scale[arm])),
                        "cv_of_per_model_score_std": float(np.std(per_model_scale[arm], ddof=1) / np.mean(per_model_scale[arm])),
                        "raw_average_ensemble_ic": raw_ensemble_ic,
                        "rank_average_ensemble_ic": float(rank_ic.mean()),
                        "mean_of_per_model_ic": float(np.mean(per_model_ic[arm])),
                    }
                )
            scale_df = pd.DataFrame(scale_rows)
            scale_df.to_csv(OUT_DIR / "ensemble_scale_audit.csv", index=False)
            display(scale_df)
            """
        ),
        md("## 10. Summary And Provenance"),
        code(
            r"""
            import subprocess

            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_DIR)).stdout.strip()
            confirm_primary = PAIRED["confirm"]["pearson"]
            lines = [
                "# Paired re-analysis summary (ticket 179)",
                "",
                f"- Run root: `{RUN_ROOT}`",
                f"- Repository commit: `{commit}` on branch `{BRANCH}`",
                f"- Panel: `{PANEL_NAME}` sha256 `{panel_sha}`",
                f"- Label: close[t+{LABEL_T}] / close[t+1] - 1 (arbiter); section-8 variant computed alongside",
                f"- Inference: Newey-West lags {HAC_LAGS}; block {BLOCK_SIZE}; {N_RESAMPLES} resamples; seed {BOOTSTRAP_SEED}; CI {CI_LEVEL}; BHY over {len(COMPARISON_ARMS)} comparisons",
                "",
                "## Confirm, primary (Pearson, arbiter label): paired against " + CONTROL_ARM,
                "",
                confirm_primary[["arm", "n_days", "mean_delta", "sd_delta", "win_rate", "hac_t", "hac_p", "bhy_p", "ci_lower", "ci_upper", "top_decile_share"]].to_markdown(index=False),
                "",
                "## Power (confirm, primary)",
                "",
                power_df[power_df["stage"] == "confirm"].to_markdown(index=False),
                "",
                "## Arbiter reconciliation",
                "",
                reconciliation_df.to_markdown(index=False),
            ]
            (OUT_DIR / "reanalysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print("\n".join(lines))
            print()
            print("Written:", sorted(p.name for p in OUT_DIR.iterdir()))
            """
        ),
    ]


def render() -> str:
    """Exact notebook payload; the contract test compares this to the committed file."""
    return json.dumps(build_notebook(build_cells(), metadata=NOTEBOOK_METADATA), indent=1)


def main(out: Path = OUT) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
