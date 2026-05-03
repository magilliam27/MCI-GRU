"""Generate the rolling temporal holdout backtest Colab notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/rolling_temporal_backtest_colab.ipynb")


def md(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code(source: str) -> dict:
    source = textwrap.dedent(source).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


cells = [
    md(
        """
        # MCI-GRU Rolling Temporal Backtest - 2022, 2023, 2024

        Trains the frozen promising recipe on rolling five-year windows and backtests the untouched test year for each vintage. The notebook keeps the model recipe fixed so the earlier years are validation evidence, not a fresh model-selection search.
        """
    ),
    md("## 1. Mount Drive, Clone Repo, Install Dependencies"),
    code(
        r"""
        from pathlib import Path
        import os
        import subprocess
        import sys

        try:
            from google.colab import drive
            IN_COLAB = True
        except ImportError:
            drive = None
            IN_COLAB = False

        if IN_COLAB:
            drive.mount('/content/drive')

        REPO_URL = 'https://github.com/magilliam27/MCI-GRU.git'
        BRANCH = 'main'
        REPO_DIR = Path('/content/MCI-GRU') if IN_COLAB else Path.cwd()
        DRIVE_ROOT = Path('/content/drive/MyDrive/MCI-GRU-Ablations') if IN_COLAB else Path.cwd() / 'drive_outputs'
        GDRIVE_DATA_DIR = Path('/content/drive/MyDrive/MCI_GRU_shared/data') if IN_COLAB else Path.cwd() / 'data' / 'raw' / 'market'

        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

        if IN_COLAB:
            if not REPO_DIR.exists():
                subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin'], check=True)
                subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', BRANCH], check=True)
                subprocess.run(['git', '-C', str(REPO_DIR), 'pull', 'origin', BRANCH], check=True)

        os.chdir(REPO_DIR)
        print('Working directory:', Path.cwd())
        print('Drive data folder:', GDRIVE_DATA_DIR)
        print('Drive output root:', DRIVE_ROOT)

        if IN_COLAB:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip', 'setuptools', 'wheel'], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[dev,tracking,fred]'], check=True)

        REQUIRE_GPU = True
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('PyTorch is not installed; rerun setup.') from exc

        print('Torch:', torch.__version__)
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        elif REQUIRE_GPU:
            raise RuntimeError('No CUDA GPU is visible. In Colab, switch Runtime -> Change runtime type -> GPU.')
        """
    ),
    md("## 2. FRED API Key"),
    code(
        r"""
        # Paste your FRED API key here, or set FRED_API_KEY in the notebook environment.
        # Required when REGIME_STRICT=True and REGIME_INPUTS_CSV is blank.
        MY_FRED_KEY = ''

        if MY_FRED_KEY.strip():
            os.environ['FRED_API_KEY'] = MY_FRED_KEY.strip()

        if os.environ.get('FRED_API_KEY'):
            print('FRED_API_KEY is set.')
        else:
            print('FRED_API_KEY is not set yet. Paste it into MY_FRED_KEY before running training, or set REGIME_INPUTS_CSV in the configuration cell.')
        """
    ),
    md("## 3. Configuration"),
    code(
        r"""
        from datetime import datetime

        RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
        EXPERIMENT_SLUG = 'rolling_temporal_backtest'
        RUN_ROOT = DRIVE_ROOT / EXPERIMENT_SLUG / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / 'training_runs'
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Optional: set this to a repo-relative CSV that follows docs/REGIME_DATA_CONTRACT.md.
        # Leave blank to use FRED. With strict regime runs, FRED_API_KEY must be set.
        REGIME_INPUTS_CSV = ''
        REGIME_STRICT = True
        REGIME_ENFORCE_LAG_DAYS = 0

        NUM_MODELS = 10
        NUM_EPOCHS = 100
        EARLY_STOPPING_PATIENCE = 15
        BATCH_SIZE = 32
        LEARNING_RATE = '5e-5'
        BOOTSTRAP_RESAMPLES = 1000

        # Frozen recipe from the 2026-05-02 fairness handoff.
        MODEL_RECIPE = {
            'name': 'static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__seed-42__drop-edge-0p1',
            'seed': 42,
            'label_t': 5,
            'loss_type': 'ic',
            'label_type': 'returns',
            'selection_metric': 'val_ic',
            'drop_edge_p': 0.1,
        }

        TEMPORAL_WINDOWS = [
            {
                'test_year': 2022,
                'data_config': 'temporal_2016',
                'data_filename': 'sp500_2016_universe_data.csv',
                'train_start': '2016-01-01',
                'train_end': '2020-12-31',
                'val_start': '2021-01-22',
                'val_end': '2021-12-31',
                'test_start': '2022-01-22',
                'test_end': '2022-12-31',
            },
            {
                'test_year': 2023,
                'data_config': 'temporal_2017',
                'data_filename': 'sp500_2017_universe_data.csv',
                'train_start': '2017-01-01',
                'train_end': '2021-12-31',
                'val_start': '2022-01-22',
                'val_end': '2022-12-31',
                'test_start': '2023-01-22',
                'test_end': '2023-12-31',
            },
            {
                'test_year': 2024,
                'data_config': 'temporal_2018',
                'data_filename': 'sp500_2018_universe_data.csv',
                'train_start': '2018-01-01',
                'train_end': '2022-12-31',
                'val_start': '2023-01-22',
                'val_end': '2023-12-31',
                'test_start': '2024-01-22',
                'test_end': '2024-12-31',
            },
        ]

        BACKTEST_SCENARIOS = [
            {
                'scenario': 'k10_spread5_slip0_rankdrop30_daily',
                'top_k': 10,
                'transaction_costs': True,
                'spread_bps': 5.0,
                'slippage_bps': 0.0,
                'rank_drop_gate': True,
                'min_rank_drop': 30,
                'holding_period': 1,
                'rebalance_style': 'staggered',
            },
            {
                'scenario': 'k20_spread5_slip0_rankdrop30_daily',
                'top_k': 20,
                'transaction_costs': True,
                'spread_bps': 5.0,
                'slippage_bps': 0.0,
                'rank_drop_gate': True,
                'min_rank_drop': 30,
                'holding_period': 1,
                'rebalance_style': 'staggered',
            },
        ]

        RUN_TRAINING = True
        RUN_BACKTESTS = True
        NUM_TESTS_OVERRIDE = None

        print('Run root:', RUN_ROOT)
        print('Training jobs:', len(TEMPORAL_WINDOWS))
        print('Backtest scenarios:', [s['scenario'] for s in BACKTEST_SCENARIOS])
        """
    ),
    md("## 4. Data Availability Check"),
    code(
        r"""
        import json
        import pandas as pd
        import shutil

        repo_market_dir = REPO_DIR / 'data' / 'raw' / 'market'
        repo_market_dir.mkdir(parents=True, exist_ok=True)

        if not GDRIVE_DATA_DIR.exists():
            raise FileNotFoundError(f'Drive data folder not found: {GDRIVE_DATA_DIR}')

        for window in TEMPORAL_WINDOWS:
            src = GDRIVE_DATA_DIR / window['data_filename']
            dst = repo_market_dir / window['data_filename']
            if not dst.exists():
                if not src.exists():
                    raise FileNotFoundError(f'Missing required data file in Drive: {src}')
                shutil.copy2(src, dst)
            preview = pd.read_csv(dst, usecols=['dt', 'kdcode'])
            preview['dt'] = pd.to_datetime(preview['dt'])
            window['repo_data_path'] = dst.relative_to(REPO_DIR).as_posix()
            print(
                f"{window['test_year']}: {dst.name} | "
                f"rows={len(preview):,}, stocks={preview.kdcode.nunique():,}, "
                f"dates={preview.dt.min().date()} to {preview.dt.max().date()}"
            )

        if REGIME_INPUTS_CSV:
            regime_path = REPO_DIR / REGIME_INPUTS_CSV
            if not regime_path.exists():
                candidate = GDRIVE_DATA_DIR / Path(REGIME_INPUTS_CSV).name
                if candidate.exists():
                    regime_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, regime_path)
                else:
                    raise FileNotFoundError(f'REGIME_INPUTS_CSV not found in repo or Drive: {REGIME_INPUTS_CSV}')
            print('Using regime CSV:', regime_path)
        elif REGIME_STRICT and not os.environ.get('FRED_API_KEY'):
            raise RuntimeError('REGIME_STRICT=True and REGIME_INPUTS_CSV is blank, so set FRED_API_KEY before running.')
        """
    ),
    md("## 5. Matrix Definition"),
    code(
        r"""
        import itertools
        import re

        BASE_OVERRIDES = [
            'features=with_momentum',
            'data.source=csv',
            'tracking.enabled=true',
            'tracking.log_predictions=false',
            f'training.num_models={NUM_MODELS}',
            f'training.num_epochs={NUM_EPOCHS}',
            f'training.early_stopping_patience={EARLY_STOPPING_PATIENCE}',
            f'training.batch_size={BATCH_SIZE}',
            f'training.learning_rate={LEARNING_RATE}',
            f'evaluation.bootstrap_resamples={BOOTSTRAP_RESAMPLES}',
            'features.include_momentum=true',
            'features.include_weekly_momentum=true',
            'features.momentum_encoding=binary',
            'features.momentum_blend_mode=static',
            'features.momentum_blend_fast_weight=0.5',
            'features.include_global_regime=true',
            f'features.regime_strict={str(REGIME_STRICT).lower()}',
            f'features.regime_enforce_lag_days={REGIME_ENFORCE_LAG_DAYS}',
            'features.regime_include_subsequent_returns=false',
            'features.regime_change_months=12',
            'features.regime_norm_months=120',
            'features.regime_exclusion_months=1',
            'features.regime_similarity_quantile=0.2',
            'features.regime_min_history_months=24',
            'graph.judge_value=0.8',
            'graph.update_frequency_months=0',
            'graph.corr_lookback_days=252',
            'graph.top_k=0',
            'graph.top_k_metric=corr',
            'graph.use_multi_feature_edges=true',
            'graph.append_snapshot_age_days=false',
            'graph.use_lead_lag_features=false',
            'training.shuffle_train=true',
            'training.loss_type=ic',
            'training.label_type=returns',
            'training.selection_metric=val_ic',
            'model.label_t=5',
            'graph.drop_edge_p=0.1',
            'seed=42',
        ]
        if REGIME_INPUTS_CSV:
            BASE_OVERRIDES.append(f'features.regime_inputs_csv={REGIME_INPUTS_CSV}')

        def safe_name(value: str, max_len: int = 110) -> str:
            cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
            if len(cleaned) <= max_len:
                return cleaned
            return cleaned[:max_len]

        TRAINING_JOBS = []
        for window in TEMPORAL_WINDOWS:
            name = safe_name(f"{MODEL_RECIPE['name']}__test-{window['test_year']}")
            overrides = [
                *BASE_OVERRIDES,
                f"data={window['data_config']}",
                f"data.filename={window['repo_data_path']}",
                f"data.train_start={window['train_start']}",
                f"data.train_end={window['train_end']}",
                f"data.val_start={window['val_start']}",
                f"data.val_end={window['val_end']}",
                f"data.test_start={window['test_start']}",
                f"data.test_end={window['test_end']}",
                f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                f"experiment_name={name}",
            ]
            TRAINING_JOBS.append({**window, 'name': name, 'overrides': overrides})

        matrix_df = pd.DataFrame(TRAINING_JOBS)
        display(matrix_df[['test_year', 'name', 'data_config', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']])
        """
    ),
    md("## 6. Run, Collect, And Score Helpers"),
    code(
        r"""
        import hashlib
        import time
        from pathlib import Path

        def is_timestamp_dir(path: Path) -> bool:
            return path.is_dir() and bool(re.fullmatch(r'\d{8}_\d{6}', path.name))

        def latest_run_dir(experiment_name: str) -> Path | None:
            base = TRAINING_OUTPUT_DIR / experiment_name
            if not base.exists():
                return None
            candidates = sorted([p for p in base.iterdir() if is_timestamp_dir(p)])
            return candidates[-1] if candidates else None

        def flatten_dict(value: dict, prefix: str = '') -> dict:
            out = {}
            for key, item in value.items():
                full_key = f'{prefix}.{key}' if prefix else str(key)
                if isinstance(item, dict):
                    out.update(flatten_dict(item, full_key))
                else:
                    out[full_key] = item
            return out

        def read_json(path: Path) -> dict:
            if not path.exists():
                return {}
            with open(path, encoding='utf-8') as f:
                return json.load(f)

        def run_training_job(job: dict) -> dict:
            run_log_dir = RUN_ROOT / 'logs' / job['name']
            run_log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_log_dir / 'stdout.log'
            stderr_path = run_log_dir / 'stderr.log'
            cmd = [sys.executable, '-u', str(REPO_DIR / 'run_experiment.py'), *job['overrides']]

            print('\n' + '=' * 100)
            print('Training:', job['name'])
            print('Test year:', job['test_year'])
            print('Command:', ' '.join(cmd[:4]), '... +', len(job['overrides']), 'overrides')
            start = time.time()
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            elapsed = (time.time() - start) / 60
            stdout_path.write_text(proc.stdout, encoding='utf-8')
            stderr_path.write_text(proc.stderr, encoding='utf-8')
            print(proc.stdout[-4000:])
            if proc.returncode != 0:
                print(proc.stderr[-4000:])

            run_dir = latest_run_dir(job['name'])
            row = {
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'elapsed_minutes': elapsed,
                'run_dir': str(run_dir) if run_dir else '',
                'predictions_dir': str(run_dir / 'averaged_predictions') if run_dir else '',
                'stdout_log': str(stdout_path),
                'stderr_log': str(stderr_path),
                **{k: job[k] for k in ['test_year', 'name', 'data_config', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']},
                **{f"recipe.{k}": v for k, v in MODEL_RECIPE.items()},
            }
            if run_dir:
                row.update({f'training_summary.{k}': v for k, v in flatten_dict(read_json(run_dir / 'training_summary.json')).items()})
                row.update({f'run_metadata.{k}': v for k, v in flatten_dict(read_json(run_dir / 'run_metadata.json')).items()})
            return row

        def run_backtest(training_row: pd.Series, scenario: dict, num_tests: int) -> dict:
            pred_dir = Path(training_row['predictions_dir'])
            suffix = '_' + scenario['scenario']
            cmd = [
                sys.executable,
                str(REPO_DIR / 'tests' / 'backtest_sp500.py'),
                '--predictions_dir', str(pred_dir),
                '--data_file', str(REPO_DIR / 'data' / 'raw' / 'market' / training_row['data_filename']),
                '--test_start', training_row['test_start'],
                '--test_end', training_row['test_end'],
                '--top_k', str(scenario['top_k']),
                '--label_t', str(MODEL_RECIPE['label_t']),
                '--holding_period', str(scenario['holding_period']),
                '--rebalance_style', scenario['rebalance_style'],
                '--num_tests', str(num_tests),
                '--adjustment_method', 'bhy',
                '--auto_save',
                '--plot',
                '--disable_mlflow_autolink',
                '--backtest_suffix', suffix,
            ]
            if scenario['transaction_costs']:
                cmd.extend(['--transaction_costs', '--spread', str(scenario['spread_bps']), '--slippage', str(scenario['slippage_bps'])])
            if scenario['rank_drop_gate']:
                cmd.extend(['--enable_rank_drop_gate', '--min_rank_drop', str(scenario['min_rank_drop'])])

            print('\n' + '-' * 100)
            print('Backtest:', training_row['name'], scenario['scenario'])
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            print(proc.stdout[-3500:])
            if proc.returncode != 0:
                print(proc.stderr[-3500:])

            source_dir = pred_dir.parent / f"backtest_{scenario['scenario']}"
            copy_dir = RUN_ROOT / 'backtests' / str(training_row['test_year']) / scenario['scenario']
            if source_dir.exists():
                if copy_dir.exists():
                    shutil.rmtree(copy_dir)
                shutil.copytree(source_dir, copy_dir)

            row = {
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'test_year': training_row['test_year'],
                'name': training_row['name'],
                'scenario': scenario['scenario'],
                'predictions_dir': str(pred_dir),
                'source_backtest_dir': str(source_dir),
                'copied_backtest_dir': str(copy_dir),
                'stdout_tail': proc.stdout[-5000:],
                'stderr_tail': proc.stderr[-5000:],
                **{f'scenario_config.{k}': v for k, v in scenario.items()},
            }
            metrics = read_json(source_dir / 'backtest_metrics.json')
            row.update({f'backtest.{k}': v for k, v in metrics.items()})
            result_csv = source_dir / 'backtest_results.csv'
            if result_csv.exists():
                result_df = pd.read_csv(result_csv)
                if len(result_df):
                    for key, value in result_df.iloc[0].to_dict().items():
                        row.setdefault(f'backtest.{key}', value)
            return row

        def add_decision_score(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            score = pd.Series(0.0, index=out.index)
            weights = {
                'backtest.ASR': 0.35,
                'backtest.excess_return': 0.30,
                'backtest.MDD': -0.20,
                'backtest.avg_daily_turnover': -0.15,
            }
            for col, weight in weights.items():
                if col in out.columns:
                    vals = pd.to_numeric(out[col], errors='coerce')
                    denom = vals.std(skipna=True)
                    if pd.notna(denom) and denom != 0:
                        score += weight * ((vals - vals.mean(skipna=True)) / denom).fillna(0.0)
            out['decision_score'] = score
            return out
        """
    ),
    md("## 7. Matrix Execution"),
    code(
        r"""
        manifest_path = RUN_ROOT / 'rolling_temporal_backtest_manifest.json'
        manifest_path.write_text(json.dumps({
            'run_tag': RUN_TAG,
            'drive_data_dir': str(GDRIVE_DATA_DIR),
            'run_root': str(RUN_ROOT),
            'model_recipe': MODEL_RECIPE,
            'temporal_windows': TEMPORAL_WINDOWS,
            'backtest_scenarios': BACKTEST_SCENARIOS,
            'base_overrides': BASE_OVERRIDES,
            'budget': {
                'num_models': NUM_MODELS,
                'num_epochs': NUM_EPOCHS,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
            },
        }, indent=2), encoding='utf-8')

        training_rows = []
        if RUN_TRAINING:
            for job in TRAINING_JOBS:
                training_rows.append(run_training_job(job))
                pd.DataFrame(training_rows).to_csv(RUN_ROOT / 'training_results_interim.csv', index=False)
        else:
            for job in TRAINING_JOBS:
                run_dir = latest_run_dir(job['name'])
                training_rows.append({
                    'status': 'OK' if run_dir else 'FAILED',
                    'returncode': 0 if run_dir else 1,
                    'run_dir': str(run_dir) if run_dir else '',
                    'predictions_dir': str(run_dir / 'averaged_predictions') if run_dir else '',
                    **{k: job[k] for k in ['test_year', 'name', 'data_config', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']},
                })

        training_df = pd.DataFrame(training_rows)
        training_results_path = RUN_ROOT / 'training_results.csv'
        training_df.to_csv(training_results_path, index=False)
        display(training_df[[c for c in ['status', 'test_year', 'name', 'elapsed_minutes', 'run_dir', 'predictions_dir'] if c in training_df.columns]])

        backtest_rows = []
        if RUN_BACKTESTS:
            ok_training = training_df[training_df['status'].eq('OK')].copy()
            num_tests = NUM_TESTS_OVERRIDE or (len(ok_training) * len(BACKTEST_SCENARIOS))
            for _, train_row in ok_training.iterrows():
                for scenario in BACKTEST_SCENARIOS:
                    backtest_rows.append(run_backtest(train_row, scenario, num_tests))
                    pd.DataFrame(backtest_rows).to_csv(RUN_ROOT / 'backtest_results_interim.csv', index=False)
        backtest_df = add_decision_score(pd.DataFrame(backtest_rows)) if backtest_rows else pd.DataFrame()

        raw_backtest_path = RUN_ROOT / 'backtest_results_raw.csv'
        decision_path = RUN_ROOT / 'backtest_decision_table.csv'
        html_path = RUN_ROOT / 'backtest_decision_table.html'
        backtest_df.to_csv(raw_backtest_path, index=False)
        if not backtest_df.empty and 'status' in backtest_df.columns:
            backtest_df['_status_rank'] = backtest_df['status'].map({'OK': 0, 'FAILED': 1}).fillna(2)
        sort_cols = [c for c in ['_status_rank', 'decision_score', 'backtest.ASR'] if c in backtest_df.columns]
        decision_df = backtest_df.sort_values(sort_cols, ascending=[True, False, False][:len(sort_cols)]) if sort_cols else backtest_df
        if '_status_rank' in decision_df.columns:
            decision_df = decision_df.drop(columns=['_status_rank'])
        decision_df.to_csv(decision_path, index=False)
        decision_df.to_html(html_path, index=False)

        display_cols = [c for c in [
            'status', 'test_year', 'scenario', 'decision_score', 'backtest.ARR', 'backtest.ASR',
            'backtest.MDD', 'backtest.total_return_calendar_aligned', 'backtest.benchmark_return',
            'backtest.excess_return', 'backtest.avg_daily_turnover', 'backtest.haircutted_sharpe',
            'backtest.adjusted_p_value', 'copied_backtest_dir'
        ] if c in decision_df.columns]
        display(decision_df[display_cols])
        print('Manifest:', manifest_path)
        print('Training results:', training_results_path)
        print('Decision table:', decision_path)
        """
    ),
    md("## 8. Main-Effect And Interaction Analysis"),
    code(
        r"""
        effect_dir = RUN_ROOT / 'effects'
        effect_dir.mkdir(exist_ok=True)
        ok = decision_df[decision_df['status'].eq('OK')].copy() if not decision_df.empty else pd.DataFrame()
        metric_cols = [c for c in [
            'decision_score',
            'backtest.ARR',
            'backtest.ASR',
            'backtest.MDD',
            'backtest.excess_return',
            'backtest.avg_daily_turnover',
            'backtest.haircutted_sharpe',
        ] if c in ok.columns]

        effect_tables = {}
        for factor in ['test_year', 'scenario']:
            if not ok.empty and factor in ok.columns and metric_cols:
                table = ok.groupby(factor, dropna=False)[metric_cols].agg(['mean', 'median', 'count'])
                effect_tables[factor] = table
                table.to_csv(effect_dir / f'main_effect_{factor}.csv')
                display(table)

        if not ok.empty and {'test_year', 'scenario'}.issubset(ok.columns) and 'decision_score' in ok.columns:
            interaction = ok.pivot_table(index='test_year', columns='scenario', values='decision_score', aggfunc='mean')
            interaction.to_csv(effect_dir / 'interaction_test_year_by_scenario.csv')
            display(interaction)
        """
    ),
    md("## 9. Visualization"),
    code(
        r"""
        import matplotlib.pyplot as plt
        import numpy as np

        plot_dir = RUN_ROOT / 'plots'
        plot_dir.mkdir(exist_ok=True)

        ok = decision_df[decision_df['status'].eq('OK')].copy() if not decision_df.empty else pd.DataFrame()
        if ok.empty:
            print('No successful backtests to plot.')
        else:
            plot_metrics = [(c, t) for c, t in [
                ('backtest.ASR', 'Annualized Sharpe'),
                ('backtest.excess_return', 'Excess Return'),
                ('backtest.MDD', 'Maximum Drawdown'),
                ('backtest.avg_daily_turnover', 'Average Daily Turnover'),
            ] if c in ok.columns]
            fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(12, max(4, 3.6 * len(plot_metrics))))
            if len(plot_metrics) == 1:
                axes = [axes]
            for ax, (col, title) in zip(axes, plot_metrics):
                plot_df = ok.sort_values(['test_year', 'scenario']).copy()
                plot_df['label'] = plot_df['test_year'].astype(str) + '\n' + plot_df['scenario']
                ax.barh(plot_df['label'], pd.to_numeric(plot_df[col], errors='coerce'), color='#2f6f73')
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_title(title)
                ax.grid(True, axis='x', alpha=0.25)
            plt.tight_layout()
            metric_plot_path = plot_dir / 'rolling_backtest_metric_bars.png'
            plt.savefig(metric_plot_path, dpi=160, bbox_inches='tight')
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(13, 6))
            for _, row in ok.iterrows():
                returns_file = Path(row['source_backtest_dir']) / 'daily_returns.csv'
                if not returns_file.exists():
                    continue
                returns_df = pd.read_csv(returns_file)
                returns_df['date'] = pd.to_datetime(returns_df['date'])
                values = np.cumprod(1 + returns_df['portfolio_return'].astype(float))
                ax.plot(returns_df['date'], values, label=f"{row['test_year']} {row['scenario']}", linewidth=1.4)
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0)
            ax.set_title('Rolling Temporal Equity Curves')
            ax.set_ylabel('Cumulative value')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best', fontsize=8)
            plt.tight_layout()
            equity_plot_path = plot_dir / 'rolling_equity_curves.png'
            plt.savefig(equity_plot_path, dpi=160, bbox_inches='tight')
            plt.show()
            print('Metric bars:', metric_plot_path)
            print('Equity curves:', equity_plot_path)
        """
    ),
    md("## 10. Failed-Run Inspection"),
    code(
        r"""
        def print_tail(path: str, n_chars: int = 4000):
            p = Path(path)
            if p.exists():
                text = p.read_text(encoding='utf-8', errors='replace')
                print(f'\n--- {p} ---')
                print(text[-n_chars:])

        failed_training = training_df[~training_df['status'].eq('OK')] if not training_df.empty else pd.DataFrame()
        if failed_training.empty:
            print('No failed training runs.')
        else:
            display(failed_training)
            for _, row in failed_training.iterrows():
                print_tail(row.get('stdout_log', ''))
                print_tail(row.get('stderr_log', ''))

        failed_backtests = decision_df[~decision_df['status'].eq('OK')] if not decision_df.empty else pd.DataFrame()
        if failed_backtests.empty:
            print('No failed backtests.')
        else:
            display(failed_backtests[['test_year', 'scenario', 'stdout_tail', 'stderr_tail']])
        """
    ),
    md("## 11. Summary Report Export"),
    code(
        r"""
        def fmt_pct(value):
            if pd.isna(value):
                return ''
            return f'{100 * float(value):.2f}%'

        summary_cols = [c for c in [
            'status', 'test_year', 'scenario', 'decision_score', 'backtest.ARR', 'backtest.ASR',
            'backtest.MDD', 'backtest.total_return_calendar_aligned', 'backtest.benchmark_return',
            'backtest.excess_return', 'backtest.avg_daily_turnover', 'backtest.haircutted_sharpe',
            'backtest.adjusted_p_value', 'copied_backtest_dir'
        ] if c in decision_df.columns]
        report_table = decision_df[summary_cols].copy() if summary_cols else pd.DataFrame()
        for pct_col in [
            'backtest.ARR', 'backtest.MDD', 'backtest.total_return_calendar_aligned',
            'backtest.benchmark_return', 'backtest.excess_return', 'backtest.avg_daily_turnover'
        ]:
            if pct_col in report_table.columns:
                report_table[pct_col] = report_table[pct_col].apply(fmt_pct)

        report_path = RUN_ROOT / 'rolling_temporal_backtest_summary.md'
        lines = [
            '# MCI-GRU Rolling Temporal Backtest Summary',
            '',
            f'Run root: `{RUN_ROOT}`',
            f'Drive data folder: `{GDRIVE_DATA_DIR}`',
            f'Model recipe: `{MODEL_RECIPE["name"]}`',
            '',
            '## Temporal Windows',
            '',
            pd.DataFrame(TEMPORAL_WINDOWS)[['test_year', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']].to_markdown(index=False),
            '',
            '## Backtest Results',
            '',
            report_table.to_markdown(index=False) if not report_table.empty else 'No backtest rows were produced.',
            '',
            '## Interpretation Notes',
            '',
            '- The recipe is fixed across all years; do not re-rank or retune by year when interpreting this as untouched-year evidence.',
            '- Each year uses the older fixed S&P 500 universe CSV from `MCI_GRU_shared/data`, copied into `data/raw/market/` before training.',
            '- Backtests use `tests/backtest_sp500.py` with open-to-open portfolio and benchmark returns, transaction costs, BHY haircut, and the rank-drop gate.',
            '- Haircutted Sharpe can be zero even when raw realized return is positive if the multiple-testing adjustment pushes adjusted p-value to 1.0.',
            '',
            '## Artifacts',
            '',
            f'- Manifest: `{manifest_path}`',
            f'- Training results: `{training_results_path}`',
            f'- Raw backtest results: `{raw_backtest_path}`',
            f'- Decision table: `{decision_path}`',
            f'- HTML decision table: `{html_path}`',
        ]
        if 'metric_plot_path' in globals():
            lines.append(f'- Metric bars: `{metric_plot_path}`')
        if 'equity_plot_path' in globals():
            lines.append(f'- Equity curves: `{equity_plot_path}`')

        report_path.write_text('\n'.join(lines), encoding='utf-8')
        print(report_path.read_text(encoding='utf-8')[:8000])
        print('Summary report:', report_path)
        """
    ),
    md("## 12. Zip Results"),
    code(
        r"""
        archive_path = shutil.make_archive(str(RUN_ROOT), 'zip', root_dir=str(RUN_ROOT.parent), base_dir=RUN_ROOT.name)
        print('Archive:', archive_path)
        """
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
