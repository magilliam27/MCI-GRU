"""Generate the Colab notebook for proving MCI-GRU performance robustness."""

from __future__ import annotations

from pathlib import Path

from nb_lib import LOCAL_PY310_METADATA, write_notebook
from nb_lib import code_lines as code
from nb_lib import md_lines as md

OUT = Path("notebooks/performance_proof_tests_colab.ipynb")


cells = [
    md(
        """
        # MCI-GRU Performance Proof Tests

        This notebook stress-tests the frozen ensemble recipe after the initial multi-year evidence. It is meant to prove robustness, not search for a new winner.

        It can:

        - retrain the frozen recipe across multiple base seeds;
        - compare full-model, no-regime, and near-zero graph baselines;
        - backtest top-k construction and transaction-cost stress grids;
        - import an already-run 2025 result folder;
        - produce pooled daily excess-return significance across all available years.
        """
    ),
    md("## 1. Mount Drive, Clone Repo, Install Dependencies"),
    code(
        r"""
        from pathlib import Path
        import os
        import shutil
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
        DRIVE_DATA_DIR = Path('/content/drive/MyDrive/MCI_GRU_shared/data') if IN_COLAB else Path.cwd() / 'data' / 'raw' / 'market'

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
        print('Drive output root:', DRIVE_ROOT)
        print('Drive data folder:', DRIVE_DATA_DIR)

        if IN_COLAB:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip', 'setuptools', 'wheel'], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[dev,tracking,fred]'], check=True)

        REQUIRE_GPU = True
        import torch
        print('Python:', sys.executable)
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
        # Paste a FRED API key here, or set FRED_API_KEY in the notebook environment.
        # Required for strict regime runs when REGIME_INPUTS_CSV is blank.
        MY_FRED_KEY = ''

        if MY_FRED_KEY.strip():
            os.environ['FRED_API_KEY'] = MY_FRED_KEY.strip()

        print('FRED_API_KEY is set:', bool(os.environ.get('FRED_API_KEY')))
        """
    ),
    md("## 3. Proof Matrix Configuration"),
    code(
        r"""
        from datetime import datetime

        RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
        EXPERIMENT_SLUG = 'performance_proof_tests'
        RUN_ROOT = DRIVE_ROOT / EXPERIMENT_SLUG / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / 'training_runs'
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Set to False if you only want to import existing result folders and summarize them.
        RUN_TRAINING = True
        RUN_BACKTESTS = True

        # Budget knobs. A full matrix is expensive: num windows * variants * base seeds * 20 models.
        NUM_MODELS = 20
        NUM_EPOCHS = 100
        EARLY_STOPPING_PATIENCE = 15
        BATCH_SIZE = 32
        LEARNING_RATE = '5e-5'
        BOOTSTRAP_RESAMPLES = 1000

        # The ensemble uses base_seed + model_id internally. These test base-seed robustness at the ensemble level.
        BASE_SEEDS = [1729, 2718, 3141]

        # Keep 2025 as an import by default because it has already been run elsewhere.
        # Add exact folders that contain backtest_decision_table.csv or backtest_results_raw.csv.
        IMPORT_EXISTING_RESULT_FOLDERS = [
            # Example:
            # '/content/drive/MyDrive/MCI-GRU-Ablations/recommended_backtests/20260501_235210',
        ]

        REGIME_INPUTS_CSV = ''
        REGIME_STRICT = True
        REGIME_ENFORCE_LAG_DAYS = 0

        MODEL_RECIPE = {
            'name': 'static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1',
            'label_t': 5,
            'loss_type': 'ic',
            'label_type': 'returns',
            'selection_metric': 'val_ic',
            'drop_edge_p': 0.1,
        }

        # Known fixed-universe windows already used for 2022-2024.
        # 2025 can be added as a training window by setting enabled=True, but defaults to import.
        PROOF_WINDOWS = [
            {
                'enabled': True,
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
                'enabled': True,
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
                'enabled': True,
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
            {
                'enabled': False,
                'test_year': 2025,
                'data_config': 'temporal_2019',
                'data_filename': 'sp500_2019_universe_data_through_2026.csv',
                'train_start': '2019-01-01',
                'train_end': '2023-12-31',
                'val_start': '2024-01-22',
                'val_end': '2024-12-31',
                'test_start': '2025-01-22',
                'test_end': '2025-12-31',
            },
        ]

        MODEL_VARIANTS = [
            {
                'enabled': True,
                'name': 'full',
                'description': 'Frozen candidate recipe: momentum + regime + static threshold graph.',
                'overrides': [],
            },
            {
                'enabled': True,
                'name': 'no_regime',
                'description': 'Same recipe with global regime features removed.',
                'overrides': [
                    'features.include_global_regime=false',
                    'features.regime_strict=false',
                    'features.regime_include_subsequent_returns=false',
                ],
            },
            {
                'enabled': True,
                'name': 'near_zero_graph',
                'description': 'Same recipe with a very high correlation threshold. If empty edges are unsupported, this baseline should fail visibly.',
                'overrides': [
                    'graph.judge_value=0.9999',
                    'graph.top_k=0',
                ],
            },
        ]

        TOP_K_VALUES = [5, 10, 15, 20, 30]
        COST_STRESS_GRID = [
            {'cost_name': 'spread5_slip0', 'spread_bps': 5.0, 'slippage_bps': 0.0},
            {'cost_name': 'spread10_slip2', 'spread_bps': 10.0, 'slippage_bps': 2.0},
            {'cost_name': 'spread20_slip5', 'spread_bps': 20.0, 'slippage_bps': 5.0},
        ]
        RANK_DROP_GATE = {'enabled': True, 'min_rank_drop': 30}
        HOLDING_PERIOD = 1
        REBALANCE_STYLE = 'staggered'
        NUM_TESTS_OVERRIDE = None

        print('Run root:', RUN_ROOT)
        print('Enabled windows:', [w['test_year'] for w in PROOF_WINDOWS if w['enabled']])
        print('Base seeds:', BASE_SEEDS)
        print('Enabled variants:', [v['name'] for v in MODEL_VARIANTS if v['enabled']])
        print('Top-k values:', TOP_K_VALUES)
        print('Cost stress grid:', COST_STRESS_GRID)
        """
    ),
    md("## 4. Data Availability Check"),
    code(
        r"""
        import json
        import pandas as pd

        repo_market_dir = REPO_DIR / 'data' / 'raw' / 'market'
        repo_market_dir.mkdir(parents=True, exist_ok=True)

        if not DRIVE_DATA_DIR.exists():
            raise FileNotFoundError(f'Drive data folder not found: {DRIVE_DATA_DIR}')

        for window in [w for w in PROOF_WINDOWS if w['enabled']]:
            src = DRIVE_DATA_DIR / window['data_filename']
            dst = repo_market_dir / window['data_filename']
            if not dst.exists():
                if not src.exists():
                    raise FileNotFoundError(f'Missing required data file in Drive: {src}')
                shutil.copy2(src, dst)
            preview = pd.read_csv(dst, usecols=['dt', 'kdcode'])
            preview['dt'] = pd.to_datetime(preview['dt'])
            print(
                f"{window['test_year']}: {dst.name} | "
                f"rows={len(preview):,}, stocks={preview.kdcode.nunique():,}, "
                f"dates={preview.dt.min().date()} to {preview.dt.max().date()}"
            )

        if REGIME_INPUTS_CSV:
            regime_path = REPO_DIR / REGIME_INPUTS_CSV
            if not regime_path.exists():
                candidate = DRIVE_DATA_DIR / Path(REGIME_INPUTS_CSV).name
                if candidate.exists():
                    regime_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, regime_path)
                else:
                    raise FileNotFoundError(f'REGIME_INPUTS_CSV not found in repo or Drive: {REGIME_INPUTS_CSV}')
            print('Using regime CSV:', regime_path)
        elif REGIME_STRICT and any(v['name'] == 'full' and v['enabled'] for v in MODEL_VARIANTS) and not os.environ.get('FRED_API_KEY'):
            raise RuntimeError('REGIME_STRICT=True and REGIME_INPUTS_CSV is blank, so set FRED_API_KEY before running full-regime training.')
        """
    ),
    md("## 5. Build Training And Backtest Matrices"),
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
            f"training.loss_type={MODEL_RECIPE['loss_type']}",
            f"training.label_type={MODEL_RECIPE['label_type']}",
            f"training.selection_metric={MODEL_RECIPE['selection_metric']}",
            f"model.label_t={MODEL_RECIPE['label_t']}",
            f"graph.drop_edge_p={MODEL_RECIPE['drop_edge_p']}",
        ]
        if REGIME_INPUTS_CSV:
            BASE_OVERRIDES.append(f'features.regime_inputs_csv={REGIME_INPUTS_CSV}')

        def safe_name(value: str, max_len: int = 120) -> str:
            cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
            if len(cleaned) <= max_len:
                return cleaned
            return cleaned[:max_len]

        def repo_data_path_for(window: dict) -> str:
            repo_path = REPO_DIR / 'data' / 'raw' / 'market' / window['data_filename']
            if not repo_path.exists():
                raise FileNotFoundError(f"Run the data check first; missing {repo_path}")
            return repo_path.relative_to(REPO_DIR).as_posix()

        training_jobs = []
        for window in [w for w in PROOF_WINDOWS if w['enabled']]:
            repo_data_path = repo_data_path_for(window)
            for variant in [v for v in MODEL_VARIANTS if v['enabled']]:
                for base_seed in BASE_SEEDS:
                    name = safe_name(f"{MODEL_RECIPE['name']}__{variant['name']}__base-seed-{base_seed}__test-{window['test_year']}")
                    overrides = [
                        *BASE_OVERRIDES,
                        *variant['overrides'],
                        f"seed={base_seed}",
                        f"data={window['data_config']}",
                        f"data.filename={repo_data_path}",
                        f"data.train_start={window['train_start']}",
                        f"data.train_end={window['train_end']}",
                        f"data.val_start={window['val_start']}",
                        f"data.val_end={window['val_end']}",
                        f"data.test_start={window['test_start']}",
                        f"data.test_end={window['test_end']}",
                        f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                        f"experiment_name={name}",
                    ]
                    training_jobs.append({
                        **{k: window[k] for k in ['test_year', 'data_config', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']},
                        'repo_data_path': repo_data_path,
                        'variant': variant['name'],
                        'variant_description': variant['description'],
                        'base_seed': base_seed,
                        'name': name,
                        'overrides': overrides,
                    })

        backtest_scenarios = []
        for top_k, cost in itertools.product(TOP_K_VALUES, COST_STRESS_GRID):
            scenario = {
                'scenario': f"k{top_k}_{cost['cost_name']}_rankdrop{RANK_DROP_GATE['min_rank_drop']}_daily",
                'top_k': top_k,
                'transaction_costs': True,
                'spread_bps': cost['spread_bps'],
                'slippage_bps': cost['slippage_bps'],
                'rank_drop_gate': RANK_DROP_GATE['enabled'],
                'min_rank_drop': RANK_DROP_GATE['min_rank_drop'],
                'holding_period': HOLDING_PERIOD,
                'rebalance_style': REBALANCE_STYLE,
            }
            backtest_scenarios.append(scenario)

        training_matrix_df = pd.DataFrame(training_jobs)
        scenario_df = pd.DataFrame(backtest_scenarios)

        display(training_matrix_df[['test_year', 'variant', 'base_seed', 'name', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']])
        display(scenario_df)
        print('Training jobs:', len(training_jobs))
        print('Backtests after training:', len(training_jobs) * len(backtest_scenarios))
        """
    ),
    md("## 6. Execution Helpers"),
    code(
        r"""
        import math
        import time

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

            print('\n' + '=' * 110)
            print('Training:', job['name'])
            print('Variant:', job['variant'], '| Base seed:', job['base_seed'], '| Test year:', job['test_year'])
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
                **{k: job[k] for k in ['test_year', 'variant', 'base_seed', 'name', 'data_config', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']},
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

            print('\n' + '-' * 110)
            print('Backtest:', training_row['name'], '|', scenario['scenario'])
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            print(proc.stdout[-3000:])
            if proc.returncode != 0:
                print(proc.stderr[-3000:])

            source_dir = pred_dir.parent / f"backtest_{scenario['scenario']}"
            copy_dir = RUN_ROOT / 'backtests' / str(training_row['test_year']) / training_row['variant'] / f"base_seed_{training_row['base_seed']}" / scenario['scenario']
            if source_dir.exists():
                if copy_dir.exists():
                    shutil.rmtree(copy_dir)
                shutil.copytree(source_dir, copy_dir)

            row = {
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'test_year': training_row['test_year'],
                'variant': training_row['variant'],
                'base_seed': training_row['base_seed'],
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
        """
    ),
    md("## 7. Run Training And Backtests"),
    code(
        r"""
        manifest_path = RUN_ROOT / 'performance_proof_manifest.json'
        manifest_path.write_text(json.dumps({
            'run_tag': RUN_TAG,
            'run_root': str(RUN_ROOT),
            'drive_data_dir': str(DRIVE_DATA_DIR),
            'model_recipe': MODEL_RECIPE,
            'base_seeds': BASE_SEEDS,
            'proof_windows': PROOF_WINDOWS,
            'model_variants': MODEL_VARIANTS,
            'top_k_values': TOP_K_VALUES,
            'cost_stress_grid': COST_STRESS_GRID,
            'rank_drop_gate': RANK_DROP_GATE,
            'import_existing_result_folders': IMPORT_EXISTING_RESULT_FOLDERS,
        }, indent=2), encoding='utf-8')

        training_rows = []
        if RUN_TRAINING:
            for job in training_jobs:
                training_rows.append(run_training_job(job))
                pd.DataFrame(training_rows).to_csv(RUN_ROOT / 'training_results_interim.csv', index=False)
        else:
            for job in training_jobs:
                run_dir = latest_run_dir(job['name'])
                training_rows.append({
                    'status': 'OK' if run_dir else 'FAILED',
                    'returncode': 0 if run_dir else 1,
                    'run_dir': str(run_dir) if run_dir else '',
                    'predictions_dir': str(run_dir / 'averaged_predictions') if run_dir else '',
                    **{k: job[k] for k in ['test_year', 'variant', 'base_seed', 'name', 'data_config', 'data_filename', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']},
                })

        training_df = pd.DataFrame(training_rows)
        training_results_path = RUN_ROOT / 'training_results.csv'
        training_df.to_csv(training_results_path, index=False)
        display(training_df[[c for c in ['status', 'test_year', 'variant', 'base_seed', 'elapsed_minutes', 'run_dir', 'predictions_dir'] if c in training_df.columns]])

        backtest_rows = []
        if RUN_BACKTESTS:
            ok_training = training_df[training_df['status'].eq('OK')].copy()
            num_tests = NUM_TESTS_OVERRIDE or (len(ok_training) * len(backtest_scenarios) + len(IMPORT_EXISTING_RESULT_FOLDERS))
            for _, train_row in ok_training.iterrows():
                for scenario in backtest_scenarios:
                    backtest_rows.append(run_backtest(train_row, scenario, num_tests))
                    pd.DataFrame(backtest_rows).to_csv(RUN_ROOT / 'backtest_results_interim.csv', index=False)
        else:
            num_tests = NUM_TESTS_OVERRIDE or 1

        backtest_df = pd.DataFrame(backtest_rows)
        raw_backtest_path = RUN_ROOT / 'backtest_results_raw.csv'
        backtest_df.to_csv(raw_backtest_path, index=False)
        display(backtest_df.head())
        """
    ),
    md("## 8. Import Existing Result Folders"),
    code(
        r"""
        imported_rows = []

        def load_existing_result_folder(folder: Path) -> pd.DataFrame:
            candidates = [
                folder / 'backtest_decision_table.csv',
                folder / 'backtest_results_raw.csv',
                folder / 'backtest_comparison.csv',
            ]
            for path in candidates:
                if path.exists():
                    df = pd.read_csv(path)
                    df['imported_result_folder'] = str(folder)
                    df['source_table'] = str(path)
                    if 'test_year' not in df.columns:
                        df['test_year'] = 2025
                    if 'variant' not in df.columns:
                        df['variant'] = 'imported'
                    if 'base_seed' not in df.columns:
                        df['base_seed'] = 'imported'
                    return df
            raise FileNotFoundError(f'No known result CSV found in {folder}')

        for folder_text in IMPORT_EXISTING_RESULT_FOLDERS:
            folder = Path(folder_text)
            try:
                imported_rows.append(load_existing_result_folder(folder))
                print('Imported:', folder)
            except Exception as exc:
                print('Failed to import:', folder, exc)

        imported_df = pd.concat(imported_rows, ignore_index=True) if imported_rows else pd.DataFrame()
        if imported_df.empty:
            print('No existing result folders imported. Add 2025 folders to IMPORT_EXISTING_RESULT_FOLDERS if you want pooled 2025 included.')
        else:
            imported_path = RUN_ROOT / 'imported_existing_results.csv'
            imported_df.to_csv(imported_path, index=False)
            display(imported_df.head())
            print('Imported results:', imported_path)
        """
    ),
    md("## 9. Decision Tables And Robustness Summaries"),
    code(
        r"""
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

        combined_parts = []
        if not backtest_df.empty:
            combined_parts.append(backtest_df)
        if not imported_df.empty:
            combined_parts.append(imported_df)
        combined_df = pd.concat(combined_parts, ignore_index=True, sort=False) if combined_parts else pd.DataFrame()
        if combined_df.empty:
            raise RuntimeError('No backtest or imported rows available.')

        if 'decision_score' not in combined_df.columns:
            combined_df = add_decision_score(combined_df)
        elif combined_df['decision_score'].isna().all():
            combined_df = add_decision_score(combined_df.drop(columns=['decision_score']))

        if 'status' in combined_df.columns:
            combined_df['_status_rank'] = combined_df['status'].map({'OK': 0, 'FAILED': 1}).fillna(2)
        else:
            combined_df['_status_rank'] = 0
        sort_cols = [c for c in ['_status_rank', 'decision_score', 'backtest.ASR', 'backtest.excess_return'] if c in combined_df.columns]
        combined_df = combined_df.sort_values(sort_cols, ascending=[True, False, False, False][:len(sort_cols)]).drop(columns=['_status_rank'])

        raw_combined_path = RUN_ROOT / 'combined_proof_results_raw.csv'
        decision_path = RUN_ROOT / 'combined_proof_decision_table.csv'
        html_path = RUN_ROOT / 'combined_proof_decision_table.html'
        combined_df.to_csv(raw_combined_path, index=False)
        combined_df.to_csv(decision_path, index=False)
        combined_df.to_html(html_path, index=False)

        display_cols = [c for c in [
            'status', 'test_year', 'variant', 'base_seed', 'scenario', 'decision_score',
            'backtest.ARR', 'backtest.ASR', 'backtest.MDD', 'backtest.total_return_calendar_aligned',
            'backtest.benchmark_return', 'backtest.excess_return', 'backtest.avg_daily_turnover',
            'backtest.haircutted_sharpe', 'backtest.adjusted_p_value', 'copied_backtest_dir',
            'imported_result_folder'
        ] if c in combined_df.columns]
        display(combined_df[display_cols])

        metric_cols = [c for c in [
            'decision_score', 'backtest.ARR', 'backtest.ASR', 'backtest.MDD',
            'backtest.excess_return', 'backtest.avg_daily_turnover', 'backtest.haircutted_sharpe',
        ] if c in combined_df.columns]
        ok = combined_df[combined_df['status'].eq('OK')].copy() if 'status' in combined_df.columns else combined_df.copy()

        summary_dir = RUN_ROOT / 'summaries'
        summary_dir.mkdir(exist_ok=True)
        for group_cols in [['variant'], ['base_seed'], ['test_year'], ['scenario'], ['variant', 'scenario'], ['variant', 'base_seed']]:
            present = [c for c in group_cols if c in ok.columns]
            if present and metric_cols:
                table = ok.groupby(present, dropna=False)[metric_cols].agg(['mean', 'median', 'std', 'count'])
                table.to_csv(summary_dir / ('summary_by_' + '_'.join(present) + '.csv'))
                display(table)
        print('Decision table:', decision_path)
        print('HTML table:', html_path)
        """
    ),
    md("## 10. Pooled Daily Excess-Return Significance"),
    code(
        r"""
        import numpy as np
        from scipy import stats

        def daily_returns_path(row: pd.Series) -> Path | None:
            for key in ['source_backtest_dir', 'copied_backtest_dir']:
                if key in row and pd.notna(row[key]):
                    path = Path(str(row[key])) / 'daily_returns.csv'
                    if path.exists():
                        return path
            return None

        pooled_rows = []
        for _, row in ok.iterrows():
            path = daily_returns_path(row)
            if path is None:
                continue
            daily = pd.read_csv(path)
            if 'portfolio_return' not in daily.columns or 'benchmark_return' not in daily.columns:
                continue
            daily['date'] = pd.to_datetime(daily['date'])
            daily['excess_return_daily'] = daily['portfolio_return'].astype(float) - daily['benchmark_return'].astype(float)
            for _, drow in daily.iterrows():
                pooled_rows.append({
                    'test_year': row.get('test_year'),
                    'variant': row.get('variant'),
                    'base_seed': row.get('base_seed'),
                    'scenario': row.get('scenario'),
                    'date': drow['date'],
                    'portfolio_return': drow['portfolio_return'],
                    'benchmark_return': drow['benchmark_return'],
                    'excess_return_daily': drow['excess_return_daily'],
                })

        pooled_daily_df = pd.DataFrame(pooled_rows)
        pooled_daily_path = RUN_ROOT / 'pooled_daily_returns.csv'
        pooled_daily_df.to_csv(pooled_daily_path, index=False)

        sig_rows = []
        if not pooled_daily_df.empty:
            group_cols = ['variant', 'scenario']
            for keys, group in pooled_daily_df.groupby(group_cols, dropna=False):
                vals = group['excess_return_daily'].dropna().astype(float)
                if len(vals) < 3:
                    continue
                mean_daily = vals.mean()
                t_stat, p_value = stats.ttest_1samp(vals, 0.0)
                ann_excess = (1 + mean_daily) ** 252 - 1
                ann_vol = vals.std(ddof=1) * np.sqrt(252)
                ann_ir = ann_excess / ann_vol if ann_vol else np.nan
                sig_rows.append({
                    'variant': keys[0],
                    'scenario': keys[1],
                    'n_days': len(vals),
                    'mean_daily_excess': mean_daily,
                    'annualized_excess': ann_excess,
                    'annualized_excess_vol': ann_vol,
                    'annualized_information_ratio': ann_ir,
                    't_statistic': t_stat,
                    'p_value_two_sided': p_value,
                })
        pooled_sig_df = pd.DataFrame(sig_rows)
        pooled_sig_path = RUN_ROOT / 'pooled_daily_significance.csv'
        pooled_sig_df.to_csv(pooled_sig_path, index=False)

        print('Pooled daily returns:', pooled_daily_path)
        print('Pooled significance:', pooled_sig_path)
        display(pooled_sig_df)
        """
    ),
    md("## 11. Visualizations"),
    code(
        r"""
        import matplotlib.pyplot as plt

        plot_dir = RUN_ROOT / 'plots'
        plot_dir.mkdir(exist_ok=True)

        plot_metrics = [(c, t) for c, t in [
            ('backtest.ASR', 'Annualized Sharpe'),
            ('backtest.excess_return', 'Excess Return'),
            ('backtest.MDD', 'Maximum Drawdown'),
            ('backtest.avg_daily_turnover', 'Average Daily Turnover'),
        ] if c in ok.columns]

        if plot_metrics:
            compact = ok.copy()
            compact['label'] = compact.get('test_year', '').astype(str) + ' | ' + compact.get('variant', '').astype(str) + ' | ' + compact.get('base_seed', '').astype(str) + ' | ' + compact.get('scenario', '').astype(str)
            compact = compact.head(80)
            fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(15, max(4, 4 * len(plot_metrics))))
            if len(plot_metrics) == 1:
                axes = [axes]
            for ax, (col, title) in zip(axes, plot_metrics):
                plot_df = compact.sort_values(col, ascending=True)
                ax.barh(plot_df['label'], pd.to_numeric(plot_df[col], errors='coerce'), color='#2f6f73')
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_title(title)
                ax.grid(True, axis='x', alpha=0.25)
            plt.tight_layout()
            metric_plot_path = plot_dir / 'proof_metric_bars.png'
            plt.savefig(metric_plot_path, dpi=160, bbox_inches='tight')
            plt.show()
            print('Metric bars:', metric_plot_path)

        if not pooled_daily_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(14, 7))
            for keys, group in pooled_daily_df.groupby(['variant', 'scenario'], dropna=False):
                if keys[1] != f"k10_spread5_slip0_rankdrop{RANK_DROP_GATE['min_rank_drop']}_daily":
                    continue
                group = group.sort_values('date')
                values = np.cumprod(1 + group['excess_return_daily'].astype(float))
                ax.plot(group['date'], values, label=f'{keys[0]} | {keys[1]}', linewidth=1.4)
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0)
            ax.set_title('Pooled Excess Return Curves - Primary Top-10 Low-Cost Scenario')
            ax.set_ylabel('Cumulative excess value')
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best', fontsize=8)
            plt.tight_layout()
            excess_plot_path = plot_dir / 'pooled_excess_curves_primary.png'
            plt.savefig(excess_plot_path, dpi=160, bbox_inches='tight')
            plt.show()
            print('Pooled excess curves:', excess_plot_path)
        """
    ),
    md("## 12. Summary Report Export"),
    code(
        r"""
        def fmt_pct(value):
            if pd.isna(value):
                return ''
            return f'{100 * float(value):.2f}%'

        report_table = combined_df[[c for c in [
            'status', 'test_year', 'variant', 'base_seed', 'scenario', 'decision_score',
            'backtest.ARR', 'backtest.ASR', 'backtest.MDD', 'backtest.total_return_calendar_aligned',
            'backtest.benchmark_return', 'backtest.excess_return', 'backtest.avg_daily_turnover',
            'backtest.haircutted_sharpe', 'backtest.adjusted_p_value'
        ] if c in combined_df.columns]].copy()
        for pct_col in [
            'backtest.ARR', 'backtest.MDD', 'backtest.total_return_calendar_aligned',
            'backtest.benchmark_return', 'backtest.excess_return', 'backtest.avg_daily_turnover'
        ]:
            if pct_col in report_table.columns:
                report_table[pct_col] = report_table[pct_col].apply(fmt_pct)

        report_path = RUN_ROOT / 'performance_proof_summary.md'
        lines = [
            '# MCI-GRU Performance Proof Summary',
            '',
            f'Run root: `{RUN_ROOT}`',
            f'Drive data folder: `{DRIVE_DATA_DIR}`',
            f'Model recipe: `{MODEL_RECIPE["name"]}`',
            f'Base seeds tested: `{BASE_SEEDS}`',
            f'Imported result folders: `{IMPORT_EXISTING_RESULT_FOLDERS}`',
            '',
            '## What This Tests',
            '',
            '- Base-seed robustness at the ensemble level. Each training run still uses independently seeded members via `base_seed + model_id`.',
            '- Portfolio construction robustness across top-k values.',
            '- Transaction-cost robustness across spread/slippage stress levels.',
            '- Baseline robustness against no-regime and near-zero correlation graph variants.',
            '- Pooled daily excess-return significance across every run with available `daily_returns.csv`.',
            '',
            '## Decision Table',
            '',
            report_table.to_markdown(index=False),
            '',
            '## Pooled Significance',
            '',
            pooled_sig_df.to_markdown(index=False) if not pooled_sig_df.empty else 'No pooled daily-return rows were available.',
            '',
            '## Artifacts',
            '',
            f'- Manifest: `{manifest_path}`',
            f'- Training results: `{training_results_path}`',
            f'- Raw backtest results: `{raw_backtest_path}`',
            f'- Combined decision table: `{decision_path}`',
            f'- HTML decision table: `{html_path}`',
            f'- Pooled daily returns: `{pooled_daily_path}`',
            f'- Pooled significance: `{pooled_sig_path}`',
        ]
        if 'metric_plot_path' in globals():
            lines.append(f'- Metric bars: `{metric_plot_path}`')
        if 'excess_plot_path' in globals():
            lines.append(f'- Pooled excess curves: `{excess_plot_path}`')

        report_path.write_text('\n'.join(lines), encoding='utf-8')
        print(report_path.read_text(encoding='utf-8')[:8000])
        print('Summary report:', report_path)
        """
    ),
    md("## 13. Zip Results"),
    code(
        r"""
        archive_path = shutil.make_archive(str(RUN_ROOT), 'zip', root_dir=str(RUN_ROOT.parent), base_dir=RUN_ROOT.name)
        print('Archive:', archive_path)
        """
    ),
]


write_notebook(cells, OUT, metadata=LOCAL_PY310_METADATA, indent=1)
