"""Generate the Colab notebook for issue #23 long-history PIT evaluation."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path("notebooks/long_history_pit_eval_colab.ipynb")


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
        # MCI-GRU Long-History PIT Evaluation

        This generated Colab notebook evaluates issue #23: whether longer
        MCI-GRU temporal windows help under the frozen production-style recipe.

        The full performance-bearing matrix uses true PIT masked-panel data for
        2022-2025 and compares `his_t=10`, `his_t=21`, `his_t=63`, and
        `his_t=126` in the same notebook. `his_t=252` is available only behind
        a manual flag because it is a gated long-window candidate.

        Non-PIT mechanics smoke checks are wiring evidence only. Full
        interpretation must come from the PIT masked-panel rows generated here.

        Recipe reference: `docs/DEFAULT_EXPERIMENT_RECIPE.md`.
        """
    ),
    md("## 1. Setup: Mount Drive, Clone Repo, Install Dependencies"),
    code(
        r"""
        from pathlib import Path
        import hashlib
        import json
        import os
        import re
        import shutil
        import subprocess
        import sys
        import time

        try:
            from google.colab import drive
            IN_COLAB = True
        except ImportError:
            drive = None
            IN_COLAB = False

        if IN_COLAB:
            drive.mount('/content/drive')

        REPO_URL = 'https://github.com/magilliam27/MCI-GRU.git'
        BRANCH = 'codex/pit-universe-validation'
        REPO_DIR = Path('/content/MCI-GRU') if IN_COLAB else Path.cwd()
        DRIVE_ROOT = (
            Path('/content/drive/MyDrive/MCI-GRU-Ablations')
            if IN_COLAB
            else Path.cwd() / 'drive_outputs'
        )
        DRIVE_DATA_DIR = (
            Path('/content/drive/MyDrive/MCI_GRU_shared/data')
            if IN_COLAB
            else Path.cwd() / 'data' / 'raw' / 'market'
        )
        LOCAL_RUN_BASE = (
            Path('/content/mci_gru_work/long_history_pit_eval')
            if IN_COLAB
            else Path.cwd() / 'results' / 'long_history_pit_eval'
        )
        NOTEBOOK_GENERATED_REPO_FILES = [
            Path('configs/experiment/pit_temporal_2022.yaml'),
            Path('configs/experiment/pit_temporal_2023.yaml'),
            Path('configs/experiment/pit_temporal_2024.yaml'),
            Path('configs/experiment/pit_temporal_2025.yaml'),
        ]

        def clear_notebook_generated_repo_files() -> None:
            for relative_path in NOTEBOOK_GENERATED_REPO_FILES:
                path = REPO_DIR / relative_path
                if path.exists():
                    path.unlink()
                    print('Unlinked generated repo file before branch checkout:', path)

        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        LOCAL_RUN_BASE.mkdir(parents=True, exist_ok=True)

        if IN_COLAB:
            if not REPO_DIR.exists():
                subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin'], check=True)
                clear_notebook_generated_repo_files()
                subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '-B', BRANCH, f'origin/{BRANCH}'], check=True)
                subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only', 'origin', BRANCH], check=True)

        os.chdir(REPO_DIR)
        print('Working directory:', Path.cwd())
        print('Branch:', BRANCH)
        print('Drive data folder:', DRIVE_DATA_DIR)
        print('Drive output root:', DRIVE_ROOT)
        print('Local run base:', LOCAL_RUN_BASE)

        if IN_COLAB:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip', 'setuptools', 'wheel'], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[dev,tracking,fred]'], check=True)

        REQUIRE_GPU = True

        import numpy as np
        import pandas as pd
        import torch

        print('Python:', sys.executable)
        print('Torch:', torch.__version__)
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        elif REQUIRE_GPU:
            raise RuntimeError('No CUDA GPU is visible. In Colab, switch Runtime -> Change runtime type -> GPU.')

        from dataclasses import fields
        from mci_gru.config import DataConfig

        REQUIRED_DATA_CONFIG_FIELDS = {
            'pit_universe_mode',
            'pit_min_scoreable_stocks',
            'pit_breadth_policy',
        }
        data_config_fields = {field.name for field in fields(DataConfig)}
        missing_data_config_fields = REQUIRED_DATA_CONFIG_FIELDS - data_config_fields
        if missing_data_config_fields:
            raise RuntimeError(
                f"Branch {BRANCH!r} does not support true PIT masked-panel fields: "
                f"{sorted(missing_data_config_fields)}. Push the PIT-capable branch "
                "and rerun the setup cell."
            )
        print('PIT DataConfig fields available:', sorted(REQUIRED_DATA_CONFIG_FIELDS))
        """
    ),
    md("## 2. PIT Data Inputs"),
    code(
        r"""
        # Leave these blank to auto-discover by filename under common Drive folders.
        # Set explicit paths if your Drive layout differs.
        MARKET_CSV_PATH = ''
        PIT_UNIVERSE_CSV_PATH = ''
        MARKET_META_JSON_PATH = ''

        MARKET_FILENAME = 'sp500_pit_union_lseg_20150101_20260513.csv'
        PIT_FILENAME = 'sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv'
        MARKET_META_FILENAME = 'sp500_pit_union_lseg_20150101_20260513.meta.json'

        DRIVE_SEARCH_ROOTS = [
            DRIVE_DATA_DIR,
            Path('/content/drive/MyDrive') if IN_COLAB else REPO_DIR,
            Path('/content/drive/MyDrive/MCI_GRU_shared/data') if IN_COLAB else REPO_DIR / 'data' / 'raw' / 'market',
            Path('/content/drive/MyDrive/Stock universe data') if IN_COLAB else REPO_DIR / 'data' / 'raw' / 'constituents',
        ]

        def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b''):
                    digest.update(chunk)
            return digest.hexdigest()

        def resolve_input(explicit_path: str, filename: str) -> Path:
            if explicit_path:
                path = Path(explicit_path).expanduser()
                if path.exists():
                    return path
                raise FileNotFoundError(f'Explicit path does not exist: {path}')

            candidates = []
            for root in DRIVE_SEARCH_ROOTS:
                if root is None:
                    continue
                candidates.append(root / filename)
                candidates.extend(root.glob(f'**/{filename}') if root.exists() else [])

            for candidate in candidates:
                if candidate.exists():
                    return candidate

            searched = '\n'.join(str(root / filename) for root in DRIVE_SEARCH_ROOTS if root is not None)
            raise FileNotFoundError(
                f'Could not find {filename}.\n\n'
                f'Put it in Drive, preferably {DRIVE_DATA_DIR}, or set the explicit path above.\n\n'
                f'Searched:\n{searched}'
            )

        def stage_file(source: Path, dest: Path) -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or source.stat().st_size != dest.stat().st_size:
                shutil.copy2(source, dest)
                print('Copied:', source, '->', dest)
            else:
                print('Already staged:', dest)
            return dest

        market_source = resolve_input(MARKET_CSV_PATH, MARKET_FILENAME)
        pit_source = resolve_input(PIT_UNIVERSE_CSV_PATH, PIT_FILENAME)

        try:
            meta_source = resolve_input(MARKET_META_JSON_PATH, MARKET_META_FILENAME)
        except FileNotFoundError:
            meta_source = None
            print('Market meta JSON not found; continuing with CSV hash/shape checks.')

        repo_market_csv = stage_file(market_source, REPO_DIR / 'data' / 'raw' / 'market' / MARKET_FILENAME)
        repo_pit_csv = stage_file(pit_source, REPO_DIR / 'data' / 'raw' / 'constituents' / PIT_FILENAME)
        repo_market_meta = None
        if meta_source is not None:
            repo_market_meta = stage_file(meta_source, REPO_DIR / 'data' / 'raw' / 'market' / MARKET_META_FILENAME)

        market_preview = pd.read_csv(repo_market_csv, usecols=['kdcode', 'dt'])
        market_preview['dt'] = pd.to_datetime(market_preview['dt'])
        pit_preview = pd.read_csv(repo_pit_csv)

        print('\nMarket CSV:', repo_market_csv)
        print('Rows:', f'{len(market_preview):,}')
        print('Stocks:', f'{market_preview.kdcode.nunique():,}')
        print('Dates:', market_preview.dt.min().date(), 'to', market_preview.dt.max().date())
        print('SHA256:', sha256_file(repo_market_csv))

        print('\nPIT universe CSV:', repo_pit_csv)
        print('Rows:', f'{len(pit_preview):,}')
        print('Columns:', list(pit_preview.columns))
        display(pit_preview.head())
        """
    ),
    md("## 3. FRED Key And Matrix Configuration"),
    code(
        r"""
        from datetime import datetime

        # Option A: add a Colab Secret named FRED_API_KEY.
        # Option B: paste the key here for this runtime.
        MY_FRED_KEY = ''

        if IN_COLAB and not MY_FRED_KEY.strip():
            try:
                from google.colab import userdata
                secret = userdata.get('FRED_API_KEY')
                if secret:
                    MY_FRED_KEY = secret
            except Exception:
                pass

        if MY_FRED_KEY.strip():
            os.environ['FRED_API_KEY'] = MY_FRED_KEY.strip()

        if os.environ.get('FRED_API_KEY'):
            print('FRED_API_KEY is set.')
        else:
            print('FRED_API_KEY is not set. Full frozen-recipe runs require it.')

        RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
        EXPERIMENT_SLUG = 'long_history_pit_eval'
        LOCAL_RUN_ROOT = LOCAL_RUN_BASE / RUN_TAG
        TRAINING_OUTPUT_DIR = LOCAL_RUN_ROOT / 'training_runs'
        SUMMARY_DIR = LOCAL_RUN_ROOT / 'summaries'
        LOG_DIR = LOCAL_RUN_ROOT / 'logs'
        DRIVE_RUN_ROOT = DRIVE_ROOT / EXPERIMENT_SLUG / RUN_TAG

        for path in [LOCAL_RUN_ROOT, TRAINING_OUTPUT_DIR, SUMMARY_DIR, LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        RUN_TRAINING = True
        RUN_BACKTESTS = True
        REUSE_LATEST_RUNS = True
        MAX_JOBS = None

        HIS_T_VALUES = [10, 21, 63, 126]
        INCLUDE_HIS_T_252 = False
        if INCLUDE_HIS_T_252 and 252 not in HIS_T_VALUES:
            HIS_T_VALUES.append(252)

        YEARS = [2022, 2023, 2024, 2025]

        # Set this true only for a cheap mechanics check. Those rows must not be
        # interpreted as model-performance evidence.
        MECHANICS_SMOKE_MODE = False

        MODEL_RECIPE = {
            'name': 'static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1',
            'seed': 1729,
            'label_t': 5,
            'loss_type': 'ic',
            'label_type': 'returns',
            'selection_metric': 'val_ic',
            'drop_edge_p': 0.1,
        }

        NUM_MODELS = 1 if MECHANICS_SMOKE_MODE else 20
        NUM_EPOCHS = 1 if MECHANICS_SMOKE_MODE else 100
        EARLY_STOPPING_PATIENCE = 2 if MECHANICS_SMOKE_MODE else 15
        BATCH_SIZE = 32
        LEARNING_RATE = '5e-5'
        BOOTSTRAP_RESAMPLES = 50 if MECHANICS_SMOKE_MODE else 1000
        TRACKING_ENABLED = False
        LABEL_T = MODEL_RECIPE['label_t']
        PIT_MIN_SCOREABLE_STOCKS = 450
        PIT_BREADTH_POLICY = 'error'

        PIT_WINDOWS = {
            2022: {
                'experiment': 'pit_temporal_2022',
                'train_start': '2016-01-01',
                'train_end': '2020-12-31',
                'val_start': '2021-01-22',
                'val_end': '2021-12-31',
                'test_start': '2022-01-22',
                'test_end': '2022-12-31',
            },
            2023: {
                'experiment': 'pit_temporal_2023',
                'train_start': '2017-01-01',
                'train_end': '2021-12-31',
                'val_start': '2022-01-22',
                'val_end': '2022-12-31',
                'test_start': '2023-01-22',
                'test_end': '2023-12-31',
            },
            2024: {
                'experiment': 'pit_temporal_2024',
                'train_start': '2018-01-01',
                'train_end': '2022-12-31',
                'val_start': '2023-01-22',
                'val_end': '2023-12-31',
                'test_start': '2024-01-22',
                'test_end': '2024-12-31',
            },
            2025: {
                'experiment': 'pit_temporal_2025',
                'train_start': '2019-01-01',
                'train_end': '2023-12-31',
                'val_start': '2024-01-08',
                'val_end': '2024-12-31',
                'test_start': '2025-01-08',
                'test_end': '2025-12-31',
            },
        }

        print('Local run root:', LOCAL_RUN_ROOT)
        print('Drive run root:', DRIVE_RUN_ROOT)
        print('HIS_T_VALUES:', HIS_T_VALUES)
        print('YEARS:', YEARS)
        print('MAX_JOBS:', MAX_JOBS)
        print('MECHANICS_SMOKE_MODE:', MECHANICS_SMOKE_MODE)
        print('Model recipe:', MODEL_RECIPE['name'])
        print('Budget:', {
            'num_models': NUM_MODELS,
            'num_epochs': NUM_EPOCHS,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
        })
        """
    ),
    md("## 4. Build Resumable Job Matrix"),
    code(
        r"""
        def safe_name(value: str, max_len: int = 130) -> str:
            cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
            return cleaned[:max_len].strip('_')

        def read_json(path: Path) -> dict:
            if not path.exists():
                return {}
            with path.open(encoding='utf-8') as handle:
                return json.load(handle)

        def flatten_dict(value: dict, prefix: str = '') -> dict:
            out = {}
            for key, item in value.items():
                name = f'{prefix}.{key}' if prefix else str(key)
                if isinstance(item, dict):
                    out.update(flatten_dict(item, name))
                else:
                    out[name] = item
            return out

        def is_complete_training_run(path: Path) -> bool:
            predictions_dir = path / 'averaged_predictions'
            return (path / 'training_summary.json').exists() and predictions_dir.exists()

        def latest_run_dir(experiment_name: str) -> Path | None:
            root = TRAINING_OUTPUT_DIR / experiment_name
            if not root.exists():
                return None
            candidates = [p for p in root.iterdir() if p.is_dir()]
            if not candidates:
                return None
            complete = []
            for candidate in candidates:
                if is_complete_training_run(candidate):
                    complete.append(candidate)
                else:
                    print('Ignoring incomplete run dir:', candidate)
            if not complete:
                return None
            return max(complete, key=lambda p: p.stat().st_mtime)

        def write_pit_experiment_presets() -> None:
            # Write PIT preset YAMLs when the cloned branch does not have them yet.
            experiment_dir = REPO_DIR / 'configs' / 'experiment'
            experiment_dir.mkdir(parents=True, exist_ok=True)
            preset_names = [
                'pit_temporal_2022.yaml',
                'pit_temporal_2023.yaml',
                'pit_temporal_2024.yaml',
                'pit_temporal_2025.yaml',
            ]
            for year in YEARS:
                window = PIT_WINDOWS[year]
                path = experiment_dir / f"{window['experiment']}.yaml"
                text = '\n'.join([
                    '# @package _global_',
                    '# Auto-written by long_history_pit_eval_colab.ipynb.',
                    f'# True rolling PIT S&P 500 panel for test year {year}.',
                    '',
                    f"experiment_name: {window['experiment']}",
                    '',
                    'data:',
                    '  universe: sp500',
                    '  source: csv',
                    f'  filename: data/raw/market/{MARKET_FILENAME}',
                    f"  train_start: \"{window['train_start']}\"",
                    f"  train_end: \"{window['train_end']}\"",
                    f"  val_start: \"{window['val_start']}\"",
                    f"  val_end: \"{window['val_end']}\"",
                    f"  test_start: \"{window['test_start']}\"",
                    f"  test_end: \"{window['test_end']}\"",
                    '  filter_stocks_per_split: false',
                    '  use_pit_universe: true',
                    f'  pit_universe_csv: data/raw/constituents/{PIT_FILENAME}',
                    '  pit_universe_mode: masked_panel',
                    f'  pit_min_scoreable_stocks: {PIT_MIN_SCOREABLE_STOCKS}',
                    f'  pit_breadth_policy: {PIT_BREADTH_POLICY}',
                    '',
                ])
                path.write_text(text, encoding='utf-8')
                print('Wrote PIT preset:', path)
            print('PIT preset files:', preset_names)

        write_pit_experiment_presets()

        BASE_OVERRIDES = [
            f"seed={MODEL_RECIPE['seed']}",
            'data.source=csv',
            'features=with_momentum',
            'features.include_momentum=true',
            'features.include_weekly_momentum=true',
            'features.momentum_encoding=binary',
            'features.momentum_blend_mode=static',
            'features.momentum_blend_fast_weight=0.5',
            'features.include_global_regime=true',
            'features.regime_strict=true',
            'features.regime_enforce_lag_days=0',
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
            f"graph.drop_edge_p={MODEL_RECIPE['drop_edge_p']}",
            'training.lr_scheduler=cosine',
            f'training.learning_rate={LEARNING_RATE}',
            f'training.num_epochs={NUM_EPOCHS}',
            f'training.num_models={NUM_MODELS}',
            f'training.early_stopping_patience={EARLY_STOPPING_PATIENCE}',
            f'training.batch_size={BATCH_SIZE}',
            f"training.loss_type={MODEL_RECIPE['loss_type']}",
            f"training.label_type={MODEL_RECIPE['label_type']}",
            f"training.selection_metric={MODEL_RECIPE['selection_metric']}",
            'training.shuffle_train=true',
            f"model.label_t={MODEL_RECIPE['label_t']}",
            'model.temporal_encoder=gru_attn',
            f'evaluation.bootstrap_resamples={BOOTSTRAP_RESAMPLES}',
            f'tracking.enabled={str(TRACKING_ENABLED).lower()}',
            'tracking.log_predictions=false',
            f"data.filename={(repo_market_csv.relative_to(REPO_DIR)).as_posix()}",
            f"data.pit_universe_csv={(repo_pit_csv.relative_to(REPO_DIR)).as_posix()}",
            'data.use_pit_universe=true',
            'data.pit_universe_mode=masked_panel',
            f'data.pit_min_scoreable_stocks={PIT_MIN_SCOREABLE_STOCKS}',
            f'data.pit_breadth_policy={PIT_BREADTH_POLICY}',
        ]

        TRAINING_JOBS = []
        for his_t in HIS_T_VALUES:
            for year in YEARS:
                window = PIT_WINDOWS[year]
                name = safe_name(f'long_history_his_t_{his_t}__pit_{year}')
                job = {
                    'name': name,
                    'his_t': his_t,
                    'year': year,
                    'pit_experiment': window['experiment'],
                    'train_start': window['train_start'],
                    'train_end': window['train_end'],
                    'val_start': window['val_start'],
                    'val_end': window['val_end'],
                    'test_start': window['test_start'],
                    'test_end': window['test_end'],
                }
                job['overrides'] = [
                    f"+experiment={window['experiment']}",
                    *BASE_OVERRIDES,
                    f"experiment_name={name}",
                    f"model.his_t={job['his_t']}",
                    f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                ]
                TRAINING_JOBS.append(job)

        if MAX_JOBS is not None:
            TRAINING_JOBS = TRAINING_JOBS[:MAX_JOBS]

        manifest = {
            'run_tag': RUN_TAG,
            'branch': BRANCH,
            'experiment_slug': EXPERIMENT_SLUG,
            'local_run_root': str(LOCAL_RUN_ROOT),
            'drive_run_root': str(DRIVE_RUN_ROOT),
            'market_csv': str(repo_market_csv),
            'market_csv_sha256': sha256_file(repo_market_csv),
            'pit_universe_csv': str(repo_pit_csv),
            'model_recipe': MODEL_RECIPE,
            'mechanics_smoke_mode': MECHANICS_SMOKE_MODE,
            'his_t_values': HIS_T_VALUES,
            'include_his_t_252': INCLUDE_HIS_T_252,
            'years': YEARS,
            'max_jobs': MAX_JOBS,
            'budget': {
                'num_models': NUM_MODELS,
                'num_epochs': NUM_EPOCHS,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
            },
            'base_overrides': BASE_OVERRIDES,
            'jobs': TRAINING_JOBS,
        }
        manifest_path = LOCAL_RUN_ROOT / 'long_history_pit_eval_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        print('Training jobs:', len(TRAINING_JOBS))
        display(pd.DataFrame(TRAINING_JOBS)[['name', 'his_t', 'year', 'pit_experiment', 'test_start', 'test_end']])
        print('Manifest:', manifest_path)
        """
    ),
    md("## 5. Training And Backtest Helpers"),
    code(
        r"""
        def summarize_run_dir(job: dict, run_dir: Path | None, status: str, returncode: int, elapsed_minutes: float | None, stdout_path: Path | None, stderr_path: Path | None) -> dict:
            row = {
                'status': status,
                'returncode': returncode,
                'elapsed_minutes': elapsed_minutes,
                'name': job['name'],
                'his_t': job['his_t'],
                'year': job['year'],
                'pit_experiment': job['pit_experiment'],
                'train_start': job['train_start'],
                'train_end': job['train_end'],
                'val_start': job['val_start'],
                'val_end': job['val_end'],
                'test_start': job['test_start'],
                'test_end': job['test_end'],
                'stdout_log': str(stdout_path) if stdout_path else '',
                'stderr_log': str(stderr_path) if stderr_path else '',
                'run_dir': str(run_dir) if run_dir else '',
                'predictions_dir': str(run_dir / 'averaged_predictions') if run_dir else '',
            }
            if run_dir:
                metadata = read_json(run_dir / 'run_metadata.json')
                summary = read_json(run_dir / 'training_summary.json')
                row.update({f'run_metadata.{k}': v for k, v in flatten_dict(metadata).items()})
                row.update({f'training_summary.{k}': v for k, v in flatten_dict(summary).items()})
            return row

        def run_training_job(job: dict) -> dict:
            existing = latest_run_dir(job['name']) if REUSE_LATEST_RUNS else None
            if existing is not None:
                print('Reusing existing run:', job['name'], existing)
                return summarize_run_dir(job, existing, 'REUSED', 0, 0.0, None, None)

            run_log_dir = LOG_DIR / job['name']
            run_log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_log_dir / 'stdout.log'
            stderr_path = run_log_dir / 'stderr.log'
            cmd = [sys.executable, '-u', str(REPO_DIR / 'run_experiment.py'), *job['overrides']]

            print('\n' + '-' * 100)
            print('Training:', job['name'])
            print('Command:', ' '.join(cmd))
            started = time.time()
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            elapsed_minutes = (time.time() - started) / 60.0
            stdout_path.write_text(proc.stdout, encoding='utf-8')
            stderr_path.write_text(proc.stderr, encoding='utf-8')
            print(proc.stdout[-5000:])
            if proc.returncode != 0:
                print(proc.stderr[-5000:])

            run_dir = latest_run_dir(job['name'])
            status = 'OK' if proc.returncode == 0 and run_dir else 'FAILED'
            return summarize_run_dir(job, run_dir, status, proc.returncode, elapsed_minutes, stdout_path, stderr_path)

        def run_backtest(training_row: pd.Series) -> dict:
            pred_dir = Path(training_row['predictions_dir'])
            suffix = f"_his_t_{int(training_row['his_t'])}_pit_daily"
            stdout_path = LOG_DIR / str(training_row['name']) / 'backtest_stdout.log'
            stderr_path = LOG_DIR / str(training_row['name']) / 'backtest_stderr.log'
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                '-X',
                'utf8',
                str(REPO_DIR / 'tests' / 'backtest_sp500_daily.py'),
                '--predictions_dir',
                str(pred_dir),
                '--data_file',
                str(repo_market_csv),
                '--pit_universe_csv',
                str(repo_pit_csv),
                '--test_start',
                str(training_row['test_start']),
                '--test_end',
                str(training_row['test_end']),
                '--label_t',
                str(LABEL_T),
                '--auto_save',
                '--backtest_suffix',
                suffix,
            ]
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            env['PYTHONUTF8'] = '1'
            print('\n' + '-' * 100)
            print('PIT-aware daily backtest:', training_row['name'])
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True, env=env)
            stdout_path.write_text(proc.stdout, encoding='utf-8')
            stderr_path.write_text(proc.stderr, encoding='utf-8')
            print(proc.stdout[-4000:])
            if proc.returncode != 0:
                print(proc.stderr[-4000:])

            run_dir = Path(training_row['run_dir'])
            backtest_dir = run_dir / f'backtest{suffix}'
            row = {
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'name': training_row['name'],
                'his_t': int(training_row['his_t']),
                'year': int(training_row['year']),
                'run_dir': str(run_dir),
                'predictions_dir': str(pred_dir),
                'backtest_dir': str(backtest_dir),
                'stdout_log': str(stdout_path),
                'stderr_log': str(stderr_path),
            }
            result_csv = backtest_dir / 'backtest_results.csv'
            if result_csv.exists():
                result_df = pd.read_csv(result_csv)
                if len(result_df):
                    for key, value in result_df.iloc[0].to_dict().items():
                        row[f'backtest.{key}'] = value
            return row
        """
    ),
    md("## 6. Execute Resumable Matrix"),
    code(
        r"""
        training_rows = []
        if RUN_TRAINING:
            for job in TRAINING_JOBS:
                training_rows.append(run_training_job(job))
                pd.DataFrame(training_rows).to_csv(SUMMARY_DIR / 'training_results_interim.csv', index=False)
        else:
            for job in TRAINING_JOBS:
                run_dir = latest_run_dir(job['name'])
                status = 'REUSED' if run_dir else 'FAILED'
                training_rows.append(summarize_run_dir(job, run_dir, status, 0 if run_dir else 1, 0.0, None, None))

        training_df = pd.DataFrame(training_rows)
        training_path = SUMMARY_DIR / 'training_results.csv'
        training_df.to_csv(training_path, index=False)
        display(training_df[[c for c in [
            'status', 'name', 'his_t', 'year', 'elapsed_minutes',
            'run_metadata.model.his_t', 'training_summary.mean_best_val_ic',
            'run_dir', 'predictions_dir'
        ] if c in training_df.columns]])

        ok_training = training_df[training_df['status'].isin(['OK', 'REUSED'])].copy()
        backtest_rows = []
        if RUN_BACKTESTS:
            for _, row in ok_training.iterrows():
                backtest_rows.append(run_backtest(row))
                pd.DataFrame(backtest_rows).to_csv(SUMMARY_DIR / 'backtest_results_interim.csv', index=False)

        backtest_df = pd.DataFrame(backtest_rows)
        backtest_path = SUMMARY_DIR / 'backtest_results.csv'
        backtest_df.to_csv(backtest_path, index=False)
        display(backtest_df[[c for c in [
            'status', 'name', 'his_t', 'year', 'backtest.ARR', 'backtest.ASR',
            'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return',
            'backtest.excess_return', 'backtest.avg_daily_turnover', 'backtest_dir'
        ] if c in backtest_df.columns]])
        """
    ),
    md("## 7. Decision Table And Grouped History-Length Summary"),
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
                        score = score + weight * ((vals - vals.mean(skipna=True)) / denom).fillna(0.0)
            out['decision_score'] = score
            return out

        decision_df = add_decision_score(backtest_df) if not backtest_df.empty else pd.DataFrame()
        if not decision_df.empty and 'status' in decision_df.columns:
            decision_df['_status_rank'] = decision_df['status'].map({'OK': 0, 'FAILED': 1}).fillna(2)
            sort_cols = [c for c in ['_status_rank', 'decision_score', 'his_t', 'year'] if c in decision_df.columns]
            decision_df = decision_df.sort_values(sort_cols, ascending=[True, False, True, True][:len(sort_cols)])
            decision_df = decision_df.drop(columns=['_status_rank'])

        decision_path = SUMMARY_DIR / 'long_history_decision_table.csv'
        decision_html_path = SUMMARY_DIR / 'long_history_decision_table.html'
        decision_df.to_csv(decision_path, index=False)
        decision_df.to_html(decision_html_path, index=False)

        ok = decision_df[decision_df['status'].eq('OK')].copy() if not decision_df.empty else pd.DataFrame()
        metric_cols = [c for c in [
            'decision_score',
            'backtest.ARR',
            'backtest.ASR',
            'backtest.MDD',
            'backtest.total_return',
            'backtest.benchmark_return',
            'backtest.excess_return',
            'backtest.avg_daily_turnover',
        ] if c in ok.columns]

        grouped = pd.DataFrame()
        if not ok.empty and metric_cols:
            grouped = ok.groupby('his_t', dropna=False)[metric_cols].agg(['mean', 'median', 'count'])
            grouped.columns = ['.'.join(col).strip('.') for col in grouped.columns.to_flat_index()]
            failure_counts = decision_df.assign(failed=~decision_df['status'].eq('OK')).groupby('his_t')['failed'].mean()
            grouped['failure_rate'] = failure_counts
            grouped = grouped.reset_index()

        grouped_path = SUMMARY_DIR / 'grouped_his_t_summary.csv'
        grouped.to_csv(grouped_path, index=False)

        display_cols = [c for c in [
            'status', 'his_t', 'year', 'decision_score', 'backtest.ARR', 'backtest.ASR',
            'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return',
            'backtest.excess_return', 'backtest.avg_daily_turnover', 'backtest_dir'
        ] if c in decision_df.columns]
        display(decision_df[display_cols] if display_cols else decision_df)
        display(grouped)

        print('Decision table:', decision_path)
        print('Grouped his_t summary:', grouped_path)
        """
    ),
    md("## 8. Export Summary And Sync To Drive"),
    code(
        r"""
        def fmt_pct(value):
            if pd.isna(value):
                return ''
            return f'{100 * float(value):.2f}%'

        report_table = decision_df[[c for c in [
            'status', 'his_t', 'year', 'decision_score', 'backtest.ARR', 'backtest.ASR',
            'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return',
            'backtest.excess_return', 'backtest.avg_daily_turnover', 'backtest_dir'
        ] if c in decision_df.columns]].copy() if not decision_df.empty else pd.DataFrame()

        for pct_col in ['backtest.ARR', 'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return', 'backtest.excess_return', 'backtest.avg_daily_turnover']:
            if pct_col in report_table.columns:
                report_table[pct_col] = report_table[pct_col].apply(fmt_pct)

        grouped_report = grouped.copy()
        for col in grouped_report.columns:
            if col.startswith('backtest.') and not col.endswith('.count'):
                grouped_report[col] = grouped_report[col].apply(lambda x: '' if pd.isna(x) else f'{x:.4f}')

        report_path = SUMMARY_DIR / 'long_history_pit_eval_summary.md'
        lines = [
            '# Long-History PIT Evaluation Summary',
            '',
            f'Run tag: `{RUN_TAG}`',
            f'Local run root: `{LOCAL_RUN_ROOT}`',
            f'Drive run root: `{DRIVE_RUN_ROOT}`',
            f'Market CSV: `{repo_market_csv}`',
            f'Market SHA256: `{sha256_file(repo_market_csv)}`',
            f'PIT universe CSV: `{repo_pit_csv}`',
            f'Model recipe: `{MODEL_RECIPE["name"]}`',
            f'MECHANICS_SMOKE_MODE: `{MECHANICS_SMOKE_MODE}`',
            f'HIS_T_VALUES: `{HIS_T_VALUES}`',
            f'YEARS: `{YEARS}`',
            '',
            '## Interpretation Notes',
            '',
            '- `his_t=10` is the same-notebook frozen-default baseline.',
            '- `his_t=21`, `his_t=63`, and `his_t=126` are the first-pass long-history candidates.',
            '- `his_t=252` is gated by `INCLUDE_HIS_T_252 = False` until shorter windows pass runtime and memory checks.',
            '- Mechanics smoke rows are wiring evidence, not model-performance evidence.',
            '- The decision score is only a sorting aid; read IC, Sharpe, return, drawdown, turnover, and failure-rate evidence directly.',
            '',
            '## Per-Year Decision Rows',
            '',
            report_table.to_markdown(index=False) if not report_table.empty else 'No backtest rows were produced.',
            '',
            '## Grouped his_t Summary',
            '',
            grouped_report.to_markdown(index=False) if not grouped_report.empty else 'No grouped summary was produced.',
            '',
            '## Artifacts',
            '',
            f'- Manifest: `{manifest_path}`',
            f'- Training results: `{training_path}`',
            f'- Backtest results: `{backtest_path}`',
            f'- Decision table: `{decision_path}`',
            f'- Grouped his_t summary: `{grouped_path}`',
        ]
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        print(report_path.read_text(encoding='utf-8')[:12000])

        DRIVE_RUN_ROOT.parent.mkdir(parents=True, exist_ok=True)
        if DRIVE_RUN_ROOT.exists():
            shutil.rmtree(DRIVE_RUN_ROOT)
        shutil.copytree(LOCAL_RUN_ROOT, DRIVE_RUN_ROOT)
        archive_path = shutil.make_archive(str(DRIVE_RUN_ROOT), 'zip', root_dir=str(DRIVE_RUN_ROOT.parent), base_dir=DRIVE_RUN_ROOT.name)

        print('\nSynced run to Drive:', DRIVE_RUN_ROOT)
        print('Drive archive:', archive_path)
        print('Summary report:', DRIVE_RUN_ROOT / 'summaries' / 'long_history_pit_eval_summary.md')
        """
    ),
    md("## 9. Failed-Run Inspection"),
    code(
        r"""
        def print_tail(path_text: str, n_chars: int = 5000) -> None:
            if not path_text:
                return
            path = Path(path_text)
            if path.exists():
                print('\n' + '=' * 100)
                print(path)
                print(path.read_text(encoding='utf-8', errors='replace')[-n_chars:])

        if 'training_df' in globals() and len(training_df):
            failed_training = training_df[~training_df['status'].isin(['OK', 'REUSED'])]
            if len(failed_training):
                display(failed_training)
                for _, row in failed_training.iterrows():
                    print_tail(row.get('stdout_log', ''))
                    print_tail(row.get('stderr_log', ''))
            else:
                print('No failed training runs.')

        if 'backtest_df' in globals() and len(backtest_df):
            failed_backtests = backtest_df[~backtest_df['status'].eq('OK')]
            if len(failed_backtests):
                display(failed_backtests)
                for _, row in failed_backtests.iterrows():
                    print_tail(row.get('stdout_log', ''))
                    print_tail(row.get('stderr_log', ''))
            else:
                print('No failed backtests.')
        """
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
