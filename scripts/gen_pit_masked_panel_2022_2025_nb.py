"""Generate the Colab notebook for strict PIT masked-panel year tests."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path("notebooks/pit_masked_panel_2022_2025_colab.ipynb")


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
        # MCI-GRU True PIT Masked-Panel Tests: 2022-2025

        This notebook runs strict point-in-time masked-panel checks for the 2022, 2023, 2024, and 2025 temporal presets.

        The default is a one-epoch, one-model smoke pass. That is deliberate: first prove the LSEG PIT-union data, daily masks, prediction export, and PIT-aware backtest mechanics. Once every year passes breadth and prediction-count validation, switch `SMOKE_MODE = False` or raise the budget knobs in the configuration cell.

        Required Drive inputs:

        - `sp500_pit_union_lseg_20150101_20260513.csv`
        - `sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv`

        Put them in `/content/drive/MyDrive/MCI_GRU_shared/data`, Drive root, or set explicit paths in the data cell.

        By default, this notebook does not use FRED-backed global regime features. A later setup cell has `USE_GLOBAL_REGIME` and `FRED_API_KEY` controls if you want the regime-enhanced recipe after the PIT mechanics pass.
        """
    ),
    md("## 1. Setup: Mount Drive, Clone Repo, Install Dependencies"),
    code(
        r"""
        from pathlib import Path
        import hashlib
        import json
        import math
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

        # Use this branch until the PIT masked-panel work is merged to main.
        # If you have already merged, set BRANCH = 'main'.
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
            Path('/content/mci_gru_pit_masked_panel_runs')
            if IN_COLAB
            else Path.cwd() / 'results' / 'colab_like_pit_masked_panel'
        )

        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        LOCAL_RUN_BASE.mkdir(parents=True, exist_ok=True)

        if IN_COLAB:
            if not REPO_DIR.exists():
                subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin'], check=True)
                subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', BRANCH], check=True)
                subprocess.run(['git', '-C', str(REPO_DIR), 'pull', 'origin', BRANCH], check=True)

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
        """
    ),
    md("## 2. Data Inputs"),
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

        if repo_market_meta is not None:
            with repo_market_meta.open(encoding='utf-8') as handle:
                market_meta = json.load(handle)
            print('\nMarket meta:')
            print(json.dumps({k: market_meta.get(k) for k in [
                'source',
                'requested_identifiers',
                'resolved_identifiers_with_rows',
                'rows',
                'date_min',
                'date_max',
                'currently_unresolved_original_failure_count',
            ]}, indent=2))
        """
    ),
    md("## 3. Optional FRED Key And Regime Toggle"),
    code(
        r"""
        # The default PIT smoke does NOT use global regime features, so it does not need FRED.
        # Turn USE_GLOBAL_REGIME on after the PIT data/mask mechanics pass if you want the richer regime recipe.
        USE_GLOBAL_REGIME = False

        # Option A: add a Colab Secret named FRED_API_KEY.
        # Option B: paste the key here for this runtime.
        MY_FRED_KEY = ''

        REGIME_STRICT = True
        REGIME_ENFORCE_LAG_DAYS = 0

        if IN_COLAB and not MY_FRED_KEY.strip():
            try:
                from google.colab import userdata

                secret_value = userdata.get('FRED_API_KEY')
                if secret_value:
                    os.environ['FRED_API_KEY'] = secret_value
                    print('FRED_API_KEY loaded from Colab Secrets.')
            except Exception as exc:
                print('No FRED_API_KEY loaded from Colab Secrets:', exc)

        if MY_FRED_KEY.strip():
            os.environ['FRED_API_KEY'] = MY_FRED_KEY.strip()
            print('FRED_API_KEY loaded from MY_FRED_KEY.')

        print('USE_GLOBAL_REGIME:', USE_GLOBAL_REGIME)
        print('FRED_API_KEY is set:', bool(os.environ.get('FRED_API_KEY')))

        if USE_GLOBAL_REGIME and REGIME_STRICT and not os.environ.get('FRED_API_KEY'):
            raise RuntimeError(
                'USE_GLOBAL_REGIME=True with REGIME_STRICT=True requires FRED_API_KEY. '
                'Add it as a Colab Secret named FRED_API_KEY or paste it into MY_FRED_KEY.'
            )
        """
    ),
    md("## 4. Test Configuration"),
    code(
        r"""
        from datetime import datetime

        RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
        EXPERIMENT_SLUG = 'pit_masked_panel_2022_2025'
        LOCAL_RUN_ROOT = LOCAL_RUN_BASE / RUN_TAG
        TRAINING_OUTPUT_DIR = LOCAL_RUN_ROOT / 'training_runs'
        SUMMARY_DIR = LOCAL_RUN_ROOT / 'summaries'
        LOG_DIR = LOCAL_RUN_ROOT / 'logs'
        DRIVE_RUN_ROOT = DRIVE_ROOT / EXPERIMENT_SLUG / RUN_TAG

        for path in [LOCAL_RUN_ROOT, TRAINING_OUTPUT_DIR, SUMMARY_DIR, LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        SMOKE_MODE = True
        RUN_TRAINING = True
        RUN_BACKTESTS = True

        YEARS = [2022, 2023, 2024, 2025]

        # Default smoke budget. Increase these after all four years pass strict breadth.
        NUM_MODELS = 1 if SMOKE_MODE else 10
        NUM_EPOCHS = 1 if SMOKE_MODE else 100
        EARLY_STOPPING_PATIENCE = 2 if SMOKE_MODE else 10
        BATCH_SIZE = 32
        BOOTSTRAP_RESAMPLES = 50 if SMOKE_MODE else 1000

        PIT_MIN_SCOREABLE_STOCKS = 450
        PIT_BREADTH_POLICY = 'error'

        TRACKING_ENABLED = False
        BACKTEST_SUFFIX = '_pit_daily'
        LABEL_T = 5

        PIT_WINDOWS = {
            2022: {
                'experiment': 'pit_temporal_2022',
                'experiment_name': 'pit_true_rolling_2022',
                'train_start': '2016-01-01',
                'train_end': '2020-12-31',
                'val_start': '2021-01-22',
                'val_end': '2021-12-31',
                'test_start': '2022-01-22',
                'test_end': '2022-12-31',
            },
            2023: {
                'experiment': 'pit_temporal_2023',
                'experiment_name': 'pit_true_rolling_2023',
                'train_start': '2017-01-01',
                'train_end': '2021-12-31',
                'val_start': '2022-01-22',
                'val_end': '2022-12-31',
                'test_start': '2023-01-22',
                'test_end': '2023-12-31',
            },
            2024: {
                'experiment': 'pit_temporal_2024',
                'experiment_name': 'pit_true_rolling_2024',
                'train_start': '2018-01-01',
                'train_end': '2022-12-31',
                'val_start': '2023-01-22',
                'val_end': '2023-12-31',
                'test_start': '2024-01-22',
                'test_end': '2024-12-31',
            },
            2025: {
                'experiment': 'pit_temporal_2025',
                'experiment_name': 'pit_true_rolling_2025',
                'train_start': '2019-01-01',
                'train_end': '2023-12-31',
                'val_start': '2024-01-22',
                'val_end': '2024-12-31',
                'test_start': '2025-01-22',
                'test_end': '2025-12-31',
            },
        }

        print('Local run root:', LOCAL_RUN_ROOT)
        print('Drive run root:', DRIVE_RUN_ROOT)
        print('Years:', YEARS)
        print('SMOKE_MODE:', SMOKE_MODE)
        print('Budget:', {'num_models': NUM_MODELS, 'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE})
        """
    ),
    md("## 4. Execution And Validation Helpers"),
    code(
        r"""
        def is_timestamp_dir(path: Path) -> bool:
            return path.is_dir() and bool(re.fullmatch(r'\d{8}_\d{6}', path.name))

        def latest_run_dir(experiment_name: str) -> Path | None:
            base = TRAINING_OUTPUT_DIR / experiment_name
            if not base.exists():
                return None
            candidates = sorted([p for p in base.iterdir() if is_timestamp_dir(p)])
            return candidates[-1] if candidates else None

        def read_json(path: Path) -> dict:
            if not path.exists():
                return {}
            with path.open(encoding='utf-8') as handle:
                return json.load(handle)

        def flatten_dict(value: dict, prefix: str = '') -> dict:
            out = {}
            for key, item in value.items():
                full_key = f'{prefix}.{key}' if prefix else str(key)
                if isinstance(item, dict):
                    out.update(flatten_dict(item, full_key))
                else:
                    out[full_key] = item
            return out

        def summarize_breadth(metadata: dict, year: int, run_dir: Path) -> list[dict]:
            rows = []
            breadth = metadata.get('pit_breadth') or {}
            for split, split_rows in breadth.items():
                frame = pd.DataFrame(split_rows)
                if frame.empty:
                    rows.append({'year': year, 'split': split, 'run_dir': str(run_dir), 'status': 'missing'})
                    continue
                row = {
                    'year': year,
                    'split': split,
                    'run_dir': str(run_dir),
                    'status': 'OK',
                    'dates': len(frame),
                }
                for col in ['active_count', 'feature_ready_count', 'loss_count', 'scoreable_count']:
                    if col in frame.columns:
                        values = pd.to_numeric(frame[col], errors='coerce')
                        row[f'{col}.min'] = int(values.min())
                        row[f'{col}.median'] = float(values.median())
                        row[f'{col}.max'] = int(values.max())
                        row[f'{col}.below_threshold'] = int((values < PIT_MIN_SCOREABLE_STOCKS).sum())
                rows.append(row)
            return rows

        def check_prediction_counts(metadata: dict, run_dir: Path, year: int) -> dict:
            pred_dir = run_dir / 'averaged_predictions'
            test_rows = (metadata.get('pit_breadth') or {}).get('test') or []
            missing_files = []
            mismatches = []
            matched = 0
            total_rows = 0
            for item in test_rows:
                date = str(item['date'])
                expected = int(item['scoreable_count'])
                path = pred_dir / f'{date}.csv'
                if not path.exists():
                    missing_files.append(date)
                    continue
                actual = len(pd.read_csv(path))
                total_rows += actual
                if actual != expected:
                    mismatches.append({'date': date, 'expected': expected, 'actual': actual})
                else:
                    matched += 1
            return {
                'year': year,
                'run_dir': str(run_dir),
                'prediction_files_expected': len(test_rows),
                'prediction_files_matched': matched,
                'prediction_files_missing': len(missing_files),
                'prediction_count_mismatches': len(mismatches),
                'prediction_rows_total': total_rows,
                'first_missing_dates': ','.join(missing_files[:5]),
                'first_mismatches': json.dumps(mismatches[:5]),
            }

        def run_training_year(year: int) -> dict:
            window = PIT_WINDOWS[year]
            log_dir = LOG_DIR / str(year)
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / 'train_stdout.log'
            stderr_path = log_dir / 'train_stderr.log'
            cmd = [
                sys.executable,
                '-u',
                str(REPO_DIR / 'run_experiment.py'),
                f"+experiment={window['experiment']}",
                f"training.num_epochs={NUM_EPOCHS}",
                f"training.num_models={NUM_MODELS}",
                f"training.early_stopping_patience={EARLY_STOPPING_PATIENCE}",
                f"training.batch_size={BATCH_SIZE}",
                f"evaluation.bootstrap_resamples={BOOTSTRAP_RESAMPLES}",
                f"tracking.enabled={str(TRACKING_ENABLED).lower()}",
                'tracking.log_artifacts=false',
                'tracking.log_checkpoints=false',
                'tracking.log_predictions=false',
                f"features.include_global_regime={str(USE_GLOBAL_REGIME).lower()}",
                f"features.regime_strict={str(REGIME_STRICT if USE_GLOBAL_REGIME else False).lower()}",
                f"features.regime_enforce_lag_days={REGIME_ENFORCE_LAG_DAYS}",
                'features.regime_include_subsequent_returns=false',
                f"data.filename={(repo_market_csv.relative_to(REPO_DIR)).as_posix()}",
                f"data.pit_universe_csv={(repo_pit_csv.relative_to(REPO_DIR)).as_posix()}",
                f"data.pit_min_scoreable_stocks={PIT_MIN_SCOREABLE_STOCKS}",
                f"data.pit_breadth_policy={PIT_BREADTH_POLICY}",
                f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
            ]
            print('\n' + '=' * 100)
            print('Training strict PIT year:', year)
            print('Command:', ' '.join(cmd))
            start = time.time()
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            elapsed = (time.time() - start) / 60
            stdout_path.write_text(proc.stdout, encoding='utf-8')
            stderr_path.write_text(proc.stderr, encoding='utf-8')
            print(proc.stdout[-5000:])
            if proc.returncode != 0:
                print(proc.stderr[-5000:])

            run_dir = latest_run_dir(window['experiment_name'])
            row = {
                'year': year,
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'elapsed_minutes': elapsed,
                'run_dir': str(run_dir) if run_dir else '',
                'stdout_log': str(stdout_path),
                'stderr_log': str(stderr_path),
                **window,
            }
            if run_dir:
                metadata = read_json(run_dir / 'run_metadata.json')
                row['pit_universe_mode'] = metadata.get('pit_universe_mode')
                row['n_union_axis'] = len(metadata.get('kdcode_list', []))
                row['data_file_sha256'] = metadata.get('data_file_sha256')
                row.update({f'training_summary.{k}': v for k, v in flatten_dict(read_json(run_dir / 'training_summary.json')).items()})
            return row

        def run_backtest_year(training_row: pd.Series) -> dict:
            year = int(training_row['year'])
            run_dir = Path(training_row['run_dir'])
            pred_dir = run_dir / 'averaged_predictions'
            stdout_path = LOG_DIR / str(year) / 'backtest_stdout.log'
            stderr_path = LOG_DIR / str(year) / 'backtest_stderr.log'
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
                BACKTEST_SUFFIX,
            ]
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            env['PYTHONUTF8'] = '1'
            print('\n' + '-' * 100)
            print('PIT-aware daily backtest year:', year)
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True, env=env)
            stdout_path.write_text(proc.stdout, encoding='utf-8')
            stderr_path.write_text(proc.stderr, encoding='utf-8')
            print(proc.stdout[-4000:])
            if proc.returncode != 0:
                print(proc.stderr[-4000:])

            backtest_dir = run_dir / f'backtest{BACKTEST_SUFFIX}'
            row = {
                'year': year,
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'run_dir': str(run_dir),
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
    md("## 5. Run Strict PIT Training Smokes"),
    code(
        r"""
        def write_pit_experiment_presets() -> None:
            # Write PIT preset YAMLs when the cloned branch does not have them yet.
            experiment_dir = REPO_DIR / 'configs' / 'experiment'
            experiment_dir.mkdir(parents=True, exist_ok=True)
            for year in YEARS:
                window = PIT_WINDOWS[year]
                path = experiment_dir / f"{window['experiment']}.yaml"
                text = '\n'.join([
                    '# @package _global_',
                    '# Auto-written by pit_masked_panel_2022_2025_colab.ipynb.',
                    f'# True rolling PIT S&P 500 panel for test year {year}.',
                    '',
                    f"experiment_name: {window['experiment_name']}",
                    '',
                    'data:',
                    '  universe: sp500',
                    '  source: csv',
                    '  filename: data/raw/market/sp500_data.csv',
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

        write_pit_experiment_presets()

        # Safe defaults if the optional FRED/regime cell was skipped in a resumed notebook.
        USE_GLOBAL_REGIME = globals().get('USE_GLOBAL_REGIME', False)
        REGIME_STRICT = globals().get('REGIME_STRICT', True)
        REGIME_ENFORCE_LAG_DAYS = globals().get('REGIME_ENFORCE_LAG_DAYS', 0)

        manifest = {
            'run_tag': RUN_TAG,
            'branch': BRANCH,
            'smoke_mode': SMOKE_MODE,
            'years': YEARS,
            'market_csv': str(repo_market_csv),
            'market_csv_sha256': sha256_file(repo_market_csv),
            'pit_universe_csv': str(repo_pit_csv),
            'use_global_regime': USE_GLOBAL_REGIME,
            'fred_api_key_set': bool(os.environ.get('FRED_API_KEY')),
            'regime_strict': REGIME_STRICT,
            'regime_enforce_lag_days': REGIME_ENFORCE_LAG_DAYS,
            'pit_min_scoreable_stocks': PIT_MIN_SCOREABLE_STOCKS,
            'pit_breadth_policy': PIT_BREADTH_POLICY,
            'budget': {
                'num_models': NUM_MODELS,
                'num_epochs': NUM_EPOCHS,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'batch_size': BATCH_SIZE,
                'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
            },
            'pit_windows': PIT_WINDOWS,
        }
        (LOCAL_RUN_ROOT / 'pit_masked_panel_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        training_rows = []
        if RUN_TRAINING:
            for year in YEARS:
                training_rows.append(run_training_year(year))
                pd.DataFrame(training_rows).to_csv(SUMMARY_DIR / 'training_results_interim.csv', index=False)
        else:
            for year in YEARS:
                window = PIT_WINDOWS[year]
                run_dir = latest_run_dir(window['experiment_name'])
                training_rows.append({
                    'year': year,
                    'status': 'OK' if run_dir else 'FAILED',
                    'returncode': 0 if run_dir else 1,
                    'run_dir': str(run_dir) if run_dir else '',
                    **window,
                })

        training_df = pd.DataFrame(training_rows)
        training_df.to_csv(SUMMARY_DIR / 'training_results.csv', index=False)
        display(training_df[[c for c in [
            'year',
            'status',
            'elapsed_minutes',
            'n_union_axis',
            'pit_universe_mode',
            'data_file_sha256',
            'run_dir',
        ] if c in training_df.columns]])
        """
    ),
    md("## 6. Inspect PIT Breadth And Prediction Counts"),
    code(
        r"""
        breadth_rows = []
        prediction_rows = []

        ok_training = training_df[training_df['status'].eq('OK')].copy()
        for _, row in ok_training.iterrows():
            year = int(row['year'])
            run_dir = Path(row['run_dir'])
            metadata = read_json(run_dir / 'run_metadata.json')
            breadth_rows.extend(summarize_breadth(metadata, year, run_dir))
            prediction_rows.append(check_prediction_counts(metadata, run_dir, year))

        breadth_df = pd.DataFrame(breadth_rows)
        prediction_check_df = pd.DataFrame(prediction_rows)
        breadth_df.to_csv(SUMMARY_DIR / 'pit_breadth_summary.csv', index=False)
        prediction_check_df.to_csv(SUMMARY_DIR / 'prediction_count_checks.csv', index=False)

        display(breadth_df)
        display(prediction_check_df)

        if not prediction_check_df.empty:
            bad = prediction_check_df[
                (prediction_check_df['prediction_files_missing'] > 0)
                | (prediction_check_df['prediction_count_mismatches'] > 0)
            ]
            if len(bad):
                raise RuntimeError('Prediction count validation failed. Inspect prediction_count_checks.csv.')

        if not breadth_df.empty and 'scoreable_count.below_threshold' in breadth_df.columns:
            low = breadth_df[pd.to_numeric(breadth_df['scoreable_count.below_threshold'], errors='coerce').fillna(0) > 0]
            if len(low):
                raise RuntimeError('One or more splits fell below PIT_MIN_SCOREABLE_STOCKS.')
        """
    ),
    md("## 7. Run PIT-Aware Daily Backtests"),
    code(
        r"""
        backtest_rows = []
        if RUN_BACKTESTS:
            for _, row in ok_training.iterrows():
                backtest_rows.append(run_backtest_year(row))
                pd.DataFrame(backtest_rows).to_csv(SUMMARY_DIR / 'backtest_results_interim.csv', index=False)

        backtest_df = pd.DataFrame(backtest_rows)
        backtest_df.to_csv(SUMMARY_DIR / 'backtest_results.csv', index=False)
        display(backtest_df[[c for c in [
            'year',
            'status',
            'backtest.ARR',
            'backtest.ASR',
            'backtest.MDD',
            'backtest.total_return',
            'backtest.benchmark_return',
            'backtest.excess_return',
            'backtest.num_trading_days',
            'backtest_dir',
        ] if c in backtest_df.columns]])
        """
    ),
    md("## 8. Build Final Report And Sync To Drive"),
    code(
        r"""
        def pct(value) -> str:
            try:
                if pd.isna(value):
                    return ''
                return f'{float(value):.2%}'
            except Exception:
                return str(value)

        lines = [
            '# PIT Masked-Panel 2022-2025 Validation Summary',
            '',
            f'Run tag: `{RUN_TAG}`',
            f'Local run root: `{LOCAL_RUN_ROOT}`',
            f'Drive run root: `{DRIVE_RUN_ROOT}`',
            f'Market CSV: `{repo_market_csv}`',
            f'Market SHA256: `{sha256_file(repo_market_csv)}`',
            f'PIT universe CSV: `{repo_pit_csv}`',
            f'SMOKE_MODE: `{SMOKE_MODE}`',
            f'USE_GLOBAL_REGIME: `{USE_GLOBAL_REGIME}`',
            f'FRED_API_KEY set: `{bool(os.environ.get("FRED_API_KEY"))}`',
            '',
            '## Training Runs',
            '',
            training_df[[c for c in [
                'year',
                'status',
                'elapsed_minutes',
                'n_union_axis',
                'pit_universe_mode',
                'run_dir',
            ] if c in training_df.columns]].to_markdown(index=False),
            '',
            '## PIT Breadth',
            '',
            breadth_df.to_markdown(index=False) if len(breadth_df) else 'No breadth rows.',
            '',
            '## Prediction Count Checks',
            '',
            prediction_check_df.to_markdown(index=False) if len(prediction_check_df) else 'No prediction checks.',
            '',
        ]

        if len(backtest_df):
            report_backtest = backtest_df[[c for c in [
                'year',
                'status',
                'backtest.ARR',
                'backtest.ASR',
                'backtest.MDD',
                'backtest.total_return',
                'backtest.benchmark_return',
                'backtest.excess_return',
                'backtest.num_trading_days',
                'backtest_dir',
            ] if c in backtest_df.columns]].copy()
            for col in ['backtest.ARR', 'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return', 'backtest.excess_return']:
                if col in report_backtest.columns:
                    report_backtest[col] = report_backtest[col].map(pct)
            lines.extend([
                '## PIT-Aware Daily Backtests',
                '',
                report_backtest.to_markdown(index=False),
                '',
                '> Backtest metrics from one-epoch smoke runs are mechanics evidence, not model-performance evidence.',
                '',
            ])

        report_path = SUMMARY_DIR / 'pit_masked_panel_2022_2025_summary.md'
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        print(report_path.read_text(encoding='utf-8')[:12000])

        DRIVE_RUN_ROOT.parent.mkdir(parents=True, exist_ok=True)
        if DRIVE_RUN_ROOT.exists():
            shutil.rmtree(DRIVE_RUN_ROOT)
        shutil.copytree(LOCAL_RUN_ROOT, DRIVE_RUN_ROOT)
        archive_path = shutil.make_archive(str(DRIVE_RUN_ROOT), 'zip', root_dir=str(DRIVE_RUN_ROOT.parent), base_dir=DRIVE_RUN_ROOT.name)

        print('\nSynced run to Drive:', DRIVE_RUN_ROOT)
        print('Drive archive:', archive_path)
        print('Summary report:', DRIVE_RUN_ROOT / 'summaries' / 'pit_masked_panel_2022_2025_summary.md')
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
                print('\n---', path, '---')
                print(path.read_text(encoding='utf-8', errors='replace')[-n_chars:])

        failed_training = training_df[~training_df['status'].eq('OK')] if len(training_df) else pd.DataFrame()
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
        "colab": {"provenance": []},
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
