"""Generate the Colab notebook for issue #31 repeated-seed PIT replication."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path("notebooks/pit_repeated_seed_replication_colab.ipynb")


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
        # MCI-GRU Repeated-Seed Full PIT Masked-Panel Replication

        Issue #31: run a repeated-seed full PIT masked-panel replication of the frozen default recipe.

        This notebook repeats the 2022-2025 true PIT masked-panel workflow with additional base seed(s), compares the result against the reviewed `20260514_043539` run, and emits the same decision-grade artifacts:

        - training results
        - PIT breadth summary
        - prediction count checks
        - cost-aware/rank-gated backtest results
        - pooled daily significance
        - reference comparison
        - summary markdown

        The notebook defaults to one additional base seed because one seed already means 4 yearly jobs x 20 models = 80 trained models. Add more values to `REPLICATION_BASE_SEEDS` only when you intentionally want a larger compute budget.

        Frozen recipe reference: `docs/DEFAULT_EXPERIMENT_RECIPE.md`.
        Recipe name: `static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1`.

        The pooled significance cell is intentionally written to help close issue #29. The 2022 monthly/drawdown/holdings diagnostics are intentionally written to feed issue #30, although issue #30 may still need a final interpretation pass after the run finishes.
        """
    ),
    md(
        """
        ## Known Drive Locations

        Open this notebook from the PIT branch:
        [pit_repeated_seed_replication_colab.ipynb](https://colab.research.google.com/github/magilliam27/MCI-GRU/blob/codex/pit-universe-validation/notebooks/pit_repeated_seed_replication_colab.ipynb)

        Required data folder:
        [MCI_GRU_shared/data](https://drive.google.com/drive/folders/1mlM6KQISlXl3Bnrk20ebEv7Mo8CLg8LO)

        Required files:

        - [sp500_pit_union_lseg_20150101_20260513.csv](https://drive.google.com/file/d/1hbJ7lg45tgyDKv6ecyuh8NxuTr6GpCpm/view?usp=drivesdk)
        - [sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv](https://drive.google.com/file/d/1d5NQC2y9JeKZF-90zh1ko04VzHUsb8PM/view?usp=drivesdk)
        - [sp500_pit_union_lseg_20150101_20260513.meta.json](https://drive.google.com/file/d/1Zn4njEOgtfuchMesHlhNZYdz_emgbBz-/view?usp=drivesdk)

        Reference run folder:
        [MCI-GRU-Ablations/pit_masked_panel_2022_2025/20260514_043539](https://drive.google.com/drive/folders/1p1F2NqY5C6ISBzjm7-JBkbvsE4K2E2LF)
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
            Path('/content/mci_gru_work/pit_repeated_seed_replication')
            if IN_COLAB
            else Path.cwd() / 'results' / 'pit_repeated_seed_replication'
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
    md("## 3. FRED Key And Run Configuration"),
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

        # To resume a partial run in the same Colab runtime, set this to the
        # existing tag printed in the failed run, for example '20260520_024841'.
        RUN_TAG_OVERRIDE = ''
        RUN_TAG = RUN_TAG_OVERRIDE.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
        EXPERIMENT_SLUG = 'pit_repeated_seed_replication'
        LOCAL_RUN_ROOT = LOCAL_RUN_BASE / RUN_TAG
        TRAINING_OUTPUT_DIR = LOCAL_RUN_ROOT / 'training_runs'
        SUMMARY_DIR = LOCAL_RUN_ROOT / 'summaries'
        LOG_DIR = LOCAL_RUN_ROOT / 'logs'
        DRIVE_RUN_ROOT = DRIVE_ROOT / EXPERIMENT_SLUG / RUN_TAG

        for path in [LOCAL_RUN_ROOT, TRAINING_OUTPUT_DIR, SUMMARY_DIR, LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        # Reviewed single-seed full PIT run used by issue #31 as the reference point.
        REFERENCE_RUN_TAG = '20260514_043539'
        REFERENCE_RUN_ROOT = ''

        # One additional base seed is already 4 years x 20 models = 80 trained models.
        REPLICATION_BASE_SEEDS = [314159]

        SMOKE_MODE = False
        RUN_TRAINING = True
        RUN_BACKTESTS = True
        RUN_REFERENCE_COST_BACKTESTS = True
        ALLOW_MISSING_REFERENCE = False
        REUSE_LATEST_RUNS = True
        MAX_JOBS = None
        USE_STATIC_REGIME_INPUTS = True
        STATIC_REGIME_INPUTS_CSV = ''
        TRAINING_RETRY_ON_REGIME_FETCH_FAILURE = 2
        TRAINING_RETRY_SLEEP_SECONDS = 60
        FRED_SERIES_MAX_ATTEMPTS = 3
        FRED_SERIES_RETRY_SECONDS = 10

        os.environ['MCI_GRU_FRED_MAX_ATTEMPTS'] = str(FRED_SERIES_MAX_ATTEMPTS)
        os.environ['MCI_GRU_FRED_RETRY_SECONDS'] = str(FRED_SERIES_RETRY_SECONDS)

        YEARS = [2022, 2023, 2024, 2025]

        MODEL_RECIPE = {
            'name': 'static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1',
            'reference_seed': 1729,
            'label_t': 5,
            'loss_type': 'ic',
            'label_type': 'returns',
            'selection_metric': 'val_ic',
            'drop_edge_p': 0.1,
        }

        # Full-budget defaults follow docs/DEFAULT_EXPERIMENT_RECIPE.md.
        NUM_MODELS = 1 if SMOKE_MODE else 20
        NUM_EPOCHS = 1 if SMOKE_MODE else 100
        EARLY_STOPPING_PATIENCE = 2 if SMOKE_MODE else 15
        BATCH_SIZE = 32
        LEARNING_RATE = '5e-5'
        BOOTSTRAP_RESAMPLES = 50 if SMOKE_MODE else 1000
        TRACKING_ENABLED = False
        LABEL_T = MODEL_RECIPE['label_t']
        PIT_MIN_SCOREABLE_STOCKS = 450
        PIT_BREADTH_POLICY = 'error'

        BACKTEST_SUFFIX = '_pit_daily_tc_rank_gate'
        SPREAD_BPS = 10.0
        SLIPPAGE_BPS = 5.0
        MIN_RANK_DROP = 30
        TOP_K = 10
        ADJUSTMENT_METHOD = 'bhy'
        NUM_TESTS = max(1, len(REPLICATION_BASE_SEEDS) + 1)
        NEWEY_WEST_LAGS = 5
        BLOCK_BOOTSTRAP_BLOCK_SIZE = 21
        SIGNIFICANCE_BOOTSTRAP_RESAMPLES = 200 if SMOKE_MODE else 1000

        if os.environ.get('FRED_API_KEY'):
            print('FRED_API_KEY is set.')
        else:
            print('FRED_API_KEY is not set. Full frozen-recipe runs require it.')

        if not os.environ.get('FRED_API_KEY'):
            raise RuntimeError(
                'Frozen full PIT recipe requires FRED_API_KEY because global regime features are enabled. '
                'Add it as a Colab Secret named FRED_API_KEY or paste it into MY_FRED_KEY.'
            )

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
                'val_start': '2024-01-22',
                'val_end': '2024-12-31',
                'test_start': '2025-01-22',
                'test_end': '2025-12-31',
            },
        }

        RUN_BUDGET = {
            'replication_base_seeds': REPLICATION_BASE_SEEDS,
            'yearly_training_jobs': len(REPLICATION_BASE_SEEDS) * len(YEARS),
            'models_per_yearly_job': NUM_MODELS,
            'total_models': len(REPLICATION_BASE_SEEDS) * len(YEARS) * NUM_MODELS,
            'num_epochs': NUM_EPOCHS,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
        }

        print('Local run root:', LOCAL_RUN_ROOT)
        print('Drive run root:', DRIVE_RUN_ROOT)
        print('RUN_TAG_OVERRIDE:', RUN_TAG_OVERRIDE or '(fresh run)')
        print('REFERENCE_RUN_TAG:', REFERENCE_RUN_TAG)
        print('REPLICATION_BASE_SEEDS:', REPLICATION_BASE_SEEDS)
        print('YEARS:', YEARS)
        print('SMOKE_MODE:', SMOKE_MODE)
        print('USE_STATIC_REGIME_INPUTS:', USE_STATIC_REGIME_INPUTS)
        print('FRED series attempts:', FRED_SERIES_MAX_ATTEMPTS)
        print('FRED series retry seconds:', FRED_SERIES_RETRY_SECONDS)
        print('Training job retries on regime fetch failure:', TRAINING_RETRY_ON_REGIME_FETCH_FAILURE)
        print('Run budget:', RUN_BUDGET)
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

        def run_matches_static_regime_inputs(path: Path) -> bool:
            if not USE_STATIC_REGIME_INPUTS:
                return True

            config_path = path / 'config.yaml'
            marker_path = path / 'static_regime_inputs_marker.json'
            if not config_path.exists() or not marker_path.exists():
                return False

            try:
                import yaml

                cfg = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
                configured_path = (cfg.get('features') or {}).get('regime_inputs_csv')
            except Exception:
                configured_path = None

            marker = read_json(marker_path)
            return (
                str(configured_path) == STATIC_REGIME_INPUTS_RELATIVE_PATH
                and marker.get('relative_path') == STATIC_REGIME_INPUTS_RELATIVE_PATH
                and marker.get('sha256') == STATIC_REGIME_INPUTS_SUMMARY.get('sha256')
            )

        def is_complete_training_run(path: Path, require_static_regime: bool = True) -> bool:
            predictions_dir = path / 'averaged_predictions'
            if not (path / 'training_summary.json').exists() or not predictions_dir.exists():
                return False
            if require_static_regime and not run_matches_static_regime_inputs(path):
                print('Ignoring run with mismatched/missing static regime marker:', path)
                return False
            return True

        def latest_run_dir(experiment_name: str, require_static_regime: bool = True) -> Path | None:
            root = TRAINING_OUTPUT_DIR / experiment_name
            if not root.exists():
                return None
            candidates = [p for p in root.iterdir() if p.is_dir()]
            complete = []
            for candidate in candidates:
                if is_complete_training_run(candidate, require_static_regime=require_static_regime):
                    complete.append(candidate)
                else:
                    print('Ignoring incomplete run dir:', candidate)
            if not complete:
                return None
            return max(complete, key=lambda p: p.stat().st_mtime)

        def write_pit_experiment_presets() -> None:
            # Write runtime-generated PIT union presets for this Colab clone.
            experiment_dir = REPO_DIR / 'configs' / 'experiment'
            experiment_dir.mkdir(parents=True, exist_ok=True)
            for year in YEARS:
                window = PIT_WINDOWS[year]
                path = experiment_dir / f"{window['experiment']}.yaml"
                text = '\n'.join([
                    '# @package _global_',
                    '# Auto-written by pit_repeated_seed_replication_colab.ipynb.',
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

        write_pit_experiment_presets()
        print('Using runtime-generated PIT union presets so the Colab clone points at the staged PIT union CSV.')

        def validate_static_regime_inputs(path: Path) -> dict:
            from mci_gru.features.regime import REGIME_VARIABLES

            frame = pd.read_csv(path)
            missing = sorted(({'dt'} | set(REGIME_VARIABLES)) - set(frame.columns))
            if missing:
                raise RuntimeError(f'Static regime input CSV is missing required columns: {missing}')

            frame['dt'] = pd.to_datetime(frame['dt'])
            target_end = pd.Timestamp(max(window['test_end'] for window in PIT_WINDOWS.values()))
            first_dt = frame['dt'].min()
            last_dt = frame['dt'].max()
            if pd.isna(first_dt) or pd.isna(last_dt) or last_dt < target_end:
                raise RuntimeError(
                    'Static regime input CSV does not cover the full test horizon: '
                    f'last_dt={last_dt}, target_end={target_end}'
                )

            null_counts = {}
            empty_columns = []
            for col in REGIME_VARIABLES:
                values = pd.to_numeric(frame[col], errors='coerce')
                null_counts[col] = int(values.isna().sum())
                if values.notna().sum() == 0:
                    empty_columns.append(col)
            if empty_columns:
                raise RuntimeError(f'Static regime input CSV has all-null columns: {empty_columns}')

            summary = {
                'path': str(path),
                'relative_path': path.relative_to(REPO_DIR).as_posix(),
                'sha256': sha256_file(path),
                'rows': int(len(frame)),
                'first_dt': first_dt.date().isoformat(),
                'last_dt': last_dt.date().isoformat(),
                'target_end': target_end.date().isoformat(),
                'null_counts': null_counts,
            }
            summary_path = LOCAL_RUN_ROOT / 'static_regime_inputs_summary.json'
            summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
            display(pd.DataFrame([summary]))
            display(pd.DataFrame(list(null_counts.items()), columns=['column', 'null_count']))
            print('Static regime input summary:', summary_path)
            return summary

        def build_static_regime_inputs() -> tuple[Path | None, str, dict]:
            if not USE_STATIC_REGIME_INPUTS:
                return None, '', {}

            from mci_gru.data.data_manager import DataManager

            static_dir = REPO_DIR / 'data' / 'raw' / 'regime'
            static_dir.mkdir(parents=True, exist_ok=True)
            dest = static_dir / f'pit_repeated_seed_regime_inputs_{RUN_TAG}.csv'
            drive_static = DRIVE_RUN_ROOT / 'inputs' / dest.name

            def draw_static_regime_inputs(dest: Path) -> None:
                fetch_start = min(window['train_start'] for window in PIT_WINDOWS.values())
                fetch_end = max(window['test_end'] for window in PIT_WINDOWS.values())
                fetch_config = DataConfig(
                    source='csv',
                    filename=(repo_market_csv.relative_to(REPO_DIR)).as_posix(),
                    train_start=fetch_start,
                    train_end=fetch_start,
                    val_start=fetch_start,
                    val_end=fetch_start,
                    test_start=fetch_end,
                    test_end=fetch_end,
                )
                print('Drawing down static regime inputs from FRED-backed loader:', fetch_start, 'to', fetch_end)
                regime_df = DataManager(fetch_config).load_regime_inputs(end=fetch_end)
                regime_df.to_csv(dest, index=False)
                print('Wrote static regime inputs:', dest)

            copied_explicit_static = bool(STATIC_REGIME_INPUTS_CSV.strip())
            if STATIC_REGIME_INPUTS_CSV.strip():
                source = Path(STATIC_REGIME_INPUTS_CSV).expanduser()
                if not source.exists():
                    raise FileNotFoundError(f'STATIC_REGIME_INPUTS_CSV does not exist: {source}')
                if source.resolve() != dest.resolve():
                    shutil.copy2(source, dest)
                    print('Copied static regime inputs:', source, '->', dest)
            elif drive_static.exists():
                shutil.copy2(drive_static, dest)
                print('Recovered static regime inputs from Drive:', drive_static, '->', dest)
            elif dest.exists():
                print('Reusing static regime inputs:', dest)
            else:
                draw_static_regime_inputs(dest)

            try:
                summary = validate_static_regime_inputs(dest)
            except RuntimeError as exc:
                if copied_explicit_static:
                    raise
                print('Cached static regime inputs failed validation; redrawing from FRED-backed loader:', exc)
                if dest.exists():
                    dest.unlink()
                draw_static_regime_inputs(dest)
                summary = validate_static_regime_inputs(dest)

            inputs_dir = LOCAL_RUN_ROOT / 'inputs'
            inputs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, inputs_dir / dest.name)
            (inputs_dir / 'static_regime_inputs_summary.json').write_text(
                json.dumps(summary, indent=2),
                encoding='utf-8',
            )
            drive_inputs_dir = DRIVE_RUN_ROOT / 'inputs'
            drive_inputs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, DRIVE_RUN_ROOT / 'inputs' / dest.name)
            (drive_inputs_dir / 'static_regime_inputs_summary.json').write_text(
                json.dumps(summary, indent=2),
                encoding='utf-8',
            )
            return dest, dest.relative_to(REPO_DIR).as_posix(), summary

        STATIC_REGIME_INPUTS_PATH, STATIC_REGIME_INPUTS_RELATIVE_PATH, STATIC_REGIME_INPUTS_SUMMARY = build_static_regime_inputs()
        if STATIC_REGIME_INPUTS_RELATIVE_PATH:
            print('Static regime inputs:', STATIC_REGIME_INPUTS_RELATIVE_PATH)
            print('Static regime SHA256:', STATIC_REGIME_INPUTS_SUMMARY.get('sha256'))

        BASE_OVERRIDES = [
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
            'tracking.log_artifacts=false',
            'tracking.log_checkpoints=false',
            'tracking.log_predictions=false',
            f"data.filename={(repo_market_csv.relative_to(REPO_DIR)).as_posix()}",
            f"data.pit_universe_csv={(repo_pit_csv.relative_to(REPO_DIR)).as_posix()}",
            'data.use_pit_universe=true',
            'data.pit_universe_mode=masked_panel',
            f'data.pit_min_scoreable_stocks={PIT_MIN_SCOREABLE_STOCKS}',
            f'data.pit_breadth_policy={PIT_BREADTH_POLICY}',
        ]
        if STATIC_REGIME_INPUTS_RELATIVE_PATH:
            BASE_OVERRIDES.append(f'features.regime_inputs_csv={STATIC_REGIME_INPUTS_RELATIVE_PATH}')

        TRAINING_JOBS = []
        for base_seed in REPLICATION_BASE_SEEDS:
            for year in YEARS:
                window = PIT_WINDOWS[year]
                name = safe_name(f'pit_seed_{base_seed}_replication_{year}')
                job = {
                    'name': name,
                    'base_seed': int(base_seed),
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
                    f"seed={job['base_seed']}",
                    f"experiment_name={name}",
                    f"output_dir={TRAINING_OUTPUT_DIR.as_posix()}",
                ]
                TRAINING_JOBS.append(job)

        if MAX_JOBS is not None:
            TRAINING_JOBS = TRAINING_JOBS[:MAX_JOBS]

        manifest = {
            'run_tag': RUN_TAG,
            'run_tag_override': RUN_TAG_OVERRIDE,
            'branch': BRANCH,
            'issue': 31,
            'experiment_slug': EXPERIMENT_SLUG,
            'local_run_root': str(LOCAL_RUN_ROOT),
            'drive_run_root': str(DRIVE_RUN_ROOT),
            'reference_run_tag': REFERENCE_RUN_TAG,
            'market_csv': str(repo_market_csv),
            'market_csv_sha256': sha256_file(repo_market_csv),
            'pit_universe_csv': str(repo_pit_csv),
            'static_regime_inputs_path': str(STATIC_REGIME_INPUTS_PATH) if STATIC_REGIME_INPUTS_PATH else '',
            'static_regime_inputs_relative_path': STATIC_REGIME_INPUTS_RELATIVE_PATH,
            'static_regime_inputs_sha256': STATIC_REGIME_INPUTS_SUMMARY.get('sha256'),
            'static_regime_inputs_summary': STATIC_REGIME_INPUTS_SUMMARY,
            'model_recipe': MODEL_RECIPE,
            'smoke_mode': SMOKE_MODE,
            'years': YEARS,
            'replication_base_seeds': REPLICATION_BASE_SEEDS,
            'run_budget': RUN_BUDGET,
            'fred_series_max_attempts': FRED_SERIES_MAX_ATTEMPTS,
            'fred_series_retry_seconds': FRED_SERIES_RETRY_SECONDS,
            'training_retry_on_regime_fetch_failure': TRAINING_RETRY_ON_REGIME_FETCH_FAILURE,
            'training_retry_sleep_seconds': TRAINING_RETRY_SLEEP_SECONDS,
            'backtest': {
                'suffix': BACKTEST_SUFFIX,
                'spread_bps': SPREAD_BPS,
                'slippage_bps': SLIPPAGE_BPS,
                'min_rank_drop': MIN_RANK_DROP,
                'top_k': TOP_K,
                'num_tests': NUM_TESTS,
                'adjustment_method': ADJUSTMENT_METHOD,
            },
            'base_overrides': BASE_OVERRIDES,
            'jobs': TRAINING_JOBS,
        }
        manifest_path = LOCAL_RUN_ROOT / 'pit_repeated_seed_replication_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        print('Training jobs:', len(TRAINING_JOBS))
        display(pd.DataFrame(TRAINING_JOBS)[['name', 'base_seed', 'year', 'pit_experiment', 'test_start', 'test_end']])
        print('Manifest:', manifest_path)
        """
    ),
    md("## 5. Training, Validation, And Backtest Helpers"),
    code(
        r"""
        def summarize_breadth(metadata: dict, job: dict, run_dir: Path) -> list[dict]:
            rows = []
            breadth = metadata.get('pit_breadth') or {}
            for split, split_rows in breadth.items():
                frame = pd.DataFrame(split_rows)
                if frame.empty:
                    rows.append({
                        'base_seed': job['base_seed'],
                        'year': job['year'],
                        'split': split,
                        'run_dir': str(run_dir),
                        'status': 'missing',
                    })
                    continue
                row = {
                    'base_seed': job['base_seed'],
                    'year': job['year'],
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

        def check_prediction_counts(metadata: dict, job: dict, run_dir: Path) -> dict:
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
                'base_seed': job['base_seed'],
                'year': job['year'],
                'run_dir': str(run_dir),
                'prediction_files_expected': len(test_rows),
                'prediction_files_matched': matched,
                'prediction_files_missing': len(missing_files),
                'prediction_count_mismatches': len(mismatches),
                'prediction_rows_total': total_rows,
                'first_missing_dates': ','.join(missing_files[:5]),
                'first_mismatches': json.dumps(mismatches[:5]),
            }

        def summarize_run_dir(job: dict, run_dir: Path | None, status: str, returncode: int, elapsed_minutes: float | None, stdout_path: Path | None, stderr_path: Path | None) -> dict:
            row = {
                'status': status,
                'returncode': returncode,
                'elapsed_minutes': elapsed_minutes,
                'name': job['name'],
                'base_seed': job['base_seed'],
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
                evaluation = read_json(run_dir / 'evaluation_summary.json')
                row['pit_universe_mode'] = metadata.get('pit_universe_mode')
                row['n_union_axis'] = len(metadata.get('kdcode_list', []))
                row['data_file_sha256'] = metadata.get('data_file_sha256')
                row.update({f'run_metadata.{k}': v for k, v in flatten_dict(metadata).items()})
                row.update({f'training_summary.{k}': v for k, v in flatten_dict(summary).items()})
                row.update({f'evaluation_summary.{k}': v for k, v in flatten_dict(evaluation).items()})
            return row

        def write_static_regime_marker(run_dir: Path | None) -> None:
            if run_dir is None or not USE_STATIC_REGIME_INPUTS:
                return
            marker = {
                'relative_path': STATIC_REGIME_INPUTS_RELATIVE_PATH,
                'sha256': STATIC_REGIME_INPUTS_SUMMARY.get('sha256'),
                'rows': STATIC_REGIME_INPUTS_SUMMARY.get('rows'),
                'first_dt': STATIC_REGIME_INPUTS_SUMMARY.get('first_dt'),
                'last_dt': STATIC_REGIME_INPUTS_SUMMARY.get('last_dt'),
            }
            (run_dir / 'static_regime_inputs_marker.json').write_text(
                json.dumps(marker, indent=2),
                encoding='utf-8',
            )

        def is_regime_fetch_failure(stdout_text: str, stderr_text: str) -> bool:
            text = f'{stdout_text}\n{stderr_text}'
            return (
                'Unable to load required regime input series' in text
                and ('regime_oil' in text or 'regime_' in text)
            )

        def run_training_job(job: dict) -> dict:
            existing = latest_run_dir(job['name']) if REUSE_LATEST_RUNS else None
            if existing is not None:
                print('Reusing existing run:', job['name'], existing)
                return summarize_run_dir(job, existing, 'REUSED', 0, 0.0, None, None)

            run_log_dir = LOG_DIR / job['name']
            run_log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_log_dir / 'train_stdout.log'
            stderr_path = run_log_dir / 'train_stderr.log'
            cmd = [sys.executable, '-u', str(REPO_DIR / 'run_experiment.py'), *job['overrides']]

            max_attempts = 1 + max(0, int(TRAINING_RETRY_ON_REGIME_FETCH_FAILURE))
            stdout_chunks = []
            stderr_chunks = []
            final_returncode = 1
            started = time.time()
            for attempt in range(1, max_attempts + 1):
                print('\n' + '-' * 100)
                print('Training:', job['name'])
                print(f'Attempt: {attempt}/{max_attempts}')
                print('Command:', ' '.join(cmd))
                proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
                final_returncode = proc.returncode
                stdout_chunks.append(
                    f"\n\n{'=' * 100}\nAttempt {attempt}/{max_attempts} stdout\n{'=' * 100}\n"
                    + proc.stdout
                )
                stderr_chunks.append(
                    f"\n\n{'=' * 100}\nAttempt {attempt}/{max_attempts} stderr\n{'=' * 100}\n"
                    + proc.stderr
                )
                print(proc.stdout[-5000:])

                if proc.returncode == 0:
                    break

                print(proc.stderr[-5000:])
                if attempt >= max_attempts or not is_regime_fetch_failure(proc.stdout, proc.stderr):
                    break

                print(
                    'Retrying after transient regime input fetch failure '
                    f'in {TRAINING_RETRY_SLEEP_SECONDS} seconds...'
                )
                time.sleep(TRAINING_RETRY_SLEEP_SECONDS)

            elapsed_minutes = (time.time() - started) / 60.0
            stdout_path.write_text(''.join(stdout_chunks), encoding='utf-8')
            stderr_path.write_text(''.join(stderr_chunks), encoding='utf-8')

            run_dir = latest_run_dir(job['name'], require_static_regime=False)
            if final_returncode == 0:
                write_static_regime_marker(run_dir)
                run_dir = latest_run_dir(job['name'])
            status = 'OK' if final_returncode == 0 and run_dir else 'FAILED'
            return summarize_run_dir(job, run_dir, status, final_returncode, elapsed_minutes, stdout_path, stderr_path)

        def run_backtest(training_row: pd.Series) -> dict:
            pred_dir = Path(training_row['predictions_dir'])
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
                '--top_k',
                str(TOP_K),
                '--label_t',
                str(LABEL_T),
                '--num_tests',
                str(NUM_TESTS),
                '--adjustment_method',
                ADJUSTMENT_METHOD,
                '--auto_save',
                '--backtest_suffix',
                BACKTEST_SUFFIX,
                '--transaction_costs',
                '--spread',
                str(SPREAD_BPS),
                '--slippage',
                str(SLIPPAGE_BPS),
                '--enable_rank_drop_gate',
                '--min_rank_drop',
                str(MIN_RANK_DROP),
            ]
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            env['PYTHONUTF8'] = '1'
            print('\n' + '-' * 100)
            print('Cost/rank-gated PIT backtest:', training_row['name'])
            print('Command:', ' '.join(cmd))
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True, env=env)
            stdout_path.write_text(proc.stdout, encoding='utf-8', errors='replace')
            stderr_path.write_text(proc.stderr, encoding='utf-8', errors='replace')
            print(proc.stdout[-4000:])
            if proc.returncode != 0:
                print(proc.stderr[-4000:])

            run_dir = Path(training_row['run_dir'])
            backtest_dir = run_dir / f'backtest{BACKTEST_SUFFIX}'
            row = {
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'name': training_row['name'],
                'base_seed': int(training_row['base_seed']),
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

        def assert_expected_training_complete(training_df: pd.DataFrame) -> None:
            expected = {(int(job['base_seed']), int(job['year'])) for job in TRAINING_JOBS}
            ok_rows = training_df[training_df['status'].isin(['OK', 'REUSED'])].copy()
            complete = {(int(row['base_seed']), int(row['year'])) for _, row in ok_rows.iterrows()}
            missing = sorted(expected - complete)
            failed = training_df[~training_df['status'].isin(['OK', 'REUSED'])][
                ['name', 'base_seed', 'year', 'status', 'returncode', 'stdout_log', 'stderr_log']
            ].to_dict('records')
            if missing or failed:
                raise RuntimeError(
                    'Expected training jobs did not all complete. '
                    f'Missing={missing}; failed={failed[:5]}. Inspect training_results.csv and logs.'
                )

        def assert_expected_backtests_complete(backtest_df: pd.DataFrame) -> None:
            expected = {(int(job['base_seed']), int(job['year'])) for job in TRAINING_JOBS}
            ok_rows = backtest_df[backtest_df['status'].eq('OK')].copy() if len(backtest_df) else pd.DataFrame()
            complete = {(int(row['base_seed']), int(row['year'])) for _, row in ok_rows.iterrows()}
            missing = sorted(expected - complete)
            failed = backtest_df[~backtest_df['status'].eq('OK')][
                ['name', 'base_seed', 'year', 'status', 'returncode', 'stdout_log', 'stderr_log']
            ].to_dict('records') if len(backtest_df) else []
            if missing or failed:
                raise RuntimeError(
                    'Expected cost/rank-gated backtests did not all complete. '
                    f'Missing={missing}; failed={failed[:5]}. Inspect backtest_results.csv and logs.'
                )
        """
    ),
    md("## 6. Execute Training, Breadth Checks, And Backtests"),
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
            'status', 'name', 'base_seed', 'year', 'elapsed_minutes',
            'n_union_axis', 'pit_universe_mode', 'data_file_sha256',
            'training_summary.mean_best_val_ic', 'run_dir', 'predictions_dir'
        ] if c in training_df.columns]])
        assert_expected_training_complete(training_df)

        breadth_rows = []
        prediction_rows = []
        ok_training = training_df[training_df['status'].isin(['OK', 'REUSED'])].copy()
        job_by_name = {job['name']: job for job in TRAINING_JOBS}
        for _, row in ok_training.iterrows():
            job = job_by_name[str(row['name'])]
            run_dir = Path(row['run_dir'])
            metadata = read_json(run_dir / 'run_metadata.json')
            breadth_rows.extend(summarize_breadth(metadata, job, run_dir))
            prediction_rows.append(check_prediction_counts(metadata, job, run_dir))

        breadth_df = pd.DataFrame(breadth_rows)
        prediction_check_df = pd.DataFrame(prediction_rows)
        breadth_path = SUMMARY_DIR / 'pit_breadth_summary.csv'
        prediction_checks_path = SUMMARY_DIR / 'prediction_count_checks.csv'
        breadth_df.to_csv(breadth_path, index=False)
        prediction_check_df.to_csv(prediction_checks_path, index=False)
        display(breadth_df)
        display(prediction_check_df)

        if not prediction_check_df.empty:
            bad = prediction_check_df[
                (prediction_check_df['prediction_files_missing'] > 0)
                | (prediction_check_df['prediction_count_mismatches'] > 0)
            ]
            if len(bad):
                raise RuntimeError('Prediction count validation failed. Inspect prediction_count_checks.csv.')

        backtest_rows = []
        if RUN_BACKTESTS:
            for _, row in ok_training.iterrows():
                backtest_rows.append(run_backtest(row))
                pd.DataFrame(backtest_rows).to_csv(SUMMARY_DIR / 'backtest_results_interim.csv', index=False)

        backtest_df = pd.DataFrame(backtest_rows)
        backtest_path = SUMMARY_DIR / 'backtest_results.csv'
        backtest_df.to_csv(backtest_path, index=False)
        display(backtest_df[[c for c in [
            'status', 'name', 'base_seed', 'year', 'backtest.ARR', 'backtest.ASR',
            'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return',
            'backtest.excess_return', 'backtest.avg_daily_turnover',
            'backtest.rank_gate_enabled', 'backtest.transaction_costs_enabled', 'backtest_dir'
        ] if c in backtest_df.columns]])
        assert_expected_backtests_complete(backtest_df)
        """
    ),
    md("## 7. Reference Cost/Rank-Gate Backtest And Yearly Comparison"),
    code(
        r"""
        def resolve_reference_run_root() -> Path | None:
            if REFERENCE_RUN_ROOT:
                path = Path(REFERENCE_RUN_ROOT).expanduser()
                if path.exists():
                    return path
                raise FileNotFoundError(f'REFERENCE_RUN_ROOT does not exist: {path}')

            candidates = [
                DRIVE_ROOT / 'pit_masked_panel_2022_2025' / REFERENCE_RUN_TAG,
                Path('/content/drive/MyDrive/MCI-GRU-Ablations/pit_masked_panel_2022_2025') / REFERENCE_RUN_TAG,
                REPO_DIR / 'drive_outputs' / 'pit_masked_panel_2022_2025' / REFERENCE_RUN_TAG,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate

            for root in [DRIVE_ROOT, Path('/content/drive/MyDrive') if IN_COLAB else REPO_DIR / 'drive_outputs']:
                if root.exists():
                    matches = list(root.glob(f'**/{REFERENCE_RUN_TAG}'))
                    if matches:
                        return matches[0]
            return None

        def run_reference_cost_rank_gate(reference_root: Path | None) -> tuple[pd.DataFrame, Path | None]:
            if reference_root is None:
                if not ALLOW_MISSING_REFERENCE:
                    raise RuntimeError('Reference run root not found for 20260514_043539. Set REFERENCE_RUN_ROOT to the mounted Drive folder or set ALLOW_MISSING_REFERENCE=True for a mechanics-only run.')
                print('Reference run root not found; comparison rows will be empty because ALLOW_MISSING_REFERENCE=True.')
                return pd.DataFrame(), None

            output_dir = SUMMARY_DIR / 'reference_cost_rank_gate'
            yearly_csv = output_dir / 'cost_rank_gate_yearly_backtest_results.csv'
            if yearly_csv.exists():
                print('Using existing reference cost/rank-gate rows:', yearly_csv)
                return pd.read_csv(yearly_csv), output_dir

            if not RUN_REFERENCE_COST_BACKTESTS:
                candidates = [
                    reference_root / 'summaries' / 'pit_saved_prediction_cost_rank_gate' / 'cost_rank_gate_yearly_backtest_results.csv',
                    reference_root / 'summaries' / 'reference_cost_rank_gate' / 'cost_rank_gate_yearly_backtest_results.csv',
                ]
                for path in candidates:
                    if path.exists():
                        return pd.read_csv(path), path.parent
                if not ALLOW_MISSING_REFERENCE:
                    raise RuntimeError('Reference cost/rank-gate backtest did not produce a reusable yearly CSV. Enable RUN_REFERENCE_COST_BACKTESTS or set ALLOW_MISSING_REFERENCE=True for a mechanics-only run.')
                return pd.DataFrame(), None

            cmd = [
                sys.executable,
                str(REPO_DIR / 'scripts' / 'run_pit_saved_prediction_backtests.py'),
                '--run-root',
                str(reference_root),
                '--output-dir',
                str(output_dir),
                '--repo-dir',
                str(REPO_DIR),
                '--data-file',
                str(repo_market_csv),
                '--pit-universe-csv',
                str(repo_pit_csv),
                '--years',
                *[str(year) for year in YEARS],
                '--top-k',
                str(TOP_K),
                '--label-t',
                str(LABEL_T),
                '--spread',
                str(SPREAD_BPS),
                '--slippage',
                str(SLIPPAGE_BPS),
                '--min-rank-drop',
                str(MIN_RANK_DROP),
                '--num-tests',
                str(NUM_TESTS),
                '--adjustment-method',
                ADJUSTMENT_METHOD,
                '--backtest-suffix',
                BACKTEST_SUFFIX,
            ]
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = output_dir / 'reference_cost_rank_gate_stdout.log'
            stderr_path = output_dir / 'reference_cost_rank_gate_stderr.log'
            print('Reference cost/rank-gate command:', ' '.join(cmd))
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            stdout_path.write_text(proc.stdout, encoding='utf-8', errors='replace')
            stderr_path.write_text(proc.stderr, encoding='utf-8', errors='replace')
            print(proc.stdout[-4000:])
            if proc.returncode != 0:
                print(proc.stderr[-4000:])

            if yearly_csv.exists():
                return pd.read_csv(yearly_csv), output_dir
            if not ALLOW_MISSING_REFERENCE:
                raise RuntimeError('Reference cost/rank-gate backtest did not produce cost_rank_gate_yearly_backtest_results.csv. Inspect reference_cost_rank_gate_stdout.log and stderr.')
            return pd.DataFrame(), output_dir

        def build_reference_comparison(replication_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
            metric_cols = [
                'backtest.total_return',
                'backtest.benchmark_return',
                'backtest.excess_return',
                'backtest.ARR',
                'backtest.ASR',
                'backtest.MDD',
                'backtest.avg_daily_turnover',
                'backtest.num_trading_days',
            ]
            if replication_df.empty or reference_df.empty:
                return pd.DataFrame(columns=[
                    'base_seed', 'year', 'metric', 'reference_run_tag',
                    'reference_value', 'replication_value', 'delta'
                ])

            rows = []
            for _, rep in replication_df.iterrows():
                if str(rep.get('status')).upper() != 'OK':
                    continue
                year = int(rep['year'])
                ref_rows = reference_df[reference_df['year'].astype(int).eq(year)]
                if ref_rows.empty:
                    continue
                ref = ref_rows.iloc[0]
                for metric in metric_cols:
                    if metric not in replication_df.columns:
                        continue
                    ref_value = ref.get(metric, np.nan)
                    rep_value = rep.get(metric, np.nan)
                    ref_num = pd.to_numeric(ref_value, errors='coerce')
                    rep_num = pd.to_numeric(rep_value, errors='coerce')
                    delta = np.nan if pd.isna(ref_num) or pd.isna(rep_num) else float(rep_num - ref_num)
                    rows.append({
                        'base_seed': int(rep['base_seed']),
                        'year': year,
                        'metric': metric,
                        'reference_run_tag': REFERENCE_RUN_TAG,
                        'reference_value': ref_value,
                        'replication_value': rep_value,
                        'delta': delta,
                    })
            return pd.DataFrame(rows)

        reference_run_root = resolve_reference_run_root()
        print('Reference run root:', reference_run_root)
        reference_backtest_df, reference_cost_dir = run_reference_cost_rank_gate(reference_run_root)
        reference_comparison_df = build_reference_comparison(backtest_df, reference_backtest_df)
        reference_comparison_path = SUMMARY_DIR / 'pit_repeated_seed_reference_comparison.csv'
        reference_comparison_df.to_csv(reference_comparison_path, index=False)
        display(reference_comparison_df)
        """
    ),
    md("## 8. Pooled Daily Significance"),
    code(
        r"""
        from scipy import stats
        from mci_gru.evaluation.statistics import moving_block_bootstrap_ci, newey_west_std

        def load_daily_returns_from_backtests(rows_df: pd.DataFrame, scenario_prefix: str) -> pd.DataFrame:
            frames = []
            if rows_df is None or rows_df.empty:
                return pd.DataFrame()
            for _, row in rows_df.iterrows():
                if str(row.get('status', 'OK')).upper() not in {'OK', 'REUSED'}:
                    continue
                backtest_dir = Path(str(row.get('backtest_dir', '')))
                daily_path = backtest_dir / 'daily_returns.csv'
                if not daily_path.exists():
                    print('Missing daily returns:', daily_path)
                    continue
                frame = pd.read_csv(daily_path)
                frame['date'] = pd.to_datetime(frame['date'])
                year_value = int(row.get('year', frame['date'].dt.year.iloc[0]))
                base_seed = row.get('base_seed', np.nan)
                scenario = f'{scenario_prefix}_seed_{int(base_seed)}' if pd.notna(base_seed) else scenario_prefix
                frame['scenario'] = scenario
                frame['base_seed'] = base_seed
                frame['year'] = year_value
                frame['backtest_dir'] = str(backtest_dir)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        def _information_ratio(values: np.ndarray) -> float:
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size <= 1:
                return float('nan')
            vol = float(np.std(values, ddof=1) * np.sqrt(252))
            if vol <= 0 or not np.isfinite(vol):
                return float('nan')
            return float(np.mean(values) * 252 / vol)

        def pooled_significance_row(frame: pd.DataFrame, scenario: str) -> dict:
            values = pd.to_numeric(frame['excess_return'], errors='coerce').dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                return {'scenario': scenario, 'n_days': 0}
            mean_daily = float(np.mean(values))
            daily_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            nw_lags = min(NEWEY_WEST_LAGS, max(0, values.size - 1))
            nw_std = newey_west_std(values, lags=nw_lags)
            nw_se = nw_std / math.sqrt(values.size) if values.size > 0 else float('nan')
            t_stat = mean_daily / nw_se if nw_se and np.isfinite(nw_se) and nw_se > 0 else float('nan')
            p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if np.isfinite(t_stat) else float('nan')

            block_size = min(BLOCK_BOOTSTRAP_BLOCK_SIZE, max(1, values.size))
            mean_ci = moving_block_bootstrap_ci(
                values,
                statistic=lambda x: float(np.mean(x)),
                block_size=block_size,
                n_resamples=SIGNIFICANCE_BOOTSTRAP_RESAMPLES,
                seed=314159,
                ci_level=0.95,
            )
            ir_ci = moving_block_bootstrap_ci(
                values,
                statistic=_information_ratio,
                block_size=block_size,
                n_resamples=SIGNIFICANCE_BOOTSTRAP_RESAMPLES,
                seed=271828,
                ci_level=0.95,
            )
            annualized_excess_return = mean_daily * 252
            annualized_excess_volatility = daily_std * math.sqrt(252)
            information_ratio = (
                annualized_excess_return / annualized_excess_volatility
                if annualized_excess_volatility > 0
                else float('nan')
            )
            return {
                'scenario': scenario,
                'n_days': int(values.size),
                'mean_daily_excess_return': mean_daily,
                'annualized_excess_return': annualized_excess_return,
                'annualized_excess_volatility': annualized_excess_volatility,
                'information_ratio': information_ratio,
                'newey_west_lags': nw_lags,
                'newey_west_se_daily_excess': nw_se,
                'newey_west_t_stat': t_stat,
                'p_value': p_value,
                'mean_daily_excess_ci95_low': mean_ci['lower'],
                'mean_daily_excess_ci95_high': mean_ci['upper'],
                'information_ratio_ci95_low': ir_ci['lower'],
                'information_ratio_ci95_high': ir_ci['upper'],
                'bootstrap_resamples': SIGNIFICANCE_BOOTSTRAP_RESAMPLES,
                'bootstrap_block_size': block_size,
            }

        def adjust_p_values_bhy(frame: pd.DataFrame, p_col: str = 'p_value') -> pd.Series:
            if frame.empty or p_col not in frame.columns:
                return pd.Series(dtype=float)
            p_values = pd.to_numeric(frame[p_col], errors='coerce')
            valid = p_values.dropna().sort_values()
            if valid.empty:
                return pd.Series(np.nan, index=frame.index)
            m = len(valid)
            c_m = sum(1.0 / i for i in range(1, m + 1))
            adjusted_sorted = []
            for rank, (idx, value) in enumerate(valid.items(), start=1):
                adjusted_sorted.append((idx, min(float(value) * m * c_m / rank, 1.0)))
            running = 1.0
            out = pd.Series(np.nan, index=frame.index, dtype=float)
            for idx, value in reversed(adjusted_sorted):
                running = min(running, value)
                out.loc[idx] = running
            return out

        replication_daily_df = load_daily_returns_from_backtests(backtest_df, 'replication')
        reference_daily_df = load_daily_returns_from_backtests(reference_backtest_df, 'reference')
        pooled_daily_df = pd.concat(
            [frame for frame in [reference_daily_df, replication_daily_df] if not frame.empty],
            ignore_index=True,
        ) if (not reference_daily_df.empty or not replication_daily_df.empty) else pd.DataFrame()
        pooled_daily_path = SUMMARY_DIR / 'pit_repeated_seed_pooled_daily_returns.csv'
        pooled_daily_df.to_csv(pooled_daily_path, index=False)

        significance_rows = []
        if not pooled_daily_df.empty:
            for scenario, frame in pooled_daily_df.groupby('scenario', sort=True):
                significance_rows.append(pooled_significance_row(frame, str(scenario)))
            if not replication_daily_df.empty:
                significance_rows.append(pooled_significance_row(replication_daily_df, 'replication_all_seeds'))

        pooled_significance_df = pd.DataFrame(significance_rows)
        if not pooled_significance_df.empty:
            pooled_significance_df['multiple_testing_adjusted_p_value'] = adjust_p_values_bhy(pooled_significance_df)
            pooled_significance_df['multiple_testing_method'] = ADJUSTMENT_METHOD
        pooled_significance_path = SUMMARY_DIR / 'pit_repeated_seed_pooled_daily_significance.csv'
        pooled_significance_df.to_csv(pooled_significance_path, index=False)

        display(pooled_daily_df.head())
        display(pooled_significance_df)
        print('Pooled daily returns:', pooled_daily_path)
        print('Pooled significance:', pooled_significance_path)
        print('Inference note: Newey-West t-stats and moving-block bootstrap CIs are decision aids, not a guarantee of robustness.')
        """
    ),
    md("## 9. 2022 Diagnostics For Issue #30"),
    code(
        r"""
        def collect_2022_diagnostics(rows_df: pd.DataFrame, scenario_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            monthly_frames = []
            drawdown_rows = []
            holdings_frames = []
            if rows_df is None or rows_df.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            for _, row in rows_df.iterrows():
                if int(row.get('year', 0)) != 2022:
                    continue
                if str(row.get('status', 'OK')).upper() not in {'OK', 'REUSED'}:
                    continue
                backtest_dir = Path(str(row.get('backtest_dir', '')))
                base_seed = row.get('base_seed', np.nan)
                scenario = f'{scenario_prefix}_seed_{int(base_seed)}' if pd.notna(base_seed) else scenario_prefix

                monthly_path = backtest_dir / 'monthly_performance.csv'
                if monthly_path.exists():
                    monthly = pd.read_csv(monthly_path)
                    monthly['scenario'] = scenario
                    monthly['base_seed'] = base_seed
                    monthly['backtest_dir'] = str(backtest_dir)
                    monthly_frames.append(monthly)

                perf_path = backtest_dir / 'cumulative_performance.csv'
                if perf_path.exists():
                    perf = pd.read_csv(perf_path)
                    if len(perf):
                        perf['date'] = pd.to_datetime(perf['date'])
                        dd = pd.to_numeric(perf['portfolio_drawdown'], errors='coerce')
                        bottom_idx = dd.idxmin()
                        bottom_date = perf.loc[bottom_idx, 'date']
                        bottom_drawdown = float(dd.loc[bottom_idx])
                        pre_bottom = perf.loc[:bottom_idx, 'portfolio_value']
                        prior_peak = float(pre_bottom.max())
                        after_bottom = perf.loc[bottom_idx:].copy()
                        recovered = after_bottom[after_bottom['portfolio_value'] >= prior_peak]
                        recovery_date = recovered['date'].iloc[0] if len(recovered) else pd.NaT
                        drawdown_rows.append({
                            'scenario': scenario,
                            'base_seed': base_seed,
                            'year': 2022,
                            'max_drawdown': bottom_drawdown,
                            'drawdown_bottom_date': bottom_date.date().isoformat(),
                            'prior_peak_value': prior_peak,
                            'recovery_date': '' if pd.isna(recovery_date) else recovery_date.date().isoformat(),
                            'recovered_by_year_end': bool(len(recovered)),
                            'backtest_dir': str(backtest_dir),
                        })

                holdings_path = backtest_dir / 'holdings_summary.csv'
                if holdings_path.exists():
                    holdings = pd.read_csv(holdings_path)
                    holdings['scenario'] = scenario
                    holdings['base_seed'] = base_seed
                    holdings['year'] = 2022
                    holdings['backtest_dir'] = str(backtest_dir)
                    holdings_frames.append(holdings.head(25))

            monthly_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
            drawdown_df = pd.DataFrame(drawdown_rows)
            holdings_df = pd.concat(holdings_frames, ignore_index=True) if holdings_frames else pd.DataFrame()
            return monthly_df, drawdown_df, holdings_df

        reference_monthly, reference_drawdown, reference_holdings = collect_2022_diagnostics(reference_backtest_df, 'reference')
        replication_monthly, replication_drawdown, replication_holdings = collect_2022_diagnostics(backtest_df, 'replication')

        monthly_diag_df = pd.concat(
            [frame for frame in [reference_monthly, replication_monthly] if not frame.empty],
            ignore_index=True,
        ) if (not reference_monthly.empty or not replication_monthly.empty) else pd.DataFrame()
        drawdown_diag_df = pd.concat(
            [frame for frame in [reference_drawdown, replication_drawdown] if not frame.empty],
            ignore_index=True,
        ) if (not reference_drawdown.empty or not replication_drawdown.empty) else pd.DataFrame()
        holdings_diag_df = pd.concat(
            [frame for frame in [reference_holdings, replication_holdings] if not frame.empty],
            ignore_index=True,
        ) if (not reference_holdings.empty or not replication_holdings.empty) else pd.DataFrame()

        monthly_diag_path = SUMMARY_DIR / 'pit_repeated_seed_2022_monthly_diagnostics.csv'
        drawdown_diag_path = SUMMARY_DIR / 'pit_repeated_seed_2022_drawdown_diagnostics.csv'
        holdings_diag_path = SUMMARY_DIR / 'pit_repeated_seed_2022_holdings_diagnostics.csv'
        monthly_diag_df.to_csv(monthly_diag_path, index=False)
        drawdown_diag_df.to_csv(drawdown_diag_path, index=False)
        holdings_diag_df.to_csv(holdings_diag_path, index=False)

        display(monthly_diag_df)
        display(drawdown_diag_df)
        display(holdings_diag_df.head(50) if not holdings_diag_df.empty else holdings_diag_df)
        print('2022 monthly diagnostics:', monthly_diag_path)
        print('2022 drawdown diagnostics:', drawdown_diag_path)
        print('2022 holdings diagnostics:', holdings_diag_path)
        """
    ),
    md("## 10. Summary And Drive Sync"),
    code(
        r"""
        def fmt_pct(value):
            if pd.isna(value):
                return ''
            return f'{100 * float(value):.2f}%'

        report_backtest = backtest_df[[c for c in [
            'status', 'name', 'base_seed', 'year', 'backtest.ARR', 'backtest.ASR',
            'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return',
            'backtest.excess_return', 'backtest.avg_daily_turnover', 'backtest_dir'
        ] if c in backtest_df.columns]].copy() if not backtest_df.empty else pd.DataFrame()
        for col in ['backtest.ARR', 'backtest.MDD', 'backtest.total_return', 'backtest.benchmark_return', 'backtest.excess_return', 'backtest.avg_daily_turnover']:
            if col in report_backtest.columns:
                report_backtest[col] = report_backtest[col].apply(fmt_pct)

        report_significance = pooled_significance_df.copy()
        for col in ['mean_daily_excess_return', 'annualized_excess_return', 'annualized_excess_volatility']:
            if col in report_significance.columns:
                report_significance[col] = report_significance[col].apply(lambda x: '' if pd.isna(x) else f'{x:.6f}')
        for col in ['information_ratio', 'p_value', 'multiple_testing_adjusted_p_value']:
            if col in report_significance.columns:
                report_significance[col] = report_significance[col].apply(lambda x: '' if pd.isna(x) else f'{x:.4f}')

        report_path = SUMMARY_DIR / 'pit_repeated_seed_replication_summary.md'
        lines = [
            '# PIT Repeated-Seed Replication Summary',
            '',
            f'Issue: `#31`',
            f'Run tag: `{RUN_TAG}`',
            f'Local run root: `{LOCAL_RUN_ROOT}`',
            f'Drive run root: `{DRIVE_RUN_ROOT}`',
            f'Reference run tag: `{REFERENCE_RUN_TAG}`',
            f'Reference run root: `{reference_run_root}`',
            f'Market CSV: `{repo_market_csv}`',
            f'Market SHA256: `{sha256_file(repo_market_csv)}`',
            f'PIT universe CSV: `{repo_pit_csv}`',
            f'Static regime inputs: `{STATIC_REGIME_INPUTS_RELATIVE_PATH}`',
            f'Static regime SHA256: `{STATIC_REGIME_INPUTS_SUMMARY.get("sha256", "")}`',
            f'Model recipe: `{MODEL_RECIPE["name"]}`',
            f'SMOKE_MODE: `{SMOKE_MODE}`',
            f'REPLICATION_BASE_SEEDS: `{REPLICATION_BASE_SEEDS}`',
            f'YEARS: `{YEARS}`',
            f'Run budget: `{RUN_BUDGET}`',
            '',
            '## Interpretation Notes',
            '',
            '- Full performance evidence must come from PIT masked-panel rows only.',
            '- The primary promotion path here is transaction costs enabled plus rank-drop gate enabled.',
            '- Pooled significance uses daily excess returns across 2022-2025.',
            '- The reference comparison uses `20260514_043539` when that run root is available.',
            '- If the added seed agrees with the reference year by year and pooled, promotion confidence improves.',
            '- If 2022 remains uniquely weak across seeds, issue #30 should treat it as a repeatable regime stress case before blaming randomness.',
            '',
            '## Replication Backtest Rows',
            '',
            report_backtest.to_markdown(index=False) if not report_backtest.empty else 'No replication backtest rows were produced.',
            '',
            '## Pooled Daily Significance',
            '',
            report_significance.to_markdown(index=False) if not report_significance.empty else 'No pooled significance rows were produced.',
            '',
            '## Artifact Checklist',
            '',
            f'- Manifest: `{manifest_path}`',
            f'- Training results: `{training_path}`',
            f'- PIT breadth summary: `{breadth_path}`',
            f'- Prediction count checks: `{prediction_checks_path}`',
            f'- Backtest results: `{backtest_path}`',
            f'- Pooled daily returns: `{pooled_daily_path}`',
            f'- Pooled significance: `{pooled_significance_path}`',
            f'- Reference comparison: `{reference_comparison_path}`',
            f'- 2022 monthly diagnostics: `{monthly_diag_path}`',
            f'- 2022 drawdown diagnostics: `{drawdown_diag_path}`',
            f'- 2022 holdings diagnostics: `{holdings_diag_path}`',
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
        print('Summary report:', DRIVE_RUN_ROOT / 'summaries' / 'pit_repeated_seed_replication_summary.md')
        """
    ),
    md("## 11. Failed-Run Inspection"),
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
                print('No failed training jobs.')

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


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
