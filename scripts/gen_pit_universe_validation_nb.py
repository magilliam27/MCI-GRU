"""Generate the Colab notebook for PIT/date-aware universe validation."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path("notebooks/pit_universe_validation_colab.ipynb")


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
        # MCI-GRU PIT Universe Validation

        This notebook tests whether the frozen MCI-GRU proof recipe survives survivorship and future-completeness controls.

        It keeps the model recipe fixed and compares:

        - the current baseline complete-stock filter;
        - per-split complete-stock filtering;
        - point-in-time/date-aware row filtering through `data.use_pit_universe=true`;
        - PIT plus per-split filtering.

        If `PIT_UNIVERSE_CSV` is blank, the notebook first runs the Joiner/Leaver PIT exporter and uses the generated `*_pit_universe.csv`. That export requires an LSEG/Refinitiv-enabled environment. If you already generated the PIT CSV and stored it in Drive, set `PIT_UNIVERSE_CSV` to that path and skip generation.
        """
    ),
    md("## 1. Mount Drive, Clone Repo, Install Dependencies"),
    code(
        r"""
        from pathlib import Path
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
        DRIVE_ROOT = Path('/content/drive/MyDrive/MCI-GRU-Ablations') if IN_COLAB else Path.cwd() / 'drive_outputs'
        DRIVE_DATA_DIR = Path('/content/drive/MyDrive/MCI_GRU_shared/data') if IN_COLAB else Path.cwd() / 'data' / 'raw' / 'market'
        DRIVE_STOCK_UNIVERSE_DIR = Path('/content/drive/MyDrive/Stock universe data') if IN_COLAB else DRIVE_DATA_DIR

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
        import pandas as pd
        import numpy as np
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
    md("## 2. FRED API Key And PIT Inputs"),
    code(
        r"""
        # Paste a FRED API key here, or set FRED_API_KEY in the notebook environment.
        MY_FRED_KEY = ''
        if MY_FRED_KEY.strip():
            os.environ['FRED_API_KEY'] = MY_FRED_KEY.strip()

        # Optional existing PIT file. Accepts an absolute Drive path, repo-relative path, or blank.
        # Required schema: kdcode, valid_from, valid_to. If blank, this notebook generates one.
        PIT_UNIVERSE_CSV = ''
        GENERATE_PIT_UNIVERSE = True
        PIT_EXPORT_START = '2016-01-01'
        PIT_EXPORT_END = '2026-05-13'
        PIT_EXPORT_INDEX_RIC = '.SPX'
        PIT_EXPORT_CHAIN_RIC = '0#.SPX'

        # Candidate names searched under the known Drive data folders when PIT_UNIVERSE_CSV is blank.
        PIT_CANDIDATE_FILENAMES = [
            'sp500_pit_universe.csv',
            'sp500_pit_membership.csv',
            'sp500_pit_daily_membership.csv',
            'sp500_pit_joiner_leaver_20160101_20260513_pit_universe.csv',
        ]

        print('FRED_API_KEY is set:', bool(os.environ.get('FRED_API_KEY')))
        print('PIT_UNIVERSE_CSV:', PIT_UNIVERSE_CSV or '(generate or auto-discover)')
        print('GENERATE_PIT_UNIVERSE:', GENERATE_PIT_UNIVERSE)
        """
    ),
    md("## 3. Validation Matrix"),
    code(
        r"""
        from datetime import datetime
        import itertools

        FAST_MODE = True  # True = smoke run; False = fuller proof matrix.
        RUN_TRAINING = True
        RUN_BACKTESTS = True

        RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
        EXPERIMENT_SLUG = 'pit_universe_validation'
        RUN_ROOT = DRIVE_ROOT / EXPERIMENT_SLUG / RUN_TAG
        TRAINING_OUTPUT_DIR = RUN_ROOT / 'training_runs'
        PIT_OUTPUT_DIR = REPO_DIR / 'data' / 'raw' / 'constituents' / 'pit_validation'
        for path in [RUN_ROOT, TRAINING_OUTPUT_DIR, PIT_OUTPUT_DIR, RUN_ROOT / 'logs']:
            path.mkdir(parents=True, exist_ok=True)

        BASE_SEEDS = [1729, 2718, 3141]
        ACTIVE_BASE_SEEDS = [1729] if FAST_MODE else BASE_SEEDS
        NUM_MODELS = 3 if FAST_MODE else 20
        NUM_EPOCHS = 15 if FAST_MODE else 100
        EARLY_STOPPING_PATIENCE = 5 if FAST_MODE else 15
        BATCH_SIZE = 32
        LEARNING_RATE = '5e-5'

        MODEL_RECIPE = {
            'name': 'static-threshold-shuffle__pure-ic-returns-5d-val-ic__regime-current-only__ensemble__drop-edge-0p1',
            'label_t': 5,
            'loss_type': 'ic',
            'label_type': 'returns',
            'selection_metric': 'val_ic',
            'drop_edge_p': 0.1,
        }

        REGIME_INPUTS_CSV = ''
        REGIME_STRICT = True
        REGIME_ENFORCE_LAG_DAYS = 0

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
                'name': 'full',
                'description': 'Frozen full model with current-only global regime features.',
                'overrides': [],
            },
            {
                'name': 'no_regime',
                'description': 'Same frozen recipe with global regime features disabled.',
                'overrides': [
                    'features.include_global_regime=false',
                    'features.regime_strict=false',
                    'features.regime_include_subsequent_returns=false',
                ],
            },
        ]
        ACTIVE_MODEL_VARIANTS = ['full'] if FAST_MODE else ['full', 'no_regime']

        UNIVERSE_CONTROLS = [
            {
                'name': 'baseline',
                'description': 'Current complete-stock filter across train through test.',
                'requires_pit': False,
                'overrides': [],
            },
            {
                'name': 'per_split_filter',
                'description': 'Complete-stock filtering per train/val/test split.',
                'requires_pit': False,
                'overrides': ['+data.filter_stocks_per_split=true'],
            },
            {
                'name': 'pit_universe',
                'description': 'PIT row validity filter before normalization and stock filtering.',
                'requires_pit': True,
                'overrides': ['+data.use_pit_universe=true', '+data.pit_universe_csv={pit_csv}'],
            },
            {
                'name': 'pit_plus_per_split',
                'description': 'PIT row validity plus per-split complete-stock filtering.',
                'requires_pit': True,
                'overrides': [
                    '+data.use_pit_universe=true',
                    '+data.pit_universe_csv={pit_csv}',
                    '+data.filter_stocks_per_split=true',
                ],
            },
        ]
        ACTIVE_UNIVERSE_CONTROLS = ['baseline', 'pit_plus_per_split'] if FAST_MODE else [
            'baseline',
            'per_split_filter',
            'pit_universe',
            'pit_plus_per_split',
        ]

        TOP_K_VALUES = [15, 20]
        COST_SCENARIOS = [
            {'cost_name': 'spread5_slip0', 'spread_bps': 5.0, 'slippage_bps': 0.0},
            {'cost_name': 'spread10_slip2', 'spread_bps': 10.0, 'slippage_bps': 2.0},
        ]
        RANK_DROP_GATE = {'enabled': True, 'min_rank_drop': 30}
        HOLDING_PERIOD = 1
        REBALANCE_STYLE = 'staggered'
        NUM_TESTS_OVERRIDE = None

        print('RUN_ROOT:', RUN_ROOT)
        print('FAST_MODE:', FAST_MODE)
        print('Active years:', [w['test_year'] for w in PROOF_WINDOWS if w['enabled']])
        print('Active seeds:', ACTIVE_BASE_SEEDS)
        print('Active variants:', ACTIVE_MODEL_VARIANTS)
        print('Active universe controls:', ACTIVE_UNIVERSE_CONTROLS)
        """
    ),
    md("## 4. Data And PIT File Preparation"),
    code(
        r"""
        DATA_ROOTS = [
            DRIVE_DATA_DIR,
            DRIVE_STOCK_UNIVERSE_DIR,
            Path('/content/drive/MyDrive') if IN_COLAB else DRIVE_DATA_DIR,
            Path('/content/drive/MyDrive/MCI_GRU_shared/data') if IN_COLAB else DRIVE_DATA_DIR,
            Path('/content/drive/MyDrive/Stock universe data') if IN_COLAB else DRIVE_DATA_DIR,
        ]
        DATA_ROOTS = [p for p in DATA_ROOTS if p is not None]

        def existing_path(path_value: str | Path) -> Path | None:
            if not path_value:
                return None
            path = Path(path_value).expanduser()
            if path.exists():
                return path
            repo_path = REPO_DIR / path
            if repo_path.exists():
                return repo_path
            return None

        def copy_data_file(filename: str) -> Path:
            dest = REPO_DIR / 'data' / 'raw' / 'market' / filename
            if dest.exists():
                return dest
            for root in DATA_ROOTS:
                source = root / filename
                if source.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    print('Copied data file:', source, '->', dest)
                    return dest
            searched = [str(root / filename) for root in DATA_ROOTS]
            raise FileNotFoundError(f'Missing {filename}. Searched: {searched}')

        def normalize_pit_schema(source: Path, dest: Path) -> Path:
            pit = pd.read_csv(source)
            pit.columns = [str(c).strip().lower() for c in pit.columns]
            rename_map = {}
            if 'ticker' in pit.columns and 'kdcode' not in pit.columns:
                rename_map['ticker'] = 'kdcode'
            if 'start_date' in pit.columns and 'valid_from' not in pit.columns:
                rename_map['start_date'] = 'valid_from'
            if 'end_date' in pit.columns and 'valid_to' not in pit.columns:
                rename_map['end_date'] = 'valid_to'
            pit = pit.rename(columns=rename_map)
            required = {'kdcode', 'valid_from', 'valid_to'}
            missing = required - set(pit.columns)
            if missing:
                raise ValueError(f'PIT CSV {source} is missing required columns: {sorted(missing)}')
            pit = pit[['kdcode', 'valid_from', 'valid_to']].copy()
            pit['kdcode'] = pit['kdcode'].astype(str)
            pit['valid_from'] = pd.to_datetime(pit['valid_from']).dt.strftime('%Y-%m-%d')
            pit['valid_to'] = pd.to_datetime(pit['valid_to']).dt.strftime('%Y-%m-%d')
            suffix_mask = pit['kdcode'].str.contains('^', regex=False, na=False)
            aliases = pit.loc[suffix_mask].copy()
            if not aliases.empty:
                aliases['kdcode'] = aliases['kdcode'].str.split('^', n=1).str[0].str.strip()
                aliases = aliases[aliases['kdcode'] != '']
                pit = pd.concat([pit, aliases], ignore_index=True)
            pit['_valid_from_dt'] = pd.to_datetime(pit['valid_from'])
            pit['_valid_to_dt'] = pd.to_datetime(pit['valid_to'])
            pit = pit.sort_values(['kdcode', '_valid_from_dt', '_valid_to_dt'])
            rows = []
            for kdcode, group in pit.groupby('kdcode', sort=True):
                current_start = None
                current_end = None
                for row_start, row_end in group[['_valid_from_dt', '_valid_to_dt']].itertuples(index=False, name=None):
                    if current_start is None or current_end is None:
                        current_start = row_start
                        current_end = row_end
                    elif row_start <= current_end + pd.Timedelta(days=1):
                        current_end = max(current_end, row_end)
                    else:
                        rows.append({
                            'kdcode': kdcode,
                            'valid_from': current_start.strftime('%Y-%m-%d'),
                            'valid_to': current_end.strftime('%Y-%m-%d'),
                        })
                        current_start = row_start
                        current_end = row_end
                if current_start is not None and current_end is not None:
                    rows.append({
                        'kdcode': kdcode,
                        'valid_from': current_start.strftime('%Y-%m-%d'),
                        'valid_to': current_end.strftime('%Y-%m-%d'),
                    })
            pit = pd.DataFrame(rows, columns=['kdcode', 'valid_from', 'valid_to'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            pit.to_csv(dest, index=False)
            return dest

        def find_pit_file() -> Path | None:
            explicit = existing_path(PIT_UNIVERSE_CSV)
            if explicit is not None:
                return explicit
            for root in DATA_ROOTS + [DRIVE_ROOT, REPO_DIR / 'data' / 'raw' / 'constituents']:
                for filename in PIT_CANDIDATE_FILENAMES:
                    candidate = root / filename
                    if candidate.exists():
                        return candidate
            return None

        def repo_or_abs(path: Path) -> str:
            try:
                return path.relative_to(REPO_DIR).as_posix()
            except ValueError:
                return path.as_posix()

        def run_pit_export() -> Path:
            export_dir = REPO_DIR / 'data' / 'raw' / 'constituents'
            export_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(REPO_DIR / 'scripts' / 'data' / 'export_sp500_joiner_leaver_pit.py'),
                '--start',
                PIT_EXPORT_START,
                '--end',
                PIT_EXPORT_END,
                '--index-ric',
                PIT_EXPORT_INDEX_RIC,
                '--chain-ric',
                PIT_EXPORT_CHAIN_RIC,
                '--output-dir',
                str(export_dir),
            ]
            print('Generating PIT universe with:', ' '.join(cmd))
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            (RUN_ROOT / 'logs' / 'pit_export_stdout.log').write_text(proc.stdout, encoding='utf-8')
            (RUN_ROOT / 'logs' / 'pit_export_stderr.log').write_text(proc.stderr, encoding='utf-8')
            print(proc.stdout[-5000:])
            if proc.returncode != 0:
                print(proc.stderr[-5000:])
                raise RuntimeError(
                    'PIT export failed. Run this notebook where Refinitiv Workspace/LSEG access is available, '
                    'or set PIT_UNIVERSE_CSV to an existing *_pit_universe.csv in Drive.'
                )
            safe_start = PIT_EXPORT_START.replace('-', '')
            safe_end = PIT_EXPORT_END.replace('-', '')
            pit_path = export_dir / f'sp500_pit_joiner_leaver_{safe_start}_{safe_end}_pit_universe.csv'
            if not pit_path.exists():
                raise FileNotFoundError(f'PIT export completed but did not create {pit_path}')
            drive_copy = RUN_ROOT / pit_path.name
            shutil.copy2(pit_path, drive_copy)
            meta_path = pit_path.with_name(pit_path.name.replace('_pit_universe.csv', '_meta.json'))
            if meta_path.exists():
                shutil.copy2(meta_path, RUN_ROOT / meta_path.name)
            return pit_path

        pit_source = find_pit_file()
        generated_pit = False
        if pit_source is None and GENERATE_PIT_UNIVERSE:
            pit_source = run_pit_export()
            generated_pit = True

        if pit_source is None:
            raise FileNotFoundError(
                'No PIT universe CSV found. Set PIT_UNIVERSE_CSV, place one of PIT_CANDIDATE_FILENAMES in Drive, '
                'or leave GENERATE_PIT_UNIVERSE=True in an LSEG-enabled environment.'
            )

        pit_universe_normalized = normalize_pit_schema(
            pit_source,
            PIT_OUTPUT_DIR / 'sp500_pit_universe_normalized.csv',
        )
        pit_source_kind = 'missing'
        pit_source_kind = 'generated_joiner_leaver' if generated_pit else 'existing_pit'
        print('Using PIT universe:', pit_source)
        print('Normalized PIT universe:', pit_universe_normalized)

        for window in PROOF_WINDOWS:
            if not window['enabled']:
                continue
            data_path = copy_data_file(window['data_filename'])
            window['repo_data_path'] = repo_or_abs(data_path)
            window['pit_universe_csv'] = repo_or_abs(pit_universe_normalized)
            window['pit_source_kind'] = pit_source_kind

        display(pd.DataFrame([
            {
                'test_year': w['test_year'],
                'data_filename': w['data_filename'],
                'repo_data_path': w.get('repo_data_path', ''),
                'pit_universe_csv': w.get('pit_universe_csv', ''),
                'pit_source_kind': w.get('pit_source_kind', ''),
            }
            for w in PROOF_WINDOWS
            if w['enabled']
        ]))
        """
    ),
    md("## 5. Build Training And Backtest Jobs"),
    code(
        r"""
        BASE_OVERRIDES = [
            'training.lr_scheduler=cosine',
            f'training.learning_rate={LEARNING_RATE}',
            f'training.num_epochs={NUM_EPOCHS}',
            f'training.num_models={NUM_MODELS}',
            f'training.early_stopping_patience={EARLY_STOPPING_PATIENCE}',
            f'training.batch_size={BATCH_SIZE}',
            f'features.include_global_regime={str(REGIME_STRICT).lower()}',
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
            'tracking.enabled=false',
        ]
        if REGIME_INPUTS_CSV:
            BASE_OVERRIDES.append(f'features.regime_inputs_csv={REGIME_INPUTS_CSV}')

        def safe_name(value: str, max_len: int = 150) -> str:
            cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
            return cleaned if len(cleaned) <= max_len else cleaned[:max_len]

        def control_overrides(control: dict, window: dict) -> list[str]:
            out = []
            for item in control['overrides']:
                out.append(item.format(pit_csv=window['pit_universe_csv']))
            return out

        selected_controls = [c for c in UNIVERSE_CONTROLS if c['name'] in ACTIVE_UNIVERSE_CONTROLS]
        selected_variants = [v for v in MODEL_VARIANTS if v['name'] in ACTIVE_MODEL_VARIANTS]

        training_jobs = []
        for window in [w for w in PROOF_WINDOWS if w['enabled']]:
            for control in selected_controls:
                if control['requires_pit'] and not window.get('pit_universe_csv'):
                    raise RuntimeError(f"Universe control {control['name']} requires PIT csv for {window['test_year']}")
                for variant in selected_variants:
                    for base_seed in ACTIVE_BASE_SEEDS:
                        name = safe_name(
                            f"{MODEL_RECIPE['name']}__{control['name']}__{variant['name']}__base-seed-{base_seed}__test-{window['test_year']}"
                        )
                        overrides = [
                            *BASE_OVERRIDES,
                            *control_overrides(control, window),
                            *variant['overrides'],
                            f"seed={base_seed}",
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
                        training_jobs.append({
                            **{k: window[k] for k in [
                                'test_year',
                                'data_config',
                                'data_filename',
                                'repo_data_path',
                                'train_start',
                                'train_end',
                                'val_start',
                                'val_end',
                                'test_start',
                                'test_end',
                                'pit_universe_csv',
                                'pit_source_kind',
                            ]},
                            'universe_control': control['name'],
                            'universe_control_description': control['description'],
                            'model_variant': variant['name'],
                            'model_variant_description': variant['description'],
                            'base_seed': base_seed,
                            'name': name,
                            'overrides': overrides,
                        })

        backtest_scenarios = []
        for top_k, cost in itertools.product(TOP_K_VALUES, COST_SCENARIOS):
            backtest_scenarios.append({
                'scenario': f"k{top_k}_{cost['cost_name']}_rankdrop{RANK_DROP_GATE['min_rank_drop']}_daily",
                'top_k': top_k,
                'transaction_costs': True,
                'spread_bps': cost['spread_bps'],
                'slippage_bps': cost['slippage_bps'],
                'rank_drop_gate': RANK_DROP_GATE['enabled'],
                'min_rank_drop': RANK_DROP_GATE['min_rank_drop'],
                'holding_period': HOLDING_PERIOD,
                'rebalance_style': REBALANCE_STYLE,
            })

        training_matrix_df = pd.DataFrame(training_jobs)
        scenario_df = pd.DataFrame(backtest_scenarios)
        display(training_matrix_df[['test_year', 'universe_control', 'model_variant', 'base_seed', 'data_filename', 'pit_source_kind', 'name']])
        display(scenario_df)
        print('Training jobs:', len(training_jobs))
        print('Backtests after training:', len(training_jobs) * len(backtest_scenarios))
        """
    ),
    md("## 6. Execution Helpers"),
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
            print('Control:', job['universe_control'], '| Variant:', job['model_variant'], '| Year:', job['test_year'])
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
                **{k: job[k] for k in [
                    'test_year',
                    'universe_control',
                    'model_variant',
                    'base_seed',
                    'name',
                    'data_config',
                    'data_filename',
                    'pit_universe_csv',
                    'pit_source_kind',
                    'train_start',
                    'train_end',
                    'val_start',
                    'val_end',
                    'test_start',
                    'test_end',
                ]},
            }
            if run_dir:
                metadata = read_json(run_dir / 'run_metadata.json')
                row['n_stocks'] = len(metadata.get('kdcode_list', []))
                row.update({f'training_summary.{k}': v for k, v in flatten_dict(read_json(run_dir / 'training_summary.json')).items()})
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
            copy_dir = (
                RUN_ROOT
                / 'backtests'
                / str(training_row['test_year'])
                / training_row['universe_control']
                / training_row['model_variant']
                / f"base_seed_{training_row['base_seed']}"
                / scenario['scenario']
            )
            if source_dir.exists():
                if copy_dir.exists():
                    shutil.rmtree(copy_dir)
                shutil.copytree(source_dir, copy_dir)

            row = {
                'status': 'OK' if proc.returncode == 0 else 'FAILED',
                'returncode': proc.returncode,
                'test_year': training_row['test_year'],
                'universe_control': training_row['universe_control'],
                'model_variant': training_row['model_variant'],
                'base_seed': training_row['base_seed'],
                'n_stocks': training_row.get('n_stocks', np.nan),
                'pit_source_kind': training_row.get('pit_source_kind', ''),
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
        manifest_path = RUN_ROOT / 'pit_universe_validation_manifest.json'
        manifest_path.write_text(json.dumps({
            'run_tag': RUN_TAG,
            'run_root': str(RUN_ROOT),
            'fast_mode': FAST_MODE,
            'drive_data_dir': str(DRIVE_DATA_DIR),
            'pit_universe_csv_input': PIT_UNIVERSE_CSV,
            'pit_source_kind': pit_source_kind,
            'model_recipe': MODEL_RECIPE,
            'base_seeds': BASE_SEEDS,
            'active_base_seeds': ACTIVE_BASE_SEEDS,
            'proof_windows': PROOF_WINDOWS,
            'model_variants': MODEL_VARIANTS,
            'active_model_variants': ACTIVE_MODEL_VARIANTS,
            'universe_controls': UNIVERSE_CONTROLS,
            'active_universe_controls': ACTIVE_UNIVERSE_CONTROLS,
            'top_k_values': TOP_K_VALUES,
            'cost_scenarios': COST_SCENARIOS,
            'rank_drop_gate': RANK_DROP_GATE,
        }, indent=2), encoding='utf-8')

        training_rows = []
        if RUN_TRAINING:
            for job in training_jobs:
                training_rows.append(run_training_job(job))
                pd.DataFrame(training_rows).to_csv(RUN_ROOT / 'pit_training_results_interim.csv', index=False)
        else:
            for job in training_jobs:
                run_dir = latest_run_dir(job['name'])
                training_rows.append({
                    'status': 'OK' if run_dir else 'FAILED',
                    'returncode': 0 if run_dir else 1,
                    'run_dir': str(run_dir) if run_dir else '',
                    'predictions_dir': str(run_dir / 'averaged_predictions') if run_dir else '',
                    **{k: job[k] for k in [
                        'test_year',
                        'universe_control',
                        'model_variant',
                        'base_seed',
                        'name',
                        'data_config',
                        'data_filename',
                        'pit_universe_csv',
                        'pit_source_kind',
                        'train_start',
                        'train_end',
                        'val_start',
                        'val_end',
                        'test_start',
                        'test_end',
                    ]},
                })

        training_df = pd.DataFrame(training_rows)
        training_path = RUN_ROOT / 'pit_training_results.csv'
        training_df.to_csv(training_path, index=False)
        display(training_df[[c for c in ['status', 'test_year', 'universe_control', 'model_variant', 'base_seed', 'n_stocks', 'elapsed_minutes', 'run_dir'] if c in training_df.columns]])

        backtest_rows = []
        if RUN_BACKTESTS:
            ok_training = training_df[training_df['status'].eq('OK')].copy()
            num_tests = NUM_TESTS_OVERRIDE or (len(ok_training) * len(backtest_scenarios))
            for _, train_row in ok_training.iterrows():
                for scenario in backtest_scenarios:
                    backtest_rows.append(run_backtest(train_row, scenario, num_tests))
                    pd.DataFrame(backtest_rows).to_csv(RUN_ROOT / 'pit_backtest_results_interim.csv', index=False)

        backtest_df = pd.DataFrame(backtest_rows)
        raw_backtest_path = RUN_ROOT / 'pit_backtest_results_raw.csv'
        backtest_df.to_csv(raw_backtest_path, index=False)
        display(backtest_df[[c for c in ['status', 'test_year', 'universe_control', 'model_variant', 'base_seed', 'scenario', 'backtest.excess_return', 'backtest.ASR', 'backtest.MDD', 'n_stocks'] if c in backtest_df.columns]])
        """
    ),
    md("## 8. Build Comparison Tables"),
    code(
        r"""
        def add_decision_score(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in ['backtest.ASR', 'backtest.excess_return', 'backtest.MDD', 'backtest.avg_daily_turnover']:
                if col not in out.columns:
                    out[col] = np.nan
            out['decision_score'] = (
                out['backtest.ASR'].astype(float).fillna(0.0) * 0.35
                + out['backtest.excess_return'].astype(float).fillna(0.0) * 0.35
                + out['backtest.MDD'].astype(float).fillna(0.0) * 0.20
                - out['backtest.avg_daily_turnover'].astype(float).fillna(0.0) * 0.10
            )
            return out

        decision_df = add_decision_score(backtest_df) if len(backtest_df) else pd.DataFrame()
        if len(decision_df):
            decision_df = decision_df.sort_values(
                ['status', 'test_year', 'model_variant', 'universe_control', 'base_seed', 'decision_score'],
                ascending=[True, True, True, True, True, False],
            )

        key_cols = ['test_year', 'model_variant', 'base_seed', 'scenario']
        if len(decision_df) and 'baseline' in set(decision_df['universe_control']):
            metric_cols = ['backtest.excess_return', 'backtest.ASR', 'backtest.MDD', 'n_stocks', 'decision_score']
            baseline = decision_df[decision_df['universe_control'].eq('baseline')][key_cols + metric_cols].copy()
            baseline = baseline.rename(columns={c: f'baseline.{c}' for c in metric_cols})
            controls = decision_df[~decision_df['universe_control'].eq('baseline')].copy()
            comparison_df = controls.merge(baseline, on=key_cols, how='left')
            for metric in metric_cols:
                comparison_df[f'delta.{metric}'] = comparison_df[metric] - comparison_df[f'baseline.{metric}']
        else:
            comparison_df = decision_df.copy()

        decision_path = RUN_ROOT / 'pit_vs_baseline_decision_table.csv'
        html_path = RUN_ROOT / 'pit_vs_baseline_decision_table.html'
        comparison_df.to_csv(decision_path, index=False)
        comparison_df.to_html(html_path, index=False)

        display_cols = [
            'status',
            'test_year',
            'universe_control',
            'model_variant',
            'base_seed',
            'scenario',
            'n_stocks',
            'baseline.n_stocks',
            'delta.n_stocks',
            'backtest.excess_return',
            'baseline.backtest.excess_return',
            'delta.backtest.excess_return',
            'backtest.ASR',
            'baseline.backtest.ASR',
            'delta.backtest.ASR',
            'backtest.MDD',
            'decision_score',
        ]
        display(comparison_df[[c for c in display_cols if c in comparison_df.columns]])
        print('Decision CSV:', decision_path)
        print('Decision HTML:', html_path)
        """
    ),
    md("## 9. Pooled Daily Significance"),
    code(
        r"""
        def daily_file_for(row: pd.Series) -> Path:
            copied = Path(str(row.get('copied_backtest_dir', '')))
            return copied / 'daily_returns.csv'

        pooled_rows = []
        if len(decision_df):
            for _, row in decision_df.iterrows():
                daily_path = daily_file_for(row)
                if not daily_path.exists():
                    continue
                daily = pd.read_csv(daily_path)
                if 'excess_return' not in daily.columns:
                    continue
                daily['test_year'] = row['test_year']
                daily['universe_control'] = row['universe_control']
                daily['model_variant'] = row['model_variant']
                daily['base_seed'] = row['base_seed']
                daily['scenario'] = row['scenario']
                pooled_rows.append(daily)

        pooled_daily = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
        pooled_daily_path = RUN_ROOT / 'pit_pooled_daily_returns.csv'
        pooled_daily.to_csv(pooled_daily_path, index=False)

        sig_rows = []
        if len(pooled_daily):
            for keys, group in pooled_daily.groupby(['universe_control', 'model_variant', 'scenario']):
                excess = pd.to_numeric(group['excess_return'], errors='coerce').dropna().to_numpy()
                n_days = int(len(excess))
                if n_days < 2:
                    continue
                mean_daily = float(np.mean(excess))
                vol = float(np.std(excess, ddof=1) * math.sqrt(252))
                ann_excess = float((1.0 + mean_daily) ** 252 - 1.0)
                ir = float(ann_excess / vol) if vol else np.nan
                stderr = float(np.std(excess, ddof=1) / math.sqrt(n_days))
                t_stat = float(mean_daily / stderr) if stderr else np.nan
                p_value = float(math.erfc(abs(t_stat) / math.sqrt(2))) if np.isfinite(t_stat) else np.nan
                sig_rows.append({
                    'universe_control': keys[0],
                    'model_variant': keys[1],
                    'scenario': keys[2],
                    'n_days': n_days,
                    'mean_daily_excess': mean_daily,
                    'annualized_excess': ann_excess,
                    'annualized_excess_vol': vol,
                    'annualized_information_ratio': ir,
                    't_statistic': t_stat,
                    'p_value_two_sided_normal_approx': p_value,
                    'sig_p05': bool(p_value < 0.05) if np.isfinite(p_value) else False,
                    'sig_p01': bool(p_value < 0.01) if np.isfinite(p_value) else False,
                })

        significance_df = pd.DataFrame(sig_rows)
        significance_path = RUN_ROOT / 'pit_pooled_daily_significance.csv'
        significance_df.to_csv(significance_path, index=False)
        display(significance_df.sort_values('annualized_information_ratio', ascending=False) if len(significance_df) else significance_df)
        print('Pooled daily returns:', pooled_daily_path)
        print('Pooled significance:', significance_path)
        """
    ),
    md("## 10. Write Summary"),
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
            '# MCI-GRU PIT Universe Validation Summary',
            '',
            f'Run root: `{RUN_ROOT}`',
            f'PIT source kind: `{pit_source_kind}`',
            f'FAST_MODE: `{FAST_MODE}`',
            f'Model recipe: `{MODEL_RECIPE["name"]}`',
            '',
            '## What This Tests',
            '',
            '- Whether the frozen recipe survives `data.use_pit_universe=true`.',
            '- Whether per-split completeness changes the result relative to the full-period complete-stock baseline.',
            '- Whether stock counts collapse when survivorship controls are applied.',
            '- Whether pooled daily excess return remains positive after transaction costs.',
            '',
            '## Artifact Paths',
            '',
            f'- Manifest: `{RUN_ROOT / "pit_universe_validation_manifest.json"}`',
            f'- Training results: `{RUN_ROOT / "pit_training_results.csv"}`',
            f'- Raw backtest results: `{RUN_ROOT / "pit_backtest_results_raw.csv"}`',
            f'- Comparison decision table: `{RUN_ROOT / "pit_vs_baseline_decision_table.csv"}`',
            f'- Pooled daily significance: `{RUN_ROOT / "pit_pooled_daily_significance.csv"}`',
            '',
        ]

        if len(comparison_df):
            summary_cols = [
                'test_year',
                'universe_control',
                'model_variant',
                'base_seed',
                'scenario',
                'n_stocks',
                'baseline.n_stocks',
                'delta.n_stocks',
                'backtest.excess_return',
                'baseline.backtest.excess_return',
                'delta.backtest.excess_return',
                'backtest.ASR',
                'delta.backtest.ASR',
                'backtest.MDD',
            ]
            report = comparison_df[[c for c in summary_cols if c in comparison_df.columns]].head(40).copy()
            for col in report.columns:
                if 'excess_return' in col or col.endswith('MDD') or col.startswith('delta.backtest.excess_return'):
                    report[col] = report[col].map(pct)
            lines.extend(['## Top Comparison Rows', '', report.to_markdown(index=False), ''])

        if len(significance_df):
            sig = significance_df.sort_values('annualized_information_ratio', ascending=False).head(20).copy()
            for col in ['mean_daily_excess', 'annualized_excess', 'annualized_excess_vol']:
                sig[col] = sig[col].map(pct)
            lines.extend(['## Pooled Daily Significance', '', sig.to_markdown(index=False), ''])

        summary_path = RUN_ROOT / 'pit_universe_validation_summary.md'
        summary_path.write_text('\n'.join(lines), encoding='utf-8')
        print(summary_path.read_text(encoding='utf-8'))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
