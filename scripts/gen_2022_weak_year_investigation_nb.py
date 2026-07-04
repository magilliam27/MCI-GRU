"""Generate the guided notebook for investigating weak 2022 backtests."""

from __future__ import annotations

from pathlib import Path

from nb_lib import code_lines as code
from nb_lib import md_lines as md
from nb_lib import write_notebook

OUT = Path("notebooks/2022_weak_year_investigation.ipynb")


cells = [
    md(
        """
        # MCI-GRU 2022 Weak-Year Investigation

        This notebook is a guided test harness for explaining why 2022 was weak in the latest completed backtest proof grid. It is designed to be rerunnable: each section states a hypothesis, reproduces the relevant evidence, and records a `PASS`, `WARN`, or `FAIL` style result.

        Primary artifacts discovered in Drive:

        - `/content/drive/MyDrive/MCI-GRU-Ablations/performance_proof_missing_grid/20260505_030758/completed_proof_decision_table.csv`
        - `/content/drive/MyDrive/MCI-GRU-Ablations/performance_proof_missing_grid/20260505_030758/completed_pooled_daily_returns.csv`
        - `/content/drive/MyDrive/MCI-GRU-Ablations/performance_proof_missing_grid/20260505_030758/20260505_030758.zip`

        If you move artifacts, set `WEAK_YEAR_ARTIFACT_DIR` to the folder containing the two CSVs, or set `WEAK_YEAR_ZIP` to a zip containing those CSVs.

        Working vocabulary:

        - **2022 walk-forward segment**: the temporal test segment using the 2016-universe training window.
        - **Excess return**: strategy return minus benchmark return for the same evaluation window.
        - **Rank-drop gate**: portfolio policy that exits a held name only when its rank worsens by at least `min_rank_drop`.
        """
    ),
    md("## 1. Setup"),
    code(
        r"""
        from pathlib import Path
        import os
        import re
        import subprocess
        import sys
        import zipfile

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
        if not (REPO_DIR / 'AGENTS.md').exists() and Path.cwd().name == 'notebooks':
            REPO_DIR = Path.cwd().parent

        if IN_COLAB:
            if not REPO_DIR.exists():
                subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
            else:
                subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin'], check=True)
                subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', BRANCH], check=True)
                subprocess.run(['git', '-C', str(REPO_DIR), 'pull', 'origin', BRANCH], check=True)

        os.chdir(REPO_DIR)

        DRIVE_ROOT = Path('/content/drive/MyDrive/MCI-GRU-Ablations') if IN_COLAB else REPO_DIR / 'drive_outputs'
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

        REQUIRED_ARTIFACT_FILES = [
            'completed_proof_decision_table.csv',
            'completed_pooled_daily_returns.csv',
        ]
        RUN_TAG = '20260505_030758'
        DEFAULT_RUN_DIR = DRIVE_ROOT / 'performance_proof_missing_grid' / RUN_TAG
        DEFAULT_WEAK_YEAR_DIR = DRIVE_ROOT / 'weak_year_diagnostic'
        ARTIFACT_DIR_OVERRIDE = os.environ.get('WEAK_YEAR_ARTIFACT_DIR', '').strip()
        ZIP_PATH_OVERRIDE = os.environ.get('WEAK_YEAR_ZIP', '').strip()

        artifact_candidates = []
        if ARTIFACT_DIR_OVERRIDE:
            artifact_candidates.append(Path(ARTIFACT_DIR_OVERRIDE))
        artifact_candidates.extend([
            DEFAULT_RUN_DIR,
            DEFAULT_WEAK_YEAR_DIR,
            REPO_DIR / 'drive_outputs' / 'weak_year_diagnostic',
        ])

        def has_required_artifacts(path: Path) -> bool:
            return all((path / name).exists() for name in REQUIRED_ARTIFACT_FILES)

        def first_existing_zip(candidates: list[Path]) -> Path | None:
            zip_candidates = []
            if ZIP_PATH_OVERRIDE:
                zip_candidates.append(Path(ZIP_PATH_OVERRIDE))
            for candidate in candidates:
                zip_candidates.extend([
                    candidate / f'{RUN_TAG}.zip',
                    candidate.with_suffix('.zip'),
                ])
            zip_candidates.extend([
                DRIVE_ROOT / 'performance_proof_missing_grid' / RUN_TAG / f'{RUN_TAG}.zip',
                DRIVE_ROOT / 'performance_proof_missing_grid' / f'{RUN_TAG}.zip',
                DRIVE_ROOT / 'weak_year_diagnostic' / f'{RUN_TAG}.zip',
            ])
            for candidate in zip_candidates:
                if candidate.exists():
                    return candidate
            return None

        def extract_required_artifacts(zip_path: Path, destination: Path) -> bool:
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                for required in REQUIRED_ARTIFACT_FILES:
                    matches = [
                        name for name in names
                        if name == required or name.endswith('/' + required)
                    ]
                    if not matches:
                        return False
                    with zf.open(matches[0]) as src, (destination / required).open('wb') as dst:
                        dst.write(src.read())
            return True

        ARTIFACT_DIR = next((p for p in artifact_candidates if has_required_artifacts(p)), None)
        ZIP_PATH = first_existing_zip(artifact_candidates)

        if ARTIFACT_DIR is None and ZIP_PATH is not None:
            ARTIFACT_DIR = DEFAULT_WEAK_YEAR_DIR
            extracted = extract_required_artifacts(ZIP_PATH, ARTIFACT_DIR)
            if not extracted:
                raise FileNotFoundError(f'Zip did not contain required weak-year CSVs: {ZIP_PATH}')

        if ARTIFACT_DIR is None:
            searched = '\n'.join(f'  - {candidate}' for candidate in artifact_candidates)
            raise FileNotFoundError(
                'Missing weak-year artifacts. Expected both '
                f"{', '.join(REQUIRED_ARTIFACT_FILES)}. Searched artifact dirs:\n"
                f'{searched}\n'
                'Set WEAK_YEAR_ARTIFACT_DIR to the folder containing the CSVs, '
                'or set WEAK_YEAR_ZIP to a zip containing them.'
            )

        if ZIP_PATH is None:
            ZIP_PATH = ARTIFACT_DIR / f'{RUN_TAG}.zip'

        DECISION_CSV = ARTIFACT_DIR / 'completed_proof_decision_table.csv'
        DAILY_CSV = ARTIFACT_DIR / 'completed_pooled_daily_returns.csv'
        OUTPUT_DIR = ARTIFACT_DIR / 'notebook_outputs'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print('Repo:', REPO_DIR)
        print('Drive output root:', DRIVE_ROOT)
        print('Artifact dir:', ARTIFACT_DIR)
        print('Zip path:', ZIP_PATH)
        print('Output dir:', OUTPUT_DIR)
        """
    ),
    code(
        r"""
        import importlib
        import subprocess

        REQUIRED_PACKAGES = ['pandas', 'numpy', 'matplotlib']
        OPTIONAL_PACKAGES = ['seaborn']

        missing = []
        for pkg in REQUIRED_PACKAGES:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)

        if missing:
            print('Installing missing packages:', missing)
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns
            HAS_SEABORN = True
        except ImportError:
            sns = None
            HAS_SEABORN = False

        pd.set_option('display.max_columns', 120)
        pd.set_option('display.width', 180)
        plt.style.use('default')
        """
    ),
    md("## 2. Load Artifacts And Build Test Helpers"),
    code(
        r"""
        required_paths = [DECISION_CSV, DAILY_CSV]
        missing_paths = [p for p in required_paths if not p.exists()]
        if missing_paths:
            raise FileNotFoundError('Missing required artifact(s): ' + ', '.join(map(str, missing_paths)))

        decision = pd.read_csv(DECISION_CSV)
        daily_raw = pd.read_csv(DAILY_CSV)

        def parse_scenario(s: str) -> pd.Series:
            m = re.match(r'k(\d+)_spread(\d+)_slip(\d+)_rankdrop(\d+)_daily', str(s))
            if not m:
                return pd.Series({'top_k': np.nan, 'spread_bps': np.nan, 'slippage_bps': np.nan, 'rankdrop': np.nan})
            return pd.Series({
                'top_k': int(m.group(1)),
                'spread_bps': int(m.group(2)),
                'slippage_bps': int(m.group(3)),
                'rankdrop': int(m.group(4)),
            })

        decision = pd.concat([decision, decision['scenario'].apply(parse_scenario)], axis=1)
        daily_raw = pd.concat([daily_raw, daily_raw['scenario'].apply(parse_scenario)], axis=1)

        def parse_mixed_dates(values) -> pd.Series:
            return pd.to_datetime(values, format='mixed').dt.normalize()

        daily_raw['date'] = parse_mixed_dates(daily_raw['date'])
        daily_raw['year_month'] = daily_raw['date'].dt.to_period('M').astype(str)

        def compound_return(values) -> float:
            values = pd.Series(values).dropna().astype(float)
            if values.empty:
                return np.nan
            return float((1.0 + values).prod() - 1.0)

        def annualized_return(values, trading_days: int = 252) -> float:
            values = pd.Series(values).dropna().astype(float)
            if values.empty:
                return np.nan
            total = (1.0 + values).prod()
            return float(total ** (trading_days / len(values)) - 1.0)

        def annualized_sharpe(values, trading_days: int = 252) -> float:
            values = pd.Series(values).dropna().astype(float)
            sd = values.std(ddof=1)
            if values.empty or sd == 0 or pd.isna(sd):
                return np.nan
            return float(values.mean() / sd * np.sqrt(trading_days))

        def date_average_daily(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
            # Average repeated seed/model rows to one row per calendar date and group.
            cols = group_cols + ['date']
            out = (
                df.groupby(cols, dropna=False)
                .agg(
                    rows=('excess_return_daily', 'size'),
                    portfolio_return=('portfolio_return', 'mean'),
                    benchmark_return=('benchmark_return', 'mean'),
                    excess_return_daily=('excess_return_daily', 'mean'),
                )
                .reset_index()
            )
            out['year_month'] = out['date'].dt.to_period('M').astype(str)
            return out

        checks = []

        def check(name: str, condition: bool, detail: str = '', severity: str = 'WARN') -> None:
            status = 'PASS' if bool(condition) else severity
            checks.append({'check': name, 'status': status, 'detail': detail})
            print(f'[{status}] {name}: {detail}')

        print('Decision rows:', len(decision))
        print('Daily rows:', len(daily_raw))
        display(decision.head())
        """
    ),
    code(
        r"""
        check('required_years_present', set(decision['test_year'].astype(int)) >= {2022, 2023, 2024}, str(sorted(decision['test_year'].unique())))
        check('required_variants_present', set(decision['variant']) >= {'full', 'near_zero_graph', 'no_regime'}, str(sorted(decision['variant'].unique())))
        check('three_seed_grid_present', decision['base_seed'].nunique() >= 3, f"seeds={sorted(decision['base_seed'].unique())}")
        check('scenario_parser_ok', decision['top_k'].notna().all(), 'parsed top_k/spread/slippage/rankdrop from scenario names', severity='FAIL')

        grid_counts = (
            decision.groupby(['test_year', 'variant', 'base_seed'])
            .size()
            .rename('rows')
            .reset_index()
            .sort_values(['test_year', 'variant', 'base_seed'])
        )
        display(grid_counts)
        check('balanced_grid_cells', grid_counts['rows'].nunique() == 1, f"row-counts={sorted(grid_counts['rows'].unique())}")
        """
    ),
    md("## 3. Reproduce The Weak-Year Symptom"),
    code(
        r"""
        metric_cols = [
            'backtest.total_return',
            'backtest.benchmark_return',
            'backtest.excess_return',
            'backtest.ARR',
            'backtest.ASR',
            'backtest.MDD',
            'backtest.avg_daily_turnover',
            'backtest.total_transaction_cost',
        ]

        year_variant = (
            decision.groupby(['test_year', 'variant'])[metric_cols]
            .mean()
            .reset_index()
            .sort_values(['variant', 'test_year'])
        )
        display(year_variant.style.format({c: '{:.4f}' for c in metric_cols}))

        pivot_excess = year_variant.pivot(index='test_year', columns='variant', values='backtest.excess_return')
        display(pivot_excess.style.format('{:.2%}'))

        y2022_full = year_variant.query("test_year == 2022 and variant == 'full'")['backtest.excess_return'].iloc[0]
        y2023_full = year_variant.query("test_year == 2023 and variant == 'full'")['backtest.excess_return'].iloc[0]
        y2024_full = year_variant.query("test_year == 2024 and variant == 'full'")['backtest.excess_return'].iloc[0]
        check('2022_full_weaker_than_2023_2024', y2022_full < y2023_full and y2022_full < y2024_full, f"2022={y2022_full:.2%}, 2023={y2023_full:.2%}, 2024={y2024_full:.2%}")

        ax = pivot_excess.plot(kind='bar', figsize=(9, 4), title='Mean Excess Return By Year And Variant')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Mean excess return')
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 4. Hypothesis A - Top-K Concentration Drives 2022 Weakness"),
    code(
        r"""
        topk_2022 = (
            decision.query('test_year == 2022')
            .groupby('top_k')
            .agg(
                rows=('scenario', 'size'),
                mean_excess=('backtest.excess_return', 'mean'),
                mean_asr=('backtest.ASR', 'mean'),
                mean_mdd=('backtest.MDD', 'mean'),
                mean_turnover=('backtest.avg_daily_turnover', 'mean'),
                mean_cost=('backtest.total_transaction_cost', 'mean'),
            )
            .reset_index()
        )
        display(topk_2022.style.format({
            'mean_excess': '{:.2%}',
            'mean_asr': '{:.3f}',
            'mean_mdd': '{:.2%}',
            'mean_turnover': '{:.2%}',
            'mean_cost': '{:.2%}',
        }))

        k5 = topk_2022.loc[topk_2022['top_k'] == 5, 'mean_excess'].iloc[0]
        k15plus = topk_2022.loc[topk_2022['top_k'] >= 15, 'mean_excess'].mean()
        check('top_k_5_is_worse_than_top_k_15plus', k5 < k15plus, f"k5={k5:.2%}, k15plus={k15plus:.2%}")

        ax = topk_2022.plot(x='top_k', y='mean_excess', kind='bar', figsize=(8, 4), legend=False, title='2022 Mean Excess By Top-K')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Mean excess return')
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        no_cost = decision.query("spread_bps == 5 and slippage_bps == 0")
        no_cost_topk = (
            no_cost.groupby(['test_year', 'variant', 'top_k'])
            .agg(
                mean_excess=('backtest.excess_return', 'mean'),
                mean_asr=('backtest.ASR', 'mean'),
                mean_total=('backtest.total_return', 'mean'),
            )
            .reset_index()
        )
        display(no_cost_topk.pivot_table(index=['test_year', 'variant'], columns='top_k', values='mean_excess').style.format('{:.2%}'))

        full_2022_no_cost = no_cost_topk.query("test_year == 2022 and variant == 'full'")
        display(full_2022_no_cost.style.format({'mean_excess': '{:.2%}', 'mean_asr': '{:.3f}', 'mean_total': '{:.2%}'}))
        check(
            'full_2022_no_cost_k15plus_repairs_excess',
            full_2022_no_cost.query('top_k >= 15')['mean_excess'].mean() > 0,
            f"mean k>=15 excess={full_2022_no_cost.query('top_k >= 15')['mean_excess'].mean():.2%}",
        )
        """
    ),
    md("## 5. Hypothesis B - Weakness Is Month-Clustered"),
    code(
        r"""
        daily_by_scenario = date_average_daily(
            daily_raw,
            ['test_year', 'variant', 'scenario', 'base_seed', 'top_k', 'spread_bps', 'slippage_bps', 'rankdrop'],
        )

        # Average across the full grid to one row per date for broad stress mapping.
        all_grid_daily = date_average_daily(daily_raw.query('test_year == 2022'), ['test_year'])
        monthly_all_grid = (
            all_grid_daily.groupby('year_month')
            .agg(
                dates=('date', 'nunique'),
                portfolio_total=('portfolio_return', compound_return),
                benchmark_total=('benchmark_return', compound_return),
                mean_daily_excess=('excess_return_daily', 'mean'),
                hit_rate=('excess_return_daily', lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        monthly_all_grid['simple_excess'] = monthly_all_grid['portfolio_total'] - monthly_all_grid['benchmark_total']
        display(monthly_all_grid.style.format({
            'portfolio_total': '{:.2%}',
            'benchmark_total': '{:.2%}',
            'simple_excess': '{:.2%}',
            'mean_daily_excess': '{:.3%}',
            'hit_rate': '{:.1%}',
        }))

        worst_months = monthly_all_grid.sort_values('simple_excess').head(4)['year_month'].tolist()
        check('weak_months_include_known_2022_stress_pockets', {'2022-04', '2022-06', '2022-12'}.issubset(set(worst_months)), f"worst_months={worst_months}")

        ax = monthly_all_grid.plot(x='year_month', y='simple_excess', kind='bar', figsize=(11, 4), legend=False, title='2022 Monthly Excess - Date-Averaged Full Grid')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Simple excess return')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        focus_scenarios = [
            'k5_spread5_slip0_rankdrop30_daily',
            'k10_spread5_slip0_rankdrop30_daily',
            'k15_spread5_slip0_rankdrop30_daily',
            'k20_spread5_slip0_rankdrop30_daily',
            'k30_spread5_slip0_rankdrop30_daily',
        ]
        focus_daily = date_average_daily(
            daily_raw.query("test_year == 2022 and variant == 'full' and scenario in @focus_scenarios"),
            ['scenario', 'top_k'],
        )
        focus_month = (
            focus_daily.groupby(['scenario', 'top_k', 'year_month'])
            .agg(
                portfolio_total=('portfolio_return', compound_return),
                benchmark_total=('benchmark_return', compound_return),
                mean_daily_excess=('excess_return_daily', 'mean'),
                hit_rate=('excess_return_daily', lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        focus_month['simple_excess'] = focus_month['portfolio_total'] - focus_month['benchmark_total']
        display(focus_month.pivot_table(index='year_month', columns='top_k', values='simple_excess').style.format('{:.2%}'))

        pivot = focus_month.pivot(index='year_month', columns='top_k', values='simple_excess')
        ax = pivot.plot(kind='bar', figsize=(12, 5), title='2022 Full Variant Monthly Excess By Top-K')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Simple excess return')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 6. Hypothesis C - Graph/Regime Are Secondary, Not Primary"),
    code(
        r"""
        paired = decision.pivot_table(
            index=['test_year', 'base_seed', 'scenario'],
            columns='variant',
            values='backtest.excess_return',
            aggfunc='mean',
        ).reset_index()

        paired['full_minus_no_regime'] = paired['full'] - paired['no_regime']
        paired['full_minus_near_zero_graph'] = paired['full'] - paired['near_zero_graph']

        variant_delta = (
            paired.groupby('test_year')[['full_minus_no_regime', 'full_minus_near_zero_graph']]
            .agg(['mean', 'std', 'min', 'max'])
        )
        display(variant_delta.style.format('{:.2%}'))

        y2022_delta = paired.query('test_year == 2022')[['full_minus_no_regime', 'full_minus_near_zero_graph']].mean().abs().max()
        y2022_year_gap = y2023_full - y2022_full
        check(
            'variant_deltas_smaller_than_2022_to_2023_gap',
            y2022_delta < y2022_year_gap,
            f"max abs 2022 variant delta={y2022_delta:.2%}, 2022->2023 full gap={y2022_year_gap:.2%}",
        )
        """
    ),
    md("## 7. Hypothesis D - Rank-Drop Gate Is Too Sticky In Narrow 2022 Portfolios"),
    code(
        r"""
        gate_topk = (
            decision.query('test_year == 2022')
            .groupby('top_k')
            .agg(
                gate_exit_days=('backtest.days_with_gate_exits', 'mean'),
                total_trades=('backtest.total_trades', 'mean'),
                turnover=('backtest.avg_daily_turnover', 'mean'),
                cost=('backtest.total_transaction_cost', 'mean'),
                excess=('backtest.excess_return', 'mean'),
            )
            .reset_index()
        )
        display(gate_topk.style.format({
            'gate_exit_days': '{:.1f}',
            'total_trades': '{:.1f}',
            'turnover': '{:.2%}',
            'cost': '{:.2%}',
            'excess': '{:.2%}',
        }))

        corr_topk_excess = gate_topk['top_k'].corr(gate_topk['excess'])
        corr_turnover_excess = gate_topk['turnover'].corr(gate_topk['excess'])
        check('higher_top_k_improves_2022_excess', corr_topk_excess > 0, f"corr(top_k, excess)={corr_topk_excess:.3f}")
        check('higher_turnover_not_root_cause_in_2022', corr_turnover_excess > 0, f"corr(turnover, excess)={corr_turnover_excess:.3f}")

        ax = gate_topk.plot(x='top_k', y=['excess', 'turnover'], secondary_y='turnover', marker='o', figsize=(9, 4), title='2022 Excess vs Rank-Gate Turnover By Top-K')
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 8. Hypothesis E - 2022 Is A High-Beta Down-Day Problem"),
    code(
        r"""
        sign_rows = []
        selected = [
            ('full', 'k15_spread5_slip0_rankdrop30_daily'),
            ('full', 'k10_spread5_slip0_rankdrop30_daily'),
            ('full', 'k5_spread5_slip0_rankdrop30_daily'),
            ('no_regime', 'k15_spread5_slip0_rankdrop30_daily'),
            ('near_zero_graph', 'k20_spread5_slip0_rankdrop30_daily'),
        ]

        for variant, scenario in selected:
            one = date_average_daily(
                daily_raw.query("variant == @variant and scenario == @scenario"),
                ['test_year', 'variant', 'scenario'],
            )
            for (year, v, s), group in one.groupby(['test_year', 'variant', 'scenario']):
                for bucket, mask in [
                    ('bench_up', group['benchmark_return'] > 0),
                    ('bench_down', group['benchmark_return'] < 0),
                    ('big_up', group['benchmark_return'] > 0.015),
                    ('big_down', group['benchmark_return'] < -0.015),
                ]:
                    sub = group[mask]
                    if sub.empty:
                        continue
                    sign_rows.append({
                        'test_year': year,
                        'variant': v,
                        'scenario': s,
                        'bucket': bucket,
                        'n_dates': len(sub),
                        'mean_excess': sub['excess_return_daily'].mean(),
                        'mean_portfolio': sub['portfolio_return'].mean(),
                        'mean_benchmark': sub['benchmark_return'].mean(),
                        'hit_rate': (sub['excess_return_daily'] > 0).mean(),
                    })

        sign_df = pd.DataFrame(sign_rows)
        display(sign_df.query("variant == 'full' and scenario == 'k15_spread5_slip0_rankdrop30_daily'").style.format({
            'mean_excess': '{:.3%}',
            'mean_portfolio': '{:.3%}',
            'mean_benchmark': '{:.3%}',
            'hit_rate': '{:.1%}',
        }))

        full_k15_2022 = sign_df.query("test_year == 2022 and variant == 'full' and scenario == 'k15_spread5_slip0_rankdrop30_daily'")
        up_excess = full_k15_2022.query("bucket == 'bench_up'")['mean_excess'].iloc[0]
        down_excess = full_k15_2022.query("bucket == 'bench_down'")['mean_excess'].iloc[0]
        check('2022_full_k15_wins_up_days_loses_down_days', up_excess > 0 and down_excess < 0, f"up={up_excess:.3%}, down={down_excess:.3%}")
        """
    ),
    code(
        r"""
        # Optional raw-market context. This reads the temporal universe files directly.
        universe_files = {
            2022: REPO_DIR / 'data/raw/market/sp500_2016_universe_data.csv',
            2023: REPO_DIR / 'data/raw/market/sp500_2017_universe_data.csv',
            2024: REPO_DIR / 'data/raw/market/sp500_2018_universe_data.csv',
        }

        def raw_open_to_open_context(year: int, path: Path, date_set: set[pd.Timestamp]) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()
            usecols = ['kdcode', 'dt', 'open']
            raw = pd.read_csv(path, usecols=usecols, parse_dates=['dt'])
            raw = raw.sort_values(['kdcode', 'dt'])
            raw['next_open'] = raw.groupby('kdcode')['open'].shift(-1)
            raw['open_to_open_return'] = raw['next_open'] / raw['open'] - 1
            raw = raw[raw['dt'].isin(date_set)]
            out = (
                raw.groupby('dt')['open_to_open_return']
                .agg(benchmark_return='mean', xsec_dispersion='std', n_stocks='count')
                .reset_index()
            )
            out['test_year'] = year
            return out

        contexts = []
        for year, path in universe_files.items():
            year_dates = set(parse_mixed_dates(daily_raw.loc[daily_raw['test_year'] == year, 'date']))
            contexts.append(raw_open_to_open_context(year, path, year_dates))
        market_context = pd.concat([c for c in contexts if not c.empty], ignore_index=True) if contexts else pd.DataFrame()

        if market_context.empty:
            print('Raw universe files not available; skipping raw-market context.')
        else:
            context_summary = (
                market_context.groupby('test_year')
                .agg(
                    dates=('dt', 'nunique'),
                    avg_n_stocks=('n_stocks', 'mean'),
                    benchmark_total=('benchmark_return', compound_return),
                    mean_daily=('benchmark_return', 'mean'),
                    annualized_vol=('benchmark_return', lambda s: float(s.std(ddof=1) * np.sqrt(252))),
                    down_day_share=('benchmark_return', lambda s: float((s < 0).mean())),
                    avg_xsec_dispersion=('xsec_dispersion', 'mean'),
                    p95_abs_benchmark=('benchmark_return', lambda s: float(s.abs().quantile(0.95))),
                )
                .reset_index()
            )
            display(context_summary.style.format({
                'avg_n_stocks': '{:.1f}',
                'benchmark_total': '{:.2%}',
                'mean_daily': '{:.3%}',
                'annualized_vol': '{:.2%}',
                'down_day_share': '{:.1%}',
                'avg_xsec_dispersion': '{:.2%}',
                'p95_abs_benchmark': '{:.2%}',
            }))

            vol_2022 = context_summary.query('test_year == 2022')['annualized_vol'].iloc[0]
            vol_later = context_summary.query('test_year in [2023, 2024]')['annualized_vol'].mean()
            check('2022_market_vol_above_2023_2024_average', vol_2022 > vol_later, f"2022={vol_2022:.2%}, later_avg={vol_later:.2%}")
        """
    ),
    md("## 9. Ticker Attribution From Backtest Zip"),
    code(
        r"""
        def zip_member_exists(member: str) -> bool:
            if not ZIP_PATH.exists():
                return False
            with zipfile.ZipFile(ZIP_PATH) as z:
                return member in set(z.namelist())

        def read_zip_csv(member: str) -> pd.DataFrame:
            if not ZIP_PATH.exists():
                raise FileNotFoundError(f'Zip not found: {ZIP_PATH}')
            with zipfile.ZipFile(ZIP_PATH) as z:
                with z.open(member) as f:
                    return pd.read_csv(f)

        def backtest_member(year: int, variant: str, base_seed: int | str, scenario: str, filename: str) -> str:
            return f'20260505_030758/backtests/{year}/{variant}/base_seed_{base_seed}/{scenario}/{filename}'

        ATTR_YEAR = 2022
        ATTR_VARIANT = 'near_zero_graph'
        ATTR_SEED = 3141
        ATTR_SCENARIO = 'k5_spread5_slip0_rankdrop30_daily'

        holdings_member = backtest_member(ATTR_YEAR, ATTR_VARIANT, ATTR_SEED, ATTR_SCENARIO, 'daily_holdings.csv')
        composition_member = backtest_member(ATTR_YEAR, ATTR_VARIANT, ATTR_SEED, ATTR_SCENARIO, 'portfolio_composition.csv')
        returns_member = backtest_member(ATTR_YEAR, ATTR_VARIANT, ATTR_SEED, ATTR_SCENARIO, 'daily_returns.csv')

        if not ZIP_PATH.exists():
            print('Zip artifact not found; set ZIP_PATH to a completed proof bundle before running ticker attribution.')
        elif not zip_member_exists(holdings_member):
            print('Selected rich artifacts are not present in zip. Try a missing_grid_completion row or retrieve imported prior folders.')
            print('Missing member:', holdings_member)
        else:
            holdings = read_zip_csv(holdings_member)
            composition = read_zip_csv(composition_member)
            returns = read_zip_csv(returns_member)
            display(holdings.head())
            print('holding rows:', len(holdings), 'composition rows:', len(composition), 'return days:', len(returns))
            display(composition['status'].value_counts(dropna=False).rename('rows').to_frame())
        """
    ),
    code(
        r"""
        if 'holdings' in globals() and not holdings.empty:
            ticker_attr = (
                holdings.groupby('kdcode')
                .agg(
                    days_held=('entry_date', 'nunique'),
                    total_contribution=('contribution', 'sum'),
                    avg_rank=('rank', 'mean'),
                    avg_stock_return=('stock_return', 'mean'),
                )
                .sort_values('total_contribution')
                .reset_index()
            )
            display(ticker_attr.head(15).style.format({
                'total_contribution': '{:.2%}',
                'avg_rank': '{:.1f}',
                'avg_stock_return': '{:.2%}',
            }))
            display(ticker_attr.tail(10).sort_values('total_contribution', ascending=False).style.format({
                'total_contribution': '{:.2%}',
                'avg_rank': '{:.1f}',
                'avg_stock_return': '{:.2%}',
            }))

            long_held_losers = ticker_attr.query('days_held >= 40 and total_contribution < 0')
            check('ticker_attribution_has_long_held_losers', len(long_held_losers) > 0, f"long-held loser count={len(long_held_losers)}")

            ax = ticker_attr.head(15).plot(x='kdcode', y='total_contribution', kind='bar', figsize=(10, 4), legend=False, title='Worst Cumulative Ticker Contributions')
            ax.axhline(0, color='black', linewidth=1)
            ax.set_ylabel('Total contribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        """
    ),
    code(
        r"""
        if 'holdings' in globals() and not holdings.empty:
            returns['date'] = parse_mixed_dates(returns['date'])
            worst_days = returns.sort_values('excess_return').head(8)
            display(worst_days[['date', 'portfolio_return', 'benchmark_return', 'excess_return']].style.format({
                'portfolio_return': '{:.2%}',
                'benchmark_return': '{:.2%}',
                'excess_return': '{:.2%}',
            }))

            holdings['entry_date'] = parse_mixed_dates(holdings['entry_date'])
            for dt in worst_days['date'].head(5):
                pieces = holdings.loc[holdings['entry_date'] == dt].sort_values('contribution').head(8)
                print('\\nWorst contributors on', dt.date())
                display(pieces[['kdcode', 'rank', 'score', 'stock_return', 'contribution']].style.format({
                    'score': '{:.5f}',
                    'stock_return': '{:.2%}',
                    'contribution': '{:.2%}',
                }))
        """
    ),
    md("## 10. Optional Counterfactual Backtest Harness"),
    code(
        r"""
        # This harness is intentionally lightweight and pandas-only. Use it for quick policy checks
        # on a prediction folder. For final numbers, rerun tests/backtest_sp500.py.

        PREDICTIONS_DIR = REPO_DIR / 'seed_results/2022/seed7/averaged_predictions'
        STOCK_DATA_FILE = REPO_DIR / 'data/raw/market/sp500_2016_universe_data.csv'
        TEST_START = '2022-01-01'
        TEST_END = '2022-12-31'

        def load_prediction_folder(predictions_dir: Path, start: str, end: str) -> pd.DataFrame:
            rows = []
            if not predictions_dir.exists():
                return pd.DataFrame()
            for fp in sorted(predictions_dir.glob('*.csv')):
                date = fp.stem
                if start <= date <= end:
                    part = pd.read_csv(fp)
                    rows.append(part)
            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        def stock_open_to_open(stock_file: Path) -> pd.DataFrame:
            raw = pd.read_csv(stock_file, usecols=['kdcode', 'dt', 'open'], parse_dates=['dt'])
            raw = raw.sort_values(['kdcode', 'dt'])
            raw['next_open'] = raw.groupby('kdcode')['open'].shift(-1)
            raw['open_to_open_return'] = raw['next_open'] / raw['open'] - 1
            return raw.dropna(subset=['open_to_open_return'])

        def quick_policy_backtest(preds: pd.DataFrame, stock: pd.DataFrame, top_k: int, rankdrop_enabled: bool, min_rank_drop: int = 30, spread_bps: float = 5.0, slippage_bps: float = 0.0) -> dict:
            if preds.empty or stock.empty:
                return {}
            preds = preds.copy()
            preds['dt'] = parse_mixed_dates(preds['dt'])
            stock_dates = sorted(stock['dt'].drop_duplicates())
            date_pos = {d: i for i, d in enumerate(stock_dates)}
            returns_by_date = {d: g.set_index('kdcode')['open_to_open_return'] for d, g in stock.groupby('dt')}
            benchmark_by_date = stock.groupby('dt')['open_to_open_return'].mean()

            prev_holdings = None
            prev_ranks = None
            portfolio_returns = []
            benchmark_returns = []
            turnovers = []
            gate_exit_days = 0

            spread = spread_bps / 10000.0
            slippage = slippage_bps / 10000.0

            for pred_date, day_preds in preds.groupby('dt'):
                if pred_date not in date_pos:
                    continue
                day_preds = day_preds.sort_values('score', ascending=False).reset_index(drop=True)
                if len(day_preds) < top_k:
                    continue
                day_preds['_rank'] = np.arange(1, len(day_preds) + 1)
                current_ranks = dict(zip(day_preds['kdcode'], day_preds['_rank']))
                ordered = day_preds['kdcode'].tolist()

                if (not rankdrop_enabled) or prev_holdings is None or prev_ranks is None:
                    holdings = ordered[:top_k]
                else:
                    survivors = []
                    exits_today = 0
                    for kd in prev_holdings:
                        if kd not in current_ranks:
                            continue
                        rank_drop = current_ranks[kd] - prev_ranks.get(kd, current_ranks[kd])
                        if rank_drop >= min_rank_drop:
                            exits_today += 1
                        else:
                            survivors.append(kd)
                    if exits_today:
                        gate_exit_days += 1
                    survivor_set = set(survivors)
                    refill = [kd for kd in ordered if kd not in survivor_set]
                    holdings = survivors + refill[: max(0, top_k - len(survivors))]

                entry_idx = date_pos[pred_date] + 1
                if entry_idx >= len(stock_dates):
                    break
                entry_date = stock_dates[entry_idx]
                rets = returns_by_date.get(entry_date, pd.Series(dtype=float)).reindex(holdings).dropna()
                if rets.empty:
                    prev_holdings = holdings
                    prev_ranks = current_ranks
                    continue

                prev_set = set(prev_holdings or [])
                curr_set = set(holdings)
                one_way_turnover = (len(prev_set - curr_set) + len(curr_set - prev_set)) / (2 * max(top_k, 1))
                cost = one_way_turnover * (spread + 2 * slippage)
                portfolio_returns.append(float(rets.mean() - cost))
                benchmark_returns.append(float(benchmark_by_date.loc[entry_date]))
                turnovers.append(one_way_turnover)
                prev_holdings = holdings
                prev_ranks = current_ranks

            return {
                'top_k': top_k,
                'rankdrop_enabled': rankdrop_enabled,
                'min_rank_drop': min_rank_drop,
                'n_days': len(portfolio_returns),
                'portfolio_total': compound_return(portfolio_returns),
                'benchmark_total': compound_return(benchmark_returns),
                'simple_excess': compound_return(portfolio_returns) - compound_return(benchmark_returns),
                'ARR': annualized_return(portfolio_returns),
                'ASR': annualized_sharpe(portfolio_returns),
                'avg_turnover': float(np.mean(turnovers)) if turnovers else np.nan,
                'gate_exit_days': gate_exit_days,
            }

        preds_cf = load_prediction_folder(PREDICTIONS_DIR, TEST_START, TEST_END)
        stock_cf = stock_open_to_open(STOCK_DATA_FILE) if STOCK_DATA_FILE.exists() else pd.DataFrame()
        print('Counterfactual predictions:', len(preds_cf), 'stock rows:', len(stock_cf))
        """
    ),
    code(
        r"""
        if preds_cf.empty or stock_cf.empty:
            print('Skipping counterfactual harness: prediction folder or stock data missing.')
        else:
            cf_rows = []
            for k in [5, 10, 15, 20, 30, 40]:
                for enabled in [True, False]:
                    cf_rows.append(quick_policy_backtest(preds_cf, stock_cf, top_k=k, rankdrop_enabled=enabled, min_rank_drop=30, spread_bps=5, slippage_bps=0))
            cf = pd.DataFrame(cf_rows)
            display(cf.style.format({
                'portfolio_total': '{:.2%}',
                'benchmark_total': '{:.2%}',
                'simple_excess': '{:.2%}',
                'ARR': '{:.2%}',
                'ASR': '{:.3f}',
                'avg_turnover': '{:.2%}',
            }))

            out_path = OUTPUT_DIR / 'seed7_quick_policy_counterfactuals.csv'
            cf.to_csv(out_path, index=False)
            print('Saved:', out_path)
        """
    ),
    md("## 11. Save Test Summary"),
    code(
        r"""
        checks_df = pd.DataFrame(checks)
        display(checks_df)
        checks_path = OUTPUT_DIR / '2022_investigation_checks.csv'
        checks_df.to_csv(checks_path, index=False)
        print('Saved checks:', checks_path)

        summary_lines = [
            '# 2022 Investigation Notebook Summary',
            '',
            f'- Decision rows: {len(decision):,}',
            f'- Daily rows: {len(daily_raw):,}',
            f'- Checks: {checks_df.status.value_counts().to_dict() if not checks_df.empty else {}}',
            '',
            '## Suggested next probes',
            '',
            '1. Retrieve rich holdings artifacts for imported `full` 2022 prior rows.',
            '2. Rerun exact `tests/backtest_sp500.py` counterfactuals for rankdrop on/off and min_rank_drop values.',
            '3. Compute monthly rank IC / top-minus-bottom realized spread from prediction files when available.',
            '4. Compare long-held loser lists across seeds for k5/k10/k15.',
        ]
        summary_path = OUTPUT_DIR / '2022_investigation_notebook_summary.md'
        summary_path.write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')
        print('Saved summary:', summary_path)
        """
    ),
]


METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10",
    },
}


def main() -> None:
    write_notebook(cells, OUT, metadata=METADATA, indent=2, trailing_newline=True)


if __name__ == "__main__":
    main()
