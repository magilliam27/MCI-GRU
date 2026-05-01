"""Generate the Colab notebook for backtesting promising confirmation models."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path("notebooks/promising_models_backtest_colab.ipynb")


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
        # MCI-GRU Promising Models Backtest - Google Colab

        Runs the strongest recommended-confirmation candidates through the repository backtesting framework. The notebook uses `tests/backtest_sp500.py` for portfolio simulation so the outputs follow the same open-to-open timing, rank-drop gate, transaction-cost, turnover, and reporting conventions as the rest of the project.
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
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

        if IN_COLAB:
            if not REPO_DIR.exists():
                !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}
            else:
                os.chdir(REPO_DIR)
                !git fetch origin
                !git checkout {BRANCH}
                !git pull origin {BRANCH}

        os.chdir(REPO_DIR)
        print('Working directory:', Path.cwd())

        if IN_COLAB:
            !python -m pip install -q --upgrade pip setuptools wheel
            !python -m pip install -q -r requirements.txt
            !python -m pip install -q -e ".[dev,tracking,fred]"

        print('Python:', sys.executable)
        print('Drive root:', DRIVE_ROOT)
        """
    ),
    md("## 2. Backtest Configuration"),
    code(
        r"""
        from datetime import datetime

        CONFIRMATION_ROOT = DRIVE_ROOT / 'recommended_confirmation_ablation'
        CONFIRMATION_RUN_OVERRIDE = ''  # Optional absolute path to a specific confirmation run folder.
        CANDIDATE_NAME_FILTER = []      # Optional exact candidate names; empty selects by decision table rank.
        MAX_CANDIDATES = 5             # The latest run had five successful promising candidates.

        RUN_PRIMARY_SCENARIO = True
        RUN_TOPK_SENSITIVITY = True
        RUN_NO_GATE_CONTROL = False

        TEST_START = '2025-01-01'
        TEST_END = '2025-12-31'
        LABEL_T = 5
        NUM_TESTS_OVERRIDE = None      # Leave None to use len(candidates) * len(scenarios).

        RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
        BACKTEST_RUN_ROOT = DRIVE_ROOT / 'recommended_backtests' / RUN_TAG
        BACKTEST_RUN_ROOT.mkdir(parents=True, exist_ok=True)

        DATA_FILE_CANDIDATES = [
            'sp500_2019_universe_data_through_2026.csv',
            'sp500_2019_universe_data.csv',
            'sp500_data.csv',
        ]
        DATA_SEARCH_DIRS = [
            REPO_DIR,
            REPO_DIR / 'data' / 'raw' / 'market',
            Path('/content/drive/MyDrive/MCI-GRU-Data') if IN_COLAB else Path.cwd(),
        ]

        PRIMARY_SCENARIO = {
            'scenario': 'k20_spread5_slip0_rankdrop30',
            'top_k': 20,
            'transaction_costs': True,
            'spread_bps': 5.0,
            'slippage_bps': 0.0,
            'rank_drop_gate': True,
            'min_rank_drop': 30,
            'holding_period': 1,
            'rebalance_style': 'staggered',
        }

        SENSITIVITY_SCENARIOS = [
            {**PRIMARY_SCENARIO, 'scenario': 'k10_spread5_slip0_rankdrop30', 'top_k': 10},
            {**PRIMARY_SCENARIO, 'scenario': 'k50_spread5_slip0_rankdrop30', 'top_k': 50},
        ]

        NO_GATE_CONTROL = {
            **PRIMARY_SCENARIO,
            'scenario': 'k20_spread5_slip0_no_rankgate',
            'rank_drop_gate': False,
            'min_rank_drop': 0,
        }

        scenarios = []
        if RUN_PRIMARY_SCENARIO:
            scenarios.append(PRIMARY_SCENARIO)
        if RUN_TOPK_SENSITIVITY:
            scenarios.extend(SENSITIVITY_SCENARIOS)
        if RUN_NO_GATE_CONTROL:
            scenarios.append(NO_GATE_CONTROL)

        print('Backtest output root:', BACKTEST_RUN_ROOT)
        print('Scenarios:', [s['scenario'] for s in scenarios])
        """
    ),
    md("## 3. Locate Latest Confirmation Run And Candidate Predictions"),
    code(
        r"""
        import json
        import pandas as pd
        import shutil

        def is_timestamp_dir(path: Path) -> bool:
            return path.is_dir() and len(path.name) == 15 and path.name[8] == '_' and path.name.replace('_', '').isdigit()

        def find_latest_run(root: Path) -> Path:
            if not root.exists():
                raise FileNotFoundError(f'Confirmation root not found: {root}')
            candidates = sorted([p for p in root.iterdir() if is_timestamp_dir(p)])
            if not candidates:
                raise FileNotFoundError(f'No timestamped confirmation runs found under {root}')
            return candidates[-1]

        confirmation_run = Path(CONFIRMATION_RUN_OVERRIDE) if CONFIRMATION_RUN_OVERRIDE else find_latest_run(CONFIRMATION_ROOT)
        decision_table = confirmation_run / 'recommended_confirmation_decision_table.csv'
        if not decision_table.exists():
            raise FileNotFoundError(f'Missing decision table: {decision_table}')

        decision_df = pd.read_csv(decision_table)
        if 'status' not in decision_df.columns:
            raise ValueError('Decision table is missing a status column.')

        ok_df = decision_df[decision_df['status'].eq('OK')].copy()
        if CANDIDATE_NAME_FILTER:
            ok_df = ok_df[ok_df['name'].isin(CANDIDATE_NAME_FILTER)].copy()

        if 'decision_score' in ok_df.columns:
            ok_df = ok_df.sort_values('decision_score', ascending=False)

        ok_df['predictions_dir'] = ok_df['run_dir'].apply(lambda value: str(Path(value) / 'averaged_predictions'))
        ok_df['has_predictions'] = ok_df['predictions_dir'].apply(lambda value: Path(value).is_dir())
        candidates_df = ok_df[ok_df['has_predictions']].head(MAX_CANDIDATES).copy()

        if candidates_df.empty:
            missing = ok_df[['name', 'run_dir', 'predictions_dir', 'has_predictions']] if len(ok_df) else ok_df
            display(missing)
            raise RuntimeError('No successful candidates with averaged_predictions were found.')

        manifest_path = BACKTEST_RUN_ROOT / 'selected_candidates.csv'
        candidates_df.to_csv(manifest_path, index=False)

        display_cols = [c for c in [
            'name',
            'decision_score',
            'evaluation.metrics.avg_ic',
            'evaluation.metrics.return_top_20',
            'evaluation.metrics.sharpe_top_20_newey_west',
            'graph_factor',
            'objective_factor',
            'regime_factor',
            'drop_edge_p',
            'seed',
            'predictions_dir',
        ] if c in candidates_df.columns]

        print('Confirmation run:', confirmation_run)
        print('Selected candidates:', len(candidates_df))
        print('Candidate manifest:', manifest_path)
        display(candidates_df[display_cols])
        """
    ),
    md("## 4. Resolve Market Data"),
    code(
        r"""
        def resolve_market_data() -> Path:
            checked = []
            for directory in DATA_SEARCH_DIRS:
                if directory is None:
                    continue
                directory = Path(directory)
                for filename in DATA_FILE_CANDIDATES:
                    path = directory / filename
                    checked.append(path)
                    if path.exists():
                        return path
            raise FileNotFoundError('Could not find market data. Checked: ' + ', '.join(str(p) for p in checked))

        DATA_FILE = resolve_market_data()
        preview = pd.read_csv(DATA_FILE, usecols=['dt', 'kdcode'])
        preview['dt'] = pd.to_datetime(preview['dt'])
        print('Using data file:', DATA_FILE)
        print(f'Rows: {len(preview):,}')
        print(f'Stocks: {preview.kdcode.nunique():,}')
        print(f'Dates: {preview.dt.min().date()} to {preview.dt.max().date()}')
        del preview
        """
    ),
    md("## 5. Run Backtests"),
    code(
        r"""
        import hashlib
        import re

        def safe_name(value: str, max_len: int = 96) -> str:
            cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
            if len(cleaned) <= max_len:
                return cleaned
            digest = hashlib.sha1(cleaned.encode('utf-8')).hexdigest()[:8]
            return cleaned[: max_len - 9] + '_' + digest

        def backtest_dir_for(predictions_dir: Path, scenario: dict) -> Path:
            return predictions_dir.parent / f"backtest_{scenario['scenario']}"

        def run_backtest(row: pd.Series, scenario: dict, num_tests: int) -> dict:
            predictions_dir = Path(row['predictions_dir'])
            suffix = '_' + scenario['scenario']
            cmd = [
                sys.executable,
                str(REPO_DIR / 'tests' / 'backtest_sp500.py'),
                '--predictions_dir', str(predictions_dir),
                '--data_file', str(DATA_FILE),
                '--test_start', TEST_START,
                '--test_end', TEST_END,
                '--top_k', str(scenario['top_k']),
                '--label_t', str(LABEL_T),
                '--holding_period', str(scenario['holding_period']),
                '--rebalance_style', scenario['rebalance_style'],
                '--num_tests', str(num_tests),
                '--adjustment_method', 'bhy',
                '--auto_save',
                '--plot',
                '--backtest_suffix', suffix,
            ]
            if scenario['transaction_costs']:
                cmd.extend(['--transaction_costs', '--spread', str(scenario['spread_bps']), '--slippage', str(scenario['slippage_bps'])])
            if scenario['rank_drop_gate']:
                cmd.extend(['--enable_rank_drop_gate', '--min_rank_drop', str(scenario['min_rank_drop'])])

            print('\n' + '=' * 100)
            print('Candidate:', row['name'])
            print('Scenario:', scenario['scenario'])
            print('Command:', ' '.join(cmd))
            proc = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
            print(proc.stdout[-5000:])
            if proc.returncode != 0:
                print(proc.stderr[-5000:])

            source_backtest_dir = backtest_dir_for(predictions_dir, scenario)
            copied_backtest_dir = BACKTEST_RUN_ROOT / safe_name(row['name']) / scenario['scenario']
            if source_backtest_dir.exists():
                if copied_backtest_dir.exists():
                    shutil.rmtree(copied_backtest_dir)
                shutil.copytree(source_backtest_dir, copied_backtest_dir)

            result = {
                'name': row['name'],
                'scenario': scenario['scenario'],
                'returncode': proc.returncode,
                'predictions_dir': str(predictions_dir),
                'source_backtest_dir': str(source_backtest_dir),
                'copied_backtest_dir': str(copied_backtest_dir),
                'stdout_tail': proc.stdout[-5000:],
                'stderr_tail': proc.stderr[-5000:],
            }

            metrics_path = source_backtest_dir / 'backtest_metrics.json'
            if metrics_path.exists():
                with open(metrics_path, encoding='utf-8') as f:
                    metrics = json.load(f)
                result.update({f'backtest.{k}': v for k, v in metrics.items()})

            adjusted_results_path = source_backtest_dir / 'backtest_results.csv'
            if adjusted_results_path.exists():
                adjusted_df = pd.read_csv(adjusted_results_path)
                if len(adjusted_df) > 0:
                    adjusted = adjusted_df.iloc[0].to_dict()
                    result.update({f'backtest.{k}': v for k, v in adjusted.items() if f'backtest.{k}' not in result})

            for key in [
                'decision_score',
                'evaluation.metrics.avg_ic',
                'evaluation.metrics.avg_ic_ci_lower',
                'evaluation.metrics.return_top_20',
                'evaluation.metrics.top_20_return_ci_lower',
                'evaluation.metrics.sharpe_top_20_newey_west',
                'graph_factor',
                'objective_factor',
                'regime_factor',
                'objective_factor',
                'drop_edge_p',
                'seed',
            ]:
                if key in row:
                    result[key] = row[key]

            result.update({f"scenario_config.{k}": v for k, v in scenario.items()})
            return result

        num_tests = NUM_TESTS_OVERRIDE or (len(candidates_df) * len(scenarios))
        rows = []
        for _, candidate in candidates_df.iterrows():
            for scenario in scenarios:
                rows.append(run_backtest(candidate, scenario, num_tests))
                interim_df = pd.DataFrame(rows)
                interim_df.to_csv(BACKTEST_RUN_ROOT / 'backtest_comparison_interim.csv', index=False)

        results_df = pd.DataFrame(rows)
        raw_path = BACKTEST_RUN_ROOT / 'backtest_comparison_raw.csv'
        results_df.to_csv(raw_path, index=False)

        sort_cols = [c for c in ['backtest.ASR', 'backtest.excess_return', 'decision_score'] if c in results_df.columns]
        if sort_cols:
            results_df = results_df.sort_values(sort_cols, ascending=False)

        comparison_path = BACKTEST_RUN_ROOT / 'backtest_comparison.csv'
        html_path = BACKTEST_RUN_ROOT / 'backtest_comparison.html'
        results_df.to_csv(comparison_path, index=False)
        results_df.to_html(html_path, index=False)

        print('Saved raw comparison:', raw_path)
        print('Saved sorted comparison:', comparison_path)
        print('Saved HTML comparison:', html_path)

        display_cols = [c for c in [
            'name',
            'scenario',
            'decision_score',
            'backtest.ARR',
            'backtest.ASR',
            'backtest.IR',
            'backtest.MDD',
            'backtest.total_return',
            'backtest.total_return_calendar_aligned',
            'backtest.benchmark_return',
            'backtest.excess_return',
            'backtest.avg_daily_turnover',
            'backtest.avg_daily_cost_bps',
            'backtest.total_trades',
            'backtest.haircutted_sharpe',
            'backtest.adjusted_p_value',
            'source_backtest_dir',
        ] if c in results_df.columns]
        display(results_df[display_cols])
        """
    ),
    md("## 6. Visualize Backtest Comparison"),
    code(
        r"""
        import matplotlib.pyplot as plt
        import numpy as np

        ok_results = results_df[results_df['returncode'].eq(0)].copy()
        if ok_results.empty:
            raise RuntimeError('No successful backtests to plot.')

        plot_metrics = [
            ('backtest.ASR', 'Annualized Sharpe'),
            ('backtest.excess_return', 'Excess Return'),
            ('backtest.total_return_calendar_aligned', 'Calendar-Aligned Total Return'),
            ('backtest.MDD', 'Maximum Drawdown'),
        ]
        plot_metrics = [(col, title) for col, title in plot_metrics if col in ok_results.columns]

        fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(14, max(4, 3.8 * len(plot_metrics))))
        if len(plot_metrics) == 1:
            axes = [axes]

        for ax, (col, title) in zip(axes, plot_metrics):
            plot_df = ok_results.copy()
            plot_df['label'] = plot_df['name'].str.replace('__', '\n', regex=False) + '\n' + plot_df['scenario']
            plot_df = plot_df.sort_values(col, ascending=True)
            ax.barh(plot_df['label'], plot_df[col], color='#2f6f73')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(title)
            ax.grid(True, axis='x', alpha=0.25)

        plt.tight_layout()
        metric_plot_path = BACKTEST_RUN_ROOT / 'backtest_metric_bars.png'
        plt.savefig(metric_plot_path, dpi=160, bbox_inches='tight')
        plt.show()
        print('Saved metric bars:', metric_plot_path)

        primary_name = PRIMARY_SCENARIO['scenario']
        primary_results = ok_results[ok_results['scenario'].eq(primary_name)].copy()
        if not primary_results.empty:
            fig, ax = plt.subplots(1, 1, figsize=(13, 6))
            benchmark_drawn = False
            for _, row in primary_results.iterrows():
                returns_file = Path(row['source_backtest_dir']) / 'daily_returns.csv'
                if not returns_file.exists():
                    continue
                returns_df = pd.read_csv(returns_file)
                returns_df['date'] = pd.to_datetime(returns_df['date'])
                values = np.cumprod(1 + returns_df['portfolio_return'].astype(float))
                ax.plot(returns_df['date'], values, label=row['name'], linewidth=1.8)
                if not benchmark_drawn and 'benchmark_return' in returns_df.columns:
                    benchmark = np.cumprod(1 + returns_df['benchmark_return'].astype(float))
                    ax.plot(returns_df['date'], benchmark, label='Equal-weight benchmark', color='black', linestyle='--', linewidth=1.2)
                    benchmark_drawn = True
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0)
            ax.set_title(f'Equity Curves - {primary_name}')
            ax.set_ylabel('Cumulative value')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best', fontsize=8)
            plt.tight_layout()
            equity_plot_path = BACKTEST_RUN_ROOT / 'primary_equity_curves.png'
            plt.savefig(equity_plot_path, dpi=160, bbox_inches='tight')
            plt.show()
            print('Saved primary equity curves:', equity_plot_path)
        """
    ),
    md("## 7. Write Summary Report"),
    code(
        r"""
        def fmt_pct(value):
            if pd.isna(value):
                return ''
            return f'{100 * float(value):.2f}%'

        summary_cols = [c for c in [
            'name',
            'scenario',
            'decision_score',
            'backtest.ARR',
            'backtest.ASR',
            'backtest.IR',
            'backtest.MDD',
            'backtest.total_return_calendar_aligned',
            'backtest.benchmark_return',
            'backtest.excess_return',
            'backtest.avg_daily_turnover',
            'backtest.avg_daily_cost_bps',
            'backtest.haircutted_sharpe',
            'backtest.adjusted_p_value',
            'source_backtest_dir',
        ] if c in ok_results.columns]

        report_table = ok_results[summary_cols].copy()
        for pct_col in [
            'backtest.ARR',
            'backtest.MDD',
            'backtest.total_return_calendar_aligned',
            'backtest.benchmark_return',
            'backtest.excess_return',
            'backtest.avg_daily_turnover',
        ]:
            if pct_col in report_table.columns:
                report_table[pct_col] = report_table[pct_col].apply(fmt_pct)

        top_primary = ok_results[ok_results['scenario'].eq(PRIMARY_SCENARIO['scenario'])].copy()
        if 'backtest.ASR' in top_primary.columns:
            top_primary = top_primary.sort_values('backtest.ASR', ascending=False)

        report_path = BACKTEST_RUN_ROOT / 'promising_models_backtest_summary.md'
        lines = [
            '# MCI-GRU Promising Models Backtest Summary',
            '',
            f'Confirmation run: `{confirmation_run}`',
            f'Backtest output root: `{BACKTEST_RUN_ROOT}`',
            f'Market data: `{DATA_FILE}`',
            f'Evaluation window: `{TEST_START}` to `{TEST_END}`',
            f'Label horizon: `{LABEL_T}` trading days',
            f'Multiple-testing count used for BHY haircut: `{num_tests}`',
            '',
            '## Selected Candidates',
            '',
            candidates_df[[c for c in ['name', 'decision_score', 'evaluation.metrics.avg_ic', 'evaluation.metrics.return_top_20', 'run_dir'] if c in candidates_df.columns]].to_markdown(index=False),
            '',
            '## Backtest Results',
            '',
            report_table.to_markdown(index=False),
            '',
            '## Primary Scenario',
            '',
            f"- `{PRIMARY_SCENARIO['scenario']}` uses top-k={PRIMARY_SCENARIO['top_k']}, daily open-to-open rebalancing, transaction costs enabled with spread={PRIMARY_SCENARIO['spread_bps']} bps and slippage={PRIMARY_SCENARIO['slippage_bps']} bps, and rank-drop gate min_rank_drop={PRIMARY_SCENARIO['min_rank_drop']}.",
            '- Backtest outputs are produced by `tests/backtest_sp500.py`, including `backtest_metrics.json`, `daily_returns.csv`, `portfolio_holdings.csv`, `trades.csv`, and `equity_curve.png` for each candidate/scenario.',
            '- Treat this as the next filter after IC/top-20 return confirmation: prefer candidates that keep positive excess return and reasonable drawdown after costs and turnover.',
            '',
            '## Artifacts',
            '',
            f'- Candidate manifest: `{manifest_path}`',
            f'- Sorted comparison CSV: `{comparison_path}`',
            f'- HTML comparison: `{html_path}`',
            f'- Metric bars: `{metric_plot_path}`',
        ]
        if 'equity_plot_path' in globals():
            lines.append(f'- Primary equity curves: `{equity_plot_path}`')

        report_path.write_text('\n'.join(lines), encoding='utf-8')
        print(report_path.read_text(encoding='utf-8')[:6000])
        print('Saved summary report:', report_path)
        """
    ),
    md("## 8. Zip Results"),
    code(
        r"""
        archive_path = shutil.make_archive(str(BACKTEST_RUN_ROOT), 'zip', root_dir=str(BACKTEST_RUN_ROOT.parent), base_dir=BACKTEST_RUN_ROOT.name)
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
