# CLAUDE.md

> Claude-specific guidance. For the tool-agnostic agent map, see `AGENTS.md`.
> For deep architecture details, see `docs/ARCHITECTURE.md`.

## Commands

```bash
# Tests
python -m pytest tests/ -v                                        # all tests
python -m pytest tests/test_dynamic_graph_updates.py -v            # single file
python -m pytest tests/test_momentum_blend_modes.py::TestDynamicBlend -v  # single class

# Lint
ruff check .          # lint
ruff format --check . # format check

# Smoke run
python run_experiment.py training.num_epochs=2 training.num_models=1

# Full experiment
python run_experiment.py +experiment=paper_faithful

# Backtest
python tests/backtest_sp500.py --predictions_dir results/<name>/<ts>/averaged_predictions --auto_save --plot

# Paper trade
python paper_trade/scripts/run_nightly.py [--skip-refresh]
```

## Architecture (summary)

See `docs/ARCHITECTURE.md` for the full version.

**Data flow**: CSV/LSEG/FRED → DataManager → FeatureEngineer → `prepare_data()` (normalize, window, graph) → DataLoaders → Trainer → `averaged_predictions/`

**Model**: Four streams `[A1 temporal, A2 cross-sectional, B1 latent, B2 latent]` → concat → optional self-attention → final GAT → per-stock score.

**Config**: Hydra YAML in `configs/`. Override: `python run_experiment.py model.his_t=20 training.loss_type=ic`

## Invariants

1. **No lookahead** — normalization, graph, labels use train-period cutoffs only.
2. **Dynamic graph requires `batch_size=1`** — enforced in `ExperimentConfig.__post_init__`.
3. **`combined_collate_fn` returns a 7-tuple** — `(time_series, labels, graph_features, edge_index, edge_weight, n_stocks, batch_dates)`.
4. **Ensemble** — `train_multiple_models` trains N models; prediction = mean.
5. **Paper trade** — frozen checkpoints, no `GraphBuilder` import at inference.
6. **FRED_API_KEY** required for credit spread / regime features.
