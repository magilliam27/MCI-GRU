---
name: minimal-dropout-plan
overview: Add a single configurable dropout layer at the concatenated feature vector before the final prediction GAT, with a Hydra model-level parameter and validation checks for regression risk.
todos:
  - id: add-model-dropout-config
    content: Add model.dropout to dataclass/config validation and dictionary export.
    status: pending
  - id: wire-dropout-into-model
    content: Add nn.Dropout in StockPredictionModel and apply it only on concatenated Z before final_gat.
    status: pending
  - id: plumb-factory-and-hydra
    content: Pass dropout through create_model and set default in configs/config.yaml.
    status: pending
  - id: ab-test-no-regression
    content: Run 0.0 vs 0.1 dropout comparison with fixed seeds/windows and confirm no leakage-related process changes.
    status: pending
isProject: false
---

# Minimal Dropout Integration Plan

## Objective

Introduce one dropout point in the MCI-GRU model with minimal architectural disruption: apply dropout to concatenated representation `Z` right before `final_gat`, controlled by one config value (`model.dropout`).

## Files To Update

- [c:/Users/magil/MCI-GRU/mci_gru/models/mci_gru.py](c:/Users/magil/MCI-GRU/mci_gru/models/mci_gru.py)
- [c:/Users/magil/MCI-GRU/mci_gru/config.py](c:/Users/magil/MCI-GRU/mci_gru/config.py)
- [c:/Users/magil/MCI-GRU/configs/config.yaml](c:/Users/magil/MCI-GRU/configs/config.yaml)
- Optional sanity check alignment: [c:/Users/magil/MCI-GRU/scripts/check_config.py](c:/Users/magil/MCI-GRU/scripts/check_config.py)

## Implementation Steps

- Add `dropout: float = 0.0` to `ModelConfig` in `config.py` with range validation (`0.0 <= dropout < 1.0`) and include it in `to_dict()`.
- Add `model.dropout: 0.1` (or your preferred default) in `configs/config.yaml` so Hydra can override via CLI.
- In `StockPredictionModel.__init`__ (`mci_gru/models/mci_gru.py`):
  - accept a `dropout` argument,
  - instantiate `self.dropout = nn.Dropout(p=dropout)`.
- In `StockPredictionModel.forward`:
  - keep current feature construction,
  - apply dropout only at the minimal point: `Z = self.dropout(Z)` before `self.final_gat(...)`.
- Update `create_model(...)` to pass `dropout` from config into `StockPredictionModel`.

## Validation And Safety Checks

- Verify behavior parity at `dropout=0.0` (outputs/training pipeline should remain functionally unchanged).
- Run one A/B comparison (`dropout=0.0` vs `dropout=0.1`) on the same seeds/windows.
- Track train/val divergence and stability across windows; keep this as a regularization test, not an architecture overhaul.

## Bias / Methodology Guardrails

- Dropout itself does not add look-ahead or survivorship bias.
- Preserve existing temporal splits and lag policies exactly during A/B runs to avoid confounding and accidental leakage.

