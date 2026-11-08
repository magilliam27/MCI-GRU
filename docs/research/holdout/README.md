# Holdout Promotions

The locked 2026 holdout and its three looks (PRD #369, decision 3; #374). The
rules are in [`../EXPERIMENT_PROTOCOL.md`](../EXPERIMENT_PROTOCOL.md), under
**Promoting a candidate**; `mci_gru/promotion.py` enforces them.

- `<id>.yaml` is a promotion record, committed before its runs.
- `ledger.yaml` lists every look taken, from the receipts. Three looks retire the
  holdout.
- `<id>.receipt.json` is the receipt `first_run_colab.py compare --promotion`
  wrote, committed with its ledger entry.

A change here is the only change a promotion's checkout may carry beyond the code
its verdict ran.
