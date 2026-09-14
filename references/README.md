# References

The papers and vendor documentation this repository draws on. Only citations
and links are kept here; the documents themselves are the authors' and
publishers' to distribute.

## Papers

1. *MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and
   Improved GRU*, Neurocomputing 2025. arXiv:2410.20679.
   <https://arxiv.org/abs/2410.20679>. The four-stream architecture.
2. Goulding, Harvey, Mazzoleni, *Momentum Turning Points*. SSRN 3489539.
   <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489539>. The slow/fast
   momentum signals, the Bull/Correction/Bear cycle states, and the dynamic
   speed selection in `mci_gru/features/momentum.py`.
3. Harvey, Liu, *Evaluating Trading Strategies*. SSRN 2474755.
   <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2474755>. The
   multiple-testing haircut applied in the selection audit.
4. Veličković et al., *Graph Attention Networks*, ICLR 2018. arXiv:1710.10903.
   <https://arxiv.org/abs/1710.10903>. The graph-attention blocks.
5. Man Group, *Regimes, Systematic Models and the Power of Prediction*,
   March 2025, and a companion note on market regimes. Background for the
   global regime features in `mci_gru/features/regime.py`; see
   `docs/REGIME_DATA_CONTRACT.md` for what the code actually computes.

## Vendor documentation

- LSEG Refinitiv Data Platform, *Historical Pricing API data guide*. Consulted
  for the field names in `mci_gru/data/lseg_loader.py`. Available to licensed
  LSEG users from the developer portal.

The full-text extractions and PDFs that used to sit in this directory are
readable at tag `archive/pre-cleanup-2026-09`.
