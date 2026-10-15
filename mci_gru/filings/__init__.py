"""SEC filings as text inputs, judged by a decision model (#289).

Pure functions only: no network access and no credentials. Acquisition lives in
``scripts/data/export_sec_risk_factors.py`` and the decision-model calls in
``scripts/data/score_risk_factor_pairs.py``.

- ``edgar``: the filing index from EDGAR submissions files, and when each filing
  was knowable under the #225 prediction clock.
- ``risk_factors``: Item 1A of a 10-K, cut into individual risk factors.
- ``pairing``: each risk factor paired with its most similar prior-year risks.
- ``jev``: the frozen question set, the answer cache, and the blind label sheet.
- ``grading``: Jev's answers graded against the hand labels and two code baselines.
"""
