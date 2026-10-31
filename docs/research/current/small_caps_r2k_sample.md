# Small-caps universe: a random point-in-time Russell 2000 sample

Status: data built, run not yet made (#379). Exploratory: the run is stamped
`protocol.exploratory=true` and gets no verdict.

## Question

How does the model do on much smaller companies than the S&P 500's largest,
with everything else the same? The owner asked (2026-10-31) for the recipe to
score Russell 2000 names and hold the top 50 each week, with IWM as a reference
benchmark. A random sample of about 400 members a day was chosen over the whole
index to keep the run inside the GCP budget.

## What changes and what does not

Only the data config: `data=r2k_sample_400_eodhd` in place of
`gics_top10_110_2016_eodhd`. Model, features, graph, loss, seeds, ensemble size,
budget and dates are the recipe's (train 2020 to 2023, validate 2024, test
2025). The weekly top 50 is the evaluation's existing `top_50` book: a top-50
portfolio formed every session and held for the label's four sessions, in four
staggered tranches. Its headline benchmark is still the equal-weight tradable
universe, here the sample's own members. IWM is compared after the run, from
EODHD's `IWM.US` close (price basis, like the panel).

## How the universe is built

1. **Membership** is from the SEC N-PORT filings of the iShares Russell 2000 ETF
   (IWM, series S000004344), quarterly from 2019-09-30, built by
   `scripts/data/build_r2k_members.py` (#317). Each holding was tied to an EODHD
   symbol by matching the fund's price (value over shares) to the symbol's close
   within 3%, and to an SEC CIK checked against SEC's names. In this build 55,134
   of 55,729 holdings rows were price-confirmed, and 99.4% have a CIK. A company is
   a member from a report date to the day before the next report it is missing
   from. Delisted and acquired companies stay in.
2. **Frame**: the 3,398 companies with a membership day from 2020-01-02 to
   2025-12-31, less 29 with no price-confirmed holding, leaving 3,369.
3. **Draw**: 688 companies, `random.Random(379).sample` of the sorted CIKs. 688 is
   the frame scaled by 400 over the median of 1,958 frame companies held per
   report date. On any weekday 384 to 434 are members (median 402). The draw reads
   nothing about prices or returns.
4. **Prices**: EODHD daily rows from 2015-01-02 to 2026-01-21 (the day before the
   locked holdout, so the last test dates have their labels), split-adjusted, with
   every name's trailing zero-volume repeats of its last close trimmed. 1,426,226
   rows, 0 blocking findings.
5. **Symbols**: each company's EODHD symbol comes from its price-matched holdings.
   When a company is held under two share classes on one report date, the class
   with the larger total value over all report dates is used for its whole history,
   so the series never switches class. A ticker change hands over on the old
   symbol's last close before the new symbol's first report date.

No reference panel exists for these names, so every decision below was read from
EODHD's own rows and is recorded with its reason in
`data/mappings/eodhd_review_r2k_sample_400.json`. The builder refuses a review entry
without a reason.

**Split-basis findings.** The first pull stopped on 18 names. Three were ticker
changes after the window end, which the builder now ignores. The other 15 were
split-sized disagreements between EODHD's raw close and its `adjusted_close`:

- 7 distributions (special dividends, a spin-off) take the vendor's adjusted step.
- 2 reverse splits that EODHD dates a session after its raw close shows them lose
  that one row.
- 6 names whose vendor adjustment looks wrong before their first membership day
  have those earlier rows dropped.

**Ticker-change joins.** The export checks each symbol's own series but not the
price level where one symbol hands over to the next, so the builder's
`--check-joins` measures the one-day return across every change and stops on one
over 20% without a decision. Seven joins were on two different bases (a
restructuring, a bankruptcy, a separation, a restated split history). Each company
keeps one side only, the one covering more of its membership days, and is
unpriced on the other side's days. The final package has no flagged join.

**Extreme returns.** Every member-window daily move above +100% or below -60% from
mid-2019 (65 moves in 51 names) was read against the vendor's raw rows. Six were
vendor errors and are fixed:

| Company | Date | Move | Finding | Fix |
|---|---|---|---|---|
| Parker Drilling | 2020-02-14 | -99.5% | raw close $14.66 to $0.072 with no split record | series ends 2020-02-07 |
| DigitalBridge | 2022-08-23 | +6298% | raw close before the 1-for-4 reverse split reads about $0.08 | rows before 2022-08-23 dropped |
| Knoll | 2021-07-20 | -97.3% | rows printed after the acquisition closed | series ends 2021-07-19 |
| CorePoint Lodging | 2022-03-03 | -79.0% | last row after the $16 take-private | series ends 2022-03-02 |
| Equity Commonwealth | 2024-12-09 | -92.6% | $19 liquidating distribution not in the vendor's prices | series ends 2024-12-06 |
| Cryptyde | 2022-06-30 | -67.8% | when-issued quotes on volumes of 0 to 36 shares | rows before 2022-06-30 dropped |

The other 59 moves look genuine: trial readouts, bankruptcies, takeover bids and
the January 2021 squeezes. They stay in, because small caps do move this way and
the test should see it.

## Known biases and limits

- **Membership dates are quarterly.** A company added between reports joins up to
  a quarter late. One deleted between reports stays a member until the next
  report, but a delisted name has no price after its last trade, so it simply
  stops being tradable.
- **Delisting returns are missing.** A name's last return is its last traded
  close, not what holders finally received. In a one-model CPU check of the
  first package, 125 of the test year's book exits found no close. Small caps delist far more often than the S&P 500's largest, so this
  matters more here, and it biases both the book and the universe benchmark.
- **ETF holdings are not the index file.** IWM tracks the Russell 2000 closely,
  but it can hold a few names the index does not, and miss a few it does.
- **No second price source.** The S&P panels were proven against LSEG daily
  returns; these names have no reference panel. The cross-check is the
  holdings-price match at every report date, plus the export's own split-basis
  findings.
- **Price returns.** Dividends are not applied, as in the recipe.
- **Regime inputs stay S&P 500-based**, because they are part of the recipe.
- **The graph is nearly empty.** At the recipe's 0.8 threshold, 29 linked pairs
  form among 688 names.

## Run

The package is at
`gs://mci-gru-runs-8a195c19-data/preservation/r2k_sample_400_20200102_20251231_eodhd_r2/`,
with the committed manifest copied beside it (manifest sha256
`317644ae7e340d1b5e4d7d095a822e161c56f8ac3c4af146e0937326ca7d8a13`). Revision r1,
before the join and extreme-return fixes, is superseded. The run is one recipe arm
on Vertex (g2-standard-8 + L4), after a timing probe, each cleared by the GCP budget
tracker first.
