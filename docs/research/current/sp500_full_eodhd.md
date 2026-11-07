# The full point-in-time S&P 500 on EODHD prices

Status: package r1 built 2026-11-07 (#386). Data config
`configs/data/sp500_full_eodhd.yaml`. No model has been trained on it yet.

## Question

The recipe trains on 110 names, the ten largest by market cap in each GICS sector.
The owner asked for the whole index, about 500 names a day, so that the baselines,
new model families and the recipe itself can be scored on the full S&P 500 with
nothing else changed.

## Membership

Membership is the preserved LSEG S&P 500 membership: the
`..._sp500_membership_intervals.csv` file of the r1 110-name package
(`sp500_pit_gics_top10_mcap_monthly_20160104_20260731.r1.json`, sha256
`5d7f8191...`), from which the 110-name universe was cut. The owner chose it
(2026-11-07) over EODHD's own constituents list.

`scripts/data/build_sp500_full_universe.py membership` checks the file's size and
digest against that manifest and writes it under the `kdcode,valid_from,valid_to`
header with every row unchanged: 760 intervals, 753 RICs, 129 of them delisted
(with their `^` codes), 503 to 509 members on any weekday from 2016-01-04 to
2026-07-31.

EODHD's `fundamentals/GSPC.INDX?historical=1` is the cross-check. Over 2,760
sessions the two lists share 99.04% of LSEG's member-days. Most of the rest is
EODHD filing a past member under the ticker the company carries today, or under a
ticker since given to someone else: Equity Residential's EQR days appear under
another current ticker, Paramount's under PSKY, and Discovery's class C (DISCK) is
missing. That is current-ticker filing, not a different membership, and it is why
EODHD's list was not used. The comparison is in the evidence folder
(`..._membership_check.json`).

## Prices

`scripts/data/export_eodhd_pit_prices.py` pulls every RIC with the symbol map in
`data/mappings/eodhd_symbols_sp500_full.json`, from 2015-01-01 (for warm-up) to
2026-07-31. The prices follow r3's conventions: split-adjusted from EODHD's split
records, volume as delivered less later splits (#347), regular dividends not applied.

750 of the 753 RICs are priced, 1,881,725 rows. Three have no EODHD series and are
declared in the manifest's `provenance.unknowns` and in `data.pit_absent_kdcodes`:

* `DD.N^I17`, old DuPont, as in the 110-name package;
* `IILGV.OQ^E16`, a when-issued line EODHD does not carry;
* `JCI.N^I16`, the pre-merger Johnson Controls. EODHD's JCI.US before 2016-09-02 is
  Tyco's series (LSEG's monthly market cap for this RIC is Tyco's too), and no other
  listing holds it.

Priced members per day run from 501 to 508 over 2016-01-04 to 2026-07-31, so
`pit_min_scoreable_stocks: 450` has room.

### The symbol map

The map keeps every decision of the 110-name map, which was proven against the LSEG
price panel, with one exception pinned by a test: EODHD now files Paramount's whole
history under `PSKY.US`. On top of that it adds:

* **Old symbols.** EODHD moves a delisted company to `X_old` (or `X_old1`, `CB1`,
  `FTI1`, `MNKKQ`) when its ticker is reused. Twenty-odd past members map there, for
  example `CA_old.US`, `MON_old.US`, `STI_old.US`, and Coca-Cola Enterprises to
  `CCE_old.US`.
* **Bounded segments** where EODHD's series runs into another company or a bogus
  tail: the old Chubb (`CB1`), Envision (AmSurg to 2016-12-01, then EVHC), the old
  TechnipFMC (`FTI1`), Diamond Offshore, Mallinckrodt, Valaris (`ESV`), Chesapeake
  before its 2021 restructuring, Signature Bank to 2023-03-10, Garrett to 2020-09-30,
  Noble Energy to 2020-10-02, ZimVie to 2025-10-17, Alcoa from its 2016-11-01 listing,
  Delphi Technologies from 2017-12-05, and Victoria's Secret's change of ticker.
* **Adjustments** for spin-offs EODHD records oddly: Ventas's 2015 CCP spin (EODHD's
  split runs backwards), the Adient and Ingevity spins, and steps in Weyerhaeuser,
  Keurig Dr Pepper, Leidos, Gen Digital and AIMCO where EODHD's own series already
  steps and the map records the vendor step rather than inventing a factor.

Cash special dividends of 3% to 18% in names the LSEG panel does not cover stay on
the raw basis, like r3: LSEG itself does not restate them consistently, and the
export reports them as non-blocking `large_adjustment_step` findings (39 in this
build) rather than altering prices.

## Checks

The 206 RICs the LSEG price panel covers were compared with it by the exporter:
0 reference failures, and the split-adjusted basis is the closer one (median
absolute difference 2.5e-8).

The other names have no LSEG prices, so `build_sp500_full_universe.py check` tests
each against LSEG's monthly market caps (`all_metadata_snapshots.csv`, every member
2016 to 2026-07):

* **Identity.** Between adjacent month-ends, the change in log market cap minus the
  change in log EODHD close is the change in shares outstanding, which for the right
  company is almost always zero. A median share step above 2% over at least three
  pairs flags a name as possibly the wrong company.
* **Price steps.** A share step above 10% *and* a close move above 25% in the same
  month flags a missed split or a wrong splice; a merger moves shares without moving
  the price, so it does not flag.
* **Extreme moves.** A daily rise above 40% or fall below 35% inside membership must
  be corroborated by the market cap or reviewed.
* **Delisting month.** For a delisted RIC, EODHD's last trade must fall within a
  month of the month its `^` code names.
* **Missing sessions** inside membership, against a calendar of the days at least 100
  names traded.

Identity rests on market cap for 626 names, on the delisting month for 87, and on a
reviewed reason for 37. Every remaining flag (10 price steps, 6 uncorroborated
extreme moves, 2 names with missing sessions) is decided with its reason in
`data/mappings/eodhd_review_sp500_full.json`; the check exits non-zero on any flag
that file does not decide. Before this build the checks found eleven mappings that ran into another company
or a bogus tail, among them the old Chubb, Johnson Controls, TechnipFMC and
Envision; all are fixed in the map.

Two gaps are deliberate and reviewed:

* Howmet (`HWM.N`, formerly Arconic) is unpriced from 2016-01-04 to 2016-10-31.
  EODHD's series for the old Alcoa sits under `AA.US`, which is the new Alcoa's after
  the split, and splicing two companies' series was not done.
* Marathon Petroleum's (`MRP`) 11 when-issued sessions are unpriced.

## Package

Manifest `data/manifests/sp500_pit_full_20160104_20260731_eodhd.r1.json` (sha256
`8a3f8e4e...`); the data config pins it. The files are in
`gs://mci-gru-runs-8a195c19-data/preservation/sp500_pit_full_20160104_20260731_eodhd_r1/`
under the manifest's paths (`market/`, `constituents/`), with a copy of the manifest
and, in `evidence/`, the membership check, the identity-check report and the EODHD
constituents response it was compared with. The LSEG files it was checked against are
not copied: they are the preserved 110-name package in
`gs://mci-gru-runs-8a195c19-data/preservation/2026-10-02-110-universe-2016/` (the
intervals `5d7f8191...`, the price panel `d64c4d04...`, which is the r1 manifest's
record, and `..._all_metadata_snapshots.csv` `31673ec6...`). The bucket copy was
downloaded again and every file matched the manifest.

## Known limits

* Volume for names whose spin-off EODHD records as a split carries that "split".
* Howmet's first ten months and the three absent names reduce breadth by at most
  four names on any day.
* Past members' prices depend on EODHD's archive, checked against market cap, not
  against a second price source.
