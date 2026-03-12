"""
Quick LSEG diagnostic: test what comes back for a small set of RICs
across two dates, with varying batch sizes.

Usage:
    python paper_trade/scripts/diag_lseg.py
"""

import time
import refinitiv.data as rd
import pandas as pd

TEST_RICS = [
    "AAPL.OQ", "MSFT.OQ", "AMZN.OQ", "NVDA.OQ", "INTC.OQ",
    "TSLA.OQ", "PSKY.OQ", "F.N", "T.N", "VZ.N",
]

FIELDS = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"]
DATES = [("2026-03-09", "2026-03-09"), ("2026-03-10", "2026-03-10")]


def test_fetch(rics, fields, start, end, label=""):
    """Fetch and report what came back."""
    print(f"\n  [{label}] {len(rics)} RICs, fields={fields is not None}, "
          f"range {start} to {end}")
    try:
        kwargs = dict(universe=rics, start=start, end=end, interval="1D")
        if fields:
            kwargs["fields"] = fields
        df = rd.get_history(**kwargs)

        if df is None:
            print("    Result: None")
            return
        if df.empty:
            print("    Result: empty DataFrame")
            return

        if isinstance(df.columns, pd.MultiIndex):
            returned_rics = [
                r for r in df.columns.get_level_values(0).unique()
                if r not in ("Date", "Instrument", "index")
            ]
        else:
            returned_rics = ["(flat columns)"]

        print(f"    Shape: {df.shape}")
        print(f"    RICs returned: {len(returned_rics)}/{len(rics)}")
        print(f"    Returned: {returned_rics}")

        missing = set(rics) - set(returned_rics)
        if missing:
            print(f"    Missing:  {sorted(missing)}")

        print(f"    Index (dates): {df.index.tolist()}")
        print(f"    Sample:\n{df.head(3)}")

    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")


def check_session(session):
    """Print session state info."""
    print(f"\n  Session type:  {type(session).__name__}")
    print(f"  Session state: {session.open_state if hasattr(session, 'open_state') else 'unknown'}")
    try:
        print(f"  Session repr:  {session}")
    except Exception:
        pass


def main():
    print("=" * 70)
    print("  LSEG Diagnostic")
    print("=" * 70)

    print("\n  Connecting...")
    rd.open_session()
    session = rd.session.Definition().get_session()
    check_session(session)

    try:
        print("\n  --- Test 1: Known historical range (should always work) ---")
        test_fetch(["AAPL.OQ"], None, "2026-03-03", "2026-03-07",
                   label="AAPL last week")
        time.sleep(2)

        print("\n  --- Test 2: Single RIC, recent dates ---")
        for start, end in DATES:
            test_fetch(["AAPL.OQ"], None, start, end,
                       label=f"AAPL {start}")
            time.sleep(2)

        print("\n  --- Test 3: PSKY.OQ (the one that worked before) ---")
        test_fetch(["PSKY.OQ"], None, "2026-03-10", "2026-03-10",
                   label="PSKY 2026-03-10")
        time.sleep(2)

        print("\n  --- Test 4: With vs without fields ---")
        test_fetch(["AAPL.OQ"], FIELDS, "2026-03-03", "2026-03-07",
                   label="AAPL with fields")
        time.sleep(2)

        print("\n  --- Test 5: Full 10 RICs, wider range ---")
        test_fetch(TEST_RICS, None, "2026-03-03", "2026-03-10",
                   label="10 RICs full week")

    finally:
        rd.close_session()
        print("\n  Disconnected.")

    print("=" * 70)


if __name__ == "__main__":
    main()
