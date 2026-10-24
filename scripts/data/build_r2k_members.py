"""Build Russell 2000 membership from IWM's SEC holdings filings, and draw E14's companies.

E14 repeats E11's 8-K returns test on Russell 2000 companies. No free history of
the index exists, so membership comes from what the iShares Russell 2000 ETF
(IWM, series S000004344) reported holding (``mci_gru.filings.iwm_holdings``)::

    python -m scripts.data.build_r2k_members \\
        --user-agent "Your Name you@example.com" \\
        --cache-dir /content/r2k_cache --output-dir /content/r2k_membership \\
        --seed 314 --draw 1000 --first-release 500 --rest-release

Steps, each read from ``--cache-dir`` when already there and fetched otherwise
(EODHD calls send no ``api_token``: the cloud proxy adds the key, and
``EODHD_API_KEY`` is sent when set):

1. **Holdings.** IWM's N-Q filings for 31 Dec 2012 to 31 Dec 2018 and every
   N-PORT, from EDGAR. One list per report date; an N-PORT re-filed for a date
   already read is skipped. Common stocks only (the N-Q's "Common Stocks"
   section, N-PORT asset category EC).
2. **Candidates.** For each holding name (``holding_key``) and ISIN: EODHD
   listings whose name has the same key; for an ISIN, EODHD's id map and the
   listings carrying it; and the listings EODHD maps to each SEC CIK that has
   ever gone by the name (SEC ``cik-lookup-data.txt``). Preferred-share symbols
   never count.
3. **Price check.** A holding is a candidate's when the fund's price (value over
   shares) is within ``MATCH_TOLERANCE`` of the candidate's close on the report
   date (``price_error``). Where no candidate passes, EODHD's splits for the
   candidates are applied and the check is repeated. Weaker rules, flagged in
   ``how``: the only candidate (``single_candidate``), the closest within 40%
   (``closest_40pct``), or the symbol the same name's checked rows got
   (``same_name``).
4. **Company.** A symbol's CIK comes from EODHD's id map, then SEC's ticker file,
   then a name SEC lists under one CIK only. Every (name, CIK) pair is then
   checked against SEC's current and former names for that CIK; a pair that
   fails is re-keyed to the CIK whose name matches and whose SEC tickers include
   the symbol, else to the only CIK with the name, else left out.
5. **Spells and draw.** ``membership.spells`` over the dated lists of CIKs held:
   a company is a member from a report date to the next report date it is
   missing from. The frame is every company with a membership day from
   ``--frame-start``; the draw is ``random.Random(seed).sample`` of its sorted
   CIKs. A larger draw with the same seed keeps a smaller one as its first
   companies, so ``--first-release`` marks the first N drawn as release ``r1``
   and the rest ``r2``: E14 drew 500, E15 adds the next 500. ``--rest-release``
   adds every company not drawn as ``r3`` (E16, the whole frame).

What it writes under ``--output-dir``: ``holdings_matched.csv.gz`` (every
holding row with its candidates' best symbol, price error, symbol, CIK and how
each was found), ``r2k_membership.csv`` (the frame: one row per CIK with its
names, symbols and spells), ``r2k_sample.csv`` (the draw), ``r2k_companies.csv``
(the draw as ``export_sec_member_8k``'s companies table) and
``r2k_members.meta.json``.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import io
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mci_gru.filings.iwm_holdings import (
    holding_key,
    nport_holdings,
    nq_holdings,
    price_error,
    split_factor,
)
from mci_gru.filings.membership import spells

SERIES = "S000004344"
TRUST_CIK = "1100663"
SEC_BROWSE = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={series}&type={form}"
    "&dateb=&owner=include&count=100&output=atom"
)
EODHD = "https://eodhd.com/api"
#: The fund's price and a candidate's close agree within this share.
MATCH_TOLERANCE = 0.03
#: The weaker rule's limit for the closest candidate.
LOOSE_TOLERANCE = 0.40
#: First day of price history fetched for each candidate.
PRICE_FROM = "2012-06-01"
SEC_INTERVAL = 0.125


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-agent", required=True, help="Name and contact email for SEC")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nq-from", default="2012-12-31", help="First N-Q report date")
    parser.add_argument("--nq-to", default="2018-12-31", help="Last N-Q report date")
    parser.add_argument("--frame-start", default="2013-01-01")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--draw", type=int, default=500)
    parser.add_argument(
        "--first-release", type=int, default=None, help="First N drawn are r1, the rest r2"
    )
    parser.add_argument(
        "--rest-release", action="store_true", help="Add every company not drawn as r3"
    )
    parser.add_argument("--workers", type=int, default=8, help="Threads for EODHD calls")
    args = parser.parse_args(argv)
    if "@" not in args.user_agent:
        parser.error("--user-agent must include a contact email, as SEC asks")
    return args


class Sec:
    """Paced GETs against SEC, each response kept under the cache."""

    def __init__(self, user_agent: str) -> None:
        self.headers = {"User-Agent": user_agent}
        self._lock = threading.Lock()
        self._last = 0.0
        self.requests = 0

    def get(self, url: str) -> bytes:
        for attempt in range(6):
            with self._lock:
                wait = self._last + SEC_INTERVAL - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
                self.requests += 1
            try:
                request = urllib.request.Request(url, headers=self.headers)
                return urllib.request.urlopen(request, timeout=120).read()
            except urllib.error.HTTPError as error:
                if error.code == 404 or attempt == 5:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 5:
                    raise
            time.sleep(2**attempt)
        raise RuntimeError(url)

    def cached(self, path: Path, url: str) -> bytes:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.get(url))
        return path.read_bytes()


def eodhd_get(path: str, params: dict[str, str]) -> bytes | None:
    """One EODHD GET; None for a 404. The key is the proxy's, or ``EODHD_API_KEY``."""
    params = dict(params)
    if os.environ.get("EODHD_API_KEY"):
        params["api_token"] = os.environ["EODHD_API_KEY"]
    url = f"{EODHD}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"EODHD {path} failed")


def cached_json_map(path: Path, keys: list[str], fetch, workers: int) -> dict[str, Any]:
    """A JSON file of ``key -> response``, filled for the keys it lacks."""
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = sorted(set(keys) - set(cache))
    if todo:
        with ThreadPoolExecutor(workers) as pool:
            for key, value in zip(todo, pool.map(fetch, todo), strict=True):
                if value is not None:
                    cache[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return cache


def _id_map(field: str):
    def fetch(value: str) -> list[dict[str, Any]] | None:
        body = eodhd_get("id-mapping", {f"filter[{field}]": value, "fmt": "json"})
        return json.loads(body)["data"] if body else []

    return fetch


# ---- 1. holdings ---------------------------------------------------------------


def iwm_filings(sec: Sec, cache: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    """IWM's N-Q and N-PORT filings in range, oldest report first, one per report date."""
    found = []
    for form in ("N-Q", "NPORT-P"):
        atom = sec.cached(
            cache / "raw" / f"listing_{form}.xml", SEC_BROWSE.format(series=SERIES, form=form)
        )
        for entry in re.findall(r"<entry>.*?</entry>", atom.decode("latin1"), re.S):
            filed = re.search(r"<filing-date>([^<]*)", entry).group(1)
            index = re.search(r"<filing-href>([^<]*)", entry).group(1)
            found.append({"form": form, "filed": filed, "index": index})
    kept, seen = [], set()
    for filing in sorted(found, key=lambda f: f["filed"]):
        folder = cache / "raw" / f"{filing['form']}_{filing['filed']}"
        page = sec.cached(folder / "index.htm", filing["index"]).decode("latin1")
        period = re.search(r"Period of Report</div>\s*<div[^>]*>([^<]+)<", page).group(1).strip()
        if filing["form"] == "N-Q" and not (args.nq_from <= period <= args.nq_to):
            continue
        if period in seen:
            continue
        seen.add(period)
        links = re.findall(r'href="(/Archives/[^"]+)"', page)
        if filing["form"] == "NPORT-P":
            doc = next(
                link for link in links if link.endswith("primary_doc.xml") and "xsl" not in link
            )
            name = "primary_doc.xml"
        else:
            doc = next(
                link
                for link in links
                if link.endswith(".htm") and "nq" in link.lower() and "ex99" not in link.lower()
            )
            name = "nq.htm"
        kept.append(
            {
                **filing,
                "period": period,
                "path": str(folder / name),
                "url": "https://www.sec.gov" + doc,
            }
        )
    for filing in kept:
        sec.cached(Path(filing["path"]), filing["url"])
    return kept


def read_holdings(filings: list[dict[str, str]]) -> pd.DataFrame:
    """Every common-stock holding on every report date."""
    frames = []
    for filing in filings:
        raw = Path(filing["path"]).read_bytes()
        if filing["form"] == "N-Q":
            table = nq_holdings(raw.decode("latin1"))
            table = table[table["section"] == "COMMON STOCKS"]
        else:
            series, period, table = nport_holdings(raw)
            if series != SERIES or period != filing["period"]:
                raise ValueError(f"{filing['path']} is {series} on {period}")
            table = table[table["asset_cat"] == "EC"]
        frames.append(
            table.assign(snap=filing["period"], source=filing["form"].replace("NPORT-P", "N-PORT"))
        )
    holdings = pd.concat(frames, ignore_index=True)
    holdings["isin"] = holdings["isin"].fillna("")
    holdings["norm"] = holdings["name"].map(holding_key)
    holdings["implied_price"] = holdings["value"] / holdings["shares"].where(holdings["shares"] > 0)
    return holdings[
        ["snap", "source", "name", "shares", "value", "isin", "cusip", "norm", "implied_price"]
    ]


# ---- 2. candidates and 3. price check -------------------------------------------


def sec_names(lookup: Path) -> dict[str, set[str]]:
    """Every name EDGAR has held, by ``holding_key``, with its CIKs."""
    by_key: dict[str, set[str]] = defaultdict(set)
    for line in lookup.read_text(encoding="latin1").split("\n"):
        parts = line.rsplit(":", 2)
        if len(parts) == 3 and parts[0]:
            by_key[holding_key(parts[0])].add(parts[1])
    return by_key


def candidates(
    holdings: pd.DataFrame, cache: Path, sec_by_key: dict[str, set[str]], workers: int
) -> dict[tuple[str, str], list[str]]:
    lists = []
    for name, extra in (("active", ""), ("delisted", "1")):
        path = cache / "eodhd" / f"{name}.json"
        if not path.exists():
            params = {"fmt": "json", **({"delisted": extra} if extra else {})}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(eodhd_get("exchange-symbol-list/US", params))
        lists += json.loads(path.read_text())
    by_key, by_isin = defaultdict(set), defaultdict(set)
    for listing in lists:
        if listing["Type"] == "Common Stock":
            by_key[holding_key(listing["Name"])].add(listing["Code"] + ".US")
            if listing.get("Isin"):
                by_isin[listing["Isin"]].add(listing["Code"] + ".US")
    isins = sorted(set(holdings.loc[holdings["isin"] != "", "isin"]))
    id_isin = cached_json_map(cache / "eodhd" / "idmap_isin.json", isins, _id_map("isin"), workers)
    ciks = sorted(set().union(*(sec_by_key.get(key, set()) for key in set(holdings["norm"]))))
    id_cik = cached_json_map(cache / "eodhd" / "idmap_cik.json", ciks, _id_map("cik"), workers)
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in holdings[["norm", "isin"]].drop_duplicates().itertuples():
        symbols = set(by_key.get(row.norm, ()))
        for cik in sec_by_key.get(row.norm, ()):
            symbols |= {m["symbol"] for m in id_cik.get(cik) or [] if m["symbol"].endswith(".US")}
        if row.isin:
            symbols |= {
                m["symbol"] for m in id_isin.get(row.isin, []) if m["symbol"].endswith(".US")
            }
            symbols |= by_isin.get(row.isin, set())
        found[(row.norm, row.isin)] |= symbols
    return {key: sorted(value) for key, value in found.items()}


def fetch_prices(symbols: list[str], cache: Path, workers: int) -> None:
    folder = cache / "eod"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(symbol: str) -> None:
        path = folder / f"{symbol}.csv.gz"
        if not path.exists():
            body = eodhd_get(
                f"eod/{urllib.parse.quote(symbol)}", {"from": PRICE_FROM, "fmt": "csv"}
            )
            path.write_bytes(gzip.compress(body or b""))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, sorted(set(symbols))))


def price_match(holdings: pd.DataFrame, cands: dict, cache: Path, workers: int) -> pd.DataFrame:
    @functools.cache
    def closes(symbol: str) -> pd.Series | None:
        body = gzip.decompress((cache / "eod" / f"{symbol}.csv.gz").read_bytes())
        if not body.strip():
            return None
        table = pd.read_csv(io.BytesIO(body))
        if "Date" not in table:
            return None
        table = table[pd.to_numeric(table["Close"], errors="coerce").notna()]
        return pd.Series(table["Close"].astype(float).values, index=pd.to_datetime(table["Date"]))

    def close_on(symbol: str, day: str) -> float:
        series = closes(symbol)
        if series is None:
            return np.nan
        stamp = pd.Timestamp(day)
        near = series[(series.index <= stamp) & (series.index > stamp - pd.Timedelta(days=6))]
        return float(near.iloc[-1]) if len(near) else np.nan

    def best(row, splits: dict[str, list]) -> tuple[float, str]:
        top = (np.inf, "")
        for symbol in cands.get((row.norm, row.isin), []):
            close = close_on(symbol, row.snap)
            if np.isfinite(close) and np.isfinite(row.implied_price):
                error = price_error(
                    row.implied_price, close, split_factor(splits.get(symbol) or [], row.snap)
                )
                if error < top[0]:
                    top = (error, symbol)
        return top

    first = [best(row, {}) for row in holdings.itertuples()]
    unmatched = {
        symbol
        for row, (error, _) in zip(holdings.itertuples(), first, strict=True)
        if error >= MATCH_TOLERANCE
        for symbol in cands.get((row.norm, row.isin), [])
    }

    def fetch_splits(symbol: str) -> list | None:
        body = eodhd_get(f"splits/{urllib.parse.quote(symbol)}", {"fmt": "json"})
        return json.loads(body) if body else []

    all_splits = cached_json_map(
        cache / "eodhd" / "splits.json", sorted(unmatched), fetch_splits, workers
    )
    splits = {symbol: all_splits.get(symbol) for symbol in unmatched}
    second = [
        best(row, splits) if error >= MATCH_TOLERANCE else (error, symbol)
        for row, (error, symbol) in zip(holdings.itertuples(), first, strict=True)
    ]
    out = holdings.copy()
    out["err"] = [error for error, _ in second]
    out["best_symbol"] = [symbol for _, symbol in second]
    out["symbol"] = np.where(out["err"] < MATCH_TOLERANCE, out["best_symbol"], "")
    out["n_cands"] = [len(cands.get((row.norm, row.isin), [])) for row in holdings.itertuples()]
    out["how"] = np.where(out["symbol"] != "", "price", "")
    common = {key: [s for s in value if "-P" not in s] for key, value in cands.items()}
    for i in out.index[out["symbol"] == ""]:
        options = common.get((out.at[i, "norm"], out.at[i, "isin"]), [])
        if len(options) == 1 and out.at[i, "best_symbol"] in ("", options[0]):
            out.at[i, "symbol"], out.at[i, "how"] = options[0], "single_candidate"
        elif out.at[i, "best_symbol"] and out.at[i, "err"] < LOOSE_TOLERANCE:
            out.at[i, "symbol"], out.at[i, "how"] = out.at[i, "best_symbol"], "closest_40pct"
    checked = (
        out[out["how"] == "price"]
        .groupby("norm")["symbol"]
        .agg(lambda s: s.value_counts().index[0])
    )
    for i in out.index[out["symbol"] == ""]:
        if out.at[i, "norm"] in checked:
            out.at[i, "symbol"], out.at[i, "how"] = checked[out.at[i, "norm"]], "same_name"
    return out


# ---- 4. company ------------------------------------------------------------------


def companies_for(
    out: pd.DataFrame, cache: Path, sec: Sec, sec_by_key: dict[str, set[str]], workers: int
) -> pd.DataFrame:
    sym_cik: dict[str, str] = {}
    for name in ("idmap_isin.json", "idmap_cik.json"):
        for value in json.loads((cache / "eodhd" / name).read_text()).values():
            for entry in value or []:
                if entry.get("cik") and entry["symbol"].endswith(".US"):
                    sym_cik.setdefault(entry["symbol"], entry["cik"])
    missing = sorted(set(out["symbol"]) - {""} - set(sym_cik))
    id_symbol = cached_json_map(
        cache / "eodhd" / "idmap_symbol.json", missing, _id_map("symbol"), workers
    )
    for symbol in missing:
        for entry in id_symbol.get(symbol) or []:
            if entry.get("cik"):
                sym_cik[symbol] = entry["cik"]
                break
    tickers = json.loads(
        sec.cached(
            cache / "sec_company_tickers.json", "https://www.sec.gov/files/company_tickers.json"
        )
    )
    by_ticker = {
        v["ticker"].replace(".", "-") + ".US": str(v["cik_str"]).zfill(10) for v in tickers.values()
    }
    out = out.copy()
    out["cik"] = out["symbol"].map(sym_cik).fillna("")
    out["cik_how"] = np.where(out["cik"] != "", "eodhd_idmap", "")
    for i in out.index[(out["cik"] == "") & (out["symbol"] != "")]:
        if out.at[i, "symbol"] in by_ticker:
            out.at[i, "cik"], out.at[i, "cik_how"] = by_ticker[out.at[i, "symbol"]], "sec_ticker"
    for i in out.index[out["cik"] == ""]:
        only = sec_by_key.get(out.at[i, "norm"], set())
        if len(only) == 1:
            out.at[i, "cik"], out.at[i, "cik_how"] = next(iter(only)), "sec_unique_name"
    out["cik"] = out["cik"].map(lambda c: str(c).zfill(10) if c else "")

    def submissions(cik: str) -> dict[str, Any]:
        try:
            raw = sec.cached(
                cache / "submissions" / f"CIK{cik}.json",
                f"https://data.sec.gov/submissions/CIK{cik}.json",
            )
        except urllib.error.HTTPError:
            return {}
        return json.loads(raw)

    def sec_keys(cik: str) -> set[str]:
        data = submissions(cik)
        names = [data.get("name", "")] + [f["name"] for f in data.get("formerNames", [])]
        return {holding_key(n) for n in names if n}

    def close(a: str, b: str) -> bool:
        wa, wb = set(a.split()), set(b.split())
        a2, b2 = a.replace(" ", ""), b.replace(" ", "")
        prefix = min(len(a2), len(b2)) >= 8 and (a2.startswith(b2) or b2.startswith(a2))
        return a == b or prefix or bool(wa and wb and (wa <= wb or wb <= wa))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(submissions, sorted(set(out["cik"]) - {""})))
    fixes = {}
    for pair in out.loc[out["cik"] != "", ["norm", "cik", "symbol"]].drop_duplicates().itertuples():
        if any(close(pair.norm, key) for key in sec_keys(pair.cik)):
            fixes[(pair.norm, pair.cik)] = (pair.cik, "pass")
            continue
        others = [c for c in sec_by_key.get(pair.norm, set()) if c != pair.cik]
        ticker = pair.symbol.replace(".US", "").split("_")[0]
        with_ticker = [c for c in others if ticker in submissions(c).get("tickers", [])]
        if len(with_ticker) == 1:
            fixes[(pair.norm, pair.cik)] = (with_ticker[0], "fixed_by_name_and_ticker")
        elif len(others) == 1:
            fixes[(pair.norm, pair.cik)] = (others[0], "fixed_by_name")
        else:
            fixes[(pair.norm, pair.cik)] = ("", "unconfirmed_left_out")
    out["cik_check"] = ""
    for i in out.index[out["cik"] != ""]:
        out.at[i, "cik"], out.at[i, "cik_check"] = fixes[(out.at[i, "norm"], out.at[i, "cik"])]
    return out


# ---- 5. spells and draw ------------------------------------------------------------


def membership(out: pd.DataFrame, frame_start: str) -> pd.DataFrame:
    held = out[out["cik"] != ""]
    lists = [(pd.Timestamp(snap), frozenset(group["cik"])) for snap, group in held.groupby("snap")]
    runs = defaultdict(list)
    for spell in spells(lists):
        end = "" if spell.end is None else (spell.end - pd.Timedelta(days=1)).date().isoformat()
        runs[spell.ticker].append(f"{spell.start.date().isoformat()}:{end}")
    rows = []
    for cik, group in held.groupby("cik"):
        group = group.sort_values("snap")
        symbols = list(dict.fromkeys(s for s in group["symbol"] if s))
        rows.append(
            {
                "cik": cik,
                "name": group.iloc[-1]["name"],
                "symbols": ";".join(symbols),
                "first_snap": group["snap"].iloc[0],
                "last_snap": group["snap"].iloc[-1],
                "n_snaps": group["snap"].nunique(),
                "spells": ";".join(runs[cik]),
                "price_confirmed_share": round((group["how"] == "price").mean(), 3),
            }
        )
    frame = pd.DataFrame(rows)
    keep = frame["spells"].map(
        lambda value: any(
            end == "" or end >= frame_start for _, end in (s.split(":") for s in value.split(";"))
        )
    )
    return frame[keep].sort_values("cik").reset_index(drop=True)


def draw(
    ciks: list[str],
    seed: int,
    size: int,
    first_release: int | None = None,
    rest_release: bool = False,
) -> pd.DataFrame:
    """``cik``, ``draw_order`` and ``release`` of ``size`` CIKs drawn from ``ciks``, by CIK.

    ``random.sample`` picks one index at a time, so the first N of a draw are the
    whole of a draw of N with the same seed and population. The first
    ``first_release`` drawn (all, by default) are ``r1``; the rest are ``r2``. With
    ``rest_release`` every CIK not drawn is added as ``r3``, with no draw order.
    """
    picked = random.Random(seed).sample(sorted(ciks), size)
    cut = size if first_release is None else first_release
    table = pd.DataFrame(
        {
            "cik": picked,
            "draw_order": pd.array(range(len(picked)), dtype="Int64"),
            "release": ["r1" if i < cut else "r2" for i in range(len(picked))],
        }
    )
    if rest_release:
        rest = sorted(set(ciks) - set(picked))
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    {
                        "cik": rest,
                        "draw_order": pd.array([pd.NA] * len(rest), dtype="Int64"),
                        "release": "r3",
                    }
                ),
            ]
        )
    return table.sort_values("cik").reset_index(drop=True)


def companies_table(sample: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """The draw in ``export_sec_member_8k``'s companies format, named as SEC names it.

    Keys are tickers; a second company with the same ticker gets its CIK appended.
    Earlier releases take keys first, so adding a release never renames a company.
    """
    rows, seen = [], set()
    for row in sample.sort_values(["release", "cik"]).itertuples():
        symbols = [s for s in row.symbols.split(";") if s]
        ticker = symbols[-1].replace(".US", "").split("_")[0] if symbols else "NA"
        key = ticker.lower()
        if key in seen or key == "na":
            key = f"{key}_{row.cik.lstrip('0')}"
        seen.add(key)
        sec_name = json.loads((cache / "submissions" / f"CIK{row.cik}.json").read_text()).get(
            "name", ""
        )
        rows.append(
            {
                "key": key,
                "name": sec_name or row.name,
                "ciks": row.cik,
                "tickers": ticker,
                "spells": ";".join(f"{ticker}:{s}" for s in row.spells.split(";")),
                "release": row.release,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache, out_dir = args.cache_dir, args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sec = Sec(args.user_agent)
    started = datetime.now(timezone.utc).isoformat()
    filings = iwm_filings(sec, cache, args)
    holdings = read_holdings(filings)
    lookup = cache / "sec_cik_lookup.txt"
    sec.cached(lookup, "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt")
    sec_by_key = sec_names(lookup)
    cands = candidates(holdings, cache, sec_by_key, args.workers)
    fetch_prices([s for value in cands.values() for s in value], cache, args.workers)
    matched = price_match(holdings, cands, cache, args.workers)
    final = companies_for(matched, cache, sec, sec_by_key, args.workers)
    frame = membership(final, args.frame_start)
    drawn = draw(list(frame["cik"]), args.seed, args.draw, args.first_release, args.rest_release)
    sample = frame.merge(drawn, on="cik").sort_values("cik").reset_index(drop=True)
    final.to_csv(out_dir / "holdings_matched.csv.gz", index=False, compression="gzip")
    frame.to_csv(out_dir / "r2k_membership.csv", index=False)
    sample.to_csv(out_dir / "r2k_sample.csv", index=False)
    companies_table(sample, cache).to_csv(out_dir / "r2k_companies.csv", index=False)
    meta = {
        "source": "IWM (iShares Russell 2000 ETF, series S000004344) N-Q and N-PORT holdings on SEC EDGAR; EODHD listings, id map, prices and splits; SEC names",
        "started": started,
        "report_dates": [f["period"] for f in filings],
        "filings": [{k: f[k] for k in ("form", "filed", "period", "url")} for f in filings],
        "holdings_rows": int(len(final)),
        "rows_by_how": final["how"].value_counts().to_dict(),
        "rows_by_cik_how": final["cik_how"].value_counts().to_dict(),
        "rows_by_cik_check": final["cik_check"].value_counts().to_dict(),
        "rows_with_cik": round(float((final["cik"] != "").mean()), 4),
        "frame_start": args.frame_start,
        "frame_companies": int(len(frame)),
        "seed": args.seed,
        "drawn": int(len(sample)),
        "drawn_by_release": sample["release"].value_counts().sort_index().to_dict(),
        "drawn_member_at_last_date": int((sample["last_snap"] == filings[-1]["period"]).sum()),
        "sec_requests": sec.requests,
    }
    (out_dir / "r2k_members.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "filings"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
