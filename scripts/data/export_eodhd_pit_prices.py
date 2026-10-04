"""Pull daily EODHD prices for every name in a point-in-time universe (#281).

Run from the repository root where EODHD is reachable (Colab; this cloud's
network refuses it), with the key in the environment only::

    EODHD_API_KEY=... python -m scripts.data.export_eodhd_pit_prices \\
        --pit-universe <package>/constituents/<prefix>_pit_universe.csv \\
        --reference-panel <package>/market/<prefix>_lseg_20150101_20260731.csv \\
        --package-root /content/eodhd_package \\
        --cache-dir /content/eodhd_cache \\
        --manifest-output /content/<prefix>_eodhd.r1.json

What it writes under ``--package-root``, as a package beside the LSEG one:

- ``constituents/<prefix>_pit_universe.csv``: the membership file, byte-copied.
- ``market/<prefix>_eodhd_<start>_<end>.csv``: the ``kdcode, dt, open, high,
  low, close, volume`` panel the CSV loader reads, keyed by the same RICs.
- ``..._coverage.csv``, ``..._reference_check.csv``, ``..._symbols.json``,
  ``..._symbol_map.json``, ``..._raw.jsonl`` and ``.meta.json``: the evidence.
  That is per-name coverage, agreement with the reference panel, the symbol each
  stretch was read from, the reviewed map used, every vendor response as
  received, and the pull's provenance and findings.

A blocking finding stops the package: the files are written for inspection,
no manifest is published, and the exit status is 1. The key is read from
``EODHD_API_KEY`` and never written, logged or put in an error message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.data.eodhd_prices import (
    Finding,
    SymbolPlan,
    SymbolSegment,
    adjustment_basis_summary,
    agreement_passes,
    assemble_panel,
    build_symbol_plans,
    check_window,
    clean_values,
    clip_segment,
    coverage_table,
    eod_rows_frame,
    load_symbol_map,
    name_hint_candidates,
    needed_spans,
    panel_for_write,
    reference_check,
    return_agreement,
    split_adjust,
    split_basis_findings,
    splits_frame,
)
from mci_gru.data.input_manifest import InputFileSpec, read_input_manifest, write_input_manifest

API_ROOT = "https://eodhd.com/api"
KEY_ENV = "EODHD_API_KEY"
DEFAULT_SYMBOL_MAP = Path("data/mappings/eodhd_symbols_gics_top10_110_2016.json")
RETRY_DELAYS = (2, 4, 8, 16)


class VendorError(RuntimeError):
    """A vendor request failed; the message never carries the key."""


class EodhdClient:
    """Minimal EODHD JSON client with an on-disk response cache."""

    def __init__(self, api_key: str, cache_dir: Path, raw_log: list[dict[str, Any]]) -> None:
        if not api_key:
            raise VendorError(f"{KEY_ENV} is not set")
        self._key = api_key
        self._cache = cache_dir
        self._raw = raw_log
        self.calls = 0

    def _get(self, kind: str, name: str, path: str, params: dict[str, str]) -> Any:
        cache_path = self._cache / kind / f"{name}.json"
        record: dict[str, Any] = {"endpoint": path, "params": params}
        if cache_path.exists():
            envelope = json.loads(cache_path.read_text(encoding="utf-8"))
            record.update(envelope, from_cache=True)
            self._raw.append(record)
            return envelope["body"]
        query = urllib.parse.urlencode({**params, "api_token": self._key, "fmt": "json"})
        url = f"{API_ROOT}/{path}?{query}"
        body, status = None, None
        for attempt, delay in enumerate((0, *RETRY_DELAYS)):
            time.sleep(delay)
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    status = response.status
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                status = error.code
                if error.code == 404:
                    body = []
                    break
                if error.code in (401, 403):
                    raise VendorError(f"EODHD refused {path} (HTTP {error.code})") from None
                if error.code != 429 and error.code < 500:
                    raise VendorError(f"EODHD returned HTTP {error.code} for {path}") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                status = type(error).__name__
            if attempt == len(RETRY_DELAYS):
                raise VendorError(f"EODHD request for {path} failed after retries ({status})")
        self.calls += 1
        envelope = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "body": body,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(envelope), encoding="utf-8")
        record.update(envelope, from_cache=False)
        self._raw.append(record)
        return body

    def eod(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        body = self._get("eod", symbol, f"eod/{symbol}", {"period": "d", "from": start, "to": end})
        return body if isinstance(body, list) else []

    def splits(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        body = self._get("splits", symbol, f"splits/{symbol}", {"from": start, "to": end})
        return body if isinstance(body, list) else []

    def listing(self, delisted: bool) -> list[dict[str, Any]]:
        params = {"delisted": "1"} if delisted else {}
        name = "US_delisted" if delisted else "US_active"
        body = self._get("listing", name, "exchange-symbol-list/US", params)
        return body if isinstance(body, list) else []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pit-universe", type=Path, required=True)
    parser.add_argument(
        "--reference-panel",
        type=Path,
        help="Preserved LSEG panel used to prove each mapping; omit only to skip that proof",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="Manifest of the package the PIT file and reference panel come from; both "
        "files must match its records by name, size and SHA-256",
    )
    parser.add_argument("--reference-manifest-sha256")
    parser.add_argument("--symbol-map", type=Path, default=DEFAULT_SYMBOL_MAP)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--package-revision", default="r1")
    return parser.parse_args(argv)


def _prefix(pit_path: Path) -> str:
    suffix = "_pit_universe.csv"
    if not pit_path.name.endswith(suffix):
        raise SystemExit(f"--pit-universe must be a '*{suffix}' file")
    return pit_path.name[: -len(suffix)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_exact(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target``; an existing different target stops the pull."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != _sha256(source):
            raise SystemExit(f"{target} exists with different bytes; refusing to overwrite")
        return
    shutil.copyfile(source, target)


def verify_inputs(manifest: Path, expected_sha256: str | None, inputs: list[Path]) -> str:
    """Check each input against the record with its file name; return the manifest digest."""
    snapshot = read_input_manifest(manifest, expected_sha256=expected_sha256)
    records = {Path(record.path).name: record for record in snapshot.manifest.files}
    for path in inputs:
        record = records.get(path.name)
        if record is None:
            raise SystemExit(f"{path.name} is not declared in {manifest.name}")
        if path.stat().st_size != record.size_bytes or _sha256(path) != record.sha256:
            raise SystemExit(f"{path} differs from its record in {manifest.name}")
    return snapshot.sha256


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def read_symbol(
    client: EodhdClient, kdcode: str, symbol: str, args: argparse.Namespace
) -> tuple[pd.DataFrame, list[Finding], int]:
    """One symbol's split-adjusted rows over the pull span, with its checks."""
    eod = eod_rows_frame(client.eod(symbol, args.start, args.end))
    eod = eod[(eod["dt"] >= args.start) & (eod["dt"] <= args.end)]
    splits = splits_frame(client.splits(symbol, args.start, args.end), as_of=args.end)
    adjusted = split_adjust(eod, splits)
    findings = split_basis_findings(kdcode, symbol, adjusted)
    adjusted, blanked = clean_values(adjusted)
    return adjusted.reset_index(drop=True), findings, blanked


def resolve_segment(
    client: EodhdClient,
    plan: SymbolPlan,
    segment: SymbolSegment,
    need: dict[str, str] | None,
    reference: pd.DataFrame | None,
    listings: dict[bool, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], list[Finding]]:
    """Pick the candidate symbol for one segment and return its clipped rows.

    With a reference, the first candidate whose returns agree with it over the
    segment's needed span wins; name-hint candidates are tried only after the
    listed ones fail. Without one, the first candidate with rows wins.
    """
    window = None
    if need is not None and reference is not None:
        window = check_window(need, reference, segment)
    tried: list[dict[str, Any]] = []
    findings: list[Finding] = []
    fallback: tuple[pd.DataFrame, str, int] | None = None

    def candidates():
        yield from segment.candidates
        if segment.name_hint:
            for delisted in (True, False):
                if delisted not in listings:
                    listings[delisted] = client.listing(delisted)
                for symbol in name_hint_candidates(listings[delisted], segment.name_hint):
                    if symbol not in segment.candidates:
                        yield symbol

    seen = set()
    for symbol in candidates():
        if symbol in seen:
            continue
        seen.add(symbol)
        rows, symbol_findings, blanked = read_symbol(client, plan.kdcode, symbol, args)
        rows = clip_segment(rows, segment)
        entry: dict[str, Any] = {"symbol": symbol, "rows": len(rows)}
        if rows.empty:
            tried.append(entry)
            continue
        if fallback is None:
            fallback = (rows, symbol, blanked)
            fallback_findings = symbol_findings
        if window is None:
            tried.append(entry)
            findings.extend(symbol_findings)
            return rows, _chosen(segment, symbol, tried, blanked, "first_with_rows"), findings
        stats = return_agreement(rows, reference, *window)
        entry.update(stats)
        tried.append(entry)
        if agreement_passes(stats):
            findings.extend(symbol_findings)
            return rows, _chosen(segment, symbol, tried, blanked, "reference_match"), findings
    if fallback is None:
        findings.append(
            Finding(
                plan.kdcode,
                "no_vendor_rows",
                True,
                "No candidate symbol returned rows inside this segment",
                {"segment": _segment_dict(segment), "tried": tried},
            )
        )
        return pd.DataFrame(), _chosen(segment, None, tried, 0, "none"), findings
    rows, symbol, blanked = fallback
    findings.extend(fallback_findings)
    findings.append(
        Finding(
            plan.kdcode,
            "no_candidate_matched",
            True,
            "No candidate agreed with the reference over this segment; the first with rows "
            "is kept so the full check shows the disagreement",
            {"segment": _segment_dict(segment), "window": list(window), "tried": tried},
        )
    )
    return rows, _chosen(segment, symbol, tried, blanked, "unmatched_fallback"), findings


def _segment_dict(segment: SymbolSegment) -> dict[str, Any]:
    return {
        "candidates": list(segment.candidates),
        "start": segment.start,
        "end": segment.end,
        "name_hint": segment.name_hint,
    }


def _chosen(segment, symbol, tried, blanked, how) -> dict[str, Any]:
    return {
        **_segment_dict(segment),
        "symbol": symbol,
        "chosen_by": how,
        "values_blanked": blanked,
        "tried": tried,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(KEY_ENV, "")
    prefix = _prefix(args.pit_universe)
    reference_manifest_sha256 = None
    if args.reference_manifest is not None:
        inputs = [args.pit_universe]
        if args.reference_panel is not None:
            inputs.append(args.reference_panel)
        reference_manifest_sha256 = verify_inputs(
            args.reference_manifest, args.reference_manifest_sha256, inputs
        )
    start_c, end_c = args.start.replace("-", ""), args.end.replace("-", "")
    stem = f"{prefix}_eodhd_{start_c}_{end_c}"
    market = args.package_root / "market"
    market.mkdir(parents=True, exist_ok=True)
    pit_target = args.package_root / "constituents" / args.pit_universe.name
    _copy_exact(args.pit_universe, pit_target)
    map_target = market / f"{stem}_symbol_map.json"
    _copy_exact(args.symbol_map, map_target)

    pit = pd.read_csv(args.pit_universe, dtype=str)
    plans = build_symbol_plans(pit["kdcode"], load_symbol_map(args.symbol_map))
    spans = needed_spans(pit).set_index("kdcode")
    reference = None
    if args.reference_panel is not None:
        reference = pd.read_csv(args.reference_panel, usecols=["kdcode", "dt", "close"])
        reference["dt"] = reference["dt"].astype(str)
    ref_by_code = dict(tuple(reference.groupby("kdcode"))) if reference is not None else {}

    raw_log: list[dict[str, Any]] = []
    client = EodhdClient(api_key, args.cache_dir, raw_log)
    listings: dict[bool, list[dict[str, Any]]] = {}
    pieces: dict[str, list[pd.DataFrame]] = {}
    resolved: dict[str, Any] = {}
    findings: list[Finding] = []
    blanked_total = 0
    for index, (kdcode, plan) in enumerate(plans.items(), start=1):
        need = spans.loc[kdcode].to_dict() if kdcode in spans.index else None
        ref = ref_by_code.get(kdcode) if reference is not None else None
        if reference is not None and ref is None:
            ref = pd.DataFrame(columns=["kdcode", "dt", "close"])
        parts, chosen = [], []
        for segment in plan.segments:
            rows, choice, segment_findings = resolve_segment(
                client, plan, segment, need, ref, listings, args
            )
            parts.append(rows)
            chosen.append(choice)
            findings.extend(segment_findings)
            blanked_total += choice["values_blanked"]
        pieces[kdcode] = parts
        resolved[kdcode] = {"overridden": plan.overridden, "note": plan.note, "segments": chosen}
        symbols = ", ".join(str(c["symbol"]) for c in chosen)
        print(f"[{index}/{len(plans)}] {kdcode}: {symbols}", flush=True)

    panel = assemble_panel(pieces)
    missing = sorted(set(plans) - set(panel["kdcode"]))
    for kdcode in missing:
        findings.append(Finding(kdcode, "no_rows", True, "The panel has no rows for this name"))
    check = pd.DataFrame()
    basis: dict[str, Any] = {}
    if reference is not None:
        check, check_findings = reference_check(panel, reference, pit)
        findings.extend(check_findings)
        basis = adjustment_basis_summary(check)

    panel_path = market / f"{stem}.csv"
    panel_for_write(panel).to_csv(panel_path, index=False)
    coverage_path = market / f"{stem}_coverage.csv"
    coverage_table(panel).to_csv(coverage_path, index=False)
    check_path = market / f"{stem}_reference_check.csv"
    check.to_csv(check_path, index=False)
    symbols_path = market / f"{stem}_symbols.json"
    symbols_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    raw_path = market / f"{stem}_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in raw_log:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    blocking = [f for f in findings if f.blocking]
    acquired_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "created_at_utc": acquired_at,
        "source": "eodhd.com /api/eod, /api/splits, /api/exchange-symbol-list",
        "adjustment": "split-adjusted from EODHD split records as of the panel end; "
        "dividends not applied",
        "repository_commit": _git_commit(),
        "start": args.start,
        "end": args.end,
        "pit_universe": args.pit_universe.name,
        "pit_universe_sha256": _sha256(args.pit_universe),
        "symbol_map_sha256": _sha256(args.symbol_map),
        "reference_panel": args.reference_panel.name if args.reference_panel else None,
        "reference_manifest": args.reference_manifest.name if args.reference_manifest else None,
        "reference_manifest_sha256": reference_manifest_sha256,
        "reference_panel_sha256": _sha256(args.reference_panel) if args.reference_panel else None,
        "requested_identifiers": len(plans),
        "resolved_identifiers_with_rows": int(panel["kdcode"].nunique()) if len(panel) else 0,
        "missing_identifiers": missing,
        "rows": len(panel),
        "date_min": str(panel["dt"].min()) if len(panel) else None,
        "date_max": str(panel["dt"].max()) if len(panel) else None,
        "values_blanked": blanked_total,
        "vendor_calls": client.calls,
        "vendor_responses_cached": sum(1 for r in raw_log if r.get("from_cache")),
        "reference_check_failures": int((check.get("verdict") == "fail").sum())
        if len(check)
        else 0,
        "adjustment_basis_vs_reference": basis,
        "blocking_findings": len(blocking),
        "findings": [f.as_dict() for f in findings],
    }
    meta_path = market / f"{stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in meta.items() if k != "findings"}, indent=2))
    for finding in blocking:
        print(f"BLOCKING {finding.kdcode}: {finding.code}: {finding.detail}")
    if blocking:
        print("No manifest published: resolve the blocking findings above and re-run.")
        return 1

    if args.manifest_output is not None:
        files = [
            InputFileSpec(
                f"constituents/{args.pit_universe.name}",
                "Point-in-time membership used by the model",
            ),
            InputFileSpec(f"market/{panel_path.name}", "Price/volume panel and warmup history"),
            InputFileSpec(f"market/{meta_path.name}", "Market pull provenance and findings"),
            InputFileSpec(f"market/{coverage_path.name}", "Export coverage"),
            InputFileSpec(f"market/{check_path.name}", "Agreement with the reference panel"),
            InputFileSpec(f"market/{symbols_path.name}", "EODHD symbol read for each stretch"),
            InputFileSpec(f"market/{map_target.name}", "Reviewed symbol map used"),
            InputFileSpec(f"market/{raw_path.name}", "Vendor responses as received"),
        ]
        snapshot = write_input_manifest(
            args.manifest_output,
            args.package_root,
            package_id=f"{prefix}_eodhd",
            package_revision=args.package_revision,
            files=files,
            provenance={
                "source": meta["source"],
                "acquisition_mode": "live" if not meta["vendor_responses_cached"] else "cache",
                "acquired_at": acquired_at,
                "producing_command": "python -m scripts.data.export_eodhd_pit_prices",
                "producing_arguments": sys.argv[1:] if argv is None else list(argv),
                "unknowns": [],
            },
            metadata={
                "description": "EODHD daily prices for the preserved 110-name PIT universe "
                "(#281); membership is the LSEG-derived file, byte-identical.",
                "repository_commit": meta["repository_commit"],
            },
        )
        print(json.dumps({"manifest": str(args.manifest_output), "sha256": snapshot.sha256}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
