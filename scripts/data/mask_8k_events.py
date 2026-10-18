"""E13: the 8-K events with the filer's identity hidden, ready for Jev.

Writes the E11 test events twice, once per mask level of
:mod:`mci_gru.filings.masking`, in the event contract that
``scripts/data/score_8k_events.py`` reads, so the same frozen question can be
asked of them unchanged::

    python -m scripts.data.mask_8k_events \\
        --events <member_8k_events.jsonl> \\
        --sample <E11 events.csv> \\
        --companies <membership/sp500_companies.csv> \\
        --sec-submissions <folder of CIK json files with name, formerNames, tickers> \\
        --output-dir /mnt/project-files/filings/experiments/E13-name-masking/masked

Under ``--output-dir``: ``name/events.jsonl`` and ``deep/events.jsonl``, each
with a ``mask_counts.csv`` (what was replaced, per event); ``identities.csv``
(the names and tickers each filer is masked by); and ``masked.meta.json``
(counts, the residual check and every file's SHA-256).

The residual check is the deterministic half of the leak test: after the name
mask, no event's text may still hold its filer's legal-name core, a
distinctive short name, or a ticker after an exchange name. Any that does
stops the run. Jev's own leak check (can it still tell who filed?) is asked
separately, in ``scripts/data/identify_8k_company.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from mci_gru.filings.events import EVENT_FIELDS, Event, load_events
from mci_gru.filings.masking import (
    EXCHANGES,
    Identity,
    MaskCounts,
    capital_words,
    company_vocabulary,
    core_tokens,
    mask_entities,
    mask_names,
    name_patterns,
    text_words,
    ticker_patterns,
)
from scripts.data.score_8k_events import sampled_events
from scripts.data.score_company_risk_changes import _commit, _sha256

try:
    import spacy
except ImportError:  # only the deep mask needs the tagger
    spacy = None

TAGGER = "en_core_web_sm"
TAGGER_BATCH = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True, help="CSV with an event_id column")
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--sec-submissions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1, help="Tagger processes")
    parser.add_argument("--skip-deep", action="store_true", help="Name mask only")
    return parser.parse_args(argv)


def identities(
    companies: pd.DataFrame, submissions: Path, events: list[Event]
) -> dict[str, Identity]:
    """Each filer's legal names (member table, its events, SEC current), former names and tickers."""
    seen_names: dict[str, set[str]] = {}
    for event in events:
        seen_names.setdefault(event.company, set()).add(event.company_name)
    found: dict[str, Identity] = {}
    for row in companies.itertuples():
        names = {str(row.name)} | seen_names.get(row.key, set())
        former: set[str] = set()
        tickers = {t for t in str(row.tickers).split(";") if t and t != "nan"}
        for cik in re.split(r"[;|]", str(row.ciks)):
            path = submissions / f"{cik}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("name"):
                names.add(record["name"])
            former.update(entry["name"] for entry in record.get("formerNames") or [])
            tickers.update(record.get("tickers") or [])
        found[row.key] = Identity(
            names=tuple(sorted(names)),
            tickers=tuple(sorted(tickers)),
            former=tuple(sorted(former - names)),
        )
    return found


def residuals(event: Event, identity: Identity) -> list[str]:
    """Identifiers of the filer the name mask should have removed but did not."""
    left = []
    lowered = event.text.lower()
    for name in identity.names + identity.former:
        core = core_tokens(name)
        if len(core) > 1:
            found = re.search(
                r"\b" + r"[\s,.]*".join(re.escape(t.lower()) for t in core) + r"(?![a-z0-9])",
                lowered,
            )
        elif core:
            # A one-word core is always masked when capitalised or in capitals.
            spellings = {core[0].capitalize(), core[0].upper()}
            found = re.search(
                r"\b(?:" + "|".join(map(re.escape, spellings)) + r")(?![A-Za-z0-9])", event.text
            )
        else:
            found = None
        if found:
            left.append(" ".join(core))
    for ticker in identity.tickers:
        if re.search(
            r"\b" + EXCHANGES + r"\s*:\s*" + re.escape(ticker) + r"(?![A-Za-z0-9])", event.text
        ):
            left.append(ticker)
    return sorted(set(left))


def write_events(path: Path, events: list[Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            record: dict[str, Any] = {
                "event_id": event.event_id,
                "company": event.company,
                "company_name": event.company_name,
                "items": list(event.items),
                "text": event.text,
                "prior_revenue_usd": event.prior_revenue_usd,
                "prior_revenue_fiscal_year": event.prior_revenue_fiscal_year,
                "prior_90d_items": [list(codes) for codes in event.prior_90d_items],
            }
            assert tuple(record) == EVENT_FIELDS
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def counts_table(events: list[Event], counts: list[MaskCounts]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event_id": e.event_id, "company": e.company, **asdict(c)}
            for e, c in zip(events, counts, strict=True)
        ]
    )


def main(argv: list[str] | None = None, *, tagger: Any = None) -> int:
    args = parse_args(argv)
    every = load_events(args.events)
    sample = pd.read_csv(args.sample, usecols=["event_id"], dtype=str)
    events = sampled_events(every, sample)
    companies = pd.read_csv(args.companies, dtype=str)
    who = identities(companies, args.sec_submissions, every)
    missing = sorted({e.company for e in events} - set(who))
    if missing:
        print(
            f"No identity for {len(missing)} filers: {', '.join(missing[:10])}. Refusing to go on."
        )
        return 1
    # Vocabulary over every published event, so "few other filers use it" is judged on all of them.
    vocabulary = company_vocabulary((e.company, e.text) for e in every)
    own_lower: dict[str, set[str]] = {}
    own_capital: dict[str, set[str]] = {}
    for event in every:
        own_lower.setdefault(event.company, set()).update(text_words(event.text))
        own_capital.setdefault(event.company, set()).update(capital_words(event.text))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    patterns = {
        key: (
            name_patterns(
                who[key],
                vocabulary,
                own_words=own_lower.get(key, set()),
                own_capitals=own_capital.get(key, set()),
            ),
            ticker_patterns(who[key], vocabulary, own_words=own_lower.get(key, set())),
        )
        for key in sorted({e.company for e in events})
    }
    pd.DataFrame(
        [
            {
                "company": key,
                "names": " | ".join(who[key].names),
                "former_names": " | ".join(who[key].former),
                "tickers": " | ".join(who[key].tickers),
                "name_patterns": " | ".join(
                    ("[case] " if p.case_sensitive else "") + p.regex for p in names
                ),
                "ticker_patterns": len(tickers),
            }
            for key, (names, tickers) in patterns.items()
        ]
    ).to_csv(out / "identities.csv", index=False)

    named: list[Event] = []
    name_counts: list[MaskCounts] = []
    left: dict[str, list[str]] = {}
    for event in events:
        masked, counts = mask_names(event, *patterns[event.company])
        named.append(masked)
        name_counts.append(counts)
        found = residuals(masked, who[event.company])
        if found:
            left[event.event_id] = found
    if left:
        print(f"{len(left)} events keep an identifier after the name mask, such as:")
        for event_id, found in list(left.items())[:10]:
            print(f"  {event_id}: {', '.join(found)}")
        print("Refusing to go on.")
        return 1
    write_events(out / "name" / "events.jsonl", named)
    counts_table(named, name_counts).to_csv(out / "name" / "mask_counts.csv", index=False)
    meta: dict[str, Any] = {
        "experiment": "E13-name-masking",
        "commit": _commit(),
        "events": len(named),
        "name": {
            "events_with_a_name_replaced": sum(c.names > 0 for c in name_counts),
            "names_replaced": sum(c.names for c in name_counts),
            "tickers_replaced": sum(c.tickers for c in name_counts),
            "web_and_email_replaced": sum(c.web for c in name_counts),
            "cut_to_limit": sum(c.cut_to_limit for c in name_counts),
            "residual_identifiers": 0,
        },
    }
    files = ["identities.csv", "name/events.jsonl", "name/mask_counts.csv"]

    if not args.skip_deep:
        if tagger is None:
            if spacy is None:
                print("The deep mask needs spaCy and its en_core_web_sm model.")
                return 1
            tagger = spacy.load(
                TAGGER, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"]
            )
            meta["tagger"] = {
                "model": TAGGER,
                "version": tagger.meta.get("version"),
                "spacy": spacy.__version__,
            }
        deep: list[Event] = []
        deep_counts: list[MaskCounts] = []
        docs = tagger.pipe(
            (e.text for e in named), batch_size=TAGGER_BATCH, n_process=args.processes
        )
        for event, doc in zip(named, docs, strict=True):
            spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            masked, counts = mask_entities(event, spans)
            deep.append(masked)
            deep_counts.append(counts)
        write_events(out / "deep" / "events.jsonl", deep)
        counts_table(deep, deep_counts).to_csv(out / "deep" / "mask_counts.csv", index=False)
        meta["deep"] = {
            "entities_replaced": sum(c.entities for c in deep_counts),
            "phones_replaced": sum(c.phones for c in deep_counts),
            "cut_to_limit": sum(c.cut_to_limit for c in deep_counts),
            "revenue": "rounded to one significant figure",
        }
        files += ["deep/events.jsonl", "deep/mask_counts.csv"]

    meta["sha256"] = {
        "events": _sha256(args.events),
        "sample": _sha256(args.sample),
        "companies": _sha256(args.companies),
        **{name: _sha256(out / name) for name in files},
    }
    (out / "masked.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
