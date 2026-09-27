"""Parse per-epoch validation IC from the smoke logs into a markdown table."""
import json
import os
import re
import sys

runs = sys.argv[1]
variants = ["baseline", "no_self_attention", "residual", "pre_ln_residual"]
pat = re.compile(r"Epoch \[(\d+)/(\d+)\].*?Val IC: (-?[0-9.]+), Val Rank IC: (-?[0-9.]+)")
rows = {}
best = {}
test = {}
params = {}
for v in variants:
    p = os.path.join(runs, v + ".out.log")
    if not os.path.exists(p):
        continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    m = re.search(r"params=(\d+)", txt)
    params[v] = int(m.group(1)) if m else None
    rows[v] = {int(a): (float(c), float(d)) for a, b, c, d in pat.findall(txt)}
    summ = os.path.join(runs, v, "evaluation_summary.json")
    if os.path.exists(summ):
        try:
            d = json.load(open(summ))
            test[v] = d
        except Exception as e:  # noqa: BLE001
            test[v] = {"error": str(e)}
    ts = os.path.join(runs, v, "training_summary.json")
    if os.path.exists(ts):
        best[v] = json.load(open(ts))

n_ep = max((max(r) for r in rows.values() if r), default=0)
hdr = "| Epoch | " + " | ".join(variants) + " |"
print(hdr)
print("|---:|" + "---:|" * len(variants))
for e in range(1, n_ep + 1):
    cells = []
    for v in variants:
        r = rows.get(v, {}).get(e)
        cells.append(f"{r[0]:+.4f}" if r else "")
    print(f"| {e} | " + " | ".join(cells) + " |")
print()
print("params:", params)
for v in variants:
    if v in best:
        b = best[v]
        print(f"{v}: best_val_ics={b.get('best_val_ics')} best_val_rank_ics={b.get('best_val_rank_ics')}")
for v in variants:
    if v in test:
        d = test[v]
        keys = [k for k in d if isinstance(d[k], (int, float)) and ("ic" in k.lower())]
        print(f"{v} evaluation_summary numeric ic-like keys:", {k: round(d[k], 5) for k in keys[:8]})
        if not keys:
            print(f"{v} evaluation_summary top-level keys:", list(d)[:15])
