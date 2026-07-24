# MCI-GRU Repo Rearchitecture — Addendum Spec (WS-L … WS-Q)

> Companion to `docs/REARCHITECTURE_TECHNICAL_SPEC_2026-07-01.md` (the "base
> spec", workstreams WS-A and WS-C…WS-K). That spec remains the plan of record for
> everything it covers. This addendum formalizes the six workstreams surfaced
> by the 2026-07-02 full-repo architecture review that the base spec does not
> cover, and replaces the base spec's Sequencing Summary with a consolidated
> master sequencing across all sixteen retained workstreams.
>
> **Same ground rules as the base spec:** behavior-preserving first, one
> workstream = one PR = one green test run
> (`.\.venv\Scripts\python.exe scripts\run_pytest_isolated.py tests/ -v`
> and `ruff check .`), rollback unit = one workstream, no new abstractions
> without a second caller. Workstreams below flag explicitly where a change is
> **not** behavior-preserving.

## Contents

- [WS-L: Notebook generator shared library](#ws-l-notebook-generator-shared-library-scriptsnb_libpy)
- [WS-M: Single config ingestion path + runner slimming](#ws-m-single-config-ingestion-path--runner-slimming)
- [WS-N: Merge the two backtest engines](#ws-n-merge-the-two-backtest-engines)
- [WS-O: Structured logging in core modules](#ws-o-structured-logging-in-core-modules)
- [WS-P: Cross-layer coupling fixes](#ws-p-cross-layer-coupling-fixes)
- [WS-Q: Repository hygiene sweep](#ws-q-repository-hygiene-sweep)
- [Master sequencing (all 16 retained workstreams)](#master-sequencing-all-16-retained-workstreams)
- [Acceptance criteria (addendum)](#acceptance-criteria-addendum)

---

## WS-L: Notebook generator shared library (`scripts/nb_lib.py`)

**Priority note: this workstream must land BEFORE base-spec WS-C.** WS-C
requires editing the embedded `tests/backtest_sp500*.py` path string in 11
generators. After WS-L, that becomes a one-line change in the library instead
of an 11-file sweep.

**Current state (verified):** 16 `scripts/gen_*_nb.py` generators share zero
code. Duplication exists at two distinct levels, and the fix differs per
level:

1. **Generator-level Python helpers** — duplicated verbatim in all 16 files:
   - `md(text) -> dict` and `code(text) -> dict` cell constructors
     (e.g. `scripts/gen_lambdarank_ic_pit_nb.py:12-27`,
     `scripts/gen_portfolio_ic_pit_nb.py:12-27`,
     `scripts/gen_volatility_targeting_ablation_nb.py:12-27` — identical).
   - `build_notebook(...) -> dict` nbformat-4 shell (e.g.
     `gen_lambdarank_ic_pit_nb.py:648-662`, `gen_portfolio_ic_pit_nb.py:499-513`;
     some generators inline it instead).
2. **Generated cell source text** — near-identical Colab preamble strings
   emitted *into* the notebooks (e.g. `gen_lambdarank_ic_pit_nb.py:56-156`):
   `IN_COLAB` detection, `REPO_URL`/`BRANCH`/`REPO_DIR` constants, GPU
   preflight (`detect_gpu_name`, `BLOCKED_GPU_NAMES`, `ALLOWED_GPU_MARKERS`),
   Drive mount, clone/checkout/pull, the three `pip install` commands, and
   `sys.path.insert`. Parameters that genuinely vary per generator: `BRANCH`,
   pip extras (`[dev,tracking,fred]` vs subsets), GPU strictness flags, and
   the experiment-specific probe block at the end of the setup cell.

Additionally, 11 generators embed `tests/backtest_sp500*.py` subprocess
invocation strings (enumerated in base spec WS-C), one more
(`gen_2022_weak_year_investigation_nb.py:768,924`) references the path in
markdown/comment text only, and two `run_*` orchestration scripts plus
`paper_trade/scripts/compare_regime.py` invoke the same paths outside the
generator family.

**Target layout:**

```
scripts/
├── nb_lib.py            # new shared module
└── gen_*_nb.py          # keep experiment-specific cells only
```

**`nb_lib.py` exact public surface:**

```python
# scripts/nb_lib.py

def md(text: str) -> dict: ...          # moved verbatim
def code(text: str) -> dict: ...        # moved verbatim

def build_notebook(cells: list[dict]) -> dict:
    """nbformat-4 shell. Single definition replacing per-file copies."""

def write_notebook(cells: list[dict], out_path: Path) -> None:
    """json.dump with the indent/newline conventions the contract tests pin."""

def colab_setup_cell(
    *,
    branch: str,
    pip_extras: str = "dev,tracking,fred",
    require_gpu: bool = True,
    blocked_gpu_names: tuple[str, ...] = ("T4",),
    allowed_gpu_markers: tuple[str, ...] = (
        "G4", "L4", "A100", "H100", "V100", "RTX PRO", "BLACKWELL",
    ),
    strict_gpu_markers: tuple[str, ...] = (),
    extra_setup_source: str = "",
) -> dict:
    """Returns the standard Colab setup *code cell* (repo clone, pip install,
    GPU preflight). `extra_setup_source` is appended verbatim so generators
    keep their experiment-specific probe blocks (e.g. the
    `build_training_loss` probe in gen_lambdarank_ic_pit_nb.py:150-155)."""

def backtest_invocation_source(
    *,
    engine: str,           # "backtest_sp500" | "backtest_sp500_daily"
    args: Sequence[str],
) -> str:
    """Returns the subprocess-invocation source string. THIS is the single
    place WS-C later edits when the engines move from tests/ to scripts/."""
```

**Import mechanics (no packaging change needed):** generators are invoked as
`python scripts/gen_*_nb.py` from repo root; Python puts the script's own
directory (`scripts/`) at `sys.path[0]`, so a plain `import nb_lib` resolves
without making `scripts/` a package. Verified: no test imports a generator as
a module — the 14 notebook contract tests read generator **source text** and
the generated `.ipynb` only (e.g. `tests/test_pit_masked_panel_notebook.py:5-7`).

**Migration protocol (strictness is the point):**

1. Create `nb_lib.py` with `md`/`code`/`build_notebook`/`write_notebook`
   only. Port all 16 generators to import them. Regenerate all 16 notebooks:
   output must be **byte-identical** (`git diff --stat notebooks/` empty).
   This is PR 1.
2. Add `colab_setup_cell(...)` and port generators one at a time, comparing
   regenerated notebooks. Where a generator's setup cell has drifted from the
   canonical form, the diff makes the drift visible — decide per case whether
   to preserve it via `extra_setup_source` (behavior-preserving) or adopt the
   canonical cell (flag in the PR as a deliberate change; contract tests may
   need updating). This is PR 2.
3. Add `backtest_invocation_source(...)` and port the 11 generators that
   embed backtest paths. PR 3. WS-C is unblocked after this merges.

**Contract-test interaction:** several contract tests assert on generator
source text. Any test asserting on the *duplicated helper text* (rather than
experiment-specific cells) must be updated to assert against the generated
notebook instead — the notebook is the real contract. List the affected
assertions in the PR description.

**Risk:** low-medium. Pure text-plumbing; the byte-identical gate in step 1
and per-generator diffs in steps 2–3 make regressions visible immediately.
**Verification:** all 16 notebooks regenerate byte-identical after PR 1; the
14 notebook contract tests green after each PR; `ruff check .` clean.

---

## WS-M: Single config ingestion path + runner slimming

**Depends on:** nothing. Can run in Phase 0.

### M1 — Delete `run_experiment.py::dict_to_config`

**Current state (verified):** two parallel Hydra-dict → `ExperimentConfig`
converters:

| | `run_experiment.py:161-182` (`dict_to_config`) | `mci_gru/config.py:896-928` (`create_config_from_dict`) |
|---|---|---|
| Input | `DictConfig` (calls `OmegaConf.to_container` itself) | plain `dict` |
| Sub-config construction | always `DataConfig(**cfg_dict.get("data", {}))` | `DataConfig(**data_dict) if data_dict else DataConfig()` — equivalent |
| `seed` fallback | **42** | **1729** |
| `experiment_name`/`output_dir` fallback | `"baseline"` / `"results"` | same |

The `seed` fallback divergence is live proof of the drift risk: an entry
point that omits `seed` gets a different experiment depending on which
converter it used. In practice `configs/config.yaml:111` always sets
`seed: 1729` and `ExperimentConfig.seed` defaults to 1729
(`mci_gru/config.py:707`), so **42 is the outlier** — `dict_to_config` is the
buggy copy.

**Change:**

```python
# run_experiment.py — replace lines 161-182 and the call at line 202 with:
from mci_gru.config import create_config_from_dict
...
config = create_config_from_dict(OmegaConf.to_container(cfg, resolve=True))
```

**Not behavior-preserving, flagged:** the `seed` fallback changes 42 → 1729
for any caller that passes a config dict with no `seed` key. No committed
Hydra config hits this path (`configs/config.yaml` sets it), but state the
change in the PR description and add a regression test pinning the fallback:

```python
def test_hydra_ingestion_seed_fallback_is_1729(): ...
```

**Call sites requiring change (exhaustive):**

| File:Line | Change |
|---|---|
| `run_experiment.py:161-182,202` | delete function, call `create_config_from_dict` |
| `tests/test_mlflow_tracking.py:18,52` | currently imports **both** converters and compares them (`:45-52`) — after M1 that comparison is an identity check. Rewrite the test to assert `create_config_from_dict` correctly ingests an `OmegaConf.to_container` product (keep the Hydra-round-trip coverage, drop the dual-path comparison) |

### M2 — Move evaluation-summary helpers into the package

**Current state (verified):** four helpers in `run_experiment.py` are
package logic living in the entry point:

| Helper | Lines | Nature |
|---|---|---|
| `_data_file_fingerprint` | 66-88 | SHA-256 + stat of the data CSV → run metadata |
| `_resolved_evaluation_kwargs` | 91-104 | config → `evaluate_predictions` kwargs (block_size/NW-lag defaults) |
| `_compute_evaluation_summary` | 107-121 | wraps `evaluate_predictions` into the summary dict |
| `_select_training_objective_value` | 124-141 | selection-metric → summary-key mapping |

**Target:** new module `mci_gru/evaluation/experiment_summary.py` with the
four functions public (drop the leading underscore):
`data_file_fingerprint`, `resolved_evaluation_kwargs`,
`compute_evaluation_summary`, `select_training_objective_value`. Move
verbatim; `run_experiment.py` imports them. `setup_logging`
(`run_experiment.py:144-158`) **stays in the runner** — process-global
`logging.basicConfig(force=True)` is an entry-point concern (see WS-O).

Rationale: these encode research policy (Newey-West lag defaults derived
from `label_t`, the selection-metric key mapping). Policy belongs in the
package where it is importable and unit-testable; the runner keeps only
wiring. Post-M1+M2, `run_experiment.py` drops from 530 to roughly 350 lines;
further slimming (the per-window artifact I/O loop, lines 283-494) is out of
scope until base-spec WS-J stabilizes `prepare_data`'s return contract.

**Risk:** low. Pure move plus one flagged fallback change.
**Verification:** full pytest; new unit tests for the four moved functions
(previously untestable without importing the runner); one smoke
`run_experiment.py` run whose `run_metadata.json` matches a pre-change run
field-for-field.

---

## WS-N: Merge the two backtest engines

**Depends on:** base-spec WS-C (engines relocated out of `tests/`, core in
`mci_gru/evaluation/backtest_engine.py`, thin CLIs in `scripts/`). Do NOT
attempt relocation and merge in one PR — WS-C is already the largest-surface
Phase 0 workstream.

**Current state (verified via function-surface diff):**
`tests/backtest_sp500.py` (3,573 lines, 43 top-level functions) and
`tests/backtest_sp500_daily.py` (2,603 lines, 39 functions) are a fork, not
siblings. The daily file is a strict feature-subset with drifted line
numbers:

**Only in `backtest_sp500.py`:**

| Function | Lines | Purpose |
|---|---|---|
| `simulate_trading_strategy_staggered` | 1419-1784 | multi-day holding, staggered tranches |
| `simulate_trading_strategy_block` | 1785-2142 | multi-day holding, block rebalance |
| `_infer_experiment_name` | 2909-2916 | MLflow run naming |
| `setup_backtest_tracking` | 2917-2976 | MLflow backtest tracking |
| `_log_backtest_artifacts` | 2977-2995 | MLflow artifact logging |

**Shared by name in both files (drift status unknown until step 1):** all 38
remaining functions — metric calculators (`calculate_arr` … `haircut_sharpe_ratio`),
data loading (`resolve_data_file`, `load_stock_data`, `calculate_forward_returns`,
`load_predictions`), PIT helpers (`load_pit_universe_for_backtest`,
`_pit_active_*`, `equal_weight_benchmark_daily_series`,
`calendar_returns_for_evaluation_window`), the daily
`simulate_trading_strategy`, `evaluate`, results I/O
(`print_results`, `save_results`, `save_backtest_results`,
`derive_portfolio_composition`, `derive_holdings_summary`), and `main`.

**Target:** one engine module; "daily" is a configuration, not a fork.

```
mci_gru/evaluation/backtest_engine.py   # single source of truth
scripts/backtest_sp500.py               # thin CLI (full argparse surface)
scripts/backtest_sp500_daily.py         # thin CLI: fixed holding_period=1,
                                        # tracking off — preserved so the 12
                                        # subprocess call sites keep a stable
                                        # command name
```

**Steps:**

1. **Divergence audit (its own PR-less artifact, before any code change).**
   For each of the 38 shared-name functions, diff the two bodies
   (post-WS-C locations). Produce
   `docs/audits/BACKTEST_ENGINE_DIVERGENCE_<date>.md` with three buckets:
   *identical*, *cosmetic drift* (comments/format), *behavioral drift*.
   Every behavioral-drift entry needs a decision: which behavior is correct,
   recorded with rationale. This audit is the merge's requirements document —
   do not skip it; the whole reason this fork is dangerous is that nobody
   currently knows the drift set.
2. **Golden-output fixtures (before the merge).** Build a small synthetic
   prediction set + stock CSV under `tests/fixtures/backtest_golden/`
   (deterministic, committed). Run **both legacy engines** on it across the
   argument matrix: daily; staggered `holding_period=5`; block
   `holding_period=5`; PIT-universe on/off; transaction costs on/off. Commit
   the resulting metrics JSONs as goldens and add
   `tests/test_backtest_engine_golden.py` asserting each engine reproduces
   its own goldens. Merge this while both engines still exist — it pins
   current behavior.
3. **Merge.** Fold the daily file into the engine: daily = the existing
   `simulate_trading_strategy` path; `setup_backtest_tracking` gated by a
   `--tracking/--no-tracking` flag (default off for the daily CLI, matching
   today). Where step 1 found behavioral drift, apply the recorded decision
   and update the affected golden **in the same commit, with the audit line
   cited**. Delete the daily engine body; its CLI becomes a wrapper.
4. **Re-run goldens.** Daily goldens must match bit-for-bit unless a step-1
   decision says otherwise; staggered/block goldens must match bit-for-bit
   (that code has a single source already).

**Call sites:** none beyond WS-C's — both CLI command names survive, and
`load_pit_universe_for_backtest` importers move to the engine module (already
enumerated in base spec WS-C).

**Not behavior-preserving, flagged:** wherever step 1 finds behavioral drift,
one of the two current behaviors changes by definition. The audit + golden
mechanism exists precisely to make each such change explicit, reviewed, and
pinned.

**Risk:** medium-high in aggregate, but decomposed into low-risk steps: the
audit is read-only, the goldens are additive, and the merge lands against
pinned behavior. **Verification:** golden suite green; full pytest;
`scripts/run_pit_saved_prediction_backtests.py` smoke run produces outputs
matching a pre-merge run.

---

## WS-O: Structured logging in core modules

**Depends on:** nothing, but sequence AFTER Phase 2 (WS-I/J/K) to avoid
churning `pipeline.py` while its no-lookahead-sensitive refactor is in
flight. Fine to run alongside WS-N.

**Current state (verified counts of `print(` in core modules):**

| Module | print calls |
|---|---|
| `mci_gru/pipeline.py` | 27 |
| `mci_gru/data/lseg_loader.py` | 25 |
| `mci_gru/data/data_manager.py` | 20 |
| `mci_gru/training/trainer.py` | 18 |
| `mci_gru/features/momentum.py` | 7 |
| `mci_gru/features/registry.py` | 6 |
| `mci_gru/graph/builder.py` | 5 |

~108 calls in these seven files alone; zero `logging` usage in core paths.
Meanwhile `run_experiment.py::setup_logging` (lines 144-158) already
configures the root logger with a `FileHandler` + `StreamHandler(sys.stdout)`
— so today's training log files **miss everything the pipeline prints**.

**Change pattern (mechanical, per module):**

```python
import logging

logger = logging.getLogger(__name__)
# print(f"...") → logger.info("...")   [f-string → %-style or keep f-string; pick one, apply uniformly]
```

**Rules:**

- Library code never calls `logging.basicConfig` — configuration stays in
  entry points (`run_experiment.py`, `setup_backtest_logging` in the backtest
  engine, `paper_trade/scripts/run_nightly.py`).
- Diagnostics that are warnings in spirit (soft-fail VIX/credit/regime loads,
  e.g. `pipeline.py::_load_auxiliary_data`) become `logger.warning`; existing
  `warnings.warn` calls that mark API deprecation (e.g.
  `data_manager.py:228-233`) stay `warnings.warn`.
- Entry points must keep a `StreamHandler(sys.stdout)` so console behavior is
  visually unchanged for interactive/Colab runs. Colab notebooks invoke
  training via subprocess and surface stdout — this is load-bearing, not
  cosmetic.

**Verified non-blocker:** no test uses `capsys`/`capfd` (grep: zero hits), so
no stdout-assertion updates are needed.

**Scope control:** the seven files above only. `scripts/` and notebook
generators keep `print` (they are CLIs; stdout is their interface).

**Risk:** low. **Verification:** full pytest; one smoke `run_experiment.py`
run confirming the `training_*.log` file now contains pipeline/trainer lines
that previously went only to the console.

---

## WS-P: Cross-layer coupling fixes

**Depends on:** nothing; each fix rides along whenever its file is next
touched (WS-F/G/H/I/J all touch these files). Three independent fixes —
they can be one PR or three ride-alongs.

### P1 — `config` must not import `features`

**Current:** `FeatureConfig.__post_init__` lazily imports
`resolve_volatility_targeting_components` from `mci_gru/features/volatility.py`
(`mci_gru/config.py:244-248`). Config — the layer everything else depends on
— reaches *up* into a domain layer. The lazy import avoids a load-time cycle
but the dependency is real.

**Fix (move the leaf, point both callers at it):**
`resolve_volatility_targeting_components` (`features/volatility.py:44-66`)
and `DEFAULT_VOLATILITY_TARGETING_COMPONENTS` are pure, dependency-free
validation/constants. Move both into `mci_gru/config.py` (they are config
vocabulary — the function validates a config field). `features/volatility.py`
imports them from `config` (features → config is the correct direction;
`features/registry.py` already lazily imports `FeatureConfig`). Keep a
re-export in `features/volatility.py` for any external caller:

| Call site | Change |
|---|---|
| `mci_gru/config.py:244` | delete lazy import; call the now-local function |
| `mci_gru/features/volatility.py:44-66` | replace with `from mci_gru.config import resolve_volatility_targeting_components, DEFAULT_VOLATILITY_TARGETING_COMPONENTS` re-export |
| grep `resolve_volatility_targeting_components` repo-wide before merging | update any importer of the old path |

### P2 — `data` must not import `features` for the regime contract

**Current:** `DataManager.load_regime_inputs` imports `REGIME_VARIABLES`
from `mci_gru/features/regime.py` (`data_manager.py:225`, used at
`:239,248,249,252,254`). The data layer knows feature-layer column names.

**Fix:** the regime column lists (`REGIME_REQUIRED_VARIABLES`,
`REGIME_OPTIONAL_VARIABLES`, `REGIME_VARIABLES` — `features/regime.py:27-40`)
are a **data contract**, already documented as one in
`docs/REGIME_DATA_CONTRACT.md`. Move the three constants to a new leaf module
`mci_gru/regime_contract.py` (no imports beyond stdlib). Both
`features/regime.py` and `data/data_manager.py` import from it. Re-export
from `features/regime.py` for compatibility. Update
`docs/REGIME_DATA_CONTRACT.md` to name the module as the contract's home.

### P3 — Break the `training` ↔ `evaluation` cycle

**Current:** `training/metrics.py:9-14` imports `evaluation.portfolio` and
`evaluation.statistics`; `evaluation/prediction_report.py:338` lazily imports
`evaluate_predictions` back from `training.metrics`. Soft cycle held together
by one lazy import.

**Fix (relocate, one-way afterwards):** `evaluate_predictions` computes
ranking/IC metrics on saved predictions — it is evaluation logic that
happens to live in `training/`. Move `evaluate_predictions` (and its private
helpers) from `training/metrics.py` to `mci_gru/evaluation/metrics.py`.
`training/metrics.py` re-exports it (its lazy `training/__init__` facade and
`run_experiment.py:61` keep working unchanged). `prediction_report.py:338`
switches to the intra-package import and stops being lazy. Result:
`training → evaluation` one-way; no cycle.

**Explicitly out of scope:** the duplicated `top_k`/`top_k_metric` validation
in `GraphConfig.__post_init__` and `GraphBuilder.__init__`
(base spec WS-G territory) — defense-in-depth at two construction sites is
acceptable; dedupe only if WS-G's split makes it free.

**Risk:** low. All three are move-plus-re-export. **Verification:** full
pytest; `ruff check .` (catches dangling imports);
`rg "from mci_gru.features.volatility import resolve"`,
`rg "from mci_gru.features.regime import REGIME_VARIABLES"`, and
`rg "from mci_gru.training.metrics import evaluate_predictions"` each show
only the sanctioned re-export/import sites.

---

## WS-Q: Repository hygiene sweep

**Depends on:** nothing. Any time. Cheap.

**Current state (verified `git ls-files` counts):**

| Item | Tracked files | Status |
|---|---|---|
| `Seed_test (1).ipynb` (repo root) | 1 | legacy notebook, filename with space + `(1)` download suffix |
| `Program explainers/` | 1 (`paper_trade_system_reference_1393a11f.plan.md`) | orphaned plan doc in a spaced dirname |
| `seed_results/` | 46 | committed experiment artifacts (backtest metrics, equity curves, configs), ruff-excluded, documented as non-authoritative |
| `_uncertain/` | 22 | duplicate seed-test configs/partial results, ruff-excluded |
| `archive/` | 6 | legacy training scripts, ruff-excluded, AGENTS.md says "do not treat as current" |

**Actions (ordered so nothing is lost):**

1. **Preserve first.** Create an annotated tag before any deletion so every
   artifact remains reachable:
   `git tag archive/pre-hygiene-2026-07 && git push origin archive/pre-hygiene-2026-07`.
2. **Delete from the working tree** (history retains everything; the tag
   makes it findable): `Seed_test (1).ipynb`, `_uncertain/`, `archive/`,
   `seed_results/`. Before each deletion, `rg` the path/dirname repo-wide;
   the only expected hits are AGENTS.md's gotchas section and ruff excludes —
   update both.
3. **Relocate** `Program explainers/paper_trade_system_reference_1393a11f.plan.md`
   → `docs/agent_references/cursor/plans/` (where the other `.plan.md` files
   live); delete the spaced directory.
4. **Trim tooling config:** remove the now-dead `archive/`, `seed_results/`,
   `_uncertain/` entries from `[tool.ruff]` excludes (`pyproject.toml:45-57`)
   and from AGENTS.md's "Key Gotchas".
5. **Land base-spec WS-E** (docs source-of-truth CI check) in the same
   sweep — it stops the next generation of root-level dated reports.

**Decision required from the repo owner before step 2:** whether any
`seed_results/` artifacts are still referenced by open research threads. If
yes, move those specific files to `docs/research/current/` attachments
instead of deleting. Grep evidence (`rg "seed_results" docs/ scripts/ tests/`)
must be attached to the PR.

**Not behavior-preserving in the trivial sense** (files disappear) but zero
code-path impact: all four targets are ruff-excluded and imported by nothing
(verify with `rg` per step 2 before deleting).

**Risk:** near-zero with the tag in place. **Verification:** full pytest;
`ruff check .`; CI green; `rg` shows no dangling references.

---

## Master sequencing (all 16 retained workstreams)

Supersedes the base spec's Sequencing Summary. Phases 0–2 keep their
base-spec gates; addendum workstreams slot in as follows.

```
Phase 0 (parallel)                Phase 1 (parallel, after Phase 0)   Phase 2 (sequential, after Phase 1)
──────────────────────────       ─────────────────────────────      ───────────────────────────────────
WS-A  deps/lockfile               WS-F  models split                  WS-I  transforms.py extraction
WS-D  pytest markers              WS-G  graph/builder split                  ↓
WS-E  docs SOT check              WS-H  trainer/ensemble split        WS-J  pipeline.py staging
WS-M  config ingestion + slim                                                ↓
WS-Q  hygiene sweep                                                   WS-K  infer.py adopts transforms
WS-L  nb_lib  ──────┐
                    ▼
WS-C  backtest relocation         Phase 3 (after WS-C merged)
      (GATED on WS-L PR 3)        ─────────────────────────────
                                  WS-N  backtest engine merge
                                        (audit → goldens → merge)

Anytime ride-alongs: WS-P (with WS-F/G/H/I/J touches)
After Phase 2:       WS-O  logging (avoids pipeline.py churn during WS-J)
```

**Hard ordering constraints (everything else is parallelizable):**

1. WS-L PR 3 → WS-C (backtest path becomes a one-line library edit).
2. WS-C → WS-N (merge needs the engines' post-relocation home).
3. WS-N step 1 (audit) → step 2 (goldens) → step 3 (merge), strictly.
4. WS-I → WS-J → WS-K (base spec, unchanged).
5. WS-J merged → WS-O touches `pipeline.py`.

**Per-PR gate (unchanged from base spec):** `ruff check .` clean, full
pytest green on Windows venv through `.\.venv\Scripts\python.exe scripts\run_pytest_isolated.py tests/ -v`, plus the
workstream-specific verification listed above. Phase 2 additionally requires
the base spec's before/after artifact-diff protocol.

---

## Acceptance criteria (addendum)

1. Exactly one Hydra-dict → `ExperimentConfig` converter exists
   (`create_config_from_dict`); `rg "def dict_to_config"` returns nothing.
2. `run_experiment.py` ≤ ~350 lines; the four evaluation-summary helpers are
   importable from `mci_gru/evaluation/experiment_summary.py` with direct
   unit tests.
3. `scripts/nb_lib.py` exists; `rg "def md\(text" scripts/gen_*` returns
   zero hits; all 16 notebooks regenerate from generators that import the
   library; WS-L PR 1 regeneration was byte-identical.
4. One backtest engine module; `tests/`/`scripts/` contain no second
   implementation of `simulate_trading_strategy`; the divergence audit doc
   exists and every behavioral-drift decision in it maps to a golden-test
   update commit.
5. Golden backtest fixtures + `test_backtest_engine_golden.py` run in CI.
6. Zero `print(` calls in the seven WS-O core modules
   (`rg -c "print\(" mci_gru/pipeline.py mci_gru/data/data_manager.py ...`
   → no matches); training log files contain pipeline/trainer lines.
7. `rg "from mci_gru.features" mci_gru/config.py mci_gru/data/` → no hits;
   `rg "from mci_gru.training" mci_gru/evaluation/` → no hits.
8. `Seed_test (1).ipynb`, `Program explainers/`, `seed_results/`,
   `_uncertain/`, `archive/` absent from the working tree; the
   `archive/pre-hygiene-2026-07` tag exists on origin.
