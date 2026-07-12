# Dynamic Workstream Seed Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase-by-phase. Each phase has its own Acceptance Criteria and Verification, and phases are meant to land in the Rollout Order below.

**Tracking issue:** [GitHub issue #76](https://github.com/magilliam27/MCI-GRU/issues/76) — "Cockpit: derive workstream seeds from registry and recent git activity instead of hardcoded `INITIAL_WORKSTREAMS`".

**Verification environment:** Windows / PowerShell, repo venv at `.venv`.

---

## Overview

### Problem

Every workstream row in the cockpit register (`docs/agents/workstreams.md`) ultimately originates from a hardcoded Python list, `INITIAL_WORKSTREAMS`, in `cockpit/runner.py` (lines 64-120). That list holds nine `WorkstreamSeed` dataclass instances (definition at `cockpit/runner.py:48-54`), each carrying a `name`, default `status`, `next_action`, optional `tracker`, and a tuple of substring `branch_terms`. Because the seed list is code, a human must edit and ship Python to add, rename, or retire a workstream. As real work shifts week to week, the seed list drifts away from what is actually in flight, and the register slowly stops reflecting reality.

There is also a **latent crash**. `run_local_cockpit_refresh` derives the set of "known" workstream names directly from the seeds:

```133:135:cockpit/runner.py
    registry = load_decision_registry(
        repo_root,
        known_workstreams={seed.name for seed in INITIAL_WORKSTREAMS},
    )
```

`load_decision_registry` (`cockpit/decisions.py:54-77`) then hard-rejects any workstream name in `docs/agents/cockpit/workstream-decisions.json` that is not in that known set — both for the `workstreams` section (`cockpit/decisions.py:87-88`) and for workstream references inside the `surfaces` section (`cockpit/decisions.py:131-135`). If a recorded decision ever names a workstream that is not in the hardcoded seeds, the whole daily refresh raises `ValueError` on the user's own recorded decisions. Today this is only avoided because the seed names happen to be a superset of the registry's workstream keys; nothing structurally enforces that.

### Guiding principle: "sources propose, registry disposes"

Rather than have one hardcoded list own the workstream identities, we introduce **workstream sources** that each *propose* candidate `WorkstreamSeed`s from evidence they can see, and a **merge step** that combines them deterministically. The recorded decision registry (`workstream-decisions.json`) remains authoritative for *status, canonical surface, next action, and last-reviewed date* — it *disposes*. A source can suggest that a workstream exists and is `ACTIVE`; the registry can override that to `parked`, `done`, `archive`, and so on through the existing `_resolve_workstream` machinery (`cockpit/runner.py:266-317`). Sources never get the last word on disposition, and the registry never has to know how a workstream was discovered.

This inverts the current dependency: instead of the registry being validated against a static code list, the "known" set becomes the union of what the sources proposed and what the registry itself already records, so recorded decisions can never crash the run.

---

## Current State (verified file:line references)

- **Seed type:** `WorkstreamSeed` frozen dataclass — `cockpit/runner.py:48-54` (`name`, `status`, `next_action`, `tracker=""`, `branch_terms=()`).
- **Hardcoded seed list:** `INITIAL_WORKSTREAMS` — `cockpit/runner.py:64-120` (nine entries; the ninth is `"Git and worktree hygiene"` at `cockpit/runner.py:114-119`).
- **Known-set derivation + crash surface:** `cockpit/runner.py:132-135` feeds `load_decision_registry`.
- **Resolution loop:** `_resolve_workstreams` — `cockpit/runner.py:207-263`. It iterates `INITIAL_WORKSTREAMS` at `cockpit/runner.py:218`, special-cases the hygiene seed by literal name at `cockpit/runner.py:219-221`, computes `live_topology` at `cockpit/runner.py:214`, and suppresses unmatched seeds when topology is live via the `elif not live_topology:` branch at `cockpit/runner.py:236`.
- **Per-seed resolution + escalation:** `_resolve_workstream` — `cockpit/runner.py:266-317`. Competing-surface escalation to `NEEDS_USER_DECISION` (owner becomes `User`) is at `cockpit/runner.py:300-304`; owner assignment at `cockpit/runner.py:315`.
- **Branch-term matching:** `_workstream_surfaces` — `cockpit/runner.py:368-379` (heuristic substring match plus registry surface assignment).
- **Hygiene row builder:** `_git_hygiene_workstream` — `cockpit/runner.py:382-415`.
- **Reserved `"Git surface: "` prefix:** emitted by `_topology_surface_workstream` at `cockpit/runner.py:330` and `cockpit/runner.py:345`; consumed by `_decision_workstreams` at `cockpit/runner.py:628` and `_classification_surface_count` at `cockpit/runner.py:652`. Run color depends on `_classification_surface_count` via `_run_color` (`cockpit/runner.py:562-567`).
- **Registry loader + contract:** `cockpit/decisions.py`. `DECISION_REGISTRY_PATH` at line 15, `FORMAT_VERSION = 1` at line 16, top-level key allow-list `{"format_version", "workstreams", "surfaces"}` enforced by `_keys(...)` at `cockpit/decisions.py:69`, version check at `cockpit/decisions.py:70-72`, unknown-workstream rejection at `cockpit/decisions.py:87-88` and `cockpit/decisions.py:131-135`, and the rule that a persisted status may not be `needs-user-decision` at `cockpit/decisions.py:96-100`.
- **Evidence available to sources:** `LocalEvidence` — `cockpit/evidence.py:15-24`. Fields: `repo_root`, `required_docs`, `recent_handoffs`, `dirty_paths`, `branches`, `worktrees`, `recent_commits`, `git_topology`. Branch names come from `git branch --format=%(refname:short)` (`cockpit/evidence.py:42`); handoff filenames come from `_recent_handoffs`, which globs `docs/handoffs/*.md` newest-first, capped at 10 (`cockpit/evidence.py:69-74`). **There is no per-branch committer-date field today** (see Phase 3 Design and Resolved Decision 4).
- **Register schema (do not change):** `REGISTER_COLUMNS` — `cockpit/render.py:11-23`; `render_workstream_register` — `cockpit/render.py:26-38`.
- **Registry keys today (`docs/agents/cockpit/workstream-decisions.json`):** the `workstreams` section has 7 keys — `LambdaRankIC`, `Portfolio-IC`, `Issue #8 volatility targeting`, `Colab operations`, `Regime CSV contract`, `Daily bug scans`, `Docs and research evidence`. `"LSEG access"` and `"Git and worktree hygiene"` are in the hardcoded seeds but **not** registry workstream keys. Several `surfaces` entries reference `"Git and worktree hygiene"`, which is why that name must stay in the known set.
- **Decisions contract tests:** `tests/test_cockpit_decisions.py`. `known_workstreams={"LambdaRankIC"}` is used at lines 49, 104, 108; unknown-name rejection is pinned at lines 66-81; **`format_version: 2` is currently asserted to be *rejected*** at lines 62-64 (this is a conflict for Phase 3 — see that phase and Discrepancies).

---

## Non-Goals

- **No GitHub sync changes.** `run_github_cockpit_refresh` and `cockpit/github.py` behavior stays as-is.
- **No Tier 2 agent-session mining.** Parsing Codex `~/.codex/session_index.jsonl` / rollout `session_meta.cwd` or Cursor agent-transcript JSONL is explicitly deferred: formats are undocumented, files are heavy, and git signals are sufficient to reconstruct the current seed list.
- **No render/schema changes.** No new register columns, no change to `REGISTER_COLUMNS`, run-color rules, or the register table layout.
- **No branch/worktree deletion or cleanup automation.** Sources only read git state; they never mutate it.
- **No new persisted `needs-user-decision`.** Sources never emit `NEEDS_USER_DECISION`; escalation stays a runtime-only outcome of competing surfaces.

---

## Phase 1 — Registry self-declaration (bug-fix tier, land first)

### Scope

Guarantee that recorded decisions can never crash the daily refresh, and let the registry declare workstreams that exist nowhere in code. This is a standalone, behavior-preserving-plus-safety change that does not yet touch the source protocol or git mining.

### Files

- `cockpit/runner.py` — add `RegistryWorkstreamSource`; change how `known_workstreams` is derived.
- `cockpit/decisions.py` — add a lightweight helper to read the registry's `workstreams` keys without full validation (needed to break the chicken-and-egg between "known set" and "load registry"), *or* expose the raw workstream keys during load. See Design.
- `tests/test_cockpit_runner.py` — add a regression test proving a registry-only workstream name no longer crashes.

### Design

The known set must include the registry's own workstream keys, but `load_decision_registry` needs a known set to validate against — a chicken-and-egg. Break it by reading the registry file's `workstreams` object keys *before* full validation:

1. Add `RegistryWorkstreamSource`. Its `provide(...)` reads `docs/agents/cockpit/workstream-decisions.json` and emits one `WorkstreamSeed` per key in the `workstreams` section, using a neutral default (`status=WorkstreamStatus.ACTIVE`, a generic `next_action`, empty `branch_terms`). The registry's recorded status/next-action will override these in `_resolve_workstream`, so the seed defaults are only placeholders for the "declared-but-not-yet-resolved" case.
2. Add a helper — e.g. `read_registry_workstream_names(repo_root) -> set[str]` in `cockpit/decisions.py` — that JSON-parses the file and returns `set(root["workstreams"])` defensively (empty set if the file is missing or the section is absent/ill-typed). This does **not** replace validation; it only peeks at keys so we can build the known set. `load_decision_registry` continues to fully validate afterward.
3. In `run_local_cockpit_refresh`, compute the known set as the union of the hardcoded/derived seed names and the registry's declared workstream keys:
   `known = {seed.name for seed in <seeds>} | read_registry_workstream_names(repo_root)`.
   Because `RegistryWorkstreamSource` already emits a seed per registry key, this union is belt-and-suspenders; keep it explicit for clarity and to stay correct even if the source is later filtered.
4. **Keep genuinely-unknown-name rejection.** Do not weaken `load_decision_registry`. A `surfaces` entry (or `workstreams` entry) that references a name present neither in any source's output nor in the registry's own `workstreams` keys must still raise `ValueError`. The contract test in `tests/test_cockpit_decisions.py:66-81` (which passes `known_workstreams={"LambdaRankIC"}`) must still pass unchanged.

> Note: in Phase 1, `_resolve_workstreams` may still iterate `INITIAL_WORKSTREAMS`; Phase 1 only needs the known-set union and the registry source to exist and be merged into the seed list. Full parameterization of `_resolve_workstreams` is Phase 2. A minimal Phase 1 can prepend `RegistryWorkstreamSource` output to the seeds it iterates, deduped by name with hardcoded seeds winning, so behavior is unchanged for existing names while new registry-only names become known.

### Acceptance Criteria

- A `workstream-decisions.json` containing a `workstreams` key that is **not** in `INITIAL_WORKSTREAMS` loads and refreshes without raising.
- A `surfaces` entry referencing that same registry-declared workstream also loads without raising.
- A `surfaces` or `workstreams` reference to a name that exists in neither the seeds nor the registry's `workstreams` keys still raises `ValueError` (unknown-name rejection preserved).
- All existing `tests/test_cockpit_decisions.py` tests pass unchanged.

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cockpit_decisions.py tests/test_cockpit_runner.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check cockpit scripts/refresh_cockpit.py
```

---

## Phase 2 — Source protocol and merge

### Scope

Introduce the `WorkstreamSource` protocol, a deterministic merge, and parameterize `_resolve_workstreams` to consume merged seeds instead of the module constant.

### Files

- `cockpit/runner.py` — define `WorkstreamSource` protocol, `merge_workstream_sources(...)`, parameterize `_resolve_workstreams`, keep the hygiene seed static.
- `tests/test_cockpit_runner.py` — add merge dedupe/determinism tests.

### Design

- **Protocol:**
  ```python
  class WorkstreamSource(Protocol):
      def provide(self, evidence: LocalEvidence, run_date: date) -> list[WorkstreamSeed]: ...
  ```
- **Merge:** `merge_workstream_sources(sources, evidence, run_date) -> list[WorkstreamSeed]`:
  - Call each source's `provide(...)` in list order.
  - **Dedupe by `name`; earlier source wins.** Order the registry source *before* the git source so registry-declared identities win over git-derived ones.
  - **Stable-sort the merged output before returning.** The register is committed to git, so a nondeterministic row order produces daily diff noise. Sort by a stable key (e.g. `name`), except keep the static `"Git and worktree hygiene"` seed handled where it is today (it is appended last inside `_resolve_workstreams`, `cockpit/runner.py:261-262`, and must remain last).
- **Parameterize `_resolve_workstreams`:** change the loop at `cockpit/runner.py:218` to iterate a `seeds: list[WorkstreamSeed]` argument (the merged output) rather than the module-level `INITIAL_WORKSTREAMS`. Keep the hygiene special-case (`cockpit/runner.py:219-221`) by literal name, and keep the `live_topology` suppression (`cockpit/runner.py:236`) unchanged so derived seeds with matching `branch_terms` survive while placeholder-only seeds are hidden when topology is live.
- **Static hygiene seed:** `"Git and worktree hygiene"` stays hardcoded and is special-cased by literal name; it is not produced by any source. Tests require its presence.
- **Reserve the `"Git surface: "` name prefix.** No source may emit a `WorkstreamSeed` whose name starts with `"Git surface: "`, because run-color (`_classification_surface_count`, `cockpit/runner.py:648-654`) and the decision queue (`_decision_workstreams`, `cockpit/runner.py:623-629`) key off that prefix to identify unclassified topology surfaces. Enforce this in `merge_workstream_sources` by **dropping** any such seed (a source emitting a reserved prefix is a bug, but the daily automated refresh must not crash over it); the drop is covered by a merge test in Phase 4.

### Acceptance Criteria

- Two sources emitting the same `name` produce exactly one merged seed; the earlier (registry) source's fields win.
- Merged output order is deterministic across repeated runs with identical inputs.
- Any seed a source emits with a `"Git surface: "` name prefix is excluded from the merged output (or rejected).
- `_resolve_workstreams` produces identical rows to the pre-change behavior when fed the equivalent seed list (validated by the register-parity check in Phase 4).

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cockpit_runner.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check cockpit scripts/refresh_cockpit.py
```

---

## Phase 3 — GitActivitySource

### Scope

Add the git-derived source: topic tokens from recent branch names and handoff filenames, filtered by stopwords and mapped through an alias table, emitted as `ACTIVE` seeds.

### Files

- `cockpit/runner.py` — add `GitActivitySource` and token-extraction helpers.
- `cockpit/decisions.py` — bump `FORMAT_VERSION`, add the alias section to the contract and its parser, extend the top-level key allow-list.
- `docs/agents/cockpit/workstream-decisions.json` — bump `format_version` and add the alias section.
- `tests/test_cockpit_runner.py` and `tests/test_cockpit_decisions.py` — token extraction, stopword filtering, and alias-section parsing tests (see Phase 4 for the full test list).

### Design

- **Signals:**
  - **Branch names** following the `codex/<topic>-<date>` convention. Extract topic tokens by stripping the `codex/` (or `cursor/`) prefix and any trailing date (`YYYYMMDD` or `YYYY-MM-DD`) or short-hash segment, then splitting the remainder on hyphens. Branch names are available from `evidence.branches` (`cockpit/evidence.py:42`) and `evidence.git_topology.branches`.
  - **Handoff filenames** of the form `docs/handoffs/YYYY-MM-DD-<topic-slug>.md`, available from `evidence.recent_handoffs` (`cockpit/evidence.py:69-74`). Extract the date prefix and the topic slug tokens.
- **Lookback: 14 days by committer date (Resolved Decision 4 — committer date).** `LocalEvidence` exposes no per-branch committer date today (`cockpit/evidence.py:15-24`), so extend it: add a new field (e.g. `recent_branches: list[tuple[str, date]]`) populated in `collect_local_evidence` (`cockpit/evidence.py:38-66`) by `git for-each-ref --sort=-committerdate --format="%(committerdate:short) %(refname:short)" refs/heads`, and filter to the last 14 days by that date. Handoffs reuse their `YYYY-MM-DD` filename-prefix date (there is no commit object per handoff).
- **Stopword layer (Resolved Decision 3 — code constant).** Keep the stopword set as a module constant in `cockpit/runner.py`, filtering process meta-branch tokens so they never become fake workstreams: `cockpit-refresh`, `salvage`, `pr-repair`, `worktree-snapshot`, `registry-closeout`, `decision-closeout`, `ci-repair`. A branch whose tokens are entirely stopwords contributes nothing. The stopword set is **not** part of the JSON contract.
- **Alias map (token → canonical workstream name) (Resolved Decision 1 — in-file):** e.g. `lambdarank` / `top10` → `LambdaRankIC`. Stored as a **new optional section** in `workstream-decisions.json`, so curation lives in the JSON the user already edits, not in code. Concretely in `cockpit/decisions.py`:
  - Bump `FORMAT_VERSION` from `1` to `2` (`cockpit/decisions.py:16`).
  - Extend the top-level key allow-list at `cockpit/decisions.py:69` to permit the new `"workstream_aliases"` section (and nothing else — stopwords stay in code).
  - Add a parser that validates the alias section as a mapping of token → non-empty canonical name string, rejecting malformed shapes (non-string values, empty strings, non-object).
  - **Migrate the JSON file to `format_version: 2` in this same PR** and update the version check at `cockpit/decisions.py:70-72` to accept version 2. The `format_version`-rejection assertion at `tests/test_cockpit_decisions.py:62-64` must move to an unsupported version (e.g. `3`); see Discrepancies. (An empty/absent `workstream_aliases` section is valid, so migration is a one-line `format_version` change plus an optional alias block.)
- **Derived seeds:** for each surviving topic (after stopwords + alias resolution):
  - `status` defaults to `WorkstreamStatus.ACTIVE` — never `NEEDS_USER_DECISION`. Emitting `NEEDS_USER_DECISION` would spam the decision queue and force `owner="User"` (`cockpit/runner.py:315`); escalation already happens naturally in `_resolve_workstream` when competing surfaces exist (`cockpit/runner.py:300-304`).
  - `branch_terms` set to the extracted tokens so `_workstream_surfaces` (`cockpit/runner.py:368-379`) matches the live surface and the `live_topology` suppression (`cockpit/runner.py:236`) does not hide the row.
  - `name` (Resolved Decision 5 — title-case joined tokens): when a token matches the alias map, use the alias-mapped canonical name; otherwise the `name` is the title-cased, space-joined surviving token set (e.g. `paper-trade-frozen-graph` → `Paper Trade Frozen Graph`). This lets new topics auto-appear; the alias map only normalizes known synonyms onto canonical names.

### Acceptance Criteria

- Given real branch names (e.g. `codex/top10-lambdarank-screen-20260625`, `codex/portfolio-ic-hybrid-testing`), token extraction yields the expected token sets with the trailing date/hash removed.
- Stopword-only branches (e.g. `codex/cockpit-refresh-20260710`) produce no seed.
- Alias tokens resolve to the canonical name (e.g. `lambdarank`/`top10` → `LambdaRankIC`).
- `workstream-decisions.json` at the new `format_version` loads; a malformed alias section raises `ValueError`; version-1 input still behaves per the chosen compatibility rule.
- Derived seeds are `ACTIVE` and carry their tokens as `branch_terms`.

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cockpit_runner.py tests/test_cockpit_decisions.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check cockpit scripts/refresh_cockpit.py
```

---

## Phase 4 — Test updates and parity check

### Scope

Update the tests that pin seed behavior by literal name, add new tests for the new machinery, and gate on register row parity.

### Files

- `tests/test_cockpit_runner.py` — update the five pinned tests; add new tests.
- `tests/test_cockpit_decisions.py` — update the `format_version` rejection assertion; add alias-section tests.

### Design

**Update the five pinned tests in `tests/test_cockpit_runner.py`** (verified names and current line anchors):

1. `test_run_local_cockpit_refresh_writes_register_and_packet` — register/packet writer test (`tests/test_cockpit_runner.py:207`). Currently asserts `"LambdaRankIC" not in register` and the `"Git surface: ..."` row present; the assertions must be re-derived from the new source-driven seed set for its fake topology.
2. `test_run_local_cockpit_refresh_surfaces_git_topology_without_placeholders` — topology-without-placeholders test (`tests/test_cockpit_runner.py:250`).
3. `test_decision_registry_keeps_reviewed_surfaces_resolved` — decision-registry resolution test (`tests/test_cockpit_runner.py:343`).
4. `test_decision_registry_reopens_only_for_new_unreviewed_surface` — decision-registry reopen test (`tests/test_cockpit_runner.py:419`).
5. `test_run_local_cockpit_refresh_suppresses_seeds_for_main_divergence` — main-divergence suppression test (`tests/test_cockpit_runner.py:612`).

For each, the fake `run_command` returns branch/topology sets that previously were matched (or not) against `INITIAL_WORKSTREAMS`; with source-derived seeds, decide the expected rows from the merged sources for that fixture and update the string assertions accordingly. Where a test relies on a registry-declared workstream, ensure the fixture registry (`_write_decision_registry`, `tests/test_cockpit_runner.py:907-923`, or `_fake_topology_runner`, `tests/test_cockpit_runner.py:929-950`) still supplies it.

**Add new tests:**

- **Registry source emission:** `RegistryWorkstreamSource.provide(...)` emits one seed per `workstreams` key.
- **Merge dedupe/determinism:** duplicate names collapse with earlier-source precedence; repeated runs produce identical order; a `"Git surface: "`-prefixed seed is excluded/rejected.
- **Token extraction:** real branch-name examples strip the trailing date/hash and split on hyphens correctly.
- **Stopword filtering:** stopword-only branches yield no seed.
- **`format_version` alias parsing:** the new version parses, including rejection of a malformed alias section (non-object, empty/non-string canonical value).

**Register-parity verification (acceptance gate):** run `scripts/refresh_cockpit.py` locally on the same date before and after the change and diff the generated `docs/agents/workstreams.md`. Row parity — or explicitly explained differences — is the gate. A known, expected difference: `"LSEG access"` is a hardcoded seed but not a registry `workstreams` key and (absent recent LSEG git activity) will drop from the register. That is an intended consequence of removing the stale hardcoded list and must be recorded as an explained difference, not treated as a regression.

### Acceptance Criteria

- All five updated runner tests pass with assertions that reflect source-driven seeds.
- New unit tests for registry emission, merge, token extraction, stopwords, and alias parsing pass.
- `tests/test_cockpit_decisions.py` passes with the updated `format_version` expectation.
- The before/after register diff is either identical or every differing row is explained (e.g. `LSEG access` drop).

### Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cockpit_runner.py tests/test_cockpit_decisions.py tests/test_cockpit_render.py tests/test_cockpit_cli.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check cockpit scripts/refresh_cockpit.py
.\.venv\Scripts\python.exe scripts\refresh_cockpit.py --date 2026-07-11
```

---

## Rollout Order

1. **Phase 1 — Registry self-declaration.** Standalone bug fix; removes the latent crash. **Ships as its own PR** (Resolved Decision 2).
2. **Phase 2 — Source protocol and merge.** Refactor with behavior parity; no new signals yet.
3. **Phase 3 — GitActivitySource.** New git-derived seeds and the `format_version` bump + alias section.
4. **Phase 4 — Test updates and parity check.** Land alongside Phases 2-3 (tests must move with the behavior); final parity gate before merge.

---

## Combined Verification

Run the full focused suite, lint, and a local dry-run refresh (Windows, repo venv):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cockpit_runner.py tests/test_cockpit_decisions.py tests/test_cockpit_render.py tests/test_cockpit_cli.py -v --basetemp .tmp_pytest\pytest
.\.venv\Scripts\ruff.exe check cockpit scripts/refresh_cockpit.py
.\.venv\Scripts\python.exe scripts\refresh_cockpit.py --date 2026-07-11
```

Register-parity gate: capture `docs/agents/workstreams.md` before the change, run the refresh after, and diff. All differing rows must be identical or explicitly explained.

---

## Resolved Decisions

Resolved 2026-07-11.

1. **Alias section location — in-file.** The token→canonical-name alias map lives in `workstream-decisions.json` as a `format_version` 2 section. Rationale: keep a single curation surface in the file the user already edits; the version bump is cheap. (Affects Phase 3.)
2. **Phase 1 as its own PR — yes.** Ship the registry self-declaration crash fix separately from Phases 2-4: fast, low-risk, independently reviewable, and it de-risks the larger refactor. (Affects Rollout Order.)
3. **Stopword-list ownership — code constant.** The stopword set stays a module constant in `cockpit/runner.py`. Rationale: stopwords are mechanical process-noise filtering tied to the extraction logic, not user-facing curation; keeping them in code avoids expanding the validated JSON contract. Only the alias map goes into the JSON. (Affects Phase 3.)
4. **Lookback — committer date.** Extend `LocalEvidence` with a per-branch committer-date field populated by `git for-each-ref --sort=-committerdate --format="%(committerdate:short) %(refname:short)" refs/heads`, and filter to the last 14 days by that date. Rationale: branch names are not uniformly dated (`codex/Top10nLambda`, `codex/main_2`, `origin/lC_loss` have no date suffix), so a name-date proxy would silently drop them. Handoffs keep their filename date (no commit object per handoff). (Affects Phase 3 and `cockpit/evidence.py`.)
5. **Unaliased-token naming — title-case joined tokens.** Tokens that survive stopword filtering but match no alias become a workstream whose `name` is the title-cased, space-joined token set, so genuinely new topics auto-appear without a prior alias entry. Noise is bounded by the stopword layer, the 14-day committer-date window, and live-topology suppression. The alias map only normalizes known synonyms onto canonical names. A frequency gate (require a topic on 2+ branches before it surfaces) is noted as a possible future refinement, out of scope for this plan. (Affects Phase 3.)

---

## Discrepancies Between the Design Brief and the Code (resolved)

Each place where the incoming design and the code did not line up, with its locked resolution:

- **`format_version: 2` is currently a *rejection* test — RESOLVED.** `tests/test_cockpit_decisions.py:62-64` asserts that `format_version: 2` raises "Unsupported cockpit decision registry format_version: 2". Phase 3 makes version 2 valid. **Action (Phase 4):** move the rejection assertion to an unsupported version — assert `format_version: 3` raises — and add a positive test that `format_version: 2` with a valid (or absent) `workstream_aliases` section loads. This is an additional required test edit beyond the five `tests/test_cockpit_runner.py` tests, and it is now noted on issue #76.
- **No committer-date field in evidence — RESOLVED (Decision 4).** Add a per-branch committer-date field to `LocalEvidence` (`cockpit/evidence.py:15-24`) populated by `git for-each-ref --sort=-committerdate --format="%(committerdate:short) %(refname:short)" refs/heads` in `collect_local_evidence` (`cockpit/evidence.py:38-66`); filter branch signals to the last 14 days by that date. Handoffs keep their filename date.
- **Top-level key allow-list is strict — RESOLVED.** `_keys(root, {"format_version", "workstreams", "surfaces"}, ...)` at `cockpit/decisions.py:69` rejects unknown sections. **Action (Phase 3):** extend the allow-list to `{"format_version", "workstreams", "surfaces", "workstream_aliases"}` — only the alias section is added, because stopwords stay in code (Decision 3).
- **Chicken-and-egg in the known-set union — RESOLVED.** The known set must include registry workstream keys, but loading the registry needs the known set. **Action (Phase 1):** add `read_registry_workstream_names(repo_root) -> set[str]` in `cockpit/decisions.py` that JSON-parses the file and returns `set(root["workstreams"])` defensively (empty set on missing file/section/ill-typed data) **without** validating, purely to seed the known set; `load_decision_registry` still fully validates afterward.
- **`LSEG access` will drop — ACCEPTED, confirmed to drop (2026-07-11).** `"LSEG access"` is a hardcoded seed but neither a registry `workstreams` key nor present in recent git activity, so it disappears once the hardcoded list is removed. This is the intended consequence of "sources propose, registry disposes", and the user has confirmed the drop is fine — **do not** add an `LSEG access` entry to `workstream-decisions.json`. **Action (Phase 4):** record it as the known, expected row in the register-parity diff (an explained difference, not a regression). If LSEG work resumes later, it will re-surface automatically via `GitActivitySource` from LSEG-tagged branch activity, or can be pinned then with a registry entry.
