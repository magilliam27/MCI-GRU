# Run Input Preservation Contract

Status: **owner-approved target and implementation handoff; not implemented
behavior or live preservation evidence.** Decision issue: [#192](https://github.com/magilliam27/MCI-GRU/issues/192).
This is the detailed input contract supporting the human-led
[target architecture](target-architecture.md). Read [implemented architecture](../ARCHITECTURE.md)
for current behavior and [the agent guide](guide.md) for ownership.

## Outcome and scope

A saved run must identify and recover every input it actually consumed. The
same named file fetched again is not proof of the same input. Preserve the
expected package declarations and the observed consumed-input identities.

The selected stock dataset is
`sp500_pit_gics_top10_mcap_monthly_20160104_20260731`: the July 31, 2026
110-universe pull completed through [#141](https://github.com/magilliam27/MCI-GRU/issues/141).
Its complete package contains the following ten outputs; this is an inventory,
not a universal ten-file limit in the manifest format:

| Directory | Suffix after the shared vintage prefix | Purpose |
| --- | --- | --- |
| `constituents/` | `_snapshots.csv` | Monthly selected universe |
| `constituents/` | `_all_metadata_snapshots.csv` | Sector and selection metadata |
| `constituents/` | `_pit_universe.csv` | Point-in-time membership used by the model |
| `constituents/` | `_sp500_membership_intervals.csv` | Source membership history |
| `constituents/` | `_current_members.csv` | Current-member source record |
| `constituents/` | `_changes.csv` | Membership-change source record |
| `constituents/` | `_meta.json` | Universe pull provenance |
| `market/` | `_lseg_20150101_20260731.csv` | Price/volume panel and warmup history |
| `market/` | `_lseg_20150101_20260731_coverage.csv` | Export coverage |
| `market/` | `_lseg_20150101_20260731.meta.json` | Market pull provenance |

Older stock datasets are excluded from preservation/backfill and from the
supported retrieval workflow. Do not revive them to satisfy a broad inventory
test, silently substitute them, or retarget historical experiments. Exclusion
does not delete or relocate existing files. Future stock vintages require an
explicit experiment choice; a moving `latest` selector cannot identify a run.

The owner additionally requires **all actual run inputs** to be kept. Auxiliary
packages therefore include the exact regime/FRED inputs and any consumed VIX,
credit, benchmark/index, external prediction, or other source data. Inventory
enabled dependencies; do not fetch or preserve every possible feature family.
The stock package remains distinct from these auxiliary packages.

For a run that consumes a pre-existing checkpoint, frozen graph, normalization
state, or other derived file, that file is an input and its exact bytes must be
retained. Intermediate features, panels and graphs built entirely within the
run can remain reproducible from retained inputs plus the resolved recipe.
Calling a pre-existing artifact "derived" does not exempt it from preservation.
Implementation starts with the reference training/evaluation path; any separate
entry point not yet covered must report that limitation. This contract does
not authorize changes to `paper_trade/` or its frozen-graph invariant.

Resolved configuration, seeds, source code state and environment identity are
also replay dependencies, coordinated with #144, #143 and #142. A code hash must
resolve to recoverable code; it does not reconstruct an unretained dirty tree.
Credentials are access mechanisms and must not be embedded in manifests
or run artifacts. Input retention does not imply bitwise training determinism.

## Manifest and storage contracts

One extensible, Git-tracked manifest describes each complete data package. Its
stable required core carries a format version, readable package identity and
file records with package-relative path, purpose, SHA-256 and byte size.
Provenance records available producing commands/arguments, acquisition time,
source and acquisition mode. Keep unknown historical facts explicitly unknown;
never invent a pull time or claim today's entitlement describes an old pull.
Preserve the selected pull's original metadata unchanged as package files.

The engineering specification must distinguish the manifest's format version
from a package revision. Use a list of file records and optional descriptive
extensions, rather than ten hard-coded fields. Optional metadata must not
override required identity or validation semantics. Reject unsupported required
semantics; define compatibility behavior for every format version implemented.
Specify deterministic serialization and hash the exact retained manifest bytes,
without placing a self-referential digest inside those bytes. Adding files or
changing the declaration yields a separately identifiable manifest revision;
prior runs keep their exact earlier declaration.

Validate unique, safe relative paths, valid hash/size fields and a complete
declared inventory. Do not permit absolute paths, traversal outside the package
or duplicate destinations. Exact schema keys and helper names are engineering
choices to specify in the implementation PR, within this contract.

**Storage locations are separate from dataset identity.** A location record
links a manifest identity to Google Drive and local copies. External roots,
addresses and machine-specific settings belong there; package-relative paths
belong in the manifest. Moving a copy changes its locator, not the dataset.
Shared location configuration must support machine-local roots and existing
Drive access without embedding credentials. Exact initial locators must be
inspected and verified before use; old filenames, size matches and copy reports
do not establish current preservation completeness.

Google Drive is the off-machine archive/retrieval home. An independent local
preservation copy supports recovery and quick local tests without repeated
downloads. Working copies are separate from preservation originals; deletions
must not automatically mirror into the independent archive. Identical input
packages can be shared across runs by identity; raw files need not be duplicated
inside every run bundle. No automatic retention/deletion schedule is introduced.

## Retrieval, capture and replay

For the exact manifest requested by the run:

1. Check the intended working location and reuse matching files.
2. Fill missing files from the configured local preservation source when
   available, checking the declared hashes.
3. Retrieve remaining missing files from the configured Drive copy using
   available access, including mounted Drive in Colab, then verify them.
4. A checksum mismatch stops setup and reports the expected/observed identity.
   Leave conflicting existing files untouched. Do not hide the discrepancy by
   overwriting it, changing expected hashes or choosing a different dataset.
5. Missing/unavailable files or required access produce an actionable failure
   before training uses incomplete inputs. Recovery never performs a fresh
   vendor pull or automatically creates account access.

Use staging and publish retrieved files for use only after verification.
An unavailable source may lead to the next configured source; a present but
wrong source must produce a mismatch report. A partly staged package must not
be reported as a complete preserved copy. Ensure the loader consumes the
verified files rather than resolving the basename again to another location.
Independent preservation proof must exercise each copy without fallback to the
other, even though routine retrieval uses the accepted priority order.

At the external-data boundary, capture the exact values supplied to the run
before application-owned lagging, filling, resampling or feature calculations.
For an SDK that already returns parsed observations, preserve those observations
losslessly with the relevant schema, types, ordering, missing values, series
identifiers, request parameters and source/acquisition metadata. If raw response
bytes are available and used by an application parser, retain them and identify
that parser. Replay must use the retained representation at the same boundary;
an approximate re-export of final features is not a source snapshot.

Record which source actually supplied data, including cache hits and external
fallbacks. Preserving an unrelated existing cache path is insufficient. Replay
loads the captured snapshot without a live provider fetch. Initial acquisition
is a separate workflow governed by the existing recipe and pull policy; replay
does not initiate it. Existing no-lookahead and regime contracts remain in force.
An exact snapshot is replay evidence, not proof that historical observations
were available at each historical decision date.

For each reference run, enumerate all enabled input dependencies and their
retention state. A missing capture or unverified preservation copy prevents a
claim that the run's complete inputs are preserved, even if training finished.
Record recoverable failure/progress evidence rather than emitting a false
success. The exact operational time at which off-machine replication completes
must be visible; a local-only partial result is not preservation completion.

## What each saved run retains

- Exact manifest snapshots for every consumed data package, each with its
  SHA-256 and readable package identity in the run record.
- Observed identities of files actually consumed, including resolved paths,
  input roles and any source/fallback provenance. The expected declaration does
  not replace this observation.
- Resolved config and seed settings, recoverable source identity/state and
  environment provenance through the existing #144/#142 work.
- References to preserved auxiliary snapshots or initial model/graph state,
  plus input verification/capture outcomes. Access locations can evolve through
  their separate records without changing the run's identities.

Integrate with #191 / PR #201's observed `data_inputs` and legacy `data_file_*`
fields. Do not build a second fingerprint collector or erase evidence of what
was loaded. All required manifest snapshots must travel with a saved result;
reviewing it later must not depend on a current branch path or current locator.

## Execution packet and ownership

These slices describe implementation work; this documentation is not evidence
that they are complete. Claim the linked ticket and exact owned paths before
editing, re-read current assignees/PRs, and use scoped worktrees and draft PRs.

| Slice | Responsibility and likely seams | Dependencies and boundaries |
| --- | --- | --- |
| [#205: manifest/backfill](https://github.com/magilliam27/MCI-GRU/issues/205) | Input-manifest reader/writer/validator, selected manifest and original metadata; new focused tests; reuse hashing ideas from `mci_gru/evaluation/run_bundle.py` without coupling inputs to output layout | #192 documentation; selected stock package only; protected source hashing is read-only |
| [#206: capture/replay](https://github.com/magilliam27/MCI-GRU/issues/206) | Auxiliary snapshot boundary around `mci_gru/data/data_manager.py`, `fred_loader.py`, `pipeline.py`; explicit snapshot replay and enabled-input inventory | #192, #193 and #205; no policy relaxation, recipe change or legacy-data backfill |
| [#207: retrieval/preservation](https://github.com/magilliam27/MCI-GRU/issues/207) | Shared locators/retrieval helpers, working/local/Drive setup, independent copy verification and restoration reports | #192, #205 and #206; coordinate #185 before any notebook-generator edits |
| [#208: run attachment](https://github.com/magilliam27/MCI-GRU/issues/208) | Manifest snapshots and expected/observed linkage in the saved-run path, extending `run_experiment.py` / `mci_gru/evaluation/experiment_summary.py` through narrow helpers | #192, #191, #144, #205 and #206; do not duplicate their metadata ownership |

Every slice has a bounded ticket with explicit dependencies. Test additions
regenerate `docs/TEST_REGISTRY.md`; concurrent edits coordinate this shared file.
The exact implementation boundaries can be refined without changing the owner
policy. Data-quality choices belong to #193; pull immutability/automation policy
belongs to #194. Input snapshots do not qualify G4/T4 or settle calibration
counts, numerical tolerances, comparison budgets, or promotion decisions.

## Required completion evidence

Follow [the testing guide](../TESTING_GUIDE.md), including meaningful synthetic
fixtures and mutation checks for load-bearing guards. Evidence must establish:

- Schema extensibility plus rejection of altered bytes, missing/duplicate
  declarations, unsafe paths and unsupported required semantics.
- The selected manifest matches all ten actual outputs; original provenance
  remains intact and unknowns remain explicit.
- Working-file reuse, local retrieval without Drive access, Drive retrieval
  into an empty destination, actionable missing-access failures, and mismatch
  behavior that leaves conflicting files untouched.
- Replay of each enabled external-input path with provider calls disabled;
  capture before feature transformations and preservation of actual source
  choice, missing values and ordering. Test existing no-lookahead contracts.
- A saved run's manifest snapshots hash correctly and correspond to the
  files the loader actually opened, including auxiliary and initial-state inputs.
- Both required preservation copies independently pass complete-inventory
  checks and restore successfully into empty working locations. Retain manifest
  identity, source/copy identity, verification time, expected/observed hashes,
  file count, outcome and failure details. Counts alone are insufficient.
- Live Colab restoration claims include visible execution and retained Drive
  artifacts per [the Colab runbook](../workflows/COLAB_CHROME_CONTROL_GUIDE.md).
  Generated-notebook tests and local tests are not live Colab proof.

## Decision record

| Owner decision | Durable source |
| --- | --- |
| Git manifests plus required retrieval and preservation | [#192, mechanism](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5565361507) |
| Drive plus independent local copy; quick local testing | [#192, storage](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5565443982) |
| Only the latest selected 110-universe stock pull | [#192, stock scope](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5565580759) |
| One extensible manifest per complete package | [#192, granularity](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5565675589) |
| Storage locations separate from identity | [#192, locators](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5629055392) |
| Working/local/Drive retrieval; stop on mismatch | [#192, retrieval](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5629142214) |
| Exact manifest retained with every saved run | [#192, run snapshots](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5629299880) |
| Preserve every input actually consumed, including auxiliaries | [#192, complete inputs](https://github.com/magilliam27/MCI-GRU/issues/192#issuecomment-5643250220) |

The detailed serialization, capture and verification specifications above are
engineering consequences of these decisions. They are not claims that current
code implements them. The accepted reference-run repeatability policy is tracked
by #142 / PR #204; this input contract complements that work without changing it.
