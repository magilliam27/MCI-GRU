# MCI-GRU Cockpit Runbook

The cockpit refresh writes the repo-local operating picture:

- `docs/agents/workstreams.md`
- `docs/agents/cockpit/YYYY-MM-DD.md`
- `docs/agents/cockpit/auto-decisions.json`

It reads durable human decisions from:

- `docs/agents/cockpit/workstream-decisions.json`

Processed curator comment IDs are stored in:

- `docs/agents/cockpit/override-receipts.json`

Manual local refresh:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20
```

Auto-decisions are enabled by default. Use `--no-auto-decisions` only to
reproduce the legacy feature-off output for debugging or parity verification:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20 --no-auto-decisions
```

GitHub sync is disabled by default. Live sync requires `--github-sync`:

```bash
python scripts/refresh_cockpit.py --date 2026-06-20 --github-sync
```

Before running live sync, confirm GitHub CLI authentication:

```bash
gh auth status
```

Live sync uses a dated branch named `codex/cockpit-refresh-YYYYMMDD`, commits
only cockpit files, pushes the branch, opens or reuses `Cockpit refresh:
YYYY-MM-DD`, creates or reuses the `MCI-GRU Cockpit` issue, and comments there
with the run color, PR link, and decision queue. The producer refuses a primary
checkout, including a clean `main`: it must start in a clean, pre-provisioned
linked worktree already on the dated branch. It verifies the linked-worktree
git-dir/common-dir relationship and never switches the caller checkout. It
fetches and reuses an existing dated branch only when the local and remote OIDs
match exactly; an unpublished dated branch must already equal freshly fetched
`main`. Before mutation, it compares the complete
rename-aware fetched branch diff with the cockpit artifact allowlist. If a PR
already exists, its exact title, `main` base, dated head, open state, base/head
OIDs, and paginated API file list must match the fetched evidence before any
commit or push. It finalizes the packet before one artifact commit, validates
the complete staged index and the resulting full branch diff, and creates or
updates exactly one date-marked Cockpit issue digest instead of duplicating
comments on rerun. Digest reporting distinguishes created, updated, and
unchanged outcomes. The shared producer/curator path allowlist includes
`workstream-decisions.json`, so a normal same-day refresh can safely reuse a
dated branch after an accepted curator override without extra commit churn.

### Disposable producer checkout

Provision the dated branch in a temporary linked worktree before live sync. On
Windows PowerShell, the following outline avoids reset, force, or mutation of
the source checkout. Create the local dated branch at the fetched OID only when
it does not already exist; if it exists, first verify that its full OID matches
`FETCH_HEAD` and that another worktree is not using it.

```powershell
$Source = (Resolve-Path C:\Users\magil\MCI-GRU).Path
$Branch = "codex/cockpit-refresh-20260620"
$RunDate = "2026-06-20"
$TempRoot = Join-Path ([IO.Path]::GetTempPath()) ("mci-gru-cockpit-" + [guid]::NewGuid().ToString("N"))
$Checkout = Join-Path $TempRoot "repo"
New-Item -ItemType Directory -Path $TempRoot | Out-Null

git -C $Source fetch origin main
$Remote = git -C $Source ls-remote --heads origin $Branch
if ($Remote) {
    git -C $Source fetch origin $Branch
}
# Stop here if an existing local $Branch does not equal FETCH_HEAD or is already checked out.
if (-not (git -C $Source branch --list $Branch)) {
    git -C $Source branch $Branch FETCH_HEAD
}
git -C $Source worktree add $Checkout $Branch

Push-Location $Checkout
try {
    & (Join-Path $Source ".venv\Scripts\python.exe") scripts\refresh_cockpit.py --date $RunDate --github-sync
} finally {
    Pop-Location
}
```

Cleanup is a separate, verified step. Remove only the temporary surface created
by this execution; never compute a broad parent path or remove a pre-existing
worktree. Resolve the exact checkout and confirm it stays under the system temp
root before calling `git worktree remove` and `Remove-Item`:

```powershell
$ResolvedCheckout = [IO.Path]::GetFullPath($Checkout)
$ResolvedTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
if (-not $ResolvedCheckout.StartsWith($ResolvedTemp, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean a cockpit checkout outside the system temp root."
}
git -C $Source worktree remove $ResolvedCheckout
Remove-Item -LiteralPath $TempRoot -Recurse -Force
```

## Automated Policy And Curator

The generated policy always chooses a surface disposition and workstream
status from current git and read-only GitHub evidence. Ambiguity lowers
confidence and records alternatives; it does not create a terminal
`needs-user-decision` result. Explicit entries in
`workstream-decisions.json` remain durable overrides and always win over
generated output. Generated association, disposition, status, and review-date
metadata remain the independent policy counterfactual; surface/workstream
overrides apply only in the effective overlay. The autonomous policy never
writes that override file. The packet audits generated choice and confidence
changes separately from field-specific metadata changes, including rule,
evidence, association/canonical surface, alternatives, and review date. Override
added, changed, and cleared events are retained even when the generated target
has disappeared; historical committed registries admit their own previously
declared workstream names for this comparison while current registry validation
stays strict. The cleared state is rendered as no generated decision.

`association_basis` preserves the source association as
`explicit-surface`, `explicit-alias`, `branch-term`, `title-case-fallback`, or
`implied-alias`; linked PR or issue evidence may strengthen the decision without
rewriting that provenance. Only high-confidence independently grounded bases
may propose implied aliases. Title-case fallbacks, implied aliases, and legacy
`linked-*` metadata cannot teach a later generated decision.

The dated cockpit PR is the correction surface. The repository
`.github/workflows/cockpit-overrides.yml` workflow routes new structured PR
comments to `scripts/apply_cockpit_overrides.py`. For the initial rollout, only
a comment whose trusted GitHub API evidence has both the repository owner's
login and `OWNER` author association is authorized. The workflow does
not call the Administration-read collaborator-permission endpoint because that
permission is unavailable to its standard `GITHUB_TOKEN`. Other collaborators,
malformed commands, unsafe PR branches, forks, and closed PRs fail closed and
receive no registry mutation.

Supported commands are exact, single-line instructions:

```text
/cockpit override workstream "LambdaRankIC" status parked reason "Pause until data contract review completes."
/cockpit override surface "codex/example-branch" disposition archive workstream "LambdaRankIC" reason "Superseded by PR #90."
/cockpit clear-override workstream "LambdaRankIC"
/cockpit clear-override surface "codex/example-branch"
```

Leading or trailing whitespace is not normalized; a command must match one of
these forms exactly.

An accepted command changes only the named registry entry, refreshes the
generated register/packet/auto-decisions, commits the exact command and source
comment URL to the same dated cockpit PR branch, and posts an acknowledgement
only after the push succeeds. That acknowledgement repeats the exact accepted
command and links the full pushed commit OID.
`override-receipts.json` makes repeated comment delivery idempotent. A duplicate
with an existing response marker produces no second response, registry change,
or commit. If the original push succeeded but its response post failed, retry
recovers exactly one full commit OID from validated branch history, verifies the
exact commit subject, source comment URL, and command, then posts the same
evidence-rich response. Missing, ambiguous, or mismatched history fails closed.
Clearing a generated-only target is rejected: `clear-override` requires the
named explicit registry entry to exist.

Idempotency markers are evidence only when authored by the repository owner
with `OWNER` association or by `github-actions[bot]`. Curator response markers
must also match the complete expected command/commit response body. Untrusted
lookalike markers are ignored; duplicate or mismatched trusted markers fail
closed. The same author rule protects the date-marked Cockpit issue digest.

The curator requires a clean disposable checkout and validates an open
same-repository PR whose branch, exact title, date, and `main` base match the
cockpit contract. It fetches and binds both validated base/head OIDs, then
creates the dated local branch from `FETCH_HEAD` or reuses it only at the exact
validated head; mismatches fail instead of being reset. The workflow configures
the GitHub Actions bot identity. Before checkout, both GitHub's path evidence
and a local rename-aware diff of the fetched object IDs must contain only
cockpit artifact paths; rename source and destination paths are both checked.
Refresh runs through modules already imported from the trusted startup
checkout, so it never executes a script from the mutable PR head while the
write-scoped token is present. Producer and curator refreshes exclude the dated
automation branch from policy evidence while still comparing curator lifecycle
changes against its current committed `HEAD`; this prevents synthetic branch,
ahead-count, or self-dirty churn. Before commit, each path validates the complete
staged index. If refresh, staging, validation, or commit fails before a commit,
the allowlisted files are restored to their pre-command bytes and unrelated
paths remain untouched; a post-commit push failure fails the disposable job
without resetting history.

The immediate `issue_comment` workflow is paired with nightly reconciliation.
The daily control-plane automation should find the current open dated cockpit
PR and run the curator without `--comment-id` before or after its normal
refresh, so any missed structured comments are processed in stable comment-ID
order. Do not enable that production reconciliation command until this Phase 4
code is merged to the automation's `origin/main` base.

The curator never deletes branches, removes worktrees, closes PRs, closes
issues, or force-pushes. Surface dispositions remain advisory labels; those
actions keep their existing approval gates.

Labels are applied only when they already exist in GitHub. The dated cockpit PR
reconciles the exact existing labels `cockpit-reviewed`, `codex`, and
`codex-automation`; it never creates a missing or near-duplicate label. The sync
reads the PR labels back after mutation and fails closed if an eligible label is
still absent. Its trusted dated Cockpit issue digest records deterministic
`applied`, `already-present`, `skipped-missing`, and
`verified-after-readback` receipt fields. Same-date retries retain labels that
the cockpit previously applied instead of rewriting that history as
`already-present`. This external digest is post-mutation evidence; the pre-sync
repository packet remains deterministic generated output.
Other safe issue labels include `ready-for-agent`, `needs-info`, and
`needs-triage` when present.

Issue closure must go through `close_issue_with_evidence`, which comments with
the closure evidence before closing the issue. Do not close ambiguous research,
experiment, or data-access issues without clear evidence such as a merged PR,
existing code/tests/docs, a linked duplicate, or an explicit user decision.

Before committing cockpit output, inspect:

```bash
git status --short
git diff -- docs/agents/workstreams.md docs/agents/cockpit/
```

Only cockpit files should be staged for a cockpit refresh.

## Daily Branch And PR Integration Review

Every cockpit refresh must include a branch-integration pass before writing the
workstream register.

Read-only evidence to collect:

```bash
git status --short
git status --short --branch
git branch --show-current
git worktree list --porcelain
git branch --all --no-merged origin/main
git rev-list --left-right --count origin/main...HEAD
gh pr list --state all --json number,title,state,isDraft,headRefName,baseRefName,mergedAt,updatedAt,url
gh issue view 38 --json number,title,state,updatedAt,url,comments
```

On Windows/Codex worktrees, per-worktree status checks should use a local
safe-directory override rather than changing global git config:

```bash
git -c safe.directory=<worktree-path> -C <worktree-path> status --porcelain=v1 -b --untracked-files=all
```

For each durable workstream, the cockpit must name the canonical continuation
surface in `docs/agents/workstreams.md`. Do not use generic placeholders such as
"See latest branch, worktree, issue, or handoff evidence" when a branch,
worktree, PR, issue, handoff, or merged commit is known.

The canonical continuation surface should be one of:

- a GitHub issue when implementation work should start from the tracker;
- an open PR when review or merge is the next action;
- a named branch or worktree when local continuation is expected;
- `main` plus a merged PR or commit when the work is already integrated;
- a dated cockpit PR when the row is cockpit-status-only;
- the policy-selected best candidate when two or more plausible continuation
  surfaces exist, with low confidence and the rejected candidates listed as
  alternatives.

With default auto-decisions enabled, competing evidence must produce a concrete
best choice rather than `needs-user-decision`. The legacy
`--no-auto-decisions` diagnostic mode may still emit `needs-user-decision` and
list the competing surfaces in `Blocked On` or `Next Action`.

## Workstream Decision Registry

`workstream-decisions.json` is the canonical, versioned input for reviewed
workstream and git-surface decisions. Generated files such as
`docs/agents/workstreams.md` must not be edited to preserve a decision; the next
refresh overwrites them.

Each `workstreams` entry records the chosen status, canonical continuation
surface, rationale, next action, and review date. Each `surfaces` entry records
which workstreams reviewed a normalized branch name and whether that surface is
`canonical`, `parked`, `archive`, or `stale`.

Resolution rules:

1. A registry decision wins over generated policy and branch-name heuristics
   for every reviewed surface or workstream.
2. Reviewed historical surfaces remain visible as parked, stale, or archive
   candidates, but they do not reopen the workstream decision.
3. A newly matching branch is classified automatically. Competing evidence
   produces a deterministic best choice with lower confidence and listed
   alternatives.
4. Explicit `surfaces[*].workstreams` assignments classify the effective
   register without changing the independently generated association retained
   in `auto-decisions.json`.
5. Missing registry files make the cockpit red through the required-doc check.
   Invalid JSON, unknown fields, unknown workstreams, remote-prefixed surface
   keys, and unsupported format versions fail fast.

Surface keys must use the normalized branch name emitted by the cockpit, for
example `codex/example`, not `origin/codex/example`,
`remotes/origin/codex/example`, or `refs/heads/codex/example`.

When a decision changes, update the existing entry and `last_reviewed`. When a
new branch is reviewed, add it to `surfaces`; do not broaden branch-name terms
just to suppress a decision. Branch deletion, worktree removal, PR closure, and
force-pushes remain separately approval-gated even when a surface is marked
`archive` or `stale`.

## Stale Cockpit PR Handling

Dated cockpit PRs are audit records, but only the newest relevant cockpit PR
should remain the active review surface for the day.

During daily review:

1. List open `Cockpit refresh: YYYY-MM-DD` PRs.
2. Compare each open dated cockpit PR with newer merged cockpit PRs.
3. If a newer cockpit PR has merged and covers the same generated files, mark
   the older open PR as `stale` or `superseded` in the cockpit packet.
4. Do not close the superseded PR automatically unless the user has approved PR
   cleanup for that run.
5. Do not delete the dated branch automatically.
6. Add the proposed action to the GitHub Cockpit issue comment when live sync is
   enabled.

A superseded cockpit PR should be described like this:

```markdown
- PR #41 (`codex/cockpit-refresh-20260622`) is still open, but PR #42
  (`codex/cockpit-refresh-20260623`) has merged newer cockpit output. Proposed
  action: close PR #41 as superseded after user approval; keep the branch unless
  separately approved for deletion.
```

## Integration Branch Selection

The cockpit must keep branch integration separate from cockpit refreshes.

Rules:

- Cockpit branches may update only cockpit/status files and small cockpit policy
  docs under `docs/agents/`.
- Active research, experiment, Colab, paper-trade, or pipeline work must name
  its own canonical branch/worktree before continuation.
- If a workstream has multiple plausible branches, use the deterministic policy
  choice, lower its confidence, and list the rejected branch choices as
  alternatives. Only the legacy `--no-auto-decisions` diagnostic mode may emit
  `needs-user-decision` for that ambiguity.
- If a branch has already been merged to `main`, the canonical continuation
  surface becomes `main` plus the merged PR/commit, unless follow-up work is
  explicitly opened elsewhere.
- If a branch/worktree is detached, local-only, or stale, say so directly in the
  `Branch / Worktree` column.

Branch integration decisions that require user approval:

- closing a superseded PR;
- deleting or pruning a branch/worktree;
- force-updating any branch;
- choosing between competing non-cockpit implementation branches;
- moving code from a local-only worktree into an integration branch.

## GitHub Hygiene

Each cockpit run should include a short GitHub hygiene section when any branch
or PR state needs attention.

Check for:

- the latest merged PR to `main`;
- open cockpit PRs that appear superseded by newer merged cockpit output;
- draft PRs older than 14 days, especially when dirty against `main`;
- pushed branches with no open PR, grouped by workstream when possible;
- active or `ready-for-agent` issues that lack a named branch, open PR, or
  explicit "not started" state.

## Git Tree Impact

Each daily packet must include a `Git Tree Impact` section that summarizes:

- the refresh/control-plane checkout path, branch, dirty state, and
  `origin/main...HEAD` divergence;
- the canonical active checkout (Git's primary worktree) path, branch, dirty
  state, and separately parsed `origin/main` divergence;
- snapshot timing, especially whether the packet was collected before GitHub
  sync commits/pushes;
- count and names of branches not merged into `origin/main`, labelled as
  local, remote-only, or local+remote;
- total, detached, and dirty worktrees;
- dirty or detached worktree paths that require review.

The actionable packet queues keep canonical/active and `ready-for-agent`
continuations together, list `parked` work separately, and reserve the
archive/cleanup queue for `stale` and `archive` candidates. `done`, blocked,
local-only, and decision rows are not mixed into those queues: blocked,
local-only, and decision rows keep their dedicated packet sections, while
`done` rows remain in the generated workstream register and are omitted from
the actionable packet queues.

Detached surface IDs combine the seven-character HEAD prefix with a stable
digest of the normalized worktree path. Two detached worktrees at the same
commit therefore remain separate surfaces, retain separate path evidence, and
receive separate generated decisions.

An ordinary attached worktree keeps its branch name as the surface ID. If
multiple attached worktrees report the same branch, the deterministically first
normalized path keeps that ordinary ID and each additional path receives
`worktree:<branch>@<path-digest>`. The colon places collision IDs outside valid
Git branch syntax, so a literal branch cannot shadow them. Duplicate normalized
paths or duplicate synthetic IDs fail closed instead of being deduplicated.

Primary repo git commands should run with `safe.directory` set for the current
checkout. Secondary worktree status probes should set `safe.directory` for the
worktree path being inspected.

The workstream register should be live-topology-first. Seeded workstream names
may enrich matching branches/worktrees, but unmatched local branches, worktree
branches, and remote-only unmerged branches should get explicit `Git surface:`
rows so they cannot disappear from the solo-dev continuation map.

Use `green` only when no branch/worktree attention items are found. Use
`yellow` when there is normal solo-dev branch pressure, such as unmerged
branches, detached worktrees, or non-current dirty worktrees. Use `red` when
the current checkout itself is dirty before cockpit generation or required
docs are missing.

If live sync goes sideways, inspect and recover with:

```bash
git status --short
git branch --show-current
gh pr list --head codex/cockpit-refresh-YYYYMMDD
gh issue list --search "MCI-GRU Cockpit in:title"
```
