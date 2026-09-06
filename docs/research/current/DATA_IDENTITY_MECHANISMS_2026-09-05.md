# Dataset identity mechanisms compared (2026-09-05)

Research report for GitHub issue #190. Every vendor fact carries a source tag
[Sn] resolved in the Sources section, each read on 2026-09-05. Facts observed
on this machine rather than in vendor documentation are tagged [Ln] and listed
under Local observations. Where a fact could not be established from a
primary source the text says so. No ruling is made here; the ruling belongs to
the human grilling ticket.

## Question

How do the four candidate identity mechanisms (A hardened sidecars, B DVC,
C Git LFS, D a home-grown content-addressed store) fare against this
repository's actual constraints: corpus shape, where the data lives, hosting
quotas, DVC specifics, manifest portability to Colab, blast radius, and prior
art already in the repository?

## Constraints

Verified by reading the worktree at origin/main @ 125abda and the reference
checkout `C:\Users\magil\MCI-GRU` (read-only listing only).

Corpus shape

- `data/` in the reference checkout holds 70 files totalling 1,013,195,582
  bytes (966.3 MiB; the ticket's "967 MB" is this figure in MiB). The largest
  file is `data/raw/market/sp500_yf_download.csv` at 147,668,406 bytes. No
  data CSV has ever been committed (`git log --all -- 'data/raw/market/*.csv'`
  is empty).
- The 2026-07-31 vintage listed in the external MANIFEST.txt totals
  46,502,119 bytes across 10 files (sum of its `bytes` column), which is the
  size of one whole-vintage increment for the 110-name universe family.
- `.gitignore` line 1 is `*.csv`; line 2 re-admits only
  `tests/fixtures/backtest_golden/**/*.csv`.

Tracked sidecars

- `git ls-files data` returns 10 JSON sidecars, not eight: seven `*_meta.json`
  (five under `data/raw/constituents`, two `lseg_regime_export_*` under
  `data/raw/market`) and three `*.meta.json` under `data/raw/market`. None has
  a key containing sha, md5, hash or checksum.
- Two carry absolute paths in their `outputs` block, all pointing at
  `C:\Users\magil\MCI-GRU\.claude\worktrees\elegant-mendeleev-12bad9\data\...`:
  `data/raw/constituents/sp500_pit_gics_top10_mcap_monthly_20160104_20260731_meta.json`
  (6 of 6 outputs absolute) and
  `data/raw/market/sp500_pit_gics_top10_mcap_monthly_20160104_20260731_lseg_20150101_20260731.meta.json`
  (2 of 2). The cause is visible in
  `scripts/data/export_sp500_pit_gics_top10_mcap.py`: `write_constituent_outputs`
  (lines 390-436) and `fetch_history` (lines 444-489) store `str(path)` of
  whatever directory argparse received, so an absolute `--constituents-dir`
  produces absolute sidecar paths. Four pull scripts write sidecars this way:
  `export_sp500_gics_top10_mcap.py`, `export_sp500_joiner_leaver_pit.py`,
  `export_sp500_pit_gics_top10_mcap.py`, `export_sp500_pit_membership.py`.

Where it lives

- One read-only reference checkout on Windows 11 (git 2.49.0.windows.1);
  agent sessions run in git worktrees under `.claude/worktrees/` that hold no
  data. A Google Drive folder `MyDrive/MCI_GRU_shared/data` is mounted by
  Colab and staged from by flat filename.
- 16 of the 19 `scripts/gen_*_nb.py` generators reference
  `/content/drive/MyDrive/MCI_GRU_shared/data`. In
  `scripts/gen_graph_specification_ablation_nb.py` the staging cell defines
  `sha256_file` (lines 484-489), copies each file with `shutil.copy2`
  (line 517), hashes the copy afterwards (line 523) and records
  `staged_files` and `staged_sha256` in the run manifest (lines 691-692). No
  generator compares the computed hash against an expected value; the
  comparison against MANIFEST.txt recorded in
  `docs/research/current/GRAPH_SPECIFICATION_ABLATION_2026-09-01.md` (section
  3.1) was done by a person.
- Configs: 16 files under `configs/` reference `data/raw/` paths, e.g.
  `configs/data/gics_top10_110_2016.yaml` lines 36 (`filename`) and 39
  (`pit_universe_csv`). `mci_gru/data/path_resolver.py` resolves a configured
  path as given, then relative to the package root, then by basename across
  seven fallback directories, using only `Path.exists()` at each step.
  `mci_gru/data/data_manager.py` calls it at lines 74, 107, 157 and 239.

Hosting and environment

- GitHub repository on the free or personal tier. CI (`.github/workflows/ci.yml`)
  runs on `ubuntu-latest` with Python 3.10 and `pip install -e ".[dev,fred]"`;
  it uses `actions/checkout` without an `lfs` input.
- Local venv is uv-managed Python 3.12 with `requirements.lock`; neither
  `dvc` nor any LFS package appears in `requirements.lock` or `pyproject.toml`.
  There is no `.dvc/` directory, `.gitattributes` or `.lfsconfig` in the tree.
- `git-lfs` 3.6.1 is on PATH at `/mingw64/bin/git-lfs` (inside the Git for
  Windows installation); `dvc` is not on PATH [L3].

Prior art

- `mci_gru/evaluation/run_bundle.py`: `sha256_file` (lines 50-55, 1 MiB
  chunks), `sha256_directory` (58-61), `describe_artifact` (64-74, returns
  `path`, `size_bytes`, `sha256`), `build_run_manifest` with
  `schema_version` 1 (77-142), `validate_run_bundle` (145-158),
  `write_run_manifest` (161-199), and `_describe_file_collection` (282-302),
  which emits one `{path, sha256, size_bytes}` entry per file plus an aggregate
  digest over relative path and hash. Covered by five tests in
  `tests/test_run_bundle_manifest.py`.
- `C:\Users\magil\mci-gru-data-preservation\2026-07-31-110-universe-2016\MANIFEST.txt`:
  a comment header (vintage, capture date, source path, selector) followed by
  one `sha256  bytes  path` line per file, paths relative to `data/raw`. No
  script in the repository writes this format (grep for `MANIFEST.txt` finds
  only the 2026-09-01 research doc).
- Tests coupled to data paths or sidecars: `tests/test_data_loading_helpers.py`
  (path resolver, lines 98-102), `tests/test_experiment_summary.py`
  (`data/raw/market/sp500_data.csv` as a config value and a
  `data_file_sha256` field, lines 75-121),
  `tests/test_sp500_pit_gics_top10_baseline_notebook.py` (asserts the
  `..._20260622_meta.json` filename token, line 66),
  `tests/test_pit_repeated_seed_replication_notebook.py` (asserts hardwired
  Drive paths, lines 189-191). Fifteen test files reference `gen_*_nb`.
- `docs/CONFIGURATION_GUIDE.md` lines 76, 153 and 159 print `data/raw` paths
  and describe them as "(gitignored)" and "not in the repository".

## Comparison table

| Axis | A Hardened sidecars | B DVC | C Git LFS | D Content-addressed store |
|---|---|---|---|---|
| Corpus shape (70 files, 0.94 GiB, whole vintages a few times a year) | Neutral: one sidecar per pull, no size limit, vintage arrives as CSV plus sidecar as today. | Neutral: one `.dvc` file per file or directory; cache holds a full copy of every version in `.dvc/cache/files/md5/...` [S30]. | 0.94 GiB is 9.4 percent of the 10 GiB storage quota; every pushed version counts at full size and storage accrues hourly [S1]; largest file (147.7 MB) is under the 2 GB per-file limit [S4]. | Neutral: store grows by one vintage at a time; layout is whatever the repo defines. |
| Where it lives (read-only checkout, data-less worktrees, flat Drive folder) | Sidecars are plain files in the tree; no tool touches worktrees or Drive; Drive folder stays flat. | Cache is per project copy and Git-ignored [S30]; sharing across worktrees needs `dvc cache dir <absolute path>` [S29]; a Google Drive remote needs its own OAuth app since the default app is blocked [S12][S13]; a plain local-path remote on a mounted drive is supported [S28]; a DVC remote is not a flat folder of named files (layout on the remote: not established from a primary source; the cache layout is by md5 [S30]). | Objects live in the common git dir, so linked worktrees share `.git/lfs/objects` and inherit the smudge filter [L1]; keeping worktrees data-less needs `GIT_LFS_SKIP_SMUDGE` or `lfs.fetchexclude` [S8]; Drive folder is untouched by LFS. | Store can live anywhere; if it lives on Drive it must either duplicate the flat folder or change how notebooks stage. |
| Hosting (GitHub free tier) | No quota involved. | No GitHub quota involved; the data goes to the DVC remote. | Free plan: 10 GiB storage and 10 GiB bandwidth per month, per repository owner; downloads count, uploads do not; Actions downloads and forks count against the owner [S1][S2]; data packs replaced by metered billing [S2]; calculator states additional storage $0.07 per GiB and additional data transfer out $0.0875 per GiB [S3]. | No quota involved. |
| DVC specifics | n/a (hash is chosen by the repo; run_bundle uses SHA-256). | Google Drive remote supported with caveats [S12]; `.dvc` files carry `md5` and the `hash` field allows "only md5" [S14]; SHA-256 request open since 2020-01-06 [S15]; git worktree: "has not been tested" (2020) [S23], `exp run` failures fixed by a PR merged 2025-10-04 [S24][S25]; Windows: installers available, Python 3.9+, Command Prompt called inadequate, path-length and antivirus caveats [S20][S21]; `dvc.lock` is written only by DVC pipeline commands from `dvc.yaml` stages [S17][S18], `dvc add` writes `.dvc` files and never `dvc.lock` [S19]. | n/a (pointer oid is SHA-256 only [S6]). | n/a (hash is chosen by the repo). |
| Portability of the manifest to Colab without the tool | Yes: `sha256  bytes  path` text or a JSON sidecar is checkable with `hashlib` (the notebooks already compute SHA-256 [gen_graph_specification_ablation_nb.py 484-489]). | Partly: a single-file `.dvc` entry holds an md5 of the raw bytes (DVC 3 treats all files as binary [S16]), checkable with `hashlib.md5`; directory entries point to a cache-side `.dir` object (format not fetched). | Yes: a pointer file is three text lines including `oid sha256:<hex>` and `size` [S6], checkable with `hashlib`; observed locally that `sha256sum` of the file equals the oid [L2]. The pointer must be readable, which is what a checkout without git-lfs yields [S8]. | Yes: the repo defines the manifest; `run_bundle._describe_file_collection` already emits `{path, sha256, size_bytes}`. |
| Blast radius (see table below) | Smallest: 4 pull scripts, backfill of 10 sidecars, optional expected-hash check in 16 generators. | Medium: `.dvc/` dir and config committed, `.dvc` files, DVC install on Windows, Colab auth or local remote, tests. | Medium to large: `.gitignore` line 1 must change, `.gitattributes` added, existing untracked CSVs added to LFS, `path_resolver` mistakes pointers for data, CI checkout policy, bandwidth. | Largest: new store layout, manifest writer and verifier, resolver changes, notebook staging changes. |
| Prior art in the repo | MANIFEST.txt format and `run_bundle` hashing are directly reusable; sidecars already exist. | None (no `.dvc`, no DVC in locks). | git-lfs binary present on the machine [L3]; no `.gitattributes`. | `run_bundle.sha256_file`, `_describe_file_collection`, and the MANIFEST.txt format are the store's manifest half; the store half does not exist. |

## A. Hardened sidecars

What exists. Ten tracked `*meta.json` files carry provenance (source,
selector, dates, row counts, output paths) but no checksum, and two carry
absolute Windows paths (Constraints). The external MANIFEST.txt already
demonstrates the missing half: `sha256  bytes  path` with relative paths.

What hardening means in repo terms. Add `sha256` and `bytes` per output to the
sidecar written by the four pull scripts, and store output paths relative to
the repository root instead of `str(path)`. `run_bundle.describe_artifact`
(lines 64-74) already returns exactly `path`, `size_bytes`, `sha256` for one
file and `_describe_file_collection` (282-302) does it for a set, so the
writer is a call into existing, tested code rather than new hashing code.

Corpus and hosting. No quota, no tool, no size limit. A vintage arrives as it
does today (CSV plus sidecar on Drive, CSV ignored in git, sidecar tracked).

Colab. The generated notebooks already compute SHA-256 of each staged file
after copying (gen_graph_specification_ablation_nb.py lines 508-524). A
hardened sidecar gives them an expected value to compare against, which is
the step that was done by hand for the 2026-09-01 ablation (research doc
section 3.1). Verification needs only `hashlib`.

Limits established from the repo. A sidecar is written by the same script
that writes the CSV, so it attests to what the pull produced, not to what a
later copy contains; the check has to run at the consumer (notebook or
loader). Nothing in the repo verifies sidecars today, and `path_resolver`
does not consult them.

## B. DVC

Google Drive remote. The remote type is supported [S12]. The page carries the
banner "There is an ongoing issue and the default Google DVC app is affected.
If you see 'This app is blocked' message, check this ticket for a workaround
and more details" [S12]; the ticket (#10516, opened 2024-08-10) is closed as
not planned, the error is "This app is blocked. This app tried to access
sensitive info in your Google Account", the maintainer's workaround is to
create your own Google Cloud project and set `gdrive_client_id` and
`gdrive_client_secret`, described as what "was always the recommended way",
and no statement says the default app will be restored [S13].

Authentication from Windows and Colab. First use opens "a special Google
authentication web page"; credentials are then cached locally, at
`%CSIDL_LOCAL_APPDATA%` on Windows and under `~/.cache` on Linux, or at the
path given by `gdrive_user_credentials_file` [S12]. For non-interactive use
"a `GDRIVE_CREDENTIALS_DATA` can be set to pass user credentials in CI/CD
systems, production setup, read-only file systems, etc." with the same JSON
as the credentials file [S12]. Service accounts are configured with
`gdrive_use_service_account true` and
`gdrive_service_account_json_file_path` [S12]. The page has no Colab-specific
or Windows-specific note; how the OAuth browser step behaves inside a Colab
cell is not established from a primary source. An alternative that avoids the
Drive API is a local-path remote: "You can also use system directories,
mounted drives, network resources e.g. network-attached storage (NAS), and
other external devices as storage" [S28], which would point at the Colab
FUSE mount on one side and a Drive for Desktop mount on the other. Whether a
DVC remote keeps files under their original names is not established from
the pages fetched; the local cache is laid out as
`.dvc/cache/files/md5/xx/yyyy` [S30] and `dvc add` says "The hash value (md5
field) corresponds to a file path in the cache" [S19], so a DVC remote would
not be the flat, name-addressed folder the 16 generators stage from.

Hash. A `.dvc` output entry has `path`, `hash` ("Hash algorithm for the file
or directory being tracked with DVC (only `md5` is currently supported)"),
`md5`, `size`, `nfiles` [S14]. The request "Support hashings other than MD5"
(#3069) was opened 2020-01-06 and is open, with an earlier attempt (#2022)
closed unmerged [S15]. SHA-256 therefore cannot be selected. DVC 3.0 removed
the CRLF-to-LF conversion before hashing and "treats all files as if they
contain binary data" [S16], so the md5 in a `.dvc` file is the md5 of the raw
bytes.

Git worktrees. A DVC maintainer wrote in October 2020 that DVC "has not been
tested with `git worktree`" [S23]. Issue #5456 (2021-02-12) reported
`dvc exp run --queue` failing with `[Errno 2] No such file or directory:
'<prefix>/.git/worktrees/<name>/logs/refs/stash'` [S24]; it was closed by PR
#10834 "exp run: add test for running inside a linked worktree", merged
2025-10-04 [S25]. Which DVC release carries that fix is not established from
a primary source (the releases page did not render usably; see Sources).
Separately, DVC's own use of the word "worktree" (issue #8293, "For the
worktree workflow, `dvc add` cannot just remove existing stages") refers to a
cloud-versioning remote mode, which supports only S3, Azure Blob and Google
Cloud Storage [S26][S27], not to git worktrees. For data-less agent
worktrees the operative fact is cache placement: the cache is under
`.dvc/cache` and is Git-ignored at init [S30]; sharing it across copies of the
project needs `dvc cache dir /absolute/path` outside the workspace, with
`cache.type symlink` and `cache.shared group` recommended [S29]. Each linked
worktree would otherwise carry its own copy of any data it pulls.

Windows. Install via winget, Chocolatey, Scoop, conda, pip or a self-contained
installer; "Python 3.9+ is needed to get the latest version of DVC"; the
installer "by default enables symlink permissions for all users" [S20]. The
"Running DVC on Windows" guide says the regular Windows Command Prompt is
inadequate and recommends WSL 2, Cmder or Anaconda Prompt; DVC commands may
fail on paths over 260 characters unless long paths are enabled; whitelist
DVC in Windows Security; symlinks need the "Create symbolic links" privilege;
CRLF versus LF can trigger unnecessary reproduction [S21]. Neither page makes
a support-status statement. PyPI lists dvc 3.67.1 (2026-03-31), "Requires:
Python >=3.9", and a `gdrive` extra [S22]; the repo's Python 3.10 CI and 3.12
venv satisfy the floor.

What `dvc.lock` would add. `dvc.lock` is kept per `dvc.yaml`, records per
stage the `cmd`, `deps` and `outs` with md5 and size, and `params`; "Avoid
editing these files. DVC will create and update them for you" [S17]. It is
updated by `dvc repro`, which runs "the stages listed in `dvc.yaml`" and
updates "the hash values of changed dependencies and outputs ... in
`dvc.lock`" [S18]. `dvc add` creates a `.dvc` file per target, adds the
target to `.gitignore` and stores it in the cache; the documentation does not
indicate it touches `dvc.lock` or `dvc.yaml` [S19]. For a pipeline that is
not run through `dvc repro`, `dvc.lock` would not exist; only `.dvc` md5
records from `dvc add` would.

## C. Git LFS

Quotas. "GitHub Free, Pro, and Free for organizations" receive 10 GiB of
storage and 10 GiB of bandwidth per month; Team and Enterprise Cloud receive
250 GiB each [S1][S2]. Usage counts against the repository owner's account
and resets on the first of the month [S1]. Downloads count: "When you
download a Git LFS file, the bandwidth you use is included in the repository
owner's bandwidth usage"; pulling updated versions, GitHub Actions downloads
("If GitHub Actions downloads a 500 MB file that is tracked with Git LFS, it
will use 500 MB of the repository owner's bandwidth"), source-archive
downloads and pulls from forks all count against the owner; uploads are not
measured [S1][S2]. Storage accrues continuously and is calculated hourly;
every version pushed counts at its full size; deleting objects mid-month does
not recalculate that month [S1].

Overage. "Previously, Git LFS billing used pre-paid data packs. These have
been removed and replaced with metered billing" [S2]. The docs page states no
price and says "To estimate costs for paid Git LFS usage, use the GitHub
pricing calculator" [S2]; the calculator states "Additional storage $0.07 per
GiB, additional data transfer out $0.0875 per GiB" [S3] (the storage billing
period is expressed on the docs page as an hourly usage rate rather than a
per-month figure [S2]). With a budget of $0, "Git LFS usage is blocked for
the rest of the calendar month"; with no payment method, usage is blocked
once the quota is used and new files cannot be pushed [S1][S2].

Per-file limits. GitHub Free 2 GB, Pro 2 GB, Team 4 GB, Enterprise Cloud
5 GB [S4]. The largest corpus file is 147.7 MB.

Arithmetic (derived, not a vendor fact). The corpus is 0.94 GiB, 9.4 percent
of the storage quota; each whole vintage of the 110-name family adds about
46.5 MB. One full LFS download of the corpus consumes 0.94 GiB of the
10 GiB monthly bandwidth, so about ten full downloads per month fit; a
single-vintage download (about 0.043 GiB) fits about 230 times.

Pointer files and hash. A pointer is UTF-8 text of `{key} {value}` lines,
sorted, under 1024 bytes: `version https://git-lfs.github.com/spec/v1`,
`oid sha256:<hex>`, `size <bytes>`; "Currently, only sha256 is supported";
objects are stored at `.git/lfs/objects/OID[0:2]/OID[2:4]/OID` [S6]. Observed
locally: the committed pointer for a test file read exactly those three lines
and `sha256sum` of the working file equalled the oid [L2].

Worktrees. `git lfs install --worktree` sets the filters in the working
tree's config and "If multiple working trees are in use, the Git config
extension `worktreeConfig` must be enabled" [S7]. Objects default to "`lfs`
in Git repository directory (usually `.git/lfs`)" [S8]. Observed locally with
git-lfs 3.6.1: in a linked worktree `LocalGitDir` was `main/.git/worktrees/wt`
while `LocalGitStorageDir` and `LocalMediaDir` were `main/.git` and
`main/.git/lfs/objects`, and the smudge filter set with `--local` was visible
from the worktree [L1]. So worktrees share one object store and, by default,
materialize CSVs on checkout; keeping them data-less needs
`GIT_LFS_SKIP_SMUDGE` ("Sets whether or not Git LFS will skip attempting to
convert pointers of files tracked into their corresponding objects when
checked out") or `lfs.fetchexclude` [S8]. The 2015 investigation issue #545 is
closed; the fetched content records the $GIT_DIR versus $GIT_COMMON_DIR
question but no conclusion [S10].

Setup. `git lfs track <pattern>` "amends your repository's .gitattributes
file"; commit `.gitattributes`; "If there are existing files in your
repository that you'd like to use with GitHub, you need to first remove them
from the repository and then add them to Git LFS locally" [S5]. git-lfs.com
offers v3.8.0 downloads and requires `git lfs install` once per user account
[S9]; whether Git for Windows bundles git-lfs is not stated on
gitforwindows.org [S31], but git-lfs 3.6.1 is present inside this machine's
Git for Windows install [L3]. Whether Colab runtimes ship git-lfs is not
established from a primary source. CI: `actions/checkout`'s `lfs` input,
"Whether to download Git-LFS files", defaults to `false` [S11], so CI would
receive pointers unless opted in, and opting in spends the owner's bandwidth
[S1].

Interaction with the resolver. `path_resolver.resolve_project_data_path`
accepts any existing path; a 130-byte pointer at
`data/raw/market/<vintage>.csv` satisfies `Path.exists()` and would be handed
to the CSV loader as if it were data (path_resolver.py lines 19-21 and
34-38). That is a new failure mode LFS would introduce in every checkout
where smudge is skipped or the object was not fetched.

Colab. The pointer is verifiable with `hashlib` and no git-lfs, but the
notebook must obtain the pointer text; a clone without git-lfs installed
yields pointers [S8], which is sufficient for verification against a file
staged from Drive by flat path. Pulling the objects themselves from GitHub in
Colab would count against the 10 GiB monthly bandwidth [S1].

## D. Home-grown content-addressed store

No vendor facts apply; everything below is from the repository.

What exists. The manifest half already exists twice: MANIFEST.txt
(`sha256  bytes  path`, relative paths, per-vintage header) and
`run_bundle._describe_file_collection` (JSON `{path, sha256, size_bytes}`
entries plus an aggregate digest). `sha256_file` reads in 1 MiB chunks, the
same loop the notebooks embed. `validate_run_bundle` shows the shape of a
presence check with a `status` and `missing_artifacts` result.

What does not exist. A store layout (for example `store/sha256/xx/<hash>`),
a writer that moves pull outputs into it, a resolver that maps a configured
name to a hash and then to a store path, and a verifier the notebooks call.
`path_resolver` resolves names, not hashes, so either configs keep naming
files and the manifest maps name to hash (a manifest-plus-verify design that
is A at the vintage level), or configs name hashes and every config path,
the resolver and the generator staging tables change.

Where it would live. The Drive folder is staged from by flat filename by 16
generators; a hashed store on Drive would need either a parallel flat copy
(duplicating 0.94 GiB) or a change to every generator's staging cell. On the
Windows side the store could sit beside the reference checkout, outside git,
with the manifest tracked.

Colab. The manifest is plain text or JSON and verifiable with `hashlib`.

Corpus fit. Content addressing dedupes identical bytes across vintages, but
the vintages are distinct CSV exports with different date ranges, so the
saving is not established and would need measuring.

## Blast radius

"Change" means a file must be edited or created for the option to function
as an identity mechanism, not merely to be nice.

| Surface | A Hardened sidecars | B DVC | C Git LFS | D Content-addressed store |
|---|---|---|---|---|
| Config paths (16 files under `configs/`) | No change | No change (files reappear at the same paths after `dvc pull`/checkout) | No change to paths; `.gitignore` line 1 `*.csv` must be removed or narrowed and `.gitattributes` added [S5] | Change if files are stored by hash; no change if the manifest maps names to hashes and the resolver consults it |
| Pull scripts (4 sidecar writers) | Change: write `sha256`, `bytes`, relative paths (reuse `run_bundle.describe_artifact`) | Change or manual step: `dvc add` after each pull; `.dvc` files committed | Change or manual step: `git add` of tracked CSVs; existing untracked CSVs added fresh | Change: write into the store and append to the manifest |
| Notebook generators (16 stage from Drive) | Optional: compare computed SHA-256 to the sidecar value (hash code already present) | Change if `dvc pull` is used on Colab (install `dvc[gdrive]`, custom OAuth app or `GDRIVE_CREDENTIALS_DATA`, or a local remote on the mount); otherwise optional md5 check against `.dvc` | Change: read the pointer from the checkout and compare, or `git lfs pull` (bandwidth) | Change: stage from the store or verify against the manifest |
| Tests (15 reference `gen_*_nb`; 4 touch data paths or sidecars) | Add: sidecar schema test; existing tests pass (they assert filename tokens only) | Add: `.dvc` presence/format tests; `tests/test_data_loading_helpers.py` unaffected | Add: pointer-versus-data guard in `path_resolver` and its test (lines 98-102 of test_data_loading_helpers.py); notebook tests that assert Drive paths unaffected | Add: store, manifest and resolver tests |
| CI (`ci.yml`, ubuntu, Python 3.10) | No change | No change unless a test imports `dvc` (then `pip install dvc`) | Policy decision: `actions/checkout` `lfs` defaults to `false` [S11]; leaving it false means pointers in CI; `true` spends owner bandwidth [S1] | No change |
| Docs (`CONFIGURATION_GUIDE.md` lines 76, 153, 159; handoffs) | Add: sidecar contract | Add: DVC setup, Windows caveats, Colab auth | Change: "(gitignored)" and "not in the repository" wording at lines 153 and 159 becomes false | Add: store and manifest contract |
| New tracked artefacts | None beyond edited sidecars | `.dvc/config`, `.dvc/.gitignore`, one `.dvc` file per target | `.gitattributes`, one pointer per CSV (about 70) | Manifest file(s) |
| Tooling on Windows | None | DVC install; Command Prompt inadequate per DVC [S21] | git-lfs already present [L3] | None |
| Tooling on Colab | None | `pip install dvc[gdrive]` plus credentials, or none if only verifying md5 | None if only verifying pointers; git-lfs on Colab not established | None |

## Sources

Vendor documentation and primary project pages, all read 2026-09-05.

- [S1] GitHub Docs, About storage and bandwidth usage.
  https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage
- [S2] GitHub Docs, About billing for Git Large File Storage.
  https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage
- [S3] GitHub pricing calculator, LFS feature.
  https://github.com/pricing/calculator?feature=lfs
- [S4] GitHub Docs, About Git Large File Storage (per-file limits).
  https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage
- [S5] GitHub Docs, Configuring Git Large File Storage.
  https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage
- [S6] Git LFS specification (pointer format, sha256, object path).
  https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md
- [S7] git-lfs-install man page (--local, --worktree).
  https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-install.adoc
- [S8] git-lfs-config man page (GIT_LFS_SKIP_SMUDGE, lfs.fetchexclude, lfs.storage).
  https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-config.adoc
- [S9] git-lfs.com (downloads, `git lfs install`).
  https://git-lfs.com/
- [S10] git-lfs issue #545, Investigate Git LFS and Git worktrees.
  https://github.com/git-lfs/git-lfs/issues/545
- [S11] actions/checkout README (`lfs` input default false).
  https://github.com/actions/checkout
- [S12] DVC docs, Google Drive remote (dvc.org/doc URL returned 301 to this address).
  https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive
- [S13] DVC issue #10516, Unable connect dvc to Google Drive. Access blocked!
  https://github.com/iterative/dvc/issues/10516
- [S14] DVC docs, .dvc files (fields, "only md5 is currently supported").
  https://doc.dvc.org/user-guide/project-structure/dvc-files
- [S15] DVC issue #3069, Support hashings other than MD5 (open).
  https://github.com/iterative/dvc/issues/3069
- [S16] DVC docs, Upgrading to DVC 3.0 (line-ending conversion removed).
  https://doc.dvc.org/user-guide/upgrade
- [S17] DVC docs, dvc.yaml files (dvc.lock section).
  https://doc.dvc.org/user-guide/project-structure/dvcyaml-files
- [S18] DVC command reference, repro.
  https://doc.dvc.org/command-reference/repro
- [S19] DVC command reference, add.
  https://doc.dvc.org/command-reference/add
- [S20] DVC docs, Install on Windows.
  https://doc.dvc.org/install/windows
- [S21] DVC docs, Running DVC on Windows.
  https://doc.dvc.org/user-guide/how-to/run-dvc-on-windows
- [S22] PyPI, dvc (3.67.1, 2026-03-31, Requires Python >=3.9, extras).
  https://pypi.org/project/dvc/
- [S23] DVC community forum, DVC with git worktree (2020-10-16).
  https://discuss.dvc.org/t/dvc-with-git-worktree/532
- [S24] DVC issue #5456, exp: git worktree is unsupported.
  https://github.com/iterative/dvc/issues/5456
- [S25] DVC PR #10834, exp run: add test for running inside a linked worktree (merged 2025-10-04).
  https://github.com/iterative/dvc/pull/10834
- [S26] DVC issue #8293, worktree add: dvc add removes valid metadata (cloud-versioning sense).
  https://github.com/iterative/dvc/issues/8293
- [S27] DVC docs, Cloud versioning (supported providers).
  https://doc.dvc.org/user-guide/data-management/cloud-versioning
- [S28] DVC docs, Remote storage (local file system remotes).
  https://doc.dvc.org/user-guide/data-management/remote-storage
- [S29] DVC docs, How to share a DVC cache.
  https://doc.dvc.org/user-guide/how-to/share-a-dvc-cache
- [S30] DVC docs, Internal files (.dvc/cache layout, gitignored).
  https://doc.dvc.org/user-guide/project-structure/internal-files
- [S31] gitforwindows.org (no statement about Git LFS bundling).
  https://gitforwindows.org/

Note on the DVC GitHub pages: the issue, PR and release pages fetched under
`github.com/iterative/dvc/...` rendered under the `treeverse` organization.
This report records the fact and draws no conclusion from it.

Pages that could not be used

- https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive,
  https://dvc.org/doc/user-guide/project-structure/dvc-files,
  https://dvc.org/doc/user-guide/project-structure/dvc-lock-files,
  https://dvc.org/doc/install/windows: each returned 301 to doc.dvc.org; the
  redirect targets were fetched instead.
- https://doc.dvc.org/user-guide/project-structure/dvc-lock-files: 404. The
  dvc.lock facts come from [S17] and [S18].
- https://docs.github.com/en/billing/concepts/product-billing/git-large-file-storage: 404.
- https://doc.dvc.org/install: fetched, index page only, no facts.
- https://doc.dvc.org/command-reference/remote/modify and the docs source
  https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/command-reference/remote/modify.md:
  fetched, but the returned content did not include the `worktree` or
  `gdrive_*` option text, so the cloud-versioning `worktree` option wording
  is not established from a primary source.
- https://raw.githubusercontent.com/iterative/dvc.org/main/content/docs/user-guide/data-management/cloud-versioning.md
  and .../remote-storage/index.md: fetched; no sentence on remote file layout
  or the `worktree` option was returned.
- https://github.com/iterative/dvc/releases: fetched, but the content rendered
  inconsistently (reported 3.67.1 with a 2025 date and no releases after
  2024); not used. Version and date facts come from PyPI [S22].

Local observations (this machine, not vendor documentation)

- [L1] In a throwaway repository under the scratchpad, with git 2.49.0.windows.1
  and git-lfs 3.6.1, `git lfs env` in a linked worktree reported
  `LocalGitDir=<main>/.git/worktrees/wt`, `LocalGitStorageDir=<main>/.git`,
  `LocalMediaDir=<main>/.git/lfs/objects`; the object was found at
  `<main>/.git/lfs/objects/e5/98/e598...`; `filter.lfs.smudge` set with
  `git lfs install --local` was visible from the worktree. The repository was
  deleted afterwards.
- [L2] The committed pointer read `version https://git-lfs.github.com/spec/v1`,
  `oid sha256:e598a5f6...`, `size 4096`, and `sha256sum` of the working file
  matched the oid.
- [L3] `git-lfs` 3.6.1 is at `/mingw64/bin/git-lfs`; `dvc` is not on PATH.

## Observations

1. The repository already has a SHA-256 identity format in two places
   (MANIFEST.txt and `run_bundle._describe_file_collection`) and already
   computes SHA-256 on Colab after staging; what is missing is an expected
   value to compare against and a writer that emits it with relative paths.
2. DVC cannot emit SHA-256 (`hash` accepts only md5 [S14], request open since
   2020 [S15]), its default Google Drive app is blocked with the workaround
   closed as not planned [S13], and its remote is not the flat, name-addressed
   Drive folder the 16 generators stage from; git worktree support beyond
   `exp run` rests on a 2020 "not tested" statement and cache placement
   configuration [S23][S29].
3. Git LFS fits the corpus size today (9.4 percent of the 10 GiB storage
   quota, largest file well under 2 GB) but converts every full fetch into
   about a tenth of the monthly bandwidth [S1], shares objects across
   worktrees and smudges them by default [L1][S8], and makes a pointer file
   satisfy `path_resolver`'s existence check.
4. A home-grown store reuses the most existing code but is the only option
   that changes how names resolve to files and how notebooks stage from Drive.
5. Counting corrections for the ticket: ten sidecars are tracked, not eight;
   the corpus is 1,013,195,582 bytes (966.3 MiB); the 2026-07-31 vintage is
   46,502,119 bytes.

On these facts, the facts favour hardening the existing sidecars with the
`sha256  bytes  relative-path` triple already used by MANIFEST.txt and
`run_bundle`, with the notebooks comparing rather than merely printing.
