"""Generate notebooks/eodhd_price_pull_colab.ipynb, the Colab pull of EODHD stock prices (#281).

The notebook runs ``scripts/data/export_eodhd_pit_prices.py`` where EODHD is
reachable. It reads the membership file and the LSEG reference panel from the
preserved Drive package, checks both against the committed r1 manifest, pulls
EODHD daily prices for every name, and copies the new package and its manifest
to Drive beside the LSEG one. The notebook carries no data logic of its own.

Run from the repository root: ``python scripts/gen_eodhd_price_pull_nb.py``.
"""

from __future__ import annotations

from pathlib import Path

from nb_lib import code, md, write_notebook

NOTEBOOK_PATH = Path("notebooks/eodhd_price_pull_colab.ipynb")
#: Colab metadata without a GPU accelerator: the pull needs none.
COLAB_CPU_METADATA: dict = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def build_cells() -> list[dict]:
    return [
        md(
            """
            # EODHD stock prices for the 110-name universe (Colab)

            Pulls EODHD daily prices for all 206 names in the preserved 110-name
            point-in-time universe (#281), so the first run can train on EODHD instead
            of the LSEG export. The universe itself is unchanged: the same membership
            file is byte-copied into the new package.

            Each name's EODHD symbol is proven against the preserved LSEG panel: its
            daily returns must agree over the span the universe needs it. A name that
            does not agree stops the pull with its name, and no manifest is published.

            **Before you start**

            1. The preserved LSEG package is on Drive at
               `MCI_GRU_shared/preservation/2026-10-02-110-universe-2016`.
            2. A Colab secret named `EODHD_API_KEY` exists and this notebook has access
               to it (key icon in the left bar). The key is never printed or written.
            3. Any runtime type works; no GPU is needed.

            **Where outputs go.** The package is built on local disk, because manifest
            publication needs hard links. It is then copied to
            `MyDrive/MCI_GRU_shared/preservation/<OUTPUT_NAME>/`, beside the LSEG
            package. EODHD responses are cached on Drive, so a re-run after a symbol fix
            spends calls only on what changed. Nothing on Drive is ever overwritten or
            deleted.
            """
        ),
        code(
            """
            # Parameters
            BRANCH = "claude/first-run-eodhd-prices-g5lvy4"  # "main" once #281's pull request is merged
            EXPECTED_COMMIT = ""  # optional: the full commit SHA to run

            LSEG_PACKAGE_DRIVE_DIR = "/content/drive/MyDrive/MCI_GRU_shared/preservation/2026-10-02-110-universe-2016"
            PREFIX = "sp500_pit_gics_top10_mcap_monthly_20160104_20260731"
            REFERENCE_PANEL = f"market/{PREFIX}_lseg_20150101_20260731.csv"
            PIT_UNIVERSE = f"constituents/{PREFIX}_pit_universe.csv"
            REFERENCE_MANIFEST = f"data/manifests/{PREFIX}.r1.json"
            REFERENCE_MANIFEST_SHA256 = "1e2043dcb4f00a88de128985cc94423c122e480034c16ec741710a2460653ecf"

            START, END = "2015-01-01", "2026-07-31"  # the LSEG panel's span
            REVISION = "r2"  # r1 kept carried rows after three delistings; never reuse one
            OUTPUT_NAME = f"eodhd_{PREFIX}_{REVISION}"
            OUTPUT_DRIVE_DIR = f"/content/drive/MyDrive/MCI_GRU_shared/preservation/{OUTPUT_NAME}"
            CACHE_DRIVE_DIR = "/content/drive/MyDrive/MCI_GRU_shared/eodhd_cache/gics_top10_110_2016"
            LOCAL_PACKAGE = "/content/eodhd_package"
            LOCAL_MANIFEST = f"/content/{PREFIX}_eodhd.{REVISION}.json"
            """
        ),
        code(
            """
            # Mount Drive and read the EODHD key from Colab secrets. It is placed in this
            # process's environment for the pull's subprocess only; never printed or saved.
            import os
            import subprocess
            import sys
            from pathlib import Path

            from google.colab import drive, userdata

            drive.mount("/content/drive")
            os.environ["EODHD_API_KEY"] = userdata.get("EODHD_API_KEY") or ""
            if not os.environ["EODHD_API_KEY"].strip():
                raise RuntimeError("The Colab secret EODHD_API_KEY is empty or not shared with this notebook.")
            print("EODHD_API_KEY loaded from Colab secrets.")
            """
        ),
        code(
            """
            # Clone the repository at BRANCH (and EXPECTED_COMMIT, if set), then install the
            # Colab-facing requirements (ranges that Colab's own torch satisfies) and the repository.
            REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
            REPO_DIR = Path("/content/MCI-GRU")


            def stream(command, cwd=None, check=True):
                \"\"\"Run a command, show its output as it arrives, and return its exit code.\"\"\"
                process = subprocess.Popen(
                    command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                for line in process.stdout:
                    print(line, end="")
                code = process.wait()
                if check and code != 0:
                    raise RuntimeError(f"Command failed with exit code {code}: {command[:4]}")
                return code


            if REPO_DIR.exists():
                raise RuntimeError(
                    f"{REPO_DIR} already exists. Use Runtime > Disconnect and delete runtime, "
                    "then run the notebook from the top."
                )
            stream(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)])
            if EXPECTED_COMMIT:
                stream(["git", "-C", str(REPO_DIR), "checkout", "--detach", EXPECTED_COMMIT])
            COMMIT = subprocess.run(
                ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            if EXPECTED_COMMIT and COMMIT != EXPECTED_COMMIT:
                raise RuntimeError(f"Checked out {COMMIT}, expected {EXPECTED_COMMIT}")
            print("Commit:", COMMIT)
            PIP = [sys.executable, "-m", "pip", "install", "-q"]
            stream(PIP + ["-r", str(REPO_DIR / "requirements.txt")])
            stream(PIP + ["--no-deps", "-e", str(REPO_DIR)])
            """
        ),
        code(
            """
            # Pull. The PIT file and the LSEG panel are read from Drive and must match the
            # committed LSEG r1 manifest. Exit status 1 means a blocking finding: the evidence files
            # are still written under LOCAL_PACKAGE, but no manifest is published.
            lseg = Path(LSEG_PACKAGE_DRIVE_DIR)
            exit_code = stream(
                [
                    sys.executable, "-m", "scripts.data.export_eodhd_pit_prices",
                    "--pit-universe", str(lseg / PIT_UNIVERSE),
                    "--reference-panel", str(lseg / REFERENCE_PANEL),
                    "--reference-manifest", REFERENCE_MANIFEST,
                    "--reference-manifest-sha256", REFERENCE_MANIFEST_SHA256,
                    "--start", START,
                    "--end", END,
                    "--package-root", LOCAL_PACKAGE,
                    "--cache-dir", CACHE_DRIVE_DIR,
                    "--manifest-output", LOCAL_MANIFEST,
                    "--package-revision", REVISION,
                ],
                cwd=REPO_DIR,
                check=False,
            )
            print("Pull exit code:", exit_code)
            """
        ),
        code(
            """
            # Copy the package (and its manifest, when one was published) to Drive. Every
            # destination is checked before anything is copied: a file already there with other
            # bytes stops the copy, and nothing is overwritten. A failed pull goes to its own
            # timestamped folder, so re-runs never collide.
            import filecmp
            import hashlib
            import shutil
            from datetime import datetime, timezone

            target = Path(OUTPUT_DRIVE_DIR)
            if exit_code != 0:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                target = target.with_name(f"{target.name}_FAILED_{stamp}")
            sources = [p for p in sorted(Path(LOCAL_PACKAGE).rglob("*")) if p.is_file()]
            pairs = [(p, target / p.relative_to(LOCAL_PACKAGE)) for p in sources]
            manifest = Path(LOCAL_MANIFEST)
            if exit_code == 0 and manifest.exists():
                pairs.append((manifest, target / manifest.name))
            clashes = [d for s, d in pairs if d.exists() and not filecmp.cmp(s, d, shallow=False)]
            if clashes:
                raise RuntimeError(f"{len(clashes)} files already on Drive with other bytes, e.g. {clashes[0]}")
            copied = 0
            for source, destination in pairs:
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                copied += 1
            if exit_code == 0 and manifest.exists():
                print("Manifest SHA-256:", hashlib.sha256(manifest.read_bytes()).hexdigest())
            print(f"Copied {copied} files to {target}")
            """
        ),
        md(
            """
            ## After the pull

            - **Exit code 0:** the package and `<PREFIX>_eodhd.<REVISION>.json` are on Drive.
              Tell the thread the pull finished. The manifest gets committed to
              `data/manifests/`, and a data config pins its SHA-256, so the first run can
              stage this package the way it stages the LSEG one.
            - **Exit code 1:** the files went to a timestamped `_FAILED_` folder instead.
              `market/*.meta.json` lists each blocking finding by name, and
              `*_symbols.json` shows every symbol tried. A one-day difference from LSEG is
              usually a spin-off the split records miss: the map's `adjustments` handle it.
              Fix the symbol map
              (`data/mappings/eodhd_symbols_gics_top10_110_2016.json`), then re-run from
              the top; cached responses are reused.
            """
        ),
    ]


def main() -> None:
    write_notebook(build_cells(), NOTEBOOK_PATH, metadata=COLAB_CPU_METADATA, trailing_newline=True)


if __name__ == "__main__":
    main()
