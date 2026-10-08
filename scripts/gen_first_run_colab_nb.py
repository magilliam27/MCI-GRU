"""Generate notebooks/first_run_colab.ipynb, the Colab launcher for the first admitted run.

The run itself is the frozen recipe in docs/DEFAULT_EXPERIMENT_RECIPE.md (map #187,
ticket #278). The notebook holds no recipe values: ``scripts/first_run_colab.py``
reads them from the recipe at the cloned commit, stages and verifies the inputs, and
runs ``run_experiment.py``. Every repository step runs in a subprocess, so the
kernel never imports packages that the lock install has just replaced.

Run from the repository root: ``python scripts/gen_first_run_colab_nb.py``.
"""

from __future__ import annotations

from pathlib import Path

from nb_lib import code, md, write_notebook

NOTEBOOK_PATH = Path("notebooks/first_run_colab.ipynb")


def build_cells() -> list[dict]:
    return [
        md(
            """
            # MCI-GRU first admitted run (Colab)

            Runs the frozen recipe in `docs/DEFAULT_EXPERIMENT_RECIPE.md`, as map #187
            defines the first run. The notebook carries no recipe values of its own: it
            reads the override block and slug from the recipe at the commit it clones,
            and stages the input package that recipe's data config declares (the
            preserved LSEG package, or the EODHD-priced package once the recipe names
            it).

            **Before you start**

            1. Pull requests #270, #274 and #275 are merged to `main`. In `full` mode
               the notebook checks for each one and refuses to run without it.
            2. The EODHD S&P 500 file is on Drive at
               `MCI_GRU_shared/preservation/eodhd_sp500_2010_20260919/sp500_index.csv`,
               in its own folder beside the LSEG package
               (351,815 bytes, SHA-256 `9f5cfaf5…f080f5ac`).
            3. A Colab secret named `FRED_API_KEY` exists and this notebook has
               access to it (key icon in the left bar). The key is never printed or
               written anywhere by this notebook.
            4. Runtime type is a GPU runtime.

            **Modes.** `full` is the admitted-run candidate: 20 models x 100 epochs.
            `smoke` keeps every recipe setting except the budget (1 model x 2 epochs),
            runs even before the three pull requests land, and is labelled
            *not evidence*.

            **Where outputs go.** The run writes to local disk under
            `/content/mci_gru_runs/<run tag>`, because snapshot capture publishes with
            hard links and Colab's Drive mount is not known to support them. The folder is copied to
            `MyDrive/MCI_GRU_shared/runs/first_run/<run tag>` every
            `SYNC_EVERY_SECONDS` while it runs and once more at the end. Nothing is
            ever deleted on Drive. `colab_heartbeat.json` there says when the last
            copy was made.

            Runbook: `docs/workflows/first_run/RUNBOOK.md`.
            """
        ),
        code(
            """
            # Parameters
            RUN_MODE = "full"  # "full" = admitted-run candidate; "smoke" = 1 model x 2 epochs, not evidence
            BRANCH = "main"
            EXPECTED_COMMIT = ""  # optional: the full commit SHA to run; empty records whatever BRANCH is

            # Empty: the Drive folder of the package the recipe's data config declares.
            PACKAGE_DRIVE_DIR = ""
            EODHD_DRIVE_PATH = "/content/drive/MyDrive/MCI_GRU_shared/preservation/eodhd_sp500_2010_20260919/sp500_index.csv"
            DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/MCI_GRU_shared/runs/first_run"
            LOCAL_RUN_ROOT = "/content/mci_gru_runs"
            SYNC_EVERY_SECONDS = 600

            # Leave empty: torch is installed from PyPI at the version requirements.lock pins.
            # Only if the environment check finds no CUDA on a GPU runtime: set a PyTorch wheel
            # index that lists the same torch version for an older CUDA, for example
            # "https://download.pytorch.org/whl/cu128", then Runtime > Disconnect and delete
            # runtime and run all again. Re-running only the install cell would keep the first build.
            TORCH_INDEX_URL = ""

            assert RUN_MODE in ("full", "smoke"), RUN_MODE
            """
        ),
        code(
            """
            # Mount Drive and read the FRED key from Colab secrets, before anything is installed.
            # The key goes into this process's environment for the run's subprocess only; it is
            # never printed or saved.
            import os
            import subprocess
            import sys
            from pathlib import Path

            from google.colab import drive, userdata

            drive.mount("/content/drive")
            os.environ["FRED_API_KEY"] = userdata.get("FRED_API_KEY") or ""
            if not os.environ["FRED_API_KEY"].strip():
                raise RuntimeError("The Colab secret FRED_API_KEY is empty or not shared with this notebook.")
            print("FRED_API_KEY loaded from Colab secrets.")
            """
        ),
        code(
            """
            # Clone the repository at BRANCH (and EXPECTED_COMMIT, if set).
            REPO_URL = "https://github.com/magilliam27/MCI-GRU.git"
            REPO_DIR = Path("/content/MCI-GRU")


            def stream(command, cwd=None):
                \"\"\"Run a command and show its output as it arrives; stop on failure.\"\"\"
                process = subprocess.Popen(
                    command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                for line in process.stdout:
                    print(line, end="")
                if process.wait() != 0:
                    raise RuntimeError(f"Command failed with exit code {process.returncode}: {command[:4]}")


            if REPO_DIR.exists():
                raise RuntimeError(
                    f"{REPO_DIR} already exists. Use Runtime > Disconnect and delete runtime, "
                    "then run the notebook from the top, so the run starts from a fresh clone."
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
            print("Python:", sys.version.split()[0])
            """
        ),
        code(
            """
            # Install exactly what requirements.lock pins, then the repository without dependencies.
            # Everything runs in subprocesses, so this kernel never imports the replaced packages.
            PIP = [sys.executable, "-m", "pip", "install", "-q"]
            if TORCH_INDEX_URL:
                torch_pin = next(
                    line.split()[0]
                    for line in (REPO_DIR / "requirements.lock").read_text().splitlines()
                    if line.startswith("torch==")
                )
                stream(PIP + [torch_pin, "--index-url", TORCH_INDEX_URL])
            stream(PIP + ["-r", str(REPO_DIR / "requirements.lock")])
            stream(PIP + ["--no-deps", "-e", str(REPO_DIR)])
            stream([sys.executable, "scripts/first_run_colab.py", "check-env", "--require-gpu"], cwd=REPO_DIR)
            """
        ),
        code(
            """
            # What this commit needs from open pull requests. Full mode stops here if any is missing.
            stream([sys.executable, "scripts/first_run_colab.py", "prerequisites", "--mode", RUN_MODE], cwd=REPO_DIR)
            """
        ),
        code(
            """
            # Stage the recipe's input package and the EODHD S&P 500 file from Drive, and verify them.
            # The Drive copy must carry the inventory the committed manifest vouches for; that
            # inventory must equal the manifest's records; every staged file must pass
            # validate_input_package. A file with other bytes is never copied or replaced.
            # The run cell stages again before it starts.
            stream(
                [
                    sys.executable, "scripts/first_run_colab.py", "stage",
                    "--package-drive-dir", PACKAGE_DRIVE_DIR,
                    "--eodhd-drive-path", EODHD_DRIVE_PATH,
                ],
                cwd=REPO_DIR,
            )
            """
        ),
        code(
            """
            # Run the recipe. Outputs land under LOCAL_RUN_ROOT and are copied to DRIVE_OUTPUT_ROOT
            # every SYNC_EVERY_SECONDS and at the end. colab_run_record.json in the run folder holds
            # the commit, mode, prerequisites, staged input hashes and the exact overrides.
            TAG_FILE = "/content/first_run_tag.txt"
            stream(
                [
                    sys.executable, "scripts/first_run_colab.py", "run",
                    "--mode", RUN_MODE,
                    "--local-root", LOCAL_RUN_ROOT,
                    "--drive-root", DRIVE_OUTPUT_ROOT,
                    "--sync-every", str(SYNC_EVERY_SECONDS),
                    "--package-drive-dir", PACKAGE_DRIVE_DIR,
                    "--eodhd-drive-path", EODHD_DRIVE_PATH,
                    "--tag-file", TAG_FILE,
                ],
                cwd=REPO_DIR,
            )
            """
        ),
        code(
            """
            # Where the outputs are, and what the run decided about its inputs.
            import json

            run_folder = Path(DRIVE_OUTPUT_ROOT) / Path(TAG_FILE).read_text().strip()
            print("Drive run folder:", run_folder)
            record = json.loads((run_folder / "colab_run_record.json").read_text())
            print("Mode:", record["mode"], "-", record["evidence"])
            print("Commit:", record["commit"], "exit code:", record.get("returncode"))
            for name in ("admission.json", "run_failure.json", "run_metadata.json"):
                for path in sorted(run_folder.rglob(name)):
                    print("found", path.relative_to(run_folder))
            """
        ),
        md(
            """
            ## After the run

            - Keep the whole Drive run folder. `input_snapshots/` is the only copy of what
              FRED returned and of the index file the run read.
            - A runtime that stops part way leaves an incomplete run. Start a new run
              from the top; do not resume into the same folder.
            - The run is evidence only once the owner's admission receipt and run charter
              (`docs/workflows/first_run/ADMISSION_RECEIPT_AND_CHARTER_DRAFT.md`) name it.
            """
        ),
    ]


def main() -> None:
    write_notebook(build_cells(), NOTEBOOK_PATH, trailing_newline=True)


if __name__ == "__main__":
    main()
