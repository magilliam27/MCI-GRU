"""Publish a new manifest from an explicit JSON source specification.

Run from the repository root with ``python -m scripts.data.write_input_manifest
--spec specification.json --package-root data/raw --output data/manifests/new.json``.
The specification supplies package_id, package_revision, files (path, purpose,
optional metadata), provenance, optional metadata and optional expected_files
(complete prior path/purpose/sha256/size_bytes records for a historical backfill).
The source root and output location are command arguments, never manifest identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mci_gru.data.input_manifest import InputFileRecord, InputFileSpec, write_input_manifest


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate field in source specification")
    return result


def main() -> None:
    """Publish one fully validated manifest; report failures without repairing inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields)
        if not isinstance(spec, dict):
            raise ValueError("The source specification must be a JSON object")
        files = [InputFileSpec(**record) for record in spec.pop("files")]
        if "expected_files" in spec:
            spec["expected_files"] = [
                InputFileRecord(**record) for record in spec["expected_files"]
            ]
        result = write_input_manifest(args.output, args.package_root, files=files, **spec)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "sha256": result.sha256,
                "size_bytes": result.size_bytes,
                "package_id": result.manifest.package_id,
                "package_revision": result.manifest.package_revision,
            }
        )
    )


if __name__ == "__main__":
    main()
