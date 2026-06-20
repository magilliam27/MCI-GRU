from __future__ import annotations


class GitHubSyncDisabled(RuntimeError):
    """Raised when a caller tries to sync GitHub without explicit enablement."""


def sync_github(*, enabled: bool) -> None:
    if not enabled:
        raise GitHubSyncDisabled("GitHub cockpit sync requires --github-sync.")
    raise NotImplementedError(
        "Guarded GitHub sync is intentionally not implemented in the local-first slice."
    )
