from __future__ import annotations

import pytest

from mci_gru.cockpit.github import GitHubSyncDisabled, sync_github


def test_sync_github_requires_explicit_enablement() -> None:
    with pytest.raises(GitHubSyncDisabled, match="requires --github-sync"):
        sync_github(enabled=False)
