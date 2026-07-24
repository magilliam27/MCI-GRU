# Triage Labels

The Matt Pocock skills use five canonical triage roles. Map them to these exact
GitHub label strings:

| Label in `mattpocock/skills` | Label in MCI-GRU | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer needs to evaluate this issue |
| `needs-info` | `needs-info` | Waiting on the reporter for more information |
| `ready-for-agent` | `ready-for-agent` | Fully specified and ready for an agent |
| `ready-for-human` | `ready-for-human` | Requires human implementation |
| `wontfix` | `wontfix` | Will not be actioned |

Before applying a canonical role, verify that its exact configured label
exists. If it is missing, create that exact label with an appropriate
description and continue the requested workflow. Never substitute a
semantically similar label or create a near-duplicate spelling.

When a skill names a triage role, use the corresponding exact string from this
table.
