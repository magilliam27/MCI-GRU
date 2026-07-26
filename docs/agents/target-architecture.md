# MCI-GRU Target Architecture Workspace

Status: **human-led design workspace. Not current-state authority.**

This document does not describe what the code does today. Nothing in it may be
cited as evidence of implemented behaviour, and an agent must never satisfy a
factual question about the repository from this file.

- For implemented architecture, read [../ARCHITECTURE.md](../ARCHITECTURE.md).
- For current-state routing — owning modules, adjacent contracts, focused tests,
  and engineering constraints — read [guide.md](guide.md).
- For the source-of-truth order, read [domain.md](domain.md).

This file is the shared workspace where the project owner and an agent define
MCI-GRU's target architecture together. It is intentionally incomplete. It is
not a roadmap, not an implementation authorisation, and not a summary of
historical plans under `docs/agent_references/`.

## Working Agreement

- The project owner supplies goals, priorities, risk tolerance, and acceptable
  trade-offs.
- The agent supplies current-code evidence, alternatives, consequences,
  migration options, and verification strategies.
- No historical plan, handoff, research report, or agent proposal becomes a
  decision by default.
- Design-gated work in progress elsewhere is not a decision either. Record it as
  an open question, not as a target.
- Record unresolved questions as unresolved. Do not silently choose a target.
- A decision that changes a repository invariant listed in `AGENTS.md` must name
  its migration and validation path before implementation begins.

## Goals

> To decide together.

## Non-Goals

> To decide together.

## Design Forces And Constraints

> To decide together. Candidate inputs may include research objectives,
> experiment cost, data availability, point-in-time guarantees, operational
> simplicity, reproducibility, and the paper-trade boundary.

## Target System Boundaries

> To decide together.

## Target Data And Timing Contracts

> To decide together.

## Target Model And Graph Boundaries

> To decide together.

## Target Training, Evaluation, And Promotion Flow

> To decide together.

## Target Artifact And Provenance Contract

> To decide together.

## Target Paper-Trade And Live-Operations Boundary

> To decide together.

## Migration Strategy

> To decide together, only after the target boundaries are agreed.

## Decision Log

Add a row only after explicit human review. An empty table means no target-state
decision has been made.

| Date | Decision | Rationale | Alternatives considered | Invariants affected | Validation required |
| --- | --- | --- | --- | --- | --- |

## Open Questions

> Populate during working sessions.

## Parked Or Rejected Ideas

> Record ideas here only after discussion, so they are not repeatedly
> re-inferred from historical plans.
