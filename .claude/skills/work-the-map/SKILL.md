---
name: work-the-map
description: Work through an existing Wayfinder map in this repository — load the map, take the frontier ticket, claim it, resolve it, record the resolution. Use when continuing work on a wayfinder:map issue, when asked what the next unstarted item is, or on a cold start that routes to the tracker. Does not chart new maps and stops at any HITL ticket.
---

# Work the map

This is the **AFK half** of `wayfinder/SKILL.md`, reachable by an agent.

`/mattpocock-skills:wayfinder` is user-invoked: it carries
`disable-model-invocation: true`, so an agent cannot start it. That is deliberate
— charting a map is workflow-defining and belongs to the human. But *working
through* an existing map is what this repository's cold start already does every
session, and gating that on a typed command means the workstyle cannot hold
without intervention.

So this skill covers `wayfinder/SKILL.md` § **Work through the map** only, and
refuses the rest.

**If the maintainer has typed `/mattpocock-skills:wayfinder`, prefer it.** It is
the full protocol; this is the reachable subset.

## Hard stops

Three, and none is negotiable.

1. **Never chart.** Do not name a destination, do not run breadth-first fog
   mapping, do not create a new map. If the work needs charting, say so and stop.
2. **Never resolve more than one ticket in a session** — research tickets
   excepted. `wayfinder/SKILL.md:105`.
3. **Stop at any HITL ticket.** `wayfinder:grilling` and `wayfinder:prototype`
   are human-in-the-loop and resolve only through live exchange. *"The agent
   never stands in for the human's side of it — a grilling agent that answers
   its own questions has broken this"* (`wayfinder/SKILL.md:75`). Claim it if you
   like, then hand back with the question stated. Do not answer it yourself.
   `wayfinder:research` is AFK. `wayfinder:task` is either — read it and judge;
   if it turns on a decision the maintainer has not made, treat it as HITL.

## The loop

1. **Load the map** — the low-resolution view, not every ticket body. Read the
   body *and* the newest comments and trust neither alone; see
   `docs/agents/issue-tracker.md` § *Deliberate divergence: map state lives in
   the newest comments*. Both surfaces have been stale here.

2. **Compute the frontier yourself. Do not take a comment's word for it.**
   Open children in map order, excluding any with an open blocker or **any**
   assignee; the first remaining is the frontier. A map comment on this
   repository named the wrong frontier for eight days. Recompute from the
   tracker every time.

3. **Claim it before any work** — assign it, and declare `**Owned paths:**` in
   the ticket body *before* a branch exists. Concurrent sessions run here; the
   tracker is the only surface all of them read. See
   `docs/agents/issue-tracker.md` § *Claim Before Branching*.

4. **Resolve it.** Zoom as needed — fetch full bodies of related or closed
   tickets on demand, and invoke the skills the map's `## Notes` names. For
   implementation work use the `implement-ticket` skill.

5. **Record the resolution.** Post the answer as a durable resolution comment
   **first**, then close an eligible non-code ticket, then append a pointer to
   the map's `Decisions so far`. Implementation tickets stay open for
   merge-driven closure. A ticket here was once closed with an empty record;
   the comment is not optional, and declining the merge-closure route does not
   exempt you from it.

6. **Update the map body**, and pair the edit with a companion comment. Fold
   comment-recorded state changes back into the body so the two do not drift.
   Graduate any fog the answer made specifiable into fresh tickets, clearing
   each graduated patch from the fog section so it lives in exactly one place.

## Tracker specifics

Everything tracker-shaped — labels, native sub-issues, native dependency edges,
the exact frontier query — is in `docs/agents/issue-tracker.md` §
*Wayfinding operations*. Read it; do not reconstruct it from memory.

Two mechanical traps that have each cost a session here:

- Creating a sub-issue needs the child's **numeric database id**, not its issue
  number, and `gh api ... -F sub_issue_id=<id>` with a capital `-F`. Lowercase
  `-f` sends a string and returns 422.
- Before publishing any body, comment, or commit message, scan it for a closing
  keyword near a `#`-prefixed number. GitHub's parser is lexical and ignores
  negation. See `CLAUDE.md`.
