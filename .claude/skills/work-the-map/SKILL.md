---
name: work-the-map
description: Work through an existing Wayfinder map in this repository — load the map, take the frontier ticket, claim it, resolve it, record the resolution. Use when continuing work on a wayfinder:map issue, when asked what the next unstarted item is, or on a cold start that routes to the tracker. Does not chart new maps and stops at any HITL ticket.
---

# Work the map

The **AFK half** of `wayfinder/SKILL.md`: working through an existing map, never
charting a new one. `/mattpocock-skills:wayfinder` is user-invoked and covers
both halves — if the maintainer has typed it, prefer it.

**The loop itself is `docs/agents/issue-tracker.md` § Wayfinding operations.**
Read it; do not reconstruct it from memory. This file adds only the three
refusals below and the routing under them.

## Hard stops

1. **Never chart.** Do not name a destination, do not run breadth-first fog
   mapping, do not create a new map, and do not fan out speculative tickets.
   Note that the tracker doc *teaches* charting mechanics — map body structure,
   child creation, dependency wiring — because a human running `/wayfinder`
   needs them. Being able to follow those steps is not authorisation to. If the
   work needs charting, say so and stop.

2. **Never resolve more than one ticket in a session**, research tickets
   excepted — `wayfinder/SKILL.md` § *Invocation*: *"never resolve more than one
   ticket per session — with the exception of research tickets."*

3. **Stop at any HITL ticket.** `wayfinder:grilling` and `wayfinder:prototype`
   resolve only through live exchange — `wayfinder/SKILL.md` § *Ticket Types*:
   *"the agent never stands in for the human's side of it (a grilling agent that
   answers its own questions has broken this)."*

   Claim it if you like, then **hand back with the question stated**. Do not
   answer it yourself, and do not treat your own reasoning as the exchange.

   `wayfinder:research` is AFK. `wayfinder:task` is either — read it and judge:
   **if it turns on a decision the maintainer has not made, treat it as HITL.**

## Resolving a ticket

Zoom as needed rather than loading everything up front: fetch the full body of a
related or closed ticket **on demand**, not the whole map's history at the
start.

Invoke the skills the map's `## Notes` block names — a map may route its own
tickets at specific skills, and that routing is part of the map, not decoration.

For implementation work, use the `implement-ticket` skill. It carries this
repository's claim-before-branching, TDD, mutation-check and review rules, which
are stricter than the generic loop.
