---
name: blind-ai-reviewers
description: Run the three blind AI reviewers that label a filings sheet (10-K risk pairs) for an answer key, the same way every time. Use when a labels folder needs AI labels, when the owner has no time to label a sheet, or when checking the owner's spot-check labels against an AI majority.
---

# Three blind AI reviewers

The Labelling desk's fixed step for an AI answer key (#291). The owner accepted
three independent AI reviewers as a stand-in for hand labels when their time is
short, provided the reviewers are blind and their answers are checked against a
blind spot-check of the owner's own.

The instructions every reviewer reads are `configs/filings/labels/reviewer_v1.md`,
filled in from the sheet's label set (`configs/filings/labels/*.yaml`). **Do not
add to them in the subagent prompt.** A reviewer that is told what the
experiment is, what Jev or a baseline answered, or what the owner thinks is no
longer blind, and its labels cannot go in a key.

## Steps

1. The sheet exists (`label_desk sheet`), so the labels folder holds
   `sheet.json` and `items.csv`. Write the packets:

   ```
   python -m scripts.data.label_desk reviewers --labels-dir <labels>
   ```

   Each reviewer gets every item, in its own shuffled order, cut into batches
   of 40 so one subagent reads a manageable amount. The script prints one
   `prompt.md` per batch.

2. Start **one fresh subagent per packet**, all in one message so they run side
   by side, with exactly this prompt and nothing else:

   > Read `<absolute path to prompt.md>` and follow it exactly. Open no other
   > file. Do not call any `mcp__hearthbot__` tool.

   Use the general-purpose agent type. Never reuse a subagent across packets
   or reviewers: one reviewer's earlier batch must not colour its next, and no
   reviewer may see another's answers.

3. Combine:

   ```
   python -m scripts.data.label_desk combine --labels-dir <labels>
   ```

   It checks every answer file against the label set (yes/no only, skipped
   questions left empty, every item answered once) and writes `ai_labels.csv`
   (each reviewer's category, the majority, whether all three agreed) and
   `ai_summary.json`. An item with no majority has an empty `ai_label`; it
   goes to the owner, never to a fourth reviewer.

4. When the owner has spot-checked part of the sheet on the labelling page,
   read their labels back from the page's store (ArtifactData `list` of
   `sheets/<sheet id>/labels` with `out_dir`), then:

   ```
   python -m scripts.data.label_desk collect --labels-dir <labels> --dump <out_dir>
   python -m scripts.data.label_desk agree --labels-dir <labels>
   ```

   `agreement.json` gives the share of items where the owner and the AI
   majority agree, its 95% interval, Cohen's kappa and each disagreement.
   Report the disagreements to the owner as pairs to look at, with both texts,
   not as the owner's mistakes.

## When an answer file is rejected

`combine` names the file and the item. Move that file aside as
`answers.rejected.csv` (keep it; it is evidence) and give the same packet to a
new fresh subagent. Do not edit a reviewer's answers by hand, and do not re-run
a packet whose answers passed because you disagree with them.

## What this step never does

- Reads a Jev answer, a baseline answer or anything under a `sealed/` or
  `jev/` folder. The desk is on the blind side of the wall.
- Changes `reviewer_v1.md` or a label set that already has answers against it.
  A changed instruction is a new file (`reviewer_v2.md`), and `run.json`
  records the hash of the one used.
