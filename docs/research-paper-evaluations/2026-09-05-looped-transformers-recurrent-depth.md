---
paper_title: "Recurrent Depth / Looped Transformers (OpenAI Astra explainer video and the primary looped-model literature)"
authors:
  - "Sebastian Raschka (video and companion blog post)"
  - "Primary literature: Dehghani et al. 2018; Yang et al. 2023; Geiping et al. 2025; Saunshi et al. 2025; Bae et al. 2025; Zhu et al. 2025; Nanbeige 4.2 report 2026; Looped SSMs 2026"
paper_date: "2026-09"
evaluated_on: "2026-09-05"
source: "https://www.youtube.com/watch?v=KT4n-z_4QJU (auto-generated English transcript pulled 2026-09-05 with youtube-transcript-api) plus the primary papers listed under Intake"
status: "evaluated"
decision: "defer"
primary_landing_zone: "Model"
data_gate: "clear"
recommended_next_action: "Do not schedule a standalone looped-transformer run. Carry a config-gated looped cross-sectional block as one optional arm into the next pre-registered trunk ablation, after the graph-specification map (#157) has produced its reusable paired harness."
github_issue_urls: []
---

# Research-to-Implementation Brief: Recurrent Depth / Looped Transformers

## Intake

### Source and provenance

- Video: "OpenAI Astra and Recurrent Depth / Looped Transformers", Sebastian Raschka, about 28 minutes.
  https://www.youtube.com/watch?v=KT4n-z_4QJU
- Transcript: 748 auto-caption segments, about 4,700 words, pulled on 2026-09-05 with
  `youtube-transcript-api` (English auto-generated track; a manual `en-US` track was
  listed but the fetch returned the generated one). The transcript is third-party spoken
  content and is not committed to the repo; a copy was handed to the maintainer.
  Caption artifacts to be aware of: "Non-Bayesian" / "non-beige" is Nanbeige, "KB cache"
  is KV cache, and the Universal Transformer paper is dated "2008" in speech but is 2018.
- Companion post: https://sebastianraschka.com/blog/2026/openai-astra-looped-transformers.html
- Press context: Fortune, 2026-09-03, on the safety debate the rumour triggered.
  https://fortune.com/2026/09/03/reports-openais-astra-model-uses-a-new-more-efficient-ai-architecture-alarms-ai-safety-experts-who-worry-the-method-makes-models-harder-to-control/

This is not a finance paper, so the intake helper was not used. The rest of the
translation workflow still applies: the question is whether the mechanism transfers to a
specific MCI-GRU surface under the repo's invariants and evidence rules.

### What the video argues, paraphrased with timestamps

- 00:01 to 02:20. The Information reported that OpenAI's forthcoming Astra model uses
  "recurrent depth" or "looped transformers". OpenAI has not confirmed it. The report
  frames the technique as processing the same text several times and as obscuring some
  of the model's reasoning.
- 02:49 to 07:32. Concrete example: Nanbeige 4.2, a 3B open-weight model, runs its
  22-layer block twice. The network behaves like a 44-layer model while storing only 22
  layers of weights. Training and inference compute roughly double; weight storage does
  not.
- 08:00 to 10:17. Nanbeige's report says training with the loop from scratch beats
  retrofitting a loop onto a trained model, because a trained network depends on the exact
  computation it saw in training. Two passes gave the best trade-off, retaining roughly
  75 percent of token efficiency; more passes gave diminishing returns at higher cost.
- 10:47 to 11:42. The KV cache cannot be shared across passes even though weights are
  shared, because activations differ between passes; halving the cache hurt quality.
- 12:09 to 19:36. Mixture-of-Recursions (Bae et al. 2025) adds a router that decides
  per token how many passes it takes, in the spirit of mixture-of-experts routing. In
  the paper's plot the vanilla model slightly beats the plain recursive model; the routed
  model beats both, except at the smallest model size where vanilla is best, and the gap
  widens with size.
- 19:36 to 23:13. Rebuttal of the "obscured reasoning" claim: more effective depth does
  not by itself hide chain of thought. Larger models also need fewer visible reasoning
  tokens, and nobody calls that obscuring.
- 23:13 to 24:39. This is not recursive self-improvement. The presenter's summary is
  that looping is "more scaling than changing the architecture".
- 24:39 to 26:58. Lineage: the Universal Transformer (2018) repeats one layer T times;
  Nanbeige repeats a 22-layer stack once; Geiping et al. (2025) add re-injection of the
  embedded input at every iteration, which the presenter reads as dense residual wiring.

### Primary literature consulted

| Source | What it establishes | Why it matters here |
| --- | --- | --- |
| Dehghani et al. 2018, Universal Transformers. https://arxiv.org/abs/1807.03819 | Weight-tied depth with adaptive computation time. | The origin of every variant below. |
| Yang et al. 2023, Looped Transformers are Better at Learning Learning Algorithms. https://arxiv.org/abs/2311.12424 | On in-context regression tasks a looped model matches a standard transformer with under 10 percent of the parameters; loops emulate iterative algorithms such as gradient descent. Trains with sampled loop counts and a truncated backprop window. | The closest analogue to iterative refinement over a cross-section of assets. |
| Geiping et al. 2025, Scaling up Test-Time Compute with Latent Reasoning. https://arxiv.org/abs/2502.05171 | Prelude, recurrent block, coda. State initialised from noise, input re-injected each iteration through an adapter, loop count sampled from a log-normal Poisson, backprop through the last 8 iterations only. Largest gains on GSM8K, MATH, HumanEval; smallest on easy QA. Two failed runs from representation collapse; fixed with sandwich normalisation and a 10x lower learning rate. | Defines the injection design and the stability risks. |
| Saunshi et al. 2025, Reasoning with Latent Thoughts. https://arxiv.org/abs/2502.17416 | A k-layer block looped L times is competitive with a kL-layer model on reasoning tasks; looping helps reasoning more than memorisation. | Defines the k x L versus kL comparison used in the proposed ablation. |
| Bae et al. 2025, Mixture-of-Recursions. https://arxiv.org/abs/2507.10524 | At 135M parameters the plain recursive model underperforms vanilla (NLL 2.808 vs 2.782 at two passes, worse at three), attributed to a "recursive capacity bottleneck"; the gap closes at 360M and above. | The strongest documented negative result for small models. |
| Zhu et al. 2025, Ouro. https://arxiv.org/abs/2510.25741 | 1.4B and 2.6B looped models match much larger models; the abstract attributes the gain to knowledge manipulation rather than capacity. | Confirms the effect is about computation, not storage. |
| Nanbeige 4.2 report, 2026. https://arxiv.org/html/2607.22083v2 | Two-pass loop trained from scratch on 28T tokens; from scratch beats retrofit; two passes is the sweet spot. | The video's worked example. |
| Looped SSMs, 2026. https://arxiv.org/abs/2605.16048 | On six UEA time-series classification sets a 6-layer state-space model built from 1, 2 or 3 unique blocks matches or beats the 6 independent-layer baseline, with 1 to 6 percentage points higher accuracy on several sets and mixed results on others, over 5 seeds. Loop count fixed at 6; no inference-time loop variation; no retrofit. | The only small-model time-series evidence found. Supports the optimisation-bias reading. |
| Li and Zhang 2026, DeepLoop. https://arxiv.org/abs/2607.13491 | Residual parameterisation must stay stable as the same block is revisited many times. | Norm growth is a real design constraint. |
| Convergence Selection in Weight-Tied Looped Transformers, 2026. https://arxiv.org/abs/2607.20594 | Whether extra test-time loops help depends on the training contract; a halting rule follows from it. | Test-time loop extrapolation is not free. |
| What Makes Looped Transformers Perform Better, 2025. https://arxiv.org/abs/2510.10089 | Attributes looped gains to loss-landscape geometry that favours exploration. | Mechanistic support for the optimisation-bias reading. |
| Gu et al. 2020, Implicit Graph Neural Networks. https://proceedings.neurips.cc/paper/2020/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf and the JMLR 2025 comparison https://www.jmlr.org/papers/volume26/22-0459/22-0459.pdf | Weight-tied message passing to a fixed point with input injection captures long-range graph structure while limiting over-smoothing, at a fixed-point-solve cost. | The graph-stream version of the same idea. |

## Mechanisms

1. **Weight-tied depth as a capacity lever.** Apply one block of weights L times so the
   network has depth kL with k layers of storage. Benefits are weight-memory savings and,
   in language models, a modest quality gain per parameter. Costs are roughly L times the
   compute of the untied k-layer model, the requirement to train with the loop from the
   start, and diminishing returns beyond two passes. This is the mechanism the video is
   mostly about.

2. **Latent iterative refinement with input injection.** Keep a latent state, repeatedly
   update it with the same block while re-reading the original input, and read the answer
   off the final state. Loop count can be sampled during training and varied at test time,
   which turns loops into a test-time-compute dial. This is what Geiping, Yang and Saunshi
   study, and it is the only one of the three mechanisms with a plausible story for stock
   ranking: iterative refinement of relative scores across a cross-section resembles
   iterative de-noising or neutralisation, which is what looped models are shown to
   emulate on in-context regression.

3. **Depth-wise parameter sharing as an optimisation inductive bias in small models.**
   Tying weights across depth restricts the hypothesis class, yet the Looped SSM and
   loss-landscape papers find the tied model often trains to a better optimum than the
   untied model that contains it. This is the mechanism most likely to matter at MCI-GRU's
   scale, and it is also the one contradicted by Mixture-of-Recursions at 135M
   parameters, so it has to be tested rather than assumed.

Empirical choices, not mechanisms: loop count (2 in Nanbeige, up to 32 in Huginn, 6 in
Looped SSMs); loop-count sampling distribution; truncated backprop window; adapter
versus additive injection; sandwich versus pre-norm; learning-rate reduction;
per-token routing; KV-cache policy. Routing and KV-cache policy have no analogue in a
non-autoregressive per-date scorer.

## Data Readiness Gate

| Required input | Status |
| --- | --- |
| Time-series windows, graph features, labels, masks | Already available through `prepare_data` and `combined_collate_fn`; no change. |
| Training loop, ensemble, evaluation | Already available; a looped block is a drop-in `nn.Module`. |
| PIT masked-panel evaluation years 2022 to 2025 | Already available and used by the frozen recipe. |
| Paired-arm evaluation harness with HAC and multiplicity control | Being built under map #157 (tickets #181, #183, #185). Reusable once landed. |

Gate result: **clear** for data. The binding constraints are compute and validation
power, which are handled under Feasibility Opinion. No proxies are needed.

## MCI-GRU Landing Zone Ranking

### Where the model's parameters and time actually go

Measured on this worktree with the `configs/config.yaml` model block, 23 input features
(the frozen `with_momentum` set plus current-only regime), `edge_feature_dim=4`, and
frozen-recipe batch shapes `(B=32, N=500, T=10)`. CPU, forward plus backward, rough.

| Module | Parameters | Share |
| --- | ---: | ---: |
| Temporal encoder A1 (`MultiScaleTemporalEncoder`, `gru_attn`) | 6,338 | 7% |
| Correlation GAT A2 (`GATBlock`) | 4,512 | 5% |
| Latent learner B1/B2 (`MarketLatentStateLearner`) | 10,496 | 12% |
| Cross-stock `SelfAttention` | 49,664 | 56% |
| Final `GATBlock` | 17,544 | 20% |
| Projections and LayerNorms | 896 | 1% |
| Total | 89,450 | |

| Variant (scratchpad prototype, not repo code) | Total params | Step time | Masked nodes stay zero |
| --- | ---: | ---: | --- |
| Current trunk (single untied attention pass, no residual, no LayerNorm) | 89,450 | 1.22 s | yes |
| `use_self_attention=false` reference | 39,786 | 1.07 s | yes |
| Looped block with adapter injection, L=1 | 122,602 | 1.13 s | yes |
| Looped block with adapter injection, L=2 | 122,602 | 1.21 s | yes |
| Looped block with adapter injection, L=3 | 122,602 | 1.37 s | yes |
| Looped block with adapter injection, L=4 | 122,602 | 1.41 s | yes |
| Looped block without injection, L=3 | 89,706 | 1.34 s | yes |

Three readings. First, the cross-stock attention block is the largest module by
parameters but a small share of step time, so looping it three times costs about 12
percent more per step; the GRU over 16,000 sequences and the two GATs dominate. Second,
the adapter that concatenates state and input adds 33K parameters; the additive variant
adds 256. Third, an untrained looped block's output norm grew from 112 at L=1 to 164 at
L=6 and 460 at L=12, so test-time loop extrapolation is not a free dial and needs
pre-norm plus a training contract that includes the loop counts used at inference.

### Mechanism 2, latent iterative refinement: primary landing zone

**Primary: the cross-stock `SelfAttention` block in the trunk.**

Repo evidence:

- `mci_gru/models/trunk.py`: `StockPredictionModel.forward` reshapes `z` to
  `(B, N, 4*align_dim)`, applies `self.self_attention` exactly once, and replaces `z` with
  its output. There is no residual connection and no LayerNorm around the block; the
  `ln_z` and `drop_z` modules run before it.
- `mci_gru/models/attention.py`: `SelfAttention` is a single-head Q/K/V mixing layer
  with an optional four-slot group-type embedding and a stock-mask path that renormalises
  attention over active names. It has no feed-forward sublayer.
- `mci_gru/models/graph.py`: the final `GATBlock` only sees correlation-graph
  neighbours, so the attention block is the one place where every stock on a date
  interacts with every other stock before scoring.

Why here: the block is already a per-date transformer layer over the cross-section of
assets, with 500 to 700 tokens and a 128-wide embedding, so the attention cost of
looping it is trivial. Iterative refinement of relative representations across the
cross-section is the direct analogue of the in-context iterative algorithms looped
models are shown to learn. The contract that the four streams sit in `[A1, A2, B1, B2]`
order is preserved because the loop wraps the block rather than changing its input.

**Secondary: weight-tied hops in the final `GATBlock`.** Reusing the second GAT layer
for extra message-passing hops with input injection is the implicit-GNN version of the
same idea and would let the score head see beyond two hops of the correlation graph.
It is secondary because the 0.8-threshold graph is sparse and clustered, so extra hops
risk over-smoothing within clusters, and because map #157 is still deciding what the
graph is; a depth arm on top of an undecided graph would confound both questions.

**Rejected for now:**

- Temporal encoder A1. With `his_t=10` there are ten tokens; the GRU is already a
  recurrence over them and the transformer variant is two layers over ten tokens.
  Issue #131 also shows `gru_hidden_sizes` is reinterpreted per encoder, so encoder
  comparisons are not capacity-matched until that is fixed. The long-history results
  (2026-05-18) moved returns through `his_t`, not through encoder depth.
- Latent learner B1/B2. Alternating cross-attention between stocks and the 32 learned
  latent vectors would be a Perceiver-style redesign, not a loop.
- Whole-trunk prelude, recurrent block, coda in the Geiping layout. This has the largest
  blast radius (stream-order contract, type embedding, masks, AMP stability, checkpoint
  compatibility) and should not be attempted until the block-level test shows anything.

### Mechanism 1, weight-tied depth for parameter efficiency: rejected as a motivation

The model has 89,450 parameters. Weight storage, KV cache, and load time are not
constraints anywhere in training, Colab evaluation, or paper-trade inference. The only
cost that matters here is compute per step, which looping increases. The mechanism
survives only as the k x L versus kL comparison arm inside the ablation below.

### Mechanism 3, optimisation inductive bias: primary is the same block

Primary: the same cross-stock block, tested as a four-arm block-depth ablation that
isolates residual wiring, depth, and tying separately. Secondary: a notebook diagnostic
over ensemble members, because seed-paired per-member validation IC was the most
sensitive statistic in the graph re-analysis (#179) and is where an optimisation effect
would show first.

## Invariant Check

- **No lookahead.** A looped block consumes only the per-date tensors already in the
  batch; it introduces no dates, windows, shifts, or external series. Unaffected.
- **Train-only normalisation and reference statistics.** Unaffected; `run_metadata.json`
  does not change.
- **Graph timing and `GraphSchedule`.** Unaffected for the primary landing zone. The
  secondary GAT-hop variant would still consume the same collated edges.
- **Label embargo.** Unaffected.
- **9-tuple collate contract and `edge_feature_dim`.** Unaffected.
- **PIT masked-panel masks.** The loop must re-apply `stock_mask` after every iteration
  so inactive union nodes neither send nor accumulate state. The prototype does this and
  the masked tail stayed exactly zero at every loop count. A mutation-checked test in the
  style of `tests/test_pit_masked_panel.py::test_self_attention_mask_prevents_inactive_node_influence`
  is required.
- **Ensemble averaging.** Unaffected; each member gets its own looped block.
- **Paper-trade frozen checkpoint rule.** New config keys must default to the legacy
  block so existing `config.yaml` files in checkpoint folders still build the same module
  graph through `create_model`; `ModelConfig.to_dict` must serialise the new keys so new
  runs are auditable. Paper-trade inference stays on frozen artifacts and must not be
  touched by this work.
- **Numerical stability under AMP.** The frozen recipe trains with `use_amp=true`. A
  looped residual block needs pre-LayerNorm inside the loop and a fixed loop count that
  matches inference. Norm growth is measurable and should be asserted finite under
  autocast in tests.
- **Backtest fairness.** Unaffected; the change is upstream of prediction files.

## Feasibility Opinion

Overall: mechanically an easy win, evidentially a long shot.

The problem is not whether MCI-GRU can loop a block; the prototype shows it can in a
few dozen lines at a 12 percent step-time cost. The problem is that the mechanism's
headline benefit, capacity without weights, is irrelevant at 89K parameters, and the
benefit that could apply, a better optimum through iterative refinement, has to clear
an evidence bar the repo has already measured. The paired re-analysis for map #157 puts
the minimum detectable mean daily IC difference at 0.0035 over four pooled PIT test
years for the shipped configuration. Yearly mean best validation IC in the full PIT run
sits between 0.0034 and 0.0254. The small-model evidence for looping is one to six
percentage points of accuracy on classification benchmarks and a negative result at the
smallest language-model scale. A looped-block effect that is real but smaller than 0.0035
would land in the "undecidable at this universe and horizon" branch of the #181 outcome
map, which is the most likely outcome and is worth knowing before spending GPU hours.

That is why the decision is defer rather than pursue or reject. Defer means: build the
block behind a flag when a trunk ablation is next being run anyway, pre-register it as
one arm of that run under the paired protocol, and let the harness say whether it
separates. It does not mean a standalone looped-transformer study.

Per-slice opinions are given with each slice below. Budget reference: the 2026-05-16
full PIT run averaged about 20 minutes per fold for 20 members on Colab; four arms over
four folds is roughly 5 to 6 GPU hours before the 12 percent looped-arm surcharge.

## GitHub-Ready Slices

### Slice 1. Model: config-gated looped cross-sectional block

Category: Model issue.

Problem: the cross-stock `SelfAttention` block is applied once with no residual path and
no normalisation, so neither depth nor weight tying can be tested on the one surface
where the whole cross-section interacts.

Proposed scope:

- Add `LoopedCrossSectionalBlock` in `mci_gru/models/attention.py` that wraps the
  existing `SelfAttention` and computes, for a fixed loop count L,
  `s_0 = 0`, `h_i = adapter([s_i ; e])` (or `h_i = s_i` when injection is off),
  `s_{i+1} = s_i + SelfAttention(LayerNorm(h_i), stock_mask)`, re-applying the stock
  mask after each iteration, and returns `s_L`.
- Add `ModelConfig` fields `cross_section_block: "legacy" | "looped"` (default
  `legacy`), `cross_section_num_loops: int` (default 1), and
  `cross_section_input_injection: bool` (default true), with `__post_init__`
  validation and `to_dict` serialisation.
- Wire the fields through `create_model` with legacy-safe `config.get` defaults so old
  checkpoint `config.yaml` files build identical modules.
- Keep the `[A1, A2, B1, B2]` stream order and the group-type-embedding contract.

Acceptance criteria:

- With defaults, `create_model` builds a module graph whose state-dict keys and
  parameter count equal today's, and existing checkpoints load.
- With `cross_section_block=looped`, output shape is `(B, N)` for L in {1, 2, 3, 4},
  gradients reach every parameter of the block, and masked nodes are exactly zero at the
  output for every L.
- Under `torch.autocast` (CPU bfloat16 in tests), outputs are finite at L=4.
- The effective block configuration appears in the run's serialised `config.yaml`.

Suggested tests:

- A mutation-checked mask test: remove the per-iteration mask re-application, confirm
  the inactive-node test fails, restore, confirm it passes.
- A checkpoint-compatibility test that loads a state dict produced by the legacy path
  into a model built from a config lacking the new keys.
- A determinism test that two forwards with the same seed and L agree.

Out of scope: any change to the frozen recipe; loop-count sampling during training;
adaptive exits; the temporal encoder; paper-trade.

Feasibility Opinion: effort easy win; confidence high that it can be built as
specified; rationale, the prototype already ran inside the real trunk at frozen shapes;
main blocker, none for the code, validation cost for any claim.

### Slice 2. Config/experiment: pre-registered block-depth ablation

Category: Config/experiment issue.

Superseded the same day by the trunk-hygiene ablation in
`docs/research/current/MCI_GRU_TRUNK_ARCHITECTURE_OPPORTUNITIES_2026-09-05.md`,
which keeps the residual-block arm (its C1 equals this slice's C1), adds a
market-state gate and capacity-matched widths, and drops the depth and tying arms
until C1 has an outcome. The text below is retained as the record of the original
looped design.

Problem: without a pre-registered arm set, the looped block would be evaluated on a
single year and a single seed, which #167 and #179 have already shown cannot separate
arms from control.

Proposed scope:

- Four arms on the cross-stock block, everything else at the frozen recipe: C0 legacy
  (single pass, no residual), C1 looped block at L=1 (isolates residual plus pre-norm),
  C2 three untied residual blocks (isolates depth), C3 looped block at L=3 (isolates
  tying against C2, the k x L versus kL comparison).
- Hydra presets under `configs/experiment/` for each arm.
- Folds, seeds, pairing, arbiter, multiplicity (m = 3, each of C1 to C3 against C0),
  secondaries, and the three-branch outcome map exactly as ruled on #181, reusing the
  #183 harness once it lands. Member seeds shared across arms; random-number consumption
  held identical so `sd(delta)` measures the block, not the stream.
- Screen at reduced budget for mechanics only; confirm every arm at 20 x 100.

Acceptance criteria:

- Presets compose and pass a one-epoch mechanics smoke on the anchored snapshot data.
- The arm table, arbiter, m, and effect-worth-acting-on are written into the issue
  before any confirm run starts.
- Results are recorded as a dated report under `docs/research/current/` with per-fold
  disclosure and the outcome branch named explicitly, including the undecidable branch.

Suggested tests: a preset-composition test that each arm resolves to the intended
`ModelConfig` fields and to the frozen recipe elsewhere.

Out of scope: promotion into the frozen recipe; temporal-encoder arms (blocked on
#131); graph-depth arms (blocked on #157).

Feasibility Opinion: effort medium; confidence low that any arm clears the 0.0035 MDE;
rationale, the effect sizes in the small-model literature are modest and the repo's
year-to-year dispersion is large; main blocker, validation cost.

### Slice 3. Notebook: loop-count extrapolation and member-dynamics diagnostic

Category: Notebook issue.

Problem: if a looped model is trained at L=3, the literature says extra test-time loops
help only when the training contract supports it, and the prototype shows norm growth
with L. Whether the learned iteration is convergent is cheap to check and informative
regardless of the ablation's headline outcome.

Proposed scope:

- On saved C3 checkpoints, evaluate validation IC and output-norm trajectories at
  inference loop counts 1 to 6 without retraining, per member and for the ensemble mean.
- Plot per-member validation IC by epoch for C0 versus C3 to see whether tying changes
  training dynamics, which is where the optimisation-bias mechanism would show.
- Validation only; test predictions are not touched.

Acceptance criteria: a generated notebook under the repo's notebook-generation
conventions, artifacts exported to Drive, and a short dated note recording whether the
iteration converges (norms and IC plateau) or diverges past the training loop count.

Suggested tests: the notebook contract test pattern already used for PIT notebooks.

Out of scope: any promotion decision; loop-count sampling during training.

Feasibility Opinion: effort easy win once slice 2 checkpoints exist; confidence medium;
rationale, it reads existing artifacts; main blocker, depends on slice 2.

### Slice 4. Model, gated: weight-tied final-GAT hops with injection

Category: Model issue.

Problem: the score head sees two hops of the correlation graph; a weight-tied third and
fourth hop with input injection is the graph-stream analogue of looping and is the only
way to test depth on the graph path without adding parameters.

Proposed scope: reuse `gat2` for `final_gat_num_hops - 1` extra hops with a residual
and injected node input, behind a default-off flag; same mask re-application rule.

Acceptance criteria and tests: as slice 1, plus an over-smoothing diagnostic (mean
pairwise cosine similarity of node states by hop) on the anchored snapshot graph.

Out of scope: correlation GAT A2; any change while map #157 is open.

Feasibility Opinion: effort medium; confidence low; rationale, over-smoothing on a
sparse clustered graph is likely and the graph specification is itself under decision;
main blocker, code complexity and the #157 dependency. Do not start before slice 2 has
an outcome.

## ADR Candidates

None now. Every proposed change is config-gated and default-off, so it is reversible and
not surprising. An ADR becomes warranted only if a looped block were ever promoted into
the frozen recipe, because that would change compute per inference, checkpoint
compatibility expectations, and the test-time behaviour of paper-trade scoring.

## Rejected Ideas

- A whole-model recurrent-depth redesign (prelude, recurrent block, coda) now. Largest
  blast radius for the least evidence.
- Mixture-of-Recursions style per-stock adaptive depth. The paper itself shows plain
  recursion losing at small scale, and routing adds balancing losses and a second source
  of run-to-run variance to a model whose ensemble already exists to tame variance.
- Retrofitting a loop onto frozen checkpoints. Nanbeige found from-scratch training
  significantly better than retrofit, and the untrained prototype's norm growth agrees.
- Treating test-time loop count as a free compute dial. Without loop-count sampling in
  training, extra loops at inference are outside the training contract.
- Framing looping as "latent reasoning" for return prediction. The gains in the
  literature concentrate on multi-step reasoning tasks; the repo's binding constraint is
  signal-to-noise, not depth.
- KV-cache tricks and per-token routing. There is no autoregression in a per-date
  scorer.

## Open Questions

- Which loop count to confirm: L=2 (Nanbeige's sweet spot) or L=3 (the smallest value
  that makes the k x L versus kL comparison non-trivial)? Slice 2 proposes L=3 and
  should be ruled on with the arm set.
- Should the training contract sample L per step (Geiping) or fix it (Nanbeige, Looped
  SSMs)? Fixed is proposed for the first pass because the ensemble already provides
  variance and because sampling complicates pairing across arms.
- What mean daily IC difference on the cross-stock block would change the recipe? This
  is the #181 question again and needs the maintainer's number before the run is
  budgeted.
- Should temporal-encoder arms join the same run once #131 is resolved, so one harness
  answers both the depth and the encoder question? Doing so raises m and the budget.
- Is the 33K-parameter adapter worth its cost against additive injection at this scale?
  A mechanics-only screen at reduced budget can decide this before the confirm run.
- Does `drop_edge` random-number consumption differ between C0 and the looped arms?
  It should not, because the block is downstream of edge dropout, but the pairing check
  from #181 must be run rather than assumed.
