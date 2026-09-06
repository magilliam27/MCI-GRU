# MCI-GRU Trunk Architecture Opportunities

Date: 2026-09-05

Status: research map with mechanics-level diagnostics. The diagnostics in
section 3 are scratchpad measurements on synthetic inputs and a short CPU smoke
on the anchored 2019 snapshot universe; they prove information-flow facts about
the trunk, not model performance. Nothing here changes a default, a recipe, or
a config file.

Purpose: answer "what kind of architecture change would make sense for this
model" from the inside out. The looped-transformer brief
(`docs/research-paper-evaluations/2026-09-05-looped-transformers-recurrent-depth.md`)
concluded that depth is not the lever at 89K parameters. This map asks what is,
by measuring how information actually moves through the four-stream trunk and
comparing that against what the repo's own ablations and the cross-sectional
literature say matters.

Repo anchors reviewed: `AGENTS.md`, `docs/ARCHITECTURE.md`,
`docs/ARCHITECTURE_REVIEW.md`, `docs/ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md`,
`docs/research/current/MCI_GRU_PROGRAM_MAP_2026-06-19.md`,
`docs/research/current/MCI_GRU_TOP_UNIVERSITY_RESEARCH_SCAN_2026-06-21.md`,
`docs/research/current/GRAPH_SPECIFICATION_ABLATION_2026-09-01.md` and
`GRAPH_PAIRED_REANALYSIS_2026-09-02.md` (both read from their branch refs at the
time of writing), the phase-2 and phase-3 trunk plans under
`docs/agent_references/cursor/plans/`, `mci_gru/models/*.py`,
`mci_gru/training/losses.py`, `mci_gru/training/trainer.py`,
`references/2410.20679v3.txt` (the MCI-GRU paper), and issues #131, #157,
#164, #167, #179, #181.

## 1. Summary and ranked recommendations

**Both of the trunk's attention blocks are broken, in different ways, and each
has a specific fix given in full with code: Fix A in section 4.1, Fix B in
section 4.2.** Beyond those, one width choice and one graph-input choice are
worth revisiting.

> **What the real-universe measurements establish, and what they do not.**
> On the 110-name universe with real data and trained models (section 3.7),
> the cross-stock block destroys 94 percent of the cross-sectional variance and
> Fix A demonstrably stops it doing so. That part is solid and reproducible.
>
> **No IC ordering between the variants is solid.** A first seed showed the
> no-block arm winning by a wide margin and I briefly wrote that up as the
> headline. A second seed reversed it: Fix A went from last to first on test IC
> (0.0104 then 0.0361) while the no-block arm's lead shrank by two thirds. The
> within-variant seed spread reaches 0.026 on test IC, which is larger than
> every between-variant gap and roughly seven times the paired protocol's
> 0.0035 detectable effect. **At two seeds nothing here separates from anything
> else**, and any ordering read off these runs, including one favouring
> deletion, is noise. Section 3.8 has the numbers. This is the same
> variance-dominated regime section 3.2 flagged, now confirmed on the real
> universe, and it is the single most important practical fact in this
> document: architecture choices for this model cannot be evaluated without the
> full ensemble and the paired protocol.

- **Defect 1, the cross-stock block.** The `SelfAttention` block has no residual path,
  no normalisation, and no feed-forward sublayer. At initialisation its
  attention is uniform to three decimals, so it replaces every stock's
  128-dimensional trunk vector with the same cross-sectional mean: cosine
  similarity across stocks goes from 0.86 to 1.000 and the cross-sectional
  variance share from 0.14 to 0.000. The score head therefore starts blind to
  stock-level features. A residual connection removes the collapse at zero cost
  (section 3.1). This is the textbook rank-collapse failure of attention without
  skip connections. On real validation data the picture holds after training:
  ten epochs in, the block's attention is still uniform to within 0.06 nats,
  it discards 99 percent of the cross-sectional variance, and the score head
  reaches its validation IC through a rank-1 residue (section 3.3). Even at six
  times the learning rate it sharpens only toward a date-level pattern and still
  passes 3 percent of the variance at rank 1.1 (section 3.3.1). The block is
  functioning as a date-adaptive linear projection with a large shared offset,
  not as attention between stocks. The sharpest single number: when 300 other
  stocks are perturbed, the block moves a given stock's raw score by 0.336 but
  only 0.011 of that survives the per-date centring the IC loss applies
  (section 3.4). Fifty-six percent of the model's parameters are computing a
  quantity the objective discards. **Fix A (section 4.1)** wraps the block in a
  pre-norm residual so its output corrects `z` instead of replacing it, which
  restores that number to 0.534 for 256 extra parameters and no step-time cost.
  What the fix does **not** come with is evidence that it helps. Sections 3.2
  and 3.5 show it trains no better than the shipped block at the recipe rate,
  and the two-seed spread runs mildly against the dispersion-reduction argument
  I had pre-registered for it. Its case is that it is correct, that it is free,
  and that it is stable where the shipped block is not at higher learning rates.
  Whether the block earns its place at all is a separate question the ablation
  has to settle, and on current evidence deleting it is the live alternative.
- **The variance finding.** In the same smoke, models that share every initial
  weight except for one residual add landed 0.023 apart in validation IC before
  any training, and ten epochs at the recipe learning rate moved each model by
  only 0.008 to 0.027. The initialisation lottery is as large as the training
  gain over that horizon and an order of magnitude larger than the paired
  protocol's minimum detectable effect. That is the same story the graph
  re-analysis told through per-member IC and per-member output scale, and it
  means every architecture candidate below must be judged on paired,
  seed-shared, multi-fold evidence, never on one run.
- **Defect 2, the cross-attention itself.** This is the "MCI" the model is
  named for, and it cannot see the market. In `MarketLatentStateLearner`, `R1`
  and `R2` are plain `nn.Parameter` tensors and `forward` receives only the
  per-stock vectors, so `B1 = f(A1)` and `B2 = f(A2)` pointwise, with the
  latents frozen after training. The streams the architecture calls "market
  latent states" are a learned codebook: each stock re-expressed as a soft
  mixture of 32 prototypes. That is a per-stock nonlinearity, not a market
  state. It also explains a standing puzzle. The global regime features that do
  describe the date are broadcast identically to every stock, and the IC loss is
  invariant to any per-date constant, so with frozen latents those features have
  no path to the objective at all, which is consistent with regime context
  reading as a "secondary tuning dimension" in the April ablation rather than a
  driver. Two repairs are available. **Fix B (section 4.2)** makes the latents
  data-dependent by gathering them from the date's cross-section before
  broadcasting them back, the Set Transformer induced-set construction; it is
  the principled version and it repairs Defect 1 too when placed in the
  cross-stock position. But section 3.6 reprices it: at 110 names it costs 58
  percent more step time and twice the parameters, and the `O(N*k)` efficiency
  argument that motivated it only bites at 500 names. The cheaper **market
  gate** conditions the inputs on a per-date market vector for a few hundred
  parameters, and at this universe size it is the sensible thing to test first.
- **Structural choice.** The cross-sectional stream A2 is squeezed to four
  dimensions (`output_gat1=4`) before being projected back up, and it sees only
  the last day's raw features. Its effective rank at init is 3. The
  graph-specification ablation found that removing every correlation edge does
  not hurt, and the incumbent threshold graph isolates about three quarters of
  names, so for most stocks A2 is a two-layer MLP on one day of features
  squeezed to four numbers.

Ranked recommendations, best-supported first:

| Rank | Change | Why | Cost | How to test |
| --- | --- | --- | --- | --- |
| 0 | **Delete the cross-stock block**: `model.use_self_attention=false` | Not a change to write, a flag that already exists. It is first because it is free to test and because three structural arguments favour it: the graph ablation's control-first result, the block's 56 percent parameter share, and its measured 94 percent variance destruction. Its smoke lead did **not** replicate on a second seed (section 3.8), so this is a cheap thing to test, not a finding | **-49,664 params**; faster | Arm C5, in the first wave |
| 1 | **Fix A, section 4.1.** Residual pre-norm cross-stock block: `z + Attn(LN(z))`, four heads, optional FFN | The right repair *if* the block is kept. Stops the variance destruction (0.433-to-0.024 becomes 0.403-to-0.329 on real data) and is stable at learning rates where the shipped block spikes then decays. But it lands below the shipped block on test IC in the one real-universe smoke, so it is a correctness fix without demonstrated value | +256 params; step time inside the noise | Arm C1 |
| 2 | **Market gate.** MASTER-style softmax gate on the input features, built from a per-date masked market vector | The cheap route to Defect 2. Conditions inputs on the market for a few hundred parameters, where Fix B conditions representations for 137,000. At 110 names the cheap route is the sensible first test | A few hundred params; one masked mean per batch | Arm C2 |
| 3 | **Fix B, section 4.2.** Data-dependent latents: gather the 32 latents from the date's cross-section, then broadcast back (Set Transformer ISAB) | The principled repair of the cross-attention, and in the cross-stock position it repairs Defect 1 too. **Repriced down by section 3.6**: at 110 names its efficiency argument is gone and it costs 58 percent more step time and 2x parameters, with no smoke benefit | +136,960 params (89K to 177K); +58% step time at 110 names | Arm C3, only if C2 shows promise |
| 3 | Capacity-matched widths: raise `output_gat1` toward `hidden_size_gat1`, drop the 4-to-32 re-projection, and fix the `gru_hidden_sizes` reinterpretation from #131 so encoders are comparable | Removes a rank-3 bottleneck on the only cross-sectional stream; makes encoder ablations mean what they say | Tens of thousands of parameters, still tiny | Arm C4, only after #131 is resolved |
| 4 | Feed the graph stream a temporal summary instead of one day of raw features (the roadmap's graph-input summariser) | The graph stream is the only one that sees neighbours and it sees them through a keyhole | Small | Blocked on map #157 deciding what the graph is |
| 5 | Initial-residual skip into the score head (GCNII style) and `GATv2Conv` in place of `GATConv` | The final GAT is effectively an MLP for isolated names; a skip from `z` protects against over-smoothing if the graph ever densifies; GATv2 is a one-line expressivity fix | Trivial | Bundle with whichever graph arm map #157 keeps |
| Defer | Looped or tied depth, Perceiver-style latent rewrite, multi-horizon heads, per-stock adaptive depth | Depth is not the constraint; each is a redesign, not a fix | | See the looped brief |

What this map does not recommend: replacing the four-stream trunk, adding
transformer depth on the ten-step temporal path, or any change to the frozen
recipe before the paired protocol has run. The cross-sectional literature is
consistent on this point: on Qlib's Alpha158 CSI300 benchmark the attention
models sit at IC 0.026 to 0.036 while LightGBM is at 0.045 and a tree ensemble
at 0.052, and Gu, Kelly and Xiu found returns to depth deteriorate quickly on
monthly panels. Sophistication buys little; conditioning, hygiene and ensembles
buy more.

## 2. What the trunk does today

Measured on this worktree with the `configs/config.yaml` model block, 23 input
features, `edge_feature_dim=4`, and frozen-recipe shapes.

| Module | Parameters | Share | Information-flow fact |
| --- | ---: | ---: | --- |
| A1 `MultiScaleTemporalEncoder` (`gru_attn`) | 6,338 | 7% | Output width 10 (`gru_hidden_sizes[-1]`), then `proj_temporal` 10 to 32. Effective rank at init 4.3. Under `gru_attn` both GRU layers are width 10, not 32 then 10 (#131). |
| A2 `GATBlock` | 4,512 | 5% | Input is one day of raw features; output width 4 (`output_gat1`), then `proj_cross` 4 to 32. Effective rank at init 3.1. |
| B1/B2 `MarketLatentStateLearner` | 10,496 | 12% | Query is the stock vector, keys and values are 32 fixed learned latents. Output depends only on that stock's own A1 or A2; no date-level information enters. |
| Cross-stock `SelfAttention` | 49,664 | 56% | Single head over 128 dims, scale 128^-0.5, no residual, no LayerNorm, no FFN. Replaces `z` with its output. |
| Final `GATBlock` | 17,544 | 20% | Two GAT layers to a scalar. With the shipped threshold graph, 75 to 78 percent of names have no edges, so for them this is a two-layer MLP on `z`. |
| Projections and LayerNorms | 896 | 1% | |
| Total | 89,450 | | |

Two loss facts shape everything below:

- `ICLoss` centres predictions and labels within each date. Any function of
  the date alone, including the broadcast regime features, cannot move the loss
  except through interactions with stock-specific inputs.
- The 20-member ensemble is the prediction contract, and seed-paired member IC
  was the sharpest statistic in the graph re-analysis. Variance reduction is
  where the repo's evidence has found gains; capacity is not.

Two repo results shape the graph-side recommendations:

- The graph-zeroed control ranked first on the arbiter in the graph
  specification ablation (0.04238 vs 0.04191 for the shipped graph), and the
  paired re-analysis could not separate any arm from it. Whatever A2 and the
  final GAT contribute today, it is not coming from edges.
- The minimum detectable mean daily IC difference over four pooled PIT test
  years is 0.0035 for the shipped configuration. Any change proposed here has to
  be judged against that bar, not against a single-year point estimate.

## 3. Diagnostics

> **Scale correction, recorded 2026-09-05 after the maintainer flagged it.**
> Sections 3.1 to 3.5 were measured at roughly 500 names: synthetic batches at
> `N = 500`, and smokes on the anchored 2019 snapshot universe (472 names, no
> PIT masking, 1,258 training days). **That is not the universe this project
> works on.** The real setting is `configs/data/gics_top10_110_2016.yaml`:
> GICS sector top-10 by market cap, about **110 admissible names per session**
> on a **201-node PIT union axis** in masked-panel mode, 2,012 training days
> from 2016, and 238 test days in 2025 (the same 238 days the graph ablation
> and its 0.0035 detectable-effect bar were computed on).
>
> Section 3.6 redoes the measurements at the correct scale. **Two conclusions
> change and one strengthens**, so read 3.6 before acting on 3.1 to 3.5. The
> earlier sections are kept because the trained-model probes in 3.3 have no
> 110-name counterpart yet, and because the contrast between the two scales is
> itself the finding: this architecture's attention behaves quite differently
> at 110 names than at 500.

### 3.1 Initialisation: the cross-stock block collapses the cross-section

Scratchpad script on synthetic inputs at frozen-recipe shapes (`B=8`, `N=500`,
`T=10`, `F=23`), clustered synthetic graph, labels with a weak linear signal,
`ICLoss`, averaged over three seeds. `mean_cos` is the mean pairwise cosine
between stocks within a date; `cs_var_share` is the fraction of a tensor's
variance that lies across stocks; `eff_rank` is the participation ratio of the
centred per-date matrix's singular values.

| Stage | mean_cos | cs_var_share | eff_rank |
| --- | ---: | ---: | ---: |
| A1 after `ln_a1` | 0.844 | 0.156 | 4.3 |
| A2 after `ln_a2` | 0.876 | 0.124 | 3.1 |
| B1 | 0.895 | 0.115 | 13.2 |
| B2 | 0.856 | 0.153 | 12.4 |
| `z` before the block | 0.860 | 0.140 | 7.2 |
| `z` after the shipped block | **1.000** | **0.000** | 4.1 |
| `z` after `z + Attn(z)` | 0.821 | 0.179 | 17.3 |
| `z` after `z + Attn(LN(z))` | 0.817 | 0.183 | 17.3 |

Attention entropy of the shipped block at init is 6.205 nats against a uniform
bound of 6.215 (logit standard deviation 0.175). The block is a cross-sectional
mean. Gradient norms confirm where the optimiser's effort goes: in the shipped
trunk the attention block receives the largest gradient of any module (0.30
absolute, 2.6e-2 relative to its parameter norm), roughly three times what it
receives once a residual path exists (0.09, 7.7e-3), and the correlation GAT's
gradient rises from 0.06 to 0.17 when the block stops masking it.

Why this happens is well understood. With unit-scale inputs, a single 128-wide
head with the standard scale produces logits of standard deviation about 0.2,
which over 500 keys is indistinguishable from uniform, and pure attention
without skip connections converges to a rank-one output (Dong et al. 2021). The
network could in principle learn its way out by sharpening the attention or by
routing through the diagonal. Sections 3.2 and 3.3 show what it does instead
at this budget: it leaves the block near uniform and works through the small
residue the block lets past.

### 3.2 Training dynamics smoke on the anchored universe

Setup. The repo's own `run_experiment` path on
`sp500_2019_universe_data_through_2026.csv` (an Anchored Historical Snapshot
Universe, so mechanics only), `features=with_momentum` with global regime off
(16 features, 472 names), train 2019 to 2023 (1,258 days), validation 2024,
test 2025, pure IC loss on raw 5-day returns, `selection_metric=val_ic`,
shuffled training, `drop_edge_p=0.1`, one model, seed 1729, CPU, no AMP.
Ten epochs at the recipe learning rate 5e-5 with a 40-step warmup and cosine
decay to zero over the 400 steps, so the schedule integrates to about one tenth
of a frozen-recipe run. The cross-stock block was swapped by monkeypatching
`create_model`; nothing in the repo changed. Because module construction order
is identical, the shipped block and both residual variants start from the same
weights; the no-attention arm draws different final-GAT weights.

| Variant | Parameters | Val IC epoch 1 | Val IC epoch 10 | Test avg IC | Test avg rank IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shipped block | 86,762 | 0.0231 | 0.0376 | 0.0257 | 0.0255 |
| No cross-stock block | 37,098 | 0.0342 | 0.0417 | 0.0273 | 0.0275 |
| `z + Attn(z)` | 86,762 | 0.0004 | 0.0267 | 0.0161 | 0.0217 |
| `z + Attn(LN(z))` | 87,018 | 0.0005 | 0.0269 | 0.0161 | 0.0218 |

Per-epoch validation IC, seed 1729:

| Epoch | Shipped | No block | `z + Attn(z)` | `z + Attn(LN(z))` |
|---:|---:|---:|---:|---:|
| 1 | 0.0231 | 0.0342 | 0.0004 | 0.0005 |
| 2 | 0.0277 | 0.0388 | 0.0113 | 0.0115 |
| 3 | 0.0306 | 0.0415 | 0.0194 | 0.0196 |
| 5 | 0.0344 | 0.0420 | 0.0237 | 0.0239 |
| 7 | 0.0368 | 0.0417 | 0.0263 | 0.0264 |
| 10 | 0.0376 | 0.0417 | 0.0267 | 0.0269 |

Reading, in order of confidence:

1. The four trajectories are near-parallel and never cross. The ordering at
   epoch 10 is the ordering at epoch 1. Under this schedule the models move
   0.008 to 0.027 from wherever their initialisation lands; the smoke is an
   initialisation lottery plus drift, and it is being reported as such.
2. The shipped block and the residual variants share every initial weight.
   The single residual add moved the untrained validation IC from 0.023 to
   0.0004. The no-attention arm, with different final-GAT draws, landed at
   0.034. An untrained model's IC therefore ranges over at least 0.03 with the
   seed, which is the size of the entire ten-epoch training gain and roughly
   ten times the paired protocol's minimum detectable effect. This is the
   member-level variance the graph re-analysis measured after full training
   (per-member output scale varying up to 200x, per-member IC dispersion),
   seen at its source.
3. The smoke does not show the residual block learning faster, which was the
   hypothesis section 3.1 suggested, and it does not show it learning worse:
   one seed, one tenth of a training budget, and an init gap that explains the
   whole difference. The init collapse is real and the fix is correct
   transformer practice, but its value has to be sought in dispersion across
   members and folds, which only the paired protocol can measure.
4. The no-attention arm being the best single run here is consistent with the
   graph ablation's control-first result and with the 56 percent parameter
   share of a block that starts as a cross-sectional mean. It is one seed and
   not a recommendation to remove the block; it is a reason to include a
   no-block arm in the ablation so the block has to earn its place.

#### 3.2.1 Second seed at the recipe rate

Same setup, seed 1730 (so a different initialisation for every module).

| Epoch | Shipped | No block | `z + Attn(LN(z))` |
|---:|---:|---:|---:|
| 1 | -0.0167 | 0.0057 | 0.0128 |
| 2 | -0.0009 | 0.0167 | 0.0202 |
| 3 | 0.0162 | 0.0276 | 0.0264 |
| 5 | 0.0272 | 0.0369 | 0.0347 |
| 7 | 0.0329 | 0.0414 | 0.0379 |
| 10 | 0.0348 | 0.0425 | 0.0387 |

Test 2025 avg IC / avg rank IC: shipped 0.0270 / 0.0285; no block 0.0369 /
0.0333; residual 0.0317 / 0.0316.

Two-seed summary, validation IC at epoch 10 and test avg IC:

| Variant | Val IC seed 1729 | Val IC seed 1730 | Test IC seed 1729 | Test IC seed 1730 |
| --- | ---: | ---: | ---: | ---: |
| Shipped block | 0.0376 | 0.0348 | 0.0257 | 0.0270 |
| `z + Attn(LN(z))` | 0.0269 | 0.0387 | 0.0161 | 0.0317 |
| No block | 0.0417 | 0.0425 | 0.0273 | 0.0369 |

Reading:

1. The second seed reverses the first seed's epoch-1 ordering (residual first,
   shipped block last and starting negative), and the shipped block gains
   0.052 over ten epochs against 0.026 and 0.037 for the other two. Training
   does pull the arms toward a common level; the first seed's parallel lines
   were partly the accident of a lucky shipped-block initialisation.
2. Across two seeds the shipped and residual blocks are indistinguishable:
   validation means 0.036 and 0.033, test ranges overlapping. The comparison is
   variance-dominated, as section 2 predicted it would be at this budget.
3. The no-block arm leads on both seeds, by 0.004 to 0.008 on validation and
   0.002 to 0.010 on test. Two seeds on an anchored universe at a tenth of the
   budget is not evidence, but it is the one consistent signal in the smoke,
   and it is what the graph ablation's control-first result would predict for
   a block that starts as, and stays close to, a cross-sectional mean. It is
   why C5 is in the ablation.

#### 3.2.2 Six times the recipe learning rate

Same setup as the first seed (1729), learning rate 3e-4 with the same 40-step
warmup and cosine decay. This is a dynamics probe, not a recipe candidate.

| Epoch | Shipped | `z + Attn(LN(z))` |
|---:|---:|---:|
| 1 | 0.0298 | 0.0222 |
| 2 | 0.0348 | 0.0273 |
| 3 | **0.0471** | 0.0337 |
| 4 | 0.0377 | 0.0295 |
| 5 | 0.0460 | **0.0359** |
| 6 | 0.0357 | 0.0341 |
| 7 | 0.0300 | 0.0330 |
| 8 | 0.0277 | 0.0328 |
| 9 | 0.0297 | 0.0333 |
| 10 | 0.0283 | 0.0331 |

Test 2025 avg IC / avg rank IC from the best checkpoints: shipped 0.0340 /
0.0284; residual 0.0253 / 0.0251.

Reading:

1. At this rate the shipped block spikes to 0.047 at epoch 3 and then loses
   0.019 of validation IC over the remaining seven epochs, finishing below its
   own recipe-rate result. The residual block peaks lower, at 0.036, and then
   holds within 0.001 for the last four epochs. This is the instability that
   the rank-collapse literature predicts for an attention block without a
   skip path once the optimiser is allowed to move it, and the flatness that
   a residual path is supposed to buy.
2. One seed, and under the frozen recipe's `selection_metric=val_ic` with
   patience 15 the shipped block's epoch-3 peak would have been the saved
   checkpoint, so this is a stability finding, not a performance one. Where
   it would show in a full run is in the selection surface: fewer members
   whose best checkpoint is an early spike, and a tighter per-member
   dispersion, which is the pre-registered secondary for arm C1 in section 5.

### 3.3 Attention probe on real validation batches, untrained and trained

The same data as section 3.2, eight validation batches covering all 248
validation dates and 472 names. Each variant was rebuilt at the trainer's
member seed so "untrained"
is the exact initialisation the smoke started from, then the smoke's
`model_0_best.pth` was loaded. `self-weight` is the mean attention a stock
pays to itself; uniform is 1/472 = 0.0021. `top-5 mass` is the attention mass
on the five most-attended stocks.

| Variant | State | Entropy (nats, uniform 6.16) | Logit std | Self-weight | Top-5 mass | `z` after block: cos / cs share / eff rank | Val IC (248 days) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Shipped block | untrained | 6.146 | 0.221 | 0.0022 | 0.016 | 0.999 / 0.001 / 2.1 | +0.0188 |
| Shipped block | trained | 6.101 | 0.353 | 0.0025 | 0.021 | 0.990 / 0.010 / 1.2 | +0.0376 |
| `z + Attn(LN(z))` | untrained | 6.146 | 0.221 | 0.0022 | 0.016 | 0.779 / 0.221 / 4.0 | -0.0061 |
| `z + Attn(LN(z))` | trained | 6.135 | 0.275 | 0.0025 | 0.017 | 0.762 / 0.237 / 4.2 | +0.0269 |
| No block | untrained | | | | | | +0.0299 |
| No block | trained | | | | | | +0.0420 |

`z` before the block on real data: cosine 0.70, cross-sectional share 0.30,
effective rank 4.0 untrained; 0.65 / 0.35 / 3.9 trained. The `z + Attn(z)`
variant is identical to the pre-norm one to three decimals and is omitted.

Reading:

1. **The shipped block does not become attention within this budget.** After
   training, its attention is uniform to within 0.06 nats, the self-weight is
   1.2 times uniform, and five stocks together receive 2 percent of the mass.
   Logit standard deviation grew from 0.22 to 0.35, so the block is
   sharpening, but slowly, and it has ten times the training ahead of it in
   a frozen-recipe run. Section 3.3.1 probes the higher-rate checkpoints to
   see where that sharpening leads.
2. **Ninety-nine percent of the cross-sectional variance is discarded at the
   block, and the score head works from a rank-1 residue.** Before the block
   `z` carries 35 percent of its variance across stocks at effective rank 4;
   after it, 1 percent at effective rank 1.2. The trained model still reaches
   validation IC 0.038 through that channel, which says the useful ranking
   signal is low-dimensional, and that the final GAT is amplifying a
   near-constant input (output cross-sectional standard deviation 0.27 against
   0.61 for the no-block model).
3. **What the block actually computes at this operating point.** With
   near-uniform attention, the per-stock output is the cross-sectional mean of
   the values plus a term linear in the stock's own vector through the
   date's key-value covariance. That is a date-adaptive linear projection
   with a large shared offset, not a mechanism that relates specific stocks
   to specific other stocks. It is also why the residual variant, which adds
   the same offset to an intact `z`, and the no-block variant, which has
   neither, bracket the shipped block rather than dominate it: all three are
   linear-ish readouts of `z` that differ mainly in which random projection
   they start from.
4. **The untrained IC lottery is confirmed on real data.** Same weights,
   one residual add: +0.019 becomes -0.006. Different final-GAT draws with no
   block: +0.030. These are the epoch-1 values from 3.2 to within a few
   thousandths, so the first epoch under warmup barely moved anything.

#### 3.3.1 The same probe on the higher-rate best checkpoints

Best checkpoints from section 3.2.2 (shipped block at its epoch-3 peak,
residual block at its epoch-5 peak), same eight validation batches.

| Variant | Entropy (nats, uniform 6.16) | Logit std | Self-weight | Top-5 mass | `z` after block: cos / cs share / eff rank | Val IC (248 days) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Shipped block, 3e-4, epoch 3 | 5.775 | 0.895 | 0.0024 | 0.043 | 0.967 / 0.034 / 1.1 | +0.0471 |
| `z + Attn(LN(z))`, 3e-4, epoch 5 | 5.991 | 0.657 | 0.0030 | 0.041 | 0.660 / 0.339 / 4.3 | +0.0359 |

Reading:

1. Given six times the learning rate, the shipped block does sharpen: entropy
   falls 0.37 nats below uniform and logit spread quadruples. But it sharpens
   toward a date-level pattern, not a stock-level one: the self-weight is
   still at the uniform level and five stocks together hold 4 percent of the
   mass. Even at its best checkpoint the block passes 3.4 percent of the
   cross-sectional variance at effective rank 1.1. Its ceiling as designed is a
   rank-one, date-adaptive projection of `z`, and the validation IC of 0.047 it
   reached at that point was the spike that then decayed.
2. The residual block at its best keeps 34 percent of the cross-sectional
   variance at rank 4.3, with a self-weight 1.4 times uniform. The attention
   sits on top of an intact `z` instead of replacing it, which is the whole
   point of the skip path and is what made the trajectory in 3.2.2 flat.
3. Taken together with 3.2, the evidence on the block is: as shipped it is a
   bottleneck that the network works around rather than through; the residual
   form removes the bottleneck without a visible mean gain at this budget; and
   the no-block arm matches or beats both. The block has to earn its place in
   the ablation, and the residual form is the version it should be given to
   earn it with.

### 3.4 Do the proposed fixes actually fix it?

Same synthetic harness as 3.1, with a PIT-style mask (37 of 500 names
inactive). Three things are measured for each candidate block: how much
cross-sectional structure survives it, whether it respects the mask, and
whether a stock's score can actually respond to the rest of the cross-section
in a way the loss can see.

The last one needs care. Perturbing 300 other stocks and watching stock 0's raw
score is misleading, because `ICLoss` centres within the date: a block that
shifts every stock equally scores high on the raw measure and contributes
nothing. So the table reports the change in stock 0's **centred** score, which
is what the loss sees. Stock 0 sits in graph cluster 0, so the perturbed
stocks 100 to 400 can reach it only through the cross-stock block, never
through the graph.

| Block | Block params | cs var share after | Eff rank after | Raw \|dscore_0\| | **Centred \|dscore_0\|** | Step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped `SelfAttention` | 49,664 | 0.074 | 1.0 | 0.336 | **0.011** | 0.29 s |
| No block (floor) | 0 | | | 0.000 | 0.157 | 0.25 s |
| Fix A, `z + Attn(LN(z))` | 49,920 | 0.225 | 9.0 | 0.185 | **0.534** | 0.33 s |
| Fix B, two-way latents | 136,960 | 0.261 | 11.1 | 0.038 | **0.511** | 0.34 s |

`z` entering the block: cs var share 0.189, effective rank 5.5. Every variant
passed both masking checks: inactive nodes exactly zero at the output, and
perturbing an inactive stock moved no active score at all (leakage 0.00e+00).

Reading:

1. **The shipped block's response to the cross-section is 97 percent common
   shift.** It moves stock 0's raw score by 0.336 when 300 other stocks move,
   but only 0.011 of that survives centring. The block is passing the date's
   mean and almost nothing else, and the mean is exactly what the IC loss
   discards. This is the sharpest statement of the defect: the block is not
   merely collapsing its output, it is spending 56 percent of the model's
   parameters computing a quantity the objective cannot use.
2. **The no-block floor is 0.157, and it is an artifact.** With no cross-stock
   block, stock 0's raw score cannot move at all (0.000 exactly, confirming
   there is no other path). Its centred score still moves, because perturbing
   300 stocks moves the mean that centring subtracts. That 0.157 is therefore
   the mechanical floor for this measurement, not evidence of information flow.
   **The shipped block sits below its own no-block floor.**
3. **Both fixes clear the floor by about 3.3x** (0.534 and 0.511) and preserve
   three to four times the cross-sectional variance at 9 to 11 effective rank
   instead of 1.0. They restore a channel that is closed today.
4. **Cost.** Fix A is free: 256 extra parameters and no measurable step-time
   change. Fix B roughly doubles the model (89,450 to 176,746) because it uses
   two `nn.MultiheadAttention` modules at 4 x 128^2 each, though the step time
   is unchanged since 32 latents are cheap against 500 stocks. If that cost is
   unwelcome the latent width can be cut below `concat_size`, or the two
   projections shared, at some expressivity loss.

### 3.5 Training smoke on Fix B, and what it does not support

Same harness as 3.2 (anchored universe, 16 features, 10 epochs, recipe learning
rate, one model), with the two-way latent block in the cross-stock position.
174,058 parameters against the shipped 86,762.

| Epoch | Fix B, seed 1729 | Fix B, seed 1730 |
|---:|---:|---:|
| 1 | -0.0088 | 0.0118 |
| 3 | 0.0161 | 0.0262 |
| 5 | 0.0236 | 0.0327 |
| 7 | 0.0257 | 0.0361 |
| 10 | 0.0267 | 0.0373 |

All four variants, side by side:

| Variant | Val IC 1729 | Val IC 1730 | Seed spread | Test IC 1729 | Test IC 1730 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shipped block | 0.0376 | 0.0348 | 0.0028 | 0.0257 | 0.0270 |
| Fix A, residual | 0.0269 | 0.0387 | 0.0118 | 0.0161 | 0.0317 |
| Fix B, two-way latents | 0.0267 | 0.0373 | 0.0106 | 0.0166 | 0.0306 |
| No block | 0.0417 | 0.0425 | 0.0008 | 0.0273 | 0.0369 |

Reading, and three of these four points are negative for the fixes:

1. **Fix B trains cleanly.** Ten epochs on two seeds, no divergence, no NaN
   under the masked PIT path, monotone improvement after epoch 1. The
   mechanics are confirmed; that was the purpose of the run.
2. **Fix B and Fix A are indistinguishable at this budget.** They agree to
   0.002 on validation and 0.005 on test on both seeds. Whatever the two-way
   block's extra 87K parameters buy, it does not show here. The information
   channel the diagnostics say it opens is not converting into IC in ten
   epochs on one universe.
3. **Neither fix beats removing the block, on either seed.** The no-block arm
   leads on all four measurements. That is now three separate lines of evidence
   pointing the same way: the graph ablation's control-first result, the
   parameter-share argument, and this smoke.
4. **The seed spread runs against the dispersion hypothesis.** Fix A was
   pre-registered on the claim that a residual path would reduce
   member-to-member variance. Across these two seeds the fixes are the *more*
   dispersed arms (0.011 and 0.012) and the shipped and no-block arms the less
   dispersed (0.003 and 0.001). Two seeds cannot settle a variance question and
   this is not evidence against the hypothesis, but it is not evidence for it
   either, and section 5's expectation is corrected accordingly.

The honest summary of section 3 as a whole: the diagnostics establish beyond
doubt *what the two blocks do*, and the fixes provably restore the closed
channels. The training evidence establishes only that the fixes are trainable.
It does not show them helping, and at this budget it mildly favours deleting
the cross-stock block over repairing it.

### 3.6 The same diagnostics at the real universe size

Synthetic batches reshaped to the real setting: 201-node union axis, 110
active names, 91 masked, 16 features, 11 sector-like clusters. Everything else
as in section 3.4. Side by side with the 500-name numbers those sections
reported.

| Block | cs var share, before to after | Eff rank, before to after | Centred \|dscore_0\| | Step |
| --- | --- | --- | ---: | ---: |
| **At 110 active names (correct)** | | | | |
| Shipped `SelfAttention` | 0.501 to 0.453 | 1.5 to 1.0 | **0.0011** | 0.36 s |
| No block (floor) | | | 0.579 | 0.40 s |
| Fix A, residual | 0.501 to 0.529 | 1.5 to 1.8 | 0.437 | 0.39 s |
| Fix B, two-way latents | 0.499 to 0.550 | 1.4 to 2.1 | 0.590 | **0.57 s** |
| **At 500 names (what 3.1 to 3.5 used)** | | | | |
| Shipped `SelfAttention` | 0.189 to 0.074 | 5.5 to 1.0 | 0.011 | 0.29 s |
| No block (floor) | | | 0.157 | 0.25 s |
| Fix A, residual | 0.189 to 0.225 | 5.5 to 9.0 | 0.534 | 0.33 s |
| Fix B, two-way latents | 0.189 to 0.261 | 5.5 to 11.1 | 0.511 | 0.34 s |

Masking held in every variant at both scales: inactive nodes exactly zero,
leakage from inactive names 0.00e+00.

**What strengthens.** The central finding is scale-independent and is if
anything starker at 110 names. The shipped block's response to the
cross-section survives per-date centring at **0.0011**, against a no-block
floor of 0.579. That is a ratio of 1 to 526. Whatever the block passes to the
score head, the ranking objective cannot use it. Effective rank after the block
is 1.0 at both scales. Both fixes restore centred responsiveness to the floor
or above.

**What appeared to change, and did not.** On this synthetic harness the
variance destruction looked far milder at 110 names: 0.501 to 0.453, about a
tenth, against two thirds at 500 names. I drafted that as a correction to
section 3.3. **Section 3.7 then measured it on real data at 110 names and the
opposite is true: 0.433 to 0.024, a 94 percent reduction.** The synthetic
inputs, not the original claim, were the unreliable element. Synthetic `z` here
enters the block at effective rank 1.4 to 1.5, where real `z` enters at 3.5 to
3.6, so the synthetic batches were already close to rank-collapsed before the
block touched them and had little left to lose. **Treat section 3.7 as
authoritative on variance and this table as authoritative only on parameters,
step time and masking**, which do not depend on the input distribution.

**What changes, second, and this is the one that moves a recommendation.**
Fix B's efficiency argument dies at 110 names. Its case rested on replacing an
`O(N^2)` block with an `O(N*k)` one, but with `k = 32` latents against 110
active names that is a 3.4 to 1 compression rather than the 15 to 1 available
at 500, and two attention calls plus three LayerNorms cost more than the single
block they replace. Measured: **0.57 s against 0.36 s, a 58 percent step-time
increase**, on top of doubling the parameters. At 500 names it was
cost-neutral; at 110 it is not. Fix B now has to justify roughly 2x parameters
and 1.6x step time on the strength of its mechanism alone, with no efficiency
argument and, from section 3.5, no smoke evidence of benefit.

Fix A is unaffected by all of this. It remains 256 parameters and a step time
inside the noise (0.39 s against 0.36 s), and it clears the centred-sensitivity
floor at both scales.

### 3.7 The authoritative measurement: real data, real universe, trained models

Everything above is either synthetic or measured at the wrong universe size.
This section is neither. It runs the repo's own pipeline on the 110-name panel
(201-node axis, masked-panel PIT, 2,012 training days), trains one model per
variant for 10 epochs at the recipe learning rate, and probes the trained
checkpoints on all 239 validation dates with the PIT mask applied exactly as
the trainer applies it. Seed 1729. Global regime features off, which is a
deviation from the frozen recipe and applies equally to every variant.

**Information flow through the block, trained model, active names only:**

| Variant | Entropy (uniform 5.30) | Logit std | Self-weight (uniform 0.0050) | Top-5 mass | cs var share, before to after | Eff rank, before to after |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Shipped block | 5.280 | 0.258 | 0.0048 | 0.034 | **0.433 to 0.024** | 3.6 to 1.6 |
| Fix A, residual | 5.281 | 0.259 | 0.0046 | 0.033 | 0.403 to 0.329 | 3.5 to 3.6 |

**Validation and test IC, same runs:**

| Variant | Params | Val IC untrained | Val IC trained | Test avg IC | Test rank IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shipped block | 86,762 | -0.0064 | 0.0194 | 0.0244 | 0.0301 |
| **No block** | **37,098** | **+0.0479** | **+0.0513** | **0.0422** | **0.0434** |
| Fix A, residual | 87,018 | -0.0067 | 0.0174 | 0.0104 | 0.0126 |
| Fix B, two-way latents | 174,058 | -0.0070 | 0.0201 | 0.0113 | 0.0124 |

Four readings:

1. **The defect is confirmed at the real scale, on real data, after training.**
   The shipped block destroys 94 percent of the cross-sectional variance
   (0.433 to 0.024) and takes effective rank from 3.6 to 1.6. Its attention is
   uniform to within 0.02 nats of the bound, and it gives a stock *less* weight
   on itself than uniform (0.0048 against 0.0050). This is the strongest
   version of the finding in this document and it supersedes the softer
   synthetic reading in 3.6.
2. **Fix A repairs the information flow, exactly as designed.** Variance share
   0.403 to 0.329 instead of 0.433 to 0.024; effective rank preserved at 3.5 to
   3.6. The mechanism works.
3. **The IC column on this seed favoured removing the block, and did not
   replicate.** At seed 1729 the no-block model led everything: validation IC
   0.0513 against 0.0194, test IC 0.0422 against 0.0244, with Fix A and Fix B
   below the shipped block. **Section 3.8 repeats this at seed 1730 and the
   entire ordering inverts.** Read the IC columns above as one draw from a
   distribution whose spread exceeds every gap in the table, not as a result.
4. **The untrained gap shows why.** The no-block model starts at +0.0479
   before any training and ends at +0.0513, gaining 0.003 from ten epochs. The
   three attention variants all start slightly negative and climb to roughly
   0.02. Almost all of the seed-1729 no-block lead was present at
   initialisation, which is the effect section 3.2 documented and section 3.8
   confirms.

**Caveats, all of which cut against reading this as a result.** One seed. Ten
epochs against the recipe's 100. One model against the recipe's 20-member
ensemble. Global regime off. No transaction costs, no rank gate, no paired
inference. The second seed was running when this was written. This is a
mechanics and dynamics probe, and the only claim it settles is what the blocks
do to information, not what any of them is worth.

### 3.8 Second seed: every IC ordering inverts

The section 3.7 runs repeated at seed 1730, everything else identical.

| Variant | Val IC ep1 s1 | s2 | Best val s1 | s2 | Test IC s1 | s2 | Test mean | **Seed spread** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped block | -0.0043 | -0.0098 | 0.0194 | 0.0230 | 0.0244 | 0.0263 | 0.0254 | 0.0019 |
| No block | +0.0513 | +0.0085 | 0.0513 | 0.0335 | 0.0422 | 0.0302 | 0.0362 | 0.0120 |
| Fix A residual | -0.0027 | +0.0104 | 0.0174 | 0.0337 | 0.0104 | 0.0361 | 0.0233 | **0.0257** |
| Fix B two-way | -0.0016 | +0.0112 | 0.0201 | 0.0327 | 0.0113 | 0.0333 | 0.0223 | 0.0220 |

Test-IC ranking, seed 1729: No block, Shipped, Fix B, Fix A.
Test-IC ranking, seed 1730: **Fix A, Fix B, No block, Shipped.**

The order is exactly reversed. Fix A goes from last to first; the no-block arm
from first to third; the shipped block from second to last.

1. **No IC conclusion survives.** The largest within-variant seed spread is
   0.0257 on test IC. The between-variant range inside seed 1730 is 0.0098. The
   noise is two and a half times the signal it would have to resolve, and seven
   times the paired protocol's 0.0035 bar. Every ordering in 3.7, including the
   one I briefly promoted to this document's headline, was a seed artifact.
2. **The no-block arm's apparent dominance was one lucky initialisation.** Its
   seed-1729 run started at +0.0513 before training and finished at +0.0513;
   at seed 1730 it started at +0.0085 and reached 0.0335. Its mean is still the
   highest of the four, but it rests on a draw that did not repeat.
3. **The dispersion hypothesis fails again, and more clearly.** Fix A was
   proposed partly on the argument that a residual path would stabilise members.
   It has the *largest* seed spread of any arm (0.0257) and the shipped block
   the smallest (0.0019). Two seeds still cannot settle a variance question,
   but the claim has now pointed the wrong way in three separate smokes and
   should not be repeated without evidence.
4. **What this does establish** is the case for the paired protocol. A single
   model on this universe cannot distinguish architectures whose true effects
   are anywhere near the 0.0035 range. That is precisely why the ablation
   specifies 20-member ensembles, shared member seeds, identical random-number
   consumption and paired daily differences. These smokes are worth exactly
   what they cost: they prove the code runs and they measure information flow.
   They rank nothing.

## 4. Candidate changes

Each candidate lists mechanism, evidence, cost, invariant exposure, and the
test that would settle it. Sections 4.1 and 4.2 are the two fixes for the two
attention blocks and are given in full, since they are the point of this map.
**Read sections 3.6 and 3.7 first**: 3.6 reprices Fix B at the universe size
that actually matters, and 3.7 is the only measurement here taken on real data
at that size.

### 4.1 Fix A: residual pre-norm cross-stock block

This is the fix for the **cross-stock `SelfAttention` block** in
`mci_gru/models/attention.py`, the one section 3 shows collapsing.

Mechanism. Keep the existing attention, but stop it from replacing `z`. Wrap
it in the standard pre-norm residual form so its output is a correction to `z`
rather than a substitute for it, and re-apply the stock mask after the add.

```python
class ResidualCrossStockBlock(nn.Module):
    """z + Attn(LN(z)), with the PIT mask re-applied after the residual add."""

    def __init__(self, inner: SelfAttention, dim: int, pre_ln: bool = True):
        super().__init__()
        self.inner = inner
        self.ln = nn.LayerNorm(dim) if pre_ln else nn.Identity()

    def forward(self, x, stock_mask=None):
        out = x + self.inner(self.ln(x), stock_mask=stock_mask)
        if stock_mask is not None:
            out = out * stock_mask.unsqueeze(-1).to(out.dtype)
        return out
```

Three refinements worth taking at the same time, none of which changes the
above shape: split the attention into four heads of 32 dimensions instead of
one head of 128 (the single wide head is why the logits start near-uniform);
add an optional `z = z + FFN(LN(z))` sublayer; and keep the group-type
embedding inside `inner` so the `[A1, A2, B1, B2]` contract is untouched.

Evidence. Sections 3.1, 3.3 and 3.4. Dong et al. 2021 for why an attention
block without a skip path collapses toward rank one. MASTER's inter-stock
attention uses residual connections and layer normalisation, as does every
production transformer. The repo's own phase-2 plan added LayerNorm and dropout
*around* the trunk but stopped short of the block itself.

Cost. 256 parameters, no measurable step-time change. Add about 33K if the FFN
is included.

Invariant exposure. None on data, labels, graph timing, or paper-trade,
provided the new `ModelConfig` field defaults to the legacy block so old
`config.yaml` files in checkpoint folders build the same module graph. The mask
re-application needs a mutation-checked test alongside
`tests/test_pit_masked_panel.py::test_self_attention_mask_prevents_inactive_node_influence`;
section 3.4 verifies both mask properties on the prototype.

Test. Arm C1 in section 5.

### 4.2 Fix B: make the cross-attention's latents data-dependent

This is the fix for the **multi-head cross-attention** in
`mci_gru/models/latent.py`, the "MCI" the model is named after. Section 2
records the defect: `R1` and `R2` are plain `nn.Parameter` tensors and the only
input to `forward` is the per-stock vector, so `B1 = f(A1)` and `B2 = f(A2)`
pointwise. After training they are fixed. The streams the architecture calls
"market latent states" cannot observe the market, on any date, by construction.
They are a learned codebook: each stock's vector re-expressed as a soft mixture
of 32 prototypes. That is a per-stock nonlinearity, not a market state.

Mechanism. Make the latents read the date before the stocks read the latents.
Two cross-attentions instead of one, gather then broadcast:

```
R_d = R + MHA(query=R,  key/value=Z_active_on_date_d)   # latents read the cross-section
Z'  = Z + MHA(query=Z,  key/value=R_d)                  # each stock reads the date's latents
```

Now `R_d` is a genuine 32-vector summary of that date's market, learned rather
than hand-specified, and every stock's representation is conditioned on it.
This is the Induced Set Attention Block of the Set Transformer (Lee et al.,
2019), the same shape Perceiver uses. It costs `O(N*k)` with `k = 32` latents
against the cross-stock block's `O(N^2)`, so it is cheaper than the block it
can replace even while doing strictly more.

```python
class TwoWayLatentBlock(nn.Module):
    """Gather from the date's cross-section into learned latents, broadcast back."""

    def __init__(self, dim, num_latents=32, num_heads=4, dropout=0.0, init_scale=0.02):
        super().__init__()
        self.R = nn.Parameter(torch.randn(num_latents, dim) * init_scale)
        self.ln_z_gather = nn.LayerNorm(dim)
        self.ln_r = nn.LayerNorm(dim)
        self.ln_z_broadcast = nn.LayerNorm(dim)
        self.gather = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.broadcast = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x, stock_mask=None):
        r = self.R.unsqueeze(0).expand(x.shape[0], -1, -1)
        key_pad = None
        if stock_mask is not None:
            mask = stock_mask.to(dtype=torch.bool, device=x.device)
            key_pad = ~mask
            dead = key_pad.all(dim=-1)          # a date with no active names would
            if dead.any():                       # make the softmax all -inf; let it
                key_pad = key_pad.clone()        # attend freely and zero the result
                key_pad[dead] = False            # at the end instead
        z_kv = self.ln_z_gather(x)
        if stock_mask is not None:
            z_kv = z_kv * mask.unsqueeze(-1).to(z_kv.dtype)
        r_upd, _ = self.gather(self.ln_r(r), z_kv, z_kv,
                               key_padding_mask=key_pad, need_weights=False)
        r_d = r + r_upd
        out, _ = self.broadcast(self.ln_z_broadcast(x), r_d, r_d, need_weights=False)
        out = x + out
        if stock_mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out
```

Where to put it. Two placements, and the choice is a real decision:

- **In the cross-stock block's position**, replacing `SelfAttention` on `z`.
  This is what section 3.4 measured and what the smoke in 3.5 trains. It fixes
  both blocks at once: the latents become data-dependent *and* the degenerate
  `O(N^2)` block is gone. `B1` and `B2` stay as they are, harmless.
- **Inside `MarketLatentStateLearner`**, replacing `R1` and `R2` with `R1_d`
  and `R2_d` from the same gather step. This keeps the four-stream shape
  literally intact and is closer to what the paper claims to do, but it runs
  the gather twice and leaves the collapsing cross-stock block in place, so it
  should be combined with Fix A.

The first placement is the one to test first: it is a strictly smaller change
to the trunk's wiring and it subsumes more.

Cheaper alternative. If doubling the parameter count is unwelcome, the
MASTER-style gate gets some of the same benefit for a few hundred parameters:
build a per-date market vector `m_d` from the masked cross-sectional mean and
dispersion plus the existing regime, VIX and credit features, then
`alpha_d = F * softmax(W m_d / beta)` applied as `x = alpha_d * x` before A1 and
A2. That is feature selection conditioned on the market rather than a learned
market state, and it is the June scan's `MarketStateFeatureGate`. It is the
fallback, not the first choice, because it conditions the *inputs* while the
two-way block conditions the *representations*.

Evidence. Section 2 for the defect (`R1`, `R2` are parameters; `forward` takes
only per-stock vectors). Section 3.4 for the repair: the two-way block lifts
centred cross-sectional responsiveness from 0.011 to 0.511 and preserves
effective rank 11.1 against 1.0. Lee et al. 2019 for the ISAB construction.
MASTER (AAAI 2024) reports IC 0.064 and rank IC 0.076 on CSI300 against 0.049
and 0.041 for the best baselines from market conditioning. The IC-loss
invariance in section 2 explains why the broadcast regime features have been a
"secondary tuning dimension" in the April ablation rather than a driver: with
fixed latents and a centring loss they have no path to the objective.

Cost. 136,960 parameters in the block, taking the model from 89,450 to
176,746 on the synthetic harness (174,058 on the 16-feature smoke). The
asymptotic saving is real but does not show at this size: CPU step times were
0.29 s for the shipped block and 0.34 s for the two-way block, within the
run-to-run spread seen across repeats, because at `N = 500` and `k = 32` the
constant factors of two attention calls and three LayerNorms outweigh the
`O(N^2)` to `O(N*k)` improvement. Treat it as cost-neutral in time and roughly
double in parameters. Reducible by narrowing the latent width below
`concat_size` or sharing the two projections.

Invariant exposure. The gather must exclude PIT-inactive names, which the
`key_padding_mask` above does and section 3.4 verifies (zero leakage from
inactive stocks). It must not pool across the batch's other dates: attention is
per batch row, so this holds by construction, but it deserves its own
mutation-checked test since a reshape bug here would be silent lookahead across
dates. Softmax in float32 under AMP. Paper-trade inference already has every
input this needs; the new `ModelConfig` field must default to legacy and be
serialised by `to_dict`.

Test. Arm C2 in section 5.

### 4.3 Capacity-matched widths

Mechanism. Raise `output_gat1` from 4 to `hidden_size_gat1` (32) and make
`proj_cross` an identity when the widths match; resolve #131 so that
`gru_hidden_sizes` means the same thing under every encoder, and consider a
width of 32 on the temporal output so `proj_temporal` is not expanding a
10-dimensional signal.

Evidence. Effective ranks of 3.1 and 4.3 on the two primary streams at init;
the paper's own sensitivity table picked hidden size 32 while its
`output_gat1=4` was never ablated. Gu, Kelly and Xiu warn that depth overfits
on small panels; width on a two-layer stream is a different lever and the
parameter cost here is trivial.

Cost. Tens of thousands of parameters at most.

Invariant exposure. Checkpoint compatibility only. Old configs must keep
building old shapes.

Test. Arm C4, only after #131 has a decision, because an encoder that silently
reinterprets its width would make the arm unreadable.

### 4.4 Graph-aware input to the cross-sectional stream

Mechanism. Give the GAT a trailing summary per stock (last day, rolling mean
and volatility over `his_t`, or the A1 output itself) instead of one day of raw
features. This is the roadmap's graph-input summariser and the phase-3
`use_a1_a2_cross_attention` flag approaches it from the other side.

Evidence. Architecture review gap 8; the A1/A2 cross-attention flag exists but
has never been ablated under the paired protocol.

Cost. Small.

Invariant exposure. None new.

Test. Blocked until map #157 decides whether correlation edges stay in the
recipe. If the graph leaves, this item leaves with it; if the sector map wins,
the summariser is the natural next arm.

### 4.5 Initial-residual skip into the score head

Mechanism. `score = FinalGAT(z) + w^T z`, or GCNII-style mixing of `z` into
the second GAT layer's input.

Evidence. GCNII for the over-smoothing argument. Low priority because the
shipped graph is sparse enough that over-smoothing is not the current failure.
In the same bundle, `GATv2Conv` in place of `GATConv` in both `GATBlock`s is a
one-line change that removes the static-attention limitation of the original
GAT (the ranking of neighbours cannot depend on the query node) at the same
parameter cost.

Test. Fold into C1 if it costs nothing; otherwise defer until map #157 has
decided whether the graph stays.

Scripts and raw outputs for every measurement in section 3 are kept in
`docs/research/current/trunk_architecture_diagnostics_2026-09-05/`. They are
scratchpad drivers that monkeypatch `create_model` and hard-code this
worktree and the local anchored-universe CSV path; they are records of method,
not repo tooling.

### 4.6 Explicitly deferred

- Looped or weight-tied depth on any stream: see the looped-transformer brief.
- Perceiver-style rewrite of B1 and B2 with data-dependent latents: a redesign
  with no evidence advantage over the gate in 4.2.
- Multi-horizon auxiliary heads: an objective change, owned by the loss path
  decision note, not a trunk change.
- Transformer depth on the temporal path: ten tokens, and the long-history
  results moved returns through `his_t`, not through the encoder.

## 5. A pre-registered trunk-hygiene ablation

This supersedes the block-depth arm set in the looped brief's slice 2. The
protocol machinery is the one decided on #181 and built under #183; nothing
below adds a new arbiter.

Arms, everything else at the frozen recipe:

Universe: `configs/data/gics_top10_110_2016.yaml`, the 110-name GICS sector
top-10 panel in masked-panel PIT mode, which is what the graph ablation and the
0.0035 bar were measured on. Not the anchored 2019 snapshot universe used for
the section 3 smokes.

| Arm | Change from C0 | Cost vs C0 | Priority |
| --- | --- | --- | --- |
| C0 | Shipped trunk | | control |
| **C5** | **No cross-stock block (`use_self_attention=false`)** | **-50K params, faster** | **run first** |
| C1 | Fix A: residual pre-norm cross-stock block, four heads, mask re-applied per add | +256 params | run with C5 |
| C2 | Fix A plus the market gate: MASTER-style softmax gate from a per-date masked market vector | +~600 params | second wave |
| C3 | Fix B: two-way latent block in the cross-stock position | +137K params, +58% step | only if C2 promises |
| C4 | Capacity-matched widths on top of whichever of C1 to C3 survives, after #131 | | last |

C5 is in the first wave because it is free to test, needs no new code, and has
three structural arguments behind it. It is **not** there because the smokes
showed it winning: that lead did not survive a second seed (section 3.8). Run
C0, C5 and C1 together as the first wave, which answers "keep the block, fix
the block, or drop the block" for the cost of three arms and settles the
question the smokes could not. C2 and C3 repair Defect 2 cheaply and
expensively, and are worth running only once the block's fate is known, since
Fix B in the cross-stock position is moot if that position is empty. The second
placement of Fix B, inside `MarketLatentStateLearner`, is the one that survives
C5 winning, because the frozen-latent defect is independent of the cross-stock
block.

Pairing and inference as ruled on #181: rolling folds over the PIT masked
panel, member seeds shared across arms, identical random-number consumption,
mean paired daily Pearson IC against C0 with lag-4 HAC and 5-session blocks,
BHY across the arm-versus-C0 comparisons (m = 4 without C4, m = 5 with it),
seed-paired per-member IC and per-member IC dispersion as the pre-registered
secondaries, and the three-branch outcome map with the undecidable branch
stated up front. C1 is the arm whose pre-registered claim is a dispersion
reduction rather than a mean shift; C2 and C3 are the arms whose pre-registered
claim is a mean effect.

Budget. The 110-name panel trains far faster than the S&P-scale runs the
20-minute-per-fold figure came from, so cost that from a measured fold rather
than from this map. Relative costs: C1, C2 and C5 are at or below C0; C3 adds
about 58 percent to the step time.

Expected outcome, revised after the section 3.5 smoke rather than stated in
advance of it, and weaker than the version this map first carried.

C2 and C3 remain the arms with a mechanism for a mean effect, because they open
a channel that is closed today. But the smoke gives no support for that effect
appearing: Fix B matched Fix A to within 0.002 on both seeds and neither beat
removing the block. So the pre-registered expectation is now **that no arm
clears the 0.0035 bar**, and the arms exist to test a hypothesis the cheap
evidence already declines to confirm. Section 3.6 adds a second reason to
expect little from C3 specifically: at 110 names it buys no efficiency and
costs 58 percent more step time, so it must earn its place on mechanism alone.

C1's dispersion claim is withdrawn as a prediction. Two seeds put the fixes on
the *more* dispersed side, which settles nothing at n = 2 but removes the basis
for predicting the opposite. Per-member dispersion stays a pre-registered
secondary; it is now a genuinely open measurement rather than a confirmation.

C5 is the arm to watch. Three independent lines now point at the cross-stock
block being unnecessary: the graph ablation's control-first result, its 56
percent parameter share for a near-rank-one output, and a two-seed smoke where
deleting it led every measurement. If C5 beats C0, the block leaves the recipe
and Fix A and Fix B become moot for that position, though Fix B would still be
worth testing inside `MarketLatentStateLearner` where the frozen-latent defect
is independent of the cross-stock block.

If nothing clears BHY, the honest branch is "undecidable at this universe and
horizon". Fix A can still be adopted as the default for new experiments on
correctness grounds, since it is free and the shipped block is unstable at
higher learning rates, with the frozen recipe untouched.

One caution specific to Fix B. It roughly doubles the parameter count, and
section 3.2 established that this trunk's run-to-run variance is already large
relative to the effect sizes in play. A bigger model on an 89K-parameter
baseline that Gu, Kelly and Xiu's results would already call generously sized
for the signal could easily trade variance for nothing. That is an argument for
running C2 with the same 20-member ensemble and the same paired protocol as
everything else, not for skipping it.

## 6. Invariant checklist for these changes

- No lookahead: none of the candidates touches dates, windows, labels, or
  graph timing; the market vector in 4.2 is a per-date masked mean.
- Train-only normalisation: unchanged.
- 9-tuple collate and `edge_feature_dim`: unchanged.
- PIT masks: every residual add and the market mean must exclude inactive
  union nodes; mutation-checked tests required.
- Ensemble averaging: unchanged.
- Paper-trade frozen checkpoints: every new `ModelConfig` field defaults to
  the legacy behaviour and is serialised by `to_dict`, so old bundles load and
  new runs are auditable.
- AMP: the residual block adds no unnormalised accumulation; the gate's
  softmax should be computed in float32.

## 7. Sources

Repo evidence:

- `mci_gru/models/trunk.py`: `StockPredictionModel.forward` applies
  `self.self_attention` once and replaces `z`; `proj_cross` maps 4 to 32.
- `mci_gru/models/attention.py`: `SelfAttention` has Q, K, V projections,
  scale `embed_dim**-0.5`, no residual, no norm, no FFN.
- `mci_gru/models/latent.py`: `R1` and `R2` are `nn.Parameter`; the query is
  the stock vector and nothing else enters.
- `mci_gru/training/losses.py`: `ICLoss` centres per date.
- `docs/research/current/GRAPH_SPECIFICATION_ABLATION_2026-09-01.md` and
  `GRAPH_PAIRED_REANALYSIS_2026-09-02.md` (on their branch refs): control not
  beaten; MDE 0.0035; 75 to 78 percent of names isolated; self-loops make the
  zeroed arm a two-layer MLP.
- `docs/ABLATION_NOTEBOOK_RESULTS_REPORT_2026-04-30.md`: regime context a
  secondary dimension; pure IC loss the winner.
- `docs/ARCHITECTURE_REVIEW.md` gaps 7 and 8; issue #131.

External:

- Dong, Cordonnier, Loukas, "Attention is not all you need: pure attention
  loses rank doubly exponentially with depth", ICML 2021.
  https://arxiv.org/abs/2103.03404
- Li et al., "MASTER: Market-Guided Stock Transformer", AAAI 2024.
  https://arxiv.org/abs/2312.15235
- Lee et al., "Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks", ICML 2019. The induced set attention
  block is the construction behind Fix B. https://arxiv.org/abs/1810.00825
- Jaegle et al., "Perceiver: General Perception with Iterative Attention",
  ICML 2021, for the same gather-and-broadcast latent shape.
  https://arxiv.org/abs/2103.03206
- Chen et al., "Simple and Deep Graph Convolutional Networks" (GCNII), ICML
  2020. https://arxiv.org/abs/2007.02133
- Brody, Alon, Yahav, "How Attentive are Graph Attention Networks?" (GATv2),
  ICLR 2022. https://arxiv.org/abs/2105.14491
- Gu, Kelly, Xiu, "Empirical Asset Pricing via Machine Learning", RFS 2020.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577
- Microsoft Qlib benchmark table, Alpha158 CSI300.
  https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
- Wang et al., "MCI-GRU", Neurocomputing 2025, ablation Tables 11 and 12
  (local text at `references/2410.20679v3.txt`).
