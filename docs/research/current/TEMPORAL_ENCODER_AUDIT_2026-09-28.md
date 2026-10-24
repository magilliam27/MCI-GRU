# Temporal Encoder Audit

Date: 2026-09-28

Status: research report for issue #233, requested by the maintainer on #131.
The evidence is mechanics-level: seeded numerical probes on synthetic inputs, a
reading of the code, the paper and the git history, and one end-to-end CPU smoke
per encoder through `run_experiment.py`. It makes **no model-performance claim**.
Nothing here changes code, a default, a recipe or a config file. The one
correctness defect found is filed separately as #234.

Evidence status:

- **`[Verified]`** means one of two things:
  - reproduced by the scripts in Appendix A, run against `origin/main` at
    `dbb42ed` on torch 2.11.0+cpu, with output in Appendix B; or
  - read directly from the cited file, commit or tag.
- **`[Inferred]`** means reasoned from verified facts but not measured directly.

The probe script is seeded, and a second run produced byte-identical output.
Every probe that reports "no effect" has a positive control that fires; see
§6.4.

Purpose: answer, per encoder, what it actually builds, whether its forward pass
is correct, whether it does what its name says, and whether it is correct,
malformed-but-fixable, or needs redoing, with the compatibility cost of each
option.

Repo anchors reviewed:

- `mci_gru/models/temporal.py`, `trunk.py` and `factory.py`
- `mci_gru/config.py` (`ModelConfig`), `configs/config.yaml`, `configs/experiment/*.yaml`
- `docs/ARCHITECTURE.md`, `docs/DEFAULT_EXPERIMENT_RECIPE.md`
- `mci_gru/data/pit.py`, `mci_gru/data/preprocessing.py`, `mci_gru/training/trainer.py`
- `mci_gru/evaluation/experiment_summary.py`
- `docs/research/current/MCI_GRU_TRUNK_ARCHITECTURE_OPPORTUNITIES_2026-09-05.md`
- `docs/research-paper-evaluations/2026-09-05-looped-transformers-recurrent-depth.md`
- the MCI-GRU paper at `archive/pre-cleanup-2026-09:references/2410.20679v3.txt` (arXiv:2410.20679)
- the paper authors' released code, `WinstonLiyt/MCI-GRU` at `fcebd25`, `code/sp500.py`
- issues #131, #197, #198 and #234, and PRs #200 and #202

Environment note, not an encoder finding. `python -c` puts the current directory
first on `sys.path`. So a one-liner run from the standing workspace imports that
worktree's `mci_gru`, not the one under review, even with `PYTHONPATH` set. This
is a variant of the editable-install trap in `CLAUDE.md`. Every probe here pins
the worktree path explicitly and asserts `mci_gru.__file__` before measuring.

## 1. Summary and decision packet

**`transformer` is broken and has never been trainable. `gru_attn` is correct
code with a malformed configuration surface. `legacy` is a faithful
implementation of a weak idea from the paper.** No encoder leaks across stocks
or dates, and no label lookahead exists anywhere in the temporal path. The one
within-window causality violation is the transformer's (#234).

| Encoder | Verdict | Recommended action | Alternatives | Compatibility cost of the recommendation |
| --- | --- | --- | --- | --- |
| `legacy` (`ImprovedGRU` / `AttentionResetGRUCell`) | **Correct as an implementation of the paper's equations 5, 7 and 8**, with one necessary deviation: eq. 6's softmax over a single score is identically 1, so the code uses a sigmoid. The design it implements is not a reset gate (§4.1). | Keep it as the paper-reference encoder. Restore the lost rationale for the sigmoid in the docstring, and correct the wording in `docs/ARCHITECTURE.md` and `README.md`. Add encoder tests (§8). For #131, direction 4 only: `legacy` honours every list entry, so there is nothing to refuse. | (a) Replace the cell with a real gated design: new parameters, and it stops being the paper reference. (b) Retire it, which breaks `paper_faithful.yaml`. | None. Documentation and tests only. State dicts, `resolved_config_sha256` and past-run comparability are all unchanged. |
| `gru_attn` (`GRUWithAttention`) | **Correct forward pass, malformed parameterisation.** The shipped `[32, 10]` builds 2 layers of width 10, and the 32 is dead configuration (#131). The "attention" is parameter-free last-state pooling that starts out uniform (§4.2). | For #131, direction 1 plus a value change. Refuse non-uniform lists for `gru_attn` in `ModelConfig` validation. Change the shipped value to `[10, 10]`, which builds **today's module exactly**. Add direction 4. Pin the encoder keys in the frozen recipe. | (a) Direction 3: split the field into `gru_hidden_size` and `gru_num_layers`. (b) Honour per-layer widths. **Unsafe on its own**: it changes the model behind an unchanged config value and digest (§7.3). | Checkpoints are unaffected: `[10, 10]` reproduces the pinned production model **bit for bit**, matching its keys, parameter count, state-dict signature and scores `[Verified]` (A.5). The digest changes for new runs, which is correct: the text now says what is built. Past `gru_attn` runs stay comparable, because the module is identical. |
| `transformer` (`CausalTransformerEncoder`) | **Redo or retire.** It raises on every training step on torch 2.0.1 and later, including the locked 2.12.1 (#234). On the inference path it runs with no causal mask. It has no positional information. `d_model`, the layer count and the head count are not really configurable (§4.3). | **Retire**, unless an experiment is planned: remove it from `_VALID_TEMPORAL_ENCODERS`, together with the multi-scale transformer mode. The trunk report already defers transformer depth over ten tokens. | Redo it with the smallest correct design in §7.2: an explicit mask, a positional embedding, dedicated fields, and forward-pass tests. | None to checkpoints or evidence. No run can have trained it, and no recorded result uses it (§6.3). Retiring it removes three construct-only tests, which means regenerating `docs/TEST_REGISTRY.md`. |
| `MultiScaleTemporalEncoder`, the shipped topology | **Correct wiring, no leakage.** The slow `Conv1d` is symmetric-padded, so its last step includes 1–2 zero taps past the window end. That is an edge artifact, not a leak. In transformer mode the branches differ in kind. It is not in the paper. | Documentation only: the `configs/config.yaml:26` comment is wrong under `gru_attn`. Any change to the padding changes every checkpoint's outputs under an unchanged digest, so it would need a new flag. | Left-pad the conv behind a new flag. | None for documentation. |

What the maintainer is being asked to decide:

1. Whether to retire `transformer` (recommended) or redo it.
2. Which #131 direction to take for `gru_attn`: refuse plus `[10, 10]` (recommended), or split the field.
3. Whether the frozen recipe should pin `model.temporal_encoder`, `model.use_multi_scale` and `model.gru_hidden_sizes`. Today it inherits all three from `configs/config.yaml`.

## 2. What each encoder builds

### 2.1 Census

All figures `[Verified]` by probes P1 and P1b. Parameter counts are for the
temporal encoder alone at input width F = 23. That F is `[Inferred]` to be the
shipped feature width because it reproduces the trunk report's 6,338-parameter
A1 exactly. Structure does not depend on F; only the counts do.

| `gru_hidden_sizes` | `legacy` builds | params | `gru_attn` builds | params | `transformer` builds | params |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `[32, 10]` (shipped) | cells 23→32→10 | 7,890 | `nn.GRU` 23→10, **2 layers of 10** | 1,730 | d_model 10, **2 heads of 5**, 2 layers | 2,900 |
| `[64, 32]` | 23→64→32 | 30,112 | 2 layers of 32 | 11,872 | d_model 32, 4 heads, 2 layers | 26,176 |
| `[32, 32]` | 23→32→32 | 13,632 | 2 layers of 32 | **11,872** | d_model 32, 4 heads, 2 layers | **26,176** |
| `[16]` | 23→16 | 2,352 | 1 layer of 16 | 2,000 | d_model 16, 4 heads, **2 layers** | 6,944 |
| `[8, 4]` (CI smoke) | 23→8→4 | 1,188 | 2 layers of 4 | 476 | d_model 4, **4 heads of dimension 1** | 584 |
| `[64, 32, 16]` | 23→64→32→16 | 33,040 | 3 layers of 16 | 5,264 | d_model 16, 4 heads, 2 layers | 6,944 |

With `use_multi_scale=true`, which is shipped, `[32, 10]` builds the following:

- **`legacy`: 18,658.** Two `ImprovedGRU`s, plus the conv and the combiner.
- **`gru_attn`: 6,338.** Two 2×10 `nn.GRU`s. The conv holds 2,668 of these, or 42%.
- **`transformer`: 7,508.** The fast branch is the transformer. **The slow branch is `gru_attn`.**

Across the whole model, the temporal encoder is 18.3%, 7.1% and 8.3% of the
parameters respectively.

What the census establishes:

- **`gru_attn` reads interior entries only for their count.** `[32, 10]` and
  `[999, 10]` produce state dicts with identical keys and shapes (P1b), and
  `[64, 32]` and `[32, 32]` build the same module.
- **`transformer` ignores the list length as well as the interior values.**
  The layer count is fixed at 2 by the constructor default, so `[16]` and
  `[64, 32, 16]` build the same module. The list contributes one number:
  `d_model`.
- **The head count is silently reduced**, to the largest divisor of `d_model`
  at or below 4 (P4b):
  - `d_model=10` gives 2 heads;
  - a prime `d_model` gives 1 head;
  - `d_model=4` gives 4 heads of dimension 1.
- The warning text at `temporal.py:171-177` advises changing "nhead", but
  `nhead` is not a config field. `ModelConfig` has none, and the trunk calls
  `CausalTransformerEncoder(input_size, gru_hidden_sizes[-1])` with the
  constructor default of 4 (`trunk.py:179`).
- Under the shipped settings, `legacy` has 4.6 times the single-scale capacity
  of `gru_attn` (7,890 against 1,730) and 2.9 times the multi-scale capacity
  (18,658 against 6,338).

### 2.2 Claims against reality

| Surface | Claim | What the code does | Tag |
| --- | --- | --- | --- |
| `configs/config.yaml:17` | `gru_hidden_sizes: [32, 10]   # GRU hidden layer sizes` | Under the shipped `gru_attn`, both layers are 10 wide. | `[Verified]` P1 |
| `configs/config.yaml:26` | `use_multi_scale: true  # false = plain ImprovedGRU (paper-faithful)` | With the shipped `temporal_encoder: gru_attn`, `false` builds `GRUWithAttention` (`trunk.py:176-181`). Only `paper_faithful.yaml`, which also sets `legacy`, gets `ImprovedGRU`. | `[Verified]` |
| `mci_gru/config.py:483-486` | `"gru_attn"` = "per-step post-hoc attention"; `"transformer"` = causal `nn.TransformerEncoder` | `gru_attn` has one readout at the last step (`temporal.py:99` correctly says "single"). The transformer is not causal on the inference path and does not train. | `[Verified]` P3, P4 |
| `docs/ARCHITECTURE.md:206-209` | `legacy` replaces the reset gate with "a scaled dot-product attention term"; `transformer` is "a causal `nn.TransformerEncoder` stack" | `legacy` uses the sigmoid of one scaled dot product, times a value vector. There is no softmax and no attention over time steps. The transformer claim is false (#234). | `[Verified]` P2b, P4 |
| `README.md:25` | A1 is "an attention-gated GRU" | That describes `legacy`. The shipped A1 is `gru_attn`: a standard GRU with attention pooling. | `[Verified]` |
| `temporal.py:10-18` | `AttentionResetGRUCell` follows "Paper methodology" | It does, except that it uses a sigmoid where the paper writes softmax. The reason was recorded in a code comment at `204b847` and dropped by the `2988fdc` module split (§4.1.3). | `[Verified]` git |
| `temporal.py:49-52` | "Paper uses two layers with hidden sizes [32, 10]" | True of the paper (§4, parameter table). The authors' released code uses one layer of 256 and a different cell (§4.1.4). | `[Verified]` |
| `temporal.py:98-103` | `GRUWithAttention` uses the last size for all layers | Accurate, and has been since it was written in `76f9bce`. The `ModelConfig` docstring was corrected to match in `5759756`. | `[Verified]` P1b, git |
| `temporal.py:158` | "Causal Transformer" | See #234. | `[Verified]` P4, P5 |

## 3. Forward-pass correctness

### 3.1 `legacy`

- **Recurrence** `[Verified]`, `temporal.py:67-79`. The loop is layer-major:
  each layer runs over all T steps from a zero initial state, then feeds its
  sequence to the next layer. For stacked RNNs this is equivalent to
  step-major order.
- **Output.** It returns the final layer's last step, which is paper §3.2.4's
  A1. The trunk projects it with `proj_temporal`, then applies `ln_a1` and the
  node mask (`trunk.py:356-361`).
- **Normalisation and dropout.** The encoder has none. The trunk's LayerNorm
  and dropout apply after projection, and after the stream concatenation.
- **Reference check** `[Verified]` P2. With the attention path neutralised
  (`W_q=0` gives α=0.5; `W_v` weight 0 and bias 2 give `r'=1`), one cell over
  ten steps matches `torch.nn.GRUCell` to **1.5e-7**. For that comparison the
  torch cell's reset gate is held open, and its update gate is sign-mapped,
  because the two use opposite `z` conventions.
  - So the update gate, the candidate and the interpolation are exactly GRU
    semantics. Everything distinctive about the cell lives in `r'`.
  - Restoring a live attention path moves the output by 0.66, which shows the
    probe is sensitive.
- **Cosmetic, harmless** `[Verified]` by reading:
  - `W_z` and `U_z` both carry biases.
  - `U_h`'s bias sits inside `r' ⊙ (U_h h + b)`. That follows torch's `b_hn`
    convention, not paper eq. 8, where `b_h` is outside.
  - `forward` and `forward_sequence` are duplicated code.

### 3.2 `gru_attn`

- **`nn.GRU` standard** `[Verified]`, `temporal.py:122-131`. It runs over
  `(B·N, T, F)`, with a zero initial state and no inter-layer dropout
  (`dropout=0.0`).
- **Readout** `[Verified]` P3. The readout is `LN(h_T + Σ_t α_t h_t)`, with
  `α = softmax_t(h_T·h_t / √d)`.
  - The softmax is over the time axis: shape `(B·N, T)`, with rows summing to
    1.000000.
  - A manual recomputation matches the module exactly.
- **Output.** The pooled, normalised vector is projected as A1.
  `forward_sequence` returns the raw pre-readout GRU states, and those are what
  the optional A1-A2 cross-attention consumes (`trunk.py:376`). The two paths
  are consistent `[Verified]` P3.

### 3.3 `transformer`

- **The input is linearly projected to `d_model`**, then passed through a
  2-layer post-LN `nn.TransformerEncoder` (`dim_feedforward = 4·d_model`,
  `dropout = 0.1`). The last position is taken as the output.
  - Taking the last token is the right choice for a causal encoder.
  - The dropout is hard-wired and not tied to `trunk_dropout` or
    `use_trunk_regularisation`.
- **It cannot run a training step** `[Verified]` P4, P8, and the smoke in
  Appendix B.2. The details are in §4.3 and #234.

### 3.4 Multi-scale

- **Wiring** `[Verified]` P5, P7, P9. `forward` concatenates the fast branch's
  final output with the slow branch's final output, then applies a linear
  `combiner` with no non-linearity.
- **Slow branch.** It encodes a `Conv1d(F, F, k=5, s=2, padding=2)`
  downsampling of the window. The padding is symmetric, so the conv is
  non-causal within the window.
  - The last slow step reads raw steps 6 to 9 plus one zero-padded tap past
    the window end at `his_t=10`. It reads steps 60 to 62 plus two zero taps at
    `his_t=63` (P9).
  - This is an edge artifact: the most recent days are averaged with zeros.
    It is not a leak. The window ends at the last day before the sample date,
    and the slow sequence is consumed only at its end.
- **Fast branch.** Only the fast branch's sequence is exposed for A1-A2
  cross-attention (`temporal.py:270-276`). The first `hasattr` branch of that
  method makes its `isinstance` branch unreachable.

### 3.5 Padded and inactive stocks

- **Scored stocks never reach an encoder with a partially filled window**
  `[Verified]`.
  - The model's `stock_mask` is `tradable = active & feature_ready`
    (`pit.py:249`, wired at `run_experiment.py:233`).
  - `feature_ready_mask` requires a row on every one of the `his_t` days
    before the sample date (`pit.py:188-216`).
  - None of the encoders has a per-step mask, and none needs one.
- **Inactive stocks.** Their windows are zeroed before the encoder
  (`trunk.py:329`). The encoder's constant output for a zero window is
  re-masked after projection (`trunk.py:360-361`).
- **No leakage** `[Verified]` P7. Perturbing one (date, stock) window leaves
  every other (date, stock) output bit-identical, in train and eval mode, for
  every encoder and topology.
- **Outside PIT mode** (`data.use_pit_universe=false`) there is no mask. The
  pivot fills missing days with 0.0 (`preprocessing.py:103-111`), so an encoder
  reads zeros as data. That is a data-contract question owned by the load-path
  audit and #223, not an encoder defect.

## 4. Does each encoder do what its name says?

### 4.1 `AttentionResetGRUCell` against the paper

#### 4.1.1 The equations

Paper §3.2.2 is at `archive/pre-cleanup-2026-09:references/2410.20679v3.txt:66-87`.

| Paper | Code (`temporal.py:33-45`) | Match |
| --- | --- | --- |
| (1) `z_t = σ(W_z x_t + U_z h_{t-1} + b_z)` | `z_t = sigmoid(W_z(x_t) + U_z(h_prev))` | yes |
| (5) `q_t = W_q h_{t-1}`, `k_t = W_k x_t`, `v_t = W_v x_t` | `q_t = W_q(h_prev)`, `k_t = W_k(x_t)`, `v_t = W_v(x_t)`, all of width d_h | yes |
| (6) `a_t = softmax(q_tᵀ k_t / √d_k)`, with `a_t ∈ ℝ` | `alpha_t = sigmoid(sum(q_t * k_t) / sqrt(d_h))` | **deliberate deviation** (4.1.2) |
| (7) `r'_t = a_t v_t` | `r_prime_t = alpha_t * v_t` | yes |
| (8) `h̃'_t = tanh(W_h x_t + r'_t ⊙ (U_h h_{t-1}) + b_h)`; `h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃'_t` | same | yes |
| §4: two layers of 32 then 10 (`:299`, `:302`) | `ImprovedGRU(input, [32, 10])` under `legacy` | yes |

#### 4.1.2 The paper's own equation is degenerate

Eq. 6 applies softmax to a single scalar, and the paper itself says `a_t ∈ ℝ`.
The softmax of one element is 1, for every input. Read literally, the paper's
cell is:

- `r'_t = W_v x_t`, a linear function of the input alone;
- independent of `h_{t-1}`;
- with `W_q` and `W_k` receiving **exactly zero gradient**.

Probe P2d rebuilds the cell as first committed. Its `W_q` and `W_k` gradients
are 0.000, against 0.25 and 0.27 for the shipped sigmoid cell `[Verified]`.

The paper's prose also claims more than either version of the equations can
deliver:

- It says the mechanism "dynamically allocates weights to different time
  steps". Neither the paper's equations nor the code attend over time. Each
  step sees one key, `x_t`.
- It says `r'_t` "dynamically selects the most important parts of the current
  input x_t and the previous hidden state h_{t-1}". But `r'_t` depends on
  `h_{t-1}` only through one scalar.

#### 4.1.3 History in this repository

`[Verified]` from git.

| Commit | Date | Change |
| --- | --- | --- |
| `7ffad5f` | 2026-02-02 | First implemented the cell with `F.softmax(attn_score, dim=-1)` on a size-1 axis: the dead-parameter form. |
| `204b847` | 2026-02-23 | Replaced it with a sigmoid, with a comment giving the reason: softmax over the last dimension "would always yield 1.0 -- a no-op". |
| `2988fdc` | 2026-07-04 | The module split kept the sigmoid and dropped the comment. |

Today nothing in the code or the docs records that `legacy` deviates from the
paper's eq. 6, or why. Any run on the cell between 2026-02-02 and 2026-02-23
trained with `W_q` and `W_k` frozen at their initial values `[Inferred]`. No
current evidence depends on that window. The one notebook from that period that
compared configurations, `Seed_test (1).ipynb`, ran on 2026-02-24; §6.3 covers
it. Which commit it ran against was not established.

#### 4.1.4 The authors' released code is a third design

`WinstonLiyt/MCI-GRU` at `fcebd25`, `code/sp500.py:286-309` and `:392-397`
`[Verified]` by reading. Its `AttentionGRUCell` has three distinctive features:

- **It computes `softmax(Linear(h))` over `dim=1`.** On its
  `(batch, stocks, hidden)` tensors, that is the stock axis. It then
  multiplies the input by the result, so the attention is over input features,
  normalised across stocks.
- **It keeps a standard sigmoid reset gate and update gate.** Neither has a
  bias.
- **Its candidate is `tanh(r ⊙ h)`, with no input term.**

It is one layer of width 256 (`:439`). The model reads `h_gru[-1, :, :]`, the
last element along the **batch** axis.

This implements neither the paper's equations nor a standard GRU. This
repository's `legacy` follows the paper's text, not the released code. So here
"paper-faithful" means faithful to the published equations. It does not mean
faithful to the code the authors released alongside their results. Whether
that code produced the published tables was not established.

#### 4.1.5 What the cell actually computes

- **`r'` is not a gate.**
  - It depends on `h_{t-1}` through one scalar. The Jacobian `∂r'/∂h_{t-1}`
    has rank **1**, against rank **10** for `nn.GRUCell`'s reset gate at the
    same width `[Verified]` P2b.
  - It is not bounded to [0, 1]: `v_t = W_v x_t` is an unconstrained linear
    map. At initialisation **52.1%** of `r'` entries are negative, and the
    range is [−1.05, 1.39] `[Verified]` P2c.
- **The scalar α starts at 0.501 ± 0.020** `[Verified]` P2c. So at
  initialisation `r' ≈ ½ W_v x_t`, and the cell starts out as a GRU whose
  recurrent candidate term is modulated by the input.
- **The term `r' ⊙ U_h h` is a bilinear interaction between input and
  state**, in the family of multiplicative RNNs, scaled by a learned scalar
  `[Inferred]`. It can flip the sign of the recurrent term, which a reset gate
  cannot. It is a legitimate recurrence, but it is not a reset gate, not
  attention over history, and not what the name says.

**Verdict for `legacy`:** it correctly implements the paper, and the paper's
cell is itself malformed at eq. 6. The only way to rescue eq. 6 is the code's
sigmoid substitution. Redoing the cell would mean choosing a new design, not
correcting this one.

### 4.2 `GRUWithAttention`: is the softmax over time correct?

- **It is correct.** The softmax is taken over the T axis, the weights sum to
  1, and the module equals the formula (P3).
- **"Attention" here is parameter-free pooling.**
  - There are no learned query, key or value projections. The query is the
    last state `h_T`, and the keys and values are the raw states, including
    `h_T` itself.
  - GRU states lie in (−1, 1), so every logit satisfies `|s| < √d`. For the
    shipped d = 10 the logit spread is below 6.3, and no step can outweigh
    another by more than about 560:1 `[Inferred]`, by the Cauchy-Schwarz
    inequality.
  - At initialisation, normalised entropy is **0.9995** and the last step's
    weight is 0.107 against a uniform 0.100 `[Verified]` P3. The readout
    starts as mean pooling. It can sharpen only if the GRU learns to align its
    states with `h_T`.
- **Honest name:** "GRU with last-state attention pooling". It is correct and
  standard. The malformation is the width surface (#131), not the readout.

### 4.3 `CausalTransformerEncoder`

| Question | Finding | Tag |
| --- | --- | --- |
| Is the causal mask applied? | **No.** `forward_sequence` calls `self.encoder(z, is_causal=True)` with no `mask` (`temporal.py:195`). PyTorch treats `is_causal` as a *hint that the supplied mask is causal*, not as a request to build one. | `[Verified]` P4 |
| Training path (`train()`, grad on) | **Raises** `RuntimeError: Need attn_mask if specifying the is_causal hint`. The `except TypeError` fallback at `:196-202` never catches it. Every training forward fails, including the full model and CI's smoke with `model.temporal_encoder=transformer`, which exits 1. | `[Verified]` P4, P8, B.2 |
| Inference path (`eval()` + `no_grad()`, which is what `trainer.py:492-501` runs) | The layers take PyTorch's fast path, which **drops the hint and applies no mask**. The output equals an unmasked encoder to **0.0**, and differs from an explicitly masked one by 2.1. Perturbing step 9 moves every earlier position (P5b). | `[Verified]` P4, P5, P5b |
| Which torch versions | `F.multi_head_attention_forward` raises without a mask at every tag read: v2.0.1, v2.1.0, v2.4.0, v2.8.0 and v2.12.1. That includes the lock's pin. v2.0.0 lacks the check. `pyproject.toml` allows `torch>=2.0`. | `[Verified]` PyTorch source at those tags |
| Positional information | **None.** On the inference path the encoder is exactly invariant to permuting the nine history days (difference **3.1e-7**, P6): it is a bag of days. Even with an explicit mask, a 1-layer version is exactly order-blind (1.9e-7). The 2-layer version recovers only weak order information, through causal prefixes: order share **0.198**, against 0.349 for `gru_attn` and 0.569 for `legacy`. | `[Verified]` P6 |
| Is `d_model` divisible by `nhead`? | It is made to hold by silently reducing `nhead`; see §2.1. | `[Verified]` P4b |
| How is the output token chosen? | The last position is taken. That is correct for a causal encoder. On the bidirectional path inference actually runs, with no positional encoding, the last position is distinguished only by carrying `x_T`, not by where it sits. | `[Verified]` |
| Dead parameters, if it could train | None. The explicitly masked variant gets gradient to all 26 parameter tensors (40 in multi-scale). | `[Verified]` P8 |

## 5. No lookahead

AGENTS.md invariant 1 concerns train-period cutoffs for normalisation, graph
edges and labels. The temporal encoders cannot break it on their own:

- The input window is the `his_t` days before the sample date
  (`preprocessing.py:113-117`, `pit.py:214`).
- Readiness is judged on the same span.
- Existing tests cover the label offset.

Within the window, the probes found the following.

| Probe | `legacy` | `gru_attn` | `transformer`, shipped | `transformer`, explicit mask (probe-only reference) |
| --- | --- | --- | --- | --- |
| P5: perturb steps ≥ 6, change at steps < 6 in the consumed sequence (train / eval) | 0 / 0 | 0 / 0 | raises / **1.455** | 0 / 0 |
| same, multi-scale | 0 / 0 | 0 / 0 | raises / **1.455** | 0 / 0 |
| P7: perturb one (date, stock), largest change anywhere else | 0 | 0 | raises (train) / 0 (eval) | 0 |

**The one causality violation is the transformer's, on the only path it can
run.** That violation stays within the window: no future return enters the
model `[Inferred]` from the window construction above. The multi-scale slow conv
is non-causal only within the window, and only its end is consumed (§3.4). No
encoder mixes information across stocks or dates.

## 6. Numerical probes, timing and past experiments

### 6.1 Gradient flow

P8 runs the full model in train mode with a random-projection loss. In every
trainable configuration, every temporal parameter receives a nonzero gradient.
The configurations covered are:

- `legacy` and `gru_attn`;
- single-scale and multi-scale;
- with and without A1-A2 cross-attention.

The only dead-parameter form found is the historical softmax cell (P2d).

### 6.2 Timing

Encoder only, forward plus backward, CPU with 4 threads, batch of
32 dates × 110 stocks × F = 23, median of 5 `[Verified]` (script A.4, output B.4).

| `his_t` | `legacy` | `gru_attn` | `transformer`* | `legacy` + ms | `gru_attn` + ms | `transformer`* + ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 26 ms | 19 ms | 100 ms | 70 ms | 51 ms | 118 ms |
| 63 | 478 ms | 212 ms | 1,139 ms | 772 ms | 346 ms | 1,231 ms |

\* Explicit-mask probe variant, since the shipped module cannot run a backward pass.

The comparison is not capacity-matched: under `[32, 10]`, `legacy` is 32
wide and `gru_attn` is 10 wide. On CPU the fused GRU is 1.4 times faster at
`his_t=10` and 2.2 times faster at `his_t=63`. The phase-2 plan's "3-8×" was a
CUDA claim, and this report does not re-measure it.

### 6.3 Past encoder comparisons and blast radius

`[Verified]` by reading unless tagged.

**Which encoder runs where:**

- **Base config and frozen recipe.** `configs/config.yaml` selects `gru_attn`,
  multi-scale on, `[32, 10]`, `his_t` 10.
  - `docs/DEFAULT_EXPERIMENT_RECIPE.md` overrides only `model.label_t`. It
    inherits every encoder key, so **the recipe's architecture moves with
    `config.yaml`**. That is the gap the recipe closed for data (`:80-83`) but
    not for the model.
- **Code defaults differ.** The `ModelConfig` default is `legacy`
  (`config.py:523`), and so is the factory default (`factory.py:40`). An empty
  config, or a pre-2026-04-21 checkpoint config, builds multi-scale `legacy`.
- **Presets:**
  - `paper_faithful.yaml` is the only one that builds `legacy`, with
    multi-scale off.
  - `long_history_his_t_{21,63,126}.yaml` pin `gru_attn` and vary `his_t`.
  - Every other preset inherits.
- **Generators.** `gen_long_history_pit_eval_nb.py:491` and
  `gen_pit_repeated_seed_replication_nb.py:690` pin `gru_attn`.
  `scripts/ci_smoke.py` sets `[8, 4]`, which under `gru_attn` is 2 layers of 4.
  **CI never exercises `legacy` or `transformer` end to end.**

**When and why the default switched to `gru_attn`.** This happened at `76f9bce`
(2026-04-21). The rationale was speed, not accuracy:

- Phase-2 plan §5, at the tag
  `docs/agent_references/cursor/plans/phase-2-trunk-surgery_a4da800c.plan.md:106-119`,
  says: "At `his_t ≥ 20` the Python recurrent loop dominates epoch time;
  CuDNN-fused GRU gives 3-8× speedup".
- The same plan specifies `hidden_size=hidden_sizes[-1], num_layers=len(hidden_sizes)`.
  So the #131 reinterpretation was designed in, and the 4.6-fold capacity cut
  that came with it (§2.1) was never measured.

**Comparisons that varied the encoder or multi-scale.** Every one of them held
`gru_hidden_sizes = [32, 10]`, and none was capacity-matched.

| Where | Date | Arms | Recorded result |
| --- | --- | --- | --- |
| `Seed_test (1).ipynb` (`git show 08c1f97:"Seed_test (1).ipynb"`) | 2026-02-24 | All arms `legacy`, because the encoder switch did not yet exist. Multi-scale on against off, **confounded** with self-attention, activation and label type. | Cell output only: mean best validation loss 0.001302 (multi-scale) against 0.001291 (paper-faithful). No conclusion written. |
| `archive/pre-cleanup-2026-09:notebooks/ablation_evaluation_loop_colab.ipynb` | 2026-04-28 | `paper_faithful` (`legacy`, single-scale, plus loss and selection changes); `modern_defaults` (`gru_attn`, multi-scale); `transformer_temporal`, whose fast branch is the transformer (`:234-236`) | **None.** The notebook carries no saved output, and no result is recorded anywhere. Had the transformer arm run on any torch from 2.0.1 on, its first training step would have raised `[Inferred]`. The downstream handoff (`docs/research/archive/MODERN_DEFAULTS_HANDOFF_2026.md:14`) calls `modern_defaults` "the leading candidate" without isolating the encoder. |

Comparisons that were planned but never run:

- `archive/pre-cleanup-2026-09:docs/ARCHITECTURE_REVIEW.md:337`, `gru_attn`
  against `transformer`;
- `docs/research/archive/MCI_GRU_PROGRAM_MAP_2026-06-19.md:443`, all three
  encoders "holding graph and loss fixed", but not capacity;
- the trunk map's arm C4, which is gated on #131.

**No recorded evidence anywhere isolates the temporal encoder.**

**Affected current evidence.** None of these reports states its encoder. The
encoder is inferred from each run's generator plus `configs/config.yaml`
`[Inferred]`. Runs after 2026-07-26 carry a `resolved_config.json` on Drive
that would confirm it.

| Report | Encoder its runs used | What the finding means for it |
| --- | --- | --- |
| `GRAPH_SPECIFICATION_ABLATION_2026-09-01.md`, `GRAPH_PAIRED_REANALYSIS_2026-09-02.md` | `gru_attn`, multi-scale, 2×10 | Arm-against-arm comparisons at a fixed encoder stay valid as comparisons. Their absolute level is conditional on a 1,730-parameter-per-branch encoder whose capacity was never chosen on evidence. |
| `LONG_HISTORY_PIT_EVAL_RESULTS_2026-05-18.md` | `gru_attn` pinned; `his_t` 10, 21, 63, 126 | Valid as an `his_t` sweep at a fixed encoder. Changing `his_t` also changes the slow branch's length. |
| `PIT_REPEATED_SEED_OPTION_A_RESULTS_2026-05-21.md`, `PIT_MASKED_PANEL_2022_2025_FULL_RUN_REPORT_2026-05-16.md`, `SP500_PIT_GICS_TOP10_MULTIYEAR_BASELINE_2026-06-23.md`, the four `ISSUE8_*` volatility-targeting reports | `gru_attn`, multi-scale, 2×10 (pinned or inherited) | Same: conditional on the encoder, not invalidated by it. |
| `MCI_GRU_TRUNK_ARCHITECTURE_OPPORTUNITIES_2026-09-05.md` | `gru_attn`, multi-scale, as it states (`:187`) | Its statement that long-history results "moved returns through `his_t`, not through the encoder" (`:1029-1030`) rests on no encoder comparison. Nothing here contradicts its arm C4 being gated on #131. |
| `docs/research-paper-evaluations/2026-09-05-looped-transformers-recurrent-depth.md` | `gru_attn`, multi-scale (`:139`) | Its reason for deferring encoder work, at `:198-201` ("not capacity-matched"), is confirmed. |

**No current report used `legacy` or `transformer`.** The retired
`seed_results/` and `paper_trade/` checkpoints, readable at the tag, were
single-scale `legacy` at `his_t` 60 and predate the encoder switch.

### 6.4 Probe sensitivity (mutation checks)

Every "no effect" result above is backed by a positive control that fires.

| Probe | Null result reported | Positive control | Control result |
| --- | --- | --- | --- |
| P2 `GRUCell` reduction | 1.5e-7 | Restore a live attention path | 0.66 |
| P5 causality | 0 for the GRUs | The shipped transformer, and a single-step perturbation (P5b) | 1.455; all positions move |
| P6 order blindness | 3.1e-7 for the shipped transformer | The GRUs, and the explicitly masked transformer | 0.12 to 0.70 |
| P7 cross-stock and cross-date | 0 everywhere | `LeakAcross`: add 0.1 × the mean over stocks, or over dates, before encoding (P7b) | 3.7e-2 and 5.9e-2 |
| P8 dead parameters | none | The historical softmax cell (P2d) | `W_q` and `W_k` gradients exactly 0 |

## 7. Smallest correct designs and their cost

### 7.1 How compatibility is priced in this repository

`[Verified]` by reading.

- **Checkpoints are bare state dicts.** They are saved at `trainer.py:337` and
  reloaded strictly within a run at `trainer.py:635`. None are committed, and
  every Drive checkpoint behind current evidence is `gru_attn`, multi-scale,
  `[32, 10]`.
- **The architecture guard is `test_default_model_is_unchanged_from_main`**
  (`tests/test_market_latent_state.py:403-533`). For the production config it
  pins 75 state-dict keys, 84,026 parameters and signature `3d03a4f3…`. For the
  empty `legacy` config it pins 112 keys and 91,915 parameters.
- **`resolved_config_sha256`** is the SHA-256 of `resolved_config.json`
  (`experiment_summary.py:49-70`, since `01cefa9`, 2026-07-26), written from
  `asdict(config)`. It therefore includes every `ModelConfig` field.

### 7.2 Per encoder

- **`legacy`: no code change.**
  - Restore the eq. 6 rationale as a docstring note.
  - Correct `docs/ARCHITECTURE.md:206-209` and `README.md:25`.
  - Add the §8 tests.
  - Direction 4: record the effective shape.
  - Cost: none.
- **`gru_attn`: refuse-and-restate.** This is the recommendation.
  - Refuse a non-uniform list for `gru_attn`, and for the multi-scale slow
    branch in transformer mode, in `ModelConfig.__post_init__`. That is the
    Hydra path. `ModelConfig` already refuses there a flag combination it
    cannot honour, citing "the defect family recorded in #131"
    (`config.py:557-565`).
  - Change `configs/config.yaml` to `gru_hidden_sizes: [10, 10]`. That builds
    the identical module, so every checkpoint still loads.
  - `[Verified]` A.5, which replays `test_default_model_is_unchanged_from_main`
    with its own helpers and only the list swapped:
    - `[10, 10]` matches the pinned production model's keys, parameter count,
      signature and scores;
    - `[32, 10]` also matches, as a control;
    - `[32, 32]` does not match, as a positive control.

    So the guard test needs only its config literal edited, and it doubles as
    that change's mutation check.
  - The digest changes for new runs. That is honest: the text now matches the
    model.
  - Decide deliberately whether `create_model(dict)` also refuses. That path
    has no production caller left other than `run_experiment.py:254`. Making it
    refuse would stop old run-folder configs that say `[32, 10]` from
    rebuilding.
- **`gru_attn`, alternative: split the field** (#131 direction 3).
  - Add `gru_hidden_size` and `gru_num_layers`, and keep `gru_hidden_sizes`
    for `legacy` only.
  - Defaults that reproduce 2×10 keep checkpoints loadable.
  - New fields change every future digest.
  - `ci_smoke.py`, the generators and the tests must move to the new keys.
- **`transformer`: retire.** This is the recommendation.
  - Remove the value from `_VALID_TEMPORAL_ENCODERS`, and remove the class and
    the multi-scale transformer mode.
  - Cost: three construct-only tests and a `docs/TEST_REGISTRY.md` regeneration.
    No checkpoint, run or report depends on it (§6.3).
- **`transformer`, if kept, the smallest correct redo:**
  - pass `mask=nn.Transformer.generate_square_subsequent_mask(T)`, with
    `is_causal=True` as a hint only;
  - add a learned positional embedding over `his_t`, or a sinusoidal one;
  - add dedicated fields `transformer_d_model`, `transformer_nhead`
    (**refuse** non-divisors rather than reduce) and `transformer_num_layers`;
  - tie dropout to `trunk_dropout`;
  - add tests that run a training step, check causality as in P5, and check
    order sensitivity as in P6.
  - Cost: none to history. New fields change future digests.
- **All encoders, #131 direction 4.** Write the effective temporal shape into
  the run's artifacts, beside the resolved config: class, per-layer widths,
  heads, and parameter count. The census in P1 is that record. It is additive.

### 7.3 The one unsafe move

**Changing what an existing `(key, value)` builds, without changing the key or
the value, silently breaks comparability.** Making `gru_attn` honour `[32, 10]`
as 32 then 10 is the obvious example.

The resolved-config digest hashes the config text, not the architecture. So the
same digest would certify two different models as one configuration.

- **A width change fails loudly in one place.** Checkpoints would fail to load,
  because their shapes change. The digest and every run-to-run comparison would
  still pass, silently.
- **A change that keeps shapes fails nowhere.** Examples are changing the slow
  conv's padding, or `legacy`'s gate arithmetic, in place. Old checkpoints
  would load and produce different scores under the same digest.

Any such change must ship behind a new key or a changed value.

## 8. Test gaps

`[Verified]` by reading `tests/`.

- **No encoder has a causality test**, and no test runs a transformer forward
  pass. The three transformer tests in `tests/test_mci_gru_phase2.py`
  (`:117`, `:124`, `:141`) only construct it.
- **There is no comparison against `nn.GRUCell` or `nn.GRU`**, and no
  parameter-gradient test. Only input gradients are checked.
- `test_temporal_legacy_vs_gru_attn_shape_and_grad` (`:52`) calls `backward()`
  on the `legacy` path and asserts nothing about it.
- `tests/test_temporal_encoder.py` was specified by the phase-2 plan (`:107`)
  and never created.

Probes P2, P5, P6, P7 and P8 are the natural shape for those tests. Each needs
its mutation check, as recorded in §6.4.

## 9. Not established

- **The encoders' effect on model quality.** No comparison isolates the encoder
  at matched capacity (§6.3). This report measures mechanics only. Arm C4 of
  the trunk ablation is where that would be measured, after #131.
- **Learned behaviour.** No trained checkpoint was read, because the protected
  checkout's run folders were out of bounds for this session. Two things
  therefore remain open:
  - whether trained `legacy` cells keep α near 0.5, or `r'` mostly negative;
  - whether trained `gru_attn` readouts stay near-uniform.
- **Transformer behaviour on torch 2.0.0**, the one admitted version without
  the check. It was not exercised.
- **CUDA timing.**

## Appendix A. Reproduction

Run from any directory. The interpreter is the shared venv, and the only
argument is the worktree root, which the scripts pin and assert:

```powershell
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe probe_temporal_encoders.py <worktree root>
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe repro_transformer.py <worktree root>
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe smoke_per_encoder.py <worktree root>
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe probe_timing.py <worktree root>
C:\Users\magil\MCI-GRU\.venv\Scripts\python.exe probe_guard_equivalence.py <worktree root>
```

The PyTorch-version check in §4.3 reads
`torch/nn/functional.py` at each tag through
`gh api "repos/pytorch/pytorch/contents/torch/nn/functional.py?ref=<tag>" -H "Accept: application/vnd.github.raw"`
and searches for `Need attn_mask if specifying the is_causal hint`.

### A.1 `probe_temporal_encoders.py`

The numerical probes P0 to P9.

```python
"""Temporal-encoder audit probes (issue #233).

Usage (Windows, from anywhere):
    C:\\Users\\magil\\MCI-GRU\\.venv\\Scripts\\python.exe probe_temporal_encoders.py <worktree root>

The worktree root is pinned on sys.path and asserted, because the shared venv's
editable install otherwise imports the protected checkout (see CLAUDE.md).
Every probe is seeded; re-running prints identical numbers on the same torch.
"""

import sys

WT = sys.argv[1]
sys.path.insert(0, WT)

import math  # noqa: E402
import warnings  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mci_gru  # noqa: E402

assert mci_gru.__file__.lower().startswith(WT.lower()), mci_gru.__file__

from mci_gru.models.factory import create_model  # noqa: E402
from mci_gru.models.temporal import (  # noqa: E402
    AttentionResetGRUCell,
    CausalTransformerEncoder,
    GRUWithAttention,
    ImprovedGRU,
    MultiScaleTemporalEncoder,
)

warnings.simplefilter("ignore")  # nhead-reduction warnings are reported explicitly below
torch.set_printoptions(precision=4, sci_mode=True)

F_IN = 23  # input width of the shipped feature set (trunk report: 6,338 A1 params => F = 23)
T = 10  # model.his_t
B, N = 3, 7  # dates per batch, stocks per date

# The shipped Hydra base (configs/config.yaml) model block, minus the encoder keys.
BASE = {
    "hidden_size_gat1": 32,
    "output_gat1": 4,
    "gat_heads": 4,
    "hidden_size_gat2": 32,
    "num_hidden_states": 32,
    "cross_attn_heads": 4,
    "slow_kernel": 5,
    "slow_stride": 2,
    "use_self_attention": True,
    "activation": "elu",
    "output_activation": "none",
    "latent_init_scale": 0.02,
    "use_group_type_embed": True,
    "use_trunk_regularisation": True,
    "trunk_dropout": 0.1,
    "use_nn_multihead_attention": True,
    "use_a1_a2_cross_attention": False,
    "cross_a2_num_heads": 4,
    "cross_section_block": "legacy",
    "market_latent_mode": "static",
    "edge_feature_dim": 4,
    "drop_edge_p": 0.1,
}


def header(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def describe(enc):
    if isinstance(enc, ImprovedGRU):
        dims = [enc.layers[0].input_size] + [c.hidden_size for c in enc.layers]
        return f"ImprovedGRU cells {'->'.join(map(str, dims))}"
    if isinstance(enc, GRUWithAttention):
        g = enc.gru
        return f"nn.GRU {g.input_size}->{g.hidden_size} x{g.num_layers} layers (+attn readout, LN)"
    if isinstance(enc, CausalTransformerEncoder):
        lyr = enc.encoder.layers[0]
        return (
            f"Transformer d_model={enc.d_model} nhead={enc.nhead} "
            f"(head_dim={enc.d_model // enc.nhead}) layers={len(enc.encoder.layers)} "
            f"ff={lyr.linear1.out_features} dropout={lyr.dropout.p} no-PE"
        )
    if isinstance(enc, MultiScaleTemporalEncoder):
        return (
            f"MultiScale[fast: {describe(enc.fast_gru)} | slow: {describe(enc.slow_gru)} "
            f"| conv {enc.slow_aggregator.kernel_size[0]}/{enc.slow_aggregator.stride[0]} "
            f"| combiner {enc.combiner.in_features}->{enc.combiner.out_features}]"
        )
    return type(enc).__name__


class ExplicitMaskTransformer(CausalTransformerEncoder):
    """PROBE-ONLY reference: the shipped module with an explicit causal mask passed.

    Not a proposed fix; it exists so the probes can say what a correctly masked
    version of the same architecture does. Swapped in via ``__class__``.
    """

    def forward_sequence(self, x):
        batch, num_stocks, tlen, _ = x.shape
        z = self.input_proj(x).reshape(batch * num_stocks, tlen, self.d_model)
        mask = nn.Transformer.generate_square_subsequent_mask(tlen, device=z.device, dtype=z.dtype)
        out = self.encoder(z, mask=mask, is_causal=True)
        return out.view(batch, num_stocks, tlen, self.d_model)


def model_for(encoder, hidden, multi_scale, **extra):
    """Build through create_model (the shipped path). encoder 'transformer*' = explicit-mask probe."""
    patched = encoder.endswith("*")
    cfg = dict(
        BASE,
        gru_hidden_sizes=list(hidden),
        temporal_encoder=encoder.rstrip("*"),
        use_multi_scale=multi_scale,
    )
    cfg.update(extra)
    m = create_model(F_IN, cfg)
    if patched:
        enc = m.temporal_encoder
        target = enc.fast_gru if isinstance(enc, MultiScaleTemporalEncoder) else enc
        target.__class__ = ExplicitMaskTransformer
    return m


def attempt(fn):
    try:
        return fn(), None
    except Exception as ex:  # noqa: BLE001
        return None, f"{type(ex).__name__}: {str(ex)[:60]}"


ENCODERS = ("legacy", "gru_attn", "transformer", "transformer*")


# ---------------------------------------------------------------------------
header("P0 environment")
print("torch", torch.__version__, "| mci_gru from", mci_gru.__file__)

# ---------------------------------------------------------------------------
header("P1 census: what each (encoder, gru_hidden_sizes, use_multi_scale) builds")
print(f"input_size F={F_IN}; params are the temporal encoder only (model.temporal_encoder)")
print(f"{'encoder':<12}{'hidden':<14}{'ms':<4}{'params':>8}{'out':>5}  built")
for ms in (False, True):
    for enc_name in ("legacy", "gru_attn", "transformer"):
        for hs in ([32, 10], [64, 32], [32, 32], [16], [8, 4], [64, 32, 16]):
            torch.manual_seed(0)
            m = model_for(enc_name, hs, ms)
            e = m.temporal_encoder
            print(
                f"{enc_name:<12}{str(hs):<14}{'Y' if ms else 'N':<4}{nparams(e):>8}"
                f"{e.output_size:>5}  {describe(e)}"
            )

header("P1b interior gru_hidden_sizes entries are ignored by gru_attn and transformer")
for enc_name in ("gru_attn", "transformer"):
    a = model_for(enc_name, [32, 10], False).temporal_encoder.state_dict()
    b = model_for(enc_name, [999, 10], False).temporal_encoder.state_dict()
    same = a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)
    print(f"{enc_name}: state_dict shapes identical for [32,10] and [999,10]: {same}")
a = model_for("legacy", [32, 10], False).temporal_encoder
b = model_for("legacy", [999, 10], False).temporal_encoder
print(f"legacy: params [32,10]={nparams(a)}  [999,10]={nparams(b)}  (interior entry is honoured)")

header("P1c full-model parameter share of the temporal encoder (shipped base, multi-scale)")
for enc_name in ("legacy", "gru_attn", "transformer"):
    m = model_for(enc_name, [32, 10], True)
    print(
        f"{enc_name:<12} total={nparams(m):>7}  temporal={nparams(m.temporal_encoder):>6} "
        f"({100 * nparams(m.temporal_encoder) / nparams(m):.1f}%)"
    )

# ---------------------------------------------------------------------------
header("P2 legacy cell: attention neutralised (r' == 1) vs torch.nn.GRUCell (reset open)")


def run_cell_seq(cell_fn, x, h):
    outs = []
    for t in range(x.shape[1]):
        h = cell_fn(x[:, t, :], h)
        outs.append(h)
    return torch.stack(outs, 1)


H = 10
torch.manual_seed(1)
cell = AttentionResetGRUCell(F_IN, H)
ref = nn.GRUCell(F_IN, H)
with torch.no_grad():
    cell.W_q.weight.zero_()
    cell.W_q.bias.zero_()  # q = 0  -> score 0 -> alpha = sigmoid(0) = 0.5
    cell.W_v.weight.zero_()
    cell.W_v.bias.fill_(2.0)  # v = 2 -> r' = alpha * v = 1
    for p in (ref.weight_ih, ref.weight_hh, ref.bias_ih, ref.bias_hh):
        p.zero_()
    ref.bias_ih[:H] = 40.0  # torch reset gate r = sigmoid(40) == 1.0 in float32
    # torch: h' = (1 - z) * n + z * h ; this cell: h' = (1 - z) * h + z * h~  => z_torch = 1 - z_cell
    ref.weight_ih[H : 2 * H] = -cell.W_z.weight
    ref.bias_ih[H : 2 * H] = -cell.W_z.bias
    ref.weight_hh[H : 2 * H] = -cell.U_z.weight
    ref.bias_hh[H : 2 * H] = -cell.U_z.bias
    ref.weight_ih[2 * H :] = cell.W_h.weight
    ref.bias_ih[2 * H :] = cell.W_h.bias
    ref.weight_hh[2 * H :] = cell.U_h.weight
    ref.bias_hh[2 * H :] = cell.U_h.bias
x = torch.randn(64, T, F_IN)
h0 = torch.zeros(64, H)
with torch.no_grad():
    y_cell = run_cell_seq(cell, x, h0)
    y_ref = run_cell_seq(ref, x, h0)
print(f"max |cell - GRUCell| over {T} steps: {(y_cell - y_ref).abs().max().item():.3e}")
# Sensitivity check on the probe itself: restore a live attention path and the match must break.
with torch.no_grad():
    torch.manual_seed(2)
    live = AttentionResetGRUCell(F_IN, H)
    live.W_z.load_state_dict(cell.W_z.state_dict())
    live.U_z.load_state_dict(cell.U_z.state_dict())
    live.W_h.load_state_dict(cell.W_h.state_dict())
    live.U_h.load_state_dict(cell.U_h.state_dict())
    y_live = run_cell_seq(live, x, h0)
print(f"probe sensitivity (live attention path): max |diff| = {(y_live - y_ref).abs().max().item():.3e}")

header("P2b legacy cell: r' depends on h_{t-1} through ONE scalar (rank of d r'/d h)")
torch.manual_seed(3)
cell = AttentionResetGRUCell(F_IN, H)
gru = nn.GRUCell(F_IN, H)
ranks_attn, ranks_gru = [], []
for i in range(20):
    xt = torch.randn(F_IN)
    hp = torch.randn(H) * 0.5

    def r_attn(h, xt=xt):
        q, k, v = cell.W_q(h), cell.W_k(xt), cell.W_v(xt)
        return torch.sigmoid((q * k).sum() / math.sqrt(H)) * v

    def r_gru(h, xt=xt):
        gi = F.linear(xt, gru.weight_ih[:H], gru.bias_ih[:H])
        gh = F.linear(h, gru.weight_hh[:H], gru.bias_hh[:H])
        return torch.sigmoid(gi + gh)

    Ja = torch.autograd.functional.jacobian(r_attn, hp)
    Jg = torch.autograd.functional.jacobian(r_gru, hp)
    ranks_attn.append(int(torch.linalg.matrix_rank(Ja, atol=1e-7)))
    ranks_gru.append(int(torch.linalg.matrix_rank(Jg, atol=1e-7)))
print(f"rank d r'/d h_(t-1), attention reset (H={H}): {sorted(set(ranks_attn))}")
print(f"rank d r /d h_(t-1), nn.GRUCell reset  (H={H}): {sorted(set(ranks_gru))}")

header("P2c legacy cell: is r' a gate? range of r' = alpha * W_v x at init")
torch.manual_seed(4)
for hs in ([32, 10],):
    enc = ImprovedGRU(F_IN, hs)
    xb = torch.randn(B, N, T, F_IN)
    rs, alphas = [], []
    with torch.no_grad():
        layer_in = xb
        for li, c in enumerate(enc.layers):
            h = torch.zeros(B, N, c.hidden_size)
            outs = []
            for t in range(T):
                xt = layer_in[:, :, t, :]
                q, k, v = c.W_q(h), c.W_k(xt), c.W_v(xt)
                a = torch.sigmoid((q * k).sum(-1, keepdim=True) / math.sqrt(c.hidden_size))
                rs.append((a * v).flatten())
                alphas.append(a.flatten())
                h = c(xt, h)
                outs.append(h)
            layer_in = torch.stack(outs, 2)
    r = torch.cat(rs)
    al = torch.cat(alphas)
    print(
        f"hidden={hs}: r' fraction <0 = {(r < 0).float().mean():.3f}, >1 = {(r > 1).float().mean():.3f}, "
        f"min={r.min():.3f} max={r.max():.3f}; alpha mean={al.mean():.3f} sd={al.std():.4f}"
    )

header("P2d the paper's eq.6 literally (softmax over one score) and the Feb-2026 code: dead W_q/W_k")


class SoftmaxCell(AttentionResetGRUCell):
    """AttentionResetGRUCell as first committed (7ffad5f, 2026-02-02): softmax over a size-1 axis."""

    def forward(self, x_t, h_prev):
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        q_t, k_t, v_t = self.W_q(h_prev), self.W_k(x_t), self.W_v(x_t)
        score = torch.sum(q_t * k_t, dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        alpha_t = F.softmax(score, dim=-1)
        h_tilde = torch.tanh(self.W_h(x_t) + (alpha_t * v_t) * self.U_h(h_prev))
        return (1 - z_t) * h_prev + z_t * h_tilde


for cls in (SoftmaxCell, AttentionResetGRUCell):
    torch.manual_seed(5)
    c = cls(F_IN, H)
    xs = torch.randn(32, T, F_IN)
    y = run_cell_seq(c, xs, torch.zeros(32, H))
    (y * torch.randn_like(y)).sum().backward()
    g = {n: p.grad.abs().max().item() for n, p in c.named_parameters()}
    print(f"{cls.__name__:<22} max|grad| W_q.weight={g['W_q.weight']:.3e} W_k.weight={g['W_k.weight']:.3e} "
          f"W_v.weight={g['W_v.weight']:.3e}")

# ---------------------------------------------------------------------------
header("P3 gru_attn readout: softmax axis, weight mass, LayerNorm placement, dropout")
torch.manual_seed(6)
g = GRUWithAttention(F_IN, [32, 10])
xb = torch.randn(B, N, T, F_IN)
with torch.no_grad():
    out, _ = g.gru(xb.reshape(B * N, T, F_IN))
    hT = out[:, -1, :]
    scores = (out * hT.unsqueeze(1)).sum(-1) * g.scale
    alpha = F.softmax(scores, dim=-1)
    ent = -(alpha * alpha.clamp_min(1e-12).log()).sum(-1) / math.log(T)
    y_manual = g.ln(hT + (alpha.unsqueeze(-1) * out).sum(1)).view(B, N, -1)
    y_mod = g(xb)
print(f"alpha shape {tuple(alpha.shape)} (rows=B*N sequences, cols=T steps); row sums in "
      f"[{alpha.sum(-1).min():.6f}, {alpha.sum(-1).max():.6f}]")
print(f"module output == manual LN(h_T + softmax_t(h_T.h_t/sqrt(d)) @ H): "
      f"{torch.allclose(y_manual, y_mod, atol=1e-6)}")
print(f"init normalised attention entropy mean={ent.mean():.4f} (1.0 = uniform); "
      f"weight on last step mean={alpha[:, -1].mean():.4f} vs uniform {1 / T:.4f}")
print(f"nn.GRU inter-layer dropout = {g.gru.dropout}; layers = {g.gru.num_layers}; width = {g.gru.hidden_size}")
print(f"forward_sequence returns pre-readout GRU states: "
      f"{torch.allclose(g.forward_sequence(xb).reshape(B * N, T, -1), out, atol=1e-6)}")

# ---------------------------------------------------------------------------
header("P4 transformer: is the causal mask applied?  forward_sequence vs explicit masks")
torch.manual_seed(7)
tr = CausalTransformerEncoder(F_IN, 10)
xb = torch.randn(B, N, T, F_IN)
causal = nn.Transformer.generate_square_subsequent_mask(T)
for mode, grad in (("train", True), ("eval", True), ("eval", False)):
    getattr(tr, mode)()
    with torch.set_grad_enabled(grad):
        torch.manual_seed(100)
        seq, err = attempt(lambda: tr.forward_sequence(xb))
        z = tr.input_proj(xb).reshape(B * N, T, tr.d_model)
        torch.manual_seed(100)
        with_mask = tr.encoder(z, mask=causal).view(B, N, T, -1)
        torch.manual_seed(100)
        no_mask = tr.encoder(z).view(B, N, T, -1)
    label = f"[{mode}, grad {'on' if grad else 'off'}]"
    if seq is None:
        print(f"{label:<20} forward_sequence RAISES {err}")
        continue
    print(
        f"{label:<20} max|seq - explicit causal mask| = {(seq - with_mask).abs().max():.3e}; "
        f"max|seq - NO mask| = {(seq - no_mask).abs().max():.3e}"
    )
print("trainer paths: training step = [train, grad on]; validation/test = [eval, grad off] "
      "(mci_gru/training/trainer.py model.eval() + torch.no_grad())")

header("P4b transformer: nhead actually used for each d_model")
for d in (4, 6, 7, 8, 10, 12, 16, 32):
    print(f"d_model={d:<3} nhead={CausalTransformerEncoder(F_IN, d).nhead}")

# ---------------------------------------------------------------------------
header("P5 no lookahead inside the window: perturb steps >= t0, measure change at steps < t0")
print("(sequence = the one the trunk consumes: forward_sequence, or the fast branch in multi-scale)")


def seq_of(enc, x):
    if isinstance(enc, MultiScaleTemporalEncoder):
        return enc.forward_fast_sequence(x)
    return enc.forward_sequence(x)


t0 = 6
torch.manual_seed(8)
x = torch.randn(B, N, T, F_IN)
xp = x.clone()
xp[:, :, t0:, :] += 3.0 * torch.randn_like(xp[:, :, t0:, :])
for ms in (False, True):
    for enc_name in ENCODERS:
        enc = model_for(enc_name, [32, 10], ms).temporal_encoder
        for mode, grad in (("train", True), ("eval", False)):
            getattr(enc, mode)()
            tag = f"{enc_name}{'+ms' if ms else ''}"
            with torch.set_grad_enabled(grad):
                torch.manual_seed(9)
                s0, err = attempt(lambda: seq_of(enc, x))
                torch.manual_seed(9)
                s1, _ = attempt(lambda: seq_of(enc, xp))
            if err:
                print(f"{tag:<18}[{mode:<5}] RAISES {err}")
                continue
            early = (s1[:, :, :t0] - s0[:, :, :t0]).abs().max().item()
            late = (s1[:, :, t0:] - s0[:, :, t0:]).abs().max().item()
            print(f"{tag:<18}[{mode:<5}] steps<t0 max|d|={early:.3e}   steps>=t0 max|d|={late:.3e}")

header("P5b transformer (inference path): which positions move when ONE step is perturbed")
for enc_name in ("transformer", "transformer*"):
    tr = model_for(enc_name, [32, 10], False).temporal_encoder.eval()
    torch.manual_seed(10)
    x1 = torch.randn(1, 1, T, F_IN)
    with torch.no_grad():
        s0 = tr.forward_sequence(x1)
        for tp in (T - 1, T // 2):
            xq = x1.clone()
            xq[0, 0, tp] += 5.0
            d = (tr.forward_sequence(xq) - s0).abs().amax(-1)[0, 0]
            print(f"{enc_name:<13} perturb step {tp}: per-position max|d| = "
                  + " ".join(f"{v:.0e}" for v in d.tolist()))

# ---------------------------------------------------------------------------
header("P6 positional information: permute the first T-1 steps, keep the last step fixed")
print("order share = |f(permuted history) - f(x)| / |f(fresh random history) - f(x)|, inference path")
torch.manual_seed(11)
x = torch.randn(64, 1, T, F_IN)
perm = torch.randperm(T - 1)
xperm = torch.cat([x[:, :, perm], x[:, :, -1:]], dim=2)
xother = torch.cat([torch.randn(64, 1, T - 1, F_IN), x[:, :, -1:]], dim=2)
for ms in (False, True):
    for enc_name in ENCODERS:
        torch.manual_seed(12)
        enc = model_for(enc_name, [32, 10], ms).temporal_encoder.eval()
        with torch.no_grad():
            y, yp, yo = enc(x), enc(xperm), enc(xother)
        d_perm = (yp - y).norm(dim=-1).mean().item()
        d_other = (yo - y).norm(dim=-1).mean().item()
        tag = f"{enc_name}{'+ms' if ms else ''}"
        print(f"{tag:<18} |f(perm)-f(x)|={d_perm:.3e}  |f(new history)-f(x)|={d_other:.3e}  "
              f"order share={d_perm / d_other:.3f}")
print("one-layer explicit-mask transformer (exact invariance expected: no PE, one causal layer):")
torch.manual_seed(12)
tr1 = CausalTransformerEncoder(F_IN, 10, num_layers=1)
tr1.__class__ = ExplicitMaskTransformer
tr1.eval()
with torch.no_grad():
    print(f"  |f(perm)-f(x)| = {(tr1(xperm) - tr1(x)).norm(dim=-1).mean():.3e}")

# ---------------------------------------------------------------------------
header("P7 leakage across stocks and dates: perturb one (date, stock), others must not move")
for ms in (False, True):
    for enc_name in ENCODERS:
        enc = model_for(enc_name, [32, 10], ms).temporal_encoder
        tag = f"{enc_name}{'+ms' if ms else ''}"
        for mode, grad in (("train", True), ("eval", False)):
            getattr(enc, mode)()
            torch.manual_seed(13)
            x = torch.randn(B, N, T, F_IN)
            xp = x.clone()
            xp[1, 2] += 3.0 * torch.randn(T, F_IN)
            with torch.set_grad_enabled(grad):
                torch.manual_seed(14)
                y0, err = attempt(lambda: enc(x))
                torch.manual_seed(14)
                y1, _ = attempt(lambda: enc(xp))
            if err:
                print(f"{tag:<18}[{mode:<5}] RAISES {err}")
                continue
            moved = (y1 - y0).abs().amax(-1)
            others = moved.clone()
            others[1, 2] = 0
            print(f"{tag:<18}[{mode:<5}] target max|d|={moved[1, 2]:.3e}  "
                  f"every other (date, stock) max|d|={others.max():.3e}")

header("P7b positive controls: the P7 probe must fire on a deliberately leaky encoder")


class LeakAcross(nn.Module):
    """PROBE-ONLY mutation: adds a mean over one axis before encoding, i.e. a real leak."""

    def __init__(self, inner, dim):
        super().__init__()
        self.inner, self.dim = inner, dim

    def forward(self, x):
        return self.inner(x + 0.1 * x.mean(dim=self.dim, keepdim=True))


for dim, what in ((1, "across stocks"), (0, "across dates")):
    enc = LeakAcross(model_for("gru_attn", [32, 10], True).temporal_encoder, dim).eval()
    torch.manual_seed(13)
    x = torch.randn(B, N, T, F_IN)
    xp = x.clone()
    xp[1, 2] += 3.0 * torch.randn(T, F_IN)
    with torch.no_grad():
        moved = (enc(xp) - enc(x)).abs().amax(-1)
    others = moved.clone()
    others[1, 2] = 0
    print(f"leak {what:<14} every other (date, stock) max|d|={others.max():.3e}  (must be > 0)")

# ---------------------------------------------------------------------------
header("P8 gradient flow: temporal parameters with no gradient (full model, train mode)")


def graph_inputs(b, n, edim):
    xg = torch.randn(b * n, F_IN)
    src, dst = [], []
    for d in range(b):
        for i in range(n):
            for j in range(n):
                if i != j and (i + j) % 2 == 0:
                    src.append(d * n + i)
                    dst.append(d * n + j)
    ei = torch.tensor([src, dst], dtype=torch.long)
    ew = torch.rand(ei.shape[1], edim)
    return xg, ei, ew


for ms in (False, True):
    for enc_name in ENCODERS:
        for a1a2 in (False, True):
            torch.manual_seed(15)
            m = model_for(enc_name, [32, 10], ms, use_a1_a2_cross_attention=a1a2).train()
            x = torch.randn(B, N, T, F_IN)
            xg, ei, ew = graph_inputs(B, N, 4)
            tag = f"{enc_name}{'+ms' if ms else ''}{'+a1a2' if a1a2 else ''}"
            y, err = attempt(lambda: m(x, xg, ei, ew, N))
            if err:
                print(f"{tag:<26} full-model training forward RAISES {err}")
                continue
            (y * torch.randn_like(y)).sum().backward()
            dead = [
                n for n, p in m.temporal_encoder.named_parameters()
                if p.grad is None or p.grad.abs().max().item() == 0.0
            ]
            print(f"{tag:<26} temporal params={sum(1 for _ in m.temporal_encoder.parameters()):>3}  "
                  f"no-grad: {dead if dead else 'none'}")

# ---------------------------------------------------------------------------
header("P9 multi-scale slow branch: conv receptive field of the last slow step")
for t_len in (10, 21, 63, 126):
    enc = MultiScaleTemporalEncoder(F_IN, [32, 10], temporal_encoder="legacy").eval()
    x = torch.randn(1, 1, t_len, F_IN)
    with torch.no_grad():
        xs = enc.slow_aggregator(x.view(1, t_len, F_IN).transpose(1, 2)).transpose(1, 2)
    L = xs.shape[1]
    touches = []
    with torch.no_grad():
        for t in range(t_len):
            xp = x.clone()
            xp[0, 0, t] += 1.0
            xsp = enc.slow_aggregator(xp.view(1, t_len, F_IN).transpose(1, 2)).transpose(1, 2)
            if (xsp[0, -1] - xs[0, -1]).abs().max() > 0:
                touches.append(t)
    k, s = enc.slow_kernel, enc.slow_stride
    print(f"T={t_len:<4} slow length={L:<3} last slow step reads raw steps {touches} "
          f"(kernel {k}, stride {s}, zero-padded taps beyond T-1: {k // 2 - (t_len - 1 - (L - 1) * s)})")

print()
print("done")
```

### A.2 `repro_transformer.py`

The minimal reproduction filed on #234.

```python
import sys

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402

import mci_gru  # noqa: E402
from mci_gru.models.temporal import CausalTransformerEncoder  # noqa: E402

print("torch", torch.__version__, "| mci_gru from", mci_gru.__file__)
torch.manual_seed(0)
enc = CausalTransformerEncoder(input_size=23, d_model=10)  # shipped default gru_hidden_sizes[-1]
x = torch.randn(2, 5, 10, 23)  # (dates, stocks, his_t, features)

# 1. Training forward: raises.
enc.train()
try:
    enc(x)
except RuntimeError as ex:
    print("train forward:", type(ex).__name__, "-", ex)

# 2. Inference forward (what validation and test prediction run): no causal mask.
enc.eval()
with torch.no_grad():
    seq = enc.forward_sequence(x)
    xp = x.clone()
    xp[:, :, -1] += 5.0  # perturb only the LAST step of the window
    moved = (enc.forward_sequence(xp)[:, :, :-1] - seq[:, :, :-1]).abs().max().item()
print(f"eval: max change at steps 0..8 after perturbing step 9 = {moved:.3e} (0 if causal)")
```

### A.3 `smoke_per_encoder.py`

CI's smoke command once per encoder.

```python
"""Run scripts/ci_smoke.py's exact command once per temporal encoder (issue #233).

Only `model.temporal_encoder=<enc>` is appended; everything else is CI's smoke.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

WT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(WT))
import ci_smoke  # noqa: E402

env = dict(os.environ, PYTHONPATH=str(WT))
for enc in ("gru_attn", "legacy", "transformer"):
    with tempfile.TemporaryDirectory(prefix="mci_gru_233_") as tmp:
        tmp = Path(tmp)
        csv_path, run_dir = tmp / "synthetic_market.csv", tmp / "run"
        ci_smoke._write_synthetic_csv(csv_path)
        cmd = ci_smoke._build_smoke_command(csv_path, run_dir) + [f"model.temporal_encoder={enc}"]
        r = subprocess.run(cmd, cwd=WT, env=env, text=True, capture_output=True, timeout=600)
        made = sorted(p.name for p in run_dir.glob("*")) if run_dir.exists() else []
        print(f"--- temporal_encoder={enc}: exit {r.returncode}; "
              f"training_summary.json written: {'training_summary.json' in made}")
        if r.returncode != 0:
            tail = [ln for ln in r.stderr.splitlines() if ln.strip()][-3:]
            print("    " + "\n    ".join(tail))
```

### A.4 `probe_timing.py`

The CPU timing probe.

```python
"""CPU forward+backward time of each temporal encoder at shipped batch shapes (issue #233).

Encoder-only timing; batch = training.batch_size 32 dates x 110 stocks x his_t x 23 features.
CPU only: on CUDA the fused nn.GRU / attention kernels widen the gap to the Python-loop cell.
"""

import sys

WT = sys.argv[1]
sys.path.insert(0, WT)
import time  # noqa: E402
import warnings  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import mci_gru  # noqa: E402

assert mci_gru.__file__.lower().startswith(WT.lower()), mci_gru.__file__
from mci_gru.models.temporal import (  # noqa: E402
    CausalTransformerEncoder,
    GRUWithAttention,
    ImprovedGRU,
    MultiScaleTemporalEncoder,
)

warnings.simplefilter("ignore")
torch.set_num_threads(4)


class ExplicitMaskTransformer(CausalTransformerEncoder):
    def forward_sequence(self, x):
        b, n, t, _ = x.shape
        z = self.input_proj(x).reshape(b * n, t, self.d_model)
        mask = nn.Transformer.generate_square_subsequent_mask(t, dtype=z.dtype)
        return self.encoder(z, mask=mask, is_causal=True).view(b, n, t, self.d_model)


def build(name, ms, f=23, hs=(32, 10)):
    hs = list(hs)
    if ms:
        enc = MultiScaleTemporalEncoder(f, hs, temporal_encoder=name)
        if name == "transformer":
            enc.fast_gru.__class__ = ExplicitMaskTransformer
        return enc
    if name == "legacy":
        return ImprovedGRU(f, hs)
    if name == "transformer":
        e = CausalTransformerEncoder(f, hs[-1])
        e.__class__ = ExplicitMaskTransformer
        return e
    return GRUWithAttention(f, hs)


print("torch", torch.__version__, "threads", torch.get_num_threads())
for t_len in (10, 63):
    for ms in (False, True):
        for name in ("legacy", "gru_attn", "transformer"):
            torch.manual_seed(0)
            enc = build(name, ms).train()
            x = torch.randn(32, 110, t_len, 23)
            times = []
            for i in range(6):
                t0 = time.perf_counter()
                enc(x).sum().backward()
                times.append(time.perf_counter() - t0)
                enc.zero_grad(set_to_none=True)
            med = sorted(times[1:])[len(times[1:]) // 2]
            tag = f"{name}{'*' if name == 'transformer' else ''}{'+ms' if ms else ''}"
            print(f"his_t={t_len:<3} {tag:<16} fwd+bwd median {1000 * med:8.1f} ms")
```

### A.5 `probe_guard_equivalence.py`

Replays the production-model guard test with only gru_hidden_sizes swapped.

```python
"""Does gru_hidden_sizes=[10, 10] reproduce the pinned production model bit for bit? (issue #233)

Replays tests/test_market_latent_state.py::test_default_model_is_unchanged_from_main's
exact computation, using its own helpers and constants, with only the list swapped.
"""

import sys

WT = sys.argv[1]
sys.path.insert(0, WT)
sys.path.insert(0, WT + "/tests")
import warnings  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import mci_gru  # noqa: E402

assert mci_gru.__file__.lower().startswith(WT.lower()), mci_gru.__file__
import test_market_latent_state as t  # noqa: E402

from mci_gru.models.factory import create_model  # noqa: E402

warnings.simplefilter("ignore")
expected = t._MAIN_DEFAULT_MODELS["production"]
for hs, role in (([32, 10], "control, must match"), ([10, 10], "claim"), ([32, 32], "positive control, must differ")):
    config = dict(expected["config"], gru_hidden_sizes=hs)
    torch.manual_seed(0)
    model = create_model(7, config).eval()
    ts, gf, ei, ew, sm = t._two_date_masked_batch(expected["edge_dim"])
    with torch.no_grad():
        scores = model(ts, gf, ei, ew, 5, stock_mask=sm).flatten().tolist()
    same = (
        len(model.state_dict()) == expected["state_dict_keys"]
        and sum(p.numel() for p in model.parameters()) == expected["parameters"]
        and t._state_dict_signature(model) == expected["signature"]
        and scores == pytest.approx(expected["scores"], abs=1e-5)
    )
    print(f"gru_hidden_sizes={hs!s:<9} ({role}): keys, params, signature and scores all match pinned main: {same}")
```

## Appendix B. Raw outputs

Captured on 2026-09-28 against `dbb42ed`, torch 2.11.0+cpu, Python 3.12.11, Windows 11. Warning lines from the head-count reduction are filtered out of B.2 and B.3.

### B.1 Output of A.1.

```text

==============================================================================
P0 environment
==============================================================================
torch 2.11.0+cpu | mci_gru from C:\Users\magil\.claude\worktrees\mci-gru-233-temporal-encoder-audit\mci_gru\__init__.py

==============================================================================
P1 census: what each (encoder, gru_hidden_sizes, use_multi_scale) builds
==============================================================================
input_size F=23; params are the temporal encoder only (model.temporal_encoder)
encoder     hidden        ms    params  out  built
legacy      [32, 10]      N       7890   10  ImprovedGRU cells 23->32->10
legacy      [64, 32]      N      30112   32  ImprovedGRU cells 23->64->32
legacy      [32, 32]      N      13632   32  ImprovedGRU cells 23->32->32
legacy      [16]          N       2352   16  ImprovedGRU cells 23->16
legacy      [8, 4]        N       1188    4  ImprovedGRU cells 23->8->4
legacy      [64, 32, 16]  N      33040   16  ImprovedGRU cells 23->64->32->16
gru_attn    [32, 10]      N       1730   10  nn.GRU 23->10 x2 layers (+attn readout, LN)
gru_attn    [64, 32]      N      11872   32  nn.GRU 23->32 x2 layers (+attn readout, LN)
gru_attn    [32, 32]      N      11872   32  nn.GRU 23->32 x2 layers (+attn readout, LN)
gru_attn    [16]          N       2000   16  nn.GRU 23->16 x1 layers (+attn readout, LN)
gru_attn    [8, 4]        N        476    4  nn.GRU 23->4 x2 layers (+attn readout, LN)
gru_attn    [64, 32, 16]  N       5264   16  nn.GRU 23->16 x3 layers (+attn readout, LN)
transformer [32, 10]      N       2900   10  Transformer d_model=10 nhead=2 (head_dim=5) layers=2 ff=40 dropout=0.1 no-PE
transformer [64, 32]      N      26176   32  Transformer d_model=32 nhead=4 (head_dim=8) layers=2 ff=128 dropout=0.1 no-PE
transformer [32, 32]      N      26176   32  Transformer d_model=32 nhead=4 (head_dim=8) layers=2 ff=128 dropout=0.1 no-PE
transformer [16]          N       6944   16  Transformer d_model=16 nhead=4 (head_dim=4) layers=2 ff=64 dropout=0.1 no-PE
transformer [8, 4]        N        584    4  Transformer d_model=4 nhead=4 (head_dim=1) layers=2 ff=16 dropout=0.1 no-PE
transformer [64, 32, 16]  N       6944   16  Transformer d_model=16 nhead=4 (head_dim=4) layers=2 ff=64 dropout=0.1 no-PE
legacy      [32, 10]      Y      18658   10  MultiScale[fast: ImprovedGRU cells 23->32->10 | slow: ImprovedGRU cells 23->32->10 | conv 5/2 | combiner 20->10]
legacy      [64, 32]      Y      64972   32  MultiScale[fast: ImprovedGRU cells 23->64->32 | slow: ImprovedGRU cells 23->64->32 | conv 5/2 | combiner 64->32]
legacy      [32, 32]      Y      32012   32  MultiScale[fast: ImprovedGRU cells 23->32->32 | slow: ImprovedGRU cells 23->32->32 | conv 5/2 | combiner 64->32]
legacy      [16]          Y       7900   16  MultiScale[fast: ImprovedGRU cells 23->16 | slow: ImprovedGRU cells 23->16 | conv 5/2 | combiner 32->16]
legacy      [8, 4]        Y       5080    4  MultiScale[fast: ImprovedGRU cells 23->8->4 | slow: ImprovedGRU cells 23->8->4 | conv 5/2 | combiner 8->4]
legacy      [64, 32, 16]  Y      69276   16  MultiScale[fast: ImprovedGRU cells 23->64->32->16 | slow: ImprovedGRU cells 23->64->32->16 | conv 5/2 | combiner 32->16]
gru_attn    [32, 10]      Y       6338   10  MultiScale[fast: nn.GRU 23->10 x2 layers (+attn readout, LN) | slow: nn.GRU 23->10 x2 layers (+attn readout, LN) | conv 5/2 | combiner 20->10]
gru_attn    [64, 32]      Y      28492   32  MultiScale[fast: nn.GRU 23->32 x2 layers (+attn readout, LN) | slow: nn.GRU 23->32 x2 layers (+attn readout, LN) | conv 5/2 | combiner 64->32]
gru_attn    [32, 32]      Y      28492   32  MultiScale[fast: nn.GRU 23->32 x2 layers (+attn readout, LN) | slow: nn.GRU 23->32 x2 layers (+attn readout, LN) | conv 5/2 | combiner 64->32]
gru_attn    [16]          Y       7196   16  MultiScale[fast: nn.GRU 23->16 x1 layers (+attn readout, LN) | slow: nn.GRU 23->16 x1 layers (+attn readout, LN) | conv 5/2 | combiner 32->16]
gru_attn    [8, 4]        Y       3656    4  MultiScale[fast: nn.GRU 23->4 x2 layers (+attn readout, LN) | slow: nn.GRU 23->4 x2 layers (+attn readout, LN) | conv 5/2 | combiner 8->4]
gru_attn    [64, 32, 16]  Y      13724   16  MultiScale[fast: nn.GRU 23->16 x3 layers (+attn readout, LN) | slow: nn.GRU 23->16 x3 layers (+attn readout, LN) | conv 5/2 | combiner 32->16]
transformer [32, 10]      Y       7508   10  MultiScale[fast: Transformer d_model=10 nhead=2 (head_dim=5) layers=2 ff=40 dropout=0.1 no-PE | slow: nn.GRU 23->10 x2 layers (+attn readout, LN) | conv 5/2 | combiner 20->10]
transformer [64, 32]      Y      42796   32  MultiScale[fast: Transformer d_model=32 nhead=4 (head_dim=8) layers=2 ff=128 dropout=0.1 no-PE | slow: nn.GRU 23->32 x2 layers (+attn readout, LN) | conv 5/2 | combiner 64->32]
transformer [32, 32]      Y      42796   32  MultiScale[fast: Transformer d_model=32 nhead=4 (head_dim=8) layers=2 ff=128 dropout=0.1 no-PE | slow: nn.GRU 23->32 x2 layers (+attn readout, LN) | conv 5/2 | combiner 64->32]
transformer [16]          Y      12140   16  MultiScale[fast: Transformer d_model=16 nhead=4 (head_dim=4) layers=2 ff=64 dropout=0.1 no-PE | slow: nn.GRU 23->16 x1 layers (+attn readout, LN) | conv 5/2 | combiner 32->16]
transformer [8, 4]        Y       3764    4  MultiScale[fast: Transformer d_model=4 nhead=4 (head_dim=1) layers=2 ff=16 dropout=0.1 no-PE | slow: nn.GRU 23->4 x2 layers (+attn readout, LN) | conv 5/2 | combiner 8->4]
transformer [64, 32, 16]  Y      15404   16  MultiScale[fast: Transformer d_model=16 nhead=4 (head_dim=4) layers=2 ff=64 dropout=0.1 no-PE | slow: nn.GRU 23->16 x3 layers (+attn readout, LN) | conv 5/2 | combiner 32->16]

==============================================================================
P1b interior gru_hidden_sizes entries are ignored by gru_attn and transformer
==============================================================================
gru_attn: state_dict shapes identical for [32,10] and [999,10]: True
transformer: state_dict shapes identical for [32,10] and [999,10]: True
legacy: params [32,10]=7890  [999,10]=3133234  (interior entry is honoured)

==============================================================================
P1c full-model parameter share of the temporal encoder (shipped base, multi-scale)
==============================================================================
legacy       total= 101770  temporal= 18658 (18.3%)
gru_attn     total=  89450  temporal=  6338 (7.1%)
transformer  total=  90620  temporal=  7508 (8.3%)

==============================================================================
P2 legacy cell: attention neutralised (r' == 1) vs torch.nn.GRUCell (reset open)
==============================================================================
max |cell - GRUCell| over 10 steps: 1.490e-07
probe sensitivity (live attention path): max |diff| = 6.629e-01

==============================================================================
P2b legacy cell: r' depends on h_{t-1} through ONE scalar (rank of d r'/d h)
==============================================================================
rank d r'/d h_(t-1), attention reset (H=10): [1]
rank d r /d h_(t-1), nn.GRUCell reset  (H=10): [10]

==============================================================================
P2c legacy cell: is r' a gate? range of r' = alpha * W_v x at init
==============================================================================
hidden=[32, 10]: r' fraction <0 = 0.521, >1 = 0.001, min=-1.049 max=1.394; alpha mean=0.501 sd=0.0197

==============================================================================
P2d the paper's eq.6 literally (softmax over one score) and the Feb-2026 code: dead W_q/W_k
==============================================================================
SoftmaxCell            max|grad| W_q.weight=0.000e+00 W_k.weight=0.000e+00 W_v.weight=6.620e+00
AttentionResetGRUCell  max|grad| W_q.weight=2.493e-01 W_k.weight=2.713e-01 W_v.weight=3.440e+00

==============================================================================
P3 gru_attn readout: softmax axis, weight mass, LayerNorm placement, dropout
==============================================================================
alpha shape (21, 10) (rows=B*N sequences, cols=T steps); row sums in [1.000000, 1.000000]
module output == manual LN(h_T + softmax_t(h_T.h_t/sqrt(d)) @ H): True
init normalised attention entropy mean=0.9995 (1.0 = uniform); weight on last step mean=0.1065 vs uniform 0.1000
nn.GRU inter-layer dropout = 0.0; layers = 2; width = 10
forward_sequence returns pre-readout GRU states: True

==============================================================================
P4 transformer: is the causal mask applied?  forward_sequence vs explicit masks
==============================================================================
[train, grad on]     forward_sequence RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
[eval, grad on]      forward_sequence RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
[eval, grad off]     max|seq - explicit causal mask| = 2.120e+00; max|seq - NO mask| = 0.000e+00
trainer paths: training step = [train, grad on]; validation/test = [eval, grad off] (mci_gru/training/trainer.py model.eval() + torch.no_grad())

==============================================================================
P4b transformer: nhead actually used for each d_model
==============================================================================
d_model=4   nhead=4
d_model=6   nhead=3
d_model=7   nhead=1
d_model=8   nhead=4
d_model=10  nhead=2
d_model=12  nhead=4
d_model=16  nhead=4
d_model=32  nhead=4

==============================================================================
P5 no lookahead inside the window: perturb steps >= t0, measure change at steps < t0
==============================================================================
(sequence = the one the trunk consumes: forward_sequence, or the fast branch in multi-scale)
legacy            [train] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=6.702e-01
legacy            [eval ] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=6.702e-01
gru_attn          [train] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=9.327e-01
gru_attn          [eval ] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=9.327e-01
transformer       [train] RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer       [eval ] steps<t0 max|d|=1.455e+00   steps>=t0 max|d|=3.701e+00
transformer*      [train] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=3.992e+00
transformer*      [eval ] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=3.749e+00
legacy+ms         [train] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=7.938e-01
legacy+ms         [eval ] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=7.938e-01
gru_attn+ms       [train] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=9.327e-01
gru_attn+ms       [eval ] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=9.327e-01
transformer+ms    [train] RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer+ms    [eval ] steps<t0 max|d|=1.455e+00   steps>=t0 max|d|=3.701e+00
transformer*+ms   [train] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=3.992e+00
transformer*+ms   [eval ] steps<t0 max|d|=0.000e+00   steps>=t0 max|d|=3.749e+00

==============================================================================
P5b transformer (inference path): which positions move when ONE step is perturbed
==============================================================================
transformer   perturb step 9: per-position max|d| = 1e+00 6e-01 6e-01 3e-01 6e-01 4e-01 5e-01 5e-01 6e-01 2e+00
transformer   perturb step 5: per-position max|d| = 1e+00 7e-01 7e-01 3e-01 7e-01 1e+00 4e-01 4e-01 6e-01 6e-01
transformer*  perturb step 9: per-position max|d| = 0e+00 0e+00 0e+00 0e+00 0e+00 0e+00 0e+00 0e+00 0e+00 2e+00
transformer*  perturb step 5: per-position max|d| = 0e+00 0e+00 0e+00 0e+00 0e+00 2e+00 3e-01 1e-01 3e-01 2e-01

==============================================================================
P6 positional information: permute the first T-1 steps, keep the last step fixed
==============================================================================
order share = |f(permuted history) - f(x)| / |f(fresh random history) - f(x)|, inference path
legacy             |f(perm)-f(x)|=2.119e-01  |f(new history)-f(x)|=3.721e-01  order share=0.569
gru_attn           |f(perm)-f(x)|=6.111e-01  |f(new history)-f(x)|=1.749e+00  order share=0.349
transformer        |f(perm)-f(x)|=3.069e-07  |f(new history)-f(x)|=7.609e-01  order share=0.000
transformer*       |f(perm)-f(x)|=1.515e-01  |f(new history)-f(x)|=7.658e-01  order share=0.198
legacy+ms          |f(perm)-f(x)|=1.237e-01  |f(new history)-f(x)|=1.892e-01  order share=0.654
gru_attn+ms        |f(perm)-f(x)|=6.958e-01  |f(new history)-f(x)|=1.186e+00  order share=0.587
transformer+ms     |f(perm)-f(x)|=3.615e-01  |f(new history)-f(x)|=6.647e-01  order share=0.544
transformer*+ms    |f(perm)-f(x)|=3.696e-01  |f(new history)-f(x)|=6.683e-01  order share=0.553
one-layer explicit-mask transformer (exact invariance expected: no PE, one causal layer):
  |f(perm)-f(x)| = 1.899e-07

==============================================================================
P7 leakage across stocks and dates: perturb one (date, stock), others must not move
==============================================================================
legacy            [train] target max|d|=4.404e-01  every other (date, stock) max|d|=0.000e+00
legacy            [eval ] target max|d|=4.404e-01  every other (date, stock) max|d|=0.000e+00
gru_attn          [train] target max|d|=1.034e+00  every other (date, stock) max|d|=0.000e+00
gru_attn          [eval ] target max|d|=1.034e+00  every other (date, stock) max|d|=0.000e+00
transformer       [train] RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer       [eval ] target max|d|=1.147e+00  every other (date, stock) max|d|=0.000e+00
transformer*      [train] target max|d|=1.135e+00  every other (date, stock) max|d|=0.000e+00
transformer*      [eval ] target max|d|=1.092e+00  every other (date, stock) max|d|=0.000e+00
legacy+ms         [train] target max|d|=2.492e-01  every other (date, stock) max|d|=0.000e+00
legacy+ms         [eval ] target max|d|=2.492e-01  every other (date, stock) max|d|=0.000e+00
gru_attn+ms       [train] target max|d|=8.961e-01  every other (date, stock) max|d|=0.000e+00
gru_attn+ms       [eval ] target max|d|=8.961e-01  every other (date, stock) max|d|=0.000e+00
transformer+ms    [train] RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer+ms    [eval ] target max|d|=6.678e-01  every other (date, stock) max|d|=0.000e+00
transformer*+ms   [train] target max|d|=6.193e-01  every other (date, stock) max|d|=0.000e+00
transformer*+ms   [eval ] target max|d|=6.473e-01  every other (date, stock) max|d|=0.000e+00

==============================================================================
P7b positive controls: the P7 probe must fire on a deliberately leaky encoder
==============================================================================
leak across stocks  every other (date, stock) max|d|=3.651e-02  (must be > 0)
leak across dates   every other (date, stock) max|d|=5.918e-02  (must be > 0)

==============================================================================
P8 gradient flow: temporal parameters with no gradient (full model, train mode)
==============================================================================
legacy                     temporal params= 28  no-grad: none
legacy+a1a2                temporal params= 28  no-grad: none
gru_attn                   temporal params= 10  no-grad: none
gru_attn+a1a2              temporal params= 10  no-grad: none
transformer                full-model training forward RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer+a1a2           full-model training forward RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer*               temporal params= 26  no-grad: none
transformer*+a1a2          temporal params= 26  no-grad: none
legacy+ms                  temporal params= 60  no-grad: none
legacy+ms+a1a2             temporal params= 60  no-grad: none
gru_attn+ms                temporal params= 24  no-grad: none
gru_attn+ms+a1a2           temporal params= 24  no-grad: none
transformer+ms             full-model training forward RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer+ms+a1a2        full-model training forward RAISES RuntimeError: Need attn_mask if specifying the is_causal hint. You may use
transformer*+ms            temporal params= 40  no-grad: none
transformer*+ms+a1a2       temporal params= 40  no-grad: none

==============================================================================
P9 multi-scale slow branch: conv receptive field of the last slow step
==============================================================================
T=10   slow length=5   last slow step reads raw steps [6, 7, 8, 9] (kernel 5, stride 2, zero-padded taps beyond T-1: 1)
T=21   slow length=11  last slow step reads raw steps [18, 19, 20] (kernel 5, stride 2, zero-padded taps beyond T-1: 2)
T=63   slow length=32  last slow step reads raw steps [60, 61, 62] (kernel 5, stride 2, zero-padded taps beyond T-1: 2)
T=126  slow length=63  last slow step reads raw steps [122, 123, 124, 125] (kernel 5, stride 2, zero-padded taps beyond T-1: 1)

done
```

### B.2 Output of A.3.

```text
--- temporal_encoder=gru_attn: exit 0; training_summary.json written: True
--- temporal_encoder=legacy: exit 0; training_summary.json written: True
--- temporal_encoder=transformer: exit 1; training_summary.json written: False
        raise RuntimeError(
    RuntimeError: Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask.
    Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

### B.3 Output of A.2.

```text
torch 2.11.0+cpu | mci_gru from C:\Users\magil\.claude\worktrees\mci-gru-233-temporal-encoder-audit\mci_gru\__init__.py
train forward: RuntimeError - Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask.
eval: max change at steps 0..8 after perturbing step 9 = 9.415e-01 (0 if causal)
```

### B.4 Output of A.4.

```text
torch 2.11.0+cpu threads 4
his_t=10  legacy           fwd+bwd median     26.2 ms
his_t=10  gru_attn         fwd+bwd median     18.8 ms
his_t=10  transformer*     fwd+bwd median    100.1 ms
his_t=10  legacy+ms        fwd+bwd median     69.9 ms
his_t=10  gru_attn+ms      fwd+bwd median     51.3 ms
his_t=10  transformer*+ms  fwd+bwd median    118.4 ms
his_t=63  legacy           fwd+bwd median    477.8 ms
his_t=63  gru_attn         fwd+bwd median    212.4 ms
his_t=63  transformer*     fwd+bwd median   1139.4 ms
his_t=63  legacy+ms        fwd+bwd median    771.9 ms
his_t=63  gru_attn+ms      fwd+bwd median    345.9 ms
his_t=63  transformer*+ms  fwd+bwd median   1231.1 ms
```

### B.5 Output of A.5.

```text
gru_hidden_sizes=[32, 10]  (control, must match): keys, params, signature and scores all match pinned main: True
gru_hidden_sizes=[10, 10]  (claim): keys, params, signature and scores all match pinned main: True
gru_hidden_sizes=[32, 32]  (positive control, must differ): keys, params, signature and scores all match pinned main: False
```
