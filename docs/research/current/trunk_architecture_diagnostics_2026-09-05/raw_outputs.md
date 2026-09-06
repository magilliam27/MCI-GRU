# Raw outputs for MCI_GRU_TRUNK_ARCHITECTURE_OPPORTUNITIES_2026-09-05.md

Scratchpad diagnostics, 2026-09-05. Mechanics-level; not performance evidence.

## init_diag.py (synthetic inputs, 3 seeds)

```
mci_gru resolved to: C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea\mci_gru\__init__.py

Shapes: B=8, N=500, T=10, F=23; ln(N) = 6.21

===== baseline =====
stage        mean_cos  cs_var_share  eff_rank
a1             0.844       0.156       4.3
a2             0.876       0.124       3.1
b1             0.895       0.115      13.2
b2             0.856       0.153      12.4
z_pre          0.860       0.140       7.2
z_post         1.000       0.000       4.1
attention entropy 6.205 nats (uniform=6.21); logit std 0.175
output: cross-sectional std / mean|out| = 0.640;  IC loss at init = +0.0063
grad-norm by module (abs, rel-to-param-norm):
   temporal_encoder    1.21e-01   1.00e-02
   gat_layer           5.86e-02   6.29e-03
   proj_temporal       6.14e-02   1.77e-02
   proj_cross          4.21e-02   1.17e-02
   ln_a1               1.72e-02   3.04e-03
   ln_a2               1.56e-02   2.76e-03
   latent_learner      1.85e-02   1.71e-03
   ln_z                2.27e-02   2.01e-03
   self_attention      3.00e-01   2.64e-02
   final_gat           2.18e-01   1.70e-02

===== no_self_attention =====
stage        mean_cos  cs_var_share  eff_rank
a1             0.844       0.156       4.3
a2             0.875       0.125       3.0
b1             0.895       0.115      13.3
b2             0.855       0.154      12.4
z_pre          0.859       0.140       7.1
output: cross-sectional std / mean|out| = 0.528;  IC loss at init = +0.0069
grad-norm by module (abs, rel-to-param-norm):
   temporal_encoder    9.82e-02   8.16e-03
   gat_layer           2.08e-01   2.23e-02
   proj_temporal       4.35e-02   1.26e-02
   proj_cross          5.09e-02   1.44e-02
   ln_a1               1.13e-02   2.00e-03
   ln_a2               1.44e-02   2.55e-03
   latent_learner      1.37e-02   1.26e-03
   ln_z                1.89e-02   1.67e-03
   final_gat           2.11e-01   1.65e-02

===== residual =====
stage        mean_cos  cs_var_share  eff_rank
a1             0.844       0.156       4.3
a2             0.876       0.124       3.1
b1             0.895       0.115      13.2
b2             0.856       0.153      12.4
z_pre          0.860       0.140       7.2
z_post         0.821       0.179      17.3
attention entropy 6.205 nats (uniform=6.21); logit std 0.175
output: cross-sectional std / mean|out| = 0.367;  IC loss at init = +0.0148
grad-norm by module (abs, rel-to-param-norm):
   temporal_encoder    9.11e-02   7.56e-03
   gat_layer           1.72e-01   1.85e-02
   proj_temporal       4.25e-02   1.22e-02
   proj_cross          4.59e-02   1.30e-02
   ln_a1               1.11e-02   1.96e-03
   ln_a2               1.31e-02   2.31e-03
   latent_learner      1.12e-02   1.04e-03
   ln_z                1.66e-02   1.47e-03
   self_attention      8.75e-02   7.72e-03
   final_gat           1.76e-01   1.38e-02

===== pre_ln_residual =====
stage        mean_cos  cs_var_share  eff_rank
a1             0.844       0.156       4.3
a2             0.876       0.124       3.1
b1             0.895       0.115      13.2
b2             0.856       0.153      12.4
z_pre          0.860       0.140       7.2
z_post         0.817       0.183      17.3
attention entropy 6.205 nats (uniform=6.21); logit std 0.175
output: cross-sectional std / mean|out| = 0.367;  IC loss at init = +0.0148
grad-norm by module (abs, rel-to-param-norm):
   temporal_encoder    9.14e-02   7.59e-03
   gat_layer           1.73e-01   1.85e-02
   proj_temporal       4.27e-02   1.23e-02
   proj_cross          4.62e-02   1.30e-02
   ln_a1               1.11e-02   1.96e-03
   ln_a2               1.32e-02   2.33e-03
   latent_learner      1.12e-02   1.04e-03
   ln_z                1.68e-02   1.48e-03
   self_attention      8.45e-02   5.28e-03
   final_gat           1.77e-01   1.39e-02

```

## smoke_variant.py, seed 1729, lr 5e-5 (runs)

```
| Epoch | baseline | no_self_attention | residual | pre_ln_residual |
|---:|---:|---:|---:|---:|
| 1 | +0.0231 | +0.0342 | +0.0004 | +0.0005 |
| 2 | +0.0277 | +0.0388 | +0.0113 | +0.0115 |
| 3 | +0.0306 | +0.0415 | +0.0194 | +0.0196 |
| 4 | +0.0330 | +0.0419 | +0.0223 | +0.0225 |
| 5 | +0.0344 | +0.0420 | +0.0237 | +0.0239 |
| 6 | +0.0358 | +0.0415 | +0.0251 | +0.0253 |
| 7 | +0.0368 | +0.0417 | +0.0263 | +0.0264 |
| 8 | +0.0373 | +0.0416 | +0.0264 | +0.0266 |
| 9 | +0.0375 | +0.0417 | +0.0267 | +0.0269 |
| 10 | +0.0376 | +0.0417 | +0.0267 | +0.0269 |

params: {'baseline': 86762, 'no_self_attention': 37098, 'residual': 86762, 'pre_ln_residual': 87018}
baseline: best_val_ics=[0.037608505617226326] best_val_rank_ics=[0.03709016728305047]
no_self_attention: best_val_ics=[0.04201131644508531] best_val_rank_ics=[0.040379284009818106]
residual: best_val_ics=[0.026749714728324644] best_val_rank_ics=[0.02499263913881394]
pre_ln_residual: best_val_ics=[0.026924473563990286] best_val_rank_ics=[0.02542995228882759]
baseline test avg_ic 0.0257 avg_rank_ic 0.0255
no_self_attention test avg_ic 0.0272 avg_rank_ic 0.0275
residual test avg_ic 0.0161 avg_rank_ic 0.0217
pre_ln_residual test avg_ic 0.0161 avg_rank_ic 0.0218
```

## smoke_variant.py, seed 1730, lr 5e-5 (runs2)

```
| Epoch | baseline | no_self_attention | residual | pre_ln_residual |
|---:|---:|---:|---:|---:|
| 1 | -0.0167 | +0.0057 |  | +0.0128 |
| 2 | -0.0009 | +0.0167 |  | +0.0202 |
| 3 | +0.0162 | +0.0276 |  | +0.0264 |
| 4 | +0.0226 | +0.0336 |  | +0.0311 |
| 5 | +0.0272 | +0.0369 |  | +0.0347 |
| 6 | +0.0311 | +0.0401 |  | +0.0369 |
| 7 | +0.0329 | +0.0414 |  | +0.0379 |
| 8 | +0.0340 | +0.0419 |  | +0.0383 |
| 9 | +0.0348 | +0.0425 |  | +0.0387 |
| 10 | +0.0348 | +0.0425 |  | +0.0387 |

params: {'baseline': 86762, 'no_self_attention': 37098, 'pre_ln_residual': 87018}
baseline: best_val_ics=[0.03483733766141438] best_val_rank_ics=[0.02981961734833256]
no_self_attention: best_val_ics=[0.04250320900351771] best_val_rank_ics=[0.029422738379047762]
pre_ln_residual: best_val_ics=[0.03869484016491521] best_val_rank_ics=[0.03190792508182987]
baseline test avg_ic 0.027 avg_rank_ic 0.0285
no_self_attention test avg_ic 0.0369 avg_rank_ic 0.0333
pre_ln_residual test avg_ic 0.0317 avg_rank_ic 0.0316
```

## smoke_variant.py, seed 1729, lr 3e-4 (runs_lr)

```
| Epoch | baseline | no_self_attention | residual | pre_ln_residual |
|---:|---:|---:|---:|---:|
| 1 | +0.0298 |  |  | +0.0222 |
| 2 | +0.0348 |  |  | +0.0273 |
| 3 | +0.0471 |  |  | +0.0337 |
| 4 | +0.0377 |  |  | +0.0295 |
| 5 | +0.0460 |  |  | +0.0359 |
| 6 | +0.0357 |  |  | +0.0341 |
| 7 | +0.0300 |  |  | +0.0330 |
| 8 | +0.0277 |  |  | +0.0328 |
| 9 | +0.0297 |  |  | +0.0333 |
| 10 | +0.0283 |  |  | +0.0331 |

params: {'baseline': 86762, 'pre_ln_residual': 87018}
baseline: best_val_ics=[0.04706389144543679] best_val_rank_ics=[0.03305231383250606]
pre_ln_residual: best_val_ics=[0.03587778507461471] best_val_rank_ics=[0.02943641800553568]
baseline test avg_ic 0.034 avg_rank_ic 0.0284
pre_ln_residual test avg_ic 0.0253 avg_rank_ic 0.0251
```

## trained_attention_probe.py on runs (seed 1729, lr 5e-5)

```
mci_gru resolved to: C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea\mci_gru\__init__.py
features=16 stocks=472 val_batches=8
===== baseline =====
  [untrained] attention entropy 6.146 nats (uniform=6.16); logit std 0.221; self-weight 0.0022 (uniform=0.0021); top-5 mass 0.016
  [untrained] z before block: cos 0.698 cs_var_share 0.301 eff_rank 4.0
  [untrained] z after  block: cos 0.999 cs_var_share 0.001 eff_rank 2.1
  [untrained] validation daily IC over 248 days: mean +0.0188; output cross-sectional std 0.0181
  loaded checkpoint (missing=0, unexpected=0)
  [trained] attention entropy 6.101 nats (uniform=6.16); logit std 0.353; self-weight 0.0025 (uniform=0.0021); top-5 mass 0.021
  [trained] z before block: cos 0.646 cs_var_share 0.353 eff_rank 3.9
  [trained] z after  block: cos 0.990 cs_var_share 0.010 eff_rank 1.2
  [trained] validation daily IC over 248 days: mean +0.0376; output cross-sectional std 0.2687
===== no_self_attention =====
  [untrained] validation daily IC over 248 days: mean +0.0299; output cross-sectional std 0.3733
  loaded checkpoint (missing=0, unexpected=0)
  [trained] validation daily IC over 248 days: mean +0.0420; output cross-sectional std 0.6061
===== residual =====
  [untrained] attention entropy 6.146 nats (uniform=6.16); logit std 0.221; self-weight 0.0022 (uniform=0.0021); top-5 mass 0.016
  [untrained] z before block: cos 0.698 cs_var_share 0.301 eff_rank 4.0
  [untrained] z after  block: cos 0.779 cs_var_share 0.221 eff_rank 4.0
  [untrained] validation daily IC over 248 days: mean -0.0061; output cross-sectional std 0.4575
  loaded checkpoint (missing=0, unexpected=0)
  [trained] attention entropy 6.135 nats (uniform=6.16); logit std 0.274; self-weight 0.0025 (uniform=0.0021); top-5 mass 0.017
  [trained] z before block: cos 0.682 cs_var_share 0.318 eff_rank 4.2
  [trained] z after  block: cos 0.760 cs_var_share 0.239 eff_rank 4.2
  [trained] validation daily IC over 248 days: mean +0.0267; output cross-sectional std 0.7005
===== pre_ln_residual =====
  [untrained] attention entropy 6.146 nats (uniform=6.16); logit std 0.221; self-weight 0.0022 (uniform=0.0021); top-5 mass 0.016
  [untrained] z before block: cos 0.698 cs_var_share 0.301 eff_rank 4.0
  [untrained] z after  block: cos 0.779 cs_var_share 0.221 eff_rank 4.0
  [untrained] validation daily IC over 248 days: mean -0.0061; output cross-sectional std 0.4575
  loaded checkpoint (missing=0, unexpected=0)
  [trained] attention entropy 6.135 nats (uniform=6.16); logit std 0.275; self-weight 0.0025 (uniform=0.0021); top-5 mass 0.017
  [trained] z before block: cos 0.683 cs_var_share 0.317 eff_rank 4.2
  [trained] z after  block: cos 0.762 cs_var_share 0.237 eff_rank 4.2
  [trained] validation daily IC over 248 days: mean +0.0269; output cross-sectional std 0.7076
```

## trained_attention_probe.py on runs_lr (seed 1729, lr 3e-4), best checkpoints

```
===== baseline =====
  [trained] attention entropy 5.775 nats (uniform=6.16); logit std 0.895; self-weight 0.0024 (uniform=0.0021); top-5 mass 0.043
  [trained] z before block: cos 0.611 cs_var_share 0.388 eff_rank 3.9
  [trained] z after  block: cos 0.967 cs_var_share 0.034 eff_rank 1.1
  [trained] validation daily IC over 248 days: mean +0.0471; output cross-sectional std 0.6684
===== pre_ln_residual =====
  [trained] attention entropy 5.991 nats (uniform=6.16); logit std 0.657; self-weight 0.0030 (uniform=0.0021); top-5 mass 0.041
  [trained] z before block: cos 0.602 cs_var_share 0.398 eff_rank 4.4
  [trained] z after  block: cos 0.660 cs_var_share 0.339 eff_rank 4.3
  [trained] validation daily IC over 248 days: mean +0.0359; output cross-sectional std 1.2265
```

## Addendum 2026-09-05: fix diagnostics and Fix B smoke

### fix_diag.py (synthetic, masked, centred date-sensitivity)

```
mci_gru resolved to: C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea\mci_gru\__init__.py

Shapes B=8 N=500 T=10 F=23

===== baseline =====
  params total  89450  block  49664
  z before block: cos 0.750 cs_var_share 0.189 eff_rank   5.5
  z after  block: cos 0.857 cs_var_share 0.074 eff_rank   1.0
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 100-400 move: raw |dscore_0| 0.3356  ->  CENTRED (what ICLoss sees) 0.0111
  fwd+bwd 0.30s

===== no_self_attention =====
  params total  39786  block      0
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 100-400 move: raw |dscore_0| 0.0000  ->  CENTRED (what ICLoss sees) 0.1573
  fwd+bwd 0.30s

===== pre_ln_residual =====
  params total  89706  block  49920
  z before block: cos 0.750 cs_var_share 0.189 eff_rank   5.5
  z after  block: cos 0.717 cs_var_share 0.225 eff_rank   9.0
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 100-400 move: raw |dscore_0| 0.1846  ->  CENTRED (what ICLoss sees) 0.5338
  fwd+bwd 0.28s

===== two_way_latent =====
  params total 176746  block 136960
  z before block: cos 0.751 cs_var_share 0.189 eff_rank   5.5
  z after  block: cos 0.684 cs_var_share 0.261 eff_rank  11.1
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 100-400 move: raw |dscore_0| 0.0377  ->  CENTRED (what ICLoss sees) 0.5111
  fwd+bwd 0.35s

```

### two_way_latent smoke, seeds 1729 and 1730

```
seed 1729:
  epoch 1: -0.008806
  epoch 2: 0.005256
  epoch 3: 0.016107
  epoch 4: 0.020139
  epoch 5: 0.023611
  epoch 6: 0.024225
  epoch 7: 0.025701
  epoch 8: 0.026425
  epoch 9: 0.026680
  epoch 10: 0.026689
  TEST avg_ic 0.0166 avg_rank_ic 0.0217
seed 1730:
  epoch 1: 0.011770
  epoch 2: 0.019721
  epoch 3: 0.026171
  epoch 4: 0.032233
  epoch 5: 0.032738
  epoch 6: 0.035016
  epoch 7: 0.036114
  epoch 8: 0.036639
  epoch 9: 0.037204
  epoch 10: 0.037286
  TEST avg_ic 0.0306 avg_rank_ic 0.0303
```

## Addendum 2: the 110-name universe (gics_top10_110_2016)

Correct universe: 201-node PIT union axis, ~110 admissible names/session,
masked_panel mode, 2012 training days, 239 val days, 238 test days.
Global regime off (deviation from frozen recipe, applied to every arm).

### Synthetic fix_diag.py at the 110 shape
```
mci_gru resolved to: C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea\mci_gru\__init__.py

Shapes B=8 N=201 T=10 F=16

===== baseline =====
  params total  86762  block  49664
  z before block: cos 0.272 cs_var_share 0.501 eff_rank   1.5
  z after  block: cos 0.298 cs_var_share 0.453 eff_rank   1.0
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 40-100 move: raw |dscore_0| 0.0828  ->  CENTRED (what ICLoss sees) 0.0011
  fwd+bwd 0.20s

===== no_self_attention =====
  params total  37098  block      0
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 40-100 move: raw |dscore_0| 0.0000  ->  CENTRED (what ICLoss sees) 0.5786
  fwd+bwd 0.14s

===== pre_ln_residual =====
  params total  87018  block  49920
  z before block: cos 0.272 cs_var_share 0.501 eff_rank   1.5
  z after  block: cos 0.256 cs_var_share 0.529 eff_rank   1.8
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 40-100 move: raw |dscore_0| 0.1344  ->  CENTRED (what ICLoss sees) 0.4366
  fwd+bwd 0.12s

===== two_way_latent =====
  params total 174058  block 136960
  z before block: cos 0.273 cs_var_share 0.499 eff_rank   1.4
  z after  block: cos 0.245 cs_var_share 0.550 eff_rank   2.1
  masking: inactive_zero=True  leak_from_inactive=0.00e+00
  date-sensitivity when stocks 40-100 move: raw |dscore_0| 0.0497  ->  CENTRED (what ICLoss sees) 0.5896
  fwd+bwd 0.15s

```

### Two-seed smoke, real 110 universe
```
baseline
  seed 1729: e1=-0.004283 e2=0.001354 e3=0.004423 e4=0.007743 e5=0.011563 e6=0.017333 e7=0.017958 e8=0.019034 e9=0.019263 e10=0.019420
    best_val_ic 0.0194  test_avg_ic 0.0244  test_avg_rank_ic 0.0301
  seed 1730: e1=-0.009761 e2=-0.000909 e3=0.007103 e4=0.013507 e5=0.019394 e6=0.020978 e7=0.022373 e8=0.022885 e9=0.023033 e10=0.023048
    best_val_ic 0.0230  test_avg_ic 0.0263  test_avg_rank_ic 0.0344
no_self_attention
  seed 1729: e1=0.051251 e2=0.047773 e3=0.040105 e4=0.035798 e5=0.032143 e6=0.032642 e7=0.032120 e8=0.031649 e9=0.031611 e10=0.031610
    best_val_ic 0.0513  test_avg_ic 0.0422  test_avg_rank_ic 0.0434
  seed 1730: e1=0.008487 e2=0.014464 e3=0.019342 e4=0.024780 e5=0.028447 e6=0.030480 e7=0.032367 e8=0.033293 e9=0.033394 e10=0.033512
    best_val_ic 0.0335  test_avg_ic 0.0302  test_avg_rank_ic 0.0354
pre_ln_residual
  seed 1729: e1=-0.002681 e2=0.004606 e3=0.007912 e4=0.011366 e5=0.012792 e6=0.016771 e7=0.016494 e8=0.017279 e9=0.017361 e10=0.017444
    best_val_ic 0.0174  test_avg_ic 0.0104  test_avg_rank_ic 0.0126
  seed 1730: e1=0.010442 e2=0.019084 e3=0.025473 e4=0.028475 e5=0.031211 e6=0.032602 e7=0.033479 e8=0.033631 e9=0.033663 e10=0.033664
    best_val_ic 0.0337  test_avg_ic 0.0361  test_avg_rank_ic 0.0399
two_way_latent
  seed 1729: e1=-0.001574 e2=0.006144 e3=0.009061 e4=0.014257 e5=0.015996 e6=0.018946 e7=0.019334 e8=0.019846 e9=0.020066 e10=0.020125
    best_val_ic 0.0201  test_avg_ic 0.0113  test_avg_rank_ic 0.0124
  seed 1730: e1=0.011156 e2=0.018173 e3=0.023474 e4=0.026905 e5=0.028527 e6=0.031162 e7=0.032231 e8=0.032447 e9=0.032691 e10=0.032713
    best_val_ic 0.0327  test_avg_ic 0.0333  test_avg_rank_ic 0.0367
```

### trained_attention_probe.py on the 110 universe (seed 1729)
```
mci_gru resolved to: C:\Users\magil\MCI-GRU\.claude\worktrees\looped-transformers-mci-gru-0592ea\mci_gru\__init__.py
features=16 stocks=201 val_batches=8
===== baseline =====
  [untrained] attention entropy 5.294 nats (uniform=5.30); logit std 0.176; self-weight 0.0047 (uniform=0.0050); top-5 mass 0.030
  [untrained] z before block: cos 0.634 cs_var_share 0.362 eff_rank 3.5
  [untrained] z after  block: cos 0.998 cs_var_share 0.002 eff_rank 1.5
  [untrained] validation daily IC over 239 days: mean -0.0064; output cross-sectional std nan
  loaded checkpoint (missing=0, unexpected=0)
  [trained] attention entropy 5.280 nats (uniform=5.30); logit std 0.258; self-weight 0.0048 (uniform=0.0050); top-5 mass 0.034
  [trained] z before block: cos 0.563 cs_var_share 0.433 eff_rank 3.6
  [trained] z after  block: cos 0.977 cs_var_share 0.024 eff_rank 1.6
  [trained] validation daily IC over 239 days: mean +0.0194; output cross-sectional std nan
===== no_self_attention =====
  [untrained] validation daily IC over 239 days: mean +0.0479; output cross-sectional std nan
  loaded checkpoint (missing=0, unexpected=0)
  [trained] validation daily IC over 239 days: mean +0.0513; output cross-sectional std nan
===== pre_ln_residual =====
  [untrained] attention entropy 5.294 nats (uniform=5.30); logit std 0.176; self-weight 0.0047 (uniform=0.0050); top-5 mass 0.030
  [untrained] z before block: cos 0.634 cs_var_share 0.362 eff_rank 3.5
  [untrained] z after  block: cos 0.721 cs_var_share 0.276 eff_rank 3.5
  [untrained] validation daily IC over 239 days: mean -0.0067; output cross-sectional std nan
  loaded checkpoint (missing=0, unexpected=0)
  [trained] attention entropy 5.281 nats (uniform=5.30); logit std 0.259; self-weight 0.0046 (uniform=0.0050); top-5 mass 0.033
  [trained] z before block: cos 0.594 cs_var_share 0.403 eff_rank 3.5
  [trained] z after  block: cos 0.668 cs_var_share 0.329 eff_rank 3.6
  [trained] validation daily IC over 239 days: mean +0.0174; output cross-sectional std nan
===== two_way_latent =====
  [untrained] validation daily IC over 239 days: mean -0.0070; output cross-sectional std nan
  loaded checkpoint (missing=0, unexpected=0)
  [trained] validation daily IC over 239 days: mean +0.0201; output cross-sectional std nan
```
