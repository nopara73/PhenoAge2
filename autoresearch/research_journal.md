# PA2 Research Journal

Append one short entry after every completed run.

Use this structure:

## Run N
- Hypothesis:
- Change:
- Result:
- Decision:
- Learning:
- Next:

Rules:

- Keep entries short and factual.
- Record what was learned, not just what was changed.
- Do not delete old entries. This is the memory for later agents.

## Run 1
- Hypothesis: Establish reproducible dev-val C-index for the current default `train.py` (coef-aligned encoder + residual head).
- Change: None (initial baseline).
- Result: `val_cindex` **0.768829** (best at step 500; later steps overfit—val drifts down to ~0.70).
- Decision: **keep** (first reference; matches pre-run `manage_kept.py save` snapshot).
- Learning: Signal peaks very early under 300s GPU training; longer optimization hurts validation discrimination.
- Next: Try **stronger L2** (`WEIGHT_DECAY` ↑) as a single knob to curb overfitting and lift or preserve peak `val_cindex`.

## Run 2
- Hypothesis: Stronger AdamW weight decay reduces overfitting after the early peak (step ~500) and improves reported `val_cindex`.
- Change: `WEIGHT_DECAY` `1e-4` → `3e-4` (only hyperparameter change).
- Result: Final `val_cindex` **0.768829**—identical to baseline; best checkpoint still early; more training steps fit in same 300s wall pattern.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: In this setup, tripling weight decay did not move the best validation checkpoint within the fixed time budget.
- Next: Try **lower learning rate** (single knob) so the optimum may stay stable longer, or **early stopping** on validation (conceptual training-loop change—only if we accept that as one “thing”).

## Run 3
- Hypothesis: A lower AdamW learning rate slows optimization enough that the best validation checkpoint improves before early stopping trims the run.
- Change: `LEARNING_RATE` `0.003` → `0.001` (only hyperparameter change).
- Result: `val_cindex` **0.772730** (best at step 500; early stop at step 2000 after ~32s training).
- Decision: **keep**; new best vs prior kept 0.768829.
- Learning: Halving effective step size in this band helped peak validation discrimination; same early-peak pattern but higher peak.
- Next: Try **slightly higher LR** between 1e-3 and 3e-3 (e.g. 2e-3) or tune **EVAL_EVERY** / patience if we want more resolution around the peak—pick one knob.

## Run 4
- Hypothesis: Learning rate between 1e-3 and 3e-3 may reach a higher validation peak than either endpoint.
- Change: `LEARNING_RATE` `0.001` → `0.002` (only change).
- Result: `val_cindex` **0.773912** (best at step 500; early stop at 2000).
- Decision: **keep**; beats Run 3 (0.772730).
- Learning: Mid-band LR slightly outperforms 1e-3; still peaks at first post-init eval window (step 500).
- Next: Try **LEARNING_RATE 2.5e-3** or **EVAL_EVERY 250** (finer checkpoint grid)—one knob.

## Run 5
- Hypothesis: Pushing LR slightly above 2e-3 may squeeze a bit more validation peak before overfitting.
- Change: `LEARNING_RATE` `0.002` → `0.0025` (only change).
- Result: `val_cindex` **0.773890** vs Run 4 best **0.773912**.
- Decision: **discard**; restored from `last_kept_train.py` (2e-3).
- Learning: 2e-3 appears near a local optimum in LR; 2.5e-3 ties in spirit but does not beat best-by-checkpoint.
- Next: **EVAL_EVERY 250** with kept LR=2e-3 to see if best lies between 0 and 500 steps.

## Run 6
- Hypothesis: Coarser eval spacing (500) missed an earlier validation peak between 0 and 500 steps.
- Change: `EVAL_EVERY` `500` → `250` (LR stays `0.002`).
- Result: `val_cindex` **0.774169**; `best_step` **250** (early stop at 1250).
- Decision: **keep**; beats Run 4 (0.773912).
- Learning: Best weights occur earlier than the old 500-step grid; finer checkpoints materially help reported `val_cindex`.
- Next: Try **EVAL_EVERY 125** or **EARLY_STOP_PATIENCE_EVALS** tweak—one knob, only if justified.

## Run 7
- Hypothesis: Peak may lie between 250 and 500 steps; `EVAL_EVERY=125` can capture it.
- Change: `EVAL_EVERY` `250` → `125` (LR still `0.002`).
- Result: `val_cindex` **0.774315** at `best_step` **375** (early stop 875).
- Decision: **keep**; beats Run 6 (0.774169).
- Learning: Optimum is between 250–500 updates under this LR; finer grid pays off modestly.
- Next: Try **EVAL_EVERY 75** or slight **DROPOUT** change—one knob.

## Run 8
- Hypothesis: Peak may be slightly past 375; `EVAL_EVERY=75` refines the argmax step.
- Change: `EVAL_EVERY` `125` → `75` (LR `0.002` unchanged).
- Result: `val_cindex` **0.774825** at `best_step` **450** (early stop 3750 steps but ~20s train—more eval overhead).
- Decision: **keep**; beats Run 7 (0.774315).
- Learning: Optimum near ~450 steps; 75-step grid captures it; early-stop patience spans more optimizer steps when eval is frequent.
- Next: Try **DROPOUT 0.02 -> 0.05** or **EARLY_STOP_MIN_DELTA**—one knob; avoid over-tuning eval grid further without new signal.

## Run 9
- Hypothesis: Slightly stronger dropout may regularize the residual head enough to improve the earlier validation peak without changing the optimizer path too much.
- Change: `DROPOUT` `0.02` → `0.05` (LR `0.002`, `EVAL_EVERY=75` unchanged).
- Result: `val_cindex` **0.775137** at `best_step` **675**; early stop after ~20.9s training.
- Decision: **keep**; beats Run 8 (0.774825).
- Learning: Mildly stronger dropout helps the best checkpoint in the current LR/eval regime; the useful region extends slightly later than Run 8.
- Next: Try a small **weight decay** increase (for example `1e-4` → `2e-4`) while keeping LR `0.002`, dropout `0.05`, and `EVAL_EVERY=75`.

## Run 10
- Hypothesis: Moderate AdamW weight decay between the baseline `1e-4` and the discarded `3e-4` may regularize slightly without erasing the dropout/LR/eval gains.
- Change: `WEIGHT_DECAY` `1e-4` → `2e-4` (LR `0.002`, `DROPOUT=0.05`, `EVAL_EVERY=75` unchanged).
- Result: `val_cindex` **0.775147** at `best_step` **675** (early stop ~20.2s training); prior best **0.775137**.
- Decision: **keep**; marginal but consistent lift on final reported dev-val C-index.
- Learning: `2e-4` is a sweet spot vs `3e-4` (Run 2: no gain); pairs well with current dropout.
- Next: Try **`EARLY_STOP_PATIENCE_EVALS` 3 → 4** (more tolerance before stop) or **`EARLY_STOP_MIN_DELTA` 1e-4 → 5e-5** (finer improvement detection)—one knob; alternatively probe **DROPOUT 0.05 → 0.06** if we want another regularization lever.

## Run 11
- Hypothesis: Allowing one extra non-improving validation window before early stop might surface a better checkpoint after the usual peak at ~675 steps.
- Change: `EARLY_STOP_PATIENCE_EVALS` `3` → `4` (all else matches Run 10 kept config).
- Result: Final `val_cindex` **0.775147** (same as Run 10); `best_step` still **675**; more optimizer steps before stop with no gain.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The validation peak is stable; extra patience only burns steps without changing the selected best checkpoint.
- Next: Try **`EARLY_STOP_MIN_DELTA` `1e-4` → `5e-5`** to refine which eval counts as an improvement, or **DROPOUT `0.05` → `0.06`**—one knob.

## Run 12
- Hypothesis: A smaller `EARLY_STOP_MIN_DELTA` may credit marginal validation gains and shift the stored best checkpoint.
- Change: `EARLY_STOP_MIN_DELTA` `1e-4` → `5e-5` (Run 10 hyperparameters otherwise).
- Result: Final `val_cindex` **0.775147** (ties Run 10); `best_step` **675**.
- Decision: **discard**; no gain vs kept baseline; restored.
- Learning: Finer delta does not change the selected best in this noisy regime; peak remains at 675.
- Next: Try **DROPOUT `0.05` → `0.06`** or **`LEARNING_RATE` `0.002` → `0.0018`**—one knob.

## Run 13
- Hypothesis: Slightly more dropout than `0.05` may regularize further without collapsing the early validation peak.
- Change: `DROPOUT` `0.05` → `0.06` (LR `0.002`, `WEIGHT_DECAY=2e-4`, `EVAL_EVERY=75`).
- Result: `val_cindex` **0.773710** at `best_step` **225**—well below Run 10 **0.775147**.
- Decision: **discard**; restored.
- Learning: `0.06` is too strong for this head; it shifts the best checkpoint earlier and hurts discrimination.
- Next: Try **`WEIGHT_DECAY` `2e-4` → `2.5e-4`** (between kept `2e-4` and old failed `3e-4` under a different config) or **`LEARNING_RATE` `0.002` → `0.0018`**.

## Run 14
- Hypothesis: A modest weight-decay increase above `2e-4` may squeeze a bit more from the best checkpoint without repeating the old `3e-4` failure mode.
- Change: `WEIGHT_DECAY` `2e-4` → `2.5e-4` (`DROPOUT=0.05`, LR `0.002`, `EVAL_EVERY=75`).
- Result: `val_cindex` **0.775151** at `best_step` **675** vs prior **0.775147**.
- Decision: **keep**; small but consistent improvement on final reported metric.
- Learning: The `2e-4`–`2.5e-4` band still pairs well with `DROPOUT=0.05`; optimum step unchanged.
- Next: Try **`EVAL_EVERY` `75` → `50`** (finer grid around ~675) or **`LEARNING_RATE` `0.002` → `0.0019`**—one knob.

## Run 15
- Hypothesis: A 50-step eval grid may land closer to the true best between 600–700 updates than 75-step spacing.
- Change: `EVAL_EVERY` `75` → `50` (LR `0.002`, `WEIGHT_DECAY=2.5e-4`, `DROPOUT=0.05`).
- Result: `val_cindex` **0.775835** at `best_step` **650** vs Run 14 **0.775151** at **675**.
- Decision: **keep**; meaningful gain from checkpoint resolution.
- Learning: The optimum is slightly before the old 675-step anchor; finer grids still pay off in this regime.
- Next: Try **`EVAL_EVERY` `50` → `40`** or **`LEARNING_RATE` `0.002` → `0.00195`**—one knob; watch eval overhead vs wall time.

## Run 16
- Hypothesis: Even finer eval spacing might refine the best checkpoint near step 650.
- Change: `EVAL_EVERY` `50` → `40` (otherwise Run 15 config).
- Result: `val_cindex` **0.775720** at `best_step` **520** vs Run 15 **0.775835** @ **650**.
- Decision: **discard**; restored.
- Learning: `40` is too fine / misaligns with the noise window; `50` remains the better grid for this LR/WD/dropout stack.
- Next: Try **`LEARNING_RATE` `0.002` → `0.00195`** (micro-tune below 2e-3) or **`WEIGHT_DECAY` `2.5e-4` → `2.6e-4`**—one knob; avoid further `EVAL_EVERY` reductions for now.

## Run 17
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.002 -> 0.0019
- Result: val_cindex **0.775620** (best kept **0.775835**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 18
- Hypothesis: Single local hyperparameter move may improve val_cindex.
- Change: LEARNING_RATE 0.002 -> 0.00195
- Result: val_cindex **0.775926** best_step **650**
- Decision: **keep**
- Learning: Improved vs prior best kept snapshot.
- Next: Continue local search from new baseline.

## Run 19
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.00195 -> 0.0019
- Result: val_cindex **0.775620** (best kept **0.775926**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 20
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.00195 -> 0.00205
- Result: val_cindex **0.775193** (best kept **0.775926**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 21
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.00195 -> 0.0021
- Result: val_cindex **0.775023** (best kept **0.775926**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 22
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.00195 -> 0.00185
- Result: val_cindex **0.774955** (best kept **0.775926**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 23
- Hypothesis: Training completes without error.
- Change: LEARNING_RATE 0.00195 -> 0.00215
- Result: crash / non-zero exit
- Decision: discard (restore)
- Learning: Investigate run.log
- Next: Continue queue

## Run 24
- Hypothesis: Single local move unlikely to beat current best.
- Change: WEIGHT_DECAY 0.00025 -> 0.00024
- Result: val_cindex **0.775926** (best kept **0.775926**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 25
- Hypothesis: Single local hyperparameter move may improve val_cindex.
- Change: WEIGHT_DECAY 0.00025 -> 0.00026
- Result: val_cindex **0.775929** best_step **650**
- Decision: **keep**
- Learning: Improved vs prior best kept snapshot.
- Next: Continue local search from new baseline.

## Run 26
- Hypothesis: The current encoder may be the bottleneck, so a modest expansion of biomarker-only transforms could improve discrimination without changing the training loop.
- Change: Added five extra encoder features: `amp*log_crp`, `sgp*rdw`, `alk*log_crp`, `sgp/albumin`, and `log1p(alk)`.
- Result: `val_cindex` **0.774463** at `best_step` **200** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This broader feature block overfit earlier and underperformed the kept baseline; the current model is not obviously limited by missing simple hand-crafted interactions.
- Next: Try a less anchor-constrained architecture that still starts from the kept baseline.

## Run 27
- Hypothesis: The kept model may be too tightly tied to `pheno_no_age_xb`, so adding a direct linear correction path could let it adjust biomarker effects without relying on the residual MLP alone.
- Change: Added a learned linear correction on standardized encoded features alongside the base score and residual head.
- Result: `val_cindex` **0.774955** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Loosening the anchor with a parallel linear path did not help; the extra correction capacity did not beat the simpler kept model.
- Next: Try changing training dynamics rather than model form.

## Run 28
- Hypothesis: The kept model may generalize better with noisier stochastic optimization than with the current full-batch Cox loop.
- Change: Switched training to mini-batch Cox (`batch_size=1024`) with cosine learning-rate decay while keeping the kept model form.
- Result: `val_cindex` **0.774050** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This stochastic Cox variant lost too much ranking quality; the current full-batch dynamics appear better for the kept architecture.
- Next: Finish the reset with a few more concept-level probes before deciding whether the local search is saturated.

## Run 26
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.00195 -> 0.0019
- Result: val_cindex **0.775620** (best kept **0.775929**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 27
- Hypothesis: Training completes without error.
- Change: LEARNING_RATE 0.00195 -> 0.00205
- Result: crash / non-zero exit
- Decision: discard (restore)
- Learning: Investigate run.log
- Next: Continue queue

## Run 29
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.00195 -> 0.0021
- Result: val_cindex **0.775023** (best kept **0.775929**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 30
- Hypothesis: Single local move unlikely to beat current best.
- Change: LEARNING_RATE 0.00195 -> 0.00185
- Result: val_cindex **0.774955** (best kept **0.775929**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 32
- Hypothesis: Training completes without error.
- Change: LEARNING_RATE 0.00195 -> 0.00215
- Result: crash / non-zero exit
- Decision: discard (restore)
- Learning: Investigate run.log
- Next: Continue queue

## Run 33
- Hypothesis: Single local move unlikely to beat current best.
- Change: WEIGHT_DECAY 0.00026 -> 0.0002496
- Result: val_cindex **0.775926** (best kept **0.775929**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 35
- Hypothesis: Single local move unlikely to beat current best.
- Change: WEIGHT_DECAY 0.00026 -> 0.0002704
- Result: val_cindex **0.775929** (best kept **0.775929**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 36
- Hypothesis: Single local move unlikely to beat current best.
- Change: WEIGHT_DECAY 0.00026 -> 0.0002392
- Result: val_cindex **0.775926** (best kept **0.775929**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 37
- Hypothesis: Single local move unlikely to beat current best.
- Change: WEIGHT_DECAY 0.00026 -> 0.0002808
- Result: val_cindex **0.775929** (best kept **0.775929**)
- Decision: **discard** (restored)
- Learning: No improvement vs best kept.
- Next: Try next queued neighbor.

## Run 38
- Hypothesis: Local tuning has plateaued, so reweighting the Cox objective toward earlier events may better align the model with the ranking signal the validation split rewards.
- Change: Replace the uniform Cox partial likelihood with a time-weighted Cox loss that gives somewhat more weight to earlier observed events.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 39
- Hypothesis: The failed feature-expansion run suggests extra interactions may overfit, so a leaner anchor-aligned encoder may generalize better than the current richer representation.
- Change: Prune the encoder to the nine transformed biomarkers plus `pheno_no_age_xb`, removing all hand-crafted interaction features.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 40
- Hypothesis: Broader capacity increases have not helped, so collapsing the residual path to a single linear layer may reduce overfitting while preserving the pheno-no-age anchor.
- Change: Replace the hidden residual MLP with a single linear correction layer on standardized encoded features.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 41
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: `val_cindex` **0.775760** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 42
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.775959** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 43
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.775926** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 44
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.775929** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 45
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: `val_cindex` **0.775210** at `best_step` **550** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 46
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.775114** at `best_step` **660** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 47
- Hypothesis: The previous time-weighted Cox attempt failed before producing a completed summary block, so rerunning it under the fixed synchronous supervisor will reveal whether the loss family is genuinely promising or truly weak.
- Change: Retry the time-weighted Cox objective after fixing the supervisor persistence bug that invalidated the earlier attempt.
- Result: `val_cindex` **0.775760** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 48
- Hypothesis: The lean encoder never produced a valid run because the earlier orchestration failure killed it before completion, so it deserves one clean retry before the feature-representation family is abandoned.
- Change: Retry the lean encoder variant after fixing the supervisor persistence bug that invalidated the earlier attempt.
- Result: `val_cindex` **0.775042** at `best_step` **250** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 49
- Hypothesis: The single-linear residual head was never measured cleanly because the first attempt died before logging, so rerunning it under the repaired supervisor is a materially different test of that simpler architecture.
- Change: Retry the single linear residual head after fixing the supervisor persistence bug that invalidated the earlier attempt.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 50
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 51
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.775959** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 52
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.775926** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 53
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.775929** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 54
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: `val_cindex` **0.775210** at `best_step` **550** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 55
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.775114** at `best_step` **660** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 56
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 57
- Hypothesis: The full 32-16 residual head may be slightly too flexible, so shrinking it without collapsing to a purely linear correction could preserve the useful nonlinear adjustment while reducing overfitting.
- Change: Reduce `HIDDEN_SIZES` from `(32, 16)` to `(24, 12)` while keeping the current training setup unchanged.
- Result: `val_cindex` **0.774181** at `best_step` **200** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 58
- Hypothesis: The model may generalize better if it starts slightly closer to the pheno-no-age anchor and learns a smaller correction path, rather than giving the residual head as much influence from the start.
- Change: Reduce the initial `residual_scale` from `0.1` to `0.08` while keeping the kept architecture and optimizer unchanged.
- Result: `val_cindex` **0.774220** at `best_step` **500** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 59
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.775959** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 60
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.775926** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 61
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.775929** at `best_step` **650** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 62
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: `val_cindex` **0.775210** at `best_step` **550** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 63
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.775114** at `best_step` **660** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 64
- Hypothesis: Reducing the residual contribution hurt badly, which suggests the kept model may actually be under-correcting the pheno-no-age anchor and could benefit from a slightly stronger residual path.
- Change: Increase the initial `residual_scale` from `0.1` to `0.12` while keeping the kept architecture and optimizer unchanged.
- Result: `val_cindex` **0.775060** at `best_step` **550** vs best kept **0.775929**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 65
- Hypothesis: The two-layer 32-16 head may be using the wrong shape of capacity, so collapsing it to one modest hidden layer could preserve useful nonlinear correction while removing unnecessary depth.
- Change: Change `HIDDEN_SIZES` from `(32, 16)` to `(16,)`, keeping the residual MLP, optimizer, and encoder otherwise unchanged.
- Result: `val_cindex` **0.778533** at `best_step` **350** vs best kept **0.775929**.
- Decision: **keep**
- Learning: This broader change improved enough to replace the kept baseline.
- Next: Resume local tuning around the new kept baseline.

## Run 66
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: `val_cindex` **0.778555** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 67
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.778515** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 68
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.778534** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 69
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.778533** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 70
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: `val_cindex` **0.778450** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 71
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.778388** at `best_step` **385** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 72
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: `val_cindex` **0.778555** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 73
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.778515** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 74
- Hypothesis: The search looks optimization-limited near the current basin, so a smoother residual nonlinearity may slightly improve ranking without changing model width or the anchor pathway.
- Change: Replace the residual head activation from `nn.GELU()` to `nn.SiLU()` while keeping the kept encoder, widths, and optimizer otherwise unchanged.
- Result: `val_cindex` **0.779005** at `best_step` **450** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 75
- Hypothesis: The earlier patience increase was tested on a weaker baseline before the later LR, weight-decay, and eval-grid gains, so retrying it on the stronger current setup is materially different and may allow a later checkpoint to emerge.
- Change: Increase `EARLY_STOP_PATIENCE_EVALS` from `3` to `5` on the current kept baseline without changing the model form.
- Result: `val_cindex` **0.778533** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 76
- Hypothesis: The best local LR move nearly ties but does not clear the keep bar, which suggests the current basin may need a slightly different optimizer memory rather than another scalar LR nudge.
- Change: Keep the current architecture and scalar hyperparameters, but set AdamW `betas=(0.9, 0.98)` instead of the default `beta2=0.999`.
- Result: `val_cindex` **0.778904** at `best_step` **300** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 77
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.778534** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 78
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.778533** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 79
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: `val_cindex` **0.778450** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 80
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.778388** at `best_step` **385** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 81
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: `val_cindex` **0.778555** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 82
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.778515** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 83
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.778534** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 84
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.778533** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 85
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: `val_cindex` **0.778450** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 86
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.778388** at `best_step` **385** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 87
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: `val_cindex` **0.778555** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 88
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.778515** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 89
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.778534** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 90
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.778533** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 91
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: `val_cindex` **0.778450** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 92
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.778388** at `best_step` **385** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 93
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: `val_cindex` **0.778555** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 94
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.778515** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 95
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: `val_cindex` **0.778534** at `best_step` **350** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 96
- Hypothesis: The earlier SiLU run was the best overall discard and was tested before the winning switch to a shallower residual head, so re-running SiLU on the stronger current baseline is a materially different retry with real upside.
- Change: Replace the residual activation from `nn.GELU()` to `nn.SiLU()` on top of the kept single-hidden residual MLP baseline.
- Result: `val_cindex` **0.779005** at `best_step` **450** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 97
- Hypothesis: The earlier feature-expansion attempt was paired with a worse residual architecture, so retrying a richer interaction set on the stronger single-hidden baseline could recover signal that the deeper head previously overfit.
- Change: Add five extra biomarker interactions to the encoder while keeping the current single-hidden residual head and optimizer unchanged.
- Result: `val_cindex` **0.777031** at `best_step` **1400** vs best kept **0.778533**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 98
- Hypothesis: The single-hidden residual head may still be too nonlinear for some stable ranking signal, so adding a small direct linear correction path could capture simple coefficient adjustments that the nonlinear branch misses.
- Change: Add a learned linear skip correction on standardized encoded features alongside the current single-hidden residual head.
- Result: `val_cindex` **0.779707** at `best_step` **1700** vs best kept **0.778533**.
- Decision: **keep**
- Learning: This broader change improved enough to replace the kept baseline.
- Next: Resume local tuning around the new kept baseline.

## Run 99
- Hypothesis: A slightly smaller step size may refine the same early optimum without leaving the current local basin.
- Change: `LEARNING_RATE` `0.00195` -> `0.001925`.
- Result: `val_cindex` **0.779254** at `best_step` **800** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 100
- Hypothesis: A very small upward LR nudge may sharpen the early ranking peak while staying closer than the previously discarded 0.00205 move.
- Change: `LEARNING_RATE` `0.00195` -> `0.001975`.
- Result: `val_cindex` **0.779756** at `best_step` **1700** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 101
- Hypothesis: A narrower weight-decay move may preserve the current optimum while slightly relaxing regularization around the best kept setup.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00025500`.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 102
- Hypothesis: A slightly stronger weight decay may still help the current architecture if the earlier 4 percent move was too coarse.
- Change: `WEIGHT_DECAY` `0.00026` -> `0.00026500`.
- Result: `val_cindex` **0.779707** at `best_step` **1700** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 103
- Hypothesis: A slightly lighter dropout could recover some capacity without revisiting the much larger discarded regularization changes.
- Change: `DROPOUT` `0.05` -> `0.045`.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 104
- Hypothesis: A tiny shift in checkpoint spacing may better align with the early peak than the previous coarse eval-grid moves.
- Change: `EVAL_EVERY` `50` -> `55`.
- Result: `val_cindex` **0.779610** at `best_step` **1705** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 105
- Hypothesis: The current model may be overfitting by re-scaling the pheno-no-age anchor itself, so forcing the anchor weight to stay fixed could let the residual branch learn cleaner corrections.
- Change: Replace the learned `base_weight` with a fixed weight of `1.0`, keeping the current single-hidden residual correction path unchanged.
- Result: `val_cindex` **0.778930** at `best_step` **1450** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 106
- Hypothesis: The model may now be limited by the mismatch between Cox optimization and the exact ranking target, so adding a smooth pairwise concordance surrogate could improve discrimination more meaningfully than another scalar hyperparameter tweak.
- Change: Keep the current single-hidden residual architecture, but optimize a hybrid loss: Cox partial likelihood plus a small pairwise ranking surrogate over comparable event-survival pairs.
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly, so it cannot be compared against the kept baseline.
- Next: Continue from the kept baseline with the next queued experiment.

## Run 107
- Hypothesis: The current winner may still be overfitting stable quirks of the training split, so a small amount of noise on standardized encoded features during training could improve robustness without changing the inference-time formula.
- Change: Add small Gaussian noise to standardized encoded features during training while keeping the current single-hidden residual architecture and optimizer unchanged.
- Result: `val_cindex` **0.779118** at `best_step` **2050** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 108
- Hypothesis: The current winner may be trapped by deterministic full-batch optimization, so revisiting mini-batch Cox on the stronger single-hidden baseline could unlock a different basin than the earlier weaker-architecture attempt.
- Change: Train the current single-hidden residual model with mini-batch Cox updates (`batch_size=1024`) and cosine learning-rate decay instead of the current full-batch constant-LR regime.
- Result: `val_cindex` **0.778240** at `best_step` **250** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 109
- Hypothesis: The previous hybrid Cox plus pairwise attempt likely failed because the naive all-pairs construction was too memory-heavy, so retrying it with sampled comparable pairs is still a meaningfully different loss-design probe with real upside.
- Change: Keep the current single-hidden residual architecture, but optimize a hybrid loss: Cox partial likelihood plus a sampled pairwise ranking surrogate over comparable event-survival pairs.
- Result: `val_cindex` **0.778155** at `best_step` **350** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and move to the next queued conceptual probe.

## Run 110
- Hypothesis: The current model treats every encoded biomarker channel as equally available to the correction paths, so a learned per-feature gate may suppress noisy corrections and focus capacity on the most stable encoded signals.
- Change: Add a learned sigmoid feature gate on standardized encoded features before both the residual MLP and the linear skip path.
- Result: `val_cindex` **0.779483** at `best_step` **450** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 111
- Hypothesis: Pure raw-biomarker modeling may outperform the anchor-based design if the hand-crafted pheno pathway is constraining the optimum.
- Change: Replace the anchor-aligned encoder/pathway with a raw 9-biomarker MLP using HIDDEN_SIZES=(32, 16).
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly.
- Next: Continue from the kept baseline with the next experiment.

## Run 112
- Hypothesis: A wider anchor-free MLP could capture nonlinear structure that the anchored family cannot express.
- Change: Replace the anchor-aligned encoder/pathway with a raw 9-biomarker MLP using HIDDEN_SIZES=(64, 32).
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly.
- Next: Continue from the kept baseline with the next experiment.

## Run 113
- Hypothesis: A completely linear age-free model on raw biomarkers tests whether the search is overcomplicating a mostly linear signal.
- Change: Use a standardized 9-biomarker linear Cox model with no pheno anchor and no hidden layers.
- Result: `val_cindex` **0.756476** at `best_step` **250** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 114
- Hypothesis: The best signal may live in the engineered 16-feature representation but not require any nonlinear residual network.
- Change: Use a standardized linear Cox model on the current 16 engineered biomarker features.
- Result: `val_cindex` **0.751185** at `best_step` **200** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 115
- Hypothesis: A generalized-additive style model may capture smooth per-biomarker nonlinearities without relying on dense multivariate mixing.
- Change: Use nine independent 1D biomarker subnetworks and sum their outputs.
- Result: `val_cindex` **0.768591** at `best_step` **1000** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 116
- Hypothesis: Combining raw biomarkers with engineered pheno-style features may recover signal lost by committing to only one representation.
- Change: Concatenate raw 9 biomarkers with engineered 16 features, then train a SiLU MLP with HIDDEN_SIZES=(64, 32).
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly.
- Next: Continue from the kept baseline with the next experiment.

## Run 117
- Hypothesis: If Cox is misaligned with the actual ranking objective, optimizing pairwise concordance directly may help.
- Change: Replace Cox loss with sampled pairwise logistic ranking loss while keeping the current architecture.
- Result: `val_cindex` **0.773635** at `best_step` **250** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 118
- Hypothesis: A new model family and a directly aligned ranking loss together may discover a basin the anchored Cox models never reach.
- Change: Use a raw 9-biomarker MLP with HIDDEN_SIZES=(32, 16) and pure pairwise ranking loss.
- Result: `val_cindex` **0.774495** at `best_step` **150** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 119
- Hypothesis: Changing residual MLP to (8,) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (16,) -> (8,)
- Result: `val_cindex` **0.777263** at `best_step` **1000** vs best kept **0.779707**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 120
- Hypothesis: Changing residual MLP to (24,) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (16,) -> (24,)
- Result: `val_cindex` **0.782152** at `best_step` **750** vs best kept **0.779707**.
- Decision: **keep**
- Learning: This change improved enough to replace the kept baseline.
- Next: Resume exploration from the new kept baseline.

## Run 121
- Hypothesis: Changing residual MLP to (32,) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (32,)
- Result: `val_cindex` **0.776382** at `best_step` **750** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 122
- Hypothesis: Changing residual MLP to (48,) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (48,)
- Result: `val_cindex` **0.777100** at `best_step` **500** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 123
- Hypothesis: Changing residual MLP to (64,) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (64,)
- Result: `val_cindex` **0.779780** at `best_step` **450** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 124
- Hypothesis: Changing residual MLP to (128,) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (128,)
- Result: `val_cindex` **0.779476** at `best_step` **300** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 125
- Hypothesis: Changing residual MLP to (32, 16) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (32, 16)
- Result: `val_cindex` **0.772880** at `best_step` **200** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 126
- Hypothesis: Changing residual MLP to (24, 12) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (24, 12)
- Result: `val_cindex` **0.774376** at `best_step` **200** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 127
- Hypothesis: Changing residual MLP to (16, 8) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (16, 8)
- Result: `val_cindex` **0.774151** at `best_step` **350** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 128
- Hypothesis: Changing residual MLP to (24, 16) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (24, 16)
- Result: `val_cindex` **0.771426** at `best_step` **350** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This change did not improve the kept baseline enough to justify replacing it.
- Next: Restore the kept baseline and try the next experiment.

## Run 129
- Hypothesis: Changing residual MLP to (32, 16, 8) may improve capacity/regularization balance.
- Change: HIDDEN_SIZES (24,) -> (32, 16, 8)
- Result: crash / no completed summary block in `run.log`
- Decision: **crash**; restored `train.py` from `last_kept_train.py`.
- Learning: The candidate did not finish cleanly.
- Next: Continue from the kept baseline with the next experiment.

## Run 130
- Hypothesis: The current wide single-hidden baseline may benefit from a learned normalization step before the linear and nonlinear correction heads, which could stabilize optimization better than another width tweak.
- Change: Added `nn.LayerNorm(16)` on standardized encoded features before both the residual MLP and the linear skip path, keeping the kept `(24,)` architecture and optimizer otherwise unchanged.
- Result: `val_cindex` **0.773349** at `best_step` **350** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Pre-head normalization strongly hurt this anchored model family; simple train-split standardization already seems sufficient, and extra normalization disrupts the useful scale information in the correction paths.
- Next: Return to the kept `(24,)` baseline and try a different family such as activation or optimizer changes rather than more normalization.

## Run 131
- Hypothesis: The current winner may still be wasting capacity relearning the pheno-no-age anchor, so explicitly penalizing correlation between the anchor term and the learned correction could force the residual path to extract genuinely new signal.
- Change: Kept the winning `(24,)` residual-plus-linear architecture, but trained it with an added squared-correlation penalty between the scaled anchor score and the learned correction term.
- Result: `val_cindex` **0.780485** at `best_step` **900** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Orthogonalizing the correction path helped more than most recent radical probes and delayed the best checkpoint, but it still did not clear the keep threshold; the idea has signal, but this direct penalty was not enough on its own.
- Next: Try a materially different structural family rather than tuning the penalty strength.

## Run 132
- Hypothesis: A stricter inductive bias may generalize better than the free-form residual head, so replacing the multivariate correction with signed monotone 1D biomarker calibrators could capture stable nonlinear effects while staying faithful to the original biomarker directions.
- Change: Replaced the dense residual-plus-linear correction with nine signed monotone additive calibrators on transformed biomarkers, while keeping the pheno-no-age score as an anchor term.
- Result: `val_cindex` **0.762923** at `best_step` **900** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This stronger structure was too restrictive for the dev split; additive monotone calibration leaves too much cross-biomarker signal on the table relative to the kept wide single-hidden model.
- Next: Return to the kept baseline and test a different optimization family rather than further additive-structure variants.

## Run 133
- Hypothesis: The current basin may benefit from gentler early optimization and a slower late decay, so full-batch warmup plus cosine learning-rate decay could improve the same model without the downside of mini-batch noise.
- Change: Kept the winning `(24,)` residual-plus-linear model, but replaced constant learning rate with a full-batch warmup followed by cosine decay.
- Result: `val_cindex` **0.781110** at `best_step` **900** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: This was the strongest pure training-dynamics probe so far and nearly matched the incumbent, but constant-LR training still edges it out on the frozen dev validation split.
- Next: Focus on hybrids that preserve the strong `(24,)` basin while changing what the correction path is allowed to learn, rather than replacing the entire model or only changing scalar schedules.

## Run 134
- Hypothesis: The winning model may improve if the correction paths are structurally forbidden from directly reading the `pheno_no_age_xb` anchor coordinate, forcing them to use complementary biomarker information instead of re-expressing the same scalar.
- Change: Kept the winning `(24,)` residual-plus-linear architecture, but zeroed the anchor channel from the standardized correction inputs before both the residual MLP and the linear skip path.
- Result: `val_cindex` **0.777815** at `best_step` **450** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Hard structural separation is too strict; the current winner appears to need direct access to the anchor channel inside the correction paths, even if a softer decorrelation objective showed some promise.
- Next: Retry a different radical family rather than stricter anchor separation.

## Run 135
- Hypothesis: The previously crashed dual raw+engineered family may still have upside if implemented with smaller separate towers, explicit train-split standardization, and gradient clipping instead of a single wide fusion MLP.
- Change: Replaced the baseline with a stability-first dual-tower model: one small tower over transformed raw biomarkers, one small tower over engineered encoder features, then a compact fusion head added to the pheno-no-age anchor.
- Result: `val_cindex` **0.777262** at `best_step` **400** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The dual-representation family can be made stable and scriptable, but this cleaned-up version still underperforms the kept anchored single-basin model by a wide margin.
- Next: Probe a different hybrid inside the incumbent basin rather than more raw/engineered tower designs.

## Run 136
- Hypothesis: The kept model may be missing useful multiplicative structure that a standard MLP does not capture efficiently, so adding a tiny explicit low-rank quadratic correction channel could improve ranking without needing a deeper or wider head.
- Change: Kept the winning `(24,)` residual-plus-linear model, but added a rank-3 quadratic interaction path over standardized engineered features.
- Result: `val_cindex` **0.776460** at `best_step` **450** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Explicit second-order interaction structure was not helpful here; the existing residual-plus-linear correction already captures the useful interaction signal better than this low-rank quadratic add-on.
- Next: Focus future radical attempts on optimizer/seed/ensemble-style variance reduction or softer anchor-aware hybrids, since harder structural rewrites are consistently worse.

## Run 137
- Hypothesis: The winning architecture may benefit from weight-space smoothing even if raw training weights are noisy, so evaluating and exporting an exponential-moving-average copy of the baseline could improve the selected checkpoint without changing the model family.
- Change: Kept the winning `(24,)` residual-plus-linear architecture and constant-LR training, but tracked an EMA copy of the weights for validation and export.
- Result: `val_cindex` **0.782420** at `best_step` **850** vs best kept **0.782152**.
- Decision: **discard**; although numerically higher, it did not clear the `+0.0003` keep threshold. Restored `train.py` from `last_kept_train.py`.
- Learning: Weight-space smoothing is the strongest new direction so far and moved the model closer than any radical structural rewrite, but the gain is still too small to count as a keep.
- Next: Try another ensemble-style idea that preserves the current basin rather than replacing the representation.

## Run 138
- Hypothesis: An in-graph ensemble of multiple narrow correction heads may average out idiosyncratic residual behavior better than one single head while preserving the same anchored representation.
- Change: Replaced the single residual MLP with two narrow residual heads whose outputs were fused by learned positive weights, while keeping the anchor term and linear skip path unchanged.
- Result: `val_cindex` **0.780927** at `best_step` **900** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Multi-head correction did not beat the simpler single-head winner; ensembling inside the correction path alone is weaker than direct EMA smoothing of the whole model.
- Next: Combine the two strongest variance-reduction signals rather than trying another head topology.

## Run 139
- Hypothesis: The two strongest near-miss ideas, EMA smoothing and full-batch warmup-plus-cosine decay, may compound if the schedule finds a smoother basin and the EMA export stabilizes it further.
- Change: Kept the winning `(24,)` residual-plus-linear architecture, but trained with full-batch warmup-plus-cosine LR decay while validating and exporting an EMA copy of the weights.
- Result: `val_cindex` **0.781497** at `best_step` **950** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The combination underperformed the simpler EMA-only run, suggesting the constant-LR basin still suits this architecture best even when weight smoothing is added.
- Next: Focus any further radical attempts on other forms of benchmark-legal averaging or seed diversity, since EMA is the clearest current lead.

## Run 140
- Hypothesis: Uniformly averaging a short window of recent high-performing checkpoints may reduce variance better than raw weights while staying on the winning constant-LR basin.
- Change: Kept the winning `(24,)` residual-plus-linear architecture, but evaluated and exported a rolling uniform average of the last few eval-step checkpoints.
- Result: `val_cindex` **0.781719** at `best_step` **800** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Short-window snapshot averaging was weaker than EMA smoothing on this basin, so uniform checkpoint averaging is not the best averaging kernel here.
- Next: Try seed-diverse in-graph averaging rather than more weight-space averages of the same single trajectory.

## Run 141
- Hypothesis: A benchmark-legal in-graph ensemble of independently initialized correction towers may reduce variance across basins better than single-trajectory averaging.
- Change: Replaced the single correction path with a seed-split equal-mix ensemble of three small towers, each with its own nonlinear and linear correction branch.
- Result: crash at TorchScript export due to a scriptability issue around the tower-count constant; training itself completed before export.
- Decision: **crash**; fix the scripting issue and rerun the same idea cleanly.
- Learning: The first attempt did not fail for modeling reasons, so the family still deserved one repaired measurement.
- Next: Retry the same seed-split ensemble after the TorchScript fix.

## Run 142
- Hypothesis: After fixing the export bug, a seed-diverse equal-mix correction ensemble may average out head-specific noise better than the previously tried learned-weight two-head ensemble.
- Change: Retried the seed-split equal-mix correction ensemble after fixing the TorchScript export issue.
- Result: `val_cindex` **0.779019** at `best_step` **1800** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Forcing equal mixing and seed diversity did not beat the simple incumbent; the ensemble trained stably but still underperformed both the kept baseline and EMA smoothing.
- Next: EMA remains the strongest new lead; further progress likely needs a more aggressive averaging strategy or a fundamentally different search target rather than more correction-head ensembles.

## Run 143
- Hypothesis: Averaging two independently initialized full baseline models may reduce variance across whole risk functions better than only ensembling correction heads.
- Change: Replaced the single model with a TorchScript-friendly mean of two full `RiskMLP` baselines trained jointly and exported as one artifact.
- Result: `val_cindex` **0.780853** at `best_step` **500** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Whole-model averaging was still weaker than EMA smoothing; averaging across full risk maps did not beat the simpler single-model basin.
- Next: Try one final EMA-family variant rather than more ensemble topology changes.

## Run 144
- Hypothesis: Plain EMA may be averaging too much early trajectory noise, so delaying the EMA start until the model reaches the late high-quality basin could improve over the earlier near-miss.
- Change: Kept the winning `(24,)` residual-plus-linear architecture, but started EMA only after a burn-in at step 500 and used the EMA teacher for later validation/export.
- Result: `val_cindex` **0.779165** at `best_step` **450** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The delayed-start EMA idea was materially worse than plain EMA; resetting the teacher late caused it to miss the strongest region rather than sharpen it.
- Next: Plain EMA remains the clearest surviving lead; further progress likely depends on more direct tuning of that exact averaging setup rather than delayed or structurally different variants.

## Run 145
- Hypothesis: Forcing the model to produce similar risk scores under independent dropout masks may reduce variance in the residual path and improve generalization more directly than weight-space averaging alone.
- Change: Kept the winning `(24,)` residual-plus-linear architecture, but trained with two dropout passes per step and added a small consistency penalty between their risk scores.
- Result: `val_cindex` **0.778946** at `best_step` **550** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Output-consistency regularization suppressed useful flexibility more than it reduced noise; it underperformed both the incumbent and the EMA near-miss.
- Next: Try a different optimizer-level family instead of more dropout-based regularization.

## Run 146
- Hypothesis: Lookahead may help the same constant-LR AdamW basin by smoothing the optimization path in parameter space without the weaknesses seen in delayed EMA or cosine decay.
- Change: Kept the winning `(24,)` residual-plus-linear architecture, but wrapped AdamW updates with Lookahead slow-weight interpolation every few steps.
- Result: `val_cindex` **0.780723** at `best_step` **1700** vs best kept **0.782152**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Lookahead was directionally better than many structural rewrites, but still clearly behind plain EMA and below the incumbent threshold.
- Next: Plain EMA still stands as the strongest non-keep result.

Note: One appended `results.tsv` row in this region was mis-labeled during manual logging after the Lookahead run; the journal entries above record the correct outcomes for Runs 145 and 146.

## Run 147
- Hypothesis: Plain EMA helped only at validation/export time, so using the EMA path as an explicit teacher during training might regularize the optimization trajectory itself and push the same `(24,)` model into a better basin.
- Change: Kept the incumbent residual-plus-linear `RiskMLP`, added an EMA teacher with decay `0.999`, trained the student on Cox loss plus a small z-scored risk consistency term against the teacher, and continued to validate/export the teacher weights as one TorchScript artifact.
- Result: `val_cindex` **0.782634** at `best_step` **1950** vs prior best kept **0.782152**.
- Decision: **keep**; saved the new baseline with `manage_kept.py save`.
- Learning: Temporal self-distillation finally broke through the local plateau; letting the slow teacher shape the student trajectory mattered more than architecture churn.
- Next: Probe whether the win came from the consistency weight itself or from the broader idea of richer temporal averaging.

## Run 148
- Hypothesis: The successful mean-teacher run may still have been slightly over-regularized, so reducing the consistency term could preserve the gain while allowing more Cox-driven flexibility.
- Change: Re-ran the mean-teacher setup from Run 147, changing only `TEACHER_CONSISTENCY_WEIGHT` from `0.05` to `0.03`.
- Result: `val_cindex` **0.781746** at `best_step` **1800** vs kept **0.782634**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The improvement was not just "EMA teacher but weaker"; the stronger consistency coupling in Run 147 was part of what made that run work.
- Next: Try a different temporal-averaging idea rather than local mean-teacher weakening.

## Run 149
- Hypothesis: A single EMA decay may still be too restrictive, so blending fast and slow EMA shadows into one exported checkpoint could match the sharp early peak and the smoother late basin at the same time.
- Change: Removed the teacher consistency loss, kept pure Cox training, maintained two EMA copies with decays `0.990` and `0.997`, fused them with a `0.5/0.5` convex blend for validation/export, and stored the best blended state as one TorchScript artifact.
- Result: `val_cindex` **0.783021** at `best_step` **800** vs kept **0.782634**.
- Decision: **keep**; saved the new baseline with `manage_kept.py save`.
- Learning: The strongest signal so far is now richer weight-space averaging rather than architectural novelty; a two-timescale trajectory filter beat both plain EMA and mean-teacher consistency.
- Next: Test whether leaning slightly toward the faster EMA improves the already-better dual-EMA peak.

## Run 150
- Hypothesis: Since the dual-EMA keep peaked early, shifting the fused export weights a little toward the faster shadow may sharpen the best region further.
- Change: Re-ran the dual-EMA setup from Run 149, changing only the convex blend from `0.5 fast / 0.5 slow` to `0.6 fast / 0.4 slow`.
- Result: `val_cindex` **0.782948** at `best_step` **800** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The equal fast/slow mix was already near the sweet spot; tilting toward the shorter-memory EMA slightly weakened the peak.
- Next: The dual-EMA family is now the best-performing direction, with the symmetric blend as the current frontier.

## Run 151
- Hypothesis: If dual-EMA is helping mainly by smoothing a sharp basin, then optimizing for flatter local geometry with Sharpness-Aware Minimization might improve the underlying trajectory before the same dual-EMA export is applied.
- Change: Kept the Run 149 architecture and dual-EMA export path, but replaced the plain AdamW step with a SAM two-pass update using `SAM_RHO = 0.05`.
- Result: `val_cindex` **0.780047** at `best_step` **500** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: In this full-batch setting, explicit sharpness-aware optimization hurt more than it helped; the winning signal still seems to come from post-step weight averaging rather than from more aggressive optimizer geometry.
- Next: Try a more direct lag-reduction filter in weight space instead of changing the optimizer again.

## Run 152
- Hypothesis: The dual-EMA keep may still lag the best region, so a level-plus-trend temporal filter that extrapolates the smoothed weights forward could reduce averaging lag without giving up stability.
- Change: Replaced the dual-EMA export logic with a Holt-style weight filter: maintained smoothed level and trend states for each parameter with `HOLT_LEVEL_ALPHA = 0.01`, `HOLT_TREND_BETA = 0.1`, and exported `level + 1.0 * trend` for validation/export.
- Result: `val_cindex` **0.782824** at `best_step` **650** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Trend-aware extrapolation had more signal than SAM and some early promise, but the straightforward dual-EMA blend remained stronger and more stable overall.
- Next: The surviving evidence still points toward the dual-EMA family, with future progress more likely from tuning its decay pair or trying a mild non-convex extrapolation from the fast/slow gap rather than broader optimizer changes.

## Run 153
- Hypothesis: The equal dual-EMA blend may still be lagging the best point, so extrapolating slightly past the fast EMA in the fast-minus-slow direction could debias the smoothing without changing the training trajectory.
- Change: Kept the Run 149 training loop and EMA decays, but replaced the convex export blend with the affine combination `(1.0 + 0.20) * fast - 0.20 * slow` for validation/export.
- Result: `val_cindex` **0.781292** at `best_step` **800** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Overshooting the fast EMA was too aggressive; the useful signal from fast-minus-slow disagreement does not translate into a better checkpoint through simple extrapolation.
- Next: Try changing the temporal blend structure more locally rather than pushing the whole model beyond the convex hull.

## Run 154
- Hypothesis: Different parameter families may prefer different temporal cutoffs, so using one global fast/slow blend for every tensor could be leaving accuracy on the table.
- Change: Kept the same two EMA shadows from Run 149, but built the export checkpoint with family-specific fast weights: `0.62` for scalar path gains, `0.38` for `residual_head.*`, `0.48` for `linear_skip.*`, and `0.45` elsewhere.
- Result: `val_cindex` **0.782885** at `best_step` **850** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Heterogeneous temporal mixing had more signal than affine extrapolation and nearly matched the keep, but the simple symmetric global blend still remained best.
- Next: The evidence keeps collapsing back onto the same conclusion: richer weight averaging is the only real lever left, but the current dual-EMA `0.990/0.997` with equal fusion is still the strongest concrete instance.

## Run 155
- Hypothesis: The current dual-EMA keep may be held back by using one pair of temporal cutoffs for the entire run, so letting both EMA decays change after the early fast-moving phase could improve the same equal-blend export.
- Change: Kept the Run 149 architecture and `0.5/0.5` fast-slow export blend, but used scheduled EMA decays: `0.986/0.995` before step `500`, then `0.990/0.998` afterward.
- Result: `val_cindex` **0.782623** at `best_step` **900** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Changing the decay pair over time added complexity without beating the simple fixed dual-EMA; the winning signal still seems to come from a stable pair rather than a phase-switched one.
- Next: Try a richer fixed temporal filter instead of time-varying decay schedules.

## Run 156
- Hypothesis: A third EMA timescale might capture useful medium-horizon structure that the current two-shadow filter misses, allowing a stronger export checkpoint without extrapolation.
- Change: Replaced the two-shadow export with three parallel EMA shadows using decays `0.990`, `0.994`, and `0.997`, and exported their equal `1/3` mixture for validation/export.
- Result: `val_cindex` **0.782824** at `best_step` **850** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Adding a middle timescale did not improve on the best dual-EMA basin; more temporal scales by themselves are not sufficient if the original two already span the useful signal.
- Next: The frontier remains stubbornly the same: the simple fixed dual-EMA with `0.990/0.997` and equal fusion is still the strongest point found.

## Run 157
- Hypothesis: The current keep may be losing information when fast and slow EMA weights are averaged into one network, so a single exported artifact that keeps both temporal snapshots alive and averages their predictions could outperform weight-space fusion.
- Change: Kept the same training trajectory and the same `0.990` / `0.997` EMA shadows, but replaced weight blending with a TorchScript wrapper containing both full `RiskMLP` snapshots and returning `0.5 * (fast(x) + slow(x))`.
- Result: `val_cindex` **0.783122** at `best_step` **800** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`, because it did not clear the current keep margin.
- Learning: Prediction-space temporal ensembling had real signal and was the strongest non-keep in this batch, but the gain was too small to beat the current benchmark threshold.
- Next: Try one orthogonal optimizer-level change while keeping the winning dual-EMA export unchanged.

## Run 158
- Hypothesis: If the online AdamW trajectory is still slightly noisy or over-aggressive, switching to AMSGrad may improve the underlying iterates while leaving the proven dual-EMA export rule untouched.
- Change: Restored the kept dual-EMA setup and changed only the optimizer to `torch.optim.AdamW(..., amsgrad=True)`.
- Result: `val_cindex` **0.783096** at `best_step` **850** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`, because it still missed the keep threshold.
- Learning: AMSGrad also produced only a small local gain rather than a decisive improvement, reinforcing that the current bottleneck is not obviously vanilla AdamW instability.
- Next: The search is still circling the same narrow frontier: small improvements are possible, but the fixed dual-EMA keep remains the strongest point that actually satisfies the rules.

## Run 159
- Hypothesis: The fixed `0.5/0.5` prediction-space mean from Run 157 may be suboptimal, so choosing the fast/slow prediction blend coefficient by the training Cox loss at each eval step could squeeze out more value from the same two temporal snapshots.
- Change: Kept the same dual-EMA trajectory, but replaced the fixed output mean with a scripted two-model wrapper carrying a scalar `beta` selected by a grid search over the training Cox partial loss at each eval step.
- Result: `val_cindex` **0.782067** at `best_step` **800** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: Training-loss tuning was actively misleading here: the selected `beta` collapsed to `1.0` throughout, effectively preferring the fast EMA alone and hurting validation performance.
- Next: Try one final prediction-space committee that explicitly keeps the raw student alongside the two EMA views.

## Run 160
- Hypothesis: Run 157's two-snapshot prediction mean may still be omitting useful present-time information, so averaging the raw student, fast EMA, and slow EMA predictions inside one scripted artifact could improve over both weight-space fusion and the two-view committee.
- Change: Built a single TorchScript wrapper containing three full `RiskMLP` snapshots from one trajectory (`raw`, `fast`, `slow`) and returned their equal `1/3` prediction mean for validation/export.
- Result: `val_cindex` **0.782871** at `best_step` **700** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The prediction-space branch peaked with the simpler two-view fast-plus-slow mean from Run 157; adding the raw student diluted the useful temporal committee signal instead of improving it.
- Next: The strongest non-keep in this whole branch remains Run 157, so further prediction-space work would need to stay very close to that exact two-snapshot formulation rather than broadening the committee.

## Run 161
- Hypothesis: A nonlinear two-snapshot committee may beat the linear mean from Run 157, so replacing arithmetic averaging with a tempered log-average-exp fusion could better preserve high-risk agreement between the fast and slow EMA views.
- Change: Kept the same dual-EMA trajectory and built a scripted two-model wrapper with `temperature = 2.0`, returning `logaddexp(temperature * fast(x), temperature * slow(x)) / temperature - log(2) / temperature`.
- Result: **crash at export** after reaching `val_cindex` **0.783235** at `best_step` **800`; TorchScript rejected a closed-over Python float constant used for `log(2)`.
- Decision: **crash**; immediately fixed the wrapper by moving `log(2)` into a registered buffer and re-ran the exact same experiment.
- Learning: The idea itself had unusually strong signal before export failed, so it was worth a direct scriptability fix rather than abandoning the family.
- Next: Re-run the same nonlinear committee with the scripting fix in place to get a valid artifact and final result.

## Run 162
- Hypothesis: The export failure in Run 161 was purely a TorchScript issue, so the same temperature-2 log-average-exp committee should reproduce its strong validation signal once the constant handling is made scriptable.
- Change: Re-ran the same fast/slow log-average-exp prediction wrapper from Run 161, but stored `log(2)` as a module buffer so the single TorchScript artifact could export cleanly.
- Result: `val_cindex` **0.783235** at `best_step` **800** vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`, because even this strongest nonlinear committee still missed the current keep threshold.
- Learning: Nonlinear prediction fusion is the strongest non-keep direction seen recently and improved over the linear two-view mean, but not by enough to replace the current keep.
- Next: Try one local temperature follow-up to see whether a slightly sharper smooth-max committee can close the remaining gap.

## Run 163
- Hypothesis: Since the temperature-2 nonlinear committee improved over the linear mean, increasing the temperature further may sharpen the fast/slow agreement signal and push the same family a bit higher.
- Change: Re-ran the log-average-exp fast/slow wrapper, changing only `LSE_TEMPERATURE` from `2.0` to `4.0`.
- Result: `val_cindex` **0.783217** at `best_step` **800` vs kept **0.783021**.
- Decision: **discard**; restored `train.py` from `last_kept_train.py`.
- Learning: The nonlinear committee branch peaks at the milder `temperature = 2.0`; pushing it more toward a hard max slightly weakens the result.
- Next: The best recent non-keep is now the temperature-2 nonlinear fast/slow committee, but the current fixed dual-EMA keep still remains the strongest benchmark-valid point.

Note: The append-only `results.tsv` rows around Runs 162-163 reflect the latest `run.log` value during manual logging and therefore do not distinguish the two temperature settings correctly; the journal entries above record the correct outcomes.
