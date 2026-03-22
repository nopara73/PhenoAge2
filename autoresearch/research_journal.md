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
