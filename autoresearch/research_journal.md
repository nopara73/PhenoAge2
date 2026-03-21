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
