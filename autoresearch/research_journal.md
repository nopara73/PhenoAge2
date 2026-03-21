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
