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
