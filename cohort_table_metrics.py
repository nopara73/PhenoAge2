from autoresearch.prepare import load_joined_rows, get_rows_for_split, survival_arrays, harrell_c_index, compute_original_phenoage_scores, score_scripted_model, DEFAULT_CANDIDATE_MODEL_PATH

rows = load_joined_rows()
groups = [
    ("Full cohort", rows),
    ("Development", get_rows_for_split(rows, "development")),
    ("Test", get_rows_for_split(rows, "test")),
]

for name, grp in groups:
    times, events = survival_arrays(grp)
    pa2 = harrell_c_index(times, events, score_scripted_model(DEFAULT_CANDIDATE_MODEL_PATH, grp, device="cpu"))
    ph = harrell_c_index(times, events, compute_original_phenoage_scores(grp))
    print(f"{name}\t{len(grp)}\t{int(events.sum())}\t{pa2:.6f}\t{ph:.6f}\t{pa2-ph:+.6f}", flush=True)
