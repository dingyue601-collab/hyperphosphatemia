"""Fixed-parameter subset refitting within the training partition."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import fit_engine as engine

ROOT = Path(os.environ["HYPERPHOS_RUN"])
OUT = ROOT / "results/analysis"
DEST = OUT / "subsets"
SEARCH_MI = [1, 7, 13, 19, 25]


def main():
    job = json.loads((OUT / "Training_job.json").read_text(encoding="utf-8"))
    engine.CONFIG = json.loads((OUT / "TabM/Selected_configuration.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(OUT / "Training_reference.csv", float_precision="round_trip")
    features = list(job["features"])
    blocks = [b for b in job["blocks"] if b["fold"] > 0]
    assert len(features) == 7 and len(blocks) == 5
    DEST.mkdir(exist_ok=True)
    loaded = {}
    candidates = []
    for mask in range(1, 128):
        chosen = [v for i, v in enumerate(features) if mask & (1 << i)]
        candidates.append(dict(Mask=mask, Variables=len(chosen), Features=",".join(chosen),
            Fold_variable_counts="/".join(str(sum(v in chosen for v in b["features"])) for b in blocks)))
    pd.DataFrame(candidates).to_csv(DEST / "Candidates.csv", index=False)

    def fit_candidate(mask, imputations, stage):
        chosen = [v for i, v in enumerate(features) if mask & (1 << i)]
        matrix = np.full((len(reference), len(imputations)), np.nan)
        for block in blocks:
            fold = block["fold"]
            ii = np.asarray(block["train_indices"]) - 1
            qi = np.asarray(block["query_indices"]) - 1
            assert not set(reference.patient.iloc[ii]).intersection(reference.patient.iloc[qi])
            local = [v for v in block["features"] if v in chosen]
            code = sum(1 << features.index(v) for v in local)
            folder = DEST / "cache" / f"fold{fold:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            for column, mi in enumerate(imputations):
                path = folder / f"mask{code:03d}_m{mi:02d}.npz"
                if path.exists():
                    with np.load(path) as cached:
                        assert cached["features"].tolist() == local
                        assert np.array_equal(cached["query_index"], qi)
                        probability = cached["probability"]
                else:
                    key = (fold, mi)
                    if key not in loaded:
                        entry = block["entries"][mi - 1]
                        loaded[key] = (pd.read_csv(entry["train"], float_precision="round_trip"),
                                       pd.read_csv(entry["query"], float_precision="round_trip"))
                    a, q = loaded[key]
                    assert np.array_equal(a.label, reference.label.iloc[ii])
                    levels = block["categorical_levels"] if isinstance(block["categorical_levels"], dict) else {}
                    levels = {key: value for key, value in levels.items() if key in local}
                    if not local:
                        probability = np.full(len(q), np.average(a.label, weights=a.sample_weight))
                    elif set(local) == set(levels):
                        probability = engine.fit_categorical(a, q, local, levels, 123 + 100 * fold + mi)
                    else:
                        state, probability = engine.fitted_state(a, q, local, levels, 123 + 100 * fold + mi)
                        del state
                    engine.backend.release()
                    engine.checkpoint(path, probability=probability, features=np.asarray(local), query_index=qi)
                matrix[qi, column] = probability
        assert np.isfinite(matrix).all() and ((matrix >= 0) & (matrix <= 1)).all()
        folder = DEST / stage
        folder.mkdir(exist_ok=True)
        pd.DataFrame(matrix, columns=[f"m{m:02d}" for m in imputations]).to_csv(folder / f"subset{mask:03d}.csv", index=False)

    for mask in range(1, 128):
        fit_candidate(mask, SEARCH_MI, "search")
    scorer = Path(__file__).with_name("score_subsets.R")
    subprocess.run([os.environ["HYPERPHOS_R"], str(scorer), "search"], check=True)
    masks = json.loads((DEST / "Confirmation_queue.json").read_text())["masks"]
    for mask in masks:
        fit_candidate(mask, list(range(1, 31)), "confirmation")
    subprocess.run([os.environ["HYPERPHOS_R"], str(scorer), "confirmation"], check=True)


if __name__ == "__main__":
    main()
