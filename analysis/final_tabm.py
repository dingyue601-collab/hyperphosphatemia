"""Fixed-epoch full and two-variable TabM fits, without test outcomes."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import fit_engine as engine

ROOT = Path(os.environ["HYPERPHOS_RUN"])
OUT = ROOT / "results/analysis"
FEATURES = ["age", "creatinine"]


def main():
    mode = sys.argv[1]
    nonipw = mode == "nonipw"
    two = mode != "full"
    job_path = OUT / ("two_predictor/Training_job.json" if two else "Training_job.json")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    configuration = OUT / ("nonIPW_sensitivity/TabM/Selected_configuration.json" if nonipw else "TabM/Selected_configuration.json")
    engine.CONFIG = json.loads(configuration.read_text(encoding="utf-8"))
    reference = pd.read_csv(OUT / "Training_reference.csv", float_precision="round_trip")
    blocks = sorted(job["blocks"], key=lambda b: (b["fold"] == 0, b["fold"]))
    folder = OUT / ("two_predictor/NonIPW_TabM2" if nonipw else "two_predictor" if two else "final/TabM7")
    folder.mkdir(parents=True, exist_ok=True)
    oof = np.full((len(reference), 30), np.nan)
    for block in blocks:
        fold = block["fold"]
        if not two and fold:
            continue
        features = FEATURES if two else list(block["features"])
        levels = block.get("categorical_levels", {})
        levels = levels if isinstance(levels, dict) else {}
        levels = {k: v for k, v in levels.items() if k in features}
        ii = np.asarray(block["train_indices"]) - 1
        qi = np.asarray(block["query_indices"] or [], dtype=int) - 1
        assert not set(reference.patient.iloc[ii]).intersection(reference.patient.iloc[qi])
        for mi, entry in enumerate(block["entries"], 1):
            a = pd.read_csv(entry["train"], float_precision="round_trip")
            if nonipw:
                a["sample_weight"] = 1.0
            if fold:
                q = pd.read_csv(entry["query"], float_precision="round_trip")[features]
                prefix = None
            else:
                test = pd.read_csv(ROOT / f"results/predictors/test_inputs/test_m{mi:02d}.csv", float_precision="round_trip")[features]
                q = pd.concat([a[features], test], ignore_index=True)
                prefix = folder / f"TabM_m{mi:02d}"
            path = folder / f"fold{fold:02d}_m{mi:02d}.npz"
            if path.exists():
                with np.load(path) as record:
                    assert record["features"].tolist() == features
                    assert np.array_equal(record["query_index"], qi)
                    probability = record["probability"]
            else:
                state, probability = engine.fitted_state(a, q, features, levels, 123 + 100 * fold + mi, prefix)
                del state
                engine.backend.release()
                engine.checkpoint(path, probability=probability, features=np.asarray(features), query_index=qi)
            if fold:
                assert probability.shape == (len(qi),)
                oof[qi, mi - 1] = probability
            else:
                assert probability.shape == (6601,)
    columns = [f"m{m:02d}" for m in range(1, 31)]
    if two:
        assert np.isfinite(oof).all()
        pd.DataFrame(oof, columns=columns).to_csv(folder / "TabM2_OOF_probabilities.csv", index=False)
    final = np.column_stack([np.load(folder / f"fold00_m{m:02d}.npz")["probability"] for m in range(1, 31)])
    for name, values in (("Training", final[:4575]), ("Test", final[4575:])):
        pd.DataFrame(values, columns=columns).to_csv(folder / f"{name}_probabilities.csv", index=False)
    engine.save(folder / "Complete.json", dict(completed=True, features=features,
        imputations=30, epochs=engine.CONFIG["fixed_epochs"], training="Unweighted" if nonipw else "IPW"))


if __name__ == "__main__":
    main()
