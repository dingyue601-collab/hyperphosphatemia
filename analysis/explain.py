"""Exact empirical Shapley values of the frozen two-variable ensemble."""
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import fit_engine as engine

ROOT = Path(os.environ["HYPERPHOS_RUN"])
BASE = ROOT / "results/analysis"
DEST = BASE / "two_predictor"
PREDICTORS = ROOT / "results/predictors"
FEATURES = ["age", "creatinine"]


def read(path):
    return pd.read_csv(path, float_precision='round_trip')

def calibrate(p, coefficients):
    a, b = coefficients
    if a == 0 and b == 1:
        return p
    lp = np.log(np.clip(p, 1e-08, 1 - 1e-08) / (1 - np.clip(p, 1e-08, 1 - 1e-08)))
    return 1 / (1 + np.exp(-(a + b * lp)))

def main():
    meta = json.loads((DEST / 'Development_complete.json').read_text())
    assert meta['completed'] and meta['fixed_features_all_folds'] == FEATURES
    assert torch.cuda.is_available()
    folder = DEST / 'SHAP'
    folder.mkdir(exist_ok=True)
    donors = np.random.default_rng(123).choice(4575, 32, replace=False)
    old_train = read(DEST / 'Training_probabilities.csv').to_numpy()
    old_test = read(DEST / 'Test_probabilities.csv').to_numpy()
    pp_train, pp_test = ([], [])
    singleton_age, singleton_cr, background = ([], [], [])
    display = np.zeros((2026, 2))
    replays = []
    for mi in range(1, 31):
        started = time.monotonic()
        a = read(DEST / f'inputs/fold00_m{mi:02d}_train.csv')
        q = read(PREDICTORS / f'test_inputs/test_m{mi:02d}.csv')[FEATURES]
        assert list(a) == FEATURES + ['label', 'sample_weight'] and len(q) == 2026
        display += q.to_numpy() / 30
        path = folder / f'm{mi:02d}.npz'
        if not path.exists():
            state = engine.restore(DEST / f'TabM_m{mi:02d}', a, q)
            assert state['numeric'] == FEATURES and (not state['categorical'])
            p_train = engine.backend.predict(state['model'], state['fit_numeric'], None)
            p_test = engine.backend.predict(state['model'], state['query_numeric'], None)
            xn = state['query_numeric']
            bn = state['fit_numeric'][donors]
            p_bg = engine.backend.predict(state['model'], bn, None)
            coalitions = []
            for feature in range(2):
                values = np.empty((2026, 32))
                for start in range(0, 2026, 64):
                    stop = min(start + 64, 2026)
                    x = bn.repeat(stop - start, 1).clone()
                    x[:, feature] = xn[start:stop, feature].repeat_interleave(32)
                    values[start:stop] = engine.backend.predict(state['model'], x, None).reshape(stop - start, 32)
                coalitions.append(values)
            replay = max(float(np.max(np.abs(p_train - old_train[:, mi - 1]))), float(np.max(np.abs(p_test - old_test[:, mi - 1]))))
            assert replay <= 1e-06, f'Final checkpoint replay mismatch {replay}'
            engine.checkpoint(path, training_probability=p_train, probability=p_test, background=p_bg, age_only=coalitions[0], creatinine_only=coalitions[1], donors=donors, replay_difference=replay)
            del state, xn, bn, x
            engine.backend.release()
        with np.load(path) as z:
            assert np.array_equal(z['donors'], donors)
            pp_train.append(z['training_probability'])
            pp_test.append(z['probability'])
            singleton_age.append(z['age_only'])
            singleton_cr.append(z['creatinine_only'])
            background.append(z['background'])
            replays.append(float(z['replay_difference']))
        engine.save(DEST / 'Inference_run_status.json', dict(stage='Final_two_predictor_SHAP', imputation=mi, pid=os.getpid(), updated=time.strftime('%F %T'), model_refitted=False))
        print(time.strftime('%F %T'), 'FROZEN_REPLAY_AND_SHAP', mi, 'SECONDS', round(time.monotonic() - started, 3), flush=True)
    coeff = meta['models']['TabM2']['calibration_coefficients']
    raw = np.column_stack(pp_test).mean(1)
    value_full = calibrate(raw, coeff)
    value_empty = float(calibrate(np.mean(background, axis=0), coeff).mean())
    value_age = calibrate(np.mean(singleton_age, axis=0), coeff).mean(1)
    value_cr = calibrate(np.mean(singleton_cr, axis=0), coeff).mean(1)
    phi = np.column_stack((0.5 * (value_age - value_empty + (value_full - value_cr)), 0.5 * (value_cr - value_empty + (value_full - value_age))))
    residual = float(np.max(np.abs(value_empty + phi.sum(1) - value_full)))
    assert residual < 2e-06
    for name, values in [('Training', pp_train), ('Test', pp_test)]:
        p = np.column_stack(values)
        assert p.shape == (4575 if name == 'Training' else 2026, 30)
        pd.DataFrame(p, columns=[f'm{i:02d}' for i in range(1, 31)]).to_csv(DEST / f'TabM2_{name}_probabilities.csv', index=False)
    frame = pd.DataFrame(phi, columns=FEATURES)
    frame.insert(0, 'row', np.arange(2026) + 1)
    frame['baseline'] = value_empty
    frame['probability'] = value_full
    frame.to_csv(folder / 'SHAP_values.csv', index=False)
    pd.DataFrame(display, columns=FEATURES).to_csv(folder / 'SHAP_display_features.csv', index=False)
    w = read(BASE / 'Test_IPW_weights.csv').to_numpy()
    importance = np.average(np.abs(phi), weights=(w / w.sum(0)).mean(1), axis=0)
    pd.DataFrame(dict(Feature=FEATURES, Mean_absolute_SHAP_IPW=importance, Attribution_fraction_IPW=importance / importance.sum(), Mean_absolute_SHAP_unweighted=np.abs(phi).mean(0))).to_csv(folder / 'SHAP_importance.csv', index=False)
    engine.save(folder / 'Complete.json', dict(completed=True, records=2026, features=FEATURES, imputations=30, members=32, background_rows=32, background_seed=123, coalitions=4, baseline=value_empty, calibration=meta['models']['TabM2']['calibration_coefficients'], maximum_additivity_error=residual, maximum_replay_error=max(replays), explanation='Exact empirical Shapley of final member-then-MI averaged probability, conditional on completed MI inputs; not causal attribution'))
    engine.save(DEST / 'Inference_complete.json', dict(completed=True, model_refitted=False, training_rows=4575, test_rows=2026, imputations=30, maximum_replay_error=max(replays), final_SHAP_completed=True))
    print('FIXED_TWO_PREDICTOR_EXACT_SHAP_COMPLETE', flush=True)

if __name__ == "__main__":
    main()
