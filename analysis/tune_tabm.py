"""Weighted TabM tuning on patient-separated development folds."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

OUT = Path(os.environ["HYPERPHOS_OUTPUT"])
SEED, BUDGET = 123, 30
TUNE_MI = [1, 7, 13, 19, 25]
BACKEND = None

def backend():
    global BACKEND
    if BACKEND is None:
        import backend as module
        BACKEND = module
    return BACKEND


def note(*items):
    print(time.strftime('%F %T'), *items, flush=True)

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)

def checkpoint(path, probability):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, probability=probability)
    os.replace(temp, path)

def selection_loss(logits, y, normalized_weights):
    members = logits.shape[1]
    losses = F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(members), reduction='none')
    return (losses * normalized_weights.repeat_interleave(members)).mean()

def fit_tabm(frame, pa, seed, epochs=None):
    engine = backend()
    a, b, y = (frame['train'], frame['query'], frame['y'])
    sw = frame['weight']
    assert torch.cuda.is_available() and pa['positive_weight'] == 1
    assert np.isfinite(sw).all() and (sw > 0).all() and (len(sw) == len(a))
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, q, ac, qc, scaler, numeric, categorical, cardinalities = engine.prepare(a, b, frame['features'], frame['cats'], pa, seed)
    net = engine.make_model(pa, len(numeric), cardinalities)
    assert net.backbone.k == 32 and {p.dtype for p in net.parameters()} == {torch.float32}
    assert {p.device.type for p in net.parameters()} == {'cuda'}
    yt = torch.as_tensor(np.array(y, dtype=np.int64, copy=True), device='cuda')
    normalized = torch.as_tensor((sw / sw.mean()).astype(np.float32), device='cuda')
    optimizer = torch.optim.AdamW(net.parameters(), lr=pa['learning_rate'], weight_decay=pa['weight_decay'], foreach=False)
    rng = np.random.default_rng(seed)
    best, bestloss, stale, chosen, state = (-np.inf, np.inf, 0, 0, None)
    started = time.monotonic()
    for epoch in range(1, (epochs if epochs is not None else 200) + 1):
        net.train()
        order = rng.permutation(len(y))
        for start in range(0, len(y), pa['batch_size']):
            ii = torch.as_tensor(order[start:start + pa['batch_size']], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            logits = net(x[ii], ac[ii] if ac is not None else None)
            loss = selection_loss(logits, yt[ii], normalized[ii])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite weighted TabM loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
        if epochs is None:
            p = engine.predict(net, q, qc)
            score = roc_auc_score(frame['labels'], p, sample_weight=frame['query_weight'])
            ll = log_loss(frame['labels'], p, sample_weight=frame['query_weight'], labels=[0, 1])
            if score > best + 0.0002 or (abs(score - best) < 1e-12 and ll < bestloss):
                best, bestloss, stale, chosen = (score, ll, 0, epoch)
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
            if stale >= 20:
                break
    if epochs is None:
        net.load_state_dict(state)
    else:
        chosen = epochs
    p = engine.predict(net, q, qc)
    metadata = dict(selected_epoch=int(chosen), evaluated_epochs=epoch, seconds=time.monotonic() - started, members=32, precision='float32', loss='Member-wise cross entropy with fitting-partition mean-normalised selection IPW', early_stopping='Fitting-fold validation IPW AUC and weighted log loss' if epochs is None else 'Fixed training-selected epoch')
    del net, optimizer, x, q, ac, qc, yt, normalized, state, logits, loss
    engine.release()
    return (p, metadata)

def load(block, mi):
    entry = block['entries'][mi - 1]
    a, b = (pd.read_csv(entry['train']), pd.read_csv(entry['query']))
    features = block['features']
    features = [features] if isinstance(features, str) else features
    cats = block['categorical_levels'] if isinstance(block['categorical_levels'], dict) else {}
    assert not {'label', 'y', 'diagnostic_y', 'sample_weight'}.intersection(b.columns)
    y = a.pop('label').to_numpy(dtype=np.int64, copy=True)
    sw = a.pop('sample_weight').to_numpy(dtype=float, copy=True)
    labels = pd.read_csv(block['query_labels']).label.to_numpy(dtype=np.int64, copy=True)
    wq = pd.read_csv(block['query_weights']).iloc[:, mi - 1].to_numpy(dtype=float, copy=True)
    assert np.isfinite(a[features].to_numpy()).all() and np.isfinite(b[features].to_numpy()).all()
    assert np.isfinite(sw).all() and (sw > 0).all() and np.isfinite(wq).all() and (wq > 0).all()
    return dict(train=a[features], query=b[features], y=y, labels=labels, weight=sw, query_weight=wq, features=features, cats=cats)

def parameters(t):
    return dict(width=t.suggest_categorical('width', [128, 256, 512]), depth=t.suggest_int('depth', 2, 4), dropout=t.suggest_float('dropout', 0, 0.35), learning_rate=t.suggest_float('learning_rate', 0.0002, 0.004, log=True), weight_decay=t.suggest_float('weight_decay', 1e-05, 0.01, log=True), batch_size=t.suggest_categorical('batch_size', [128, 256]), positive_weight=1.0, scaler=t.suggest_categorical('scaler', ['quantile', 'standard']), embedding=t.suggest_categorical('embedding', ['none', 'periodic']))

def score_tabm(number, pa, imputations, frames, blocks, epochs=None):
    stage = 'search' if epochs is None else 'confirm'
    folder = OUT / 'TabM' / f'trial{number:03d}' / stage
    folder.mkdir(parents=True, exist_ok=True)
    scores, briers, selected_epochs = ([], [], [])
    for block in blocks:
        fold = block['fold']
        probabilities, weights = ([], [])
        for mi in imputations:
            target = folder / f'fold{fold:02d}_m{mi:02d}.npz'
            metadata = target.with_suffix('.json')
            frame = frames[fold, mi]
            if target.exists() and metadata.exists():
                with np.load(target) as z:
                    p = z['probability']
                meta = json.loads(metadata.read_text(encoding='utf-8'))
            else:
                p, meta = fit_tabm(frame, pa, SEED + 100 * fold + mi, epochs)
                checkpoint(target, p)
                save(metadata, meta)
                note(stage.upper(), 'TabM', number, 'FOLD', fold, 'MI', mi, 'EPOCH', meta['selected_epoch'])
            assert p.shape == frame['labels'].shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            probabilities.append(p)
            weights.append(frame['query_weight'])
            selected_epochs.append(meta['selected_epoch'])
        labels = frames[fold, imputations[0]]['labels']
        p = np.mean(probabilities, axis=0)
        scores.append(float(np.mean([roc_auc_score(labels, p, sample_weight=w) for w in weights])))
        briers.append(float(np.mean([np.average((labels - p) ** 2, weights=w) for w in weights])))
    return dict(trial=number, parameters=pa, mean_auc=float(np.mean(scores)), mean_brier=float(np.mean(briers)), fold_auc=scores, epochs=selected_epochs, fixed_epochs=epochs, imputations=imputations)

def tabm(blocks, reference):
    folder = OUT / 'TabM'
    folder.mkdir(exist_ok=True)
    frames = {(b['fold'], m): load(b, m) for b in blocks for m in range(1, 31)}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name='TabM', direction='maximize', storage='sqlite:///' + str(folder / 'Optuna.sqlite3'), load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10))
    if not study.trials:
        study.enqueue_trial(dict(width=512, depth=3, dropout=0.1, learning_rate=0.002, weight_decay=0.0003, batch_size=128, scaler='quantile', embedding='periodic'))
    save(folder / 'Search_specification.json', dict(max_trials=BUDGET, tuning_imputations=TUNE_MI, confirmation_imputations=list(range(1, 31)), optimizer='Optuna TPE', members=32, precision='float32', objective='Mean fold IPW AUC after MI probability averaging', feature_selection='Fitting-fold weighted LASSO', early_stopping='Fitting-fold IPW AUC; 200 epochs, patience 20; fixed median epoch for confirmation', test_labels_read=False, device=torch.cuda.get_device_name(0)))
    while True:
        running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        if len(study.trials) >= BUDGET and (not running):
            break
        trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
        result = score_tabm(trial.number, parameters(trial), TUNE_MI, frames, blocks)
        save(folder / f'trial{trial.number:03d}_complete.json', result)
        study.tell(trial, result['mean_auc'])
        note('TRIAL_COMPLETE', 'TabM', trial.number, 'IPW_AUC', result['mean_auc'], 'BRIER', result['mean_brier'])
    results = [json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('trial*_complete.json')]
    results.sort(key=lambda x: (-x['mean_auc'], x['mean_brier']))
    finalists = []
    for result in results[:2]:
        epochs = max(1, int(np.rint(np.median(result['epochs']))))
        confirmed = score_tabm(result['trial'], result['parameters'], list(range(1, 31)), frames, blocks, epochs)
        save(folder / f"trial{result['trial']:03d}_confirmed.json", confirmed)
        finalists.append(confirmed)
    best = min(finalists, key=lambda x: (-x['mean_auc'], x['mean_brier']))
    best['completed_trials'] = len(results)
    save(folder / 'Selected_configuration.json', best)
    probabilities = np.full((len(reference), 30), np.nan)
    for b in blocks:
        indices = np.asarray(b['query_indices']) - 1
        for m in range(1, 31):
            with np.load(folder / f"trial{best['trial']:03d}/confirm/fold{b['fold']:02d}_m{m:02d}.npz") as z:
                probabilities[indices, m - 1] = z['probability']
    finish('TabM', probabilities, blocks, best)

def finish(family, probabilities, blocks, configuration=None):
    assert probabilities.shape == (4575, 30) and np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    folder = OUT / family
    pd.DataFrame(probabilities, columns=[f'm{m:02d}' for m in range(1, 31)]).to_csv(folder / 'OOF_probabilities.csv', index=False)
    save(folder / 'Complete.json', dict(completed=True, family=family, rows=len(probabilities), folds=5, imputations=30, features_by_fold={str(b['fold']): b['features'] for b in blocks}, configuration=configuration, learning='Record-weighted training', evaluation='Stabilized testing IPW', test_labels_read=False, precision='float32', device='cuda'))
    note('MODEL_COMPLETE', family)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("The recorded TabM tuning requires a compatible CUDA device.")
    job = json.loads((OUT / "Training_job.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(OUT / "Training_reference.csv", float_precision="round_trip")
    blocks = [b for b in job["blocks"] if b["fold"] > 0]
    assert len(blocks) == 5 and reference.groupby("patient").fold.nunique().max() == 1
    tabm(blocks, reference)
