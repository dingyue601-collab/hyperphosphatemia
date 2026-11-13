"""Fixed-epoch TabM fitting and checkpoint replay."""
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import backend

CONFIG = None


def note(*items):
    print(time.strftime('%F %T'), *items, flush=True)

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)

def checkpoint(path, **values):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(temp, path)

def fitted_state(a, q, features, levels, seed, prefix=None):
    pa = CONFIG['parameters']
    assert pa['positive_weight'] == 1 and CONFIG['fixed_epochs'] > 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, query, xc, qc, scaler, numeric, categorical, cardinalities = backend.prepare(a, q, features, levels, pa, seed)
    net = backend.make_model(pa, len(numeric), cardinalities)
    assert net.backbone.k == 32 and {p.dtype for p in net.parameters()} == {torch.float32}
    assert {p.device.type for p in net.parameters()} == {'cuda'}
    y = torch.as_tensor(a.label.to_numpy(dtype=np.int64, copy=True), device='cuda')
    w = a.sample_weight.to_numpy(dtype=float, copy=True)
    assert np.isfinite(w).all() and (w > 0).all()
    weights = torch.as_tensor((w / w.mean()).astype(np.float32), device='cuda')
    optimizer = torch.optim.AdamW(net.parameters(), lr=pa['learning_rate'], weight_decay=pa['weight_decay'], foreach=False)
    rng = np.random.default_rng(seed)
    for _ in range(CONFIG['fixed_epochs']):
        net.train()
        order = rng.permutation(len(a))
        for start in range(0, len(a), pa['batch_size']):
            ii = torch.as_tensor(order[start:start + pa['batch_size']], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            logits = net(x[ii], xc[ii] if xc is not None else None)
            loss = F.cross_entropy(logits.flatten(0, 1), y[ii].repeat_interleave(32), reduction='none')
            loss = (loss * weights[ii].repeat_interleave(32)).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite weighted TabM loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
    p = backend.predict(net, query, qc)
    state = dict(model=net, numeric=numeric, categorical=categorical, levels=levels, scaler=scaler, fit_numeric=x, fit_categorical=xc, query_numeric=query, query_categorical=qc)
    if prefix is not None:
        record = dict(state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, parameters=pa, features=features, numeric=numeric, categorical=categorical, cardinalities=cardinalities, levels=levels, epochs=CONFIG['fixed_epochs'], training='Record-weighted training', classes=[0, 1])
        torch.save(record, str(prefix) + '.pt')
        joblib.dump(scaler, str(prefix) + '.preprocessing.joblib')
        restored = restore(prefix, a, q)
        delta = float(np.max(np.abs(backend.predict(restored['model'], restored['query_numeric'], restored['query_categorical']) - p)))
        assert delta <= 1e-06, f'Checkpoint reload mismatch: {delta}'
        del restored, record
    del optimizer, y, weights, logits, loss
    return (state, p)

def restore(prefix, a, q):
    record = torch.load(str(prefix) + '.pt', map_location='cpu', weights_only=False)
    scaler = joblib.load(str(prefix) + '.preprocessing.joblib')
    numeric, categorical, levels = (record['numeric'], record['categorical'], record['levels'])
    net = backend.make_model(record['parameters'], len(numeric), record['cardinalities'])
    net.load_state_dict(record['state'], strict=True)

    def encode(frame):
        x = torch.as_tensor(scaler.transform(frame[numeric]).astype(np.float32), device='cuda')
        c = torch.as_tensor(np.column_stack([frame[v].map({x: i for i, x in enumerate(levels[v])}).to_numpy(dtype=np.int64) for v in categorical]), device='cuda') if categorical else None
        return (x, c)
    x, xc = encode(a)
    query, qc = encode(q)
    return dict(model=net, numeric=numeric, categorical=categorical, levels=levels, scaler=scaler, fit_numeric=x, fit_categorical=xc, query_numeric=query, query_categorical=qc)

def fit_categorical(a, q, features, levels, seed):
    """Native TabM's supported zero-numeric case; identical weighted loss/epochs."""
    pa = CONFIG['parameters']
    torch.manual_seed(seed)
    np.random.seed(seed)

    def encode(frame):
        return torch.as_tensor(np.column_stack([frame[v].map({x: k for k, x in enumerate(levels[v])}).to_numpy(dtype=np.int64) for v in features]), device='cuda')
    x, query = (encode(a), encode(q))
    net = backend.tabm.TabM.make(n_num_features=0, cat_cardinalities=[len(levels[v]) for v in features], d_out=2, num_embeddings=None, arch_type='tabm', k=32, d_block=pa['width'], n_blocks=pa['depth'], dropout=pa['dropout']).to('cuda')
    assert net.backbone.k == 32 and {p.dtype for p in net.parameters()} == {torch.float32}
    y = torch.as_tensor(a.label.to_numpy(dtype=np.int64, copy=True), device='cuda')
    w = a.sample_weight.to_numpy(dtype=float, copy=True)
    weights = torch.as_tensor((w / w.mean()).astype(np.float32), device='cuda')
    opt = torch.optim.AdamW(net.parameters(), lr=pa['learning_rate'], weight_decay=pa['weight_decay'], foreach=False)
    rng = np.random.default_rng(seed)
    for _ in range(14):
        net.train()
        order = rng.permutation(len(a))
        for start in range(0, len(a), pa['batch_size']):
            ix = torch.as_tensor(order[start:start + pa['batch_size']], device='cuda')
            opt.zero_grad(set_to_none=True)
            logits = net(None, x[ix])
            loss = (F.cross_entropy(logits.flatten(0, 1), y[ix].repeat_interleave(32), reduction='none') * weights[ix].repeat_interleave(32)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
    net.eval()
    with torch.inference_mode():
        p = np.concatenate([net(None, query[i:i + 256]).softmax(-1).mean(1)[:, 1].cpu().numpy() for i in range(0, len(q), 256)]).astype(float)
    return p
