"""Native 32-member TabM preprocessing and prediction."""
import gc
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, QuantileTransformer

VENDOR = Path(__file__).resolve().parents[1] / "calculator/vendor"
for name in ("rtdl_num_embeddings", "tabm"):
    spec = importlib.util.spec_from_file_location(name, VENDOR / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
import tabm
from rtdl_num_embeddings import PeriodicEmbeddings
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def release():
    gc.collect()
    torch.cuda.empty_cache()

def prepare(a, b, features, categories, pa, seed):
    cat = [c for c in features if c in categories]
    num = [c for c in features if c not in categories]
    scaler = StandardScaler() if pa['scaler'] == 'standard' else QuantileTransformer(n_quantiles=min(1000, len(a)), output_distribution='normal', random_state=seed)
    x = torch.as_tensor(scaler.fit_transform(a[num]).astype(np.float32), device='cuda')
    q = torch.as_tensor(scaler.transform(b[num]).astype(np.float32), device='cuda')
    ac, bc, cardinalities = (None, None, [])
    if cat:
        aa, bb = ([], [])
        for c in cat:
            levels = list(categories[c])
            mapping = {v: i for i, v in enumerate(levels)}
            aa.append(a[c].map(mapping).to_numpy(dtype=np.int64))
            bb.append(b[c].map(mapping).to_numpy(dtype=np.int64))
            cardinalities.append(len(levels))
        ac = torch.as_tensor(np.stack(aa, axis=1), device='cuda')
        bc = torch.as_tensor(np.stack(bb, axis=1), device='cuda')
    return (x, q, ac, bc, scaler, num, cat, cardinalities)

def make_model(pa, n_num, cardinalities):
    embedding = None
    if pa['embedding'] == 'periodic':
        embedding = PeriodicEmbeddings(n_num, d_embedding=16, n_frequencies=16, frequency_init_scale=0.01, lite=False)
    return tabm.TabM.make(n_num_features=n_num, cat_cardinalities=cardinalities or None, d_out=2, num_embeddings=embedding, arch_type='tabm', k=32, d_block=pa['width'], n_blocks=pa['depth'], dropout=pa['dropout']).to('cuda')

def predict(net, q, cat):
    net.eval()
    with torch.inference_mode():
        return np.concatenate([net(q[i:i + 256], cat[i:i + 256] if cat is not None else None).softmax(-1).mean(1)[:, 1].cpu().numpy() for i in range(0, len(q), 256)]).astype(float)
