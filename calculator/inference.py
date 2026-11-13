"""Two-variable TabM inference using the supplied model artifacts.

Prediction averages member probabilities, then imputation-specific models.
Checkpoint and preprocessing files must come from a trusted source.
"""
from pathlib import Path
import importlib.util
import json
import sys
import joblib
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
for name in ("rtdl_num_embeddings", "tabm"):
    spec = importlib.util.spec_from_file_location(name, HERE / "vendor" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
import tabm
from rtdl_num_embeddings import PeriodicEmbeddings

FEATURES = ("age", "creatinine")


def clinical_input(age, creatinine, unit="umol/L"):
    """Validate adult numeric inputs and convert creatinine to micromol/L."""
    if any(v is None or isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v)
           for v in (age, creatinine)):
        raise ValueError("Both age and creatinine are required.")
    values = np.asarray([age, creatinine], dtype=float)
    if not np.isfinite(values).all() or not 18 <= values[0] <= 115 or values[1] <= 0:
        raise ValueError("Enter adult age 18-115 years and positive finite creatinine.")
    if unit not in ("umol/L", "mg/dL"):
        raise ValueError("Unsupported creatinine unit.")
    values[1] *= 88.4 if unit == "mg/dL" else 1
    if not np.isfinite(values).all():
        raise ValueError("Creatinine overflow after unit conversion.")
    return pd.DataFrame([values], columns=FEATURES)


def make_model(record, device):
    """Construct the recorded two-input architecture without fitting."""
    pa = record["parameters"]
    if record["features"] != list(FEATURES) or record["numeric"] != list(FEATURES) or record["categorical"]:
        raise ValueError("Checkpoint predictor order mismatch.")
    model = tabm.TabM.make(n_num_features=2, cat_cardinalities=None, d_out=2,
        num_embeddings=PeriodicEmbeddings(2, d_embedding=16, n_frequencies=16,
            frequency_init_scale=.01, lite=False), arch_type="tabm", k=32,
        d_block=pa["width"], n_blocks=pa["depth"], dropout=pa["dropout"])
    return model.to(device=device, dtype=torch.float32).eval()


class TwoPredictorModels:
    """Member mean within each TabM, MI mean, then frozen calibration.

    Call predict_mi for replay against MI-specific offline input matrices.
    """

    def __init__(self, directory=None, device="cpu"):
        self.directory = Path(directory or HERE / "models")
        self.spec = json.loads((self.directory / "model_specification.json").read_text(encoding="utf-8"))
        if self.spec["fixed_features_all_folds"] != list(FEATURES):
            raise ValueError("Incorrect predictor specification.")
        if self.spec["models"]["TabM2"]["calibration_coefficients"] != [0, 1]:
            raise ValueError("This release requires the recorded identity probability mapping.")
        self.device = torch.device(device)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.models = []
        for mi in range(1, 31):
            prefix = self.directory / f"TabM_m{mi:02d}"
            record = torch.load(str(prefix) + ".pt", map_location="cpu", weights_only=True)
            if record["classes"] != [0, 1] or record["epochs"] != 14:
                raise ValueError("Checkpoint outcome/epoch mismatch.")
            model = make_model(record, self.device)
            model.load_state_dict(record["state"], strict=True)
            self.models.append((model,
                                joblib.load(str(prefix) + ".preprocessing.joblib")))

    def predict_mi(self, frame, mi):
        """One MI-specific model; accepts observed or offline completed inputs."""
        if not 1 <= mi <= 30 or tuple(frame.columns) != FEATURES:
            raise ValueError("Provide exactly age, creatinine in that order.")
        raw = frame.to_numpy()
        if np.iscomplexobj(raw) or any(isinstance(v, (bool, np.bool_)) for v in raw.flat):
            raise ValueError("Inputs must be real numbers, not Boolean or complex values.")
        values = frame.to_numpy(dtype=float)
        if (not len(values) or not np.isfinite(values).all() or
                (values[:, 0] < 18).any() or (values[:, 0] > 115).any() or
                (values[:, 1] <= 0).any()):
            raise ValueError("Provide complete adult age 18-115 years and positive creatinine.")
        net, scaler = self.models[mi - 1]
        x = torch.as_tensor(scaler.transform(frame).astype(np.float32), device=self.device)
        with torch.inference_mode():
            p = np.concatenate([net(x[i:i+256], None).softmax(-1).mean(1)[:, 1].cpu().numpy()
                                for i in range(0, len(frame), 256)]).astype(float)
        return p

    def predict(self, frame):
        """Complete observed inputs; do not fill missing values in deployment."""
        predictions = [self.predict_mi(frame, mi) for mi in range(1, 31)]
        return {"TabM2": np.asarray(predictions).mean(axis=0)}
