"""Arquitectura ECGNet100 y utilidades de preprocesamiento.

Replica exactamente el modelo del notebook 02_modelo_cnn1d_100hz.ipynb para que el
state_dict guardado (models/ecgnet100_best.pt) cargue sin cambios.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# --- Constantes del problema (idénticas al notebook) ---
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100           # Hz
N_SAMPLES = 1000   # 10 s × 100 Hz

CLASS_NAMES_ES = {
    "NORM": "ECG normal (sano)",
    "MI": "Infarto de miocardio",
    "STTC": "Alteraciones ST/onda T",
    "CD": "Trastorno de la conducción",
    "HYP": "Hipertrofia",
}


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=k // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.net(x)


class ECGNet100(nn.Module):
    def __init__(self, n_leads=12, n_meta=2, n_classes=5, use_meta=True):
        super().__init__()
        self.use_meta = use_meta
        # 1000 -> 500 -> 250 -> 125 -> AdaptiveAvgPool
        self.cnn = nn.Sequential(
            ConvBlock(n_leads, 64, 7, pool=2),
            ConvBlock(64, 128, 5, pool=2),
            ConvBlock(128, 256, 3, pool=2),
            nn.AdaptiveAvgPool1d(1),
        )
        head_in = 256 + (n_meta if use_meta else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, sig, meta):
        x = self.cnn(sig).flatten(1)
        if self.use_meta:
            x = torch.cat([x, meta], dim=1)
        return self.head(x)


def preprocess(sig_raw: np.ndarray, age: float, sex: int, artifacts: dict):
    """Convierte una señal cruda (1000, 12) en mV + metadata en tensores listos para el modelo.

    Aplica exactamente el preprocesamiento del notebook:
      - señal: z-score por derivación con stats de train
      - edad: imputar NaN con mediana de train, luego z-score
      - sexo: 0/1 sin transformar
    Devuelve (sig_t (1,12,1000), meta_t (1,2)).
    """
    mean = np.asarray(artifacts["lead_mean"], dtype=np.float32)  # (12,)
    std = np.asarray(artifacts["lead_std"], dtype=np.float32)    # (12,)

    sig = np.asarray(sig_raw, dtype=np.float32)
    if sig.shape == (12, N_SAMPLES):
        sig = sig.T
    assert sig.shape == (N_SAMPLES, 12), f"señal con forma inesperada: {sig.shape}"
    sig = (sig - mean) / std                       # (1000, 12)
    sig_t = torch.from_numpy(np.ascontiguousarray(sig.T)).unsqueeze(0)  # (1, 12, 1000)

    if age is None or (isinstance(age, float) and np.isnan(age)):
        age = artifacts["age_median"]
    age_z = (float(age) - artifacts["age_mean"]) / artifacts["age_std"]
    meta = np.array([[age_z, float(sex)]], dtype=np.float32)        # (1, 2)
    meta_t = torch.from_numpy(meta)
    return sig_t, meta_t


def predict(model, sig_t, meta_t, artifacts: dict):
    """Devuelve dict con probabilidades por clase y predicción con umbrales óptimos."""
    thr = np.asarray(artifacts["thresholds"], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(sig_t, meta_t)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    preds = (probs >= thr).astype(int)
    return {
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "thresholds": {c: float(t) for c, t in zip(CLASSES, thr)},
        "predicted": [c for c, p in zip(CLASSES, preds) if p == 1],
    }
