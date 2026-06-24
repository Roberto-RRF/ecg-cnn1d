"""Regenera las constantes de inferencia que el notebook NO guardó en el repo.

Replica el split por paciente (SEED=42), calcula:
  - lead_mean / lead_std : z-score por derivación (stats de TRAIN)
  - age_median / age_mean / age_std : normalización de edad (TRAIN)
  - thresholds : umbral óptimo por clase (maximiza F1 en VALIDACIÓN)

Guarda todo en artifacts.json para que la API no necesite recargar el dataset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from ecg_model import CLASSES, ECGNet100, N_SAMPLES

SEED = 42
DATA = Path("..") / "data"
MODELS = Path("..") / "models"
OUT = Path("artifacts.json")

np.random.seed(SEED)
torch.manual_seed(SEED)


def make_split(df):
    groups = df.patient_id.values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    tv_idx, test_idx = next(gss1.split(df, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.111, random_state=SEED)
    tr_rel, val_rel = next(gss2.split(df.iloc[tv_idx], groups=groups[tv_idx]))
    return tv_idx[tr_rel], tv_idx[val_rel], test_idx


def load_signals(files):
    X = np.zeros((len(files), N_SAMPLES, 12), dtype=np.float32)
    t = time.time()
    for i, f in enumerate(files):
        sig, _ = wfdb.rdsamp(str(DATA / f))
        X[i] = sig.astype(np.float32)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)} señales cargadas ({time.time()-t:.0f}s)")
    return X


def main():
    print("Leyendo CSV...")
    df = pd.read_csv(DATA / "ptbxl_clean.csv", index_col="ecg_id")
    train_idx, val_idx, test_idx = make_split(df)
    print(f"train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")

    # --- Cargar señales de train + val (100 Hz) ---
    print("Cargando señales TRAIN (100 Hz)...")
    X_train = load_signals(df.filename_lr.values[train_idx])
    print("Cargando señales VAL (100 Hz)...")
    X_val = load_signals(df.filename_lr.values[val_idx])

    # --- Stats de señal por derivación (TRAIN) ---
    lead_mean = X_train.mean(axis=(0, 1))            # (12,)
    lead_std = X_train.std(axis=(0, 1)) + 1e-6       # (12,)

    # --- Stats de edad (TRAIN) ---
    age = df.age.values.astype(np.float32)
    age_median = float(np.nanmedian(age[train_idx]))
    age_imp = np.where(np.isnan(age), age_median, age)
    age_mean = float(age_imp[train_idx].mean())
    age_std = float(age_imp[train_idx].std() + 1e-6)

    # --- Normalizar val + construir meta para hallar umbrales ---
    Xv = (X_val - lead_mean) / lead_std
    sig_v = torch.from_numpy(np.ascontiguousarray(Xv.transpose(0, 2, 1)))  # (Nv,12,1000)
    age_v = (age_imp[val_idx] - age_mean) / age_std
    sex_v = df.sex.values.astype(np.float32)[val_idx]
    meta_v = torch.from_numpy(np.stack([age_v, sex_v], axis=1).astype(np.float32))
    y_val = df[CLASSES].values.astype(np.float32)[val_idx]

    # --- Cargar modelo y predecir en val ---
    print("Cargando ECGNet100 y evaluando en val...")
    model = ECGNet100(use_meta=True)
    state = torch.load(MODELS / "ecgnet100_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(sig_v), 256):
            logits = model(sig_v[i:i + 256], meta_v[i:i + 256])
            probs.append(torch.sigmoid(logits).numpy())
    val_probs = np.concatenate(probs)

    # --- Umbral óptimo por clase (maximiza F1 en val) ---
    thresholds = []
    for c in range(len(CLASSES)):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.01, 1.0, 0.01):
            f1 = f1_score(y_val[:, c], (val_probs[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds.append(round(best_t, 2))
    print("Umbrales óptimos:", dict(zip(CLASSES, thresholds)))

    artifacts = {
        "model": "ecgnet100_best.pt",
        "fs": 100,
        "classes": CLASSES,
        "lead_mean": [float(x) for x in lead_mean],
        "lead_std": [float(x) for x in lead_std],
        "age_median": age_median,
        "age_mean": age_mean,
        "age_std": age_std,
        "thresholds": thresholds,
        "test_ecg_ids": [int(i) for i in df.index.values[test_idx]],
    }
    OUT.write_text(json.dumps(artifacts, indent=2))
    print(f"Guardado {OUT.resolve()}  ({len(artifacts['test_ecg_ids'])} ids de test para demo)")


if __name__ == "__main__":
    main()
