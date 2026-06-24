"""Prueba rápida sin levantar el servidor: carga el modelo y clasifica unos ECG de test."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from ecg_model import CLASSES, ECGNet100, predict, preprocess

DATA = Path("..") / "data"
art = json.loads(Path("artifacts.json").read_text())
df = pd.read_csv(DATA / "ptbxl_clean.csv", index_col="ecg_id")
model = ECGNet100(use_meta=True)
model.load_state_dict(torch.load(Path("..") / "models" / art["model"], map_location="cpu"))
model.eval()

ok = 0
ids = art["test_ecg_ids"][:10]
for ecg_id in ids:
    row = df.loc[ecg_id]
    sig, _ = wfdb.rdsamp(str(DATA / row.filename_lr))
    age = None if pd.isna(row.age) else float(row.age)
    sig_t, meta_t = preprocess(sig.astype(np.float32), age, int(row.sex), art)
    res = predict(model, sig_t, meta_t, art)
    true = [c for c in CLASSES if row[c] == 1]
    hit = set(res["predicted"]) == set(true)
    ok += hit
    print(f"ECG #{ecg_id:>6} | real={true} | pred={res['predicted']} | "
          f"{'OK' if hit else 'x'} | probs=" +
          " ".join(f"{c}:{res['probabilities'][c]:.2f}" for c in CLASSES))
print(f"\nCoincidencia exacta multi-etiqueta: {ok}/{len(ids)} "
      "(nota: multi-etiqueta, coincidencia exacta es estricta)")
