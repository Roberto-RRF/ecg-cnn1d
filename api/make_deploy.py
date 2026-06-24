"""Construye una carpeta de despliegue ligera para Hugging Face Spaces.

Empaqueta solo lo necesario:
  - app.py, ecg_model.py
  - modelo ecgnet100_best.pt (716 KB)
  - artifacts.json recortado (solo los ids de muestra)
  - CSV recortado + ~150 registros WFDB de muestra (balanceados por clase) -> ~5 MB

Resultado en deploy/  (listo para subir a un Space con Dockerfile).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ecg_model import CLASSES

DATA = Path("..") / "data"
MODELS = Path("..") / "models"
DEPLOY = Path("deploy")
N_PER_CLASS = 25          # objetivo por clase (incl. multi-etiqueta)

art = json.loads(Path("artifacts.json").read_text())
df = pd.read_csv(DATA / "ptbxl_clean.csv", index_col="ecg_id")
test_ids = art["test_ecg_ids"]
test_df = df.loc[test_ids]

# --- Selección balanceada: para cada clase, tomar N positivos del set de test ---
chosen: list[int] = []
for c in CLASSES:
    pos = test_df[test_df[c] == 1].index.tolist()
    for i in pos:
        if i not in chosen:
            chosen.append(i)
        if sum(df.loc[chosen, c] == 1) >= N_PER_CLASS:
            break
# añadir algunos multi-etiqueta para casos vistosos
multi = test_df[test_df[CLASSES].sum(axis=1) >= 2].index.tolist()
for i in multi[:20]:
    if i not in chosen:
        chosen.append(i)
chosen = sorted(set(chosen))
print(f"ECGs de muestra seleccionados: {len(chosen)}")
print("cobertura por clase:", {c: int(df.loc[chosen, c].sum()) for c in CLASSES})

# --- Construir estructura deploy/ ---
if DEPLOY.exists():
    shutil.rmtree(DEPLOY)
(DEPLOY / "data" / "records100").mkdir(parents=True)
(DEPLOY / "models").mkdir(parents=True)

# código + modelo
for f in ["app.py", "ecg_model.py"]:
    shutil.copy(f, DEPLOY / f)
shutil.copy(MODELS / "ecgnet100_best.pt", DEPLOY / "models" / "ecgnet100_best.pt")

# CSV recortado
sub = df.loc[chosen].copy()
sub.to_csv(DEPLOY / "data" / "ptbxl_clean.csv")

# artifacts recortado (solo ids de muestra)
art_out = dict(art)
art_out["test_ecg_ids"] = [int(i) for i in chosen]
(DEPLOY / "artifacts.json").write_text(json.dumps(art_out, indent=2))

# copiar registros WFDB (.dat + .hea) preservando estructura
n = 0
for i in chosen:
    rel = df.loc[i, "filename_lr"]            # records100/00000/00001_lr
    for ext in (".dat", ".hea"):
        srcp = DATA / (rel + ext)
        dstp = DEPLOY / "data" / (rel + ext)
        dstp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(srcp, dstp)
        n += 1
print(f"archivos WFDB copiados: {n}")

size = sum(p.stat().st_size for p in DEPLOY.rglob("*") if p.is_file())
print(f"tamaño total de deploy/: {size/1e6:.1f} MB")
