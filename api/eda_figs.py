"""Genera figuras y estadísticas de EDA para el documento del proyecto."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("..") / "data"
OUT = Path("doc_assets"); OUT.mkdir(exist_ok=True)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NAMES = {"NORM": "Normal", "MI": "Infarto", "STTC": "ST/T", "CD": "Conducción", "HYP": "Hipertrofia"}
COL = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#9b59b6"]

df = pd.read_csv(DATA / "ptbxl_clean.csv", index_col="ecg_id")
stats = {}
stats["n_ecgs"] = len(df)
stats["n_pacientes"] = int(df.patient_id.nunique())
stats["edad_mediana"] = float(df.age.median())
stats["edad_media"] = float(df.age.mean())
stats["pct_mujeres"] = float((df.sex == 1).mean() * 100)
stats["pct_hombres"] = float((df.sex == 0).mean() * 100)
counts = {c: int(df[c].sum()) for c in CLASSES}
stats["conteo_clases"] = counts
stats["pct_clases"] = {c: round(df[c].mean() * 100, 1) for c in CLASSES}
nlab = df[CLASSES].sum(axis=1)
stats["multietiqueta"] = {
    "1_etiqueta": int((nlab == 1).sum()),
    "2_etiquetas": int((nlab == 2).sum()),
    "3+_etiquetas": int((nlab >= 3).sum()),
    "0_etiquetas": int((nlab == 0).sum()),
}

# --- Fig 1: distribución de superclases ---
fig, ax = plt.subplots(figsize=(7, 4))
vals = [counts[c] for c in CLASSES]
bars = ax.bar([NAMES[c] for c in CLASSES], vals, color=COL)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 100, f"{v:,}\n({v/len(df)*100:.0f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Nº de ECGs")
ax.set_title("Distribución de las 5 superclases diagnósticas (PTB-XL)")
ax.set_ylim(0, max(vals) * 1.18)
fig.tight_layout(); fig.savefig(OUT / "eda_clases.png", dpi=120); plt.close(fig)

# --- Fig 2: histograma de edad por sexo ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[df.sex == 0].age.dropna(), bins=30, alpha=0.6, label="Hombres", color="#3498db")
ax.hist(df[df.sex == 1].age.dropna(), bins=30, alpha=0.6, label="Mujeres", color="#e74c3c")
ax.axvline(df.age.median(), color="k", ls="--", lw=1, label=f"Mediana {df.age.median():.0f} años")
ax.set_xlabel("Edad (años)"); ax.set_ylabel("Nº de ECGs")
ax.set_title("Distribución de edad por sexo")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "eda_edad.png", dpi=120); plt.close(fig)

# --- Fig 3: número de etiquetas por ECG (multi-etiqueta) ---
fig, ax = plt.subplots(figsize=(6, 4))
order = ["0_etiquetas", "1_etiqueta", "2_etiquetas", "3+_etiquetas"]
labels = ["0", "1", "2", "3+"]
vals = [stats["multietiqueta"][k] for k in order]
bars = ax.bar(labels, vals, color="#16a085")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 100, f"{v:,}", ha="center", fontsize=9)
ax.set_xlabel("Nº de superclases por ECG"); ax.set_ylabel("Nº de ECGs")
ax.set_title("Naturaleza multi-etiqueta del problema")
fig.tight_layout(); fig.savefig(OUT / "eda_multietiqueta.png", dpi=120); plt.close(fig)

(OUT / "eda_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))
print("\nFiguras guardadas en", OUT.resolve())
