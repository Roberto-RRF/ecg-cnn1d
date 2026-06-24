"""Grad-CAM 1D del modelo ECGNet100 aplicado a un ECG concreto (por defecto #274).

Replica la implementación del notebook 02_100hz: hook en model.cnn[2] (última conv),
heatmap temporal (relu de la suma ponderada de activaciones) interpolado a 1000 muestras.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb

from ecg_model import CLASS_NAMES_ES, CLASSES, LEADS, ECGNet100

ECG_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 274
FS = 100
DATA = Path("..") / "data"
art = json.loads(Path("artifacts.json").read_text())
df = pd.read_csv(DATA / "ptbxl_clean.csv", index_col="ecg_id")
row = df.loc[ECG_ID]

# --- modelo ---
model = ECGNet100(use_meta=True)
model.load_state_dict(torch.load(Path("..") / "models" / art["model"], map_location="cpu"))
model.eval()

# --- señal cruda + preprocesamiento (igual que en producción) ---
sig_raw, _ = wfdb.rdsamp(str(DATA / row.filename_lr))          # (1000, 12) mV
mean = np.asarray(art["lead_mean"], np.float32)
std = np.asarray(art["lead_std"], np.float32)
Xn = ((sig_raw.astype(np.float32) - mean) / std)               # (1000,12) normalizado
age = art["age_median"] if pd.isna(row.age) else float(row.age)
age_z = (age - art["age_mean"]) / art["age_std"]
meta = np.array([[age_z, float(row.sex)]], dtype=np.float32)
thr = np.asarray(art["thresholds"], np.float32)

# --- hook Grad-CAM en la última convolución (cnn[2]) ---
acts, grads = {}, {}
def _fwd_hook(m, i, o):
    acts["v"] = o
    if o.requires_grad:
        o.register_hook(lambda g: grads.__setitem__("v", g))
handle = model.cnn[2].register_forward_hook(_fwd_hook)

def grad_cam(target_class):
    sig = torch.from_numpy(np.ascontiguousarray(Xn.T)[None])    # (1,12,1000)
    mt = torch.from_numpy(meta)
    model.zero_grad()
    logits = model(sig, mt)
    probs = torch.sigmoid(logits).detach().numpy()[0]
    logits[0, target_class].backward()
    a = acts["v"][0]; g = grads["v"][0]
    w = g.mean(dim=1)
    cam = torch.relu((w[:, None] * a).sum(0)).detach().cpu().numpy()
    cam = np.interp(np.arange(1000), np.linspace(0, 999, len(cam)), cam)
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    return cam, probs

# probabilidades y clases activas
_, probs = grad_cam(0)
active = [CLASSES[i] for i in range(5) if probs[i] >= thr[i]]
true = [c for c in CLASSES if row[c] == 1]
target_classes = [c for c in CLASSES if c in set(true) | set(active)] or [CLASSES[int(probs.argmax())]]

# --- graficar: una fila por clase de interés, derivación II + heatmap ---
t = np.arange(1000) / FS
LEAD = 1  # derivación II
fig, axes = plt.subplots(len(target_classes), 1, figsize=(13, 2.6 * len(target_classes)))
if len(target_classes) == 1:
    axes = [axes]
s = Xn[:, LEAD]
for ax, name in zip(axes, target_classes):
    ci = CLASSES.index(name)
    cam, _ = grad_cam(ci)
    ax.imshow(cam[None, :], aspect="auto", cmap="jet", alpha=0.35,
              extent=[0, 10, s.min(), s.max()], origin="lower")
    ax.plot(t, s, color="black", lw=0.7)
    ax.set_xlim(0, 10); ax.set_ylim(s.min(), s.max())
    ax.set_ylabel("Deriv. II", fontsize=9)
    mark = "✓ detectado" if probs[ci] >= thr[ci] else "✗ no supera umbral"
    ax.set_title(f"{name} — {CLASS_NAMES_ES[name]}   |   prob: {probs[ci]:.2f}  ({mark})",
                 fontsize=11, loc="left")
axes[-1].set_xlabel("tiempo (s)")
fig.suptitle(f"Grad-CAM 1-D · ECG #{ECG_ID}  (etiqueta real: {', '.join(true)})\n"
             "Cálido = región donde el modelo más se fija para decidir",
             fontsize=12, y=1.0)
handle.remove()
fig.tight_layout()
out = Path("doc_assets") / f"gradcam_{ECG_ID}.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("Guardado:", out.resolve())
print("probabilidades:", {c: round(float(probs[i]), 3) for i, c in enumerate(CLASSES)})
print("activas:", active, "| reales:", true)
