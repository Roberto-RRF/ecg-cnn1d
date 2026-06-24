"""API de clasificación de ECG (PTB-XL, 5 superclases) con FastAPI.

Endpoints:
  GET  /            -> interfaz web de demostración
  GET  /health      -> estado del servicio
  GET  /samples     -> lista de ecg_ids del conjunto de test para probar
  GET  /plot/{id}   -> PNG con las 12 derivaciones del ECG
  POST /predict     -> clasifica un ECG (por ecg_id de PTB-XL o señal enviada)

Ejecutar:  uv run uvicorn app:app --reload
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from ecg_model import (CLASS_NAMES_ES, CLASSES, LEADS, ECGNet100, predict,
                       preprocess)

DATA = Path(os.environ.get("ECG_DATA_DIR", Path("..") / "data"))
MODELS = Path(os.environ.get("ECG_MODELS_DIR", Path("..") / "models"))
ART_PATH = Path(os.environ.get("ECG_ARTIFACTS", "artifacts.json"))

# --- Carga única al iniciar ---
artifacts = json.loads(ART_PATH.read_text())
model = ECGNet100(use_meta=True)
model.load_state_dict(torch.load(MODELS / artifacts["model"], map_location="cpu"))
model.eval()
df = pd.read_csv(DATA / "ptbxl_clean.csv", index_col="ecg_id")

app = FastAPI(title="ECG-CNN1D API",
              description="Clasificación multi-etiqueta de ECG (PTB-XL) con CNN 1-D",
              version="1.0")


class PredictRequest(BaseModel):
    ecg_id: int | None = None          # usar un registro de PTB-XL
    signal: list[list[float]] | None = None  # o enviar señal cruda (1000×12) en mV
    age: float | None = None
    sex: int | None = None             # 0=H, 1=M


def load_record(ecg_id: int):
    if ecg_id not in df.index:
        raise HTTPException(404, f"ecg_id {ecg_id} no existe en el dataset")
    row = df.loc[ecg_id]
    sig, _ = wfdb.rdsamp(str(DATA / row.filename_lr))   # (1000, 12) en mV
    return sig.astype(np.float32), row


@app.get("/health")
def health():
    return {"status": "ok", "model": artifacts["model"], "classes": CLASSES,
            "n_test_samples": len(artifacts["test_ecg_ids"])}


@app.get("/samples")
def samples(n: int = 30):
    """Devuelve ecg_ids del conjunto de TEST (no vistos en entrenamiento) para probar."""
    ids = artifacts["test_ecg_ids"][:n]
    out = []
    for i in ids:
        row = df.loc[i]
        true = [c for c in CLASSES if row[c] == 1]
        out.append({"ecg_id": int(i), "age": None if pd.isna(row.age) else int(row.age),
                    "sex": "M" if row.sex == 1 else "H", "true_labels": true})
    return out


@app.get("/plot/{ecg_id}")
def plot(ecg_id: int):
    sig, row = load_record(ecg_id)
    fig, axes = plt.subplots(6, 2, figsize=(11, 9), sharex=True)
    t = np.arange(sig.shape[0]) / artifacts["fs"]
    for k, ax in enumerate(axes.T.flatten()):
        ax.plot(t, sig[:, k], lw=0.7, color="#c0392b")
        ax.set_ylabel(LEADS[k], rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("tiempo (s)"); axes[-1, 1].set_xlabel("tiempo (s)")
    true = ", ".join(c for c in CLASSES if row[c] == 1) or "—"
    fig.suptitle(f"ECG #{ecg_id}  ·  12 derivaciones  ·  etiqueta real: {true}", fontsize=12)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/gradcam/{ecg_id}")
def gradcam(ecg_id: int):
    """Grad-CAM 1-D: heatmap por cada clase detectada, sobre la derivación II."""
    sig, row = load_record(ecg_id)
    age = None if pd.isna(row.age) else float(row.age)
    sig_t, meta_t = preprocess(sig, age, int(row.sex), artifacts)   # (1,12,1000)
    Xn = sig_t[0].numpy().T                                          # (1000,12) normalizado
    thr = np.asarray(artifacts["thresholds"], np.float32)

    acts, grads = {}, {}
    def _hook(m, i, o):
        acts["v"] = o
        if o.requires_grad:
            o.register_hook(lambda g: grads.__setitem__("v", g))
    handle = model.cnn[2].register_forward_hook(_hook)

    def cam_for(target):
        model.zero_grad()
        logits = model(sig_t, meta_t)
        probs = torch.sigmoid(logits).detach().numpy()[0]
        logits[0, target].backward()
        a = acts["v"][0]; g = grads["v"][0]
        w = g.mean(dim=1)
        cam = torch.relu((w[:, None] * a).sum(0)).detach().cpu().numpy()
        cam = np.interp(np.arange(1000), np.linspace(0, 999, len(cam)), cam)
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        return cam, probs

    _, probs = cam_for(0)
    true = [c for c in CLASSES if row[c] == 1]
    detected = [CLASSES[i] for i in range(5) if probs[i] >= thr[i]]
    targets = detected or [CLASSES[int(probs.argmax())]]

    t = np.arange(1000) / artifacts["fs"]
    s = Xn[:, 1]  # derivación II
    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 2.4 * len(targets)), squeeze=False)
    for ax, name in zip(axes[:, 0], targets):
        ci = CLASSES.index(name)
        cam, _ = cam_for(ci)
        ax.imshow(cam[None, :], aspect="auto", cmap="jet", alpha=0.35,
                  extent=[0, 10, s.min(), s.max()], origin="lower")
        ax.plot(t, s, color="black", lw=0.7)
        ax.set_xlim(0, 10); ax.set_ylim(s.min(), s.max())
        ax.set_ylabel("Deriv. II", fontsize=9)
        ax.set_title(f"{name} — {CLASS_NAMES_ES[name]}   (prob: {probs[ci]:.2f})",
                     fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("tiempo (s)")
    fig.suptitle(f"Grad-CAM · ECG #{ecg_id}  (real: {', '.join(true) or '—'})  ·  "
                 "cálido = donde más mira el modelo", fontsize=11, y=1.0)
    handle.remove()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/predict")
def do_predict(req: PredictRequest):
    if req.ecg_id is not None:
        sig, row = load_record(req.ecg_id)
        age = req.age if req.age is not None else (None if pd.isna(row.age) else float(row.age))
        sex = req.sex if req.sex is not None else int(row.sex)
        true = [c for c in CLASSES if row[c] == 1]
    elif req.signal is not None:
        sig = np.asarray(req.signal, dtype=np.float32)
        if sig.shape not in [(1000, 12), (12, 1000)]:
            raise HTTPException(400, f"señal debe ser 1000×12 o 12×1000, recibí {sig.shape}")
        age, sex, true = req.age, (req.sex if req.sex is not None else 0), None
    else:
        raise HTTPException(400, "envía 'ecg_id' o 'signal'")

    sig_t, meta_t = preprocess(sig, age, sex, artifacts)
    result = predict(model, sig_t, meta_t, artifacts)
    result["diagnosis"] = [{"code": c, "name": CLASS_NAMES_ES[c]} for c in result["predicted"]] \
        or [{"code": "—", "name": "Sin hallazgo por encima del umbral"}]
    if true is not None:
        result["true_labels"] = true
    result["meta_used"] = {"age": age, "sex": "M" if sex == 1 else "H"}
    return result


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


INDEX_HTML = """
<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECG-CNN1D · Clasificador de ECG</title>
<style>
  :root{--bg:#0f172a;--card:#1e293b;--accent:#ef4444;--txt:#e2e8f0;--mut:#94a3b8}
  *{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--txt)}
  header{padding:20px 28px;border-bottom:1px solid #334155}
  h1{margin:0;font-size:20px} .sub{color:var(--mut);font-size:13px;margin-top:4px}
  main{display:grid;grid-template-columns:330px 1fr;gap:20px;padding:24px;max-width:1200px;margin:auto}
  .card{background:var(--card);border:1px solid #334155;border-radius:12px;padding:18px}
  label{display:block;font-size:13px;color:var(--mut);margin:10px 0 4px}
  select,button{width:100%;padding:10px;border-radius:8px;border:1px solid #475569;background:#0b1220;color:var(--txt);font-size:14px}
  button{background:var(--accent);border:none;font-weight:600;cursor:pointer;margin-top:16px}
  button:hover{opacity:.9} img{width:100%;border-radius:8px;background:#fff;margin-top:10px}
  .bar{height:22px;background:#0b1220;border-radius:6px;overflow:hidden;margin:3px 0 10px}
  .bar > div{height:100%;background:linear-gradient(90deg,#f59e0b,#ef4444);display:flex;align-items:center;padding-left:8px;font-size:12px;color:#fff;white-space:nowrap}
  .row{display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px}
  .tag{display:inline-block;background:#ef4444;color:#fff;padding:3px 10px;border-radius:20px;font-size:13px;margin:3px 4px 0 0}
  .tag.true{background:#10b981}
  .mut{color:var(--mut);font-size:12px}
</style></head><body>
<header><h1>🫀 ECG-CNN1D — Clasificador de electrocardiogramas</h1>
<div class="sub">CNN 1-D · dataset PTB-XL · 5 superclases diagnósticas (NORM, MI, STTC, CD, HYP) · multi-etiqueta</div></header>
<main>
  <div class="card">
    <label>Selecciona un ECG del conjunto de test</label>
    <select id="sel"></select>
    <button onclick="run()">Clasificar ECG</button>
    <div id="info" class="mut" style="margin-top:14px"></div>
  </div>
  <div class="card">
    <div id="out" class="mut">Elige un ECG y presiona «Clasificar».</div>
    <img id="plot" style="display:none">
    <button id="gbtn" onclick="vergradcam()" style="display:none;margin-top:14px">🔥 Ver Grad-CAM (¿dónde mira el modelo?)</button>
    <div id="gcap" class="mut" style="display:none;margin-top:10px;font-size:12px">Mapa de calor por cada clase detectada: en cálido (amarillo/rojo), las zonas del latido en que más se fija el modelo para decidir.</div>
    <img id="gcam" style="display:none">
  </div>
</main>
<script>
const NAMES={NORM:"ECG normal (sano)",MI:"Infarto de miocardio",STTC:"Alteraciones ST/onda T",CD:"Trastorno de conducción",HYP:"Hipertrofia"};
async function load(){
  const r=await fetch('/samples?n=40'); const s=await r.json();
  const sel=document.getElementById('sel');
  s.forEach(x=>{const o=document.createElement('option');o.value=x.ecg_id;
    o.textContent=`ECG #${x.ecg_id} · ${x.sex} ${x.age??'?'}a · real: ${x.true_labels.join(',')||'—'}`;sel.appendChild(o);});
}
let currentId=null;
function vergradcam(){
  if(currentId==null) return;
  const g=document.getElementById('gcam');
  g.src='/gradcam/'+currentId+'?t='+Date.now();
  g.style.display='block';
  document.getElementById('gcap').style.display='block';
  document.getElementById('gbtn').textContent='🔥 Grad-CAM (recalcular)';
}
async function run(){
  const id=+document.getElementById('sel').value;
  currentId=id;
  document.getElementById('out').innerHTML='Procesando…';
  document.getElementById('gcam').style.display='none';
  document.getElementById('gcap').style.display='none';
  const img=document.getElementById('plot'); img.src='/plot/'+id+'?t='+Date.now(); img.style.display='block';
  document.getElementById('gbtn').style.display='block';
  document.getElementById('gbtn').textContent='🔥 Ver Grad-CAM (¿dónde mira el modelo?)';
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ecg_id:id})});
  const d=await r.json();
  let h='<h3 style="margin:4px 0 10px">Diagnóstico predicho</h3>';
  h+=d.diagnosis.map(x=>`<span class="tag">${x.code} — ${x.name}</span>`).join('');
  if(d.true_labels) h+='<div style="margin-top:12px" class="mut">Etiqueta real: '+(d.true_labels.map(c=>`<span class="tag true">${c}</span>`).join('')||'—')+'</div>';
  h+='<h3 style="margin:18px 0 8px">Probabilidad por clase</h3>';
  const probs=d.probabilities, thr=d.thresholds;
  Object.keys(probs).forEach(c=>{const p=probs[c],pct=(p*100).toFixed(1);
    h+=`<div class="row"><span>${c} — ${NAMES[c]}</span><span>${pct}% (umbral ${(thr[c]*100).toFixed(0)}%)</span></div>`;
    h+=`<div class="bar"><div style="width:${Math.max(p*100,4)}%">${pct}%</div></div>`;});
  document.getElementById('out').innerHTML=h;
  document.getElementById('info').innerHTML='Metadata usada: sexo '+d.meta_used.sex+', edad '+(d.meta_used.age??'imputada');
}
load();
</script></body></html>
"""
