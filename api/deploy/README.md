---
title: ECG CNN1D API
emoji: 🫀
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ECG-CNN1D — Clasificador de electrocardiogramas

Aplicación **vía API** que clasifica electrocardiogramas (dataset PTB-XL) en 5 superclases
diagnósticas (`NORM`, `MI`, `STTC`, `CD`, `HYP`) usando una **CNN 1-D** (`ECGNet100`).

Problema **multi-etiqueta**: un ECG puede tener varias patologías a la vez, cada una con su
propio umbral de decisión.

## Uso

- **`/`** — interfaz web: elige un ECG de muestra y obtén el diagnóstico + probabilidades por clase.
- **`/docs`** — documentación interactiva (Swagger) de la API REST.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET`  | `/health` | Estado del servicio |
| `GET`  | `/samples` | ECGs de muestra disponibles |
| `GET`  | `/plot/{ecg_id}` | PNG de las 12 derivaciones |
| `POST` | `/predict` | Clasifica un ECG (`{"ecg_id": 146}` o una señal cruda) |

Esta demo incluye ~95 ECGs de muestra del conjunto de test. El modelo y el preprocesamiento son
los mismos del proyecto completo (repo `ecg-cnn1d`).
