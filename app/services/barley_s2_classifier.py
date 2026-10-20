"""
Classifieur orge / non-orge sur les valeurs Sentinel-2 de la parcelle.

Remplace, pour le verdict, la fusion CNN + regles agro : comparee a 28 parcelles
relevees sur le terrain (Antsirabe, septembre 2026), elle ne faisait pas mieux
que le hasard (CNN : AUC 0,58 ; agro : 0,32 ; hybride : 0,51), alors que les
indices Sentinel-2 de la parcelle separaient deja l'orge des autres cultures
(NDVI, EVI, SAVI : AUC ~0,82).

Regression logistique sur variables centrees-reduites, coefficients dans
app/data/barley_s2_model.json, produits par scripts/calibrate_barley_s2.py a
partir de releves terrain. Sans fichier de modele, ou si une variable manque
(parcelle sous les nuages), classify() renvoie None et l'analyse garde
l'ancienne fusion.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("agrisat.services.barley_s2_classifier")

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "barley_s2_model.json"
DECISION_THRESHOLD = 0.5
L2_PENALTY = 1.0


def fit_logistic(x: np.ndarray, y: np.ndarray, features: list[str]) -> dict[str, Any]:
    """Regression logistique penalisee (L2), par descente de gradient : peu de donnees,
    pas besoin de scikit-learn."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    w = np.zeros(z.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(5_000):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        w -= 0.1 * ((z.T @ (p - y)) / n + L2_PENALTY * w / n)
        b -= 0.1 * float(np.mean(p - y))
    return {
        "features": features,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": w.tolist(),
        "intercept": b,
        "threshold": DECISION_THRESHOLD,
    }


def predict_proba(model: dict[str, Any], values: dict[str, float]) -> float:
    logit = model["intercept"] + sum(
        c * (values[f] - m) / s for f, c, m, s in zip(model["features"], model["coef"], model["mean"], model["std"])
    )
    return 1 / (1 + math.exp(-logit))


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any] | None:
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Modele Sentinel-2 orge absent (%s) : verdict par l'ancienne fusion CNN + agro.", MODEL_PATH)
    except (OSError, ValueError) as error:
        logger.error("Modele Sentinel-2 orge illisible (%s) : %s", MODEL_PATH, error)
    return None


def classify(sat_data: dict[str, Any]) -> dict[str, Any] | None:
    """Probabilite d'orge de la parcelle, ou None si le modele ou une variable manque."""
    model = load_model()
    if model is None:
        return None
    values = {f: sat_data.get(f) for f in model["features"]}
    if any(v is None for v in values.values()):
        return None
    probability = predict_proba(model, values)
    return {
        "probability": probability,
        "is_barley": probability >= model.get("threshold", DECISION_THRESHOLD),
        "features": model["features"],
        "calibration": model.get("calibration", {}),
    }
