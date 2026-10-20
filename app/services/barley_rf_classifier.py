"""Classifieur orge / non-orge du projet `ASA/Orge` (Random Forest, 48 variables).

Remplace le modèle `rf_orge` (modele_orge) par le modèle entraîné dans le projet
`D:\\ASA\\Orge` et embarqué dans `app/data/models/orge_rf/` :

- `orge_rf.joblib` : pipeline sklearn + seuil de décision + liste des variables ;
- `metrics.json` et `threshold_selection.json` : provenance et métriques ;
- `feature_importances.csv` : importance des variables.

Le modèle classe chaque observation (une date) d'une parcelle, puis la
probabilité de la parcelle est la moyenne des probabilités de ses dates — c'est
le niveau de décision calibré pour un rappel de 85 % sur les parcelles d'orge.

Les 48 variables reproduisent exactement `orge_model.py` du projet d'origine :
bandes B2…B12, indices NDVI/EVI/NDRE/NDMI, senescence (PSRI) et stress hydrique
(MSI), contexte saisonnier, statistiques temporelles par parcelle, phénologie de
la contre-saison malgache (mai-octobre) et durée du cycle végétatif.

Aucune API externe : le pipeline joblib est chargé une seule fois en mémoire et
exécuté directement par le backend.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("agrisat.services.barley_rf_classifier")

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "orge_rf"
MODEL_PATH = MODEL_DIR / "orge_rf.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
THRESHOLD_PATH = MODEL_DIR / "threshold_selection.json"

#: verdict à trois niveaux de predict_orge.py (projet Orge)
AUTO_ORGE_THRESHOLD = 0.70
REJECT_THRESHOLD = 0.15
MAX_OCT_NDVI = 0.50
MAX_DOY_PEAK = 245

#: en dessous : pas de verdict parcelle (série inexploitable)
MIN_VALID_OBSERVATIONS = 1

SPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
VEGETATION_INDICES = ["NDVI", "EVI", "NDRE", "NDMI"]
TEMPORAL_STATS = ["mean", "max", "min", "std"]
CYCLE_FEATURES = [
    "green_span_days",
    "first_green_doy",
    "last_green_doy",
    "pct_green",
    "ndvi_falling_slope",
]
GREEN_NDVI_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any] | None:
    """Charge (une fois) le pipeline joblib, ses variables et ses métadonnées."""
    try:
        bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        logger.warning("Modèle RF orge absent (%s) : verdict par le classifieur Sentinel-2 logistique.", MODEL_PATH)
        return None
    except (OSError, ValueError) as error:
        logger.error("Modèle RF orge illisible (%s) : %s", MODEL_PATH, error)
        return None

    metrics: dict[str, Any] = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning("Métriques du modèle orge illisibles (%s) : %s", METRICS_PATH, error)

    threshold = float(bundle.get("threshold", 0.5))
    if THRESHOLD_PATH.exists():
        try:
            selection = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
            threshold = float(selection.get("seuil_retenu", threshold))
        except (OSError, ValueError, TypeError) as error:
            logger.warning("Sélection du seuil orge illisible (%s) : %s", THRESHOLD_PATH, error)

    generated_at = str(metrics.get("generated_at") or "")
    return {
        "model": bundle["model"],
        "features": list(bundle["features"]),
        "threshold": threshold,
        "positive_label": bundle.get("positive_label", "ORGE"),
        "summary": {
            "production_model": f"orge_rf_{generated_at[:10]}" if generated_at else "orge_rf",
            "final_metrics": metrics.get("metrics_threshold_tuned") or {},
            "evaluation": metrics.get("evaluation"),
            "trained_on_rows": bundle.get("trained_on_rows"),
            "trained_on_parcels": bundle.get("trained_on_parcels"),
        },
        "metrics": metrics,
    }


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    """mean / max / min / std d'une série, NaN ignorés (comme orge_model.py)."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"mean": np.nan, "max": np.nan, "min": np.nan, "std": np.nan}
    return {
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
        "min": float(np.min(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
    }


def _cycle_features(frame: pd.DataFrame) -> dict[str, float]:
    """Durée du cycle végétatif : règle agronomique de l'orge (≤ 150 jours)."""
    green = frame.loc[frame["NDVI"] >= GREEN_NDVI_THRESHOLD]
    if green.empty:
        return {name: 0.0 for name in CYCLE_FEATURES}

    first_green = float(green["day_of_year"].min())
    last_green = float(green["day_of_year"].max())
    peak_index = frame["NDVI"].idxmax()
    peak_doy = float(frame.loc[peak_index, "day_of_year"])
    after_peak = frame.loc[frame["day_of_year"] >= peak_doy].dropna(subset=["NDVI"])
    falling_slope = (
        float(np.polyfit(after_peak["day_of_year"].to_numpy(), after_peak["NDVI"].to_numpy(), 1)[0])
        if len(after_peak) >= 3
        else 0.0
    )
    return {
        "green_span_days": last_green - first_green,
        "first_green_doy": first_green,
        "last_green_doy": last_green,
        "pct_green": len(green) / len(frame),
        "ndvi_falling_slope": falling_slope,
    }


def build_features(series: list[dict[str, Any]]) -> pd.DataFrame | None:
    """Construit les 48 variables du modèle à partir de la série mensuelle S2.

    `series` : [{date: 'YYYY-MM', B2…B12, NDVI, EVI, NDRE, NDMI}, ...].
    Retourne un DataFrame (une ligne par date) ou None si la série est vide ou
    sans aucune observation NDVI exploitable.
    """
    model = load_model()
    if model is None or not series:
        return None

    frame = pd.DataFrame(series)
    if "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = (
        frame.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)
    )
    if frame.empty:
        return None

    for column in [*SPECTRAL_BANDS, *VEGETATION_INDICES]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if int(frame["NDVI"].notna().sum()) < MIN_VALID_OBSERVATIONS:
        return None

    frame["month"] = frame["date"].dt.month
    frame["day_of_year"] = frame["date"].dt.dayofyear

    # Indices complémentaires : senescence (PSRI) et teneur en eau (MSI).
    b2, b4, b6, b8, b11 = (frame[name].to_numpy(dtype=float) for name in ["B2", "B4", "B6", "B8", "B11"])
    frame["PSRI"] = (b4 - b2) / (b6 + 1e-4)
    frame["MSI"] = b11 / (b8 + 1e-4)

    # Statistiques temporelles globales de la parcelle.
    stats: dict[str, float] = {}
    for index in VEGETATION_INDICES:
        for name, value in _finite_stats(frame[index].to_numpy(dtype=float)).items():
            stats[f"{index}_{name}"] = value

    mean_ndvi = stats["NDVI_mean"]
    max_ndvi = stats["NDVI_max"]
    ndvi_by_month = {
        month: float(frame.loc[frame["month"] == month, "NDVI"].mean())
        for month in (6, 8, 9, 10)
    }
    for month, key in ((6, "jun"), (8, "aug"), (9, "sep"), (10, "oct")):
        value = ndvi_by_month[month]
        stats[f"NDVI_{key}"] = mean_ndvi if not np.isfinite(value) else value

    valid_ndvi = frame.dropna(subset=["NDVI"])
    doy_peak = float(valid_ndvi.loc[valid_ndvi["NDVI"].idxmax(), "day_of_year"]) if not valid_ndvi.empty else 230.0
    stats["doy_max_ndvi"] = doy_peak
    stats["drop_max_to_oct"] = max_ndvi - stats["NDVI_oct"]
    stats["drop_sep_oct"] = stats["NDVI_sep"] - stats["NDVI_oct"]
    stats["ratio_oct_max"] = stats["NDVI_oct"] / (max_ndvi + 1e-4)
    stats["ramp_jun_to_aug"] = stats["NDVI_aug"] - stats["NDVI_jun"]
    stats.update(_cycle_features(frame))

    row_columns = [*SPECTRAL_BANDS, *VEGETATION_INDICES, "PSRI", "MSI", "month", "day_of_year"]
    rows = frame[row_columns].copy()
    for name, value in stats.items():
        rows[name] = value

    features = model["features"]
    for name in features:
        if name not in rows.columns:
            rows[name] = np.nan
    return rows[features].astype(float).fillna(0.0)


def _verdict(probability: float, ndvi_oct: float, doy_peak: float) -> str:
    """Verdict à trois niveaux de predict_orge.py, appliqué au niveau parcelle."""
    if probability < REJECT_THRESHOLD:
        return "NON_ORGE"
    if probability >= AUTO_ORGE_THRESHOLD and ndvi_oct <= MAX_OCT_NDVI and doy_peak <= MAX_DOY_PEAK:
        return "ORGE"
    return "A_VERIFIER"


def classify(series: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Probabilité d'orge de la parcelle (moyenne des observations), ou None."""
    model = load_model()
    if model is None:
        return None
    features = build_features(series)
    if features is None:
        return None

    observation_probabilities = model["model"].predict_proba(features)[:, 1]
    probability = float(np.mean(observation_probabilities))
    ndvi_oct = float(features["NDVI_oct"].iloc[0])
    doy_peak = float(features["doy_max_ndvi"].iloc[0])
    threshold = model["threshold"]

    return {
        "probability": probability,
        "is_barley": probability >= threshold,
        "threshold": threshold,
        "verdict": _verdict(probability, ndvi_oct, doy_peak),
        "n_features": len(model["features"]),
        "n_observations": int(len(features)),
        "ndvi_oct": round(ndvi_oct, 4),
        "doy_peak": round(doy_peak, 1),
        "model_version": model["summary"].get("production_model", "orge_rf"),
    }
