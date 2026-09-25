"""Classifieur orge / non-orge par Random Forest sur la série Sentinel-2 annuelle.

Intègre dans MALTO/AGRISAT le modèle entraîné hors-ligne (projet `modele_orge`,
pipeline sklearn : SimpleImputer median + RandomForestClassifier, sauvegardé dans
app/data/models/rf_orge/final_orge_model.pkl). Décrit la saison cultural par 48
caractéristiques spectro-temporelles des 4 indices NDVI / EVI / NDRE / NDMI,
reproduisant exactement training/src/feature_engineering.py :

- stats robustes de la série : mean, median, std, min, max, amp, q10, q90
- moyennes par fenêtre saisonnière (hiver doy<=59|>=300, printemps 60-151,
  été 152-243, automne 244-299)
- 12 stats par indice = 4 indices × 12 = 48 features (le RF final n'utilise
  volontairement pas les features dynamiques ni les bandes brutes)

Fiabilité : l'échantillon d'inférence = une saison (parcelle, année glissante)
sur 12 mois ; un mois nuageux (bandes à None) ne bloque pas la prédiction, les
NaN sont gérés par l'imputer médian embarqué dans le pipeline. Si la série est
trop courte (< 4 mois exploitables), classify() renvoie None et l'analyse garde
le verdict Sentinel-2 logistique ou la fusion CNN + agro.

Le modèle HF (andritonny) et le classifieur logistique restent en place : ce RF
est une source de verdict supplémentaire, la plus prioritaire quand elle est
disponible.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("agrisat.services.barley_rf_classifier")

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "rf_orge"
MODEL_PATH = MODEL_DIR / "final_orge_model.pkl"
FEATURES_PATH = MODEL_DIR / "final_feature_names.json"
SUMMARY_PATH = MODEL_DIR / "final_model_summary.json"

MIN_VALID_MONTHS = 4  # en dessous : pas de verdict (trop de NaN même imputés)

# Fenêtres saisonnières par jour de l'année (identiques à feature_engineering.py)
_SEASON_WINDOWS: dict[str, Any] = {
    "winter": lambda doy: (doy <= 59) | (doy >= 300),
    "spring": lambda doy: (doy >= 60) & (doy <= 151),
    "summer": lambda doy: (doy >= 152) & (doy <= 243),
    "autumn": lambda doy: (doy >= 244) & (doy <= 299),
}

_INDICES = ["NDVI", "EVI", "NDRE", "NDMI"]
_STATS = ["mean", "median", "std", "min", "max", "amp", "q10", "q90"]
_SEASONS = ["winter", "spring", "summer", "autumn"]


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any] | None:
    """Charge (une fois) le pipeline joblib + noms de features + seuil. None si absent."""
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        return {
            "model": model,
            "features": list(feature_names),
            "threshold": float(summary.get("threshold", 0.5)),
            "summary": summary,
        }
    except FileNotFoundError:
        logger.warning("Modèle RF orge absent (%s) : verdict par le classifieur Sentinel-2 logistique.", MODEL_PATH)
    except (OSError, ValueError) as error:
        logger.error("Modèle RF orge illisible (%s) : %s", MODEL_PATH, error)
    return None


def _stats_block(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Stats robustes d'une série 1D (NaN ignorés), identique à _basic_stats du training."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return {f"{prefix}_{k}": np.nan for k in _STATS}
    vmax, vmin = float(np.max(v)), float(np.min(v))
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_std": float(np.std(v)),
        f"{prefix}_min": vmin,
        f"{prefix}_max": vmax,
        f"{prefix}_amp": vmax - vmin,
        f"{prefix}_q10": float(np.quantile(v, 0.10)),
        f"{prefix}_q90": float(np.quantile(v, 0.90)),
    }


def _seasonal_means(dates: pd.Series, values: np.ndarray, prefix: str) -> dict[str, float]:
    """Moyennes par fenêtre saisonnière (NaN si fenêtre vide)."""
    doy = dates.dt.dayofyear.to_numpy()
    out: dict[str, float] = {}
    for season, window in _SEASON_WINDOWS.items():
        mask = window(doy) & ~np.isnan(values)
        out[f"{prefix}_{season}_mean"] = float(np.mean(values[mask])) if mask.any() else np.nan
    return out


def build_features(series: list[dict[str, Any]]) -> pd.DataFrame | None:
    """Construit le vecteur de 48 features depuis la série mensuelle de s2_time_series.

    series : [{date: 'YYYY-MM', NDVI, EVI, NDRE, NDMI, B2..B12}, ...]
    Retourne un DataFrame à 1 ligne (colonnes = noms attendus par le pipeline),
    ou None si moins de MIN_VALID_MONTHS mois ont au moins un indice exploitable.
    """
    if not series:
        return None
    frame = pd.DataFrame(series)
    if "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset="date")

    valid_months = 0
    features: dict[str, float] = {}
    for index in _INDICES:
        values = pd.to_numeric(frame.get(index), errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).sum() >= MIN_VALID_MONTHS:
            valid_months += 1
        features.update(_stats_block(values, index))
        features.update(_seasonal_means(frame["date"], values, index))

    if valid_months == 0:
        return None

    model = load_model()
    if model is None:
        return None
    return pd.DataFrame([features])[model["features"]]


def classify(series: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Probabilité d'orge de la parcelle sur la saison écoulée, ou None si le modèle
    ou la série sont insuffisants."""
    model = load_model()
    if model is None:
        return None
    features = build_features(series)
    if features is None:
        return None
    probability = float(model["model"].predict_proba(features)[:, 1][0])
    return {
        "probability": probability,
        "is_barley": probability >= model["threshold"],
        "threshold": model["threshold"],
        "n_features": len(model["features"]),
        "model_version": MODEL_DIR.name,
    }
