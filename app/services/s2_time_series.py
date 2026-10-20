"""Séries temporelles Sentinel-2 mensuelles par parcelle, pour le classifieur RF orge.

Contrairement à `_fetch_time_series` de analyze_parcel.py (NDVI/S1 pour l'affichage),
ce module extrait les 10 bandes de réflectance + les 4 indices du dataset
d'entraînement (NDVI, EVI, NDRE, NDMI) sur les 12 derniers mois, via un appel GEE
`value:compute` par mois (médiane mensuelle + reduceRegion mean) réutilisant
l'expression éprouvée `_build_s2_expression` de analyze_parcel.py.

Conventions identiques au dataset d'entraînement (training/src) :
- bandes Sentinel-2 en réflectance de surface 0..1 (COPERNICUS/S2_SR_HARMONIZED,
  comme partout dans le backend),
- NDVI  = (B8 - B4) / (B8 + B4)
- EVI   = 2.5 (B8 - B4) / (B8 + 6 B4 - 7.5 B2 + 1)
- NDRE  = (B8 - B5) / (B8 + B5)
- NDMI  = (B8 - B11) / (B8 + B11)
- valeur sentinelle : indice == -1 et bande == 0 sont des artefacts d'extraction
  -> NaN (aligné sur preprocessing.clean_values) ; à l'inférence on renvoie None
  et le modèle final (SimpleImputer median) les gère.
"""

from __future__ import annotations

import calendar
import logging
from datetime import datetime
from typing import Any

from app.services.analyze_parcel import (
    _build_s2_expression,
    call_gee_compute,
    map_with_concurrency,
)

logger = logging.getLogger("agrisat.services.s2_time_series")

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDICES = ["NDVI", "EVI", "NDRE", "NDMI"]
N_MONTHS = 12  # 12 mois glissants : couvre un cycle cultural complet
S2_MONTHLY_CONCURRENCY = 3


def monthly_ranges(num_months: int = N_MONTHS, today: datetime | None = None) -> list[dict[str, str]]:
    """Fenêtres mensuelles glissantes (la plus ancienne en premier), même logique que
    get_monthly_ranges de analyze_parcel.py : chaque entrée couvre 1 mois calendaire."""
    now = today or datetime.now()
    total = now.year * 12 + (now.month - 1)  # index absolu du mois courant (0-based)
    ranges: list[dict[str, str]] = []
    for i in range(num_months - 1, -1, -1):
        t = total - i
        y, m0 = divmod(t, 12)  # m0 = mois 0-based
        start = f"{y:04d}-{m0 + 1:02d}-01"
        last_day = calendar.monthrange(y, m0 + 1)[1]
        end = f"{y:04d}-{m0 + 1:02d}-{last_day:02d}"
        ranges.append({"label": start[:7], "start": start, "end": end})
    return ranges


def parse_monthly_bands(results: list[dict[str, Any] | None], months: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Transforme les résultats GEE mensuels (`{B2: x, ...}` — le reducer médiane
    garde les noms de bandes) en lignes `{date, bandes, indices}`.

    - valeurs GEE en réflectance 0..10000 -> ramenées en 0..1 (échelle du dataset) ;
    - artefacts exacts (bande == 0, indice == -1) -> None, comme en training ;
    - indices recalculés depuis les bandes médianes du mois (formules training).
    """
    rows: list[dict[str, Any]] = []
    for month, result in zip(months, results):
        band_values: dict[str, float | None] = {}
        for band in S2_BANDS:
            raw = None
            if isinstance(result, dict):
                # reduce.median garde les noms de bandes ; formes suffixées acceptées par tolérance
                # (mêmes clés de repli que _parse_s2_bands de analyze_parcel.py).
                raw = result.get(band) or result.get(f"{band}_mean") or result.get(f"{band}_median")
            if isinstance(raw, (int, float)) and raw == raw:  # rejette NaN
                value = float(raw) / 10_000.0
                band_values[band] = None if value == 0.0 else value  # artefact bande == 0
            else:
                band_values[band] = None
        rows.append({"date": month["label"], **_indices_from_bands(band_values)})
    return rows


def _index(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or (numerator + denominator) == 0:
        return None
    value = (numerator - denominator) / (numerator + denominator)
    return None if value == -1.0 else value  # artefact indice == -1


def _indices_from_bands(bands: dict[str, float | None]) -> dict[str, float | None]:
    """NDVI / EVI / NDRE / NDMI recalculés des bandes médianes (échelle 0..1)."""
    b2, b4, b5, b8, b11 = bands.get("B2"), bands.get("B4"), bands.get("B5"), bands.get("B8"), bands.get("B11")

    ndvi = _index(b8, b4)
    ndre = _index(b8, b5)
    ndmi = _index(b8, b11)

    evi = None
    if b8 is not None and b4 is not None and b2 is not None:
        denom = b8 + 6 * b4 - 7.5 * b2 + 1
        if denom != 0:
            evi = 2.5 * (b8 - b4) / denom

    return {"NDVI": ndvi, "EVI": evi, "NDRE": ndre, "NDMI": ndmi, **bands}


async def fetch_s2_monthly_series(access_token: str, project_id: str, polygon: Any) -> list[dict[str, Any]]:
    """Série mensuelle S2 (bandes + indices) de la parcelle sur les 12 derniers mois.

    Un appel GEE par mois (même expression que le snapshot courant, éprouvée en
    production) ; lève RuntimeError si tous les calculs échouent (l'appelant décide
    du repli), renvoie des mois à None si certains échouent ou si aucune image
    n'est disponible (gérés par l'imputer du modèle).
    """
    months = monthly_ranges()

    async def month_mapper(month: dict[str, str]) -> dict[str, Any] | None:
        return await call_gee_compute(
            access_token,
            project_id,
            _build_s2_expression(month["start"], month["end"], polygon),
        )

    results = await map_with_concurrency(months, S2_MONTHLY_CONCURRENCY, month_mapper)
    if all(result is None for result in results):
        raise RuntimeError("Calcul GEE de la série Sentinel-2 mensuelle impossible.")
    rows = parse_monthly_bands(results, months)
    if not rows:
        logger.warning("Série Sentinel-2 mensuelle vide pour la parcelle.")
    return rows
