"""
Route /api/zone-crops : cultures des parcelles d'une zone d'analyse.

Alimente l'ecran de MALTO ou l'utilisateur pose un marqueur, choisit un rayon
et veut voir les parcelles agricoles avec leur culture.

La logique est dans app/services/zone_crops.py, qui documente aussi les limites
du jeu de donnees (echantillonnage JECAM, fiabilite des predictions, orge non
detectable). Ces limites sont renvoyees dans `warnings` : affichez-les.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.zone_crops import MAX_RADIUS_M, ZoneCropsError, zone_crops

logger = logging.getLogger("agrisat.routes.zone_crops")

router = APIRouter()


class ZoneCropsRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude du centre, en degres.")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude du centre.")
    radius_m: float = Field(1000, gt=0, le=MAX_RADIUS_M,
                            description="Rayon d'analyse en metres (50 a 20000).")
    min_confidence: float = Field(0.0, ge=0, le=1,
                                  description="Seuil de confiance des predictions.")
    crops: list[str] | None = Field(None, description="Filtrer sur certaines cultures.")
    parcels: list[dict[str, Any]] | None = Field(
        None,
        description="Parcelles deja tracees a l'ecran (Features GeoJSON de "
                    "/api/zone-parcels). Fournies, la culture est restituee par "
                    "parcelle plutot que par ilot JECAM.",
    )


@router.post("/api/zone-crops")
async def post_zone_crops(payload: ZoneCropsRequest) -> JSONResponse:
    """Parcelles et cultures d'une zone circulaire (FeatureCollection GeoJSON)."""
    try:
        result = zone_crops(
            latitude=payload.latitude,
            longitude=payload.longitude,
            radius_m=payload.radius_m,
            min_confidence=payload.min_confidence,
            crops=payload.crops,
            parcels=payload.parcels,
        )
    except ZoneCropsError as exc:
        logger.warning("zone-crops refuse : %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:  # pragma: no cover
        logger.exception("zone-crops en erreur")
        return JSONResponse(status_code=500, content={"detail": f"Erreur interne : {exc}"})
    return JSONResponse(content=result)


@router.get("/api/zone-crops")
async def get_zone_crops(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(1000, gt=0, le=MAX_RADIUS_M),
    min_confidence: float = Query(0.0, ge=0, le=1),
) -> JSONResponse:
    """Variante GET, pratique pour tester depuis un navigateur."""
    return await post_zone_crops(
        ZoneCropsRequest(latitude=lat, longitude=lon, radius_m=radius_m,
                         min_confidence=min_confidence)
    )
