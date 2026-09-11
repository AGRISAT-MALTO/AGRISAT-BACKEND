"""
Route /api/zone-parcels : delimitation automatique des parcelles d'une zone.

Source actuelle : modele JECAM sur image THR, via le service de tracage
(app/services/vhr_parcels.py). Fields of The World n'est plus utilise pour le
moment ; son service (app/services/zone_parcels.py) reste en place pour
/api/all-parcels. La reponse porte `attribution` et `warnings` (fiabilite selon
la taille des parcelles) : l'interface doit afficher les deux.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.vhr_parcels import (
    MAX_REQUEST_RADIUS_M,
    VhrParcelsError,
    VhrServiceUnavailable,
    zone_parcels_vhr,
)
from app.services.zone_parcels import all_zone_parcels

logger = logging.getLogger("agrisat.routes.zone_parcels")

router = APIRouter()


class ZoneParcelsRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(800, gt=0, le=MAX_REQUEST_RADIUS_M,
                            description="Ramene a 1 km par le service de tracage.")
    min_confidence: float = Field(0.0, ge=0, le=100, description="Score du modele, de 0 a 100.")
    min_area_ha: float = Field(0.0, ge=0, description="Surface minimale en hectares.")


@router.post("/api/zone-parcels")
async def post_zone_parcels(payload: ZoneParcelsRequest) -> JSONResponse:
    """Parcelles delimitees automatiquement dans un cercle (FeatureCollection)."""
    try:
        result = await zone_parcels_vhr(payload.latitude, payload.longitude, payload.radius_m,
                                        payload.min_confidence, payload.min_area_ha)
    except VhrServiceUnavailable as exc:
        logger.warning("zone-parcels indisponible : %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except VhrParcelsError as exc:
        logger.warning("zone-parcels refuse : %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception as exc:  # pragma: no cover
        logger.exception("zone-parcels en erreur")
        return JSONResponse(status_code=500, content={"detail": f"Erreur interne : {exc}"})
    return JSONResponse(content=result)


@router.get("/api/zone-parcels")
async def get_zone_parcels(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(800, gt=0, le=MAX_REQUEST_RADIUS_M),
    min_confidence: float = Query(0.0, ge=0, le=100),
    min_area_ha: float = Query(0.0, ge=0),
) -> JSONResponse:
    """Variante GET, pour tester depuis un navigateur."""
    return await post_zone_parcels(ZoneParcelsRequest(
        latitude=lat, longitude=lon, radius_m=radius_m,
        min_confidence=min_confidence, min_area_ha=min_area_ha))


@router.get("/api/all-parcels")
async def get_all_parcels() -> JSONResponse:
    """Retourne toutes les parcelles FTW disponibles dans la region extraite."""
    try:
        return JSONResponse(content=all_zone_parcels())
    except Exception as exc:  # pragma: no cover
        logger.exception("all-parcels en erreur")
        return JSONResponse(status_code=500, content={"detail": f"Erreur interne : {exc}"})
