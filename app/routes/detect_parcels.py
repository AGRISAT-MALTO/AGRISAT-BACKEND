import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app import cache
from app.errors import parse_or_400
from app.schemas import AutomaticDetectionRequest
from app.services.automatic_parcels import detect_automatic_parcels

logger = logging.getLogger("agrisat.routes.detect_parcels")

router = APIRouter()


def _to_service_request(parsed: AutomaticDetectionRequest) -> dict:
    return {
        "lat": parsed.lat,
        "lng": parsed.lng,
        "radiusKm": parsed.radius_km,
        "baseTemperature": parsed.base_temperature,
        "threshold": parsed.threshold,
        "periodDays": parsed.period_days,
    }


@router.post("/api/detect-parcels-fallback")
async def detect_parcels_fallback(request: Request) -> JSONResponse:
    body = await request.json()
    parsed = parse_or_400(AutomaticDetectionRequest, body, "Coordonnées ou rayon invalides.")

    result = await detect_automatic_parcels(_to_service_request(parsed))
    cache.invalidate()
    return JSONResponse(content=result)


@router.post("/api/detect-parcels")
async def detect_parcels(request: Request) -> JSONResponse:
    body = await request.json()
    parsed = parse_or_400(AutomaticDetectionRequest, body, "Coordonnées ou rayon invalides.")

    try:
        result = await detect_automatic_parcels(_to_service_request(parsed))
        cache.invalidate()
        return JSONResponse(content=result)
    except Exception as error:  # noqa: BLE001
        # detect_automatic_parcels gère déjà ses propres échecs en interne (voir son propre
        # try/except) et ne devrait normalement jamais lever — ce bloc est un filet de
        # sécurité supplémentaire, miroir du comportement de repli de server.ts.
        logger.warning("detect-parcels: discovery failed, returning fallback notice: %s", error)
        message = str(error) or "Le service de recherche des parcelles est indisponible."
        lower_message = message.lower()
        notice = (
            "La base des parcelles agricoles est indisponible. Essayez un rayon plus large ou vérifiez la configuration de la base de données."
            if "database" in message or "parcelles" in lower_message or "database" in lower_message
            else message
        )
        return JSONResponse(
            content={
                "center": {"lat": parsed.lat, "lng": parsed.lng},
                "radius_km": parsed.radius_km,
                "base_temperature": parsed.base_temperature,
                "threshold": parsed.threshold,
                "period_days": parsed.period_days,
                "candidates_found": 0,
                "analyzed_count": 0,
                "parcels": [],
                "notice": notice,
            }
        )
