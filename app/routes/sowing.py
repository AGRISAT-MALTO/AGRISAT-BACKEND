from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.errors import parse_or_400
from app.schemas import SowingDateRequest
from app.services.sowing import (
    estimate_sowing_from_ndvi,
    estimate_sowing_from_stage,
)

router = APIRouter()


@router.post("/api/sowing-date")
async def sowing_date_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(error) or "Unknown error"})
    parsed = parse_or_400(SowingDateRequest, body, "Paramètres de semis invalides.")

    # Priorité : stade observé > série NDVI > fenêtre agro (gérée dans le service).
    if parsed.zadoks_code is not None or parsed.st_target is not None:
        result = await estimate_sowing_from_stage(
            parsed.lat,
            parsed.lng,
            observation_iso=parsed.observation_date,
            zadoks_code=parsed.zadoks_code,
            st_target=parsed.st_target,
            history_days=parsed.history_days,
        )
        return JSONResponse(status_code=200, content=result)

    if parsed.s2 is not None:
        result = await estimate_sowing_from_ndvi(parsed.lat, parsed.lng, parsed.s2)
        return JSONResponse(status_code=200, content=result)

    return JSONResponse(
        status_code=400,
        content={"error": "Fournir zadoksCode ou stTarget, ou une série s2 [{date, ndvi}]."},
    )
