import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app import cache
from app.db import async_session_maker
from app.errors import parse_or_400
from app.models import FieldScanRun
from app.schemas import AnalyzeLatestQuery, AnalyzeSimpleRequest
from app.services.barley_detect_simple import analyze_fields_simple, save_simple_field_parcelles

logger = logging.getLogger("agrisat.routes.analyze_simple")

router = APIRouter()

FIELD_SCAN_VERSION = "v1-simple"
FIELD_SCAN_PROXIMITY_DEGREES = 0.01


@router.post("/api/analyze")
async def analyze_simple(request: Request) -> JSONResponse:
    body = await request.json()
    parsed = parse_or_400(AnalyzeSimpleRequest, body, "Coordonnées ou rayon invalides.")

    gdd_config = (
        {"baseTemperature": parsed.base_temperature, "threshold": parsed.threshold, "periodDays": parsed.period_days}
        if parsed.base_temperature is not None and parsed.threshold is not None and parsed.period_days is not None
        else None
    )

    result = await analyze_fields_simple(
        {
            "lat": parsed.latitude,
            "lng": parsed.longitude,
            "radiusM": parsed.radius,
            "confidenceThreshold": parsed.confidence_threshold,
            "minAreaHa": parsed.min_area_ha,
            "gddConfig": gdd_config,
        }
    )

    if result["features"]:
        await save_simple_field_parcelles(result["features"], result["warnings"])
        cache.invalidate()

    try:
        async with async_session_maker() as session:
            run = FieldScanRun(
                version=FIELD_SCAN_VERSION,
                center_lat=parsed.latitude,
                center_lng=parsed.longitude,
                radius_m=parsed.radius,
                image_date=result["imageDate"],
                image_age_days=result["imageAgeDays"],
                cloud_percentage=result["cloudPercentage"],
                confidence_threshold=result["confidenceThreshold"],
                min_area_ha=result["minAreaHa"],
                result_geojson=result,
                candidates_found=result["candidatesFound"],
                candidates_kept=result["candidatesClassified"],
                warnings=result["warnings"],
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)

        return JSONResponse(content={"analysisId": str(run.id), "analysisDate": run.created_at.isoformat(), **result})
    except Exception as error:  # noqa: BLE001
        logger.warning("analyze: persistence failed, returning result without analysisId: %s", error)
        result["warnings"].append("Résultat non sauvegardé : la base de données est temporairement indisponible.")
        return JSONResponse(content={"analysisId": None, "analysisDate": datetime.now(timezone.utc).isoformat(), **result})


@router.get("/api/analyze/latest")
async def analyze_latest(lat: str, lng: str) -> JSONResponse:
    parsed = parse_or_400(AnalyzeLatestQuery, {"lat": lat, "lng": lng}, "Coordonnées invalides.")

    async with async_session_maker() as session:
        stmt = (
            select(FieldScanRun)
            .where(
                FieldScanRun.version == FIELD_SCAN_VERSION,
                FieldScanRun.center_lat >= parsed.lat - FIELD_SCAN_PROXIMITY_DEGREES,
                FieldScanRun.center_lat <= parsed.lat + FIELD_SCAN_PROXIMITY_DEGREES,
                FieldScanRun.center_lng >= parsed.lng - FIELD_SCAN_PROXIMITY_DEGREES,
                FieldScanRun.center_lng <= parsed.lng + FIELD_SCAN_PROXIMITY_DEGREES,
            )
            .order_by(FieldScanRun.created_at.desc())
            .limit(1)
        )
        run = (await session.execute(stmt)).scalars().first()

    if run is None:
        return JSONResponse(status_code=404, content={"error": "Aucune analyse enregistrée pour cette zone."})

    result_geojson = run.result_geojson if isinstance(run.result_geojson, dict) else {}
    return JSONResponse(content={"analysisId": str(run.id), "analysisDate": run.created_at.isoformat(), **result_geojson})
