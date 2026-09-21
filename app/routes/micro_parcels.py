from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app import cache
from app.db import async_session_maker
from app.errors import parse_or_400
from app.models import FieldScanRun
from app.schemas import AnalyzeLatestQuery, MicroParcelsRequest
from app.services.micro_parcels import analyze_micro_parcels

router = APIRouter()

FIELD_SCAN_VERSION_MICRO = "v1-micro"
FIELD_SCAN_PROXIMITY_DEGREES = 0.01


@router.post("/api/micro-parcels")
async def micro_parcels_route(request: Request) -> JSONResponse:
    body = await request.json()
    parsed = parse_or_400(MicroParcelsRequest, body, "Coordonnées invalides.")

    gdd_config = (
        {"baseTemperature": parsed.base_temperature, "threshold": parsed.threshold, "periodDays": parsed.period_days}
        if parsed.base_temperature is not None and parsed.threshold is not None and parsed.period_days is not None
        else None
    )

    result = await analyze_micro_parcels(
        {
            "lat": parsed.latitude,
            "lng": parsed.longitude,
            "radiusM": parsed.radius,
            "confidenceThreshold": parsed.confidence_threshold,
            "gddConfig": gdd_config,
            "sources": parsed.sources,
        }
    )

    # Historique des runs (dashboard + restauration carte) : même table que
    # l'analyse simple, version distincte `v1-micro`.
    analysis_id: str | None = None
    analysis_date: str | None = None
    try:
        from datetime import datetime, timezone

        async with async_session_maker() as session:
            run = FieldScanRun(
                version=FIELD_SCAN_VERSION_MICRO,
                center_lat=parsed.latitude,
                center_lng=parsed.longitude,
                radius_m=parsed.radius,
                image_date=None,
                image_age_days=None,
                cloud_percentage=None,
                confidence_threshold=result.get("confidenceThreshold") or 0.7,
                min_area_ha=0.03,
                result_geojson=result,
                candidates_found=result.get("segmentation", {}).get("microParcels", 0),
                candidates_kept=len(result.get("features", [])),
                warnings=result.get("warnings", []),
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            analysis_id, analysis_date = str(run.id), run.created_at.isoformat()
    except Exception as error:  # noqa: BLE001
        import logging

        logging.getLogger("agrisat.routes.micro_parcels").warning("micro-parcels: persistence failed: %s", error)
        result["warnings"].append("Résultat non historisé : la base de données est temporairement indisponible.")

    cache.invalidate()
    return JSONResponse(content={"analysisId": analysis_id, "analysisDate": analysis_date, **result})


@router.get("/api/micro-parcels/latest")
async def micro_parcels_latest(lat: str, lng: str) -> JSONResponse:
    parsed = parse_or_400(AnalyzeLatestQuery, {"lat": lat, "lng": lng}, "Coordonnées invalides.")

    async with async_session_maker() as session:
        # Filtre de proximité explicite (lat ET lng) :
        stmt = (
            select(FieldScanRun)
            .where(
                FieldScanRun.version == FIELD_SCAN_VERSION_MICRO,
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
        return JSONResponse(status_code=404, content={"error": "Aucune analyse micro-parcelles enregistrée pour cette zone."})

    result_geojson = run.result_geojson if isinstance(run.result_geojson, dict) else {}
    return JSONResponse(content={"analysisId": str(run.id), "analysisDate": run.created_at.isoformat(), **result_geojson})
