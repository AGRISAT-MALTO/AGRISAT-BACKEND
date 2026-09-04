import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from sqlalchemy import delete as sa_delete

from app import cache
from app.db import async_session_maker
from app.errors import parse_or_400
from app.models import Parcelle
from app.schemas import DeleteParcelleParams, ParcelleCreate

logger = logging.getLogger("agrisat.routes.parcelles")

router = APIRouter()

# Champs exposés par l'API, identiques à ceux connus de Prisma (schema.prisma) — exclut
# volontairement `updated_at`, une colonne présente en base mais jamais déclarée côté Prisma
# et donc jamais renvoyée par l'ancien backend.
_PARCELLE_FIELDS = (
    "id", "label", "coordinates", "center_lat", "center_lng", "surface_ha",
    "culture_declared", "culture_detected", "ndvi_percentage", "ndre", "spectral_bands",
    "confidence", "verdict", "details", "saison", "soil_type", "risk_factors",
    "recommendations", "data_source", "owner_name", "notes", "time_series_s1",
    "time_series_s2", "time_series_rain", "estimated_planting_date", "estimated_harvest_date",
    "days_since_planting", "growth_stage", "planting_confidence", "evi", "savi", "ndwi",
    "agro_score", "hybrid_score", "cnn_prob_barley", "cnn_prob_non_barley", "created_at",
)


def _serialize(row: Parcelle) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in _PARCELLE_FIELDS:
        value = getattr(row, field)
        if isinstance(value, datetime):
            value = value.isoformat()
        else:
            value = str(value) if field == "id" else value
        out[field] = value
    return out


@router.get("/api/parcelles")
async def list_parcelles() -> JSONResponse:
    try:
        rows = await cache.load_parcelles()
        return JSONResponse(content=[_serialize(r) for r in rows])
    except Exception as error:  # noqa: BLE001
        logger.error("parcelles: read failed: %s", error)
        return JSONResponse(status_code=503, content={"error": "Les parcelles sont temporairement indisponibles."})


@router.post("/api/parcelles")
async def create_parcelle(request: Request) -> JSONResponse:
    body = await request.json()
    parsed = parse_or_400(ParcelleCreate, body, "Données de parcelle invalides")

    values = {
        "label": parsed.label,
        "coordinates": [p.model_dump() for p in parsed.coordinates],
        "center_lat": parsed.center_lat,
        "center_lng": parsed.center_lng,
        "surface_ha": parsed.surface_ha,
        "culture_declared": parsed.culture_declared,
        "culture_detected": parsed.culture_detected,
        "ndvi_percentage": parsed.ndvi_percentage,
        "ndre": parsed.ndre,
        "spectral_bands": parsed.spectral_bands,
        "confidence": parsed.confidence,
        "verdict": parsed.verdict,
        "details": parsed.details,
        "saison": parsed.saison,
        "soil_type": parsed.soil_type,
        "risk_factors": parsed.risk_factors,
        "recommendations": parsed.recommendations,
        "data_source": parsed.data_source,
        "owner_name": parsed.owner_name,
        "notes": parsed.notes,
        "time_series_s1": parsed.time_series_s1,
        "time_series_s2": parsed.time_series_s2,
        "time_series_rain": parsed.time_series_rain,
        "estimated_planting_date": parsed.estimated_planting_date,
        "estimated_harvest_date": parsed.estimated_harvest_date,
        "days_since_planting": int(parsed.days_since_planting) if parsed.days_since_planting is not None else None,
        "growth_stage": parsed.growth_stage,
        "planting_confidence": parsed.planting_confidence,
        "evi": parsed.evi,
        "savi": parsed.savi,
        "ndwi": parsed.ndwi,
        "agro_score": parsed.agro_score,
        "hybrid_score": parsed.hybrid_score,
        "cnn_prob_barley": parsed.cnn_prob_barley,
        "cnn_prob_non_barley": parsed.cnn_prob_non_barley,
    }

    async with async_session_maker() as session:
        row = Parcelle(**values)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        result = _serialize(row)

    cache.invalidate()
    return JSONResponse(status_code=201, content=result)


@router.delete("/api/parcelles/{parcelle_id}")
async def delete_parcelle(parcelle_id: str) -> Response:
    try:
        parsed = DeleteParcelleParams.model_validate({"id": parcelle_id})
    except ValidationError:
        return JSONResponse(status_code=400, content={"error": "Identifiant invalide"})

    async with async_session_maker() as session:
        result = await session.execute(sa_delete(Parcelle).where(Parcelle.id == uuid.UUID(parsed.id)))
        await session.commit()
        deleted_count = result.rowcount

    if deleted_count == 0:
        return JSONResponse(status_code=404, content={"error": "Parcelle introuvable"})

    cache.invalidate()
    return Response(status_code=204)
