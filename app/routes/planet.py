import logging

import httpx
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import settings
from app.schemas import PlanetTileParams, PlanetTileQuery
from app.services.planet_service import get_planet_tile

logger = logging.getLogger("agrisat.routes.planet")

router = APIRouter()

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


@router.get("/api/planet/status")
async def planet_status() -> JSONResponse:
    if not settings.planet_api_key:
        return JSONResponse(
            status_code=500,
            content={"success": False, "configured": False, "message": "PLANET_API_KEY est absente du fichier .env"},
        )
    return JSONResponse(content={"success": True, "configured": True, "provider": "Planet", "message": "Planet API Key configurée."})


@router.get("/api/planet/series")
async def planet_series() -> JSONResponse:
    if not settings.planet_api_key:
        return JSONResponse(status_code=500, content={"success": False, "error": "PLANET_API_KEY non configurée."})

    try:
        response = await _get_client().get(
            "https://api.planet.com/basemaps/v1/series/",
            headers={"Authorization": f"api-key {settings.planet_api_key}", "Accept": "application/json"},
        )
        if response.status_code >= 400:
            logger.error("Planet series request failed: %s %s", response.status_code, response.text)
            return JSONResponse(status_code=response.status_code, content={"success": False, "error": "Planet API error", "details": response.text})
        return JSONResponse(content=response.json())
    except Exception as error:  # noqa: BLE001
        logger.error("Planet series request failed: %s", error)
        return JSONResponse(status_code=502, content={"success": False, "error": "Impossible de contacter Planet API."})


@router.get("/api/planet/tiles/{z}/{x}/{y_ext}")
async def planet_tile(z: str, x: str, y_ext: str, mosaic: str | None = None) -> Response:
    if not y_ext.endswith(".png"):
        return JSONResponse(status_code=400, content={"success": False, "error": "Paramètres de tuile Planet invalides."})
    y = y_ext[: -len(".png")]

    try:
        params = PlanetTileParams.model_validate({"z": z, "x": x, "y": y})
        query = PlanetTileQuery.model_validate({"mosaic": mosaic})
    except ValidationError:
        return JSONResponse(status_code=400, content={"success": False, "error": "Paramètres de tuile Planet invalides."})

    max_tile = 2**params.z
    if params.x >= max_tile or params.y >= max_tile:
        return JSONResponse(status_code=400, content={"success": False, "error": "Coordonnées XYZ invalides."})

    mosaic_name = query.mosaic or "planet_medres_normalized_analytic_2024-07_mosaic"

    try:
        tile = await get_planet_tile(params.z, params.x, params.y, mosaic_name)
        return Response(content=tile["buffer"], media_type=tile["contentType"], headers={"Cache-Control": "public, max-age=3600"})
    except Exception as error:  # noqa: BLE001
        logger.error("Planet tile fetch failed: %s", error)
        return JSONResponse(status_code=502, content={"success": False, "message": "Impossible de récupérer la tuile Planet"})
