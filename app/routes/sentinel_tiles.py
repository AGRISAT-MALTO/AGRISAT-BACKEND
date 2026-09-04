"""Route + logique de src/sentinel-tiles.ts (assez petit pour être fondu directement ici
plutôt que dans un module services/ séparé — aucun autre fichier n'importe de sentinel-tiles.ts)."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.schemas import SentinelTileParams, SentinelTileQuery
from app.services.analyze_parcel import (
    GeeValue,
    build_true_color_visualize_expression,
    compute_pixels_png,
    gee_call,
    gee_constant,
    get_gee_access_token,
    get_gee_project_id,
)

logger = logging.getLogger("agrisat.routes.sentinel_tiles")

router = APIRouter()

TRANSPARENT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

WEB_MERCATOR_EXTENT_METERS = 20_037_508.342789244
TILE_SIZE_PX = 256


class PixelGrid(dict):
    """widthPx, heightPx, originXMeters, originYMeters, scaleMeters."""


def tile_to_mercator_grid(z: int, x: int, y: int) -> PixelGrid:
    tile_size_meters = (2 * WEB_MERCATOR_EXTENT_METERS) / 2**z
    origin_x_meters = -WEB_MERCATOR_EXTENT_METERS + x * tile_size_meters
    origin_y_meters = WEB_MERCATOR_EXTENT_METERS - y * tile_size_meters
    return PixelGrid(widthPx=TILE_SIZE_PX, heightPx=TILE_SIZE_PX, originXMeters=origin_x_meters, originYMeters=origin_y_meters, scaleMeters=tile_size_meters / TILE_SIZE_PX)


def _build_tile_region(grid: PixelGrid) -> GeeValue:
    west = grid["originXMeters"]
    north = grid["originYMeters"]
    east = grid["originXMeters"] + grid["widthPx"] * grid["scaleMeters"]
    south = grid["originYMeters"] - grid["heightPx"] * grid["scaleMeters"]
    return gee_call(
        "GeometryConstructors.Rectangle",
        {"coordinates": gee_constant([west, south, east, north]), "crs": gee_call("Projection", {"crs": gee_constant("EPSG:3857")}), "geodesic": gee_constant(False)},
    )


TILE_CACHE_MAX_ENTRIES = 500
TILE_CACHE_TTL_S = 30 * 60.0
_tile_cache: dict[str, dict[str, Any]] = {}


def _cache_get(key: str) -> bytes | None:
    entry = _tile_cache.get(key)
    if entry is None:
        return None
    if entry["expires_at"] <= time.monotonic():
        del _tile_cache[key]
        return None
    return entry["bytes"]


def _cache_set(key: str, data: bytes) -> None:
    if len(_tile_cache) >= TILE_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_tile_cache), None)
        if oldest_key is not None:
            del _tile_cache[oldest_key]
    _tile_cache[key] = {"expires_at": time.monotonic() + TILE_CACHE_TTL_S, "bytes": data}


async def _fetch_sentinel2_tile_png(access_token: str, project_id: str, z: int, x: int, y: int, image_timestamp_ms: float) -> bytes | None:
    """Sélectionne l'image COPERNICUS/S2_SR_HARMONIZED dont system:time_start correspond
    exactement à image_timestamp_ms (la même image que celle choisie pour l'analyse).
    Retourne None si aucune image ne couvre cette tuile à cet instant précis."""
    cache_key = f"{z}-{x}-{y}-{image_timestamp_ms}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    grid = tile_to_mercator_grid(z, x, y)
    values: dict[str, GeeValue] = {}

    def ref(name: str) -> GeeValue:
        return {"valueReference": name}

    start_iso = _iso_from_ms(image_timestamp_ms)
    end_iso = _iso_from_ms(image_timestamp_ms + 1_000)

    values["region"] = _build_tile_region(grid)
    values["intersects"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": ref("region")})})
    values["dateRange"] = gee_call("Filter.dateRangeContains", {"leftValue": gee_call("DateRange", {"start": gee_constant(start_iso), "end": gee_constant(end_iso)}), "rightField": gee_constant("system:time_start")})
    values["raw"] = gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")})
    values["byRegion"] = gee_call("Collection.filter", {"collection": ref("raw"), "filter": ref("intersects")})
    values["byDate"] = gee_call("Collection.filter", {"collection": ref("byRegion"), "filter": ref("dateRange")})
    values["image"] = gee_call("Collection.first", {"collection": ref("byDate")})
    values["visualized"] = build_true_color_visualize_expression(ref("image"))

    try:
        data = await compute_pixels_png(access_token, project_id, {"result": "visualized", "values": values}, grid)
        _cache_set(cache_key, data)
        return data
    except Exception as error:  # noqa: BLE001
        logger.warning("[sentinel-tiles] Tuile %s/%s/%s indisponible : %s", z, x, y, error)
        return None


def _iso_from_ms(ms: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ms % 1000):03d}Z"


@router.get("/api/sentinel-tiles/{z}/{x}/{y}")
async def sentinel_tile(z: str, x: str, y: str, request: Request) -> Response:
    try:
        params = SentinelTileParams.model_validate({"z": z, "x": x, "y": y})
        query = SentinelTileQuery.model_validate({"imageTimestampMs": request.query_params.get("imageTimestampMs")})
    except ValidationError:
        return JSONResponse(status_code=400, content={"error": "Coordonnées de tuile ou timestamp d'image invalides."})

    try:
        access_token = await get_gee_access_token()
        project_id = get_gee_project_id()
        png_bytes = await _fetch_sentinel2_tile_png(access_token, project_id, params.z, params.x, params.y, query.image_timestamp_ms)

        if png_bytes is None:
            return Response(content=TRANSPARENT_PNG, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

        return Response(content=png_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    except Exception as error:  # noqa: BLE001
        logger.warning("sentinel-tiles: tuile indisponible: %s", error)
        return Response(content=TRANSPARENT_PNG, media_type="image/png", headers={"Cache-Control": "public, max-age=60"})
