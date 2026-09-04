"""Port 1:1 de src/zoning.ts : zonage NDVI (VRA, 3 zones) d'une parcelle."""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.config import settings
from app.services.analyze_parcel import (
    GEE_COMPUTE_TIMEOUT_S,
    add_spectral_index,
    fetch_with_retry,
    gee_call,
    gee_constant,
    get_gee_access_token,
    get_gee_project_id,
    js_round,
    normalize_polygon,
    polygon_coordinates,
)
from app.services.field_watershed import lng_lat_to_mercator_meters, load_npy_band, mercator_meters_to_lng_lat

logger = logging.getLogger("agrisat.zoning")

GOOGLE_MAPS_API_KEY = settings.google_maps_api_key
ZONING_THUMBNAIL_SIZE = "320x200"

ZONE_DEFINITIONS = [
    {"key": "high", "label": "Zone de végétation forte", "ndviMin": 0.75, "ndviMax": 1.01, "color": "#2eb860"},
    {"key": "mid", "label": "Zone de végétation intermédiaire", "ndviMin": 0.35, "ndviMax": 0.75, "color": "#86ac39"},
    {"key": "low", "label": "Zone de végétation faible", "ndviMin": -1, "ndviMax": 0.35, "color": "#bf4040"},
]

ZONING_TARGET_SCALE_M = 10
ZONING_MAX_DIMENSION_PX = 400
ZONING_SEASON_DAYS = 180


async def _capture_zoning_thumbnail(polygon: list[dict[str, float]]) -> str | None:
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY.startswith("VOTRE_"):
        return None
    try:
        path = "|".join(f"{p['lat']},{p['lng']}" for p in polygon)
        params = {
            "size": ZONING_THUMBNAIL_SIZE,
            "maptype": "satellite",
            "path": f"color:0xfbbf24ff|weight:2|fillcolor:0x00000000|{path}",
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = await fetch_with_retry("GET", "https://maps.googleapis.com/maps/api/staticmap", params=params, timeout_s=15.0)
        if resp.status_code >= 400:
            return None
        return base64.b64encode(resp.content).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _build_zoning_ndvi_expression(polygon: list[dict[str, float]], start_date: str, end_date: str) -> dict:
    values: dict[str, Any] = {}

    def ref(name: str) -> dict:
        return {"valueReference": name}

    values["region"] = gee_call("GeometryConstructors.Polygon", {"coordinates": gee_constant(polygon_coordinates(polygon))})
    values["intersectsRegion"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": ref("region")})})
    values["dateRange"] = gee_call("Filter.dateRangeContains", {"leftValue": gee_call("DateRange", {"start": gee_constant(start_date), "end": gee_constant(end_date)}), "rightField": gee_constant("system:time_start")})
    values["lowCloudCover"] = gee_call("Filter.lessThan", {"leftField": gee_constant("CLOUDY_PIXEL_PERCENTAGE"), "rightValue": gee_constant(35)})
    values["collectionByRegion"] = gee_call("Collection.filter", {"collection": gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")}), "filter": ref("intersectsRegion")})
    values["collectionByDate"] = gee_call("Collection.filter", {"collection": ref("collectionByRegion"), "filter": ref("dateRange")})
    values["collection"] = gee_call("Collection.filter", {"collection": ref("collectionByDate"), "filter": ref("lowCloudCover")})
    values["composite"] = gee_call("reduce.median", {"collection": ref("collection")})
    values["withNdvi"] = add_spectral_index(ref("composite"), "NDVI", ("B8", "B4"))
    values["ndviBand"] = gee_call("Image.select", {"input": ref("withNdvi"), "bandSelectors": gee_constant(["NDVI"])})

    return {"result": "ndviBand", "values": values}


class PixelGrid(dict):
    """widthPx, heightPx, scaleMeters, originXMeters, originYMeters."""


def _compute_zoning_grid(polygon: list[dict[str, float]]) -> PixelGrid:
    corners = [lng_lat_to_mercator_meters(p["lng"], p["lat"]) for p in polygon]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad = ZONING_TARGET_SCALE_M
    padded_min_x, padded_max_x = min_x - pad, max_x + pad
    padded_min_y, padded_max_y = min_y - pad, max_y + pad

    raw_width_px = (padded_max_x - padded_min_x) / ZONING_TARGET_SCALE_M
    raw_height_px = (padded_max_y - padded_min_y) / ZONING_TARGET_SCALE_M
    scale_factor = max(1.0, max(raw_width_px, raw_height_px) / ZONING_MAX_DIMENSION_PX)
    scale_meters = ZONING_TARGET_SCALE_M * scale_factor

    return PixelGrid(
        widthPx=max(2, js_round((padded_max_x - padded_min_x) / scale_meters)),
        heightPx=max(2, js_round((padded_max_y - padded_min_y) / scale_meters)),
        scaleMeters=scale_meters,
        originXMeters=padded_min_x,
        originYMeters=padded_max_y,
    )


async def _fetch_ndvi_raster(access_token: str, project_id: str, expression: dict, grid: PixelGrid) -> tuple[np.ndarray, int, int]:
    url = f"https://earthengine.googleapis.com/v1/projects/{project_id}/image:computePixels"
    body = {
        "expression": expression,
        "fileFormat": "NPY",
        "bandIds": ["NDVI"],
        "grid": {
            "dimensions": {"width": grid["widthPx"], "height": grid["heightPx"]},
            "affineTransform": {
                "scaleX": grid["scaleMeters"],
                "shearX": 0,
                "translateX": grid["originXMeters"],
                "shearY": 0,
                "scaleY": -grid["scaleMeters"],
                "translateY": grid["originYMeters"],
            },
            "crsCode": "EPSG:3857",
        },
    }
    response = await fetch_with_retry(
        "POST", url, headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}, json=body, timeout_s=GEE_COMPUTE_TIMEOUT_S
    )
    if response.status_code >= 400:
        raise RuntimeError(f"GEE computePixels (NDVI) : erreur {response.status_code} {response.text[:300]}")
    data, width, height = load_npy_band(response.content, "NDVI")
    return data, width, height


async def _fetch_ndvi_raster_resilient(access_token: str, project_id: str, expression: dict, grid: PixelGrid) -> tuple[np.ndarray, int, int]:
    """fetchNdviRaster échoue parfois avec une erreur réseau générique sous charge (plusieurs
    zonings déclenchés coup sur coup) malgré les 3 tentatives internes de fetch_with_retry —
    une tentative supplémentaire après un délai plus long absorbe ces creux transitoires."""
    try:
        return await _fetch_ndvi_raster(access_token, project_id, expression, grid)
    except Exception as error:  # noqa: BLE001
        logger.warning("[zoning] fetchNdviRaster a échoué, nouvelle tentative dans 1.5s : %s", error)
        await asyncio.sleep(1.5)
        return await _fetch_ndvi_raster(access_token, project_id, expression, grid)


def _point_in_polygon(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < ((xj - xi) * (y - yi)) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _classify_zoning_raster(polygon: list[dict[str, float]], grid: PixelGrid, data: np.ndarray, width: int, height: int) -> dict[str, Any]:
    ring = [lng_lat_to_mercator_meters(p["lng"], p["lat"]) for p in polygon]
    classes = np.full(width * height, 255, dtype=np.uint8)
    sums = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]

    for row in range(height):
        y = grid["originYMeters"] - (row + 0.5) * grid["scaleMeters"]
        for col in range(width):
            index = row * width + col
            ndvi = float(data[row, col])
            if not np.isfinite(ndvi):
                continue
            x = grid["originXMeters"] + (col + 0.5) * grid["scaleMeters"]
            if not _point_in_polygon(x, y, ring):
                continue
            zone_index = -1
            for i, zone in enumerate(ZONE_DEFINITIONS):
                if zone["ndviMin"] <= ndvi < zone["ndviMax"]:
                    zone_index = i
                    break
            resolved_index = 2 if zone_index == -1 else zone_index
            classes[index] = resolved_index
            sums[resolved_index] += ndvi
            counts[resolved_index] += 1

    pixel_area_ha = (grid["scaleMeters"] * grid["scaleMeters"]) / 10_000
    zones = []
    for i, zone in enumerate(ZONE_DEFINITIONS):
        zones.append(
            {
                "key": zone["key"],
                "label": zone["label"],
                "ndviMin": zone["ndviMin"],
                "ndviMax": min(zone["ndviMax"], 1),
                "color": zone["color"],
                "avgNdvi": js_round((sums[i] / counts[i]) * 100) / 100 if counts[i] > 0 else None,
                "areaHa": js_round(counts[i] * pixel_area_ha * 100) / 100,
            }
        )
    total_area_ha = js_round(sum(z["areaHa"] for z in zones) * 100) / 100

    top_left_lng, top_left_lat = mercator_meters_to_lng_lat(grid["originXMeters"], grid["originYMeters"])
    bottom_right_lng, bottom_right_lat = mercator_meters_to_lng_lat(
        grid["originXMeters"] + width * grid["scaleMeters"], grid["originYMeters"] - height * grid["scaleMeters"]
    )

    return {
        "classes": classes.tobytes(),
        "zones": zones,
        "totalAreaHa": total_area_ha,
        "bounds": {"south": bottom_right_lat, "west": top_left_lng, "north": top_left_lat, "east": bottom_right_lng},
    }


async def compute_zoning(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        polygon = body.get("polygon")
        parcel_polygon = normalize_polygon(polygon)
        if parcel_polygon is None:
            return 400, {"error": "Le contour réel de la parcelle est requis (au moins 3 points valides)."}

        service_account_json = settings.gee_service_account_key
        if not service_account_json or service_account_json.startswith("VOTRE_"):
            return 502, {"error": "GEE_SERVICE_ACCOUNT_KEY n'est pas configurée : le zoning nécessite un accès Earth Engine réel."}

        try:
            access_token = await get_gee_access_token()
        except Exception as error:  # noqa: BLE001
            return 502, {"error": f"GEE indisponible : {error or 'authentification impossible'}"}
        project_id = get_gee_project_id()

        now = datetime.now(timezone.utc)
        end_date = now.date().isoformat()
        start_date = (now - timedelta(days=ZONING_SEASON_DAYS)).date().isoformat()

        expression = _build_zoning_ndvi_expression(parcel_polygon, start_date, end_date)
        grid = _compute_zoning_grid(parcel_polygon)

        (data, width, height), thumbnail = await asyncio.gather(
            _fetch_ndvi_raster_resilient(access_token, project_id, expression, grid),
            _capture_zoning_thumbnail(parcel_polygon),
        )

        result = _classify_zoning_raster(parcel_polygon, grid, data, width, height)

        return 200, {
            "thumbnail": thumbnail,
            "bounds": result["bounds"],
            "widthPx": width,
            "heightPx": height,
            "classes": base64.b64encode(result["classes"]).decode("ascii"),
            "zones": result["zones"],
            "totalAreaHa": result["totalAreaHa"],
            "imageStartDate": start_date,
            "imageEndDate": end_date,
        }
    except Exception as error:  # noqa: BLE001
        logger.error("zoning error: %s", error, exc_info=error)
        return 500, {"error": str(error) or "Unknown error"}
