"""Port 1:1 de src/automatic-parcels.ts : détection automatique de parcelles autour d'un
point (cascade OSM Overpass + base -> modèle U-Net -> watershed GEE -> SNIC GEE -> cellules
satellite), et calcul des degrés-jours de croissance (GDD, Open-Meteo).

Fidélité : le clipping Sutherland-Hodgman (clip_polygon_to_radius), la compacité de
Polsby-Popper (is_plausible_field_shape), et toute la géométrie locale (offset_point,
distance_between_points, radius_polygon) sont portés littéralement — ce sont ces fonctions
qui déterminent la forme finale des parcelles renvoyées."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypeVar

import httpx
import numpy as np
from sqlalchemy import select

from app.config import settings
from app.db import async_session_maker
from app.models import Parcelle
from app.services.analyze_parcel import (
    GEE_COMPUTE_TIMEOUT_S,
    add_bare_soil_index,
    add_spectral_index,
    analyze_parcel as _analyze_parcel_handler,
    approximate_polygon_area_m2,
    call_gee_compute_raw,
    extract_latlng_from_geometry,
    fetch_with_retry,
    gee_call,
    gee_constant,
    gee_image_constant,
    get_gee_access_token,
    get_gee_project_id,
    is_gee_feature_collection,
    js_round,
    map_with_concurrency,
)
from app.services.field_watershed import (
    lng_lat_to_mercator_meters,
    load_npy_all_bands,
    load_npy_band,
    mercator_meters_to_lng_lat,
    simplify_polygon,
    trace_label_contours,
    watershed_segment,
)

logger = logging.getLogger("agrisat.automatic_parcels")

OVERPASS_API_URLS = list(
    dict.fromkeys(
        [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.openstreetmap.fr/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
        ]
    )
)
OVERPASS_USER_AGENT = "fieldscan-ai/1.0 (+https://localhost)"
OVERPASS_REQUEST_TIMEOUT_S = 10.0
DATABASE_DISCOVERY_TIMEOUT_S = 8.0
MAX_CANDIDATES = 48
MAX_SATELLITE_CELLS = 36
ANALYSIS_CONCURRENCY = 4
ANALYSIS_TIME_BUDGET_S = 60.0

GeeValue = dict[str, Any]


async def _with_timeout(coro: Awaitable[Any], timeout_s: float, label: str) -> Any:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{label} : dépassement de {int(timeout_s * 1000)}ms") from exc


# ── Segmentation par modèle de segmentation de parcelles (U-Net, agri_field_segmentation) ──

FIELD_SEGMENTATION_MODEL_URL = settings.field_segmentation_model_url
FIELD_MODEL_TILE_PX = 256
FIELD_MODEL_SCALE_M = 10
FIELD_MODEL_REGION_RADIUS_KM = 2
MAX_FIELD_MODEL_RADIUS_KM = 1.2
FIELD_MODEL_MONTHS = ["03", "04", "05", "06", "07", "08"]
FIELD_MODEL_CLOUD_LIMIT = 35
FIELD_MODEL_GEE_CONCURRENCY = 3
FIELD_MODEL_MIN_SEGMENT_AREA_M2 = 1_000
FIELD_MODEL_MAX_SEGMENT_AREA_M2 = 800_000
FIELD_MODEL_REQUEST_TIMEOUT_S = 30.0


def _field_model_season_year() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 9 else now.year - 1


def _circle_region_coordinates(lat: float, lng: float, radius_km: float, vertices: int = 48) -> list[list[list[float]]]:
    lat_rad = (lat * math.pi) / 180
    meters_per_deg_lat = 111_320
    meters_per_deg_lng = 111_320 * math.cos(lat_rad)
    radius_meters = radius_km * 1000
    ring: list[list[float]] = []
    for i in range(vertices + 1):
        angle = (2 * math.pi * i) / vertices
        d_lat = (radius_meters * math.sin(angle)) / meters_per_deg_lat
        d_lng = (radius_meters * math.cos(angle)) / meters_per_deg_lng
        ring.append([lng + d_lng, lat + d_lat])
    return [ring]


def _build_monthly_bands_expression(lat: float, lng: float, year: int, month: str) -> dict:
    start_date = f"{year}-{month}-01"
    month_int = int(month)
    if month_int == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month_int + 1:02d}-01"
    values: dict[str, GeeValue] = {}

    def reference(name: str) -> GeeValue:
        return {"valueReference": name}

    values["region"] = gee_call("GeometryConstructors.Polygon", {"coordinates": gee_constant(_circle_region_coordinates(lat, lng, FIELD_MODEL_REGION_RADIUS_KM))})
    values["intersectsRegion"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": reference("region")})})
    values["dateRange"] = gee_call("Filter.dateRangeContains", {"leftValue": gee_call("DateRange", {"start": gee_constant(start_date), "end": gee_constant(end_date)}), "rightField": gee_constant("system:time_start")})
    values["lowCloud"] = gee_call("Filter.lessThan", {"leftField": gee_constant("CLOUDY_PIXEL_PERCENTAGE"), "rightValue": gee_constant(FIELD_MODEL_CLOUD_LIMIT)})
    values["byRegion"] = gee_call("Collection.filter", {"collection": gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")}), "filter": reference("intersectsRegion")})
    values["byDate"] = gee_call("Collection.filter", {"collection": reference("byRegion"), "filter": reference("dateRange")})
    values["collection"] = gee_call("Collection.filter", {"collection": reference("byDate"), "filter": reference("lowCloud")})
    values["composite"] = gee_call("reduce.median", {"collection": reference("collection")})
    values["spectral"] = gee_call("Image.select", {"input": reference("composite"), "bandSelectors": gee_constant(["B2", "B3", "B4", "B8"])})
    values["withNdvi"] = add_spectral_index(reference("spectral"), "NDVI", ("B8", "B4"))
    values["finalImage"] = gee_call("Image.unmask", {"input": reference("withNdvi"), "value": gee_image_constant(0)})

    return {"result": "finalImage", "values": values}


class _Grid(dict):
    """widthPx, heightPx, originXMeters, originYMeters, scaleMeters."""


async def _call_gee_compute_pixels_multi_band(access_token: str, project_id: str, expression: dict, band_ids: list[str], grid: _Grid) -> tuple[dict[str, np.ndarray], int, int]:
    url = f"https://earthengine.googleapis.com/v1/projects/{project_id}/image:computePixels"
    body = {
        "expression": expression,
        "fileFormat": "NPY",
        "bandIds": band_ids,
        "grid": {
            "dimensions": {"width": grid["widthPx"], "height": grid["heightPx"]},
            "affineTransform": {
                "scaleX": grid["scaleMeters"], "shearX": 0, "translateX": grid["originXMeters"],
                "shearY": 0, "scaleY": -grid["scaleMeters"], "translateY": grid["originYMeters"],
            },
            "crsCode": "EPSG:3857",
        },
    }
    response = await fetch_with_retry("POST", url, headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}, json=body, timeout_s=GEE_COMPUTE_TIMEOUT_S)
    if response.status_code >= 400:
        raise RuntimeError(f"GEE computePixels ({','.join(band_ids)}) : erreur {response.status_code} {response.text[:300]}")
    bands, width, height = load_npy_all_bands(response.content)
    return bands, width, height


async def _fetch_field_model_input_array(access_token: str, project_id: str, lat: float, lng: float) -> tuple[np.ndarray, _Grid]:
    year = _field_model_season_year()
    center_x, center_y = lng_lat_to_mercator_meters(lng, lat)
    half_size_meters = (FIELD_MODEL_TILE_PX * FIELD_MODEL_SCALE_M) / 2
    grid = _Grid(
        widthPx=FIELD_MODEL_TILE_PX, heightPx=FIELD_MODEL_TILE_PX,
        originXMeters=center_x - half_size_meters, originYMeters=center_y + half_size_meters,
        scaleMeters=FIELD_MODEL_SCALE_M,
    )

    band_names = ["B2", "B3", "B4", "B8", "NDVI"]
    month_tasks = list(enumerate(FIELD_MODEL_MONTHS))

    async def month_mapper(item: tuple[int, str]) -> tuple[int, dict[str, np.ndarray]]:
        month_index, month = item
        expression = _build_monthly_bands_expression(lat, lng, year, month)
        bands, _w, _h = await _call_gee_compute_pixels_multi_band(access_token, project_id, expression, band_names, grid)
        return month_index, bands

    monthly_results = await map_with_concurrency(month_tasks, FIELD_MODEL_GEE_CONCURRENCY, month_mapper)

    tile_size = FIELD_MODEL_TILE_PX * FIELD_MODEL_TILE_PX
    array = np.zeros(len(band_names) * len(FIELD_MODEL_MONTHS) * tile_size, dtype=np.float32)
    for month_index, bands in monthly_results:
        for band_index, band_name in enumerate(band_names):
            channel_index = band_index * len(FIELD_MODEL_MONTHS) + month_index
            array[channel_index * tile_size : (channel_index + 1) * tile_size] = bands[band_name]

    return array, grid


async def _call_field_boundary_model(raw_array: np.ndarray) -> dict[str, Any]:
    """Contrat : POST multipart "file" -> tableau .npy (30,256,256) brut, réponse
    {polygons:[{points:[{x,y},...], score}], image_width, image_height} en coordonnées
    pixels (origine haut-gauche) — voir agri_field_segmentation/api.py::segment_endpoint."""
    shape = (5 * len(FIELD_MODEL_MONTHS), FIELD_MODEL_TILE_PX, FIELD_MODEL_TILE_PX)
    buf = io.BytesIO()
    np.save(buf, raw_array.reshape(shape).astype("<f4"))
    npy_bytes = buf.getvalue()

    response = await fetch_with_retry(
        "POST", f"{FIELD_SEGMENTATION_MODEL_URL}/segment", files={"file": ("field-input.npy", npy_bytes)}, timeout_s=FIELD_MODEL_REQUEST_TIMEOUT_S
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Modèle de segmentation de parcelles : erreur {response.status_code} {response.text[:300]}")
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("polygons"), list):
        raise RuntimeError("Réponse du modèle de segmentation de parcelles invalide (champ 'polygons' manquant).")
    return data


async def _discover_agricultural_parcels_from_field_model(lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    if not FIELD_SEGMENTATION_MODEL_URL:
        return []
    if radius_km > MAX_FIELD_MODEL_RADIUS_KM:
        return []

    service_account_json = settings.gee_service_account_key
    if not service_account_json or service_account_json.startswith("VOTRE_"):
        return []

    access_token = await get_gee_access_token()
    project_id = get_gee_project_id()
    array, grid = await _fetch_field_model_input_array(access_token, project_id, lat, lng)
    result = await _call_field_boundary_model(array)

    candidates: list[dict[str, Any]] = []
    index = 0
    for polygon in result["polygons"]:
        points = polygon.get("points") if isinstance(polygon, dict) else None
        if not isinstance(points, list) or len(points) < 3:
            continue

        coordinates = []
        for point in points:
            x_meters = grid["originXMeters"] + point["x"] * grid["scaleMeters"]
            y_meters = grid["originYMeters"] - point["y"] * grid["scaleMeters"]
            point_lng, point_lat = mercator_meters_to_lng_lat(x_meters, y_meters)
            coordinates.append({"lat": point_lat, "lng": point_lng})

        area_m2 = approximate_polygon_area_m2(coordinates)
        if area_m2 < FIELD_MODEL_MIN_SEGMENT_AREA_M2 or area_m2 > FIELD_MODEL_MAX_SEGMENT_AREA_M2:
            continue
        if not _is_plausible_field_shape(coordinates, area_m2):
            continue

        index += 1
        candidates.append(
            {
                "id": f"field-model-{lat:.6f}-{lng:.6f}-{index}",
                "coordinates": coordinates,
                "center": _polygon_center(coordinates),
                "tags": {"source": "field-boundary-model"},
            }
        )

    return candidates


# ── Segmentation par watershed marqué + variance NDVI multi-temporelle (sans ML) ──

WATERSHED_RASTER_SCALE_M = 10
MAX_WATERSHED_RADIUS_KM = 1.5
WATERSHED_SEASON_DAYS = 120
WATERSHED_TEMPORAL_PERIODS = 4
WATERSHED_MIN_SEGMENT_AREA_M2 = 1_000
WATERSHED_MAX_SEGMENT_AREA_M2 = 800_000
WATERSHED_SIMPLIFY_EPS_PX = 1.2
WATERSHED_BARRIER_VALUE = 1_000_000


async def _call_gee_compute_pixels(access_token: str, project_id: str, expression: dict, band_id: str, grid: _Grid) -> tuple[np.ndarray, int, int]:
    url = f"https://earthengine.googleapis.com/v1/projects/{project_id}/image:computePixels"
    body = {
        "expression": expression,
        "fileFormat": "NPY",
        "bandIds": [band_id],
        "grid": {
            "dimensions": {"width": grid["widthPx"], "height": grid["heightPx"]},
            "affineTransform": {
                "scaleX": grid["scaleMeters"], "shearX": 0, "translateX": grid["originXMeters"],
                "shearY": 0, "scaleY": -grid["scaleMeters"], "translateY": grid["originYMeters"],
            },
            "crsCode": "EPSG:3857",
        },
    }
    response = await fetch_with_retry("POST", url, headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}, json=body, timeout_s=GEE_COMPUTE_TIMEOUT_S)
    if response.status_code >= 400:
        raise RuntimeError(f"GEE computePixels ({band_id}) : erreur {response.status_code} {response.text[:300]}")
    data, width, height = load_npy_band(response.content, band_id)
    return data.reshape(-1), width, height


def _build_boundary_strength_expression(lat: float, lng: float, radius_km: float) -> dict:
    end_date = datetime.now(timezone.utc)
    season_start = end_date - timedelta(days=WATERSHED_SEASON_DAYS)
    values: dict[str, GeeValue] = {}

    def reference(name: str) -> GeeValue:
        return {"valueReference": name}

    values["region"] = gee_call("GeometryConstructors.Polygon", {"coordinates": gee_constant(_circle_region_coordinates(lat, lng, radius_km))})
    values["intersectsRegion"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": reference("region")})})
    values["collectionByRegion"] = gee_call("Collection.filter", {"collection": gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")}), "filter": reference("intersectsRegion")})

    def build_period_ndvi(var_prefix: str, start_date: str, period_end_date: str, cloud_limit: float) -> str:
        values[f"{var_prefix}DateRange"] = gee_call("Filter.dateRangeContains", {"leftValue": gee_call("DateRange", {"start": gee_constant(start_date), "end": gee_constant(period_end_date)}), "rightField": gee_constant("system:time_start")})
        values[f"{var_prefix}ByDate"] = gee_call("Collection.filter", {"collection": reference("collectionByRegion"), "filter": reference(f"{var_prefix}DateRange")})
        values[f"{var_prefix}LowCloud"] = gee_call("Filter.lessThan", {"leftField": gee_constant("CLOUDY_PIXEL_PERCENTAGE"), "rightValue": gee_constant(cloud_limit)})
        values[f"{var_prefix}Collection"] = gee_call("Collection.filter", {"collection": reference(f"{var_prefix}ByDate"), "filter": reference(f"{var_prefix}LowCloud")})
        values[f"{var_prefix}Composite"] = gee_call("reduce.median", {"collection": reference(f"{var_prefix}Collection")})
        values[f"{var_prefix}Spectral"] = gee_call("Image.select", {"input": reference(f"{var_prefix}Composite"), "bandSelectors": gee_constant(["B2", "B3", "B4", "B8", "B11", "B12"])})
        values[f"{var_prefix}WithNdvi"] = add_spectral_index(reference(f"{var_prefix}Spectral"), "NDVI", ("B8", "B4"))
        values[f"{var_prefix}NdviBand"] = gee_call("Image.select", {"input": reference(f"{var_prefix}WithNdvi"), "bandSelectors": gee_constant(["NDVI"])})
        return f"{var_prefix}NdviBand"

    season_ndvi_ref = build_period_ndvi("season", season_start.date().isoformat(), end_date.date().isoformat(), 35)

    values["seasonWithNdwi"] = add_spectral_index(reference("seasonWithNdvi"), "NDWI", ("B3", "B8"))
    values["seasonWithNdbi"] = add_spectral_index(reference("seasonWithNdwi"), "NDBI", ("B11", "B8"))
    values["seasonWithBsi"] = add_bare_soil_index(reference("seasonWithNdbi"))

    def season_band(name: str) -> GeeValue:
        return gee_call("Image.select", {"input": reference("seasonWithBsi"), "bandSelectors": gee_constant([name])})

    values["ndviMask"] = gee_call("Image.gt", {"image1": season_band("NDVI"), "image2": gee_image_constant(0.15)})
    values["waterMask"] = gee_call("Image.lt", {"image1": season_band("NDWI"), "image2": gee_image_constant(0.1)})
    values["urbanMask"] = gee_call("Image.lt", {"image1": season_band("NDBI"), "image2": gee_image_constant(0.05)})
    values["bareSoilMask"] = gee_call("Image.lt", {"image1": season_band("BSI"), "image2": gee_image_constant(0.25)})
    values["fieldMask"] = gee_call(
        "Image.and",
        {"image1": gee_call("Image.and", {"image1": gee_call("Image.and", {"image1": reference("ndviMask"), "image2": reference("waterMask")}), "image2": reference("urbanMask")}), "image2": reference("bareSoilMask")},
    )

    values["ndviGradient"] = gee_call("Image.gradient", {"input": reference(season_ndvi_ref)})
    values["gradX"] = gee_call("Image.select", {"input": reference("ndviGradient"), "bandSelectors": gee_constant(["x"])})
    values["gradY"] = gee_call("Image.select", {"input": reference("ndviGradient"), "bandSelectors": gee_constant(["y"])})
    values["gradMagnitude"] = gee_call("Image.hypot", {"image1": reference("gradX"), "image2": reference("gradY")})

    period_s = (WATERSHED_SEASON_DAYS * 86_400) / WATERSHED_TEMPORAL_PERIODS
    stack_ref: str | None = None
    for i in range(WATERSHED_TEMPORAL_PERIODS):
        period_start = season_start + timedelta(seconds=i * period_s)
        period_end = season_start + timedelta(seconds=(i + 1) * period_s)
        ndvi_ref = build_period_ndvi(f"p{i}", period_start.date().isoformat(), period_end.date().isoformat(), 50)
        renamed_ref = f"p{i}NdviRenamed"
        values[renamed_ref] = gee_call("Image.rename", {"input": reference(ndvi_ref), "names": gee_constant([f"ndvi_{i}"])})
        if stack_ref is None:
            stack_ref = renamed_ref
        else:
            combined_ref = f"tempStack{i}"
            values[combined_ref] = gee_call("Image.addBands", {"dstImg": reference(stack_ref), "srcImg": reference(renamed_ref)})
            stack_ref = combined_ref
    values["temporalStdDev"] = gee_call("Image.reduce", {"image": reference(stack_ref), "reducer": gee_call("Reducer.stdDev", {})})

    def apply_barrier(image_ref: str, out_name: str) -> str:
        masked_ref = f"{out_name}Masked"
        unmasked_ref = f"{out_name}Barrier"
        values[masked_ref] = gee_call("Image.updateMask", {"image": reference(image_ref), "mask": reference("fieldMask")})
        values[unmasked_ref] = gee_call("Image.unmask", {"input": reference(masked_ref), "value": gee_image_constant(WATERSHED_BARRIER_VALUE)})
        return unmasked_ref

    grad_barrier_ref = apply_barrier("gradMagnitude", "grad")
    tstd_barrier_ref = apply_barrier("temporalStdDev", "tstd")

    values["gradFloat"] = gee_call("Image.rename", {"input": gee_call("Image.toFloat", {"input": reference(grad_barrier_ref)}), "names": gee_constant(["grad"])})
    values["tstdFloat"] = gee_call("Image.rename", {"input": gee_call("Image.toFloat", {"input": reference(tstd_barrier_ref)}), "names": gee_constant(["tstd"])})
    values["finalImage"] = gee_call("Image.clip", {"input": gee_call("Image.addBands", {"dstImg": reference("gradFloat"), "srcImg": reference("tstdFloat")}), "geometry": reference("region")})

    return {"result": "finalImage", "values": values}


async def _discover_agricultural_parcels_from_watershed(lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    service_account_json = settings.gee_service_account_key
    if not service_account_json or service_account_json.startswith("VOTRE_"):
        return []
    if radius_km > MAX_WATERSHED_RADIUS_KM:
        return []

    access_token = await get_gee_access_token()
    project_id = get_gee_project_id()
    expression = _build_boundary_strength_expression(lat, lng, radius_km)

    center_x, center_y = lng_lat_to_mercator_meters(lng, lat)
    radius_meters = radius_km * 1000 * 1.05
    width_px = min(400, math.ceil((2 * radius_meters) / WATERSHED_RASTER_SCALE_M))
    height_px = width_px
    grid = _Grid(widthPx=width_px, heightPx=height_px, originXMeters=center_x - radius_meters, originYMeters=center_y + radius_meters, scaleMeters=WATERSHED_RASTER_SCALE_M)

    (grad_data, width, height), (tstd_data, _w2, _h2) = await asyncio.gather(
        _call_gee_compute_pixels(access_token, project_id, expression, "grad", grid),
        _call_gee_compute_pixels(access_token, project_id, expression, "tstd", grid),
    )
    size = width * height

    barrier = np.zeros(size, dtype=np.uint8)
    grad_min, grad_max = math.inf, -math.inf
    tstd_min, tstd_max = math.inf, -math.inf
    for i in range(size):
        is_barrier = grad_data[i] >= WATERSHED_BARRIER_VALUE or tstd_data[i] >= WATERSHED_BARRIER_VALUE
        barrier[i] = 1 if is_barrier else 0
        if not is_barrier:
            if grad_data[i] < grad_min:
                grad_min = float(grad_data[i])
            if grad_data[i] > grad_max:
                grad_max = float(grad_data[i])
            if tstd_data[i] < tstd_min:
                tstd_min = float(tstd_data[i])
            if tstd_data[i] > tstd_max:
                tstd_max = float(tstd_data[i])
    if not math.isfinite(grad_min) or not math.isfinite(tstd_min):
        return []

    strength = np.zeros(size, dtype=np.float32)
    grad_range = max(grad_max - grad_min, 1e-9)
    tstd_range = max(tstd_max - tstd_min, 1e-9)
    for i in range(size):
        if barrier[i]:
            strength[i] = WATERSHED_BARRIER_VALUE
            continue
        norm_grad = (grad_data[i] - grad_min) / grad_range
        norm_tstd = (tstd_data[i] - tstd_min) / tstd_range
        strength[i] = 0.6 * norm_grad + 0.4 * norm_tstd

    labels = watershed_segment(strength, barrier, width, height)
    contours = trace_label_contours(labels, width, height)

    candidates: list[dict[str, Any]] = []
    index = 0
    for pixel_contour in contours.values():
        simplified = simplify_polygon([(float(p[0]), float(p[1])) for p in pixel_contour], WATERSHED_SIMPLIFY_EPS_PX)
        if len(simplified) < 3:
            continue

        coordinates = []
        for point_x, point_y in simplified:
            x_meters = grid["originXMeters"] + point_x * grid["scaleMeters"]
            y_meters = grid["originYMeters"] - point_y * grid["scaleMeters"]
            point_lng, point_lat = mercator_meters_to_lng_lat(x_meters, y_meters)
            coordinates.append({"lat": point_lat, "lng": point_lng})

        area_m2 = approximate_polygon_area_m2(coordinates)
        if area_m2 < WATERSHED_MIN_SEGMENT_AREA_M2 or area_m2 > WATERSHED_MAX_SEGMENT_AREA_M2:
            continue
        if not _is_plausible_field_shape(coordinates, area_m2):
            continue

        index += 1
        candidates.append(
            {"id": f"gee-watershed-{lat:.6f}-{lng:.6f}-{index}", "coordinates": coordinates, "center": _polygon_center(coordinates), "tags": {"source": "gee-watershed-segmentation"}}
        )

    return candidates


# ── Segmentation GEE/SNIC (sans modèle ML) ──

REGION_SNIC_PARAMETERS = {"size": 24, "compactness": 0.6, "connectivity": 8, "scale": 10}
MIN_PARCEL_SEGMENT_AREA_M2 = 1_000
MAX_PARCEL_SEGMENT_AREA_M2 = 800_000
MAX_SEGMENTATION_RADIUS_KM = 5
CIRCLE_REGION_VERTICES = 48


def _build_region_snic_expression(lat: float, lng: float, radius_km: float) -> dict:
    end_date = datetime.now(timezone.utc).date().isoformat()
    start_date = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
    values: dict[str, GeeValue] = {}

    def reference(name: str) -> GeeValue:
        return {"valueReference": name}

    values["region"] = gee_call("GeometryConstructors.Polygon", {"coordinates": gee_constant(_circle_region_coordinates(lat, lng, radius_km, CIRCLE_REGION_VERTICES))})
    values["intersectsRegion"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": reference("region")})})
    values["dateRange"] = gee_call("Filter.dateRangeContains", {"leftValue": gee_call("DateRange", {"start": gee_constant(start_date), "end": gee_constant(end_date)}), "rightField": gee_constant("system:time_start")})
    values["lowCloudCover"] = gee_call("Filter.lessThan", {"leftField": gee_constant("CLOUDY_PIXEL_PERCENTAGE"), "rightValue": gee_constant(35)})
    values["collectionByRegion"] = gee_call("Collection.filter", {"collection": gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")}), "filter": reference("intersectsRegion")})
    values["collectionByDate"] = gee_call("Collection.filter", {"collection": reference("collectionByRegion"), "filter": reference("dateRange")})
    values["collection"] = gee_call("Collection.filter", {"collection": reference("collectionByDate"), "filter": reference("lowCloudCover")})
    values["composite"] = gee_call("reduce.median", {"collection": reference("collection")})
    values["spectralImage"] = gee_call("Image.select", {"input": reference("composite"), "bandSelectors": gee_constant(["B2", "B3", "B4", "B8", "B11", "B12"])})
    values["withNdvi"] = add_spectral_index(reference("spectralImage"), "NDVI", ("B8", "B4"))
    values["withNdwi"] = add_spectral_index(reference("withNdvi"), "NDWI", ("B3", "B8"))
    values["withNdbi"] = add_spectral_index(reference("withNdwi"), "NDBI", ("B11", "B8"))
    values["segmentationImage"] = add_bare_soil_index(reference("withNdbi"))

    def image_band(name: str) -> GeeValue:
        return gee_call("Image.select", {"input": reference("segmentationImage"), "bandSelectors": gee_constant([name])})

    values["ndviMask"] = gee_call("Image.gt", {"image1": image_band("NDVI"), "image2": gee_image_constant(0.15)})
    values["waterMask"] = gee_call("Image.lt", {"image1": image_band("NDWI"), "image2": gee_image_constant(0.1)})
    values["urbanMask"] = gee_call("Image.lt", {"image1": image_band("NDBI"), "image2": gee_image_constant(0.05)})
    values["bareSoilMask"] = gee_call("Image.lt", {"image1": image_band("BSI"), "image2": gee_image_constant(0.25)})
    values["fieldMask"] = gee_call(
        "Image.and",
        {"image1": gee_call("Image.and", {"image1": gee_call("Image.and", {"image1": reference("ndviMask"), "image2": reference("waterMask")}), "image2": reference("urbanMask")}), "image2": reference("bareSoilMask")},
    )
    values["maskedImage"] = gee_call("Image.updateMask", {"image": reference("segmentationImage"), "mask": reference("fieldMask")})
    values["fieldImage"] = gee_call("Image.clip", {"input": reference("maskedImage"), "geometry": reference("region")})
    values["snic"] = gee_call(
        "Image.Segmentation.SNIC",
        {
            "image": reference("fieldImage"),
            "size": gee_constant(REGION_SNIC_PARAMETERS["size"]),
            "compactness": gee_constant(REGION_SNIC_PARAMETERS["compactness"]),
            "connectivity": gee_constant(REGION_SNIC_PARAMETERS["connectivity"]),
            "neighborhoodSize": gee_constant(REGION_SNIC_PARAMETERS["size"] * 4),
        },
    )
    values["snicClusters"] = gee_call("Image.select", {"input": reference("snic"), "bandSelectors": gee_constant(["clusters"])})
    values["vectorsImage"] = gee_call("Image.addBands", {"dstImg": reference("snicClusters"), "srcImg": reference("fieldImage")})
    values["vectors"] = gee_call(
        "Image.reduceToVectors",
        {
            "image": reference("vectorsImage"),
            "reducer": gee_call("Reducer.mean", {}),
            "geometry": reference("region"),
            "scale": gee_constant(REGION_SNIC_PARAMETERS["scale"]),
            "geometryType": gee_constant("polygon"),
            "eightConnected": gee_constant(True),
            "labelProperty": gee_constant("segment_id"),
            "bestEffort": gee_constant(True),
            "maxPixels": gee_constant(30_000_000),
            "tileScale": gee_constant(4),
        },
    )

    return {"expression": {"result": "vectors", "values": values}}


async def _discover_agricultural_parcels_from_gee_segmentation(lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    service_account_json = settings.gee_service_account_key
    if not service_account_json or service_account_json.startswith("VOTRE_"):
        return []
    if radius_km > MAX_SEGMENTATION_RADIUS_KM:
        return []

    access_token = await get_gee_access_token()
    project_id = get_gee_project_id()
    raw = await call_gee_compute_raw(access_token, project_id, _build_region_snic_expression(lat, lng, radius_km))
    result = raw.get("result")
    if not is_gee_feature_collection(result):
        return []

    candidates: list[dict[str, Any]] = []
    for index, feature in enumerate(result["features"]):
        if not isinstance(feature, dict):
            continue
        coordinates = extract_latlng_from_geometry(feature.get("geometry"))
        if len(coordinates) < 3:
            continue

        area_m2 = approximate_polygon_area_m2(coordinates)
        if area_m2 < MIN_PARCEL_SEGMENT_AREA_M2 or area_m2 > MAX_PARCEL_SEGMENT_AREA_M2:
            continue
        if not _is_plausible_field_shape(coordinates, area_m2):
            continue

        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        ndvi = properties.get("NDVI")
        tags = {"source": "gee-snic-segmentation"}
        if isinstance(ndvi, (int, float)) and not isinstance(ndvi, bool):
            tags["ndvi_hint"] = str(js_round(ndvi * 1000) / 1000)

        candidates.append({"id": f"gee-snic-{lat:.6f}-{lng:.6f}-{index}", "coordinates": coordinates, "center": _polygon_center(coordinates), "tags": tags})

    return candidates


# ── Overpass (OSM) ──


async def _discover_agricultural_parcels_from_overpass(lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    radius_m = radius_km * 1000
    query = (
        f'[out:json][timeout:20];(way["landuse"~"farmland|farm|orchard|vineyard|meadow"](around:{radius_m},{lat},{lng});'
        f'relation["landuse"~"farmland|farm|orchard|vineyard|meadow"](around:{radius_m},{lat},{lng});'
        f'way["crop"](around:{radius_m},{lat},{lng});relation["crop"](around:{radius_m},{lat},{lng}););out tags geom;'
    )

    async with httpx.AsyncClient() as client:
        for endpoint in OVERPASS_API_URLS:
            try:
                response = await client.post(
                    endpoint,
                    headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": OVERPASS_USER_AGENT},
                    data={"data": query},
                    timeout=OVERPASS_REQUEST_TIMEOUT_S,
                )
                if response.status_code >= 400:
                    logger.warning("Overpass endpoint %s returned %s: %s", endpoint, response.status_code, response.text)
                    continue
                payload = response.json()
                elements = payload.get("elements") if isinstance(payload, dict) else None
                if not isinstance(elements, list):
                    continue

                parsed: list[dict[str, Any]] = []
                for element in elements:
                    if not isinstance(element, dict) or element.get("type") != "way" or not isinstance(element.get("id"), int):
                        continue
                    geometry = element.get("geometry")
                    if not isinstance(geometry, list):
                        continue
                    coordinates = [
                        {"lat": p["lat"], "lng": p["lon"]}
                        for p in geometry
                        if isinstance(p, dict) and isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lon"), (int, float))
                    ]
                    if len(coordinates) < 3:
                        continue
                    first, last = coordinates[0], coordinates[-1]
                    closed_coordinates = coordinates[:-1] if first["lat"] == last["lat"] and first["lng"] == last["lng"] else coordinates
                    if len(closed_coordinates) < 3:
                        continue
                    center = {
                        "lat": sum(p["lat"] for p in closed_coordinates) / len(closed_coordinates),
                        "lng": sum(p["lng"] for p in closed_coordinates) / len(closed_coordinates),
                    }
                    tags = dict(element.get("tags") or {})
                    tags["source"] = "osm"
                    parsed.append({"id": f"osm-way-{element['id']}", "coordinates": closed_coordinates, "center": center, "tags": tags})

                candidates = [c for c in parsed if _is_parcel_within_radius(c, {"lat": lat, "lng": lng}, radius_km)]
                if candidates:
                    return candidates
            except Exception:  # noqa: BLE001
                continue

    return []


# ── Base de données ──


async def _discover_agricultural_parcels_from_database(lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    lat_delta = radius_km / 110.574
    longitude_scale = max(abs(math.cos((lat * math.pi) / 180)), 0.1)
    lng_delta = radius_km / (111.32 * longitude_scale)
    min_lat, max_lat = lat - lat_delta, lat + lat_delta
    min_lng, max_lng = lng - lng_delta, lng + lng_delta

    async def _query() -> list[Parcelle]:
        async with async_session_maker() as session:
            stmt = select(Parcelle).where(
                Parcelle.center_lat >= min_lat, Parcelle.center_lat <= max_lat, Parcelle.center_lng >= min_lng, Parcelle.center_lng <= max_lng
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    try:
        rows = await _with_timeout(_query(), DATABASE_DISCOVERY_TIMEOUT_S, "discoverAgriculturalParcelsFromDatabase")
    except Exception as error:  # noqa: BLE001
        logger.error("Database access error for parcel discovery: %s", error)
        return []

    candidates: list[dict[str, Any]] = []
    for row in rows:
        raw_coordinates = row.coordinates if isinstance(row.coordinates, list) else []
        coordinates = [
            {"lat": p["lat"], "lng": p["lng"]}
            for p in raw_coordinates
            if isinstance(p, dict) and isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lng"), (int, float))
        ]
        if len(coordinates) < 3:
            continue
        center = {"lat": sum(p["lat"] for p in coordinates) / len(coordinates), "lng": sum(p["lng"] for p in coordinates) / len(coordinates)}
        candidate = {"id": str(row.id), "coordinates": coordinates, "center": center, "tags": {"source": "database"}}
        if _is_parcel_within_radius(candidate, {"lat": lat, "lng": lng}, radius_km):
            candidates.append(candidate)

    return candidates


# ── Géométrie ──


def _point_in_polygon(point: dict[str, float], polygon: list[dict[str, float]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["lng"], polygon[i]["lat"]
        xj, yj = polygon[j]["lng"], polygon[j]["lat"]
        if (yi > point["lat"]) != (yj > point["lat"]) and point["lng"] < ((xj - xi) * (point["lat"] - yi)) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _distance_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    return math.hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy))


def _polygon_center(points: list[dict[str, float]]) -> dict[str, float]:
    return {"lat": sum(p["lat"] for p in points) / len(points), "lng": sum(p["lng"] for p in points) / len(points)}


def _polygon_perimeter_m(coords: list[dict[str, float]]) -> float:
    centroid = _polygon_center(coords)
    lat_factor = 111_320
    lng_factor = 111_320 * math.cos((centroid["lat"] * math.pi) / 180)
    points = [((p["lng"] - centroid["lng"]) * lng_factor, (p["lat"] - centroid["lat"]) * lat_factor) for p in coords]
    perimeter = 0.0
    n = len(points)
    for i in range(n):
        nxt = (i + 1) % n
        perimeter += math.hypot(points[nxt][0] - points[i][0], points[nxt][1] - points[i][1])
    return perimeter


MIN_FIELD_COMPACTNESS = 0.12


def _is_plausible_field_shape(coords: list[dict[str, float]], area_m2: float) -> bool:
    perimeter_m = _polygon_perimeter_m(coords)
    if perimeter_m <= 0:
        return False
    compactness = (4 * math.pi * area_m2) / (perimeter_m * perimeter_m)
    return compactness >= MIN_FIELD_COMPACTNESS


def _offset_point(center: dict[str, float], east_km: float, north_km: float) -> dict[str, float]:
    longitude_scale = max(abs(math.cos((center["lat"] * math.pi) / 180)), 0.1)
    return {"lat": center["lat"] + north_km / 110.574, "lng": center["lng"] + east_km / (111.32 * longitude_scale)}


def _distance_between_points(left: dict[str, float], right: dict[str, float]) -> float:
    longitude_scale = max(abs(math.cos((left["lat"] * math.pi) / 180)), 0.1)
    east_km = (right["lng"] - left["lng"]) * 111.32 * longitude_scale
    north_km = (right["lat"] - left["lat"]) * 110.574
    return math.hypot(east_km, north_km)


def _square_around(center: dict[str, float], half_side_km: float) -> list[dict[str, float]]:
    return [
        _offset_point(center, -half_side_km, -half_side_km),
        _offset_point(center, half_side_km, -half_side_km),
        _offset_point(center, half_side_km, half_side_km),
        _offset_point(center, -half_side_km, half_side_km),
    ]


def _radius_polygon(center: dict[str, float], radius_km: float) -> list[dict[str, float]]:
    radius_m = radius_km * 1_000
    longitude_scale = max(abs(math.cos((center["lat"] * math.pi) / 180)), 0.1)
    points = []
    for index in range(48):
        angle = (index / 48) * math.pi * 2
        points.append(
            {"lat": center["lat"] + (math.sin(angle) * radius_m) / 110_574, "lng": center["lng"] + (math.cos(angle) * radius_m) / (111_320 * longitude_scale)}
        )
    return points


def _cross_product(start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _line_intersection(start: tuple[float, float], end: tuple[float, float], boundary_start: tuple[float, float], boundary_end: tuple[float, float]) -> tuple[float, float]:
    direction = (end[0] - start[0], end[1] - start[1])
    boundary_direction = (boundary_end[0] - boundary_start[0], boundary_end[1] - boundary_start[1])
    denominator = direction[0] * boundary_direction[1] - direction[1] * boundary_direction[0]
    if denominator == 0:
        return end
    offset = (boundary_start[0] - start[0], boundary_start[1] - start[1])
    ratio = (offset[0] * boundary_direction[1] - offset[1] * boundary_direction[0]) / denominator
    return (start[0] + ratio * direction[0], start[1] + ratio * direction[1])


def _clip_polygon_to_radius(polygon: list[dict[str, float]], center: dict[str, float], radius_km: float) -> list[dict[str, float]]:
    longitude_scale = max(abs(math.cos((center["lat"] * math.pi) / 180)), 0.1)

    def to_local(point: dict[str, float]) -> tuple[float, float]:
        return ((point["lng"] - center["lng"]) * 111_320 * longitude_scale, (point["lat"] - center["lat"]) * 110_574)

    def from_local(point: tuple[float, float]) -> dict[str, float]:
        return {"lat": center["lat"] + point[1] / 110_574, "lng": center["lng"] + point[0] / (111_320 * longitude_scale)}

    points = list(polygon)
    if points and points[0]["lat"] == points[-1]["lat"] and points[0]["lng"] == points[-1]["lng"]:
        points = points[:-1]
    clipped = [to_local(p) for p in points]
    boundary = [to_local(p) for p in _radius_polygon(center, radius_km)]

    index = 0
    while index < len(boundary) and clipped:
        start = boundary[index]
        end = boundary[(index + 1) % len(boundary)]
        input_points = clipped
        clipped = []
        n = len(input_points)
        for point_index in range(n):
            previous = input_points[(point_index + n - 1) % n]
            current = input_points[point_index]
            previous_inside = _cross_product(start, end, previous) >= 0
            current_inside = _cross_product(start, end, current) >= 0
            if current_inside != previous_inside:
                clipped.append(_line_intersection(previous, current, start, end))
            if current_inside:
                clipped.append(current)
        index += 1

    return [from_local(p) for p in clipped]


def _is_parcel_within_radius(candidate: dict[str, Any], center: dict[str, float], radius_km: float) -> bool:
    if _point_in_polygon(center, candidate["coordinates"]):
        return True
    longitude_scale = math.cos((center["lat"] * math.pi) / 180)

    def to_km(point: dict[str, float]) -> tuple[float, float]:
        return ((point["lng"] - center["lng"]) * 111.32 * longitude_scale, (point["lat"] - center["lat"]) * 110.574)

    origin = (0.0, 0.0)
    points = [to_km(p) for p in candidate["coordinates"]]
    if any(math.hypot(p[0], p[1]) <= radius_km for p in points):
        return True
    n = len(points)
    return any(_distance_to_segment(origin, points[i], points[(i + 1) % n]) <= radius_km for i in range(n))


# ── Déduplication / couverture ──

DUPLICATE_CANDIDATE_DISTANCE_KM = 0.005
MIN_KNOWN_COVERAGE_RATIO = 0.6


def _deduplicate_by_center(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        is_duplicate = any(_distance_between_points(existing["center"], candidate["center"]) < DUPLICATE_CANDIDATE_DISTANCE_KM for existing in kept)
        if not is_duplicate:
            kept.append(candidate)
    return kept


def _known_coverage_ratio(candidates: list[dict[str, Any]], lat: float, lng: float, radius_km: float) -> float:
    circle_area_m2 = math.pi * (radius_km * 1_000) ** 2
    if circle_area_m2 <= 0:
        return 1.0
    covered_area_m2 = 0.0
    for candidate in candidates:
        clipped = _clip_polygon_to_radius(candidate["coordinates"], {"lat": lat, "lng": lng}, radius_km)
        if len(clipped) >= 3:
            covered_area_m2 += approximate_polygon_area_m2(clipped)
    return covered_area_m2 / circle_area_m2


def _ensure_coordinate_coverage(candidates: list[dict[str, Any]], lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    center = {"lat": lat, "lng": lng}
    constrained_raw = []
    for candidate in candidates:
        coordinates = _clip_polygon_to_radius(candidate["coordinates"], center, radius_km)
        if len(coordinates) < 3:
            continue
        constrained_raw.append({**candidate, "coordinates": coordinates, "center": _polygon_center(coordinates)})
    constrained_candidates = _deduplicate_by_center(constrained_raw)

    satellite_cells = [
        cell
        for cell in _create_satellite_search_cell_candidates(lat, lng, radius_km)
        if not any(_point_in_polygon(cell["center"], candidate["coordinates"]) for candidate in constrained_candidates)
    ]
    return (constrained_candidates + satellite_cells)[:MAX_CANDIDATES]


def _create_satellite_search_cell_candidates(lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    center = {"lat": lat, "lng": lng}
    cell_size_km = max(0.1, radius_km / math.sqrt(MAX_SATELLITE_CELLS / math.pi))
    half_cell_km = cell_size_km / 2
    cell_count = math.ceil(radius_km / cell_size_km)
    cells: list[dict[str, Any]] = []

    for row in range(-cell_count, cell_count + 1):
        for column in range(-cell_count, cell_count + 1):
            cell_center = _offset_point(center, column * cell_size_km, row * cell_size_km)
            if _distance_between_points(center, cell_center) > radius_km + half_cell_km:
                continue
            cell = _square_around(cell_center, half_cell_km)
            coordinates = _clip_polygon_to_radius(cell, center, radius_km)
            if len(coordinates) < 3:
                continue
            cells.append(
                {
                    "id": f"satellite-search-cell-{lat:.6f}-{lng:.6f}-{row}-{column}",
                    "coordinates": coordinates,
                    "center": _polygon_center(coordinates),
                    "tags": {"source": "satellite-search-cell"},
                    "persist": False,
                }
            )

    cells.sort(key=lambda c: _distance_between_points(center, c["center"]))
    return cells[:MAX_SATELLITE_CELLS]


async def _discover_agricultural_parcels(lat: float, lng: float, radius_km: float) -> list[dict[str, Any]]:
    osm_candidates, database_candidates = await asyncio.gather(
        _discover_agricultural_parcels_from_overpass(lat, lng, radius_km),
        _discover_agricultural_parcels_from_database(lat, lng, radius_km),
    )
    known_candidates = _deduplicate_by_center(osm_candidates + database_candidates)

    if _known_coverage_ratio(known_candidates, lat, lng, radius_km) >= MIN_KNOWN_COVERAGE_RATIO:
        return _ensure_coordinate_coverage(known_candidates, lat, lng, radius_km)

    traced_candidates: list[dict[str, Any]] = []
    try:
        traced_candidates = await _discover_agricultural_parcels_from_field_model(lat, lng, radius_km)
    except Exception as error:  # noqa: BLE001
        logger.warning("discoverAgriculturalParcelsFromFieldModel failed, trying next fallback: %s", error)

    if not traced_candidates:
        try:
            traced_candidates = await _discover_agricultural_parcels_from_watershed(lat, lng, radius_km)
        except Exception as error:  # noqa: BLE001
            logger.warning("discoverAgriculturalParcelsFromWatershed failed, trying next fallback: %s", error)

    if not traced_candidates:
        try:
            traced_candidates = await _discover_agricultural_parcels_from_gee_segmentation(lat, lng, radius_km)
        except Exception as error:  # noqa: BLE001
            logger.warning("discoverAgriculturalParcelsFromGeeSegmentation failed, trying next fallback: %s", error)

    merged_candidates = _deduplicate_by_center(known_candidates + traced_candidates)
    return _ensure_coordinate_coverage(merged_candidates, lat, lng, radius_km)


# ── Analyse des candidats + persistance ──


def _as_number(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _as_string(value: Any) -> str | None:
    return value if isinstance(value, str) and len(value) > 0 else None


def _as_string_array(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


async def _analyze_candidate(candidate: dict[str, Any], config: dict[str, float]) -> dict[str, Any]:
    try:
        status, data = await _analyze_parcel_handler(
            {"lat": candidate["center"]["lat"], "lng": candidate["center"]["lng"], "zoom": 15, "polygon": candidate["coordinates"]}
        )
    except Exception as error:  # noqa: BLE001
        return {**candidate, "analysis": None, "analysis_error": str(error) or "Analyse satellite indisponible."}

    if status != 200 or not isinstance(data, dict):
        return {**candidate, "analysis": None, "analysis_error": "Analyse satellite indisponible."}

    gdd: dict[str, Any] | None = None
    gdd_error: str | None = None
    try:
        gdd = await fetch_growing_degree_days(candidate["center"]["lat"], candidate["center"]["lng"], config)
    except Exception as error:  # noqa: BLE001
        gdd_error = str(error) or "Données degrés-jours indisponibles."

    model_detects_barley = data.get("is_barley") is True
    barley_presence = ("confirmed" if gdd and gdd.get("detected") is True else "probable") if model_detects_barley else "none"
    analysis = {
        **data,
        "barley_presence": barley_presence,
        "gdd_cumulative": gdd["cumulative"] if gdd else None,
        "gdd_threshold": config["threshold"],
        "gdd_base_temperature": config["baseTemperature"],
        "gdd_start_date": gdd["startDate"] if gdd else None,
        "gdd_end_date": gdd["endDate"] if gdd else None,
        "gdd_detected": gdd["detected"] if gdd else None,
        "gdd_daily_values": gdd["dailyValues"] if gdd else None,
        "gdd_error": gdd_error,
    }

    if candidate.get("persist") is not False:
        try:
            await _save_automatic_analysis(candidate, analysis)
        except Exception as error:  # noqa: BLE001
            logger.warning("Automatic analysis %s was not persisted: %s", candidate["id"], error)

    return {**candidate, "analysis": analysis, "analysis_error": None}


async def _save_automatic_analysis(candidate: dict[str, Any], analysis: dict[str, Any]) -> None:
    gdd_cumulative = _as_number(analysis.get("gdd_cumulative"))
    gdd_threshold = _as_number(analysis.get("gdd_threshold"))
    gdd_summary = f"Degrés-jours : {analysis.get('gdd_cumulative')}/{analysis.get('gdd_threshold')} °C" if gdd_cumulative is not None and gdd_threshold is not None else ""
    daily_values = analysis.get("gdd_daily_values")
    last_day = daily_values[-1] if isinstance(daily_values, list) and daily_values else None
    temperature_summary = (
        f"Dernier relevé : Tmax {last_day.get('tmax', '—') if isinstance(last_day, dict) else '—'}°C · Tmin {last_day.get('tmin', '—') if isinstance(last_day, dict) else '—'}°C"
        if isinstance(last_day, dict)
        else ""
    )
    details = " · ".join(p for p in [_as_string(analysis.get("details")), gdd_summary, temperature_summary] if p) or None
    recommendations = " · ".join(p for p in [_as_string(analysis.get("recommendations")), _as_string(analysis.get("details")), gdd_summary, temperature_summary] if p) or None

    days_since_planting_raw = analysis.get("days_since_planting")
    days_since_planting = days_since_planting_raw if isinstance(days_since_planting_raw, int) and not isinstance(days_since_planting_raw, bool) else None

    values = {
        "label": candidate["id"],
        "coordinates": candidate["coordinates"],
        "center_lat": candidate["center"]["lat"],
        "center_lng": candidate["center"]["lng"],
        "surface_ha": None,
        "culture_declared": _as_string(analysis.get("culture_declared")),
        "culture_detected": _as_string(analysis.get("culture_detected")),
        "ndvi_percentage": _as_number(analysis.get("percentage")),
        "confidence": _as_number(analysis.get("confidence")),
        "verdict": _as_string(analysis.get("verdict")),
        "details": details,
        "saison": _as_string(analysis.get("saison")),
        "soil_type": _as_string(analysis.get("soil_type")),
        "risk_factors": _as_string_array(analysis.get("risk_factors")),
        "recommendations": recommendations,
        "data_source": _as_string(analysis.get("data_source")),
        "owner_name": _as_string(candidate["tags"].get("owner")),
        "notes": None,
        "time_series_s1": analysis.get("time_series_s1") if isinstance(analysis.get("time_series_s1"), list) else [],
        "time_series_s2": analysis.get("time_series_s2") if isinstance(analysis.get("time_series_s2"), list) else [],
        "estimated_planting_date": _as_string(analysis.get("estimated_planting_date")),
        "estimated_harvest_date": _as_string(analysis.get("estimated_harvest_date")),
        "days_since_planting": days_since_planting,
        "growth_stage": _as_string(analysis.get("growth_stage")),
        "planting_confidence": _as_number(analysis.get("planting_confidence")),
        "evi": _as_number(analysis.get("evi")),
        "savi": _as_number(analysis.get("savi")),
        "ndwi": _as_number(analysis.get("ndwi")),
        "agro_score": _as_number(analysis.get("agro_score")),
        "hybrid_score": _as_number(analysis.get("hybrid_score")),
        "cnn_prob_barley": _as_number(analysis.get("cnn_prob_barley")),
        "cnn_prob_non_barley": _as_number(analysis.get("cnn_prob_non_barley")),
    }

    async with async_session_maker() as session:
        existing = (await session.execute(select(Parcelle).where(Parcelle.label == candidate["id"]))).scalars().first()
        if existing:
            for key, value in values.items():
                setattr(existing, key, value)
        else:
            session.add(Parcelle(**values))
        await session.commit()


# ── Degrés-jours de croissance (GDD, Open-Meteo) ──


async def fetch_growing_degree_days(lat: float, lng: float, config: dict[str, float]) -> dict[str, Any]:
    end_date = datetime.now(timezone.utc) - timedelta(days=2)
    start_date = end_date - timedelta(days=config["periodDays"] - 1)

    params = {
        "latitude": str(lat), "longitude": str(lng),
        "start_date": start_date.date().isoformat(), "end_date": end_date.date().isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min", "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    if response.status_code >= 400:
        raise RuntimeError("Les données Open-Meteo sont indisponibles.")
    payload = response.json()
    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict):
        raise RuntimeError("Réponse météo incomplète.")
    dates = daily.get("time")
    tmax_values = daily.get("temperature_2m_max")
    tmin_values = daily.get("temperature_2m_min")
    if not isinstance(dates, list) or not isinstance(tmax_values, list) or not isinstance(tmin_values, list):
        raise RuntimeError("Températures journalières indisponibles.")

    cumulative = 0.0
    valid_days = 0
    daily_values: list[dict[str, Any]] = []
    for index, date in enumerate(dates):
        tmax = tmax_values[index] if index < len(tmax_values) else None
        tmin = tmin_values[index] if index < len(tmin_values) else None
        if not isinstance(date, str) or not isinstance(tmax, (int, float)) or not isinstance(tmin, (int, float)):
            continue
        dj = js_round(((tmax + tmin) / 2 - config["baseTemperature"]) * 10) / 10
        daily_values.append({"date": date, "tmax": tmax, "tmin": tmin, "dj": dj})
        cumulative += dj
        valid_days += 1
    if valid_days == 0:
        raise RuntimeError("Aucune température valide n'a été reçue.")

    return {
        "dailyValues": daily_values,
        "cumulative": js_round(cumulative * 10) / 10,
        "detected": cumulative >= config["threshold"],
        "startDate": str(dates[0]) if dates else "",
        "endDate": str(dates[-1]) if dates else "",
    }


# ── Concurrence bornée par budget de temps ──

T = TypeVar("T")
R = TypeVar("R")


async def _map_with_concurrency_deadline(items: list[T], concurrency: int, deadline_s: float, mapper: Callable[[T], Awaitable[R]]) -> dict[int, R]:
    deadline_at = time.monotonic() + deadline_s
    results: dict[int, R] = {}
    cursor = 0
    cursor_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal cursor
        while True:
            async with cursor_lock:
                if time.monotonic() >= deadline_at:
                    return
                if cursor >= len(items):
                    return
                index = cursor
                cursor += 1
            results[index] = await mapper(items[index])

    worker_count = min(concurrency, len(items))
    if worker_count > 0:
        await asyncio.gather(*(worker() for _ in range(worker_count)))
    return results


# ── Point d'entrée principal ──


async def detect_automatic_parcels(request: dict[str, Any]) -> dict[str, Any]:
    lat, lng, radius_km = request["lat"], request["lng"], request["radiusKm"]
    base_temperature, threshold, period_days = request["baseTemperature"], request["threshold"], request["periodDays"]

    try:
        candidates = await _discover_agricultural_parcels(lat, lng, radius_km)
        config = {"baseTemperature": base_temperature, "threshold": threshold, "periodDays": period_days}
        analyzed_candidates = candidates[:MAX_CANDIDATES]
        field_model_fallback = any(c["tags"].get("source") == "field-boundary-model" for c in analyzed_candidates)
        watershed_fallback = not field_model_fallback and any(c["tags"].get("source") == "gee-watershed-segmentation" for c in analyzed_candidates)
        gee_segmentation_fallback = not field_model_fallback and not watershed_fallback and any(c["tags"].get("source") == "gee-snic-segmentation" for c in analyzed_candidates)
        satellite_window_fallback = (
            not field_model_fallback
            and not watershed_fallback
            and not gee_segmentation_fallback
            and any((c["tags"].get("source") or "").startswith("satellite-search") for c in analyzed_candidates)
        )

        analyzed_results = await _map_with_concurrency_deadline(
            analyzed_candidates, ANALYSIS_CONCURRENCY, ANALYSIS_TIME_BUDGET_S, lambda candidate: _analyze_candidate(candidate, config)
        )
        analyzed_parcels = [
            analyzed_results.get(
                index,
                {**candidate, "analysis": None, "analysis_error": "Analyse IA non exécutée : délai de recherche automatique dépassé (trop de parcelles candidates)."},
            )
            for index, candidate in enumerate(analyzed_candidates)
        ]
        analyzed_ids = {c["id"] for c in analyzed_candidates}
        parcels = analyzed_parcels + [
            {**candidate, "analysis": None, "analysis_error": "Analyse IA non exécutée pour cette parcelle."}
            for candidate in candidates
            if candidate["id"] not in analyzed_ids
        ]

        if field_model_fallback:
            notice = "Aucun contour vectoriel référencé : les limites de parcelles ont été détectées automatiquement par le modèle de segmentation U-Net (IA, entraîné sur AI4Boundaries)."
        elif watershed_fallback:
            notice = "Aucun contour vectoriel référencé : les limites de parcelles ont été détectées automatiquement par watershed + analyse NDVI multi-temporelle (Google Earth Engine, sans modèle IA)."
        elif gee_segmentation_fallback:
            notice = "Aucun contour vectoriel référencé : les limites de parcelles ont été détectées automatiquement par segmentation d'image satellite (SNIC / Google Earth Engine, sans modèle IA)."
        elif satellite_window_fallback:
            notice = "Aucun contour vectoriel n'a été trouvé : plusieurs cellules satellite autour du point sont analysées sans créer de fausse parcelle en base."
        elif len(candidates) == 0:
            notice = "Aucun contour agricole fiable n'est référencé dans ce rayon."
        else:
            notice = None

        return {
            "center": {"lat": lat, "lng": lng},
            "radius_km": radius_km,
            "base_temperature": base_temperature,
            "threshold": threshold,
            "period_days": period_days,
            "candidates_found": len(candidates),
            "analyzed_count": sum(1 for p in parcels if p["analysis"] is not None),
            "parcels": parcels,
            "notice": notice,
        }
    except Exception as error:  # noqa: BLE001
        logger.warning("detectAutomaticParcels: discovery/analysis failure: %s", error)
        message = str(error) or "Service de découverte indisponible."
        lower_message = message.lower()
        notice = (
            "La base des parcelles agricoles est indisponible. Essayez un rayon plus large ou vérifiez la configuration de la base de données."
            if "database" in message or "parcelles" in lower_message or "database" in lower_message
            else message
        )
        return {
            "center": {"lat": lat, "lng": lng},
            "radius_km": radius_km,
            "base_temperature": base_temperature,
            "threshold": threshold,
            "period_days": period_days,
            "candidates_found": 0,
            "analyzed_count": 0,
            "parcels": [],
            "notice": notice,
        }
