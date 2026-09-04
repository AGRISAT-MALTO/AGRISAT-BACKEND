"""Port 1:1 de src/barley-detect-simple.ts : pipeline "Version A" — Sentinel-2 L2A,
masque nuage/ombre SCL, NDVI/NDRE, segmentation SNIC, classification CNN externe sur
miniature Sentinel-2 (pas Google Static Maps), confirmation phénologique (GDD)."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.db import async_session_maker
from app.models import Parcelle
from app.services.analyze_parcel import (
    SNIC_PARAMETERS,
    add_spectral_index,
    call_gee_compute_raw,
    call_hf_model,
    capture_sentinel2_parcel_image,
    extract_latlng_from_geometry,
    gee_call,
    gee_constant,
    gee_image_constant,
    get_gee_access_token,
    is_gee_feature_collection,
    js_round,
    map_with_concurrency,
    polygon_centroid,
)
from app.services.automatic_parcels import fetch_growing_degree_days

logger = logging.getLogger("agrisat.barley_detect_simple")

MAX_IMAGE_AGE_DAYS_PRIMARY = 5
MAX_IMAGE_AGE_DAYS_FALLBACK = 10

TILE_RADIUS_M = 2_500
MAX_TILES = 40
TILE_CONCURRENCY = 3
CLASSIFY_CONCURRENCY = 3

DEFAULT_MIN_AREA_HA = 0.05
MAX_CANDIDATE_AREA_M2 = 800_000
MAX_CANDIDATES_TO_CLASSIFY = 20
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

DEFAULT_GDD_CONFIG = {"baseTemperature": 0, "threshold": 2200, "periodDays": 365}

NDVI_MIN = 0.3
NDWI_MAX = 0.1
NDRE_MIN = 0.1

GeeValue = dict[str, Any]


def _get_error_message(error: BaseException) -> str:
    return str(error) or "Erreur inconnue"


async def analyze_fields_simple(input_data: dict[str, Any]) -> dict[str, Any]:
    lat, lng, radius_m = input_data["lat"], input_data["lng"], input_data["radiusM"]
    confidence_threshold = input_data.get("confidenceThreshold") if input_data.get("confidenceThreshold") is not None else DEFAULT_CONFIDENCE_THRESHOLD
    min_area_ha = input_data.get("minAreaHa") if input_data.get("minAreaHa") is not None else DEFAULT_MIN_AREA_HA
    gdd_config = input_data.get("gddConfig") or DEFAULT_GDD_CONFIG
    warnings: list[str] = []

    async def _gdd_task() -> dict[str, Any] | None:
        try:
            return await fetch_growing_degree_days(lat, lng, gdd_config)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"Données degrés-jours indisponibles : {_get_error_message(error)}")
            return None

    gdd_task = asyncio.ensure_future(_gdd_task())

    def empty() -> dict[str, Any]:
        return {
            "type": "FeatureCollection", "features": [],
            "center": {"lat": lat, "lng": lng}, "radiusM": radius_m,
            "imageDate": None, "imageAgeDays": None, "cloudPercentage": None, "imageTimestampMs": None,
            "confidenceThreshold": confidence_threshold, "minAreaHa": min_area_ha,
            "candidatesFound": 0, "candidatesClassified": 0,
            "gddCumulative": None, "gddThreshold": gdd_config["threshold"],
            "warnings": warnings,
        }

    service_account_json = settings.gee_service_account_key
    if not service_account_json or service_account_json.startswith("VOTRE_"):
        warnings.append("GEE_SERVICE_ACCOUNT_KEY n'est pas configurée.")
        gdd_task.cancel()
        return empty()

    try:
        access_token = await get_gee_access_token()
        project_id = _project_id_from_service_account(service_account_json)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"GEE indisponible : {_get_error_message(error)}")
        gdd_task.cancel()
        return empty()

    window = await _select_best_image_window(access_token, project_id, lat, lng, radius_m)
    if window is None:
        warnings.append(f"Aucune image Sentinel-2 exploitable dans les {MAX_IMAGE_AGE_DAYS_FALLBACK} derniers jours.")
        gdd_task.cancel()
        return empty()

    tile_centers = _build_tile_centers(lat, lng, radius_m, TILE_RADIUS_M)[:MAX_TILES]

    async def tile_mapper(tile: dict[str, float]) -> list[dict[str, Any]]:
        return await _fetch_candidate_polygons(access_token, project_id, tile["lat"], tile["lng"], TILE_RADIUS_M, window, min_area_ha, warnings)

    candidate_lists = await map_with_concurrency(tile_centers, TILE_CONCURRENCY, tile_mapper)

    all_candidates = [c for sub in candidate_lists for c in sub]
    candidates = _dedupe_and_rank_candidates(all_candidates)[:MAX_CANDIDATES_TO_CLASSIFY]
    gdd = await gdd_task

    async def classify(candidate: dict[str, Any]) -> dict[str, Any] | None:
        try:
            center = polygon_centroid(candidate["coordinates"])
            thumbnail = await capture_sentinel2_parcel_image(access_token, project_id, center["lat"], center["lng"], 17, window["startDate"], window["endDate"])
            classification = await call_hf_model(thumbnail)
            confidence_fraction = classification["confidence"] / 100
            if not classification["is_barley"] or confidence_fraction < confidence_threshold:
                return None
            return {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[p["lng"], p["lat"]] for p in candidate["coordinates"]]]},
                "properties": {
                    "class": "ORGE",
                    "confidence": js_round(confidence_fraction * 1000) / 1000,
                    "areaHa": js_round((candidate["areaM2"] / 10_000) * 100) / 100,
                    "meanNDVI": candidate["ndvi"],
                    "meanNDRE": candidate["ndre"],
                    "imageDate": window["imageDate"],
                    "imageAgeDays": window["imageAgeDays"],
                    "cloudPercentage": window["cloudPercentage"],
                    "barleyPresence": "confirmed" if gdd and gdd.get("detected") is True else "probable",
                },
            }
        except Exception as error:  # noqa: BLE001
            warnings.append(f"Classification d'un candidat échouée : {_get_error_message(error)}")
            return None

    classified_features = await map_with_concurrency(candidates, CLASSIFY_CONCURRENCY, classify)
    features = [f for f in classified_features if f is not None]

    return {
        "type": "FeatureCollection", "features": features,
        "center": {"lat": lat, "lng": lng}, "radiusM": radius_m,
        "imageDate": window["imageDate"], "imageAgeDays": window["imageAgeDays"], "cloudPercentage": window["cloudPercentage"],
        "imageTimestampMs": window["imageTimestampMs"],
        "gddCumulative": gdd["cumulative"] if gdd else None, "gddThreshold": gdd_config["threshold"],
        "confidenceThreshold": confidence_threshold, "minAreaHa": min_area_ha,
        "candidatesFound": len(all_candidates), "candidatesClassified": len(candidates),
        "warnings": warnings,
    }


def _project_id_from_service_account(service_account_json: str) -> str:
    import json

    sa = json.loads(service_account_json)
    project_id = sa.get("project_id")
    return project_id if isinstance(project_id, str) and project_id else "earthengine-legacy"


# ── Persistance dans la table `parcelles` (registre/dashboard) ──


async def save_simple_field_parcelles(features: list[dict[str, Any]], warnings: list[str]) -> None:
    for feature in features:
        try:
            await _save_simple_field_parcelle(feature)
        except Exception as error:  # noqa: BLE001
            logger.warning("saveSimpleFieldParcelles: échec de la persistance d'un candidat : %s", error)
            warnings.append(f"Une parcelle détectée n'a pas pu être enregistrée dans le registre : {_get_error_message(error)}")


async def _save_simple_field_parcelle(feature: dict[str, Any]) -> None:
    ring = feature["geometry"]["coordinates"][0]
    coordinates = [{"lat": lat, "lng": lng} for lng, lat in ring]
    center = polygon_centroid(coordinates)
    label = f"simple-v1-{center['lat']:.5f}-{center['lng']:.5f}"
    props = feature["properties"]
    confidence, area_ha, mean_ndvi, mean_ndre = props["confidence"], props["areaHa"], props["meanNDVI"], props["meanNDRE"]
    image_date, image_age_days, cloud_percentage, barley_presence = props["imageDate"], props["imageAgeDays"], props["cloudPercentage"], props["barleyPresence"]
    confidence_percent = js_round(confidence * 1000) / 10
    presence_label = "confirmée par degrés-jours" if barley_presence == "confirmed" else "probable (CNN seul)"

    values = {
        "label": label,
        "coordinates": coordinates,
        "center_lat": center["lat"],
        "center_lng": center["lng"],
        "surface_ha": area_ha,
        "culture_declared": None,
        "culture_detected": "Orge",
        "ndvi_percentage": js_round(mean_ndvi * 1000) / 10 if mean_ndvi is not None else None,
        "ndre": mean_ndre,
        "confidence": confidence_percent,
        "verdict": f"Orge détectée (Sentinel-2, SNIC) — confiance {confidence_percent}% · présence {presence_label}",
        "details": f"Image Sentinel-2 du {image_date if image_date is not None else '—'} ({image_age_days if image_age_days is not None else '?'} j) · nuages {cloud_percentage if cloud_percentage is not None else '?'}%",
        "saison": None,
        "soil_type": None,
        "risk_factors": [],
        "recommendations": None,
        "data_source": "Sentinel-2 simple (v1, segmentation SNIC)",
        "owner_name": None,
        "notes": None,
        "time_series_s1": [],
        "time_series_s2": [],
        "estimated_planting_date": None,
        "estimated_harvest_date": None,
        "days_since_planting": None,
        "growth_stage": None,
        "planting_confidence": None,
        "evi": None,
        "savi": None,
        "ndwi": None,
        "agro_score": None,
        "hybrid_score": None,
        "cnn_prob_barley": None,
        "cnn_prob_non_barley": None,
    }

    async with async_session_maker() as session:
        existing = (await session.execute(select(Parcelle).where(Parcelle.label == label))).scalars().first()
        if existing:
            for key, value in values.items():
                setattr(existing, key, value)
        else:
            session.add(Parcelle(**values))
        await session.commit()


# ── Sélection de la meilleure image S2 L2A ──


async def _select_best_image_window(access_token: str, project_id: str, lat: float, lng: float, radius_m: float) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    end = now.date().isoformat()
    primary_start = (now - timedelta(days=MAX_IMAGE_AGE_DAYS_PRIMARY)).date().isoformat()
    fallback_start = (now - timedelta(days=MAX_IMAGE_AGE_DAYS_FALLBACK)).date().isoformat()

    primary = await _fetch_image_meta(access_token, project_id, lat, lng, radius_m, primary_start, end)
    if primary and primary["size"] > 0:
        chosen, chosen_start, is_primary = primary, primary_start, True
    else:
        chosen = await _fetch_image_meta(access_token, project_id, lat, lng, radius_m, fallback_start, end)
        chosen_start, is_primary = fallback_start, False

    if not chosen or chosen["size"] == 0 or chosen["imageDateMillis"] is None:
        return None

    image_date_millis = chosen["imageDateMillis"]
    image_date = datetime.fromtimestamp(image_date_millis / 1000, tz=timezone.utc)
    image_age_days = max(0, js_round((now.timestamp() * 1000 - image_date_millis) / 86_400_000))
    return {
        "startDate": primary_start if is_primary else chosen_start,
        "endDate": end,
        "imageDate": image_date.date().isoformat(),
        "imageAgeDays": image_age_days,
        "cloudPercentage": chosen["cloudPercentage"],
        "imageTimestampMs": image_date_millis,
    }


async def _fetch_image_meta(access_token: str, project_id: str, lat: float, lng: float, radius_m: float, start: str, end: str) -> dict[str, Any] | None:
    values: dict[str, GeeValue] = {}

    def ref(name: str) -> GeeValue:
        return {"valueReference": name}

    values["point"] = gee_call("GeometryConstructors.Point", {"coordinates": gee_constant([lng, lat])})
    values["region"] = gee_call("Geometry.buffer", {"geometry": ref("point"), "distance": gee_constant(radius_m)})
    values["intersects"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": ref("region")})})
    values["dateRange"] = gee_call("Filter.dateRangeContains", {"leftValue": gee_call("DateRange", {"start": gee_constant(start), "end": gee_constant(end)}), "rightField": gee_constant("system:time_start")})
    values["raw"] = gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")})
    values["byRegion"] = gee_call("Collection.filter", {"collection": ref("raw"), "filter": ref("intersects")})
    values["byDate"] = gee_call("Collection.filter", {"collection": ref("byRegion"), "filter": ref("dateRange")})
    values["sorted"] = gee_call("Collection.limit", {"collection": ref("byDate"), "limit": gee_constant(1), "key": gee_constant("CLOUDY_PIXEL_PERCENTAGE"), "ascending": gee_constant(True)})
    values["size"] = gee_call("Collection.size", {"collection": ref("sorted")})
    values["first"] = gee_call("Collection.first", {"collection": ref("sorted")})
    values["date"] = gee_call("Element.get", {"object": ref("first"), "property": gee_constant("system:time_start")})
    values["cloud"] = gee_call("Element.get", {"object": ref("first"), "property": gee_constant("CLOUDY_PIXEL_PERCENTAGE")})
    values["dict"] = gee_call(
        "Dictionary.set",
        {
            "dictionary": gee_call(
                "Dictionary.set",
                {"dictionary": gee_call("Dictionary.set", {"dictionary": gee_call("Dictionary", {}), "key": gee_constant("size"), "value": ref("size")}), "key": gee_constant("date"), "value": ref("date")},
            ),
            "key": gee_constant("cloud"),
            "value": ref("cloud"),
        },
    )

    try:
        raw = await call_gee_compute_raw(access_token, project_id, {"expression": {"result": "dict", "values": values}})
        result = raw.get("result")
        if not isinstance(result, dict):
            return None
        size = result.get("size") if isinstance(result.get("size"), (int, float)) else 0
        date = result.get("date")
        cloud = result.get("cloud")
        return {
            "size": size,
            "imageDateMillis": date if isinstance(date, (int, float)) else None,
            "cloudPercentage": js_round(cloud * 10) / 10 if isinstance(cloud, (int, float)) else None,
        }
    except Exception:  # noqa: BLE001
        return None


# ── Masque nuage/ombre (SCL) + bandes + NDVI/NDRE + masque de plausibilité + segmentation SNIC ──


async def _fetch_candidate_polygons(
    access_token: str, project_id: str, lat: float, lng: float, radius_m: float, window: dict[str, Any], min_area_ha: float, warnings: list[str]
) -> list[dict[str, Any]]:
    values: dict[str, GeeValue] = {}

    def ref(name: str) -> GeeValue:
        return {"valueReference": name}

    values["point"] = gee_call("GeometryConstructors.Point", {"coordinates": gee_constant([lng, lat])})
    values["region"] = gee_call("Geometry.buffer", {"geometry": ref("point"), "distance": gee_constant(radius_m)})
    values["intersects"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": ref("region")})})
    values["dateRange"] = gee_call("Filter.dateRangeContains", {"leftValue": gee_call("DateRange", {"start": gee_constant(window["startDate"]), "end": gee_constant(window["endDate"])}), "rightField": gee_constant("system:time_start")})
    values["raw"] = gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")})
    values["byRegion"] = gee_call("Collection.filter", {"collection": ref("raw"), "filter": ref("intersects")})
    values["byDate"] = gee_call("Collection.filter", {"collection": ref("byRegion"), "filter": ref("dateRange")})
    values["sorted"] = gee_call("Collection.limit", {"collection": ref("byDate"), "limit": gee_constant(1), "key": gee_constant("CLOUDY_PIXEL_PERCENTAGE"), "ascending": gee_constant(True)})
    values["image"] = gee_call("Collection.first", {"collection": ref("sorted")})

    values["scl"] = gee_call("Image.select", {"input": ref("image"), "bandSelectors": gee_constant(["SCL"])})
    values["isShadow"] = gee_call("Image.eq", {"image1": ref("scl"), "image2": gee_image_constant(3)})
    values["isCloudMed"] = gee_call("Image.eq", {"image1": ref("scl"), "image2": gee_image_constant(8)})
    values["isCloudHigh"] = gee_call("Image.eq", {"image1": ref("scl"), "image2": gee_image_constant(9)})
    values["isCirrus"] = gee_call("Image.eq", {"image1": ref("scl"), "image2": gee_image_constant(10)})
    values["isBad1"] = gee_call("Image.or", {"image1": ref("isShadow"), "image2": ref("isCloudMed")})
    values["isBad2"] = gee_call("Image.or", {"image1": ref("isCloudHigh"), "image2": ref("isCirrus")})
    values["isBad"] = gee_call("Image.or", {"image1": ref("isBad1"), "image2": ref("isBad2")})
    values["cloudMask"] = gee_call("Image.not", {"value": ref("isBad")})

    values["bands"] = gee_call("Image.select", {"input": ref("image"), "bandSelectors": gee_constant(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])})
    values["maskedBands"] = gee_call("Image.updateMask", {"image": ref("bands"), "mask": ref("cloudMask")})
    values["withNdvi"] = add_spectral_index(ref("maskedBands"), "NDVI", ("B8", "B4"))
    values["withNdre"] = add_spectral_index(ref("withNdvi"), "NDRE", ("B8A", "B5"))
    values["withNdwi"] = add_spectral_index(ref("withNdre"), "NDWI", ("B3", "B8"))

    values["ndviBand"] = gee_call("Image.select", {"input": ref("withNdwi"), "bandSelectors": gee_constant(["NDVI"])})
    values["ndreBand"] = gee_call("Image.select", {"input": ref("withNdwi"), "bandSelectors": gee_constant(["NDRE"])})
    values["ndwiBand"] = gee_call("Image.select", {"input": ref("withNdwi"), "bandSelectors": gee_constant(["NDWI"])})
    values["ndviOk"] = gee_call("Image.gt", {"image1": ref("ndviBand"), "image2": gee_image_constant(NDVI_MIN)})
    values["ndreOk"] = gee_call("Image.gt", {"image1": ref("ndreBand"), "image2": gee_image_constant(NDRE_MIN)})
    values["ndwiOk"] = gee_call("Image.lt", {"image1": ref("ndwiBand"), "image2": gee_image_constant(NDWI_MAX)})
    values["plausible1"] = gee_call("Image.and", {"image1": ref("ndviOk"), "image2": ref("ndreOk")})
    values["plausibleMask"] = gee_call("Image.and", {"image1": ref("plausible1"), "image2": ref("ndwiOk")})

    values["maskedForSegmentation"] = gee_call("Image.updateMask", {"image": ref("withNdwi"), "mask": ref("plausibleMask")})
    values["vegetationImage"] = gee_call("Image.clip", {"input": ref("maskedForSegmentation"), "geometry": ref("region")})
    values["snic"] = gee_call(
        "Image.Segmentation.SNIC",
        {
            "image": ref("vegetationImage"),
            "size": gee_constant(SNIC_PARAMETERS["size"]),
            "compactness": gee_constant(SNIC_PARAMETERS["compactness"]),
            "connectivity": gee_constant(SNIC_PARAMETERS["connectivity"]),
            "neighborhoodSize": gee_constant(SNIC_PARAMETERS["neighborhoodSize"]),
        },
    )
    values["snicClusters"] = gee_call("Image.select", {"input": ref("snic"), "bandSelectors": gee_constant(["clusters"])})
    values["vectorsImage"] = gee_call("Image.addBands", {"dstImg": ref("snicClusters"), "srcImg": ref("vegetationImage")})
    values["vectors"] = gee_call(
        "Image.reduceToVectors",
        {
            "image": ref("vectorsImage"),
            "reducer": gee_call("Reducer.mean", {}),
            "geometry": ref("region"),
            "scale": gee_constant(SNIC_PARAMETERS["scale"]),
            "geometryType": gee_constant("polygon"),
            "eightConnected": gee_constant(True),
            "labelProperty": gee_constant("segment_id"),
            "bestEffort": gee_constant(True),
            "maxPixels": gee_constant(20_000_000),
            "tileScale": gee_constant(4),
        },
    )

    try:
        raw = await call_gee_compute_raw(access_token, project_id, {"expression": {"result": "vectors", "values": values}})
        result = raw.get("result")
        if not is_gee_feature_collection(result):
            return []
        candidates = []
        for feature in result["features"]:
            candidates.extend(_to_polygon_candidate(feature, min_area_ha))
        return candidates
    except Exception as error:  # noqa: BLE001
        warnings.append(f"Tuile Sentinel-2 ({lat:.4f},{lng:.4f}) indisponible : {_get_error_message(error)}")
        return []


def _to_polygon_candidate(feature: Any, min_area_ha: float) -> list[dict[str, Any]]:
    if not isinstance(feature, dict):
        return []
    coordinates = extract_latlng_from_geometry(feature.get("geometry"))
    if len(coordinates) < 3:
        return []

    area_m2 = _approximate_polygon_area_m2_local(coordinates)
    if area_m2 < min_area_ha * 10_000 or area_m2 > MAX_CANDIDATE_AREA_M2:
        return []

    properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    ndvi = properties.get("NDVI") if isinstance(properties.get("NDVI"), (int, float)) else None
    ndre = properties.get("NDRE") if isinstance(properties.get("NDRE"), (int, float)) else None

    return [{"coordinates": coordinates, "areaM2": area_m2, "ndvi": ndvi, "ndre": ndre}]


def _approximate_polygon_area_m2_local(coords: list[dict[str, float]]) -> float:
    from app.services.analyze_parcel import approximate_polygon_area_m2

    return approximate_polygon_area_m2(coords)


def _dedupe_and_rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: list[dict[str, float]] = []
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        center = polygon_centroid(candidate["coordinates"])
        is_duplicate = any(_haversine_meters(point, center) < 15 for point in seen)
        if is_duplicate:
            continue
        seen.append(center)
        deduped.append(candidate)

    def sort_key(c: dict[str, Any]) -> tuple[float, float]:
        return (-(c["ndvi"] if c["ndvi"] is not None else -1), -c["areaM2"])

    return sorted(deduped, key=sort_key)


# ── Tuilage de la zone (GPS + rayon) en cercles couvrant l'AOI ──


def _build_tile_centers(lat: float, lng: float, radius_m: float, tile_radius_m: float) -> list[dict[str, float]]:
    if radius_m <= tile_radius_m:
        return [{"lat": lat, "lng": lng}]

    meters_per_deg_lat = 111_320
    meters_per_deg_lng = 111_320 * math.cos((lat * math.pi) / 180)
    step = tile_radius_m * 1.6
    steps = math.ceil(radius_m / step)
    centers: list[dict[str, float]] = []

    for row in range(-steps, steps + 1):
        for col in range(-steps, steps + 1):
            offset_x = col * step
            offset_y = row * step
            distance_from_center = math.hypot(offset_x, offset_y)
            if distance_from_center > radius_m + tile_radius_m:
                continue
            centers.append({"lat": lat + offset_y / meters_per_deg_lat, "lng": lng + offset_x / meters_per_deg_lng})

    origin = {"lat": lat, "lng": lng}
    centers.sort(key=lambda c: _haversine_meters(origin, c))
    return centers


def _haversine_meters(a: dict[str, float], b: dict[str, float]) -> float:
    meters_per_deg_lat = 111_320
    meters_per_deg_lng = 111_320 * math.cos((a["lat"] * math.pi) / 180)
    dx = (b["lng"] - a["lng"]) * meters_per_deg_lng
    dy = (b["lat"] - a["lat"]) * meters_per_deg_lat
    return math.hypot(dx, dy)
