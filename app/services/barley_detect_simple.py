"""Port de src/barley-detect-simple.ts : pipeline "Version A" — Sentinel-2 L2A, parcelles
déjà tracées à l'écran (modèle JECAM-VHR, envoyées par le frontend) classées entières par
le CNN externe sur miniature Sentinel-2 (pas Google Static Maps), confirmation
phénologique (GDD)."""

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
    approximate_polygon_area_m2,
    call_gee_compute_raw,
    call_hf_model,
    capture_sentinel2_parcel_image,
    gee_call,
    gee_constant,
    get_gee_access_token,
    js_round,
    map_with_concurrency,
    normalize_polygon,
    polygon_centroid,
)
from app.services.automatic_parcels import fetch_growing_degree_days

logger = logging.getLogger("agrisat.barley_detect_simple")

MAX_IMAGE_AGE_DAYS_PRIMARY = 5
MAX_IMAGE_AGE_DAYS_FALLBACK = 10

CLASSIFY_CONCURRENCY = 3

DEFAULT_MIN_AREA_HA = 0.05
MAX_CANDIDATE_AREA_M2 = 800_000
MAX_CANDIDATES_TO_CLASSIFY = 20
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

DEFAULT_GDD_CONFIG = {"baseTemperature": 0, "threshold": 2200, "periodDays": 365}

GeeValue = dict[str, Any]


def _get_error_message(error: BaseException) -> str:
    return str(error) or "Erreur inconnue"


async def analyze_fields_simple(input_data: dict[str, Any]) -> dict[str, Any]:
    lat, lng, radius_m = input_data["lat"], input_data["lng"], input_data["radiusM"]
    confidence_threshold = input_data.get("confidenceThreshold") if input_data.get("confidenceThreshold") is not None else DEFAULT_CONFIDENCE_THRESHOLD
    min_area_ha = input_data.get("minAreaHa") if input_data.get("minAreaHa") is not None else DEFAULT_MIN_AREA_HA
    gdd_config = input_data.get("gddConfig") or DEFAULT_GDD_CONFIG
    warnings: list[str] = []

    all_candidates = _traced_parcel_candidates(input_data.get("tracedParcels") or [], lat, lng, radius_m, min_area_ha)

    def empty() -> dict[str, Any]:
        return {
            "type": "FeatureCollection", "features": [],
            "center": {"lat": lat, "lng": lng}, "radiusM": radius_m,
            "imageDate": None, "imageAgeDays": None, "cloudPercentage": None, "imageTimestampMs": None,
            "confidenceThreshold": confidence_threshold, "minAreaHa": min_area_ha,
            "candidatesFound": len(all_candidates), "candidatesClassified": 0,
            "gddCumulative": None, "gddThreshold": gdd_config["threshold"],
            "warnings": warnings,
        }

    if not all_candidates:
        warnings.append("Aucune parcelle tracée dans ce rayon : laissez le traçage automatique (JECAM-VHR) finir sur la zone, puis relancez l'analyse.")
        return empty()

    async def _gdd_task() -> dict[str, Any] | None:
        try:
            return await fetch_growing_degree_days(lat, lng, gdd_config)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"Données degrés-jours indisponibles : {_get_error_message(error)}")
            return None

    gdd_task = asyncio.ensure_future(_gdd_task())

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

    candidates = all_candidates[:MAX_CANDIDATES_TO_CLASSIFY]
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
                    "meanNDVI": None,
                    "meanNDRE": None,
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
        "verdict": f"Orge détectée (Sentinel-2, parcelle tracée JECAM-VHR) — confiance {confidence_percent}% · présence {presence_label}",
        "details": f"Image Sentinel-2 du {image_date if image_date is not None else '—'} ({image_age_days if image_age_days is not None else '?'} j) · nuages {cloud_percentage if cloud_percentage is not None else '?'}%",
        "saison": None,
        "soil_type": None,
        "risk_factors": [],
        "recommendations": None,
        "data_source": "Sentinel-2 simple (v1, parcelles tracées JECAM-VHR)",
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


# ── Parcelles déjà tracées (JECAM-VHR) dans la zone ──


def _traced_parcel_candidates(
    traced_parcels: list[dict[str, Any]], lat: float, lng: float, radius_m: float, min_area_ha: float
) -> list[dict[str, Any]]:
    """Parcelles tracées dont le centre tombe dans le cercle, de la plus proche à la plus
    éloignée, sans doublons (centres à moins de 15 m)."""
    origin = {"lat": lat, "lng": lng}
    candidates: list[dict[str, Any]] = []
    for parcel in traced_parcels:
        coordinates = normalize_polygon(parcel.get("coordinates"))
        if coordinates is None:
            continue
        area_m2 = approximate_polygon_area_m2(coordinates)
        if area_m2 < min_area_ha * 10_000 or area_m2 > MAX_CANDIDATE_AREA_M2:
            continue
        center = polygon_centroid(coordinates)
        distance_m = _haversine_meters(origin, center)
        if distance_m > radius_m:
            continue
        candidates.append({"coordinates": coordinates, "areaM2": area_m2, "center": center, "distanceM": distance_m})

    candidates.sort(key=lambda c: c["distanceM"])
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(_haversine_meters(kept["center"], candidate["center"]) < 15 for kept in deduped):
            continue
        deduped.append(candidate)
    return deduped


def _haversine_meters(a: dict[str, float], b: dict[str, float]) -> float:
    meters_per_deg_lat = 111_320
    meters_per_deg_lng = 111_320 * math.cos((a["lat"] * math.pi) / 180)
    dx = (b["lng"] - a["lng"]) * meters_per_deg_lng
    dy = (b["lat"] - a["lat"]) * meters_per_deg_lat
    return math.hypot(dx, dy)
