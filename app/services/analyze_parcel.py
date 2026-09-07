"""Port 1:1 de src/analyze-parcel.ts : analyse détaillée d'une parcelle (indices
spectraux GEE, séries temporelles, SNIC, classification CNN HuggingFace, scoring
agro/hybride, détection de date de plantation) — et, en même temps, le client REST
Google Earth Engine partagé par automatic_parcels.py, barley_detect_simple.py,
zoning.py et app/routes/sentinel_tiles.py (mêmes primitives : auth, gee_call/
gee_constant, add_spectral_index, capture d'images, fetch_with_retry).

Fidélité : les formules d'indices spectraux, le scoring agro/hybride, la
détection de date de plantation et les arbres d'expression GEE sont portés
littéralement (mêmes seuils, mêmes arrondis `js_round` — équivalent exact de
Math.round JS, PAS le round() Python qui utilise l'arrondi au pair). L'auth GEE
garde la même architecture REST (JSON POSTé à value:compute / image:computePixels)
mais utilise `google-auth` au lieu d'une signature JWT RS256 à la main.
"""

from __future__ import annotations

import asyncio
import base64
import calendar
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from app.config import settings
from app.services.field_watershed import lng_lat_to_mercator_meters

logger = logging.getLogger("agrisat.analyze_parcel")

HF_MODEL_URL = settings.hf_model_url or "https://andritinatonny-agrisat.hf.space"
GOOGLE_MAPS_API_KEY = settings.google_maps_api_key

SNIC_PARAMETERS = {"size": 15, "compactness": 0.75, "connectivity": 8, "neighborhoodSize": 60, "scale": 10}
MIN_SEGMENT_AREA_M2 = 2_500
MAX_SEGMENT_AREA_M2 = 500_000
MAX_SEGMENTS_TO_CLASSIFY = 12
SEGMENT_CLASSIFY_CONCURRENCY = 3
MIN_BARLEY_CONFIDENCE = 70
TIME_SERIES_CONCURRENCY = 1
GEE_COMPUTE_TIMEOUT_S = 60.0
EXTERNAL_REQUEST_TIMEOUT_S = 30.0
RETRY_DELAYS_S = (0.25, 1.0)


def js_round(x: float) -> int:
    """Équivalent exact de Math.round JS (arrondi des .5 vers +Infini), contrairement
    à round() Python qui arrondit au pair le plus proche (banker's rounding)."""
    return math.floor(x + 0.5)


# ── HTTP client partagé + retry ──

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


async def fetch_with_retry(method: str, url: str, *, timeout_s: float, **kwargs: Any) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(len(RETRY_DELAYS_S) + 1):
        try:
            response = await _get_client().request(method, url, timeout=timeout_s, **kwargs)
            if response.status_code < 500 and response.status_code != 429:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code}")
        except Exception as exc:  # noqa: BLE001 — port littéral : toute erreur réseau déclenche un retry
            last_error = exc
        if attempt < len(RETRY_DELAYS_S):
            await asyncio.sleep(RETRY_DELAYS_S[attempt])
    raise last_error or RuntimeError("Échec réseau après plusieurs tentatives.")


def _error_message(error: BaseException) -> str:
    return str(error) or "erreur réseau inconnue"


# ── GEE Auth ──

_cached_access_token: dict[str, Any] | None = None
_access_token_lock = asyncio.Lock()
ACCESS_TOKEN_SAFETY_MARGIN_S = 5 * 60


def get_gee_project_id() -> str:
    service_account_json = settings.gee_service_account_key
    if not service_account_json:
        return "earthengine-legacy"
    try:
        sa = json.loads(service_account_json)
        project_id = sa.get("project_id")
        return project_id if isinstance(project_id, str) and project_id else "earthengine-legacy"
    except Exception:  # noqa: BLE001
        return "earthengine-legacy"


async def get_gee_access_token() -> str:
    global _cached_access_token
    now = time.time()
    if _cached_access_token and _cached_access_token["expires_at"] > now:
        return _cached_access_token["token"]
    async with _access_token_lock:
        now = time.time()
        if _cached_access_token and _cached_access_token["expires_at"] > now:
            return _cached_access_token["token"]
        return await _fetch_gee_access_token()


async def _fetch_gee_access_token() -> str:
    global _cached_access_token
    service_account_json = settings.gee_service_account_key
    if not service_account_json:
        raise RuntimeError("GEE_SERVICE_ACCOUNT_KEY is not configured")

    sa_info = json.loads(service_account_json)

    def _refresh() -> str:
        credentials = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/earthengine.readonly"]
        )
        credentials.refresh(GoogleAuthRequest())
        return credentials.token

    token = await asyncio.to_thread(_refresh)
    _cached_access_token = {"token": token, "expires_at": time.time() + 3_600 - ACCESS_TOKEN_SAFETY_MARGIN_S}
    return token


# ── GEE Expression Builders (primitives partagées) ──

GeeValue = dict[str, Any]


def gee_call(function_name: str, arguments: dict[str, GeeValue]) -> GeeValue:
    return {"functionInvocationValue": {"functionName": function_name, "arguments": arguments}}


def gee_constant(value: Any) -> GeeValue:
    return {"constantValue": value}


def gee_image_constant(value: float) -> GeeValue:
    return gee_call("Image.constant", {"value": gee_constant(value)})


def add_spectral_index(image: GeeValue, name: str, bands: tuple[str, str]) -> GeeValue:
    index = gee_call(
        "Image.rename",
        {
            "input": gee_call("Image.normalizedDifference", {"input": image, "bandNames": gee_constant(list(bands))}),
            "names": gee_constant([name]),
        },
    )
    return gee_call("Image.addBands", {"dstImg": image, "srcImg": index})


def add_bare_soil_index(image: GeeValue) -> GeeValue:
    def band(name: str) -> GeeValue:
        return gee_call("Image.select", {"input": image, "bandSelectors": gee_constant([name])})

    swir_plus_red = gee_call("Image.add", {"image1": band("B11"), "image2": band("B4")})
    nir_plus_blue = gee_call("Image.add", {"image1": band("B8"), "image2": band("B2")})
    bsi = gee_call(
        "Image.divide",
        {
            "image1": gee_call("Image.subtract", {"image1": swir_plus_red, "image2": nir_plus_blue}),
            "image2": gee_call("Image.add", {"image1": swir_plus_red, "image2": nir_plus_blue}),
        },
    )
    return gee_call("Image.addBands", {"dstImg": image, "srcImg": gee_call("Image.rename", {"input": bsi, "names": gee_constant(["BSI"])})})


# ── Géométrie parcelle ──

LatLng = dict[str, float]  # {"lat": ..., "lng": ...}


def normalize_polygon(value: Any) -> list[LatLng] | None:
    if not isinstance(value, list):
        return None
    points: list[LatLng] = []
    for point in value:
        if not isinstance(point, dict):
            continue
        lat = point.get("lat")
        lng = point.get("lng")
        if (
            isinstance(lat, (int, float))
            and not isinstance(lat, bool)
            and math.isfinite(lat)
            and -90 <= lat <= 90
            and isinstance(lng, (int, float))
            and not isinstance(lng, bool)
            and math.isfinite(lng)
            and -180 <= lng <= 180
        ):
            points.append({"lat": float(lat), "lng": float(lng)})
    if len(points) < 3:
        return None
    first, last = points[0], points[-1]
    if first["lat"] == last["lat"] and first["lng"] == last["lng"]:
        return points[:-1]
    return points


def polygon_coordinates(polygon: Any) -> list[list[list[float]]]:
    points = normalize_polygon(polygon)
    if points is None:
        raise ValueError("Le contour de la parcelle est invalide.")
    ring = [[p["lng"], p["lat"]] for p in points]
    ring.append(list(ring[0]))
    return [ring]


def extract_latlng_from_geometry(geometry: Any) -> list[LatLng]:
    if not isinstance(geometry, dict):
        return []
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    ring: Any = None
    if geom_type == "Polygon" and isinstance(coordinates, list):
        ring = coordinates[0] if coordinates else None
    elif geom_type == "MultiPolygon" and isinstance(coordinates, list):
        first = coordinates[0] if coordinates else None
        ring = first[0] if isinstance(first, list) and first else None
    if not isinstance(ring, list):
        return []
    result: list[LatLng] = []
    for coordinate in ring:
        if isinstance(coordinate, list) and len(coordinate) >= 2 and isinstance(coordinate[0], (int, float)) and isinstance(coordinate[1], (int, float)):
            result.append({"lat": float(coordinate[1]), "lng": float(coordinate[0])})
    return result


def polygon_centroid(coords: list[LatLng]) -> LatLng:
    total_lat = sum(p["lat"] for p in coords)
    total_lng = sum(p["lng"] for p in coords)
    return {"lat": total_lat / len(coords), "lng": total_lng / len(coords)}


def approximate_polygon_area_m2(coords: list[LatLng]) -> float:
    if len(coords) < 3:
        return 0.0
    centroid = polygon_centroid(coords)
    lat_factor = 111_320
    lng_factor = 111_320 * math.cos((centroid["lat"] * math.pi) / 180)
    points = [
        ((p["lng"] - centroid["lng"]) * lng_factor, (p["lat"] - centroid["lat"]) * lat_factor) for p in coords
    ]
    area = 0.0
    n = len(points)
    for index in range(n):
        nxt = (index + 1) % n
        area += points[index][0] * points[nxt][1] - points[nxt][0] * points[index][1]
    return abs(area) / 2


def segment_zoom_for_area(area_m2: float, latitude: float, fallback_zoom: int) -> int:
    target_width_m = max(180.0, math.sqrt(area_m2) * 2.2)
    zoom = math.floor(math.log2((156_543.03392 * math.cos(latitude * math.pi / 180) * 640) / target_width_m))
    return max(fallback_zoom, min(20, zoom))


def meters_per_pixel_at_zoom(lat: float, zoom: float) -> float:
    return (156_543.03392 * math.cos((lat * math.pi) / 180)) / (2**zoom)


def is_gee_feature_collection(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "FeatureCollection" and isinstance(value.get("features"), list)


# ── GEE compute helpers ──


async def call_gee_compute(access_token: str, project_id: str, expression: dict) -> dict[str, float | None] | None:
    url = f"https://earthengine.googleapis.com/v1/projects/{project_id}/value:compute"
    try:
        resp = await fetch_with_retry(
            "POST",
            url,
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json=expression,
            timeout_s=GEE_COMPUTE_TIMEOUT_S,
        )
        if resp.status_code >= 400:
            logger.error("GEE API error: %s %s", resp.status_code, resp.text[:300])
            return None
        data = resp.json()
        return data.get("result") or None
    except Exception as exc:  # noqa: BLE001
        logger.error("GEE fetch error: %s", exc)
        return None


async def call_gee_compute_raw(access_token: str, project_id: str, expression: dict) -> dict[str, Any]:
    url = f"https://earthengine.googleapis.com/v1/projects/{project_id}/value:compute"
    resp = await fetch_with_retry(
        "POST",
        url,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=expression,
        timeout_s=GEE_COMPUTE_TIMEOUT_S,
    )
    if resp.status_code >= 400:
        logger.error("GEE API error (raw): %s %s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"Erreur GEE lors de la segmentation ({resp.status_code}).")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Réponse GEE de segmentation invalide.")
    return data


class PixelGrid:
    __slots__ = ("width_px", "height_px", "origin_x_meters", "origin_y_meters", "scale_meters")

    def __init__(self, width_px: int, height_px: int, origin_x_meters: float, origin_y_meters: float, scale_meters: float):
        self.width_px = width_px
        self.height_px = height_px
        self.origin_x_meters = origin_x_meters
        self.origin_y_meters = origin_y_meters
        self.scale_meters = scale_meters


def build_true_color_visualize_expression(image_ref: GeeValue) -> GeeValue:
    return gee_call("Image.visualize", {"image": image_ref, "bands": gee_constant(["B4", "B3", "B2"]), "min": gee_constant(0), "max": gee_constant(3000)})


async def compute_pixels_png(access_token: str, project_id: str, expression: dict, grid: PixelGrid) -> bytes:
    url = f"https://earthengine.googleapis.com/v1/projects/{project_id}/image:computePixels"
    body = {
        "expression": expression,
        "fileFormat": "PNG",
        "grid": {
            "dimensions": {"width": grid.width_px, "height": grid.height_px},
            "affineTransform": {
                "scaleX": grid.scale_meters,
                "shearX": 0,
                "translateX": grid.origin_x_meters,
                "shearY": 0,
                "scaleY": -grid.scale_meters,
                "translateY": grid.origin_y_meters,
            },
            "crsCode": "EPSG:3857",
        },
    }
    response = await fetch_with_retry(
        "POST",
        url,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=body,
        timeout_s=GEE_COMPUTE_TIMEOUT_S,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"GEE computePixels (PNG) : erreur {response.status_code} {response.text[:300]}")
    return response.content


async def capture_sentinel2_parcel_image(
    access_token: str, project_id: str, lat: float, lng: float, zoom: float, start_date: str, end_date: str
) -> str:
    values: dict[str, GeeValue] = {}

    def ref(name: str) -> GeeValue:
        return {"valueReference": name}

    values["point"] = gee_call("GeometryConstructors.Point", {"coordinates": gee_constant([lng, lat])})
    values["region"] = gee_call("Geometry.buffer", {"geometry": ref("point"), "distance": gee_constant(5_000)})
    values["intersects"] = gee_call("Filter.intersects", {"leftField": gee_constant(".all"), "rightValue": gee_call("Feature", {"geometry": ref("region")})})
    values["dateRange"] = gee_call(
        "Filter.dateRangeContains",
        {"leftValue": gee_call("DateRange", {"start": gee_constant(start_date), "end": gee_constant(end_date)}), "rightField": gee_constant("system:time_start")},
    )
    values["raw"] = gee_call("ImageCollection.load", {"id": gee_constant("COPERNICUS/S2_SR_HARMONIZED")})
    values["byRegion"] = gee_call("Collection.filter", {"collection": ref("raw"), "filter": ref("intersects")})
    values["byDate"] = gee_call("Collection.filter", {"collection": ref("byRegion"), "filter": ref("dateRange")})
    values["sorted"] = gee_call("Collection.limit", {"collection": ref("byDate"), "limit": gee_constant(1), "key": gee_constant("CLOUDY_PIXEL_PERCENTAGE"), "ascending": gee_constant(True)})
    values["image"] = gee_call("Collection.first", {"collection": ref("sorted")})
    values["visualized"] = build_true_color_visualize_expression(ref("image"))

    size_px = 640
    scale_meters = meters_per_pixel_at_zoom(lat, zoom)
    half_size_meters = (size_px / 2) * scale_meters
    center_x, center_y = lng_lat_to_mercator_meters(lng, lat)
    grid = PixelGrid(size_px, size_px, center_x - half_size_meters, center_y + half_size_meters, scale_meters)

    png_bytes = await compute_pixels_png(access_token, project_id, {"result": "visualized", "values": values}, grid)
    return base64.b64encode(png_bytes).decode("ascii")


async def capture_parcel_image(lat: float, lng: float, zoom: float, polygon: Any = None) -> str:
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY.startswith("VOTRE_"):
        raise RuntimeError("GOOGLE_MAPS_API_KEY n'est pas configurée.")

    params = {"center": f"{lat},{lng}", "zoom": str(zoom), "size": "640x640", "maptype": "satellite", "key": GOOGLE_MAPS_API_KEY}
    parcel_polygon = normalize_polygon(polygon)
    if parcel_polygon:
        path = "|".join(f"{p['lat']},{p['lng']}" for p in parcel_polygon)
        params["path"] = f"color:0xfbbf24ff|weight:2|{path}"

    resp = await fetch_with_retry("GET", "https://maps.googleapis.com/maps/api/staticmap", params=params, timeout_s=EXTERNAL_REQUEST_TIMEOUT_S)
    if resp.status_code >= 400:
        raise RuntimeError(f"Google Maps Static API error: {resp.status_code}")
    return base64.b64encode(resp.content).decode("ascii")


# ── HuggingFace model ──


class HFModelResult(dict):
    """is_barley: bool, confidence: float, prob_barley: float, prob_non_barley: float."""


async def call_hf_model(satellite_image_base64: str) -> HFModelResult:
    logger.info("Calling HF model /predict at: %s", HF_MODEL_URL)
    image_bytes = base64.b64decode(satellite_image_base64)

    resp = await fetch_with_retry(
        "POST",
        f"{HF_MODEL_URL}/predict",
        files={"file": ("parcel.png", image_bytes, "image/png")},
        timeout_s=EXTERNAL_REQUEST_TIMEOUT_S,
    )
    if resp.status_code >= 400:
        logger.error("HF /predict error: %s %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"HF model error: {resp.status_code}")

    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Réponse du modèle HF invalide.")

    is_barley = data.get("is_barley")
    confidence = data.get("confidence")
    prob_barley = data.get("prob_barley")
    prob_non_barley = data.get("prob_non_barley")
    if (
        not isinstance(is_barley, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not isinstance(prob_barley, (int, float))
        or not math.isfinite(prob_barley)
        or not isinstance(prob_non_barley, (int, float))
        or not math.isfinite(prob_non_barley)
    ):
        raise RuntimeError("La réponse du modèle HF ne contient pas les valeurs attendues.")

    return HFModelResult(is_barley=is_barley, confidence=confidence, prob_barley=prob_barley, prob_non_barley=prob_non_barley)


async def _call_hf_model_safely(image_base64: str, warnings: list[str]) -> HFModelResult | None:
    try:
        return await call_hf_model(image_base64)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"Modèle Hugging Face indisponible : {_error_message(error)}")
        logger.warning("[CLASSIFICATION] Modèle indisponible : %s", error)
        return None


# ── Satellite data (Sentinel-2 / Sentinel-1) ──


def _create_empty_satellite_data() -> dict[str, Any]:
    return {
        "ndvi": None, "ndwi": None, "evi": None, "savi": None,
        "blue": None, "nir": None, "red": None, "green": None, "swir": None,
        "vv": None, "vh": None, "vhVvRatio": None, "dataSource": [],
    }


def _build_s2_expression(start_date: str, end_date: str, polygon: Any) -> dict:
    return {
        "expression": {
            "result": "0",
            "values": {
                "1": {"functionInvocationValue": {"functionName": "GeometryConstructors.Polygon", "arguments": {"coordinates": {"constantValue": polygon_coordinates(polygon)}}}},
                "0": {
                    "functionInvocationValue": {
                        "functionName": "Image.reduceRegion",
                        "arguments": {
                            "geometry": {"valueReference": "1"},
                            "image": {
                                "functionInvocationValue": {
                                    "functionName": "reduce.median",
                                    "arguments": {
                                        "collection": {
                                            "functionInvocationValue": {
                                                "functionName": "Collection.filter",
                                                "arguments": {
                                                    "collection": {
                                                        "functionInvocationValue": {
                                                            "functionName": "Collection.filter",
                                                            "arguments": {
                                                                "collection": {
                                                                    "functionInvocationValue": {
                                                                        "functionName": "Collection.filter",
                                                                        "arguments": {
                                                                            "collection": {"functionInvocationValue": {"functionName": "ImageCollection.load", "arguments": {"id": {"constantValue": "COPERNICUS/S2_SR_HARMONIZED"}}}},
                                                                            "filter": {
                                                                                "functionInvocationValue": {
                                                                                    "functionName": "Filter.intersects",
                                                                                    "arguments": {
                                                                                        "leftField": {"constantValue": ".all"},
                                                                                        "rightValue": {"functionInvocationValue": {"functionName": "Feature", "arguments": {"geometry": {"valueReference": "1"}}}},
                                                                                    },
                                                                                }
                                                                            },
                                                                        },
                                                                    }
                                                                },
                                                                "filter": {
                                                                    "functionInvocationValue": {
                                                                        "functionName": "Filter.dateRangeContains",
                                                                        "arguments": {
                                                                            "leftValue": {"functionInvocationValue": {"functionName": "DateRange", "arguments": {"start": {"constantValue": start_date}, "end": {"constantValue": end_date}}}},
                                                                            "rightField": {"constantValue": "system:time_start"},
                                                                        },
                                                                    }
                                                                },
                                                            },
                                                        }
                                                    },
                                                    "filter": {
                                                        "functionInvocationValue": {
                                                            "functionName": "Filter.lessThan",
                                                            "arguments": {"leftField": {"constantValue": "CLOUDY_PIXEL_PERCENTAGE"}, "rightValue": {"constantValue": 30}},
                                                        }
                                                    },
                                                },
                                            }
                                        },
                                    },
                                }
                            },
                            "reducer": {"functionInvocationValue": {"functionName": "Reducer.mean", "arguments": {}}},
                            "scale": {"constantValue": 10},
                        },
                    }
                },
            },
        }
    }


def _build_s1_expression(start_date: str, end_date: str, polygon: Any) -> dict:
    return {
        "expression": {
            "result": "0",
            "values": {
                "1": {"functionInvocationValue": {"functionName": "GeometryConstructors.Polygon", "arguments": {"coordinates": {"constantValue": polygon_coordinates(polygon)}}}},
                "0": {
                    "functionInvocationValue": {
                        "functionName": "Image.reduceRegion",
                        "arguments": {
                            "geometry": {"valueReference": "1"},
                            "image": {
                                "functionInvocationValue": {
                                    "functionName": "reduce.median",
                                    "arguments": {
                                        "collection": {
                                            "functionInvocationValue": {
                                                "functionName": "Collection.filter",
                                                "arguments": {
                                                    "collection": {
                                                        "functionInvocationValue": {
                                                            "functionName": "Collection.filter",
                                                            "arguments": {
                                                                "collection": {
                                                                    "functionInvocationValue": {
                                                                        "functionName": "Collection.filter",
                                                                        "arguments": {
                                                                            "collection": {"functionInvocationValue": {"functionName": "ImageCollection.load", "arguments": {"id": {"constantValue": "COPERNICUS/S1_GRD"}}}},
                                                                            "filter": {
                                                                                "functionInvocationValue": {
                                                                                    "functionName": "Filter.intersects",
                                                                                    "arguments": {
                                                                                        "leftField": {"constantValue": ".all"},
                                                                                        "rightValue": {"functionInvocationValue": {"functionName": "Feature", "arguments": {"geometry": {"valueReference": "1"}}}},
                                                                                    },
                                                                                }
                                                                            },
                                                                        },
                                                                    }
                                                                },
                                                                "filter": {
                                                                    "functionInvocationValue": {
                                                                        "functionName": "Filter.dateRangeContains",
                                                                        "arguments": {
                                                                            "leftValue": {"functionInvocationValue": {"functionName": "DateRange", "arguments": {"start": {"constantValue": start_date}, "end": {"constantValue": end_date}}}},
                                                                            "rightField": {"constantValue": "system:time_start"},
                                                                        },
                                                                    }
                                                                },
                                                            },
                                                        }
                                                    },
                                                    "filter": {
                                                        "functionInvocationValue": {
                                                            "functionName": "Filter.equals",
                                                            "arguments": {"leftField": {"constantValue": "instrumentMode"}, "rightValue": {"constantValue": "IW"}},
                                                        }
                                                    },
                                                },
                                            }
                                        },
                                    },
                                }
                            },
                            "reducer": {"functionInvocationValue": {"functionName": "Reducer.mean", "arguments": {}}},
                            "scale": {"constantValue": 10},
                        },
                    }
                },
            },
        }
    }


def _parse_s2_bands(result: dict[str, float | None]) -> dict[str, float | None]:
    return {
        "nir": result.get("B8_mean") or result.get("B8_median") or result.get("B8"),
        "red": result.get("B4_mean") or result.get("B4_median") or result.get("B4"),
        "green": result.get("B3_mean") or result.get("B3_median") or result.get("B3"),
        "blue": result.get("B2_mean") or result.get("B2_median") or result.get("B2"),
        "swir": result.get("B11_mean") or result.get("B11_median") or result.get("B11"),
    }


def _parse_s1_bands(result: dict[str, float | None]) -> dict[str, float | None]:
    return {
        "vv": result.get("VV_mean") or result.get("VV_median") or result.get("VV"),
        "vh": result.get("VH_mean") or result.get("VH_median") or result.get("VH"),
    }


def _compute_spectral_indices(nir: float | None, red: float | None, blue: float | None, green: float | None, swir: float | None) -> dict[str, float | None]:
    ndvi = ndwi = evi = savi = None

    if nir is not None and red is not None and (nir + red) != 0:
        ndvi = js_round(((nir - red) / (nir + red)) * 1000) / 10

    if green is not None and nir is not None and (green + nir) != 0:
        ndwi = js_round(((green - nir) / (green + nir)) * 1000) / 1000

    if nir is not None and red is not None and blue is not None:
        denom = nir + 6 * red - 7.5 * blue + 10_000
        if denom != 0:
            raw_evi = 2.5 * (nir - red) / denom
            evi = js_round(raw_evi * 1000) / 10

    if nir is not None and red is not None and (nir + red + 0.5) != 0:
        l_factor = 0.5
        raw_savi = ((nir - red) / (nir + red + l_factor)) * (1 + l_factor)
        savi = js_round(raw_savi * 1000) / 10

    return {"ndvi": ndvi, "ndwi": ndwi, "evi": evi, "savi": savi}


async def _fetch_current_snapshot(access_token: str, lat: float, lng: float, project_id: str, polygon: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    end = now.date().isoformat()
    start = (now - timedelta(days=180)).date().isoformat()

    s2_result, s1_result = await asyncio.gather(
        call_gee_compute(access_token, project_id, _build_s2_expression(start, end, polygon)),
        call_gee_compute(access_token, project_id, _build_s1_expression(start, end, polygon)),
    )

    result = _create_empty_satellite_data()

    if s2_result:
        bands = _parse_s2_bands(s2_result)
        result["nir"], result["red"], result["green"], result["blue"], result["swir"] = bands["nir"], bands["red"], bands["green"], bands["blue"], bands["swir"]
        indices = _compute_spectral_indices(bands["nir"], bands["red"], bands["blue"], bands["green"], bands["swir"])
        result["ndvi"] = js_round(indices["ndvi"]) if indices["ndvi"] is not None else None
        result["ndwi"] = indices["ndwi"]
        result["evi"] = js_round(indices["evi"] * 10) / 10 if indices["evi"] is not None else None
        result["savi"] = js_round(indices["savi"] * 10) / 10 if indices["savi"] is not None else None
        result["dataSource"].append("Sentinel-2")

    if s1_result:
        bands1 = _parse_s1_bands(s1_result)
        if bands1["vv"] is not None:
            result["vv"] = js_round(bands1["vv"] * 100) / 100
        if bands1["vh"] is not None:
            result["vh"] = js_round(bands1["vh"] * 100) / 100
        if bands1["vv"] is not None and bands1["vh"] is not None and bands1["vv"] != 0:
            result["vhVvRatio"] = js_round((bands1["vh"] / bands1["vv"]) * 1000) / 1000
        result["dataSource"].append("Sentinel-1")

    return result


# ── Time series ──


def _shift_month(year: int, month0: int, delta: int) -> tuple[int, int]:
    total = year * 12 + month0 + delta
    return total // 12, total % 12


def get_monthly_ranges(num_months: int) -> list[dict[str, str]]:
    now = datetime.now()
    ranges: list[dict[str, str]] = []
    for i in range(num_months - 1, -1, -1):
        y, m0 = _shift_month(now.year, now.month - 1, -i)
        start = f"{y:04d}-{m0 + 1:02d}-01"
        ey, em0 = _shift_month(y, m0, 2)
        end_y, end_m0 = _shift_month(ey, em0, -1)
        last_day = calendar.monthrange(end_y, end_m0 + 1)[1]
        end = f"{end_y:04d}-{end_m0 + 1:02d}-{last_day:02d}"
        ranges.append({"label": start[:7], "start": start, "end": end})
    return ranges


T = TypeVar("T")
R = TypeVar("R")


async def map_with_concurrency(items: list[T], concurrency: int, mapper: Callable[[T], Awaitable[R]]) -> list[R]:
    results: list[R | None] = [None] * len(items)
    cursor = 0
    cursor_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal cursor
        while True:
            async with cursor_lock:
                if cursor >= len(items):
                    return
                index = cursor
                cursor += 1
            results[index] = await mapper(items[index])

    worker_count = min(concurrency, len(items))
    if worker_count > 0:
        await asyncio.gather(*(worker() for _ in range(worker_count)))
    return results  # type: ignore[return-value]


async def _fetch_time_series(access_token: str, project_id: str, polygon: Any) -> dict[str, list[dict[str, Any]]]:
    months = get_monthly_ranges(6)

    async def s2_mapper(m: dict[str, str]) -> dict[str, Any]:
        result = await call_gee_compute(access_token, project_id, _build_s2_expression(m["start"], m["end"], polygon))
        ndvi = None
        if result:
            bands = _parse_s2_bands(result)
            if bands["nir"] is not None and bands["red"] is not None and (bands["nir"] + bands["red"]) != 0:
                ndvi = js_round(((bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"])) * 1000) / 10
        return {"date": m["label"], "ndvi": ndvi, "cloud_cover": None}

    async def s1_mapper(m: dict[str, str]) -> dict[str, Any]:
        result = await call_gee_compute(access_token, project_id, _build_s1_expression(m["start"], m["end"], polygon))
        vv = vh = None
        if result:
            parsed = _parse_s1_bands(result)
            if parsed["vv"] is not None:
                vv = js_round(parsed["vv"] * 100) / 100
            if parsed["vh"] is not None:
                vh = js_round(parsed["vh"] * 100) / 100
        return {"date": m["label"], "vv": vv, "vh": vh}

    s2 = await map_with_concurrency(months, TIME_SERIES_CONCURRENCY, s2_mapper)
    s1 = await map_with_concurrency(months, TIME_SERIES_CONCURRENCY, s1_mapper)
    return {"s2": s2, "s1": s1}


async def _fetch_precipitation_time_series(lat: float, lng: float, months: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not months:
        return []

    archive_cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    cutoff_str = archive_cutoff.date().isoformat()
    start_date = months[0]["start"]
    last_month_end = months[-1]["end"]
    end_date = cutoff_str if last_month_end > cutoff_str else last_month_end

    params = {"latitude": str(lat), "longitude": str(lng), "start_date": start_date, "end_date": end_date, "daily": "precipitation_sum", "timezone": "auto"}
    response = await fetch_with_retry("GET", "https://archive-api.open-meteo.com/v1/archive", params=params, timeout_s=EXTERNAL_REQUEST_TIMEOUT_S)
    if response.status_code >= 400:
        raise RuntimeError("Les données de précipitations Open-Meteo sont indisponibles.")
    payload = response.json()
    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict):
        raise RuntimeError("Réponse météo (pluie) incomplète.")
    dates = daily.get("time")
    precip_values = daily.get("precipitation_sum")
    if not isinstance(dates, list) or not isinstance(precip_values, list):
        raise RuntimeError("Précipitations journalières indisponibles.")

    monthly_totals: dict[str, float] = {}
    for date, value in zip(dates, precip_values):
        if not isinstance(date, str) or not isinstance(value, (int, float)):
            continue
        month_label = date[:7]
        monthly_totals[month_label] = monthly_totals.get(month_label, 0) + value

    return [
        {"date": m["label"], "precipitation_mm": js_round(monthly_totals[m["label"]] * 10) / 10 if m["label"] in monthly_totals else None}
        for m in months
    ]


# ── Planting date detection ──


def detect_planting_date(s2: list[dict[str, Any]], s1: list[dict[str, Any]]) -> dict[str, Any]:
    valid_s2 = [p for p in s2 if p["ndvi"] is not None]
    empty = {"estimated_planting_date": None, "estimated_harvest_date": None, "days_since_planting": None, "growth_stage": None, "planting_confidence": 0}
    if len(valid_s2) < 2:
        return empty

    max_jump = 0.0
    jump_index = -1
    for i in range(1, len(valid_s2)):
        delta = valid_s2[i]["ndvi"] - valid_s2[i - 1]["ndvi"]
        if delta > max_jump:
            max_jump = delta
            jump_index = i

    if max_jump < 15 or jump_index < 0:
        min_idx = 0
        for i, p in enumerate(valid_s2):
            if p["ndvi"] < valid_s2[min_idx]["ndvi"]:
                min_idx = i
        if min_idx < len(valid_s2) - 1 and valid_s2[min_idx + 1]["ndvi"] - valid_s2[min_idx]["ndvi"] > 5:
            jump_index = min_idx + 1
            max_jump = valid_s2[jump_index]["ndvi"] - valid_s2[min_idx]["ndvi"]
        else:
            return empty

    jump_month = valid_s2[jump_index]["date"]
    year_str, month_str = jump_month.split("-")
    year, month = int(year_str), int(month_str)
    planting_dt = datetime(year, month, 10).astimezone()

    confidence = min(90, js_round(max_jump * 2))
    if s1 and jump_index < len(s1):
        s1_jump = s1[jump_index]
        s1_prev = s1[jump_index - 1] if jump_index > 0 else None
        if s1_jump["vv"] is not None and s1_prev is not None and s1_prev["vv"] is not None and s1_jump["vv"] - s1_prev["vv"] > 0.5:
            confidence = min(95, confidence + 10)

    now = datetime.now().astimezone()
    days_since_planting = js_round((now - planting_dt).total_seconds() / 86400)

    if days_since_planting < 0:
        growth_stage = "Pré-semis"
    elif days_since_planting <= 7:
        growth_stage = "Semis"
    elif days_since_planting <= 20:
        growth_stage = "Levée"
    elif days_since_planting <= 45:
        growth_stage = "Tallage"
    elif days_since_planting <= 65:
        growth_stage = "Montaison"
    elif days_since_planting <= 80:
        growth_stage = "Épiaison"
    elif days_since_planting <= 110:
        growth_stage = "Maturation"
    else:
        growth_stage = "Récolte"

    harvest_dt = planting_dt + timedelta(days=100)

    return {
        "estimated_planting_date": planting_dt.astimezone(timezone.utc).date().isoformat(),
        "estimated_harvest_date": harvest_dt.astimezone(timezone.utc).date().isoformat(),
        "days_since_planting": max(0, days_since_planting),
        "growth_stage": growth_stage,
        "planting_confidence": confidence,
    }


# ── Agronomic rules scoring ──


def compute_agro_score(sat_data: dict[str, Any], planting: dict[str, Any], time_series: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ndvi_score = 0.0
    cycle_score = 0.0
    radar_score = 0.0
    ndvi_trend_score = 0.0
    details: list[str] = []

    if sat_data["ndvi"] is not None:
        ndvi = sat_data["ndvi"]
        if 25 <= ndvi <= 75:
            ndvi_score = 80 + (1 - abs(ndvi - 50) / 25) * 20
            details.append(f"NDVI {ndvi}% dans la plage orge")
        elif 15 <= ndvi <= 85:
            ndvi_score = 40
            details.append(f"NDVI {ndvi}% limite pour l'orge")
        else:
            ndvi_score = 10
            details.append(f"NDVI {ndvi}% hors plage orge typique")

    if planting["days_since_planting"] is not None and planting["growth_stage"]:
        days = planting["days_since_planting"]
        if 0 <= days <= 130:
            cycle_score = 80
            if sat_data["ndvi"] is not None:
                ndvi = sat_data["ndvi"]
                stage = planting["growth_stage"]
                if stage == "Levée" and 15 <= ndvi <= 40:
                    cycle_score = 95
                elif stage == "Tallage" and 30 <= ndvi <= 65:
                    cycle_score = 95
                elif stage == "Montaison" and 40 <= ndvi <= 80:
                    cycle_score = 95
                elif stage == "Épiaison" and 50 <= ndvi <= 80:
                    cycle_score = 95
                elif stage == "Maturation" and 30 <= ndvi <= 60:
                    cycle_score = 90
                elif stage == "Récolte" and 10 <= ndvi <= 40:
                    cycle_score = 90
            details.append(f"Cycle {days}j cohérent (stade {planting['growth_stage']})")
        elif days > 130:
            cycle_score = 30
            details.append(f"Cycle {days}j > 130j, dépasse le cycle orge")
    else:
        cycle_score = 50

    if sat_data["vv"] is not None and sat_data["vh"] is not None:
        vv = sat_data["vv"]
        vh = sat_data["vh"]
        if -14 <= vv <= -8 and -22 <= vh <= -14:
            radar_score = 90
            details.append(f"Radar VV={vv}dB VH={vh}dB typique cultures")
        elif -16 <= vv <= -6 and -24 <= vh <= -12:
            radar_score = 60
            details.append("Radar acceptable mais pas typique orge")
        else:
            radar_score = 20
            details.append("Signature radar atypique pour céréales")

        if sat_data["vhVvRatio"] is not None:
            ratio = sat_data["vhVvRatio"]
            if 0.3 <= ratio <= 0.6:
                radar_score = min(100, radar_score + 10)
    else:
        radar_score = 50

    valid_ndvi = [p["ndvi"] for p in time_series["s2"] if p["ndvi"] is not None]
    if len(valid_ndvi) >= 3:
        max_ndvi = max(valid_ndvi)
        min_ndvi = min(valid_ndvi)
        rng = max_ndvi - min_ndvi
        if rng >= 15:
            ndvi_trend_score = 70 + min(30, rng)
            details.append(f"Dynamique NDVI {rng:.0f}% (bon signal cultural)")
        elif rng >= 5:
            ndvi_trend_score = 50
            details.append(f"Dynamique NDVI faible ({rng:.0f}%)")
        else:
            ndvi_trend_score = 20
            details.append("NDVI stable - pas typique d'une culture")
    else:
        ndvi_trend_score = 50

    score = js_round(ndvi_score * 0.3 + cycle_score * 0.25 + radar_score * 0.25 + ndvi_trend_score * 0.2)

    return {
        "score": max(0, min(100, score)),
        "breakdown": {
            "ndvi_score": js_round(ndvi_score),
            "cycle_score": js_round(cycle_score),
            "radar_score": js_round(radar_score),
            "ndvi_trend_score": js_round(ndvi_trend_score),
        },
        "details": " | ".join(details),
    }


# ── Hybrid score fusion ──


def compute_hybrid_score(cnn_confidence: float, cnn_is_barley: bool, agro_score: float) -> dict[str, Any]:
    cnn_score = cnn_confidence if cnn_is_barley else (100 - cnn_confidence)
    hybrid_score = js_round(cnn_score * 0.6 + agro_score * 0.4)

    disagreement = abs(cnn_score - agro_score)
    penalty = disagreement * 0.08 if disagreement > 55 else 0
    final_confidence = max(0, min(100, js_round(hybrid_score - penalty)))

    # Seulement deux catégories : ORGE CONFIRMÉ vs NON-ORGE. Seuil à 50% (corrigé lors de
    # l'audit du portage Python — le code source TS portait `> 44` mais son propre
    # commentaire documentait l'intention `> 50` : "Remonté à 50% (depuis 43%)... 50% exige
    # une vraie majorité"; décision utilisateur du 2026-09-04 de suivre l'intention
    # documentée). Un score Agro élevé (souvent 70-95%, ses composantes retombant sur des
    # valeurs neutres généreuses quand une donnée manque) suffisait seul à faire basculer en
    # orge même quand le CNN penchait pour non-orge — 50% exige une vraie majorité CNN+Agro
    # en faveur de l'orge.
    final_is_barley = hybrid_score > 50

    verdict = "✅ ORGE CONFIRMÉE — CNN + Règles agro concordent" if final_is_barley else "❌ NON-ORGE — CNN + Règles agro concordent"
    if disagreement > 55:
        verdict += f" (⚠️ désaccord CNN/Agro: {disagreement:.0f}%)"

    return {"hybrid_score": hybrid_score, "final_is_barley": final_is_barley, "final_verdict": verdict, "final_confidence": final_confidence}


def create_agro_only_hybrid_score(agro_score: float) -> dict[str, Any]:
    final_is_barley = agro_score > 50
    verdict = (
        f"⚠️ ORGE PROBABLE (CNN indisponible, estimation agro seule) — Score agro {agro_score}%"
        if final_is_barley
        else f"⚠️ NON-ORGE PROBABLE (CNN indisponible, estimation agro seule) — Score agro {agro_score}%"
    )
    return {"hybrid_score": agro_score, "final_is_barley": final_is_barley, "final_verdict": verdict, "final_confidence": agro_score}


# ── Season detection ──


def detect_season(lat: float) -> str:
    calendar_month = datetime.now().month  # 1-12, cycle calé sur l'hémisphère nord (semis oct-nov)
    month = (((calendar_month - 1 + 6) % 12) + 1) if lat < 0 else calendar_month

    if month in (10, 11):
        return "Semis / Levée"
    if month in (12, 1, 2):
        return "Tallage"
    if month in (3, 4):
        return "Montaison"
    if month == 5:
        return "Épiaison"
    if month == 6:
        return "Maturation"
    if month in (7, 8):
        return "Récolte / Post-récolte"
    return "Jachère"  # septembre


# ── SNIC barley segment detection ──


def _feature_number(properties: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        value = properties.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return value
    return None


def _to_segment_candidates(feature: Any) -> list[dict[str, Any]]:
    if not isinstance(feature, dict):
        return []
    coordinates = extract_latlng_from_geometry(feature.get("geometry"))
    if len(coordinates) < 3:
        return []

    area_m2 = approximate_polygon_area_m2(coordinates)
    if area_m2 < MIN_SEGMENT_AREA_M2 or area_m2 > MAX_SEGMENT_AREA_M2:
        return []

    properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    ndvi = _feature_number(properties, ["NDVI", "NDVI_mean", "ndvi", "ndvi_mean"])
    ndwi = _feature_number(properties, ["NDWI", "NDWI_mean", "ndwi", "ndwi_mean"])
    ndbi = _feature_number(properties, ["NDBI", "NDBI_mean", "ndbi", "ndbi_mean"])

    if (ndvi is not None and ndvi < 0.2) or (ndwi is not None and ndwi > 0.1) or (ndbi is not None and ndbi > 0.05):
        return []

    return [{"coordinates": coordinates, "areaM2": area_m2, "ndvi": ndvi}]


def _build_snic_vectors_expression(polygon: Any) -> dict:
    end_date = datetime.now(timezone.utc).date().isoformat()
    start_date = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
    values: dict[str, GeeValue] = {}

    def reference(name: str) -> GeeValue:
        return {"valueReference": name}

    values["region"] = gee_call("GeometryConstructors.Polygon", {"coordinates": gee_constant(polygon_coordinates(polygon))})
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

    values["ndviMask"] = gee_call("Image.gt", {"image1": image_band("NDVI"), "image2": gee_image_constant(0.2)})
    values["waterMask"] = gee_call("Image.lt", {"image1": image_band("NDWI"), "image2": gee_image_constant(0.1)})
    values["urbanMask"] = gee_call("Image.lt", {"image1": image_band("NDBI"), "image2": gee_image_constant(0.05)})
    values["bareSoilMask"] = gee_call("Image.lt", {"image1": image_band("BSI"), "image2": gee_image_constant(0.2)})
    values["vegetationMask"] = gee_call(
        "Image.and",
        {
            "image1": gee_call("Image.and", {"image1": gee_call("Image.and", {"image1": reference("ndviMask"), "image2": reference("waterMask")}), "image2": reference("urbanMask")}),
            "image2": reference("bareSoilMask"),
        },
    )
    values["maskedImage"] = gee_call("Image.updateMask", {"image": reference("segmentationImage"), "mask": reference("vegetationMask")})
    values["vegetationImage"] = gee_call("Image.clip", {"input": reference("maskedImage"), "geometry": reference("region")})
    values["snic"] = gee_call(
        "Image.Segmentation.SNIC",
        {
            "image": reference("vegetationImage"),
            "size": gee_constant(SNIC_PARAMETERS["size"]),
            "compactness": gee_constant(SNIC_PARAMETERS["compactness"]),
            "connectivity": gee_constant(SNIC_PARAMETERS["connectivity"]),
            "neighborhoodSize": gee_constant(SNIC_PARAMETERS["neighborhoodSize"]),
        },
    )
    values["snicClusters"] = gee_call("Image.select", {"input": reference("snic"), "bandSelectors": gee_constant(["clusters"])})
    values["vectorsImage"] = gee_call("Image.addBands", {"dstImg": reference("snicClusters"), "srcImg": reference("vegetationImage")})
    values["vectors"] = gee_call(
        "Image.reduceToVectors",
        {
            "image": reference("vectorsImage"),
            "reducer": gee_call("Reducer.mean", {}),
            "geometry": reference("region"),
            "scale": gee_constant(SNIC_PARAMETERS["scale"]),
            "geometryType": gee_constant("polygon"),
            "eightConnected": gee_constant(True),
            "labelProperty": gee_constant("segment_id"),
            "bestEffort": gee_constant(True),
            "maxPixels": gee_constant(10_000_000),
            "tileScale": gee_constant(2),
        },
    )

    return {"expression": {"result": "vectors", "values": values}}


async def _detect_barley_segments(
    access_token: str, project_id: str, polygon: Any, fallback_zoom: int, warnings: list[str]
) -> list[dict[str, Any]]:
    try:
        logger.info("[SNIC] Segmentation démarrée")
        raw = await call_gee_compute_raw(access_token, project_id, _build_snic_vectors_expression(polygon))
        result = raw.get("result")
        if not is_gee_feature_collection(result):
            logger.warning("[SNIC] Aucune image Sentinel-2 exploitable ou aucun objet vectorisé.")
            return []

        logger.info("[SNIC] Image Sentinel-2 récupérée")
        logger.info("[SNIC] Nombre d'objets détectés : %s", len(result["features"]))

        candidates: list[dict[str, Any]] = []
        for feature in result["features"]:
            candidates.extend(_to_segment_candidates(feature))
        candidates.sort(key=lambda c: (-(c["ndvi"] if c["ndvi"] is not None else -1), -c["areaM2"]))
        candidates = candidates[:MAX_SEGMENTS_TO_CLASSIFY]

        logger.info("[CLASSIFICATION] Objets envoyés au modèle : %s", len(candidates))

        async def classify(candidate: dict[str, Any]) -> dict[str, Any] | None:
            try:
                center = polygon_centroid(candidate["coordinates"])
                thumbnail = await capture_parcel_image(
                    center["lat"], center["lng"], segment_zoom_for_area(candidate["areaM2"], center["lat"], fallback_zoom), candidate["coordinates"]
                )
                classification = await call_hf_model(thumbnail)
                if classification["is_barley"] and classification["confidence"] >= MIN_BARLEY_CONFIDENCE:
                    return {
                        "coordinates": candidate["coordinates"],
                        "confidence": js_round(classification["confidence"]),
                        "ndvi": candidate["ndvi"],
                        "area_ha": js_round((candidate["areaM2"] / 10_000) * 100) / 100,
                    }
                return None
            except Exception as error:  # noqa: BLE001
                logger.warning("[CLASSIFICATION] Échec de la classification d'un objet : %s", error)
                return None

        classified = await map_with_concurrency(candidates, SEGMENT_CLASSIFY_CONCURRENCY, classify)
        barley_segments = [s for s in classified if s is not None]

        logger.info("[CLASSIFICATION] Orge détectée : %s", len(barley_segments))
        return barley_segments
    except Exception as error:  # noqa: BLE001
        warnings.append(f"SNIC indisponible : {_error_message(error)}")
        logger.warning("[SNIC] Échec de la segmentation : %s", error)
        return []


# ── Main handler ──


async def _resolved(value: Any) -> Any:
    return value


async def analyze_parcel(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        lat = body.get("lat")
        lng = body.get("lng")
        zoom = body.get("zoom")
        polygon = body.get("polygon")

        if not isinstance(lat, (int, float)) or isinstance(lat, bool) or not isinstance(lng, (int, float)) or isinstance(lng, bool):
            return 400, {"error": "lat et lng sont requis (nombres)"}

        parcel_polygon = normalize_polygon(polygon)
        if parcel_polygon is None:
            return 400, {"error": "Le contour réel de la parcelle est requis (au moins 3 points valides)."}

        effective_zoom = zoom or 16
        warnings: list[str] = []
        project_id = "earthengine-legacy"
        access_token: str | None = None
        service_account_json = settings.gee_service_account_key

        if not service_account_json or service_account_json.startswith("VOTRE_"):
            warnings.append("GEE_SERVICE_ACCOUNT_KEY n'est pas configurée.")
        else:
            try:
                sa = json.loads(service_account_json)
                pid = sa.get("project_id")
                if isinstance(pid, str) and pid:
                    project_id = pid
                access_token = await get_gee_access_token()
            except Exception as error:  # noqa: BLE001
                warnings.append(f"GEE indisponible : {_error_message(error)}")

        snapshot_task = _fetch_current_snapshot(access_token, lat, lng, project_id, parcel_polygon) if access_token else _resolved(None)
        time_series_task = _fetch_time_series(access_token, project_id, parcel_polygon) if access_token else _resolved(None)
        rain_task = _fetch_precipitation_time_series(lat, lng, get_monthly_ranges(6))

        snapshot_result, time_series_result, rain_result = await asyncio.gather(snapshot_task, time_series_task, rain_task, return_exceptions=True)

        sat_data = snapshot_result if snapshot_result and not isinstance(snapshot_result, BaseException) else _create_empty_satellite_data()
        time_series = time_series_result if time_series_result and not isinstance(time_series_result, BaseException) else {"s2": [], "s1": []}
        time_series_rain = rain_result if not isinstance(rain_result, BaseException) else []

        if isinstance(snapshot_result, BaseException):
            warnings.append(f"Sentinel-2 indisponible : {_error_message(snapshot_result)}")
        if isinstance(time_series_result, BaseException):
            warnings.append(f"Séries temporelles indisponibles : {_error_message(time_series_result)}")
        if isinstance(rain_result, BaseException):
            warnings.append(f"Précipitations indisponibles : {_error_message(rain_result)}")

        if access_token:
            detected_segments = await _detect_barley_segments(access_token, project_id, parcel_polygon, effective_zoom, warnings)
        else:
            detected_segments = []
            warnings.append("SNIC non exécuté : authentification GEE indisponible.")

        satellite_image: str | None = None
        try:
            satellite_image = await capture_parcel_image(lat, lng, effective_zoom, parcel_polygon)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"Google Static Maps indisponible : {_error_message(error)}")

        planting = detect_planting_date(time_series["s2"], time_series["s1"])
        agro = compute_agro_score(sat_data, planting, time_series)
        hf_result = await _call_hf_model_safely(satellite_image, warnings) if satellite_image else None
        hybrid = compute_hybrid_score(hf_result["confidence"], hf_result["is_barley"], agro["score"]) if hf_result else create_agro_only_hybrid_score(agro["score"])
        season = detect_season(lat)

        data_source_parts = ["Contour parcellaire réel", *sat_data["dataSource"]]
        if hf_result:
            data_source_parts.append(f"HF OrgeDetector ({HF_MODEL_URL})")
        data_source = " + ".join(data_source_parts)

        spectral_parts = []
        if sat_data["ndvi"] is not None:
            spectral_parts.append(f"NDVI={sat_data['ndvi']}%")
        if sat_data["evi"] is not None:
            spectral_parts.append(f"EVI={sat_data['evi']}%")
        if sat_data["savi"] is not None:
            spectral_parts.append(f"SAVI={sat_data['savi']}%")
        if sat_data["ndwi"] is not None:
            spectral_parts.append(f"NDWI={sat_data['ndwi']}")
        spectral_analysis = ", ".join(spectral_parts) if spectral_parts else None

        radar_analysis = None
        if sat_data["vv"] is not None and sat_data["vh"] is not None:
            radar_analysis = f"VV={sat_data['vv']}dB, VH={sat_data['vh']}dB"
            if sat_data["vhVvRatio"] is not None:
                radar_analysis += f", VH/VV={sat_data['vhVvRatio']}"

        risk_factors = []
        if sat_data["ndvi"] is not None and sat_data["ndvi"] < 15:
            risk_factors.append("Végétation très faible")
        if sat_data["ndvi"] is not None and sat_data["ndvi"] > 85:
            risk_factors.append("Végétation trop dense pour de l'orge")
        if sat_data["vhVvRatio"] is not None and sat_data["vhVvRatio"] > 0.45:
            risk_factors.append("Structure radar type forêt")
        if not hybrid["final_is_barley"] and hybrid["final_confidence"] > 70:
            risk_factors.append("Score hybride confirme non-orge")

        detail_parts = []
        detail_parts.append(f"CNN: {'orge' if hf_result['is_barley'] else 'non-orge'} ({hf_result['confidence']:.1f}%)" if hf_result else "CNN: indisponible")
        detail_parts.append(f"Agro: {agro['score']}%")
        detail_parts.append(f"Hybride: {hybrid['hybrid_score']}%")

        soil_type = None
        if sat_data["swir"] is not None and sat_data["nir"] is not None and sat_data["nir"] != 0:
            ratio = sat_data["swir"] / sat_data["nir"]
            soil_type = "Sol argileux sec" if ratio > 1.5 else "Sol limoneux" if ratio > 1.0 else "Sol humide"

        if hf_result:
            recommendations = (
                f"Orge {'confirmée' if hybrid['final_confidence'] > 70 else 'probable'}. Score hybride {hybrid['hybrid_score']}% (CNN {js_round(hf_result['confidence'])}% + Agro {agro['score']}%)."
                if hybrid["final_is_barley"]
                else f"Non-orge détecté. Score hybride {hybrid['hybrid_score']}%. Vérification terrain recommandée."
            )
        else:
            recommendations = (
                f"{'Orge probable' if hybrid['final_is_barley'] else 'Non-orge probable'} (modèle CNN indisponible, estimation basée uniquement sur les règles agronomiques, score {agro['score']}%). Vérification terrain recommandée."
            )

        response = {
            "percentage": sat_data["ndvi"],
            "is_barley": hf_result["is_barley"] is True if hf_result else False,
            "detected_segments": detected_segments,
            "hf_model_url": HF_MODEL_URL,
            "hf_available": hf_result is not None,
            "verdict": hybrid["final_verdict"],
            "details": " | ".join(detail_parts),
            "culture_detected": ("Orge" if hf_result["is_barley"] else "Non-orge") if hf_result else None,
            "confidence": hf_result["confidence"] if hf_result else None,
            "cnn_prob_barley": hf_result["prob_barley"] if hf_result else None,
            "cnn_prob_non_barley": hf_result["prob_non_barley"] if hf_result else None,
            "cnn_available": hf_result is not None,
            "evi": sat_data["evi"],
            "savi": sat_data["savi"],
            "ndwi": sat_data["ndwi"],
            "agro_score": agro["score"],
            "agro_breakdown": agro["breakdown"],
            "agro_details": agro["details"],
            "hybrid_score": hybrid["hybrid_score"],
            "saison": season,
            "soil_type": soil_type,
            "risk_factors": risk_factors,
            "boundary_source": "Contour fourni par l'utilisateur, le cadastre ou une source agricole fiable",
            "recommendations": recommendations,
            "anomaly_level": "AUCUNE" if hybrid["final_is_barley"] else "FORTE",
            "data_source": data_source,
            "warnings": warnings,
            "radar_analysis": radar_analysis,
            "spectral_analysis": spectral_analysis,
            "time_series_s2": time_series["s2"],
            "time_series_s1": time_series["s1"],
            "time_series_rain": time_series_rain,
            "estimated_planting_date": planting["estimated_planting_date"],
            "estimated_harvest_date": planting["estimated_harvest_date"],
            "days_since_planting": planting["days_since_planting"],
            "growth_stage": planting["growth_stage"],
            "planting_confidence": planting["planting_confidence"],
        }
        return 200, response
    except Exception as error:  # noqa: BLE001
        logger.error("analyze-parcel error: %s", error, exc_info=error)
        return 500, {"error": str(error) or "Unknown error"}
