"""Micro-segmentation : découpe une zone en micro-parcelles suivant les clôtures visibles
sur l'imagerie Google haute-résolution (~0.3 m/px au zoom 19), puis analyse spectrale
Sentinel-2 + GDD de chaque micro-parcelle.

Pipeline :
  1. capture Google Static Maps satellite (640x640, zoom 19) centrée sur le point
  2. lissage gaussien (écrase la texture intra-parcelle, garde les clôtures)
  3. watershed marqué sur le gradient (luminance + excès de vert)
  4. fusion des segments adjacents si intérieurs similaires ET frontière faible
     (une vraie clôture = fort gradient de frontière -> pas de fusion)
  5. vectorisation des contours -> polygones lat/lng
  6. pour chaque micro-parcelle : indices Sentinel-2 moyens (GEE reduceRegion)
     + classification multi-cultures + GDD (même logique que analyze_fields_simple)

Dépendances : numpy + Pillow (décodage PNG) + shapely (validation géométries).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from sqlalchemy import select

from app.config import settings
from app.db import async_session_maker
from app.models import Parcelle
from app.services.analyze_parcel import (
    _build_s2_expression,
    _compute_spectral_indices,
    _parse_s2_bands,
    call_gee_compute,
    capture_parcel_image,
    get_gee_access_token,
    get_gee_project_id,
    get_monthly_ranges,
    js_round,
    map_with_concurrency,
    meters_per_pixel_at_zoom,
    polygon_centroid,
)
from app.services.automatic_parcels import fetch_growing_degree_days
from app.services.barley_detect_simple import BARLEY_GDD_THRESHOLD, classify_crop_signature, resolve_cereal
from app.services.field_watershed import (
    lng_lat_to_mercator_meters,
    mercator_meters_to_lng_lat,
    simplify_polygon,
    trace_label_contours,
    watershed_segment,
)

logger = logging.getLogger("agrisat.micro_parcels")

MICRO_ZOOM = 19
MICRO_IMAGE_PX = 640
# Rayon de découpage (comme l'analyse simple : 50 m à 20 km là-bas, 10 m à 5 km
# ici — la segmentation HR à ~0.3 m/px impose une borne basse plus fine).
MICRO_MIN_RADIUS_M = 10.0
MICRO_MAX_RADIUS_M = 5_000.0
DEFAULT_MICRO_RADIUS_M = 150.0
# Une imagette zoom 19 couvre ~190 m de côté : au-delà, on pave la zone en
# tuiles (grille centrée, pas de ~120 m soit ~37 % de recouvrement) comme
# l'analyse simple (TILE_RADIUS_M). Un pas de 150 m ne laissait que ~40 m de
# recouvrement : les clôtures coupées au bord d'une tuile étaient perdues dans
# l'autre (barrière de bordure) -> trous entre polygones aux raccords.
MICRO_TILE_STEP_M = 120.0
# Plafond de tuiles relevé (64 -> 100) : un rayon de 500 m demande déjà ~60
# tuiles avec ce pas ; tronquer silencieusement = des secteurs entiers sans
# polygones (trous). Au-delà, on prévient (warning) au lieu de trous muets.
MICRO_MAX_TILES = 100
MICRO_TILE_CONCURRENCY = 6
MICRO_BLUR_RADIUS = 2
MICRO_SEED_PERCENTILE = 0.30
MICRO_MIN_SEED_PIXELS = 150
MICRO_MIN_AREA_M2 = 300
# Seuil après découpe au rayon : un champ à cheval sur le cercle ne laisse
# parfois qu'un copeau < 300 m² dans le disque — le jeter crée un trou au
# centre. On garde les copeaux >= 100 m² (le spectral S2 reste lisible).
MICRO_MIN_AREA_CLIPPED_M2 = 100.0
MICRO_MAX_AREA_M2 = 80_000
# Filet anti-trous : si la segmentation HR laisse des vides (champs homogènes
# sans clôture visible, copeaux filtrés...), on bouche avec des mailles
# régulières de 50 m. Seuil de couverture en dessous duquel on complète.
MICRO_FALLBACK_CELL_M = 50.0
MICRO_MAX_FALLBACK_CELLS = 40
MICRO_MIN_COVERAGE_RATIO = 0.35
# Plafond relevé (40 -> 120) : avec un rayon de 150 m (~7 ha) et des clôtures
# fines, on dépasse vite 40 micro-zones ; tronquer aux plus grandes laissait
# des trous (les petites zones n'avaient aucun polygone).
MICRO_MAX_MICRO_PARCELS = 120
MICRO_SIMPLIFY_EPS_PX = 1.5
# Fusion : seuils calibrés sur zone test (-19.848219, 47.011882)
MICRO_MERGE_MAX_D_LUM = 7.0
MICRO_MERGE_MAX_D_EXG = 6.0
MICRO_MERGE_MAX_BOUNDARY_GRAD = 0.35
MICRO_CLASSIFY_CONCURRENCY = 8
# Suivi thermique des micro-zones d'orge : série NDVI mensuelle GEE (6 mois,
# même primitive que analyze-parcel) → track_parcel (semis + ST/stade).
# Concurrence basse : 1 appel GEE par mois et par micro-zone.
MICRO_PHENO_CONCURRENCY = 4
MICRO_PHENO_MONTHS = 6
# Confirmation indépendante (Sentinel-2 + Landsat + S1 + HR + météo) appliquée à
# chaque micro-zone dès le découpage : le résultat final affiché est le verdict
# confirmé, pas la seule signature Sentinel-2.
MICRO_CONFIRM_CONCURRENCY = 6
MICRO_CONFIRM_TIMEOUT_S = 30.0
# Érosion du polygone avant l'analyse spectrale : retire une bande de 10 m sur le
# pourtour (berges, clôtures, pixels mixtes Sentinel-2) pour analyser le cœur pur.
# Sans ça, un lac est dilué par ses berges (NDVI remonte, NDWI chute -> « Riz »).
MICRO_SPECTRAL_EROSION_M = 10.0
# Détection eau sur l'image haute-résolution : fraction de pixels bleu-gris
# homogènes au-dessus de laquelle la micro-zone est classée eau sans GEE.
# Seuil bas (0.30) : un polygone de berge mélange eau + rive, même minoritaire
# l'eau doit l'emporter (le spectral Sentinel-2 à 10 m/px dilue le signal).
MICRO_WATER_FRACTION = 0.30
# Pré-classification haute-résolution des surfaces non végétales (calibré sur
# lac Anosy / zone agricole / Tana centre) :
# - BATI : rouge dominant (R>G+3 et R>B+3), peu vert (exg<12), HÉTÉROGÈNE
#   (toits, routes, ombres mélangés). Tana 41.6 %, lac 13.3 % (vrais bâtiments
#   autour), champs 0.3 %.
# - SOLNU : rouge dominant, peu vert, LISSE et clair (piste, sol nu, toit clair).
# Ces fractions HR surchargent la classification spectrale quand le NDVI est bas
# (un champ cultivé a un NDVI élevé ; un NDVI bas + minéral visible = bâti/sol nu).
MICRO_BATI_FRACTION = 0.40
MICRO_SOLNU_FRACTION = 0.40
MICRO_BATI_MAX_NDVI = 0.35
MICRO_SOLNU_MAX_NDVI = 0.25

DEFAULT_GDD_CONFIG = {"baseTemperature": 0, "threshold": 2200, "periodDays": 365}
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


def _merge_adjacent_same_culture(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fusionne les micro-parcelles adjacentes ayant la même culture.

    Utilise un algorithme de union-find pour grouper les polygones adjacents
    qui partagent la même classe de culture, puis fusionne leurs géométries.
    """
    if not features:
        return features

    try:
        from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
        from shapely.ops import unary_union
    except ImportError:
        # Si shapely n'est pas disponible, on retourne les features sans fusion
        return features

    n = len(features)
    if n <= 1:
        return features

    # Union-Find pour grouper les features adjacentes de même culture
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Construire les polygones shapely pour tester l'adjacence
    polygons = []
    cultures = []
    for feat in features:
        coords = feat["geometry"]["coordinates"][0]
        # Convertir en format shapely (lng, lat)
        ring = [(c[0], c[1]) for c in coords]
        if len(ring) >= 4 and ring[0] == ring[-1]:
            ring = ring[:-1]  # Enlever le point de fermeture dupliqué
        if len(ring) >= 3:
            try:
                poly = ShapelyPolygon(ring)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
                    # Normaliser la culture pour la comparaison (strip, lower)
                    culture = feat["properties"].get("culture", "").strip().lower()
                    cultures.append(culture)
                else:
                    polygons.append(None)
                    cultures.append(feat["properties"].get("culture", "").strip().lower())
            except Exception:
                polygons.append(None)
                cultures.append(feat["properties"].get("culture", "").strip().lower())
        else:
            polygons.append(None)
            cultures.append(feat["properties"].get("culture", "").strip().lower())

    # Tester l'adjacence entre tous les paires (O(n²) mais n est petit, max 120)
    # Utiliser un petit buffer (2 mètres) pour détecter l'adjacence même si
    # les polygones ne se touchent pas exactement à cause de la précision
    # numérique ou du clipping au rayon.
    ADJACENCY_BUFFER_M = 2.0
    for i in range(n):
        if polygons[i] is None:
            continue
        for j in range(i + 1, n):
            if polygons[j] is None:
                continue
            # Même culture ? (comparaison normalisée)
            if cultures[i] != cultures[j]:
                continue
            # Adjacence : les polygones sont à moins de 2m l'un de l'autre
            # (partagent une frontière ou sont très proches)
            try:
                # Buffer léger pour capturer l'adjacence approximative
                if polygons[i].buffer(ADJACENCY_BUFFER_M).intersects(polygons[j]):
                    union(i, j)
            except Exception:
                continue

    # Grouper par racine
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Fusionner chaque groupe
    merged_features = []
    for root, indices in groups.items():
        if len(indices) == 1:
            # Pas de fusion nécessaire
            merged_features.append(features[indices[0]])
        else:
            # Fusionner les géométries
            valid_polys = [polygons[i] for i in indices if polygons[i] is not None]
            if not valid_polys:
                # Fallback : garder la première
                merged_features.append(features[indices[0]])
                continue

            try:
                # Union de tous les polygones du groupe
                merged_poly = unary_union(valid_polys)
                if merged_poly.is_empty:
                    merged_features.append(features[indices[0]])
                    continue

                # Gérer MultiPolygon (garder le plus grand)
                if merged_poly.geom_type == "MultiPolygon":
                    merged_poly = max(merged_poly.geoms, key=lambda g: g.area)

                if merged_poly.geom_type != "Polygon":
                    merged_features.append(features[indices[0]])
                    continue

                # Extraire les coordonnées
                exterior = list(merged_poly.exterior.coords)
                # Convertir en format GeoJSON (lat, lng) et fermer l'anneau
                coords = [{"lat": y, "lng": x} for x, y in exterior]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

                # Créer la feature fusionnée en combinant les propriétés
                # Prendre les propriétés de la plus grande parcelle du groupe
                largest_idx = max(indices, key=lambda i: features[i]["properties"].get("areaHa", 0))
                base_props = features[largest_idx]["properties"].copy()
                base_props["areaHa"] = sum(
                    features[i]["properties"].get("areaHa", 0) for i in indices
                )
                # Mettre à jour les alternatives pour inclure toutes les alternatives du groupe
                all_alternatives = set()
                for i in indices:
                    alts = features[i]["properties"].get("alternatives", [])
                    if isinstance(alts, list):
                        all_alternatives.update(alts)
                base_props["alternatives"] = list(all_alternatives)

                merged_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[c["lng"], c["lat"]] for c in coords]]},
                    "properties": base_props,
                })
            except Exception:
                # En cas d'erreur, garder la première feature du groupe
                merged_features.append(features[indices[0]])

    return merged_features


def _get_error_message(error: BaseException) -> str:
    return str(error) or "Erreur inconnue"


# ── Persistance au registre (dashboard) ──


async def save_micro_parcelles(features: list[dict[str, Any]], warnings: list[str]) -> None:
    for feature in features:
        try:
            await _save_micro_parcelle(feature)
        except Exception as error:  # noqa: BLE001
            logger.warning("saveMicroParcelles: échec de la persistance d'une micro-zone : %s", error)
            warnings.append(f"Une micro-zone n'a pas pu être enregistrée dans le registre : {_get_error_message(error)}")


async def _save_micro_parcelle(feature: dict[str, Any]) -> None:
    ring = feature["geometry"]["coordinates"][0]
    coordinates = [{"lat": lat, "lng": lng} for lng, lat in ring]
    center = polygon_centroid(coordinates)
    label = f"micro-v1-{center['lat']:.5f}-{center['lng']:.5f}"
    props = feature["properties"]
    confidence = props.get("confidence") or 0
    confidence_percent = js_round(confidence * 1000) / 10
    mean_ndvi = props.get("meanNDVI")
    tally = props.get("confirmationTally") or {}
    verdict = props.get("confirmationVerdict") or (
        f"{props.get('culture')} détectée (micro-parcelle HR + confirmation) — "
        f"confiance {confidence_percent}% · score confirmation {props.get('confirmationScore')}%"
    )
    values = {
        "label": label,
        "coordinates": coordinates,
        "center_lat": center["lat"],
        "center_lng": center["lng"],
        "surface_ha": props.get("areaHa"),
        "culture_declared": None,
        "culture_detected": props.get("culture"),
        "ndvi_percentage": js_round(mean_ndvi * 1000) / 10 if isinstance(mean_ndvi, (int, float)) else None,
        "ndre": props.get("meanNDRE"),
        "ndwi": props.get("meanNDWI"),
        "confidence": confidence_percent,
        "verdict": verdict,
        "details": f"{props.get('confirmationMethod')} · accords {tally.get('agree', '?')}/{tally.get('expressed', '?')} · alternatives : {', '.join(props.get('alternatives') or [])}",
        "saison": None,
        "soil_type": None,
        "risk_factors": [],
        "recommendations": None,
        "data_source": "Micro-parcelles HR (clôtures) + confirmation multi-capteurs (Sentinel-2 + Landsat + S1 + HR + météo)",
        "owner_name": None,
        "notes": None,
        "time_series_s1": [],
        "time_series_s2": props.get("timeSeriesS2") if isinstance(props.get("timeSeriesS2"), list) else [],
        "estimated_planting_date": props.get("estimatedPlantingDate") if isinstance(props.get("estimatedPlantingDate"), str) else None,
        "estimated_harvest_date": props.get("estimatedHarvestDate") if isinstance(props.get("estimatedHarvestDate"), str) else None,
        "days_since_planting": props.get("daysSincePlanting") if isinstance(props.get("daysSincePlanting"), int) else None,
        "growth_stage": props.get("growthStage") if isinstance(props.get("growthStage"), str) else None,
        "planting_confidence": props.get("plantingConfidence") if isinstance(props.get("plantingConfidence"), (int, float)) else None,
        "evi": props.get("meanEVI"),
        "savi": None,
        "ndwi": props.get("meanNDWI"),
        "agro_score": None,
        "hybrid_score": None,
        "cnn_prob_barley": None,
        "cnn_prob_non_barley": None,
        "phenology": props.get("phenology") if isinstance(props.get("phenology"), dict) else None,
    }

    async with async_session_maker() as session:
        existing = (await session.execute(select(Parcelle).where(Parcelle.label == label))).scalars().first()
        if existing:
            for key, value in values.items():
                setattr(existing, key, value)
        else:
            session.add(Parcelle(**values))
        await session.commit()


def _grad_mag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a)
    return np.sqrt(gx * gx + gy * gy)


def _water_fraction(arr: np.ndarray) -> np.ndarray:
    """Masque eau sur image haute-résolution : bleu NETTEMENT dominant
    (B > G + 6, teinte bleue franche), homogène (lstd < 12), luminosité moyenne.
    Une orge en montaison sombre a G ≈ B (vert terne, exg > 0) : elle n'est PAS
    de l'eau — l'égalité G/B ne suffit plus. Calibré : lac Anosy B>G+6 à ~60 %,
    orge montaison (G−B ≈ 0, exg ~13) à ~0 %.
    La végétation franche (exg élevé) est exclue même si bleuâtre."""
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * red + 0.587 * green + 0.114 * blue
    exg = 2 * green - red - blue
    k = 5
    p = np.pad(lum, k // 2, mode="reflect")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    ii[1:, 1:] = np.cumsum(np.cumsum(p, axis=0), axis=1)
    h, w = lum.shape
    mean = (ii[k:k + h, k:k + w] - ii[:h, k:k + w] - ii[k:k + h, :w] + ii[:h, :w]) / (k * k)
    p2 = np.pad(lum * lum, k // 2, mode="reflect")
    ii2 = np.zeros((p2.shape[0] + 1, p2.shape[1] + 1))
    ii2[1:, 1:] = np.cumsum(np.cumsum(p2, axis=0), axis=1)
    mean2 = (ii2[k:k + h, k:k + w] - ii2[:h, k:k + w] - ii2[k:k + h, :w] + ii2[:h, :w]) / (k * k)
    lstd = np.sqrt(np.maximum(mean2 - mean * mean, 0))
    blue_dominant = (blue > green + 6) & (blue >= red - 3)
    not_vegetation = exg < 8  # orge montaison sombre (exg ~13) exclue
    return blue_dominant & not_vegetation & (lstd < 12) & (lum > 40) & (lum < 170)


def _surface_masks(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Masques haute-résolution par type de surface (même image que _water_fraction).
    - eau : bleu-gris homogène (voir _water_fraction)
    - bati : rouge dominant (toits/terre), peu vert, HÉTÉROGÈNE (lstd >= 8)
    - solnu : rouge dominant, peu vert, LISSE et clair (piste, sol nu, toit clair)
    - vegetation : vert dominant (G > B + 5) et vert (exg > 10)
    Retourne un dict de masques booléens (même taille que l'image)."""
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * red + 0.587 * green + 0.114 * blue
    exg = 2 * green - red - blue
    k = 5
    p = np.pad(lum, k // 2, mode="reflect")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    ii[1:, 1:] = np.cumsum(np.cumsum(p, axis=0), axis=1)
    h, w = lum.shape
    mean = (ii[k:k + h, k:k + w] - ii[:h, k:k + w] - ii[k:k + h, :w] + ii[:h, :w]) / (k * k)
    p2 = np.pad(lum * lum, k // 2, mode="reflect")
    ii2 = np.zeros((p2.shape[0] + 1, p2.shape[1] + 1))
    ii2[1:, 1:] = np.cumsum(np.cumsum(p2, axis=0), axis=1)
    mean2 = (ii2[k:k + h, k:k + w] - ii2[:h, k:k + w] - ii2[k:k + h, :w] + ii2[:h, :w]) / (k * k)
    lstd = np.sqrt(np.maximum(mean2 - mean * mean, 0))
    mineral = (red > green + 3) & (red > blue + 3) & (exg < 12)
    return {
        "eau": _water_fraction(arr),
        "bati": mineral & (lstd >= 8),
        "solnu": mineral & (lstd < 8) & (lum > 100),
        "vegetation": (green > blue + 5) & (exg > 10),
    }


def _segment_google_image(png_bytes: bytes, lat: float) -> tuple[list[list[tuple[float, float]]], dict[str, Any], dict[int, dict[str, float]]]:
    """Retourne les contours pixels [(x, y)...] + stats + fractions de surfaces par groupe."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    raw_arr = np.asarray(img).astype(np.float32)
    masks = _surface_masks(raw_arr)
    water_mask = masks["eau"]
    smooth = img.filter(ImageFilter.GaussianBlur(radius=MICRO_BLUR_RADIUS))
    arr = np.asarray(smooth).astype(np.float32)
    height, width, _ = arr.shape
    mpp = meters_per_pixel_at_zoom(lat, MICRO_ZOOM)
    pix_m2 = mpp * mpp

    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * red + 0.587 * green + 0.114 * blue
    exg = 2 * green - red - blue

    grad = np.maximum(_grad_mag(lum) / 30.0, _grad_mag(exg) / 25.0)
    gmax = float(np.percentile(grad, 99))
    gnorm = np.clip(grad / max(gmax, 1e-6), 0, 1)

    strength = grad.astype(np.float32)
    barrier = np.zeros((height * width,), dtype=np.uint8)
    bar = barrier.reshape(height, width)
    bar[:2, :] = 1
    bar[-2:, :] = 1
    bar[:, :2] = 1
    bar[:, -2:] = 1

    labels = watershed_segment(strength, barrier, width, height, seed_percentile=MICRO_SEED_PERCENTILE, min_seed_pixels=MICRO_MIN_SEED_PIXELS)
    lab2d = labels.reshape(height, width)
    # Moyennes par segment : vectorisé (bincount) au lieu d'un scan complet par label.
    max_lab = int(lab2d.max()) if lab2d.size else 0
    flat_lab = lab2d.ravel()
    counts_arr = np.bincount(flat_lab, minlength=max_lab + 1)
    live = np.flatnonzero(counts_arr > 0)
    live = live[live > 0]
    counts: dict[int, int] = {int(l): int(counts_arr[l]) for l in live}
    lum_sum = np.bincount(flat_lab, weights=lum.ravel(), minlength=max_lab + 1)
    exg_sum = np.bincount(flat_lab, weights=exg.ravel(), minlength=max_lab + 1)
    mask_sums: dict[str, np.ndarray] = {
        key: np.bincount(flat_lab, weights=m.astype(np.float32).ravel(), minlength=max_lab + 1)
        for key, m in masks.items()
    }
    means: dict[int, tuple[float, float, int]] = {}
    surf_seg: dict[int, dict[str, float]] = {}
    for lab in counts:
        n = counts[lab]
        means[lab] = (float(lum_sum[lab] / n), float(exg_sum[lab] / n), n)
        surf_seg[lab] = {key: float(mask_sums[key][lab] / n) for key in masks}

    # Frontières entre segments adjacents : vectorisé (décalages H/V) au lieu
    # d'une double boucle Python (~100k itérations).
    bnd_sum: dict[tuple[int, int], float] = defaultdict(float)
    bnd_cnt: dict[tuple[int, int], int] = defaultdict(int)

    def _accumulate_edges(a: np.ndarray, b: np.ndarray, ga: np.ndarray, gb: np.ndarray) -> None:
        valid = (a != b) & (a > 0) & (b > 0)
        if not bool(valid.any()):
            return
        lo = np.minimum(a[valid], b[valid]).astype(np.int64)
        hi = np.maximum(a[valid], b[valid]).astype(np.int64)
        code = lo * (int(max_lab) + 1) + hi
        wsum = (ga[valid].astype(np.float64) + gb[valid].astype(np.float64))
        uniq, inv = np.unique(code, return_inverse=True)
        tot = np.bincount(inv, weights=wsum)
        cnt = np.bincount(inv)
        base = int(max_lab) + 1
        for u, t, c in zip(uniq.tolist(), tot.tolist(), cnt.tolist()):
            key = (int(u // base), int(u % base))
            bnd_sum[key] += float(t)
            bnd_cnt[key] += int(c) * 2

    # Sous-échantillonnage x2 comme avant (pas de 2) pour garder les mêmes seuils.
    sub = lab2d[::2, ::2]
    gsub = gnorm[::2, ::2]
    _accumulate_edges(sub[:, :-1], sub[:, 1:], gsub[:, :-1], gsub[:, 1:])
    _accumulate_edges(sub[:-1, :], sub[1:, :], gsub[:-1, :], gsub[1:, :])

    parent = {lab: lab for lab in counts}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (a, b), total in sorted(bnd_sum.items()):
        n = bnd_cnt[(a, b)]
        if n < 4:
            continue
        boundary_grad = total / n
        d_lum = abs(means[a][0] - means[b][0])
        d_exg = abs(means[a][1] - means[b][1])
        # Jamais de fusion entre deux surfaces différentes (eau/terre, bâti/champ) :
        # la frontière est réelle même si les moyennes sont proches (reflets, turbidité).
        sa, sb = surf_seg.get(a, {}), surf_seg.get(b, {})
        if abs(sa.get("eau", 0.0) - sb.get("eau", 0.0)) > 0.35:
            continue
        if abs(sa.get("bati", 0.0) - sb.get("bati", 0.0)) > 0.35:
            continue
        if abs(sa.get("vegetation", 0.0) - sb.get("vegetation", 0.0)) > 0.50:
            continue
        if d_lum < MICRO_MERGE_MAX_D_LUM and d_exg < MICRO_MERGE_MAX_D_EXG and boundary_grad < MICRO_MERGE_MAX_BOUNDARY_GRAD:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups: dict[int, list[int]] = defaultdict(list)
    for lab in counts:
        groups[find(lab)].append(lab)

    # Rattache les petits fragments (< seuil) au voisin adjacent avec lequel ils
    # partagent la plus longue frontière, au lieu de les jeter (un fragment jeté
    # = un trou sans polygone entre les micro-parcelles voisines).
    pix_min = MICRO_MIN_AREA_M2 / pix_m2
    group_area_px: dict[int, float] = {}
    for root, members in groups.items():
        group_area_px[root] = float(sum(counts[m] for m in members))
    group_adj: dict[tuple[int, int], float] = defaultdict(float)
    for (a, b), total in bnd_sum.items():
        if bnd_cnt[(a, b)] < 4:
            continue
        ga, gb = find(a), find(b)
        if ga != gb:
            key = (min(ga, gb), max(ga, gb))
            group_adj[key] += float(bnd_cnt[(a, b)])
    changed = True
    while changed:
        changed = False
        for root in list(groups.keys()):
            if root not in groups or group_area_px.get(root, 0) >= pix_min:
                continue
            best_nb: int | None = None
            best_len = 0.0
            for (x, y), length in group_adj.items():
                other = y if x == root else (x if y == root else None)
                if other is None or other not in groups:
                    continue
                if length > best_len:
                    best_len, best_nb = length, other
            if best_nb is None:
                continue
            # Fusionne root dans best_nb.
            groups[best_nb].extend(groups.pop(root))
            group_area_px[best_nb] = group_area_px.get(best_nb, 0) + group_area_px.get(root, 0)
            group_area_px.pop(root, None)
            for key in [k for k in group_adj if root in k]:
                length = group_adj.pop(key)
                other = key[1] if key[0] == root else key[0]
                if other == best_nb or other not in groups:
                    continue
                new_key = (min(best_nb, other), max(best_nb, other))
                group_adj[new_key] += length
            for lab in list(parent.keys()):
                if find(lab) == root:
                    parent[lab] = best_nb
            changed = True

    # Remappe les labels de segments vers les groupes fusionnés (vectorisé).
    seg_to_group = np.zeros(max_lab + 1, dtype=np.int32)
    for root, members in groups.items():
        for m in members:
            if 0 <= m <= max_lab:
                seg_to_group[m] = root
    # Compacte les ids de groupes vers 1..G
    uniq_groups = sorted(groups.keys())
    group_lut = np.zeros((max(uniq_groups) + 1) if uniq_groups else 1, dtype=np.int32)
    for i, g in enumerate(uniq_groups, start=1):
        group_lut[g] = i
    flat = group_lut[seg_to_group[lab2d]]
    n_groups = len(uniq_groups)

    # Surfaces par groupe : vectorisé (bincount) au lieu d'un scan par groupe.
    flat_groups = flat.ravel()
    group_counts = np.bincount(flat_groups, minlength=n_groups + 1)
    group_mask_sums: dict[str, np.ndarray] = {
        key: np.bincount(flat_groups, weights=m.astype(np.float32).ravel(), minlength=n_groups + 1)
        for key, m in masks.items()
    }
    contours_px: list[list[tuple[float, float]]] = []
    surf_by_group: dict[int, dict[str, float]] = {}
    try:
        import cv2 as _cv2  # type: ignore[import]

        _use_cv2 = True
    except ImportError:
        _use_cv2 = False
    for i in range(1, n_groups + 1):
        n = int(group_counts[i]) if i < group_counts.size else 0
        area = float(n) * pix_m2
        # Borne haute assouplie : un grand bloc homogène reste une micro-zone
        # valide (le jeter = un énorme trou). Seuls les résidus sans voisin
        # sous le seuil sont encore écartés ici.
        if area < MICRO_MIN_AREA_M2 or area > MICRO_MAX_AREA_M2 * 4:
            continue
        surf_by_group[len(contours_px)] = {
            key: (float(group_mask_sums[key][i] / n) if n else 0.0) for key in masks
        }
        best: list[tuple[float, float]] | None = None
        if _use_cv2:
            group_mask = (flat == i).astype(np.uint8)
            found, _ = _cv2.findContours(group_mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            for cnt in found:
                if cnt.shape[0] < 3:
                    continue
                eps = float(MICRO_SIMPLIFY_EPS_PX)
                approx = _cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
                simp = [(float(x), float(y)) for x, y in approx.tolist()]
                if len(simp) >= 3 and (best is None or len(simp) > len(best)):
                    best = simp
        else:
            group_mask = flat == i
            mask = group_mask.astype(np.int32)
            # trace_label_contours attend un tableau aplati (index y*width+x)
            traced = trace_label_contours(mask.ravel(), width, height)
            for _lab, pix_contour in traced.items():
                simp = simplify_polygon([(float(p[0]), float(p[1])) for p in pix_contour], MICRO_SIMPLIFY_EPS_PX)
                if len(simp) >= 3 and (best is None or len(simp) > len(best)):
                    best = simp
        if best:
            contours_px.append(best)

    # Les plus grandes d'abord (la zone centrale / les blocs principaux en premier)
    order = sorted(range(len(contours_px)), key=lambda i: -_pixel_contour_area(contours_px[i]))
    contours_px = [contours_px[i] for i in order]
    surf_by_group = {new_i: surf_by_group[old_i] for new_i, old_i in enumerate(order)}
    stats = {
        "segments": len(counts),
        "groups": len(groups),
        "microParcels": len(contours_px),
        "metersPerPixel": js_round(mpp * 1000) / 1000,
    }
    return contours_px[:MICRO_MAX_MICRO_PARCELS], stats, {k: v for k, v in surf_by_group.items() if k < MICRO_MAX_MICRO_PARCELS}


def _pixel_contour_area(pts: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(pts)
    for i in range(n):
        nxt = (i + 1) % n
        area += pts[i][0] * pts[nxt][1] - pts[nxt][0] * pts[i][1]
    return abs(area) / 2


def _pixels_to_latlng(contour_px: list[tuple[float, float]], lat: float, lng: float, width_px: int, height_px: int) -> list[dict[str, float]]:
    mpp = meters_per_pixel_at_zoom(lat, MICRO_ZOOM)
    center_x, center_y = lng_lat_to_mercator_meters(lng, lat)
    origin_x = center_x - (width_px / 2) * mpp
    origin_y = center_y + (height_px / 2) * mpp
    coords = []
    for x, y in contour_px:
        mx = origin_x + (x + 0.5) * mpp
        my = origin_y - (y + 0.5) * mpp
        point_lng, point_lat = mercator_meters_to_lng_lat(mx, my)
        coords.append({"lat": point_lat, "lng": point_lng})
    return coords


def _approximate_area_m2(coords: list[dict[str, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    center = polygon_centroid(coords)
    lat_factor = 111_320
    lng_factor = 111_320 * math.cos((center["lat"] * math.pi) / 180)
    pts = [((p["lng"] - center["lng"]) * lng_factor, (p["lat"] - center["lat"]) * lat_factor) for p in coords]
    area = 0.0
    n = len(pts)
    for i in range(n):
        nxt = (i + 1) % n
        area += pts[i][0] * pts[nxt][1] - pts[nxt][0] * pts[i][1]
    return abs(area) / 2


def _erode_polygon(coords: list[dict[str, float]], distance_m: float) -> list[dict[str, float]] | None:
    """Rétrécit le polygone de distance_m (buffer négatif, projection locale mètres).
    Retourne None si l'érosion vide le polygone (micro-zone trop petite)."""
    if len(coords) < 3:
        return None
    try:
        from shapely.geometry import Polygon as _ShapelyPolygon
    except ImportError:
        return coords
    center = polygon_centroid(coords)
    lat_factor = 111_320
    lng_factor = 111_320 * math.cos((center["lat"] * math.pi) / 180)
    if lng_factor <= 0:
        return coords
    ring = [((p["lng"] - center["lng"]) * lng_factor, (p["lat"] - center["lat"]) * lat_factor) for p in coords]
    try:
        eroded = _ShapelyPolygon(ring).buffer(-distance_m, join_style="mitre")
    except Exception:
        return coords
    if eroded.is_empty:
        return None
    if eroded.geom_type == "MultiPolygon":
        eroded = max(eroded.geoms, key=lambda g: g.area)
    if eroded.geom_type != "Polygon" or len(eroded.exterior.coords) < 4:
        return None
    return [{"lat": center["lat"] + y / lat_factor, "lng": center["lng"] + x / lng_factor} for x, y in eroded.exterior.coords[:-1]]


async def _fetch_micro_spectral(access_token: str, project_id: str, coords: list[dict[str, float]], start: str, end: str) -> dict[str, Any]:
    """Indices Sentinel-2 moyens sur le polygone ÉRODÉ (cœur pur, sans berges).
    L'érosion de 10 m retire les pixels mixtes du pourtour (médiane 180 j, nuages < 30 %)."""
    eroded = _erode_polygon(coords, MICRO_SPECTRAL_EROSION_M) or coords
    result = await call_gee_compute(access_token, project_id, _build_s2_expression(start, end, eroded))
    out: dict[str, Any] = {"ndvi": None, "ndre": None, "ndwi": None, "evi": None, "savi": None}
    if not result:
        return out
    bands = _parse_s2_bands(result)
    indices = _compute_spectral_indices(bands["nir"], bands["red"], bands["blue"], bands["green"], bands["swir"])
    ndvi = indices["ndvi"]
    out["ndvi"] = (ndvi / 100) if isinstance(ndvi, (int, float)) else None
    out["ndwi"] = indices["ndwi"]
    ndre = None
    b8a = result.get("B8A_mean") or result.get("B8A_median") or result.get("B8A")
    b5 = result.get("B5_mean") or result.get("B5_median") or result.get("B5")
    if isinstance(b8a, (int, float)) and isinstance(b5, (int, float)) and (b8a + b5) != 0:
        ndre = (b8a - b5) / (b8a + b5)
    out["ndre"] = ndre
    out["evi"] = (indices["evi"] / 100) if isinstance(indices["evi"], (int, float)) else None
    out["savi"] = (indices["savi"] / 100) if isinstance(indices["savi"], (int, float)) else None
    return out


async def _fetch_micro_ndvi_series(
    access_token: str, project_id: str, coords: list[dict[str, float]]
) -> list[dict[str, Any]]:
    """Série NDVI mensuelle (6 mois) sur le polygone érodé — même primitive GEE
    que analyze-parcel._fetch_time_series (médiane mensuelle, nuages < 35 %).
    Tolérante : un mois en échec → ndvi None (l'inversion thermique l'ignore)."""
    eroded = _erode_polygon(coords, MICRO_SPECTRAL_EROSION_M) or coords
    months = get_monthly_ranges(MICRO_PHENO_MONTHS)

    async def month_mapper(m: dict[str, str]) -> dict[str, Any]:
        try:
            result = await call_gee_compute(access_token, project_id, _build_s2_expression(m["start"], m["end"], eroded))
        except Exception:  # noqa: BLE001 — mois manquant, pas bloquant
            return {"date": m["label"], "ndvi": None, "cloud_cover": None}
        ndvi = None
        if result:
            bands = _parse_s2_bands(result)
            if bands["nir"] is not None and bands["red"] is not None and (bands["nir"] + bands["red"]) != 0:
                ndvi = js_round(((bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"])) * 1000) / 10
        return {"date": m["label"], "ndvi": ndvi, "cloud_cover": None}

    return await map_with_concurrency(months, 1, month_mapper)


def _is_barley_culture(culture: Any) -> bool:
    label = culture.lower().strip() if isinstance(culture, str) else ""
    return "orge" in label and "non" not in label


async def _track_micro_phenology(
    access_token: str,
    project_id: str,
    center: dict[str, float],
    coords: list[dict[str, float]],
    warnings: list[str],
) -> dict[str, Any] | None:
    """Suivi thermique d'une micro-zone : série NDVI → track_parcel.

    Retourne {"s2", "tracked"} ou None si la série est inexploitable.
    Import paresseux de sowing (même raison que analyze-parcel : pas de cycle)."""
    from app.services.sowing import track_parcel as _track  # noqa: E402

    try:
        s2 = await _fetch_micro_ndvi_series(access_token, project_id, coords)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"Phénologie ({center['lat']:.4f},{center['lng']:.4f}) : série NDVI indisponible ({_get_error_message(error)})")
        return None
    if sum(1 for p in s2 if p.get("ndvi") is not None) < 2:
        warnings.append(f"Phénologie ({center['lat']:.4f},{center['lng']:.4f}) : série NDVI inexploitable.")
        return None
    try:
        tracked = await _track(center["lat"], center["lng"], s2)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"Phénologie ({center['lat']:.4f},{center['lng']:.4f}) : suivi thermique indisponible ({_get_error_message(error)})")
        return None
    if not tracked.get("estimated_planting_date"):
        warnings.extend(tracked.get("warnings", []) or [])
        return None
    return {"s2": s2, "tracked": tracked}



def _point_in_polygon_ll(lat: float, lng: float, poly: list[dict[str, float]]) -> bool:
    """Ray-casting lat/lng : vrai si le point est dans le polygone."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]["lng"], poly[i]["lat"]
        xj, yj = poly[j]["lng"], poly[j]["lat"]
        if (yi > lat) != (yj > lat) and lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _micro_coverage_ratio(polys: list[dict[str, Any]], radius_m: float) -> float:
    disc = math.pi * radius_m * radius_m
    if disc <= 0:
        return 1.0
    covered = sum(float(m.get("areaM2", 0) or 0) for m in polys)
    return covered / disc


def _fallback_grid_cells(
    lat: float, lng: float, radius_m: float, existing: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Mailles carrées régulières bouchant les vides laissés par la segmentation HR.

    Un champ homogène sans clôture visible ne produit aucun contour (pas de
    gradient) : sans ce filet, son centre reste vide comme sur la capture
    (polygones aux bords, rien au milieu). On pave le disque en mailles de
    MICRO_FALLBACK_CELL_M et on ne garde que celles dont le centre n'est dans
    aucun polygone existant, découpées au rayon.
    """
    if radius_m <= 0:
        return []
    meters_per_deg_lat = 111_320
    meters_per_deg_lng = max(111_320 * math.cos(math.radians(lat)), 1.0)
    step = MICRO_FALLBACK_CELL_M
    half = step / 2
    cells: list[dict[str, Any]] = []
    n_steps = math.ceil(radius_m / step)
    for row in range(-n_steps, n_steps + 1):
        for col in range(-n_steps, n_steps + 1):
            oy, ox = row * step, col * step
            if math.hypot(ox, oy) > radius_m:
                continue
            c_lat = lat + oy / meters_per_deg_lat
            c_lng = lng + ox / meters_per_deg_lng
            if any(_point_in_polygon_ll(c_lat, c_lng, m["coordinates"]) for m in existing if len(m.get("coordinates", [])) >= 3):
                continue
            d_lat = half / meters_per_deg_lat
            d_lng = half / meters_per_deg_lng
            square = [
                {"lat": c_lat - d_lat, "lng": c_lng - d_lng},
                {"lat": c_lat - d_lat, "lng": c_lng + d_lng},
                {"lat": c_lat + d_lat, "lng": c_lng + d_lng},
                {"lat": c_lat + d_lat, "lng": c_lng - d_lng},
            ]
            cut = _clip_micro_to_radius(square, lat, lng, radius_m)
            if len(cut) < 3:
                continue
            area_m2 = _approximate_area_m2(cut)
            if area_m2 < MICRO_MIN_AREA_CLIPPED_M2:
                continue
            cells.append({
                "coordinates": cut, "areaM2": area_m2, "center": polygon_centroid(cut),
                "waterFraction": 0.0, "batiFraction": 0.0, "solnuFraction": 0.0,
                "vegetationFraction": 0.0, "fallbackCell": True,
            })
    # Le centre d'abord : le trou signalé est au milieu du disque.
    cells.sort(key=lambda m: math.hypot(
        (m["center"]["lng"] - lng) * meters_per_deg_lng,
        (m["center"]["lat"] - lat) * meters_per_deg_lat,
    ))
    return cells[:MICRO_MAX_FALLBACK_CELLS]


def _micro_tile_centers(lat: float, lng: float, radius_m: float) -> list[dict[str, float]]:
    """Grille de tuiles HR couvrant le disque (centre d'abord, comme S2).

    Retourne TOUTES les tuiles nécessaires (sans troncature silencieuse) : c'est
    l'appelant qui tronque à MICRO_MAX_TILES en prévenant (warning), pour éviter
    des secteurs entiers sans polygones (trous muets)."""
    half_image_m = (MICRO_IMAGE_PX / 2) * meters_per_pixel_at_zoom(lat, MICRO_ZOOM)
    if radius_m <= half_image_m:
        return [{"lat": lat, "lng": lng}]
    meters_per_deg_lat = 111_320
    meters_per_deg_lng = 111_320 * math.cos((lat * math.pi) / 180)
    steps = math.ceil(radius_m / MICRO_TILE_STEP_M)
    centers: list[dict[str, float]] = []
    for row in range(-steps, steps + 1):
        for col in range(-steps, steps + 1):
            ox, oy = col * MICRO_TILE_STEP_M, row * MICRO_TILE_STEP_M
            if math.hypot(ox, oy) > radius_m + half_image_m:
                continue
            centers.append({"lat": lat + oy / meters_per_deg_lat, "lng": lng + ox / meters_per_deg_lng})
    origin = {"lat": lat, "lng": lng}
    centers.sort(key=lambda c: math.hypot((c["lng"] - lng) * meters_per_deg_lng, (c["lat"] - lat) * meters_per_deg_lat))
    return centers


def _point_in_radius(lat: float, lng: float, center_lat: float, center_lng: float, radius_m: float) -> bool:
    meters_per_deg_lat = 111_320
    meters_per_deg_lng = 111_320 * math.cos((center_lat * math.pi) / 180)
    return math.hypot((lng - center_lng) * meters_per_deg_lng, (lat - center_lat) * meters_per_deg_lat) <= radius_m


def _clip_micro_to_radius(
    coords: list[dict[str, float]], center_lat: float, center_lng: float, radius_m: float
) -> list[dict[str, float]]:
    """Découpe Sutherland-Hodgman d'un polygone au disque (centre, rayon).

    Les contours HR suivent les clôtures réelles et débordent du cercle quand
    une parcelle est à cheval : sans découpe, l'analyse dépasse la zone
    indiquée. Projection locale mètres, 64 côtés pour le cercle."""
    if len(coords) < 3 or radius_m <= 0:
        return coords
    lng_scale = max(abs(math.cos(math.radians(center_lat))), 0.1)

    def to_local(lat: float, lng: float) -> tuple[float, float]:
        return ((lng - center_lng) * 111_320 * lng_scale, (lat - center_lat) * 110_574)

    def from_local(x: float, y: float) -> dict[str, float]:
        return {"lat": center_lat + y / 110_574, "lng": center_lng + x / (111_320 * lng_scale)}

    pts = [p for p in coords if isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lng"), (int, float))]
    if pts and pts[0]["lat"] == pts[-1]["lat"] and pts[0]["lng"] == pts[-1]["lng"]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    clipped = [to_local(p["lat"], p["lng"]) for p in pts]
    boundary = [(radius_m * math.cos(2 * math.pi * i / 64), radius_m * math.sin(2 * math.pi * i / 64)) for i in range(64)]

    def cross(s: tuple[float, float], e: tuple[float, float], p: tuple[float, float]) -> float:
        return (e[0] - s[0]) * (p[1] - s[1]) - (e[1] - s[1]) * (p[0] - s[0])

    def intersect(p1: tuple[float, float], p2: tuple[float, float], s: tuple[float, float], e: tuple[float, float]) -> tuple[float, float]:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        ex, ey = e[0] - s[0], e[1] - s[1]
        denom = dx * ey - dy * ex
        if denom == 0:
            return p2
        t = ((s[0] - p1[0]) * ey - (s[1] - p1[1]) * ex) / denom
        return (p1[0] + t * dx, p1[1] + t * dy)

    for i in range(len(boundary)):
        s, e = boundary[i], boundary[(i + 1) % len(boundary)]
        src, clipped = clipped, []
        n = len(src)
        if n == 0:
            break
        for j in range(n):
            prev, cur = src[(j + n - 1) % n], src[j]
            pin, cin = cross(s, e, prev) >= 0, cross(s, e, cur) >= 0
            if cin != pin:
                clipped.append(intersect(prev, cur, s, e))
            if cin:
                clipped.append(cur)

    return [from_local(x, y) for x, y in clipped]


async def _segment_tile(tile: dict[str, float], warnings: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Segmente une tuile HR : retourne les polygones lat/lng + stats."""
    empty_stats = {"segments": 0, "groups": 0, "microParcels": 0}
    try:
        image_b64 = await capture_parcel_image(tile["lat"], tile["lng"], MICRO_ZOOM)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"Tuile ({tile['lat']:.5f}, {tile['lng']:.5f}) : imagerie HR indisponible ({_get_error_message(error)}).")
        return [], empty_stats
    try:
        png_bytes = base64.b64decode(image_b64)
        contours_px, seg_stats, surf_by_index = await asyncio.to_thread(_segment_google_image, png_bytes, tile["lat"])
    except Exception as error:  # noqa: BLE001
        logger.warning("[MICRO] segmentation tuile échouée : %s", error)
        warnings.append(f"Tuile ({tile['lat']:.5f}, {tile['lng']:.5f}) : segmentation échouée ({_get_error_message(error)}).")
        return [], empty_stats
    polys: list[dict[str, Any]] = []
    for index, contour in enumerate(contours_px):
        coords = _pixels_to_latlng(contour, tile["lat"], tile["lng"], MICRO_IMAGE_PX, MICRO_IMAGE_PX)
        area_m2 = _approximate_area_m2(coords)
        # Même borne haute assouplie que _segment_google_image : un grand bloc
        # homogène reste valide (le jeter ici recréerait le trou bouché là-bas).
        if area_m2 < MICRO_MIN_AREA_M2 or area_m2 > MICRO_MAX_AREA_M2 * 4:
            continue
        surf = surf_by_index.get(index, {})
        polys.append({
            "coordinates": coords, "areaM2": area_m2, "center": polygon_centroid(coords),
            "waterFraction": surf.get("eau", 0.0),
            "batiFraction": surf.get("bati", 0.0),
            "solnuFraction": surf.get("solnu", 0.0),
            "vegetationFraction": surf.get("vegetation", 0.0),
        })
    return polys, {"segments": seg_stats.get("segments", 0), "groups": seg_stats.get("groups", 0), "microParcels": len(polys)}


async def analyze_micro_parcels(input_data: dict[str, Any]) -> dict[str, Any]:
    lat, lng = input_data["lat"], input_data["lng"]
    radius_m = input_data.get("radiusM", DEFAULT_MICRO_RADIUS_M)
    try:
        radius_m = float(radius_m)
    except (TypeError, ValueError):
        radius_m = DEFAULT_MICRO_RADIUS_M
    radius_m = max(MICRO_MIN_RADIUS_M, min(MICRO_MAX_RADIUS_M, radius_m))
    confidence_threshold = input_data.get("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD)
    gdd_config = input_data.get("gddConfig") or DEFAULT_GDD_CONFIG
    warnings: list[str] = []

    def empty(reason: str | None = None) -> dict[str, Any]:
        if reason:
            warnings.append(reason)
        return {
            "type": "FeatureCollection", "features": [],
            "center": {"lat": lat, "lng": lng}, "radiusM": radius_m,
            "imageZoom": MICRO_ZOOM,
            "segmentation": {"segments": 0, "groups": 0, "microParcels": 0},
            "gddCumulative": None, "gddThreshold": gdd_config["threshold"],
            "confidenceThreshold": confidence_threshold,
            "warnings": warnings,
        }

    if not settings.google_maps_api_key or settings.google_maps_api_key.startswith("VOTRE_"):
        return empty("GOOGLE_MAPS_API_KEY n'est pas configurée : la micro-segmentation nécessite l'imagerie haute-résolution.")
    service_account_json = settings.gee_service_account_key
    if not service_account_json or service_account_json.startswith("VOTRE_"):
        return empty("GEE_SERVICE_ACCOUNT_KEY n'est pas configurée.")

    all_tiles = _micro_tile_centers(lat, lng, radius_m)
    if len(all_tiles) > MICRO_MAX_TILES:
        warnings.append(
            f"Zone large ({radius_m:.0f} m) : {len(all_tiles)} tuiles nécessaires, "
            f"analyse limitée aux {MICRO_MAX_TILES} centrales — les secteurs périphériques peuvent manquer (réduisez le rayon)."
        )
    tiles = all_tiles[:MICRO_MAX_TILES]

    async def tile_mapper(tile: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        return await _segment_tile(tile, warnings)

    tile_results = await map_with_concurrency(tiles, MICRO_TILE_CONCURRENCY, tile_mapper)
    micro_polys: list[dict[str, Any]] = []
    seg_stats = {"segments": 0, "groups": 0, "microParcels": 0}
    for polys, stats in tile_results:
        micro_polys.extend(polys)
        seg_stats["segments"] += stats.get("segments", 0)
        seg_stats["groups"] += stats.get("groups", 0)
    # Verrouillage strict au disque : découpe chaque polygone au rayon
    # (Sutherland-Hodgman) au lieu du seul filtre sur le centre — une clôture
    # à cheval sur le cercle ne doit pas déborder de la zone indiquée.
    cut_polys: list[dict[str, Any]] = []
    cut_out = 0
    for m in micro_polys:
        cut = _clip_micro_to_radius(m["coordinates"], lat, lng, radius_m)
        if len(cut) < 3:
            cut_out += 1
            continue
        area_m2 = _approximate_area_m2(cut)
        if area_m2 < MICRO_MIN_AREA_CLIPPED_M2:
            cut_out += 1
            continue
        cut_polys.append({**m, "coordinates": cut, "areaM2": area_m2, "center": polygon_centroid(cut)})
    micro_polys = cut_polys
    if cut_out:
        warnings.append(f"{cut_out} micro-zone(s) hors rayon écartée(s) ou réduite(s) (rayon {radius_m:.0f} m).")
    # Déduplication inter-tuiles (recouvrement ~37 %). Seuil resserré (8 -> 5 m) :
    # deux micro-zones adjacentes de ~300 m² peuvent avoir des centres proches ;
    # un seuil trop large fusionnait des voisines distinctes -> trous apparents.
    deduped: list[dict[str, Any]] = []
    seen: list[dict[str, float]] = []
    for m in sorted(micro_polys, key=lambda x: -x["areaM2"]):
        c = m["center"]
        if any(_point_in_radius(c["lat"], c["lng"], s["lat"], s["lng"], 5.0) for s in seen):
            continue
        seen.append(c)
        deduped.append(m)
    if len(deduped) > MICRO_MAX_MICRO_PARCELS:
        warnings.append(
            f"{len(deduped)} micro-zones détectées, affichage limité aux {MICRO_MAX_MICRO_PARCELS} plus grandes — "
            "les petites zones restantes n'apparaissent pas (réduisez le rayon pour les voir)."
        )
    micro_polys = deduped[:MICRO_MAX_MICRO_PARCELS]
    # Garantie centre + filet anti-trous : le point visé ne doit jamais rester
    # vide (cas signalé : polygones aux bords, rien au milieu). Deux déclencheurs :
    # 1) le centre n'est dans aucun polygone -> on ajoute la maille centrale ;
    # 2) la couverture globale reste faible (champs homogènes sans clôture
    #    visible) -> on bouche les vides avec des mailles régulières.
    center_covered = any(
        _point_in_polygon_ll(lat, lng, m["coordinates"])
        for m in micro_polys
        if len(m.get("coordinates", [])) >= 3
    )
    need_fallback = (not center_covered) or (_micro_coverage_ratio(micro_polys, radius_m) < MICRO_MIN_COVERAGE_RATIO)
    if need_fallback:
        extra = _fallback_grid_cells(lat, lng, radius_m, micro_polys)
        if extra:
            if not center_covered:
                warnings.append(
                    "Point central non couvert par la segmentation HR : maille(s) régulière(s) "
                    "ajoutée(s) pour garantir l'analyse au point visé."
                )
            else:
                warnings.append(
                    f"Segmentation HR incomplète : {len(extra)} maille(s) régulière(s) ajoutée(s) "
                    "sur les secteurs sans clôture visible."
                )
            micro_polys = (micro_polys + extra)[:MICRO_MAX_MICRO_PARCELS + MICRO_MAX_FALLBACK_CELLS]
    seg_stats["microParcels"] = len(micro_polys)
    seg_stats["tiles"] = len(tiles)

    if not micro_polys:
        return empty("Aucune micro-parcelle délimitée détectée sur l'imagerie haute-résolution dans ce rayon.")

    try:
        access_token = await get_gee_access_token()
        project_id = get_gee_project_id()
    except Exception as error:  # noqa: BLE001
        return empty(f"GEE indisponible : {_get_error_message(error)}")

    try:
        gdd = await fetch_growing_degree_days(lat, lng, gdd_config)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"Données degrés-jours indisponibles : {_get_error_message(error)}")
        gdd = None

    # Fenêtre spectrale : 180 derniers jours (médiane, comme le zoning)
    now = datetime.now(timezone.utc)
    end = now.date().isoformat()
    start = (now - timedelta(days=180)).date().isoformat()

    excluded = {"water": 0}

    # Pré-calcul ST depuis le semis estimé (fenêtre agro, Open-Meteo seul) :
    # alimente resolve_cereal AVANT classification pour que les conditions
    # locales (ST + NDRE + NDVI) concluent orge au lieu de rester indifférencié.
    # Une seule valeur pour toute la zone (météo ~11 km) → calculée une fois.
    from app.services.sowing import st_since_estimated_sowing as _st_since_sowing  # noqa: E402

    try:
        zone_st = await _st_since_sowing(lat, lng)
    except Exception:  # noqa: BLE001 — repli : cumul calendaire seul
        zone_st = None
    if isinstance(zone_st, (int, float)):
        warnings.append(f"ST depuis semis estimé : {zone_st:.0f} °C (référence classification).")

    async def classify(micro: dict[str, Any]) -> dict[str, Any] | None:
        try:
            water_frac = micro.get("waterFraction", 0.0)
            bati_frac = micro.get("batiFraction", 0.0)
            solnu_frac = micro.get("solnuFraction", 0.0)
            # Eau détectée sur l'image haute-résolution : zone vide, aucun résultat.
            # Le signal spectral serait de toute façon dilué par les berges à 10 m/px.
            if water_frac >= MICRO_WATER_FRACTION:
                excluded["water"] += 1
                return None
            # Bâti dominant visible (toits/routes) : pas une culture, même si le
            # spectral est bruité par les jardins interstitiels.
            elif bati_frac >= MICRO_BATI_FRACTION:
                crop = {"class": "BATI", "label": "Bâti / urbain", "confidence": min(0.93, 0.55 + bati_frac * 0.38), "alternatives": ["Sol nu / Labour"]}
                spec = {"ndvi": None, "ndre": None, "ndwi": None, "evi": None}
            # Sol nu dominant visible (piste, carrière, toit clair) : pas de GEE.
            elif solnu_frac >= MICRO_SOLNU_FRACTION:
                crop = {"class": "SOL_NU", "label": "Sol nu / Labour", "confidence": min(0.90, 0.55 + solnu_frac * 0.35), "alternatives": ["Friche / Pâturage"]}
                spec = {"ndvi": None, "ndre": None, "ndwi": None, "evi": None}
            else:
                spec = await _fetch_micro_spectral(access_token, project_id, micro["coordinates"], start, end)
                crop = classify_crop_signature({"ndvi": spec["ndvi"] or 0.0, "ndre": spec["ndre"] or 0.0, "ndwi": spec["ndwi"] or 0.0})
                # Rattrapage berge : le polygone contient une part d'eau visible (> 20 %)
                # mais le spectral Sentinel-2 est dilué (NDVI < 0.25, NDWI bas) -> eau,
                # donc zone vide : aucun résultat.
                # Un vrai champ cultivé a un NDVI ≥ 0.25 ; en dessous avec de l'eau
                # visible, c'est une berge / un plan d'eau, pas une culture.
                if crop["class"] != "EAU" and water_frac >= 0.20:
                    ndvi_v = spec["ndvi"] if isinstance(spec["ndvi"], (int, float)) else 1.0
                    ndwi_v = spec["ndwi"] if isinstance(spec["ndwi"], (int, float)) else -1.0
                    if ndvi_v < 0.25 and ndwi_v < 0.18:
                        excluded["water"] += 1
                        return None
                # Le spectral seul conclut à de l'eau libre -> zone vide aussi.
                if crop["class"] == "EAU":
                    excluded["water"] += 1
                    return None
                # Rattrapage minéral : bâti/sol nu bien visibles mais sous les seuils
                # directs, avec un NDVI bas qui exclut une culture (NDVI < seuils).
                if crop["class"] not in ("EAU", "BATI", "SOL_NU"):
                    ndvi_v = spec["ndvi"] if isinstance(spec["ndvi"], (int, float)) else 1.0
                    if bati_frac >= 0.25 and ndvi_v < MICRO_BATI_MAX_NDVI:
                        crop = {"class": "BATI", "label": "Bâti / urbain", "confidence": min(0.88, 0.50 + bati_frac * 0.38), "alternatives": ["Sol nu / Labour"]}
                    elif solnu_frac >= 0.25 and ndvi_v < MICRO_SOLNU_MAX_NDVI:
                        crop = {"class": "SOL_NU", "label": "Sol nu / Labour", "confidence": min(0.86, 0.50 + solnu_frac * 0.36), "alternatives": ["Friche / Pâturage", "Bâti / urbain"]}
            if crop["class"] in ("BLE", "CEREALE"):
                gdd_cumul = gdd.get("cumulative") if isinstance(gdd, dict) else None
                gdd_cumul = gdd_cumul if isinstance(gdd_cumul, (int, float)) else None
                ndvi_v = spec["ndvi"] if isinstance(spec.get("ndvi"), (int, float)) else 0.0
                ndre_v = spec["ndre"] if isinstance(spec.get("ndre"), (int, float)) else 0.0
                st_semis = micro.get("stSinceSowing")
                st_semis = st_semis if isinstance(st_semis, (int, float)) else zone_st
                micro["stSinceSowing"] = st_semis
                resolved = resolve_cereal(ndvi_v, ndre_v, gdd_cumul, confidence_threshold, st_semis)
                crop = {**crop, **{k: v for k, v in resolved.items() if k in ("class", "label", "confidence", "alternatives")}}
                crop["confirmationMethod"] = resolved.get("method", "")
                if resolved.get("reasons"):
                    warnings.append(f"Céréale ({micro['center']['lat']:.4f},{micro['center']['lng']:.4f}) : {'; '.join(resolved['reasons'])}.")
            gdd_ok = bool(gdd and gdd.get("detected") is True)
            st_ok = isinstance(micro.get("stSinceSowing"), (int, float)) and micro["stSinceSowing"] >= BARLEY_GDD_THRESHOLD
            confirmation = "confirmée" if crop["class"] == "ORGE" and (gdd_ok or st_ok) and crop["confidence"] >= max(0.85, confidence_threshold) else "à vérifier"
            if confirmation == "confirmée":
                method = "degrés-jours + signature spectrale" if gdd_ok else "ST depuis semis + signature spectrale"
            else:
                method = crop.get("confirmationMethod") or "signature Sentinel-2 indicative"
            return {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[p["lng"], p["lat"]] for p in micro["coordinates"]]]},
                "properties": {
                    "class": crop["class"],
                    "culture": crop["label"],
                    "confidence": js_round(crop["confidence"] * 1000) / 1000,
                    "confirmation": confirmation,
                    "confirmationMethod": method,
                    "alternatives": crop["alternatives"],
                    "areaHa": js_round((micro["areaM2"] / 10_000) * 100) / 100,
                    "meanNDVI": js_round(spec["ndvi"] * 1000) / 1000 if spec["ndvi"] is not None else None,
                    "meanNDRE": js_round(spec["ndre"] * 1000) / 1000 if spec["ndre"] is not None else None,
                    "meanNDWI": spec["ndwi"],
                    "meanEVI": spec["evi"],
                    "barleyPresence": "confirmed" if crop["class"] == "ORGE" and (gdd_ok or st_ok) else "not_applicable",
                },
            }
        except Exception as error:  # noqa: BLE001
            logger.exception("[MICRO] classify failed for micro-zone area=%s", micro.get("areaM2"))
            warnings.append(f"Micro-parcelle non analysée : {_get_error_message(error)}")
            return None

    classified = await map_with_concurrency(micro_polys, MICRO_CLASSIFY_CONCURRENCY, classify)
    features = [f for f in classified if f is not None]
    if excluded["water"]:
        warnings.append(
            f"{excluded['water']} zone(s) en eau ignorée(s) — aucune culture (zone vide)."
        )

    # ── Confirmation indépendante de chaque micro-zone (résultat final) ──
    # Tolérante : si le module est absent ou lent, on renvoie le découpage +
    # l'analyse spectrale sans bloquer (la confirmation reste optionnelle).
    try:
        from app.services.confirmation_model import confirm_polygon, parse_sources  # noqa: E402

        _confirm_available = True
    except Exception as error:  # noqa: BLE001
        confirm_polygon = None  # type: ignore[assignment]
        parse_sources = lambda v: v or {}  # type: ignore[assignment]  # noqa: E731
        _confirm_available = False
        warnings.append(f"Confirmation multi-capteurs désactivée : {_get_error_message(error)}")

    confirmation_model = "confirmation-v7-no-cirad"
    if _confirm_available:
        precomputed: dict[str, Any] = {}
        try:
            precomputed["geeAccessToken"] = access_token
            precomputed["geeProjectId"] = project_id
        except Exception:  # noqa: BLE001, S110
            pass
        # Climat partagé à l'échelle de la zone (déjà calculé ci-dessus).
        if gdd is not None:
            from app.services.analyze_parcel import (  # noqa: E402
                _fetch_precipitation_time_series,
                get_monthly_ranges,
            )
            try:
                rain = await _fetch_precipitation_time_series(lat, lng, get_monthly_ranges(6))
                rain_vals = [r["precipitation_mm"] for r in rain if isinstance(r.get("precipitation_mm"), (int, float))]
                precomputed["climate"] = {
                    "available": True,
                    "gddCumulative": gdd["cumulative"],
                    "gddThreshold": gdd_config.get("threshold"),
                    "gddDetected": gdd["detected"],
                    "rain6mMm": js_round(sum(rain_vals) * 10) / 10 if rain_vals else None,
                    "rainSeries": rain,
                }
            except Exception as error:  # noqa: BLE001
                precomputed["climate"] = {"available": False}
                precomputed["climateError"] = _get_error_message(error)
        else:
            precomputed["climate"] = {"available": False}
            precomputed["climateError"] = "Données degrés-jours indisponibles."
        # Sources demandées (toggles du panneau d'analyse) — propagées à chaque micro-zone.
        requested_sources = parse_sources(input_data.get("sources"))

        async def confirm_feature(feature: dict[str, Any]) -> dict[str, Any]:
            props = feature["properties"]
            ring = feature["geometry"]["coordinates"][0]
            coords = [{"lat": c[1], "lng": c[0]} for c in ring]
            center = polygon_centroid(coords)
            try:
                confirmation = await asyncio.wait_for(
                    confirm_polygon(
                        center["lat"], center["lng"], coords,
                        props["culture"], props["confidence"], MICRO_ZOOM, gdd_config,
                        precomputed, requested_sources,
                    ),
                    timeout=MICRO_CONFIRM_TIMEOUT_S,
                )
            except Exception as error:  # noqa: BLE001
                props["confirmation"] = "à vérifier"
                props["confirmationMethod"] = "confirmation indisponible"
                props["confirmationScore"] = None
                props["confirmationVerdict"] = f"Confirmation impossible : {_get_error_message(error)}"
                props["confirmationTally"] = None
                return feature
            tally = confirmation["tally"]
            props["confirmationScore"] = confirmation["confirmationScore"]
            props["confirmationVerdict"] = confirmation["verdict"]
            props["confirmationTally"] = tally
            props["confirmationVotes"] = [
                {"source": v["source"], "label": v["label"], "confidence": v["confidence"],
                 "agreement": v["agreement"], "reason": v["reason"]}
                for v in confirmation["votes"]
            ]
            # Résultat final = verdict confirmé : la culture affichée suit la
            # confirmation quand les sources indépendantes s'accordent ou
            # contredisent nettement le candidat Sentinel-2.
            if tally["disagree"] >= 2:
                props["confirmation"] = "à vérifier"
                props["confirmationMethod"] = "confirmation multi-capteurs : désaccord"
            elif tally["agree"] >= 2:
                props["confirmation"] = "confirmée"
                props["confirmationMethod"] = "confirmation multi-capteurs (Sentinel-2 + Landsat + S1 + HR + météo)"
            return feature

        features = await map_with_concurrency(features, MICRO_CONFIRM_CONCURRENCY, confirm_feature)
        confirmed = sum(1 for f in features if f["properties"].get("confirmation") == "confirmée")
        warnings.append(
            f"Confirmation indépendante appliquée à {len(features)} micro-zone(s) : {confirmed} confirmée(s) (Sentinel-2 + Landsat + S1 + HR + météo)."
        )
    else:
        confirmation_model = "none"
        warnings.append("Confirmation multi-capteurs ignorée : résultat spectral seul.")

    # ── Fusion des micro-parcelles adjacentes de même culture ──
    # Réduit l'effet "patchwork" en fusionnant les zones voisines de même culture.
    count_before = len(features)
    features = _merge_adjacent_same_culture(features)
    count_after = len(features)
    if count_after < count_before:
        warnings.append(f"Micro-parcelles fusionnées : {count_before} → {count_after} zone(s) après fusion des cultures identiques adjacentes.")

    # ── Phénologie des micro-zones d'orge (suivi thermique Zadoks) ──
    # Seules les zones classées orge sont suivies : série NDVI mensuelle GEE
    # par micro-zone → track_parcel (semis + ST/stade + récolte). Les séries
    # sont persistées (time_series_s2) pour l'onglet Phénologie du dashboard.
    async def pheno_feature(feature: dict[str, Any]) -> dict[str, Any]:
        props = feature["properties"]
        if not _is_barley_culture(props.get("culture")):
            return feature
        ring = feature["geometry"]["coordinates"][0]
        coords = [{"lat": c[1], "lng": c[0]} for c in ring]
        center = polygon_centroid(coords)
        result = await _track_micro_phenology(access_token, project_id, center, coords, warnings)
        if not result:
            return feature
        tracked = result["tracked"]
        props["timeSeriesS2"] = result["s2"]
        props["estimatedPlantingDate"] = tracked.get("estimated_planting_date")
        props["estimatedHarvestDate"] = tracked.get("estimated_harvest_date")
        props["daysSincePlanting"] = tracked.get("days_counted") if isinstance(tracked.get("days_counted"), int) else tracked.get("days_since_planting")
        props["growthStage"] = tracked.get("growth_stage")
        props["plantingConfidence"] = tracked.get("planting_confidence")
        props["phenology"] = {
            "sowing": tracked.get("sowing"),
            "phenology": tracked.get("phenology"),
            "harvest": tracked.get("harvest"),
        }
        warnings.extend(tracked.get("warnings", []) or [])
        return feature

    features = await map_with_concurrency(features, MICRO_PHENO_CONCURRENCY, pheno_feature)
    tracked_count = sum(1 for f in features if isinstance(f["properties"].get("phenology"), dict))
    if tracked_count:
        warnings.append(f"Suivi thermique appliqué à {tracked_count} micro-zone(s) d'orge (semis + ST/stade Zadoks + récolte).")

    # Persistance au registre (dashboard) : chaque micro-zone devient une parcelle
    # requêtable, comme l'analyse simple (save_simple_field_parcelles).
    await save_micro_parcelles(features, warnings)

    return {
        "type": "FeatureCollection", "features": features,
        "center": {"lat": lat, "lng": lng}, "radiusM": radius_m,
        "imageZoom": MICRO_ZOOM,
        "segmentation": seg_stats,
        "waterExcluded": excluded["water"],
        "gddCumulative": gdd["cumulative"] if gdd else None, "gddThreshold": gdd_config["threshold"],
        "confidenceThreshold": confidence_threshold,
        "confirmationModel": confirmation_model,
        "warnings": warnings + ["Micro-parcelles suivies sur clôtures visibles (imagerie haute-résolution) ; culture finale confirmée par sources indépendantes (hors Sentinel-2)."],
    }
