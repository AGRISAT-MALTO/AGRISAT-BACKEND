"""
Cultures des parcelles d'une zone d'analyse.

Repond a la question posee par l'interface MALTO : l'utilisateur clique un
point, choisit un rayon, et veut voir les parcelles agricoles de la zone avec
la culture de chacune.

Source des donnees
------------------
`app/data/parcelles_antsirabe.geojson`, produit par le projet
agri_field_segmentation (script `src/export_malto.py`). Il contient, pour
chaque parcelle :
  - le contour releve au sol par le CIRAD (base JECAM, 8 351 parcelles) ;
  - la culture enregistree lors du releve (verite terrain, 2016-2019) ;
  - la culture predite par un classifieur temporel LTAE entraine sur des
    series Sentinel-2 (exactitude 0,469 sur 951 parcelles de test).

Limites a connaitre - elles doivent etre affichees a l'utilisateur
------------------------------------------------------------------
1. JECAM est une campagne d'ECHANTILLONNAGE : environ 2 % des parcelles d'une
   zone y figurent. Beaucoup de champs visibles a l'ecran n'auront donc pas de
   contour. Ce service ne "detecte" pas les parcelles, il restitue celles qui
   sont deja relevees.
2. La culture predite est fiable a 60-80 % sur les classes principales (riz,
   mais, perennes, maraichage) et beaucoup moins ailleurs. Le champ
   `confidence` permet de filtrer.
3. L'orge n'est PAS detectable : seules 21 parcelles de cereales a paille
   existent dans la base de reference, trop peu pour entrainer un modele.

Le fichier est charge une seule fois au demarrage et indexe spatialement, donc
les requetes se font en quelques millisecondes sans dependance lourde
(ni torch, ni geopandas : shapely suffit).
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from shapely.geometry import Point, shape
from shapely.strtree import STRtree

logger = logging.getLogger("agrisat.services.zone_crops")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "parcelles_antsirabe.geojson"

# Emprise couverte par la base (Antsirabe, Madagascar). Hors de cette zone, le
# service ne peut rien renvoyer : autant le dire clairement a l'appelant plutot
# que de retourner une liste vide sans explication.
COVERAGE_BBOX = (46.0, -20.6, 47.9, -18.6)  # lon_min, lat_min, lon_max, lat_max
COVERAGE_NAME = "Antsirabe, Madagascar (base JECAM/CIRAD)"

MAX_RADIUS_M = 20_000.0
EARTH_RADIUS_M = 6_371_000.0


class ZoneCropsError(RuntimeError):
    """Erreur metier du service de cultures par zone."""


@lru_cache(maxsize=1)
def _load() -> tuple[STRtree, list[dict[str, Any]], list[Any]]:
    """Charge le jeu de parcelles et construit l'index spatial (une seule fois)."""
    if not DATA_PATH.exists():
        raise ZoneCropsError(
            f"Jeu de parcelles absent : {DATA_PATH}. "
            "Regenerez-le avec agri_field_segmentation/src/export_malto.py."
        )
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    geometries: list[Any] = []
    properties: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            continue
        if geometry.is_empty:
            continue
        geometries.append(geometry)
        properties.append(dict(feature.get("properties") or {}))
    if not geometries:
        raise ZoneCropsError("Jeu de parcelles vide.")
    logger.info("zone_crops : %d parcelles chargees depuis %s", len(geometries), DATA_PATH.name)
    return STRtree(geometries), properties, geometries


def _degrees_for_metres(radius_m: float, latitude: float) -> tuple[float, float]:
    """
    Rayon en degres de longitude et de latitude.

    A la latitude d'Antsirabe (-19,8 deg), un degre de longitude vaut environ
    105 km contre 111 km pour un degre de latitude : les confondre decalerait
    le cercle de plusieurs pour cent.
    """
    delta_lat = math.degrees(radius_m / EARTH_RADIUS_M)
    cos_lat = max(math.cos(math.radians(latitude)), 1e-6)
    delta_lon = delta_lat / cos_lat
    return delta_lon, delta_lat


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def zone_crops(
    latitude: float,
    longitude: float,
    radius_m: float,
    min_confidence: float = 0.0,
    crops: list[str] | None = None,
) -> dict[str, Any]:
    """
    Parcelles agricoles d'une zone circulaire, avec leur culture.

    Retourne un FeatureCollection GeoJSON directement affichable, accompagne
    d'un resume et d'avertissements destines a l'utilisateur.
    """
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ZoneCropsError("Coordonnees hors bornes.")
    if radius_m <= 0:
        raise ZoneCropsError("Le rayon doit etre positif.")
    radius_m = min(float(radius_m), MAX_RADIUS_M)

    tree, properties, geometries = _load()

    delta_lon, delta_lat = _degrees_for_metres(radius_m, latitude)
    window = Point(longitude, latitude).buffer(max(delta_lon, delta_lat))
    candidates = tree.query(window)

    features: list[dict[str, Any]] = []
    for index in candidates:
        index = int(index)
        geometry = geometries[index]
        centroid = geometry.centroid
        distance = _haversine_m(longitude, latitude, centroid.x, centroid.y)
        if distance > radius_m:
            continue
        props = dict(properties[index])
        if min_confidence and float(props.get("confidence") or 0) < min_confidence:
            continue
        if crops and props.get("crop_pred") not in crops:
            continue
        props["distance_m"] = round(distance)
        features.append({
            "type": "Feature",
            "geometry": json.loads(json.dumps(geometry.__geo_interface__)),
            "properties": props,
        })

    features.sort(key=lambda f: f["properties"]["distance_m"])

    counts: dict[str, int] = {}
    area_by_crop: dict[str, float] = {}
    for feature in features:
        crop = feature["properties"].get("crop_pred") or "(non predit)"
        counts[crop] = counts.get(crop, 0) + 1
        area_by_crop[crop] = area_by_crop.get(crop, 0.0) + float(
            feature["properties"].get("area_ha") or 0.0
        )

    lon_min, lat_min, lon_max, lat_max = COVERAGE_BBOX
    hors_zone = not (lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max)

    warnings: list[str] = []
    if hors_zone:
        warnings.append(
            f"Point hors de la zone couverte ({COVERAGE_NAME}). Aucune parcelle de reference."
        )
    elif not features:
        warnings.append(
            "Aucune parcelle relevee dans ce rayon. La base JECAM est un "
            "echantillonnage : essayez un rayon plus large (1 a 2 km)."
        )
    warnings.append(
        "Seules les parcelles relevees au sol par le CIRAD sont affichees "
        "(environ 2 % des parcelles d'une zone). Les autres champs visibles ne "
        "sont pas encore delimites."
    )
    warnings.append(
        "Culture predite fiable a 60-80 % sur riz, mais, perennes et maraichage. "
        "L'orge n'est pas detectable en l'etat."
    )

    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_m": radius_m,
            "min_confidence": min_confidence,
            "crops": crops,
        },
        "summary": {
            "n_parcels": len(features),
            "area_ha_total": round(sum(area_by_crop.values()), 2),
            "crops": [
                {
                    "crop": crop,
                    "n_parcels": counts[crop],
                    "area_ha": round(area_by_crop[crop], 2),
                }
                for crop in sorted(counts, key=lambda c: -counts[c])
            ],
            "coverage": COVERAGE_NAME,
            "outside_coverage": hors_zone,
        },
        "warnings": warnings,
        "geojson": {"type": "FeatureCollection", "features": features},
    }
