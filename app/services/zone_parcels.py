"""
Delimitation automatique des parcelles d'une zone d'analyse.

Source : Fields of The World Global (modele PRUE applique a Sentinel-2),
extrait regional `app/data/parcelles_ftw.parquet` produit par
agri_field_segmentation/src/export_ftw_malto.py.

Difference avec zone_crops (contours JECAM)
-------------------------------------------
JECAM est un echantillon : les parcelles que le CIRAD a enquetees, environ 2 %
d'une zone. FTW est exhaustif : toutes les parcelles detectees, sur tout le
Vakinankaratra. Sur un rayon de 1 km a Antsirabe : 654 parcelles FTW contre
130 JECAM.

Qualite mesuree contre les 8 351 contours du CIRAD - a afficher a l'utilisateur
-------------------------------------------------------------------------------
    60,0 % des parcelles bien tracees (IoU >= 0,5), IoU median 0,690
    > 1 ha : 85,3 %   0,5-1 ha : 70,0 %   0,1-0,5 ha : 20-30 %   < 0,1 ha : 0 %
Le maraichage, sous 0,1 ha, n'est donc pas delimite.

Lecture
-------
Le fichier est trie spatialement (courbe de Morton) en blocs dont pyarrow
connait les emprises : une requete ne lit que les blocs qui recoupent la zone.
Temps de reponse de l'ordre de la centaine de millisecondes, quelle que soit la
taille du fichier. Dependances : pyarrow et shapely, rien d'autre.

Licence : CC-BY-4.0, attribution obligatoire - renvoyee dans chaque reponse.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import shapely

logger = logging.getLogger("agrisat.services.zone_parcels")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "parcelles_ftw.parquet"
MAX_RADIUS_M = 20_000.0
MAX_FEATURES = 6_000
EARTH_RADIUS_M = 6_371_000.0

ATTRIBUTION = ("Limites parcellaires : Fields of The World Global (Taylor Geospatial, "
               "Microsoft AI for Good, ASU, WashU, Oregon State, Clark) - modele PRUE - "
               "CC-BY-4.0")


class ZoneParcelsError(RuntimeError):
    """Erreur metier du service de delimitation."""


@lru_cache(maxsize=1)
def _metadata() -> dict[str, Any]:
    """Emprise couverte et informations de qualite, lues une seule fois."""
    import pyarrow.parquet as pq

    if not DATA_PATH.exists():
        raise ZoneParcelsError(
            f"Jeu de parcelles absent : {DATA_PATH}. Regenerez-le avec "
            "agri_field_segmentation/src/export_ftw_malto.py --deploy."
        )
    fichier = pq.ParquetFile(DATA_PATH)
    brut = (fichier.schema_arrow.metadata or {}).get(b"malto", b"{}")
    meta = json.loads(brut)
    meta["n_total"] = fichier.metadata.num_rows
    logger.info("zone_parcels : %d parcelles disponibles (%s)", meta["n_total"], DATA_PATH.name)
    return meta


def _haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def zone_parcels(
    latitude: float,
    longitude: float,
    radius_m: float,
    min_confidence: float = 0.0,
    min_area_ha: float = 0.0,
) -> dict[str, Any]:
    """Parcelles delimitees automatiquement dans un cercle, en GeoJSON."""
    import pyarrow.parquet as pq

    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ZoneParcelsError("Coordonnees hors bornes.")
    if radius_m <= 0:
        raise ZoneParcelsError("Le rayon doit etre positif.")
    radius_m = min(float(radius_m), MAX_RADIUS_M)

    meta = _metadata()
    lon_min_c, lat_min_c, lon_max_c, lat_max_c = meta.get("emprise", (-180, -90, 180, 90))
    hors_zone = not (lon_min_c <= longitude <= lon_max_c and lat_min_c <= latitude <= lat_max_c)

    d_lat = math.degrees(radius_m / EARTH_RADIUS_M)
    d_lon = d_lat / max(math.cos(math.radians(latitude)), 1e-6)
    filtres = [
        ("xmax", ">=", longitude - d_lon), ("xmin", "<=", longitude + d_lon),
        ("ymax", ">=", latitude - d_lat), ("ymin", "<=", latitude + d_lat),
    ]
    if min_confidence:
        filtres.append(("confidence", ">=", float(min_confidence)))
    if min_area_ha:
        filtres.append(("area_ha", ">=", float(min_area_ha)))

    table = pq.read_table(DATA_PATH, filters=filtres,
                          columns=["ftw_id", "confidence", "area_ha",
                                   "xmin", "ymin", "xmax", "ymax", "geometry"])

    features: list[dict[str, Any]] = []
    tronque = False
    if table.num_rows:
        cx = (table["xmin"].to_numpy() + table["xmax"].to_numpy()) / 2
        cy = (table["ymin"].to_numpy() + table["ymax"].to_numpy()) / 2
        distance = _haversine_m(longitude, latitude, cx, cy)
        dedans = np.flatnonzero(distance <= radius_m)
        dedans = dedans[np.argsort(distance[dedans])]
        if dedans.size > MAX_FEATURES:
            dedans = dedans[:MAX_FEATURES]
            tronque = True

        geometries = shapely.from_wkb(table["geometry"].take(dedans).to_pylist())
        ids = table["ftw_id"].take(dedans).to_pylist()
        confiance = table["confidence"].take(dedans).to_numpy()
        surface = table["area_ha"].take(dedans).to_numpy()
        for k, geom in enumerate(geometries):
            if geom is None or geom.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": shapely.geometry.mapping(geom),
                "properties": {
                    "ftw_id": ids[k],
                    "confidence": round(float(confiance[k]), 1),
                    "area_ha": round(float(surface[k]), 4),
                    "distance_m": int(distance[dedans[k]]),
                    "source": "FTW",
                },
            })

    surfaces = np.array([f["properties"]["area_ha"] for f in features]) if features else np.zeros(0)
    tranches = [(0, 0.1, "< 0,1 ha"), (0.1, 0.5, "0,1-0,5 ha"),
                (0.5, 1.0, "0,5-1 ha"), (1.0, math.inf, "> 1 ha")]

    warnings: list[str] = []
    if hors_zone:
        warnings.append("Point hors de la region extraite (Vakinankaratra) : aucune parcelle.")
    elif not features:
        warnings.append("Aucune parcelle detectee dans ce rayon.")
    warnings.append(
        "Contours traces automatiquement : fiables au-dessus de 0,5 ha "
        "(70 a 85 % bien traces), peu fiables en dessous, absents sous 0,1 ha "
        "(maraichage)."
    )
    if tronque:
        warnings.append(f"Affichage limite aux {MAX_FEATURES} parcelles les plus proches.")

    return {
        "query": {"latitude": latitude, "longitude": longitude, "radius_m": radius_m,
                  "min_confidence": min_confidence, "min_area_ha": min_area_ha},
        "summary": {
            "n_parcels": len(features),
            "area_ha_total": round(float(surfaces.sum()), 2),
            "truncated": tronque,
            "by_size": [
                {"range": nom, "n_parcels": int(((surfaces >= bas) & (surfaces < haut)).sum())}
                for bas, haut, nom in tranches
            ],
            "quality": meta.get("qualite_mesuree", {}),
            "outside_coverage": hors_zone,
        },
        "attribution": ATTRIBUTION,
        "warnings": warnings,
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def all_zone_parcels() -> dict[str, Any]:
    """Retourne toutes les parcelles FTW disponibles dans le fichier regional."""
    import pyarrow.parquet as pq

    meta = _metadata()
    table = pq.read_table(DATA_PATH, columns=[
        "ftw_id", "confidence", "area_ha", "xmin", "ymin", "xmax", "ymax", "geometry",
    ])
    features: list[dict[str, Any]] = []
    geometries = shapely.from_wkb(table["geometry"].to_pylist())
    ids = table["ftw_id"].to_pylist()
    confiance = table["confidence"].to_numpy()
    surface = table["area_ha"].to_numpy()
    for index, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": shapely.geometry.mapping(geom),
            "properties": {
                "ftw_id": ids[index],
                "confidence": round(float(confiance[index]), 1),
                "area_ha": round(float(surface[index]), 4),
                "distance_m": 0,
                "source": "FTW",
            },
        })

    surfaces = np.array([f["properties"]["area_ha"] for f in features]) if features else np.zeros(0)
    tranches = [(0, 0.1, "< 0,1 ha"), (0.1, 0.5, "0,1-0,5 ha"),
                (0.5, 1.0, "0,5-1 ha"), (1.0, math.inf, "> 1 ha")]
    return {
        "query": {"scope": "all", "min_confidence": 0.0, "min_area_ha": 0.0},
        "summary": {
            "n_parcels": len(features),
            "area_ha_total": round(float(surfaces.sum()), 2),
            "truncated": False,
            "by_size": [
                {"range": name, "n_parcels": int(((surfaces >= low) & (surfaces < high)).sum())}
                for low, high, name in tranches
            ],
            "quality": meta.get("qualite_mesuree", {}),
            "outside_coverage": False,
        },
        "attribution": ATTRIBUTION,
        "warnings": [
            "Contours traces automatiquement : fiables au-dessus de 0,5 ha "
            "(70 a 85 % bien traces), peu fiables en dessous, absents sous 0,1 ha "
            "(maraichage).",
        ],
        "geojson": {"type": "FeatureCollection", "features": features},
    }
