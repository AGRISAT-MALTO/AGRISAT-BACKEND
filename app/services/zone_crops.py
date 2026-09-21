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

Restitution PAR PARCELLE
------------------------
Les contours JECAM ne sont pas tous des parcelles : 237 d'entre eux depassent
2 ha et pesent la moitie de la surface relevee - ce sont des ilots (jachere,
pinede) qui, colories en plein, recouvraient des quartiers entiers a l'ecran.
Quand l'appelant fournit les parcelles deja tracees a l'ecran (modele JECAM-VHR,
/api/zone-parcels), chaque parcelle tracee recoit la culture de l'ilot qui la
recouvre : la carte est alors decoupee par parcelle et non par bloc. Les ilots
que le tracage n'a pas couverts restent renvoyes, marques
`granularity = "bloc"`, pour que l'interface les dessine en contour seul.

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

import shapely
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

# Au-dela, un contour JECAM n'est plus une parcelle mais un ilot : le colorier en
# plein revient a peindre tout un quartier. Mediane des contours : 0,09 ha ;
# 99e centile : 5,8 ha ; maximum : 31,6 ha.
PARCEL_MAX_HA = 2.0

# Part de la parcelle tracee qui doit tomber dans l'ilot pour en reprendre la
# culture. En dessous, le recouvrement n'est qu'un debordement de contour.
MIN_COVER_RATIO = 0.35

# Un ilot dont les parcelles tracees couvrent au moins cette part est considere
# comme decoupe : on ne renvoie plus le bloc, seulement ses parcelles.
BLOC_COVERED_ENOUGH = 0.5


class ZoneCropsError(RuntimeError):
    """Erreur metier du service de cultures par zone."""


@lru_cache(maxsize=1)
def _load() -> tuple[STRtree, list[dict[str, Any]], list[Any]]:
    """
    Charge le jeu de parcelles et construit l'index spatial (une seule fois).

    Un meme `parcel_id` figure parfois plusieurs fois (303 doublons : meme
    contour, meme annee, deux predictions). On ne garde que la plus sure, sinon
    la parcelle est dessinee deux fois et son opacite double a l'ecran.
    """
    if not DATA_PATH.exists():
        raise ZoneCropsError(
            f"Jeu de parcelles absent : {DATA_PATH}. "
            "Regenerez-le avec agri_field_segmentation/src/export_malto.py."
        )
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    meilleures: dict[str, tuple[Any, dict[str, Any]]] = {}
    ordre: list[str] = []
    anonymes: list[tuple[Any, dict[str, Any]]] = []
    for feature in payload.get("features", []):
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            continue
        if geometry.is_empty:
            continue
        props = dict(feature.get("properties") or {})
        parcel_id = props.get("parcel_id")
        if not parcel_id:
            anonymes.append((geometry, props))
            continue
        ancienne = meilleures.get(parcel_id)
        if ancienne is None:
            ordre.append(parcel_id)
            meilleures[parcel_id] = (geometry, props)
        elif float(props.get("confidence") or 0) > float(ancienne[1].get("confidence") or 0):
            meilleures[parcel_id] = (geometry, props)

    retenues = [meilleures[parcel_id] for parcel_id in ordre] + anonymes
    if not retenues:
        raise ZoneCropsError("Jeu de parcelles vide.")
    geometries = [geometry for geometry, _ in retenues]
    properties = [props for _, props in retenues]
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


def _area_ha(geometry: Any) -> float:
    """
    Surface approchee d'une geometrie en degres decimaux.

    Projection equirectangulaire locale : sur une zone de quelques kilometres,
    l'erreur est inferieure au pour cent, largement suffisant pour comparer une
    parcelle a un seuil.
    """
    latitude = geometry.centroid.y
    metres_par_degre_lat = 111_132.0
    metres_par_degre_lon = 111_320.0 * max(math.cos(math.radians(latitude)), 1e-6)
    return geometry.area * metres_par_degre_lat * metres_par_degre_lon / 10_000.0


def _parse_traced_parcels(parcels: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """
    Parcelles tracees fournies par l'appelant (Features GeoJSON de
    /api/zone-parcels). Les geometries invalides sont ignorees plutot que de
    faire echouer la requete : une parcelle en moins vaut mieux qu'une carte
    vide.

    Surfaces et validite sont calculees en une passe vectorisee : la requete est
    rejouee a chaque carre tracé et l'appel geometrie par geometrie coutait a lui
    seul plus d'une seconde sur 3 000 parcelles.
    """
    geometries: list[Any] = []
    proprietes: list[dict[str, Any]] = []
    for item in parcels or []:
        raw = item.get("geometry") if isinstance(item, dict) else None
        if raw is None and isinstance(item, dict) and "type" in item and "coordinates" in item:
            raw = item  # geometrie nue, sans enveloppe Feature
        if raw is None:
            continue
        try:
            geometry = shape(raw)
        except Exception:
            continue
        geometries.append(geometry)
        proprietes.append(dict((item.get("properties") if isinstance(item, dict) else None) or {}))

    if not geometries:
        return []

    aires = shapely.area(geometries)
    valides = shapely.is_valid(geometries)

    retenues: list[dict[str, Any]] = []
    for geometry, props, aire, valide in zip(geometries, proprietes, aires, valides):
        if aire <= 0:
            continue
        if not valide:
            geometry = geometry.buffer(0)
            if geometry.is_empty or geometry.area <= 0:
                continue
        retenues.append({"geometry": geometry, "properties": props})
    return retenues


def _decouper_par_parcelle(
    blocs: list[dict[str, Any]],
    traced: list[dict[str, Any]],
    longitude: float,
    latitude: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Reporte la culture de chaque contour JECAM sur les parcelles tracees qu'il
    recouvre.

    Une parcelle tracee prend la culture de l'ilot avec lequel elle partage le
    plus de surface, a condition d'en etre couverte a `MIN_COVER_RATIO`. Le
    contour rendu est celui de la PARCELLE, pas l'intersection : une parcelle a
    cheval sur deux ilots reste entiere, ce qui est le decoupage que l'oeil
    attend sur l'image.

    Renvoie (parcelles attribuees, ilots restes sans decoupage).
    """
    if not traced or not blocs:
        return [], blocs

    geometries = [parcel["geometry"] for parcel in traced]
    arbre = STRtree(geometries)
    # Pour chaque parcelle tracee : le meilleur ilot et la surface partagee.
    meilleur: dict[int, tuple[float, int]] = {}
    couvert: dict[int, float] = {}

    for rang_bloc, bloc in enumerate(blocs):
        geom_bloc = bloc["geometry"]
        for index in arbre.query(geom_bloc):
            index = int(index)
            geom_parcelle = geometries[index]
            try:
                partage = geom_bloc.intersection(geom_parcelle).area
            except Exception:  # geometries pathologiques : on passe
                continue
            if partage <= 0:
                continue
            if partage / geom_parcelle.area < MIN_COVER_RATIO:
                continue
            ancien = meilleur.get(index)
            if ancien is None or partage > ancien[0]:
                meilleur[index] = (partage, rang_bloc)

    parcelles: list[dict[str, Any]] = []
    for index, (partage, rang_bloc) in sorted(meilleur.items()):
        bloc = blocs[rang_bloc]
        geom_parcelle = geometries[index]
        props_trace = traced[index]["properties"]
        centroide = geom_parcelle.centroid
        props = dict(bloc["properties"])
        props.update({
            "jecam_id": bloc["properties"].get("parcel_id"),
            "parcel_id": props_trace.get("parcel_id")
            or props_trace.get("ftw_id")
            or f"TRACE_{index:06d}",
            "area_ha": round(
                float(props_trace.get("area_ha") or _area_ha(geom_parcelle)), 4
            ),
            "cover_ratio": round(partage / geom_parcelle.area, 3),
            "granularity": "parcelle",
            "geometry_source": props_trace.get("source") or "JECAM-VHR",
            "distance_m": round(_haversine_m(longitude, latitude, centroide.x, centroide.y)),
        })
        parcelles.append({"geometry": geom_parcelle, "properties": props})
        couvert[rang_bloc] = couvert.get(rang_bloc, 0.0) + partage

    restants = [
        bloc for rang, bloc in enumerate(blocs)
        if couvert.get(rang, 0.0) / bloc["geometry"].area < BLOC_COVERED_ENOUGH
    ]
    return parcelles, restants


def zone_crops(
    latitude: float,
    longitude: float,
    radius_m: float,
    min_confidence: float = 0.0,
    crops: list[str] | None = None,
    parcels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Parcelles agricoles d'une zone circulaire, avec leur culture.

    `parcels` : les parcelles deja tracees a l'ecran (Features de
    /api/zone-parcels). Fournies, la culture est reportee sur chacune d'elles
    plutot que sur les ilots JECAM : c'est ce qui donne une carte par parcelle.
    Omises, le comportement d'origine est conserve (contours JECAM tels quels).

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

    blocs: list[dict[str, Any]] = []
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
        blocs.append({"geometry": geometry, "properties": props})

    traced = _parse_traced_parcels(parcels)
    decoupees, blocs_restants = _decouper_par_parcelle(blocs, traced, longitude, latitude)

    # Un contour JECAM deja a la taille d'une parcelle reste une parcelle ; au-dela
    # c'est un ilot, rendu en contour seul par l'interface.
    entiers: list[dict[str, Any]] = []
    for bloc in blocs_restants:
        props = dict(bloc["properties"])
        surface = props.get("area_ha")
        surface = float(surface) if surface is not None else _area_ha(bloc["geometry"])
        props["jecam_id"] = props.get("parcel_id")
        props["cover_ratio"] = None
        props["granularity"] = "parcelle" if surface <= PARCEL_MAX_HA else "bloc"
        props["geometry_source"] = "JECAM"
        entiers.append({"geometry": bloc["geometry"], "properties": props})

    features = [
        {
            "type": "Feature",
            "geometry": json.loads(json.dumps(item["geometry"].__geo_interface__)),
            "properties": item["properties"],
        }
        for item in decoupees + entiers
    ]
    features.sort(key=lambda f: f["properties"]["distance_m"])

    n_decoupees = len(decoupees)
    n_blocs = sum(1 for f in features if f["properties"].get("granularity") == "bloc")

    counts: dict[str, int] = {}
    area_by_crop: dict[str, float] = {}
    for feature in features:
        if feature["properties"].get("granularity") == "bloc":
            continue  # un ilot n'est pas une parcelle : il fausserait le decompte
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
    if n_blocs:
        warnings.append(
            f"{n_blocs} contour(s) de plus de {PARCEL_MAX_HA:g} ha n'ont pas pu etre "
            "decoupes en parcelles : ce sont des ilots releves en bloc (jachere, "
            "pinede) ou des zones que le tracage automatique n'a pas encore "
            "couvertes. Ils sont dessines en contour seul, sans remplissage."
        )
    if traced and n_decoupees:
        warnings.append(
            f"{n_decoupees} parcelle(s) tracee(s) ont recu la culture de l'ilot qui "
            "les recouvre : la culture vient de JECAM, le contour du tracage "
            "automatique."
        )

    return {
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_m": radius_m,
            "min_confidence": min_confidence,
            "crops": crops,
            "n_traced_parcels": len(traced),
        },
        "summary": {
            "n_parcels": sum(counts.values()),
            "n_from_traced": n_decoupees,
            "n_blocs": n_blocs,
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
