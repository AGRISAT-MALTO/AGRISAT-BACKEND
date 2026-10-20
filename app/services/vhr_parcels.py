"""
Delimitation automatique des parcelles d'une zone : modele JECAM sur image THR.

Source : service de tracage `agri_field_segmentation/src/serve_vhr.py`
(Delineate Anything re-entraine sur les contours JECAM/CIRAD d'Antsirabe,
applique aux images Esri World Imagery a ~1,2 m). Remplace pour le moment les
contours Fields of The World (app/services/zone_parcels.py, conserve).

Pourquoi un service separe
--------------------------
Le modele exige torch et ultralytics. Contrairement au modele YOLO de
/api/field-segmentation (desormais embarque et execute dans le backend), le
tracage VHR reste un service separe : il combine le telechargement des tuiles
Esri, la mosaique et la fusion de fenetres chevauchantes autour du modele.

Deux modes de tracage
---------------------
  simple : une image. ~30-60 s pour 1 km : trace a la demande.
  mix    : 5 dates d'image (archive Esri Wayback), ~5-7 min pour 1 km : precalcule
           par agri_field_segmentation/src/precalcul_malto.py, qui remplit le
           cache ci-dessous. Un resultat mix en cache est TOUJOURS servi en
           priorite ; sinon la zone est tracee en mode simple.

Qualite mesuree - 831 parcelles JECAM de test, jamais vues a l'entrainement,
avec la chaine de production du service (detections fusionnees, sans chevauchement)
--------------------------------------------------------------------------
    simple : 15,2 % bien tracees (IoU >= 0,5), IoU median 0,148
             > 1 ha : 79 %   0,5-1 ha : 78 %   0,25-0,5 ha : 46 %   0,1-0,25 ha : 25 %   < 0,1 ha : 5 %
    mix    : 18,5 % bien tracees, IoU median 0,187
             > 1 ha : 79 %   0,5-1 ha : 70 %   0,25-0,5 ha : 43 %   0,1-0,25 ha : 29 %   < 0,1 ha : 10 %
(Le 19,9 % annonce a l'entrainement portait sur les detections brutes, qui se
chevauchent : il surestime ce que voit l'utilisateur.)

Rayon plafonne a 1 km.

Cache : le frontend trace l'ecran en zones fixes (centres toujours identiques),
chaque zone est donc gardee sur disque et resservie instantanement a la visite
suivante. Deux requetes simultanees sur la meme zone ne lancent qu'un tracage.
Le cache est versionne (VHR_CACHE_VERSION) : monter la version invalide les
traces produits par un ancien modele (ex. passage de delineate_jecam.pt a
delineate_jecam_complet.pt) — les zones seront retracees a la demande.
Relancer ensuite precalcul_malto.py pour re-precaculer le mode mix.

Licences : Delineate Anything est sous AGPL-3.0 ; les images Esri World Imagery
et Wayback sont soumises aux conditions d'Esri. A verifier avant toute mise en
production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger("agrisat.services.vhr_parcels")

MAX_RADIUS_M = 1_000.0
MAX_REQUEST_RADIUS_M = 20_000.0
MODES = ("simple", "mix")
TIMEOUT_S = {"simple": 300.0, "mix": 1_800.0}

TRANCHES = [(0, 0.1, "< 0,1 ha"), (0.1, 0.5, "0,1-0,5 ha"),
            (0.5, 1.0, "0,5-1 ha"), (1.0, math.inf, "> 1 ha")]

# Version du format de cache : a incrementer quand le modele de tracage change
# (serve_vhr.py POIDS), pour ne plus servir les traces de l'ancien modele.
VHR_CACHE_VERSION = 2

AVERTISSEMENT_FIABILITE = {
    "simple": (
        "Contours traces automatiquement sur image a ~1,2 m : fiables au-dessus de 0,5 ha "
        "(environ 78 % bien traces), incertains entre 0,1 et 0,5 ha (25 a 46 %), rarement "
        "justes sous 0,1 ha (5 %). Qualite mesuree autour d'Antsirabe seulement."
    ),
    "mix": (
        "Contours traces sur 5 dates d'image a ~1,2 m (zone precalculee) : fiables au-dessus "
        "de 0,5 ha (environ 74 % bien traces), incertains entre 0,1 et 0,5 ha (29 a 43 %), "
        "rarement justes sous 0,1 ha (10 %). Qualite mesuree autour d'Antsirabe seulement."
    ),
}


class VhrParcelsError(RuntimeError):
    """Erreur metier : requete refusee par le service de tracage."""


class VhrServiceUnavailable(RuntimeError):
    """Service de tracage injoignable ou trop lent."""


def _cache_dir() -> Path:
    if settings.vhr_cache_dir:
        return Path(settings.vhr_cache_dir)
    return Path(__file__).resolve().parents[2] / ".cache" / "vhr_parcels"


def _cache_path(latitude: float, longitude: float, radius_m: float, mode: str = "simple") -> Path:
    # Suffixe de version : les fichiers des versions precedentes (suffixe "_v1"
    # implicite : mode simple sans suffixe) ne sont plus lus ni ecrits.
    suffixe = "" if mode == "simple" else f"_{mode}"
    return (_cache_dir() /
            f"{latitude:.5f}_{longitude:.5f}_{round(radius_m)}_v{VHR_CACHE_VERSION}{suffixe}.json")


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        if time.time() - path.stat().st_mtime > settings.vhr_cache_ttl_days * 86_400:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Cache VHR non ecrit (%s) : %s", path, exc)


async def _call_trace_service(latitude: float, longitude: float, radius_m: float,
                              mode: str) -> dict[str, Any]:
    url = f"{settings.vhr_model_url.rstrip('/')}/trace"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S[mode]) as client:
            resp = await client.post(url, json={
                "latitude": latitude, "longitude": longitude, "radius_m": radius_m, "mode": mode})
    except httpx.TimeoutException as exc:
        raise VhrServiceUnavailable(
            f"Le tracage a depasse {int(TIMEOUT_S[mode])} s. Essayez un rayon plus petit.") from exc
    except httpx.HTTPError as exc:
        raise VhrServiceUnavailable(
            f"Service de tracage injoignable ({settings.vhr_model_url}). Lancez-le depuis "
            "agri_field_segmentation : python src/serve_vhr.py") from exc

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        logger.warning("serve_vhr a refuse la zone : %s %s", resp.status_code, detail)
        raise VhrParcelsError(f"Tracage impossible : {detail}")

    data = resp.json()
    _write_cache(_cache_path(latitude, longitude, radius_m, mode), data)
    return data


_inflight: dict[Path, asyncio.Task] = {}


async def _trace(latitude: float, longitude: float, radius_m: float, mode: str) -> dict[str, Any]:
    # Une zone precalculee en mode mix passe avant le mode simple.
    for mode_cache in ("mix", "simple") if mode == "simple" else ("mix",):
        cached = _read_cache(_cache_path(latitude, longitude, radius_m, mode_cache))
        if cached is not None:
            return {**cached, "cached": True}

    path = _cache_path(latitude, longitude, radius_m, mode)
    task = _inflight.get(path)
    if task is None:
        task = asyncio.ensure_future(_call_trace_service(latitude, longitude, radius_m, mode))
        _inflight[path] = task
        task.add_done_callback(lambda t: (_inflight.pop(path, None), t.cancelled() or t.exception()))
    # shield : si le navigateur abandonne la requete, le tracage va au bout et remplit le cache.
    return await asyncio.shield(task)


async def zone_parcels_vhr(
    latitude: float,
    longitude: float,
    radius_m: float,
    min_confidence: float = 0.0,
    min_area_ha: float = 0.0,
    mode: str = "simple",
) -> dict[str, Any]:
    """Parcelles tracees automatiquement dans un cercle, au format de /api/zone-parcels."""
    if radius_m <= 0:
        raise VhrParcelsError("Le rayon doit etre positif.")
    if mode not in MODES:
        raise VhrParcelsError(f"Mode inconnu : {mode} (attendu : {', '.join(MODES)}).")
    rayon_demande = float(radius_m)
    radius_m = min(rayon_demande, MAX_RADIUS_M)

    data = await _trace(latitude, longitude, radius_m, mode)
    mode_resultat = data.get("mode", "simple")
    features = [
        f for f in data.get("geojson", {}).get("features", [])
        if f["properties"]["confidence"] >= min_confidence
        and f["properties"]["area_ha"] >= min_area_ha
    ]
    logger.info("zone-parcels VHR : %d parcelles, mode %s (%.1f s%s)", len(features), mode_resultat,
                data.get("duration_s", 0.0), ", cache" if data.get("cached") else "")

    surfaces = np.array([f["properties"]["area_ha"] for f in features]) if features else np.zeros(0)
    warnings: list[str] = []
    if rayon_demande > radius_m:
        warnings.append(f"Rayon ramene a {int(MAX_RADIUS_M)} m : au-dela, le tracage prend "
                        "plusieurs minutes.")
    if not features:
        warnings.append("Aucune parcelle detectee dans ce rayon.")
    warnings.append(AVERTISSEMENT_FIABILITE.get(mode_resultat, AVERTISSEMENT_FIABILITE["simple"]))

    return {
        "query": {"latitude": latitude, "longitude": longitude, "radius_m": radius_m,
                  "min_confidence": min_confidence, "min_area_ha": min_area_ha},
        "summary": {
            "n_parcels": len(features),
            "area_ha_total": round(float(surfaces.sum()), 2),
            "truncated": False,
            "by_size": [
                {"range": nom, "n_parcels": int(((surfaces >= bas) & (surfaces < haut)).sum())}
                for bas, haut, nom in TRANCHES
            ],
            "quality": data.get("quality", {}),
            "outside_coverage": False,
            "mode": mode_resultat,
        },
        "attribution": data.get("attribution", ""),
        "warnings": warnings,
        "geojson": {"type": "FeatureCollection", "features": features},
    }
