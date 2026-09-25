"""
Auto-start du service de tracage VHR (serve_vhr.py).

Pourquoi
--------
Le modele de tracage exige torch et ultralytics (~2,5 Go) : il tourne dans un
projet a part (agri_field_segmentation) et le backend l'appelle en HTTP
(app/services/vhr_parcels.py). Pour eviter de demarrer ce service a la main
avant chaque session, le backend peut le lancer tout seul :

  - au demarrage du backend, une verification GET /health sur VHR_MODEL_URL ;
  - si le service ne repond pas et que VHR_SERVICE_DIR pointe vers le projet
    agri_field_segmentation, lancement de `python src/serve_vhr.py` en
    processus DETACHE (CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS sous
    Windows, start_new_session ailleurs) : le service survit au redemarrage
    du backend (--reload de uvicorn tue et relance le worker) ;
  - si le service repond deja (demarre a la main, ou par un backend
    precedent), rien n'est lance.

Le python utilise : VHR_SERVICE_PYTHON s'il est renseigne, sinon le venv du
projet service (.venv/Scripts/python.exe) s'il existe, sinon le python courant
du backend — torch n'est alors present que si le meme venv l'installe.

Si VHR_SERVICE_DIR est vide (defaut), rien ne se passe : le comportement reste
celui documente (service a demarrer a la main).

Verification de mise en route : le modele charge en 5-20 s (torch) ; on
interroge /health jusqu'a VHR_STARTUP_TIMEOUT_S avant de conclure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess  # noqa: S404 — commande construite depuis la config locale
import sys
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger("agrisat.services.vhr_launcher")

HEALTH_TIMEOUT_S = 3.0
STARTUP_POLL_S = 2.0


def _service_log() -> Path:
    """Fichier de sortie du service (AGRISAT-BACKEND/.cache/vhr_service.log)."""
    return Path(__file__).resolve().parents[2] / ".cache" / "vhr_service.log"


def _service_python(service_dir: Path) -> str:
    """Python du venv du service s'il existe, sinon VHR_SERVICE_PYTHON, sinon sys.executable."""
    if settings.vhr_service_python:
        return settings.vhr_service_python
    venv_python = service_dir / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _detached_kwargs() -> dict:
    """Creation du processus independant du cycle de vie du backend."""
    if os.name == "nt":
        # DETACHED_PROCESS : pas de console attachee ; CREATE_NEW_PROCESS_GROUP :
        # pas de Ctrl+C propage depuis uvicorn --reload.
        return {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                "close_fds": True}
    return {"start_new_session": True, "close_fds": True}


def _service_command(service_dir: Path) -> list[str] | None:
    """Commande a lancer, ou None si le projet service est mal configure."""
    script = service_dir / "src" / "serve_vhr.py"
    if not script.exists():
        logger.warning(
            "vhr_service_dir pointe vers %s mais %s est absent : pas d'auto-start.",
            service_dir, script)
        return None
    return [_service_python(service_dir), str(script),
            "--port", str(settings.vhr_service_port)]


async def _health_ok(client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get("/health")
        return resp.status_code == 200 and resp.json().get("model_present") is not False
    except (httpx.HTTPError, ValueError):
        return False


async def ensure_vhr_service() -> None:
    """Demarre le service de tracage s'il est configure et ne repond pas deja.

    Non bloquant pour le demarrage de l'API : a appeler dans un task arriere-plan
    (lifespan). N'eleve jamais d'exception : un echec d'auto-start degrade le
    service de tracage (message dans les reponses /api/zone-parcels), pas l'API.
    """
    if not settings.vhr_service_dir:
        logger.info("VHR_SERVICE_DIR non renseigne : pas d'auto-start du service de tracage.")
        return
    service_dir = Path(settings.vhr_service_dir)
    base_url = settings.vhr_model_url.rstrip("/")

    async with httpx.AsyncClient(base_url=base_url, timeout=HEALTH_TIMEOUT_S) as client:
        if await _health_ok(client):
            logger.info("Service de tracage deja en ligne sur %s.", base_url)
            return

        commande = _service_command(service_dir)
        if commande is None:
            return
        try:
            # Sortie vers un fichier : un processus detache sous Windows n'a pas
            # de console (sys.stdout = None, le logging du service planterait) et
            # ce fichier garde la trace pour diagnostiquer un demarrage rate.
            chemin_log = _service_log()
            chemin_log.parent.mkdir(parents=True, exist_ok=True)
            with chemin_log.open("w", encoding="utf-8") as journal:
                subprocess.Popen(commande, cwd=str(service_dir),  # noqa: S603
                                 stdout=journal, stderr=subprocess.STDOUT,
                                 **_detached_kwargs())
        except OSError as error:
            logger.error("Auto-start du service de tracage impossible (%s) : %s",
                         " ".join(commande), error)
            return
        logger.info("Service de tracage lance : %s (cwd %s, log : %s). Attente de /health...",
                    " ".join(commande), service_dir, _service_log())

        # Le chargement de torch + YOLO prend plusieurs secondes.
        for _ in range(int(60 / STARTUP_POLL_S)):
            await asyncio.sleep(STARTUP_POLL_S)
            if await _health_ok(client):
                logger.info("Service de tracage en ligne sur %s.", base_url)
                return
        logger.warning(
            "Service de tracage toujours absent de %s apres 60 s : le tracage VHR "
            "renverra une erreur jusqu'a ce qu'il soit en ligne (voir les logs du service).",
            base_url)
