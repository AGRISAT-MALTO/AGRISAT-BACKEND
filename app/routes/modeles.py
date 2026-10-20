"""Route /api/modeles : inventaire des modeles IA reels deployes (MALTO).

- Classification orge / non-orge : pipeline sklearn RandomForest (projet ASA/Orge,
  app/data/models/orge_rf), metriques et seuil reels lus dans metrics.json et
  threshold_selection.json.
- Segmentation de parcelles : modele YOLO11-seg embarque dans le backend
  (app/data/models/field_segmentation), execute en local par /api/field-segmentation.
- Delimitation automatique VHR : modele Delineate Anything re-entraine sur les contours
  JECAM/CIRAD d'Antsirabe (~1,2 m), servi par le service de tracage (settings.vhr_model_url) ;
  l'etat et le nom du modele charge sont lus sur GET /health de ce service.

L'interface « Modeles IA » du frontend affiche ces valeurs reelles en complement
(ou remplacement) des evaluations fictives de demonstration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter

from app.config import settings

logger = logging.getLogger("agrisat.routes.modeles")

router = APIRouter()

RF_MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "orge_rf"
RF_MODEL_PATH = RF_MODEL_DIR / "orge_rf.joblib"
RF_METRICS_PATH = RF_MODEL_DIR / "metrics.json"
RF_THRESHOLD_PATH = RF_MODEL_DIR / "threshold_selection.json"

SEGMENTATION_MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "field_segmentation"
SEGMENTATION_MODEL_PATH = SEGMENTATION_MODEL_DIR / "parcel_segmentation_yolo.pt"
SEGMENTATION_META_PATH = SEGMENTATION_MODEL_DIR / "model.json"

VHR_HEALTH_TIMEOUT_S = 3.0


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover
        logger.warning("fichier de metadonnees illisible (%s) : %s", path, exc)
        return {}


def _rf_orge_model() -> dict:
    """Modele RF orge/non-orge embarque : presence + metriques du projet Orge."""
    info: dict = {
        "nom": "Classification orge / non-orge",
        "type": "Random Forest (scikit-learn) — 48 variables spectrales-temporelles Sentinel-2",
        "fichier": RF_MODEL_PATH.name,
        "present": RF_MODEL_PATH.is_file(),
        "statut": "indisponible",
    }
    metrics = _read_json(RF_METRICS_PATH)
    selection = _read_json(RF_THRESHOLD_PATH)
    if metrics:
        fonctionnement = selection.get("point_de_fonctionnement") or {}
        observation = metrics.get("metrics_threshold_tuned") or {}
        data = metrics.get("data") or {}
        info.update(
            {
                "statut": "pret" if info["present"] else "fichier absent",
                "metriques": {
                    "seuil": selection.get("seuil_retenu", metrics.get("threshold")),
                    "accuracy": observation.get("accuracy"),
                    "precision_orge": observation.get("precision_orge"),
                    "rappel_orge": observation.get("recall_orge"),
                    "f1_orge": observation.get("f1_orge"),
                    "roc_auc": observation.get("roc_auc"),
                    "rappel_orge_parcelle": fonctionnement.get("rappel_orge"),
                    "precision_orge_parcelle": fonctionnement.get("precision_orge"),
                    "n_parcelles_orge": fonctionnement.get("n_parcelles_orge"),
                },
                "entrainement": {
                    "n_features": data.get("n_features"),
                    "n_lignes": data.get("n_observations"),
                    "n_parcelles": data.get("n_parcels"),
                    "validation": metrics.get("evaluation"),
                    "anti_leakage_ok": "GroupKFold" in str(metrics.get("evaluation", "")),
                    "meilleurs_parametres": metrics.get("best_params"),
                },
            }
        )
    if RF_MODEL_PATH.is_file():
        info["modifie_le"] = datetime.fromtimestamp(
            RF_MODEL_PATH.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    return info


def _segmentation_parcelle_model() -> dict:
    """Modele YOLO de segmentation execute directement par le backend."""
    meta = _read_json(SEGMENTATION_META_PATH)
    info: dict = {
        "nom": "Segmentation de parcelles (image VHR)",
        "type": meta.get("architecture", "YOLO11 instance segmentation"),
        "fichier": SEGMENTATION_MODEL_PATH.name,
        "present": SEGMENTATION_MODEL_PATH.is_file(),
        "statut": "pret" if SEGMENTATION_MODEL_PATH.is_file() else "fichier absent",
        "execution": "in-process",
        "licence": meta.get("license"),
        "parametres": meta.get("inference"),
    }
    if SEGMENTATION_MODEL_PATH.is_file():
        info["modifie_le"] = datetime.fromtimestamp(
            SEGMENTATION_MODEL_PATH.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    return info


async def _delimitation_vhr_model() -> dict:
    """Modele de delimitation VHR : etat du service de tracage (GET /health)."""
    base_url = settings.vhr_model_url.rstrip("/")
    info: dict = {
        "nom": "Delimitation automatique des parcelles (VHR)",
        "type": "Delineate Anything re-entraine sur contours JECAM/CIRAD — image ~1,2 m",
        "service": base_url,
        "statut": "injoignable",
    }
    try:
        async with httpx.AsyncClient(timeout=VHR_HEALTH_TIMEOUT_S) as client:
            resp = await client.get(f"{base_url}/health")
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("service VHR injoignable (%s) : %s", base_url, exc)
        return info
    info.update(
        {
            "statut": "pret" if data.get("status") == "ok" else "erreur",
            "modele_charge": data.get("model"),
            "modele_present": data.get("model_present"),
            "modes": data.get("modes", []),
        }
    )
    return info


@router.get("/api/modeles")
async def get_modeles() -> dict:
    """Modeles IA reels actuellement deployes dans MALTO (metriques reelles)."""
    return {
        "classification_orge": _rf_orge_model(),
        "segmentation_parcelles": _segmentation_parcelle_model(),
        "delimitation_vhr": await _delimitation_vhr_model(),
    }
