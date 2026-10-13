"""Route /api/modeles : inventaire des modeles IA reels deployes (MALTO).

- Classification orge / non-orge : pipeline sklearn RandomForest (projet modele_orge,
  app/data/models/rf_orge), metriques reelles lues dans final_model_summary.json.
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

RF_MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "rf_orge"
RF_SUMMARY_PATH = RF_MODEL_DIR / "final_model_summary.json"
RF_MODEL_PATH = RF_MODEL_DIR / "final_orge_model.pkl"

VHR_HEALTH_TIMEOUT_S = 3.0


def _rf_orge_model() -> dict:
    """Modele RF orge/non-orge : presence du .pkl + metriques du resume d'entrainement."""
    info: dict = {
        "nom": "Classification orge / non-orge",
        "type": "Random Forest (scikit-learn) — 48 features spectrales-temporelles Sentinel-2",
        "fichier": RF_MODEL_PATH.name,
        "present": RF_MODEL_PATH.is_file(),
        "statut": "indisponible",
    }
    if RF_SUMMARY_PATH.is_file():
        try:
            summary = json.loads(RF_SUMMARY_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # pragma: no cover
            logger.warning("resume RF illisible (%s) : %s", RF_SUMMARY_PATH, exc)
            return info
        metrics = summary.get("final_metrics", {})
        info.update(
            {
                "statut": "pret" if info["present"] else "fichier absent",
                "metriques": {
                    "seuil": metrics.get("threshold"),
                    "accuracy": metrics.get("accuracy"),
                    "precision_orge": metrics.get("precision_orge"),
                    "rappel_orge": metrics.get("recall_orge"),
                    "precision_non_orge": metrics.get("precision_non_orge"),
                    "rappel_non_orge": metrics.get("recall_non_orge"),
                    "f1_orge": metrics.get("f1_orge"),
                    "roc_auc": metrics.get("roc_auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "tn": metrics.get("tn"),
                    "fp": metrics.get("fp"),
                    "fn": metrics.get("fn"),
                    "tp": metrics.get("tp"),
                },
                "entrainement": {
                    "n_features": summary.get("n_features"),
                    "n_lignes": summary.get("n_rows"),
                    "n_splits": summary.get("n_splits"),
                    "anti_leakage_ok": (summary.get("anti_leakage") or {}).get("ok"),
                    "meilleurs_parametres": summary.get("best_params"),
                },
            }
        )
    if RF_MODEL_PATH.is_file():
        info["modifie_le"] = datetime.fromtimestamp(
            RF_MODEL_PATH.stat().st_mtime, tz=timezone.utc
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
        "delimitation_vhr": await _delimitation_vhr_model(),
    }
