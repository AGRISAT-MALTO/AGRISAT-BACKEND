"""Segmentation de parcelles exécutée directement dans le backend.

Le modèle YOLO11-seg entraîné dans le projet `agri_field_segmentation`
(`delineate_jecam_v3_precision.pt`, classe `field`) est embarqué dans
`app/data/models/field_segmentation/`. Il est chargé une seule fois en mémoire
puis exécuté dans le processus FastAPI : plus d'API externe, plus de service
séparé.

L'endpoint `/api/field-segmentation` conserve son contrat d'origine :

    {
      "threshold": 0.1,
      "fieldPercentage": 42.7,
      "maskShape": [hauteur, largeur],
      "maskPngBase64": "...",
      "probabilityPngBase64": "..."
    }

`maskPngBase64` est le masque binaire (union des parcelles retenues au seuil),
`probabilityPngBase64` la confiance du modèle par pixel (niveau de gris 0-255).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("agrisat.field_segmentation")

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "models" / "field_segmentation" / "parcel_segmentation_yolo.pt"
)
MODEL_IMGSZ = 768
# Seuil de détection passé au modèle : garde les détections faibles pour que
# `parcels` décrive tout ce qui a été vu (valeur de production du projet source,
# serve_vhr.py : CONF = 0.1).
DETECTION_CONFIDENCE = 0.1
# Seuil appliqué au masque renvoyé quand la requête n'en fournit pas. Le modèle
# sort des confiances faibles (médiane ~0,2 sur les tuiles JECAM) : à 0,5 seuls
# 1 à 5 objets sur 28-155 survivent (0,4-6 % de l'image) ; à 0,1 on retrouve une
# couverture de 19-53 %, conforme au paysage d'Antsirabe.
DEFAULT_THRESHOLD = 0.1
MAX_IMAGE_PIXELS = 40_000_000

# Ultralytics écrit sa configuration dans le dossier utilisateur ; en production
# (conteneur en lecture seule) on la redirige vers un dossier temporaire.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "agrisat_ultralytics"))

_MODEL_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
_MODEL: Any | None = None


def model_path() -> Path:
    """Chemin des poids : surcharge `FIELD_SEGMENTATION_MODEL_PATH` sinon embarqué."""
    configured = (settings.field_segmentation_model_path or "").strip()
    return Path(configured) if configured else DEFAULT_MODEL_PATH


def default_threshold() -> float:
    """Seuil de masque par défaut : `FIELD_SEGMENTATION_THRESHOLD`, sinon 0,1."""
    return min(max(float(settings.field_segmentation_threshold), 0.0), 1.0)


def load_model() -> Any:
    """Charge (une seule fois) le modèle YOLO de segmentation de parcelles."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        path = model_path()
        if not path.exists():
            raise RuntimeError(f"Modèle de segmentation introuvable : {path}")

        from ultralytics import YOLO  # import local : torch/ultralytics ne sont pas nécessaires au reste de l'API

        logger.info("Chargement du modèle de segmentation embarqué : %s", path)
        model = YOLO(str(path))
        logger.info("Modèle de segmentation chargé (tâche=%s, classes=%s).", model.task, model.names)
        _MODEL = model
        return model


def warmup() -> None:
    """Préchauffe le modèle au démarrage : chargement + une inférence à blanc."""
    import numpy as np

    model = load_model()
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    with _MODEL_LOCK:
        model.predict(blank, imgsz=MODEL_IMGSZ, conf=DETECTION_CONFIDENCE, verbose=False)


def _encode_png(array: Any) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _segment_sync(file_bytes: bytes, threshold: float, filename: str) -> dict[str, Any]:
    """Inférence synchrone (CPU/GPU) : exécutée hors de la boucle asyncio."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as error:  # noqa: BLE001
        raise ValueError("Image illisible (formats acceptés : JPEG, PNG, WebP).") from error

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image vide.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image trop volumineuse ({width}×{height} px) : limite {MAX_IMAGE_PIXELS // 1_000_000} Mpx."
        )

    model = load_model()
    with _MODEL_LOCK:
        results = model.predict(
            np.asarray(image),
            imgsz=MODEL_IMGSZ,
            conf=DETECTION_CONFIDENCE,
            verbose=False,
        )
    result = results[0]

    probability = np.zeros((height, width), dtype=np.float32)
    parcels: list[dict[str, Any]] = []

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is not None and masks.xy is not None and boxes is not None:
        for index, polygon in enumerate(masks.xy):
            points = [(float(x), float(y)) for x, y in polygon]
            if len(points) < 3:
                continue
            confidence = float(boxes.conf[index]) if index < len(boxes.conf) else 0.0

            # Confiance par pixel = maximum des instances qui le couvrent.
            mask_layer = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask_layer).polygon(points, fill=255)
            mask_array = np.asarray(mask_layer) > 0
            probability[mask_array] = np.maximum(probability[mask_array], confidence)

            parcels.append({"confidence": round(confidence, 4), "areaPixels": int(mask_array.sum())})

    selected = (probability >= float(threshold)) & (probability > 0)
    selected_mask = np.where(selected, 255, 0).astype(np.uint8)
    if not selected.any():
        selected_mask = np.zeros((height, width), dtype=np.uint8)

    field_percentage = float(selected_mask.mean() / 255.0 * 100.0)
    retained = sum(1 for parcel in parcels if parcel["confidence"] >= float(threshold))

    if not parcels:
        logger.info("Segmentation %s : aucune parcelle détectée.", filename)
    else:
        logger.info(
            "Segmentation %s : %d parcelles détectées, %d retenues au seuil %.2f (%.1f%% de l'image).",
            filename,
            len(parcels),
            retained,
            threshold,
            field_percentage,
        )

    return {
        "threshold": threshold,
        "fieldPercentage": round(field_percentage, 2),
        "maskShape": [height, width],
        "maskPngBase64": _encode_png(selected_mask),
        "probabilityPngBase64": _encode_png(np.clip(probability * 255.0, 0, 255).astype(np.uint8)),
        "parcelCount": retained,
        "parcels": parcels,
        "modelVersion": "delineate_jecam_v3_precision",
    }


async def call_field_segmentation_model(
    file_bytes: bytes, filename: str, threshold: float | None = None
) -> dict[str, Any]:
    """Exécute le modèle embarqué sans bloquer la boucle d'événements FastAPI."""
    threshold = min(max(float(default_threshold() if threshold is None else threshold), 0.0), 1.0)
    return await asyncio.to_thread(_segment_sync, file_bytes, threshold, filename)
