"""Port 1:1 de src/field-segmentation.ts : client HTTP trivial vers le modèle
externe de segmentation de parcelles (U-Net, service FastAPI séparé)."""

from __future__ import annotations

import logging
import math
from typing import Any

from app.config import settings
from app.services.analyze_parcel import fetch_with_retry

logger = logging.getLogger("agrisat.field_segmentation")

FIELD_SEGMENTATION_MODEL_URL = settings.field_segmentation_model_url or "http://localhost:8000"
EXTERNAL_REQUEST_TIMEOUT_S = 30.0


async def call_field_segmentation_model(file_bytes: bytes, filename: str, threshold: float = 0.5) -> dict[str, Any]:
    url = f"{FIELD_SEGMENTATION_MODEL_URL}/predict?threshold={threshold}"
    logger.info("Calling field segmentation model /predict at: %s", url)

    resp = await fetch_with_retry(
        "POST", url, files={"file": (filename, file_bytes)}, timeout_s=EXTERNAL_REQUEST_TIMEOUT_S
    )

    if resp.status_code >= 400:
        err_text = resp.text[:500] if resp.text else ""
        logger.error("Field segmentation /predict error: %s %s", resp.status_code, err_text)
        raise RuntimeError(f"Modèle de segmentation indisponible : HTTP {resp.status_code}")

    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Réponse du modèle de segmentation invalide.")

    field_percentage = data.get("field_percentage")
    mask_shape = data.get("mask_shape")
    mask_png_base64 = data.get("mask_png_base64")
    probability_png_base64 = data.get("probability_png_base64")
    if (
        not isinstance(field_percentage, (int, float))
        or isinstance(field_percentage, bool)
        or not math.isfinite(field_percentage)
        or not isinstance(mask_shape, list)
        or not isinstance(mask_png_base64, str)
        or not isinstance(probability_png_base64, str)
    ):
        raise RuntimeError("La réponse du modèle de segmentation ne contient pas les valeurs attendues.")

    response_threshold = data.get("threshold")

    return {
        "threshold": response_threshold if isinstance(response_threshold, (int, float)) and not isinstance(response_threshold, bool) else threshold,
        "fieldPercentage": field_percentage,
        "maskShape": mask_shape,
        "maskPngBase64": mask_png_base64,
        "probabilityPngBase64": probability_png_base64,
    }
