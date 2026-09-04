import base64
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.errors import parse_or_400
from app.schemas import FieldSegmentationRequest
from app.services.field_segmentation import call_field_segmentation_model

logger = logging.getLogger("agrisat.routes.field_segmentation")

router = APIRouter()

FIELD_SEGMENTATION_BODY_LIMIT = 20 * 1024 * 1024


@router.post("/api/field-segmentation")
async def field_segmentation(request: Request) -> JSONResponse:
    raw_body = await request.body()
    if len(raw_body) > FIELD_SEGMENTATION_BODY_LIMIT:
        return JSONResponse(status_code=413, content={"error": "Corps de la requête trop volumineux."})

    try:
        body = json.loads(raw_body)
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": "Corps de requête JSON invalide."})

    parsed = parse_or_400(FieldSegmentationRequest, body, "Fichier ou paramètres invalides.")

    try:
        file_bytes = base64.b64decode(parsed.file_base64, validate=True)
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": "fileBase64 invalide (attendu : encodage base64)."})

    try:
        result = await call_field_segmentation_model(file_bytes, parsed.filename, parsed.threshold if parsed.threshold is not None else 0.5)
        return JSONResponse(content=result)
    except Exception as error:  # noqa: BLE001
        logger.warning("field-segmentation: modèle indisponible: %s", error)
        message = str(error) or "Modèle de segmentation indisponible."
        return JSONResponse(status_code=502, content={"error": message})
