import asyncio
import base64
import io

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("PIL")

from PIL import Image, ImageDraw  # noqa: E402

from app.services.field_segmentation import (  # noqa: E402
    DEFAULT_THRESHOLD,
    call_field_segmentation_model,
    default_threshold,
    model_path,
)


def _image_bytes() -> bytes:
    image = Image.new("RGB", (96, 96), (104, 128, 72))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 44, 44), fill=(115, 140, 78))
    draw.rectangle((52, 52, 88, 88), fill=(96, 118, 64))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_embedded_segmentation_returns_mask_contract():
    if not model_path().exists():
        pytest.skip("poids YOLO absents")

    result = asyncio.run(call_field_segmentation_model(_image_bytes(), "test.png", threshold=0.5))

    assert set(result) >= {"threshold", "fieldPercentage", "maskShape", "maskPngBase64", "probabilityPngBase64"}
    assert result["maskShape"] == [96, 96]
    assert 0.0 <= result["fieldPercentage"] <= 100.0
    assert result["threshold"] == 0.5

    mask = Image.open(io.BytesIO(base64.b64decode(result["maskPngBase64"])))
    probability = Image.open(io.BytesIO(base64.b64decode(result["probabilityPngBase64"])))
    assert mask.size == (96, 96)
    assert probability.size == (96, 96)
    assert mask.mode == "L"
    assert probability.mode == "L"


def test_default_threshold_matches_model_calibration():
    # 0,5 ne gardait que 0,4-6 % de l'image ; 0,1 est la confiance de production.
    assert DEFAULT_THRESHOLD == pytest.approx(0.1)
    assert default_threshold() == pytest.approx(0.1)


def test_missing_threshold_falls_back_to_default():
    if not model_path().exists():
        pytest.skip("poids YOLO absents")

    result = asyncio.run(call_field_segmentation_model(_image_bytes(), "test.png"))

    assert result["threshold"] == pytest.approx(0.1)


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        asyncio.run(call_field_segmentation_model(b"not-an-image", "bad.png", threshold=0.5))
