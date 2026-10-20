import json
import math

import pandas as pd
import pytest

from app.services.barley_rf_classifier import _verdict, build_features, classify


def _series() -> list[dict[str, float | str | None]]:
    """Série mensuelle synthétique d'une parcelle (12 mois, cycle d'orge)."""
    rows: list[dict[str, float | str | None]] = []
    for offset, date in enumerate(pd.date_range("2025-10-01", periods=12, freq="MS")):
        month = date.month
        # Pic de végétation en juin-juillet, senescence en octobre.
        ndvi = 0.18 + 0.55 * math.exp(-((offset - 8) ** 2) / 5.0)
        b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12 = (
            0.10,
            0.13,
            0.16,
            0.22,
            0.30,
            0.36,
            0.38 + ndvi * 0.20,
            0.39 + ndvi * 0.20,
            0.22,
            0.16,
        )
        rows.append(
            {
                "date": date.strftime("%Y-%m"),
                "B2": b2,
                "B3": b3,
                "B4": b4,
                "B5": b5,
                "B6": b6,
                "B7": b7,
                "B8": b8,
                "B8A": b8a,
                "B11": b11,
                "B12": b12,
                "NDVI": ndvi,
                "EVI": 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1),
                "NDRE": (b8 - b5) / (b8 + b5),
                "NDMI": (b8 - b11) / (b8 + b11),
            }
        )
    return rows


def test_features_have_model_shape_and_no_nan():
    features = build_features(_series())

    assert features is not None
    assert features.shape == (12, 48)
    assert not features.isna().any().any()


def test_classify_returns_parcel_probability():
    result = classify(_series())

    assert result is not None
    assert 0.0 <= result["probability"] <= 1.0
    assert result["threshold"] == pytest.approx(0.3268)
    assert result["n_features"] == 48
    assert result["n_observations"] == 12
    assert result["verdict"] in {"ORGE", "A_VERIFIER", "NON_ORGE"}
    assert result["model_version"].startswith("orge_rf")


def test_cloudy_month_is_tolerated():
    series = _series()
    series[2] = {"date": series[2]["date"]}

    result = classify(series)

    assert result is not None
    assert result["n_observations"] == 12


def test_empty_series_returns_none():
    assert classify([]) is None
    assert build_features([]) is None


def test_model_artifacts_are_valid_json():
    """metrics.json et threshold_selection.json alimentent /api/modeles : validés ici, pas au démarrage."""
    from app.services.barley_rf_classifier import METRICS_PATH, THRESHOLD_PATH, load_model

    for path in (METRICS_PATH, THRESHOLD_PATH):
        json.loads(path.read_text(encoding="utf-8"))

    bundle = load_model()

    assert bundle is not None
    assert bundle["metrics"].get("generated_at")
    assert bundle["threshold"] == pytest.approx(0.3268)


def test_three_level_verdict():
    assert _verdict(0.90, ndvi_oct=0.20, doy_peak=210) == "ORGE"
    assert _verdict(0.90, ndvi_oct=0.80, doy_peak=210) == "A_VERIFIER"
    assert _verdict(0.05, ndvi_oct=0.20, doy_peak=210) == "NON_ORGE"
