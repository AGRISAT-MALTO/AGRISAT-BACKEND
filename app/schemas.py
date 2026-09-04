"""Pydantic — miroir des schémas zod de src/server.ts et src/routes/planet.routes.ts."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("doit être un nombre fini")
    return v


class Point(BaseModel):
    lat: float
    lng: float


class AnalyzeSimpleRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius: float = Field(ge=50, le=20_000)
    confidence_threshold: float | None = Field(default=None, ge=0.5, le=0.95, alias="confidenceThreshold")
    min_area_ha: float | None = Field(default=None, ge=0.01, le=5, alias="minAreaHa")
    base_temperature: float | None = Field(default=None, ge=-20, le=30, alias="baseTemperature")
    threshold: float | None = Field(default=None, ge=2200, le=10_000)
    period_days: int | None = Field(default=None, ge=1, le=730, alias="periodDays")

    _v_lat = field_validator("latitude", "longitude", "radius")(_finite)


class AutomaticDetectionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    radius_km: float = Field(ge=0.05, le=20, alias="radiusKm")
    base_temperature: float = Field(ge=-20, le=30, alias="baseTemperature")
    threshold: float = Field(ge=2200, le=10_000)
    period_days: int = Field(ge=1, le=730, alias="periodDays")

    _v = field_validator("lat", "lng", "radius_km", "base_temperature", "threshold")(_finite)


class FieldSegmentationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_base64: str = Field(min_length=1, alias="fileBase64")
    filename: str = Field(min_length=1)
    threshold: float | None = Field(default=None, ge=0, le=1)

    _v = field_validator("threshold")(lambda v: _finite(v) if v is not None else v)


class ParcelleCreate(BaseModel):
    label: str = Field(max_length=200)
    coordinates: list[Point] = Field(min_length=3)

    center_lat: float
    center_lng: float

    surface_ha: float | None

    culture_declared: str
    culture_detected: str | None

    ndvi_percentage: float | None
    ndre: float | None

    spectral_bands: dict[str, float | None] | None

    confidence: float | None
    verdict: str | None
    details: str | None

    saison: str | None
    soil_type: str | None

    risk_factors: list[str]

    recommendations: str | None
    data_source: str | None

    owner_name: str
    notes: str

    time_series_s2: list[Any]
    time_series_s1: list[Any]
    time_series_rain: list[Any]

    estimated_planting_date: str | None
    estimated_harvest_date: str | None

    days_since_planting: float | None

    growth_stage: str | None
    planting_confidence: float | None

    evi: float | None
    savi: float | None
    ndwi: float | None

    agro_score: float | None
    hybrid_score: float | None

    cnn_prob_barley: float | None
    cnn_prob_non_barley: float | None


class SentinelTileParams(BaseModel):
    z: int = Field(ge=10, le=19)
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class SentinelTileQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_timestamp_ms: float = Field(gt=0, alias="imageTimestampMs")

    _v = field_validator("image_timestamp_ms")(_finite)


class PlanetTileParams(BaseModel):
    z: int = Field(ge=0, le=22)
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class PlanetTileQuery(BaseModel):
    mosaic: str | None = Field(default=None, min_length=1, max_length=300)


class AnalyzeLatestQuery(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)

    _v = field_validator("lat", "lng")(_finite)


class DeleteParcelleParams(BaseModel):
    id: str

    @field_validator("id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str:
        import uuid

        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError("doit être un UUID valide") from exc
        return v
