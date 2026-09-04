import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Parcelle(Base):
    """Reflète la table `parcelles` en base (introspection live Neon — divergente à la
    fois de prisma/schema.prisma et des migrations SQL brutes, qui n'ont jamais été
    appliquées à cette base : `prisma db push` seul a créé le schéma réel)."""

    __tablename__ = "parcelles"

    # Pas de DEFAULT côté DB (confirmé par introspection : column_default = None) — Prisma
    # générait l'UUID côté client (Node) avant chaque insertion, jamais via gen_random_uuid()
    # côté serveur ; `default=uuid.uuid4` reproduit ce comportement à l'identique.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    coordinates: Mapped[list | dict] = mapped_column(JSONB, nullable=False)
    center_lat: Mapped[float] = mapped_column(Float, nullable=False)
    center_lng: Mapped[float] = mapped_column(Float, nullable=False)
    surface_ha: Mapped[float | None] = mapped_column(Float)
    culture_declared: Mapped[str | None] = mapped_column(Text)
    culture_detected: Mapped[str | None] = mapped_column(Text)
    ndvi_percentage: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    verdict: Mapped[str | None] = mapped_column(Text)
    details: Mapped[str | None] = mapped_column(Text)
    saison: Mapped[str | None] = mapped_column(Text)
    soil_type: Mapped[str | None] = mapped_column(Text)
    risk_factors: Mapped[list | None] = mapped_column(JSONB)
    recommendations: Mapped[str | None] = mapped_column(Text)
    data_source: Mapped[str | None] = mapped_column(Text)
    owner_name: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    time_series_s1: Mapped[list | None] = mapped_column(JSONB)
    time_series_s2: Mapped[list | None] = mapped_column(JSONB)
    estimated_planting_date: Mapped[str | None] = mapped_column(Text)
    estimated_harvest_date: Mapped[str | None] = mapped_column(Text)
    days_since_planting: Mapped[int | None] = mapped_column(Integer)
    growth_stage: Mapped[str | None] = mapped_column(Text)
    planting_confidence: Mapped[float | None] = mapped_column(Float)
    evi: Mapped[float | None] = mapped_column(Float)
    savi: Mapped[float | None] = mapped_column(Float)
    ndwi: Mapped[float | None] = mapped_column(Float)
    agro_score: Mapped[float | None] = mapped_column(Float)
    hybrid_score: Mapped[float | None] = mapped_column(Float)
    cnn_prob_barley: Mapped[float | None] = mapped_column(Float)
    cnn_prob_non_barley: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    ndre: Mapped[float | None] = mapped_column(Float)
    # Pas de trigger DB ni de code applicatif ne met cette colonne à jour après insertion
    # (confirmé : absente de schema.prisma, jamais lue/écrite par src/) — conservée pour
    # fidélité de schéma uniquement, ne pas y attribuer de logique métier.
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    spectral_bands: Mapped[dict | None] = mapped_column(JSONB)
    time_series_rain: Mapped[list | None] = mapped_column(JSONB)

    __table_args__ = (Index("parcelles_center_lat_center_lng_idx", "center_lat", "center_lng"),)


class FieldScanRun(Base):
    __tablename__ = "field_scan_runs"

    # Pas de DEFAULT côté DB (confirmé par introspection : column_default = None) — Prisma
    # générait l'UUID côté client (Node) avant chaque insertion, jamais via gen_random_uuid()
    # côté serveur ; `default=uuid.uuid4` reproduit ce comportement à l'identique.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    center_lat: Mapped[float] = mapped_column(Float, nullable=False)
    center_lng: Mapped[float] = mapped_column(Float, nullable=False)
    radius_m: Mapped[float] = mapped_column(Float, nullable=False)
    image_date: Mapped[str | None] = mapped_column(Text)
    image_age_days: Mapped[int | None] = mapped_column(Integer)
    cloud_percentage: Mapped[float | None] = mapped_column(Float)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    min_area_ha: Mapped[float] = mapped_column(Float, nullable=False)
    result_geojson: Mapped[dict] = mapped_column(JSONB, nullable=False)
    candidates_found: Mapped[int] = mapped_column(Integer, nullable=False)
    candidates_kept: Mapped[int] = mapped_column(Integer, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())


class VhrRefinement(Base):
    """Reflète la table `vhr_refinements`, présente en base mais sans aucune référence
    dans le code TS source (src/) ni dans l'historique git de ce dépôt (main ou Nicky) —
    créée hors bande, probablement par un autre service/script. Modélisée uniquement
    pour que la baseline Alembic n'essaie jamais de la supprimer ; aucune route/service
    ne s'appuie dessus ici, faute de comportement applicatif existant à porter."""

    __tablename__ = "vhr_refinements"

    # Pas de DEFAULT côté DB (confirmé par introspection : column_default = None) — Prisma
    # générait l'UUID côté client (Node) avant chaque insertion, jamais via gen_random_uuid()
    # côté serveur ; `default=uuid.uuid4` reproduit ce comportement à l'identique.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parcelle_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parcelles.id", ondelete="SET NULL")
    )
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    item_type: Mapped[str | None] = mapped_column(Text)
    resolution_m_per_px: Mapped[float | None] = mapped_column(Float)
    capture_date: Mapped[str | None] = mapped_column(Text)
    cloud_percentage: Mapped[float | None] = mapped_column(Float)
    confirmation_is_barley: Mapped[bool | None] = mapped_column(Boolean)
    confirmation_confidence: Mapped[float | None] = mapped_column(Float)
    request_center_lat: Mapped[float] = mapped_column(Float, nullable=False)
    request_center_lng: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (Index("vhr_refinements_parcelle_id_idx", "parcelle_id"),)
