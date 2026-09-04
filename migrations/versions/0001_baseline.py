"""baseline — documente le schéma live (Neon), vérifié par introspection directe.

Cette migration ne décrit PAS une évolution de schéma à appliquer : elle documente,
pour référence/disaster-recovery (ex. recréer une base de dev vierge), l'état exact
du schéma tel qu'il existe déjà en production. Les tables `parcelles`, `field_scan_runs`
et `vhr_refinements` existent déjà sur la base cible (créées via `prisma db push`, ou
hors bande pour `vhr_refinements`) — cette révision est appliquée uniquement via
`alembic stamp head`, JAMAIS via `alembic upgrade head`, pour ne toucher aucune donnée
existante. Un `alembic revision --autogenerate` de contrôle (généré puis supprimé sans
être appliqué) a confirmé que les modèles SQLAlchemy de app/models.py ne divergent du
schéma live que sur un détail cosmétique sans effet (clause ON UPDATE d'une FK jamais
déclenchée, car les UUID de `parcelles.id` ne sont jamais modifiés).

Revision ID: 0001_baseline
Revises:
Create Date: 2026-09-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "parcelles",
        # Pas de DEFAULT côté DB (confirmé par introspection) : UUID généré côté client.
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("coordinates", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("center_lat", sa.Float(), nullable=False),
        sa.Column("center_lng", sa.Float(), nullable=False),
        sa.Column("surface_ha", sa.Float(), nullable=True),
        sa.Column("culture_declared", sa.Text(), nullable=True),
        sa.Column("culture_detected", sa.Text(), nullable=True),
        sa.Column("ndvi_percentage", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("verdict", sa.Text(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("saison", sa.Text(), nullable=True),
        sa.Column("soil_type", sa.Text(), nullable=True),
        sa.Column("risk_factors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("recommendations", sa.Text(), nullable=True),
        sa.Column("data_source", sa.Text(), nullable=True),
        sa.Column("owner_name", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("time_series_s1", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("time_series_s2", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("estimated_planting_date", sa.Text(), nullable=True),
        sa.Column("estimated_harvest_date", sa.Text(), nullable=True),
        sa.Column("days_since_planting", sa.Integer(), nullable=True),
        sa.Column("growth_stage", sa.Text(), nullable=True),
        sa.Column("planting_confidence", sa.Float(), nullable=True),
        sa.Column("evi", sa.Float(), nullable=True),
        sa.Column("savi", sa.Float(), nullable=True),
        sa.Column("ndwi", sa.Float(), nullable=True),
        sa.Column("agro_score", sa.Float(), nullable=True),
        sa.Column("hybrid_score", sa.Float(), nullable=True),
        sa.Column("cnn_prob_barley", sa.Float(), nullable=True),
        sa.Column("cnn_prob_non_barley", sa.Float(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("ndre", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("spectral_bands", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("time_series_rain", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id", name="parcelles_pkey"),
    )
    op.create_index(
        "parcelles_center_lat_center_lng_idx", "parcelles", ["center_lat", "center_lng"], unique=False
    )

    op.create_table(
        "field_scan_runs",
        # Pas de DEFAULT côté DB (confirmé par introspection) : UUID généré côté client.
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("center_lat", sa.Float(), nullable=False),
        sa.Column("center_lng", sa.Float(), nullable=False),
        sa.Column("radius_m", sa.Float(), nullable=False),
        sa.Column("image_date", sa.Text(), nullable=True),
        sa.Column("image_age_days", sa.Integer(), nullable=True),
        sa.Column("cloud_percentage", sa.Float(), nullable=True),
        sa.Column("confidence_threshold", sa.Float(), nullable=False),
        sa.Column("min_area_ha", sa.Float(), nullable=False),
        sa.Column("result_geojson", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("candidates_found", sa.Integer(), nullable=False),
        sa.Column("candidates_kept", sa.Integer(), nullable=False),
        sa.Column("warnings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="field_scan_runs_pkey"),
    )

    op.create_table(
        "vhr_refinements",
        # Pas de DEFAULT côté DB (confirmé par introspection) : UUID généré côté client.
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parcelle_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("item_type", sa.Text(), nullable=True),
        sa.Column("resolution_m_per_px", sa.Float(), nullable=True),
        sa.Column("capture_date", sa.Text(), nullable=True),
        sa.Column("cloud_percentage", sa.Float(), nullable=True),
        sa.Column("confirmation_is_barley", sa.Boolean(), nullable=True),
        sa.Column("confirmation_confidence", sa.Float(), nullable=True),
        sa.Column("request_center_lat", sa.Float(), nullable=False),
        sa.Column("request_center_lng", sa.Float(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["parcelle_id"], ["parcelles.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="vhr_refinements_pkey"),
    )
    op.create_index("vhr_refinements_parcelle_id_idx", "vhr_refinements", ["parcelle_id"], unique=False)


def downgrade() -> None:
    op.drop_table("vhr_refinements")
    op.drop_table("field_scan_runs")
    op.drop_index("parcelles_center_lat_center_lng_idx", table_name="parcelles")
    op.drop_table("parcelles")
