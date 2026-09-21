"""Ajoute le suivi thermique Zadoks aux parcelles (colonne JSONB `phenology`).

Stocke par parcelle : semis (date, méthode, confiance), ST cumulée + stade
Zadoks courant, récolte projetée (ST 2250 °C). Colonne nullable : les lignes
existantes restent lisibles sans backfill.

Révision appliquable (table live créée via `prisma db push`, jamais par
Alembic) — contrairement à 0001_baseline qui est documentation seule.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_phenology"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "parcelles",
        sa.Column("phenology", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("parcelles", "phenology")
