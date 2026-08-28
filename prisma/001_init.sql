CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS parcelles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  label text NOT NULL DEFAULT '',
  coordinates jsonb NOT NULL,
  center_lat double precision NOT NULL,
  center_lng double precision NOT NULL,
  surface_ha double precision,
  culture_declared text,
  culture_detected text,
  ndvi_percentage double precision,
  ndre double precision,
  spectral_bands jsonb,
  confidence double precision,
  verdict text,
  details text,
  saison text,
  soil_type text,
  risk_factors jsonb,
  recommendations text,
  data_source text,
  owner_name text,
  notes text,
  time_series_s1 jsonb,
  time_series_s2 jsonb,
  estimated_planting_date text,
  estimated_harvest_date text,
  days_since_planting integer,
  growth_stage text,
  planting_confidence double precision,
  evi double precision,
  savi double precision,
  ndwi double precision,
  agro_score double precision,
  hybrid_score double precision,
  cnn_prob_barley double precision,
  cnn_prob_non_barley double precision,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS parcelles_created_at_idx ON parcelles (created_at DESC);

CREATE TABLE IF NOT EXISTS parcelle_time_series (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  acquisition_date date NOT NULL,
  cloud_cover double precision,
  created_at timestamptz DEFAULT now(),
  ndvi double precision,
  parcelle_id uuid REFERENCES parcelles(id) ON DELETE CASCADE,
  source text NOT NULL,
  vh double precision,
  vv double precision
);

CREATE INDEX IF NOT EXISTS parcelle_time_series_parcelle_id_idx ON parcelle_time_series (parcelle_id);
