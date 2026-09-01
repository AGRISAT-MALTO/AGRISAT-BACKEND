-- analyzeParcel (analyze-parcel.ts) appelle désormais Open-Meteo pour récupérer un historique
-- de précipitations réel, agrégé par mois et aligné sur time_series_s2/time_series_s1.
-- Colonne nullable, additive : n'affecte pas les lignes existantes.
ALTER TABLE parcelles ADD COLUMN IF NOT EXISTS time_series_rain jsonb;
