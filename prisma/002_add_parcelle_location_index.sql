-- discoverAgriculturalParcelsFromDatabase (automatic-parcels.ts) filtre parcelles par
-- une boîte englobante sur (center_lat, center_lng) à chaque recherche automatique.
-- Sans index, c'est un scan complet de la table, dont le coût grandit avec le nombre
-- de parcelles enregistrées.
CREATE INDEX IF NOT EXISTS parcelles_center_lat_lng_idx ON parcelles (center_lat, center_lng);
