# AGRISAT — Backend

API du projet AGRISAT : détection automatique de parcelles agricoles, analyse de cultures par imagerie satellite (NDVI/NDRE/EVI/SAVI/NDWI, détection d'orge par CNN) et diffusion de tuiles Sentinel-2/Planet. Serveur Fastify + TypeScript + Prisma (PostgreSQL).

> Ce dépôt est le backend du projet [AGRISAT](../README.md). Il est consommé par [AGRISAT-FRONTEND](../AGRISAT-FRONTEND).

## Sommaire

- [Stack technique](#stack-technique)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Base de données](#base-de-données)
- [Lancer le projet](#lancer-le-projet)
- [Scripts disponibles](#scripts-disponibles)
- [API — endpoints](#api--endpoints)
- [Structure du projet](#structure-du-projet)
- [Dépannage](#dépannage)

## Stack technique

| Domaine          | Technologie                          |
| ------------------ | --------------------------------------- |
| Runtime / langage   | Node.js (ESM) + TypeScript              |
| Serveur HTTP        | [Fastify](https://fastify.dev/) 5       |
| ORM / base de données | [Prisma](https://www.prisma.io/) + PostgreSQL (compatible [Neon](https://neon.tech/)) |
| Validation          | [Zod](https://zod.dev/)                  |
| Imagerie satellite  | Google Earth Engine (GEE), Sentinel-2, Planet, Airbus OneAtlas |
| Détection IA        | Modèle de segmentation U-Net (service externe FastAPI/Hugging Face) |

## Architecture

Le backend orchestre plusieurs sources et services externes :

- **Google Earth Engine (GEE)** : indices spectraux (NDVI, NDRE, EVI, SAVI, NDWI), séries temporelles Sentinel-1/2, tuiles Sentinel-2.
- **Modèle de segmentation de parcelles** (`FIELD_SEGMENTATION_MODEL_URL`) : service externe (ex. `agri_field_segmentation`, FastAPI/U-Net) utilisé pour la délimitation automatique des parcelles et la détection d'orge par CNN.
- **Airbus OneAtlas** : imagerie haute résolution complémentaire.
- **PostgreSQL (Prisma)** : persistance des parcelles (`parcelles`) et des exécutions d'analyse (`field_scan_runs`).

## Prérequis

- [Node.js](https://nodejs.org/) 18 ou supérieur
- npm
- Une base PostgreSQL accessible (ex. [Neon](https://neon.tech/))
- Un compte de service Google Earth Engine (clé JSON)
- (Optionnel) Un service de segmentation de parcelles compatible (`FIELD_SEGMENTATION_MODEL_URL`)
- (Optionnel) Des clés API Planet / Airbus OneAtlas si ces sources sont utilisées

## Installation

```bash
cd AGRISAT-BACKEND
npm install
```

## Configuration

Copiez le modèle fourni puis renseignez vos propres valeurs :

```bash
cp .env.example .env
```

### Variables d'environnement

| Variable                        | Obligatoire | Description                                                                                          |
| ---------------------------------- | :---------: | ---------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`                     |   **Oui**   | URL de connexion PostgreSQL (Prisma), ex. Neon : `postgresql://user:password@host/db?sslmode=require`       |
| `GEE_SERVICE_ACCOUNT_KEY`          |   **Oui**   | Clé JSON du compte de service Google Earth Engine (indices spectraux, tuiles Sentinel-2)                     |
| `HF_MODEL_URL`                     |     Non     | URL du modèle hébergé (Hugging Face Space) utilisé en complément de la détection                            |
| `FIELD_SEGMENTATION_MODEL_URL`     |     Non     | URL du service de segmentation U-Net (`/predict`, `/segment`). En local : `http://localhost:8000`             |
| `GOOGLE_MAPS_API_KEY`              |     Non     | Clé Google Maps côté serveur (si utilisée pour des appels backend)                                            |
| `AIRBUS_ONEATLAS_API_KEY`          |     Non     | Clé API Airbus OneAtlas (imagerie haute résolution)                                                           |
| `PLANET_API_KEY`                   |     Non     | Clé API Planet (mosaïques/tuiles), active les routes `/api/planet/*`                                          |
| `PORT`                             |     Non     | Port d'écoute du serveur (défaut : `3001`)                                                                    |
| `HOST`                             |     Non     | Adresse d'écoute (défaut : `0.0.0.0`)                                                                         |
| `FRONTEND_ORIGIN`                  |   **Oui**   | Origine(s) autorisée(s) en CORS pour le frontend, séparées par des virgules (ex. `http://localhost:8080`)     |

> **Important**
> - Le fichier `.env` est ignoré par Git : ne jamais committer de clé API ou d'identifiants réels.
> - `FRONTEND_ORIGIN` doit correspondre exactement à l'origine sur laquelle tourne `AGRISAT-FRONTEND` (voir son `.env` / `VITE_API_URL`).

## Base de données

Le schéma est défini avec Prisma dans [`prisma/schema.prisma`](prisma/schema.prisma) (modèles `Parcelle` et `FieldScanRun`), accompagné des scripts SQL [`001_init.sql`](prisma/001_init.sql) et [`002_add_parcelle_location_index.sql`](prisma/002_add_parcelle_location_index.sql).

```bash
# Générer le client Prisma
npm run prisma:generate

# Synchroniser le schéma avec la base (db push)
npm run prisma:push
```

## Lancer le projet

```bash
npm run dev
```

Le serveur démarre par défaut sur [http://localhost:3001](http://localhost:3001) (hot reload via `tsx watch`).

### Build de production

```bash
npm run build
npm start
```

## Scripts disponibles

| Commande                  | Description                                          |
| ---------------------------- | ---------------------------------------------------------- |
| `npm run dev`                | Démarre le serveur en mode développement (hot reload)       |
| `npm run build`              | Compile le TypeScript dans `dist/`                           |
| `npm start`                  | Démarre le serveur compilé (`dist/server.js`)                |
| `npm run prisma:generate`    | Génère le client Prisma                                      |
| `npm run prisma:push`        | Applique le schéma Prisma à la base de données                |

## API — endpoints

| Méthode | Route                              | Description                                                        |
| --------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `GET`     | `/health`                             | Vérification de l'état du serveur                                        |
| `GET`     | `/api/planet/status`                  | Vérifie que `PLANET_API_KEY` est configurée                              |
| `GET`     | `/api/planet/series`                  | Liste les séries/mosaïques Planet disponibles                            |
| `GET`     | `/api/parcelles`                      | Liste les parcelles enregistrées (avec cache court)                      |
| `POST`    | `/api/parcelles`                      | Crée une parcelle                                                        |
| `DELETE`  | `/api/parcelles/:id`                  | Supprime une parcelle                                                    |
| `POST`    | `/api/detect-parcels`                 | Détection automatique de parcelles autour d'un point (avec repli si la base est indisponible) |
| `POST`    | `/api/detect-parcels-fallback`        | Détection automatique de parcelles (variante sans repli)                 |
| `POST`    | `/api/analyze-parcel`                 | Analyse détaillée d'une parcelle (indices spectraux, séries temporelles) |
| `POST`    | `/api/analyze`                        | Analyse simplifiée d'une zone (rayon, seuils de confiance/surface)       |
| `GET`     | `/api/analyze/latest`                 | Dernière analyse enregistrée pour une position donnée                    |
| `GET`     | `/api/sentinel-tiles/:z/:x/:y`        | Tuile Sentinel-2 (PNG) pour un timestamp d'image donné                   |
| `POST`    | `/api/field-segmentation`             | Segmentation d'une image de parcelle via le modèle U-Net externe          |

Toutes les routes retournent du JSON et valident leur entrée avec [Zod](https://zod.dev/) (erreurs `400` avec détails en cas d'échec de validation).

## Structure du projet

```
AGRISAT-BACKEND/
├── src/
│   ├── server.ts               # Point d'entrée Fastify, déclaration des routes
│   ├── analyze-parcel.ts        # Analyse détaillée d'une parcelle (GEE)
│   ├── automatic-parcels.ts     # Détection automatique de parcelles
│   ├── barley-detect-simple.ts  # Analyse simplifiée / détection d'orge
│   ├── field-segmentation.ts    # Appel au modèle de segmentation externe
│   ├── field-watershed.ts       # Segmentation par ligne de partage des eaux
│   ├── sentinel-tiles.ts        # Génération des tuiles Sentinel-2
│   ├── db.ts                     # Client Prisma
│   ├── routes/                   # Routes additionnelles (Planet)
│   └── services/                 # Services externes (Planet, ...)
├── prisma/
│   ├── schema.prisma             # Schéma de base de données
│   └── *.sql                      # Scripts de migration
├── .env.example                  # Modèle des variables d'environnement
└── tsconfig.json
```

## Dépannage

| Problème                                              | Cause probable / solution                                                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Erreurs `CORS` côté frontend                                | `FRONTEND_ORIGIN` ne correspond pas à l'origine réelle du frontend                                |
| Erreurs liées à Google Earth Engine                          | `GEE_SERVICE_ACCOUNT_KEY` manquante/invalide, ou compte de service sans accès au projet GEE       |
| `/api/planet/*` renvoie une erreur 500                       | `PLANET_API_KEY` absente du fichier `.env`                                                        |
| `/api/field-segmentation` renvoie 502                        | Le service `FIELD_SEGMENTATION_MODEL_URL` n'est pas démarré ou n'est pas accessible                |
| `/api/parcelles` renvoie 503                                 | La base de données (`DATABASE_URL`) est inaccessible ou la requête a dépassé le délai imparti      |
