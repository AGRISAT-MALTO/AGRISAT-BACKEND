# AGRISAT — Backend

API du projet AGRISAT : détection automatique de parcelles agricoles, analyse de cultures par imagerie satellite (NDVI/NDRE/EVI/SAVI/NDWI, détection d'orge par CNN) et diffusion de tuiles Sentinel-2/Planet. Serveur FastAPI + Python + SQLAlchemy (PostgreSQL).

> Ce dépôt est le backend du projet [AGRISAT](../README.md). Il est consommé par [AGRISAT-FRONTEND](../AGRISAT-FRONTEND).

## Sommaire

- [Stack technique](#stack-technique)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Base de données](#base-de-données)
- [Lancer le projet](#lancer-le-projet)
- [API — endpoints](#api--endpoints)
- [Structure du projet](#structure-du-projet)
- [Dépannage](#dépannage)

## Stack technique

| Domaine          | Technologie                          |
| ------------------ | --------------------------------------- |
| Runtime / langage   | Python 3.11+                            |
| Serveur HTTP        | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| ORM / base de données | [SQLAlchemy](https://www.sqlalchemy.org/) (async) + [Alembic](https://alembic.sqlalchemy.org/) + PostgreSQL (compatible [Neon](https://neon.tech/)) |
| Validation          | [Pydantic](https://docs.pydantic.dev/)   |
| Imagerie satellite  | Google Earth Engine (GEE, via `google-auth` + API REST), Sentinel-2, Planet, Airbus OneAtlas |
| Détection IA        | Modèle de segmentation U-Net (service externe FastAPI/Hugging Face) |

## Architecture

Le backend orchestre plusieurs sources et services externes :

- **Google Earth Engine (GEE)** : indices spectraux (NDVI, NDRE, EVI, SAVI, NDWI), séries temporelles Sentinel-1/2, tuiles Sentinel-2. Accès via l'API REST (`value:compute` / `image:computePixels`), authentifié avec un compte de service via `google-auth`.
- **Modèle de segmentation de parcelles** (`FIELD_SEGMENTATION_MODEL_URL`) : service externe (ex. `agri_field_segmentation`, FastAPI/U-Net) utilisé pour la délimitation automatique des parcelles et la détection d'orge par CNN.
- **Airbus OneAtlas** : imagerie haute résolution complémentaire.
- **PostgreSQL (SQLAlchemy)** : persistance des parcelles (`parcelles`) et des exécutions d'analyse (`field_scan_runs`).

## Prérequis

- [Python](https://www.python.org/) 3.11 ou supérieur
- Une base PostgreSQL accessible (ex. [Neon](https://neon.tech/))
- Un compte de service Google Earth Engine (clé JSON)
- (Optionnel) Un service de segmentation de parcelles compatible (`FIELD_SEGMENTATION_MODEL_URL`)
- (Optionnel) Des clés API Planet / Airbus OneAtlas si ces sources sont utilisées

## Installation

```bash
cd AGRISAT-BACKEND
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS/Linux :
source .venv/bin/activate

pip install -e .
```

## Configuration

Copiez le modèle fourni puis renseignez vos propres valeurs :

```bash
cp .env.example .env
```

### Variables d'environnement

| Variable                        | Obligatoire | Description                                                                                          |
| ---------------------------------- | :---------: | ---------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`                     |   **Oui**   | URL de connexion PostgreSQL, ex. Neon : `postgresql://user:password@host/db?sslmode=require`       |
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

Le schéma est défini avec SQLAlchemy dans [`app/models.py`](app/models.py) (modèles `Parcelle`, `FieldScanRun`, et `VhrRefinement` — cette dernière présente en base mais sans code applicatif l'utilisant, conservée pour fidélité de schéma). Les migrations Alembic vivent dans [`migrations/`](migrations).

La base de données de production existe déjà (créée historiquement via `prisma db push`) : la migration [`0001_baseline`](migrations/versions/0001_baseline.py) documente ce schéma pour référence, mais **ne doit être appliquée que via `stamp`**, jamais `upgrade`, pour ne pas tenter de recréer des tables déjà existantes :

```bash
# Sur une base déjà existante (production) : marque la baseline comme "déjà appliquée"
alembic stamp head

# Sur une base neuve (dev vierge) : crée réellement le schéma
alembic upgrade head
```

Pour toute évolution de schéma ultérieure, le cycle Alembic normal s'applique :

```bash
alembic revision --autogenerate -m "description du changement"
# relire la migration générée avant de l'appliquer
alembic upgrade head
```

## Lancer le projet

```bash
uvicorn app.main:app --reload --port 3001
```

Le serveur démarre par défaut sur [http://localhost:3001](http://localhost:3001) (hot reload via `--reload`).

### Lancement en production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3001
```

## API — endpoints

| Méthode | Route                              | Description                                                        |
| --------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `GET`     | `/health`                             | Vérification de l'état du serveur                                        |
| `GET`     | `/api/planet/status`                  | Vérifie que `PLANET_API_KEY` est configurée                              |
| `GET`     | `/api/planet/series`                  | Liste les séries/mosaïques Planet disponibles                            |
| `GET`     | `/api/planet/tiles/:z/:x/:y.png`      | Tuile Planet (proxy)                                                     |
| `GET`     | `/api/parcelles`                      | Liste les parcelles enregistrées (avec cache court)                      |
| `POST`    | `/api/parcelles`                      | Crée une parcelle                                                        |
| `DELETE`  | `/api/parcelles/:id`                  | Supprime une parcelle                                                    |
| `POST`    | `/api/detect-parcels`                 | Détection automatique de parcelles autour d'un point (avec repli si la base est indisponible) |
| `POST`    | `/api/detect-parcels-fallback`        | Détection automatique de parcelles (variante sans repli)                 |
| `POST`    | `/api/analyze-parcel`                 | Analyse détaillée d'une parcelle (indices spectraux, séries temporelles) |
| `POST`    | `/api/zoning-parcel`                  | Zonage NDVI (VRA, 3 zones) d'une parcelle                                 |
| `POST`    | `/api/analyze`                        | Analyse simplifiée d'une zone (rayon, seuils de confiance/surface)       |
| `GET`     | `/api/analyze/latest`                 | Dernière analyse enregistrée pour une position donnée                    |
| `GET`     | `/api/sentinel-tiles/:z/:x/:y`        | Tuile Sentinel-2 (PNG) pour un timestamp d'image donné                   |
| `POST`    | `/api/field-segmentation`             | Segmentation d'une image de parcelle via le modèle U-Net externe          |

Toutes les routes retournent du JSON et valident leur entrée avec [Pydantic](https://docs.pydantic.dev/) (erreurs `400` avec détails en cas d'échec de validation).

La documentation interactive OpenAPI est disponible sur `/docs` (Swagger UI) et `/redoc` en développement.

## Structure du projet

```
AGRISAT-BACKEND/
├── app/
│   ├── main.py                   # Point d'entrée FastAPI, CORS, handlers d'erreur, lifespan
│   ├── config.py                 # Variables d'environnement (pydantic-settings)
│   ├── db.py                     # Moteur SQLAlchemy async
│   ├── models.py                 # Modèles SQLAlchemy (Parcelle, FieldScanRun, VhrRefinement)
│   ├── schemas.py                # Schémas Pydantic (validation des requêtes)
│   ├── errors.py                 # Gestion d'erreurs centralisée
│   ├── cache.py                  # Cache TTL des parcelles
│   ├── routes/                   # Endpoints HTTP
│   └── services/                 # Logique métier (analyse GEE, watershed, zonage, ...)
├── migrations/                   # Migrations Alembic
├── pyproject.toml                # Dépendances Python
├── alembic.ini
└── .env.example                  # Modèle des variables d'environnement
```

## Dépannage

| Problème                                              | Cause probable / solution                                                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Erreurs `CORS` côté frontend                                | `FRONTEND_ORIGIN` ne correspond pas à l'origine réelle du frontend                                |
| Erreurs liées à Google Earth Engine                          | `GEE_SERVICE_ACCOUNT_KEY` manquante/invalide, ou compte de service sans accès au projet GEE       |
| `/api/planet/*` renvoie une erreur 500                       | `PLANET_API_KEY` absente du fichier `.env`                                                        |
| `/api/field-segmentation` renvoie 502                        | Le service `FIELD_SEGMENTATION_MODEL_URL` n'est pas démarré ou n'est pas accessible                |
| `/api/parcelles` renvoie 503                                 | La base de données (`DATABASE_URL`) est inaccessible ou la requête a dépassé le délai imparti      |
