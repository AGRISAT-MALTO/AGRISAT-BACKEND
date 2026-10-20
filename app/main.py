import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import cache
from app.config import settings
from app.errors import register_error_handlers
from app.services.barley_rf_classifier import load_model as load_barley_model
from app.services.field_segmentation import warmup as warmup_field_segmentation
from app.services.vhr_launcher import ensure_vhr_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrisat")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Préchauffage non bloquant du cache parcelles au démarrage (miroir du bas de
    # server.ts : loadParcelles().catch(...) après app.listen()).
    async def _warmup() -> None:
        try:
            await cache.load_parcelles()
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "parcelles: préchauffage initial échoué, réessai à la première requête: %s",
                cache.describe_error(error),
            )

    asyncio.ensure_future(_warmup())

    # Auto-start du service de tracage VHR (serve_vhr.py, modele delineate_jecam_complet.pt)
    # si VHR_SERVICE_DIR est renseigne : evite de demarrer le service a la main a chaque
    # session. Detache du backend : survit a un redemarrage (--reload inclus), et rien
    # n'est lance si le service repond deja. Details dans app/services/vhr_launcher.py.
    async def _vhr_autostart() -> None:
        try:
            await ensure_vhr_service()
        except Exception as error:  # noqa: BLE001 — l'API doit demarrer quoi qu'il arrive
            logger.warning("auto-start service tracage VHR : echec non bloquant : %s", error)

    asyncio.ensure_future(_vhr_autostart())

    # Chargement des modèles embarqués (YOLO de segmentation + Random Forest orge)
    # hors de la boucle d'événements : la première requête ne paie pas l'import de
    # torch ni la lecture des poids. Un échec reste non bloquant.
    # WARMUP_MODELS=false saute cette étape (démarrage/reload immédiat, premier
    # appel de segmentation plus lent).
    async def _warmup_models() -> None:
        try:
            await asyncio.to_thread(warmup_field_segmentation)
            logger.info("Modèle de segmentation de parcelles chargé (inférence locale).")
        except Exception as error:  # noqa: BLE001
            logger.warning("Modèle de segmentation de parcelles indisponible : %s", error)
        try:
            model = await asyncio.to_thread(load_barley_model)
            if model is not None:
                logger.info("Modèle orge chargé (RF, %d variables).", len(model["features"]))
        except Exception as error:  # noqa: BLE001
            logger.warning("Modèle orge indisponible : %s", error)

    # Attendu avant d'ouvrir le service : la première requête de segmentation ne
    # paie ni l'import de torch ni le premier passage du modèle. Désactivable pour
    # un démarrage rapide (les modèles se chargent alors à la première requête).
    if settings.warmup_models:
        await _warmup_models()
    else:
        logger.info("Préchauffage des modèles désactivé (WARMUP_MODELS=false).")
    yield


app = FastAPI(title="AGRISAT Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origin.split(",") if settings.frontend_origin else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

from app.routes import api_router  # noqa: E402 — après la config CORS/erreurs, avant le montage

app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
