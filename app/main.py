import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import cache
from app.config import settings
from app.errors import register_error_handlers
from app.routes import api_router

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
            logger.warning("parcelles: préchauffage initial échoué, réessai à la première requête: %s", error)

    asyncio.ensure_future(_warmup())
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
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
