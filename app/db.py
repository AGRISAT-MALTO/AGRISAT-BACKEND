from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from app.config import settings


def _to_asyncpg_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError(
            "DATABASE_URL is not configured. Copy .env.example to .env and set "
            "DATABASE_URL to your PostgreSQL connection URL."
        )
    try:
        parsed_url = make_url(url)
    except ArgumentError:
        raise ValueError(
            "DATABASE_URL must be a valid PostgreSQL connection URL "
            "(for example, postgresql://USER:PASSWORD@HOST/DATABASE)."
        ) from None
    if parsed_url.drivername not in ("postgresql", "postgresql+asyncpg"):
        raise ValueError(
            "DATABASE_URL must use postgresql:// or postgresql+asyncpg://."
        )

    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    # asyncpg does not understand libpq query params negotiated over the wire;
    # SSL is handled via connect_args below instead.
    for param in ("?sslmode=require", "&sslmode=require", "?channel_binding=require", "&channel_binding=require"):
        url = url.replace(param, "" if param.startswith("?") else "")
    if "?" not in url and "&" in url:
        # a leading "&" was left behind if sslmode was the first param
        url = url.replace("&", "?", 1)
    return url


# Moteur unique partagé par tout le backend (async), miroir du client Prisma singleton :
# plusieurs moteurs ouvriraient chacun leur propre pool de connexions vers Neon.
engine = create_async_engine(
    _to_asyncpg_url(settings.database_url),
    connect_args={"ssl": True},
    pool_pre_ping=True,
)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session
        