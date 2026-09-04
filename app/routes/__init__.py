from fastapi import APIRouter

from app.routes import (
    analyze_parcel,
    analyze_simple,
    detect_parcels,
    field_segmentation,
    health,
    parcelles,
    planet,
    sentinel_tiles,
    zoning,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(planet.router)
api_router.include_router(parcelles.router)
api_router.include_router(analyze_parcel.router)
api_router.include_router(zoning.router)
api_router.include_router(detect_parcels.router)
api_router.include_router(analyze_simple.router)
api_router.include_router(sentinel_tiles.router)
api_router.include_router(field_segmentation.router)
