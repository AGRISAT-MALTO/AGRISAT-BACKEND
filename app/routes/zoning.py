from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.zoning import compute_zoning

router = APIRouter()


@router.post("/api/zoning-parcel")
async def zoning_parcel_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(error) or "Unknown error"})
    status, data = await compute_zoning(body)
    return JSONResponse(status_code=status, content=data)
