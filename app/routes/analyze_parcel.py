from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.analyze_parcel import analyze_parcel

router = APIRouter()


@router.post("/api/analyze-parcel")
async def analyze_parcel_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(error) or "Unknown error"})
    status, data = await analyze_parcel(body)
    return JSONResponse(status_code=status, content=data)
