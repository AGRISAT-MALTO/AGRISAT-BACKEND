import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("agrisat")


class AppError(Exception):
    """Erreur HTTP ad-hoc, miroir de `reply.code(x).send({error: ...})` côté Fastify."""

    def __init__(self, status_code: int, message: str, extra: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.extra = extra or {}
        super().__init__(message)


class AppValidationError(AppError):
    """Miroir de `reply.code(400).send({error, details: parsed.error.flatten()})`."""

    def __init__(self, message: str, details: dict):
        super().__init__(400, message, {"details": details})


def _flatten_validation_error(exc: ValidationError) -> dict:
    field_errors: dict[str, list[str]] = {}
    form_errors: list[str] = []
    for err in exc.errors():
        loc = [str(p) for p in err["loc"]]
        msg = err["msg"]
        if loc:
            field_errors.setdefault(loc[0], []).append(msg)
        else:
            form_errors.append(msg)
    return {"fieldErrors": field_errors, "formErrors": form_errors}


def parse_or_400(model: type[BaseModel], raw: object, message: str) -> BaseModel:
    """Valide `raw` contre `model` ; lève AppValidationError (400, forme {error, details})
    au lieu de laisser FastAPI répondre 422 {"detail": [...]}."""
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise AppValidationError(message, _flatten_validation_error(exc)) from exc


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message, **exc.extra})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        status_code = exc.status_code if exc.status_code < 500 else 500
        message = str(exc.detail) if status_code < 500 else "Erreur interne du serveur"
        return JSONResponse(status_code=status_code, content={"error": message})

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Ne devrait normalement jamais être atteint : chaque route valide manuellement
        # via parse_or_400. Filet de sécurité pour garder la forme {"error": ...} partout.
        return JSONResponse(status_code=400, content={"error": "Requête invalide.", "details": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception", exc_info=exc)
        return JSONResponse(status_code=500, content={"error": "Erreur interne du serveur"})
