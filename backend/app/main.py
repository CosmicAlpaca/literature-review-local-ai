import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth import router as auth_router
from app.api.sessions import router as sessions_router
from app.api.search import router as search_router
from app.api.admin import router as admin_router
from app.api.llm import router as llm_router
from app.api.review import router as review_router
from app.api.pdf_chat import router as pdf_chat_router
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    logger.exception(
        "Unhandled request failure: request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_server_error",
                "message": "Internal server error.",
                "request_id": request_id,
                "retryable": True,
            }
        },
        headers={"X-Request-ID": request_id},
    )

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(search_router)
app.include_router(admin_router)
app.include_router(llm_router)
app.include_router(review_router)
app.include_router(pdf_chat_router)


def _health_payload():
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": "development" if settings.debug else "production",
        "debug": settings.debug,
        "search_provider_mode": settings.search_provider_mode,
        "llm_enabled": settings.llm_enabled,
    }


@app.get("/")
def root():
    return _health_payload()


@app.get("/health")
def health():
    return _health_payload()