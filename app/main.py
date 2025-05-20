from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routers import chunks, documents, libraries, search
from app.core.config import settings
from app.core.persistence import get_persistence
from app.repos.in_memory import initialize_repo
from app.services.exceptions import NotFoundError, ValidationError

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Initializing repository with persistence...")
    await initialize_repo()
    logger.info("Repository initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down, saving data...")
    persistence = get_persistence()
    await persistence.stop_auto_save()
    logger.info("Shutdown complete")


# Register exception handlers
@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    """
    Global exception handler for NotFoundError.
    Returns a 404 response with details about the missing resource.
    """
    logger.warning(f"NotFoundError: {str(exc)}")
    # Convert UUID to string to ensure JSON serialization works
    resource_id = getattr(exc, "resource_id", "unknown")
    if hasattr(resource_id, "__str__"):
        resource_id = str(resource_id)

    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
            "resource_type": getattr(exc, "resource_type", "Resource"),
            "resource_id": resource_id,
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """
    Global exception handler for ValidationError.
    Returns a 422 response with details about the validation error.
    """
    logger.warning(f"ValidationError: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": str(exc),
        },
    )


# Include routers
app.include_router(libraries.router)
app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(search.router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """
    Health check endpoint.
    Returns a simple status message.
    """
    return {"status": "healthy", "environment": settings.ENV}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENV == "dev",
    )
