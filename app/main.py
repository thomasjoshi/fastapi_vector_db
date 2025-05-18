from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routers import libraries, search
from app.core.config import settings
from app.services.errors import NotFoundError

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
)


# Register exception handlers
@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    """
    Global exception handler for NotFoundError.
    Returns a 404 response with details about the missing resource.
    """
    logger.warning(f"NotFoundError: {str(exc)}")
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
            "resource_type": exc.resource_type,
            "resource_id": exc.resource_id,
        },
    )


# Include routers
app.include_router(libraries.router)
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
