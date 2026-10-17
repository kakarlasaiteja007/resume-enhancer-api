from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import get_settings
from app.controllers.router import api_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Reserve application startup and shutdown hooks for shared resources."""
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    debug=settings.app.debug,
    lifespan=lifespan,
)
app.include_router(api_router, prefix=settings.app.api_prefix)
