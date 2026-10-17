from fastapi import APIRouter, status

from app.models.health import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(prefix="/health", tags=["health"])
health_service = HealthService()


@router.get("", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def get_health() -> HealthResponse:
    """Return the service health and active environment."""
    return health_service.get_health()
