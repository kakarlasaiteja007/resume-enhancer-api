from app.config.settings import get_settings
from app.models.health import HealthResponse


class HealthService:
    """Provides health information without coupling it to the HTTP layer."""

    def get_health(self) -> HealthResponse:
        settings = get_settings()
        return HealthResponse(status="ok", environment=settings.environment)
