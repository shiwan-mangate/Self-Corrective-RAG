# api/routers/health.py
import logging
from fastapi import APIRouter, Depends, Response

from api.dependencies import get_application_health
from core.health import ApplicationHealth, LivenessReport, ReadinessReport

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("/liveness", response_model=LivenessReport)
def check_liveness(
    health_service: ApplicationHealth = Depends(get_application_health)
) -> LivenessReport:
    """
    K8s/Docker Liveness Probe.
    Answers: "Is the application process running and capable of answering HTTP traffic?"
    Strictly avoids touching databases or external APIs.
    """
    logger.debug("Liveness probe requested.")
    return health_service.check_liveness()


@router.get("/readiness", response_model=ReadinessReport)
def check_readiness(
    response: Response,
    health_service: ApplicationHealth = Depends(get_application_health)
) -> ReadinessReport:
    """
    K8s/Docker/LoadBalancer Readiness Probe.
    Answers: "Is the application fully connected to its infrastructure and ready to serve traffic?"
    Executes lightweight connectivity checks against PostgreSQL/Neon.
    """
    logger.debug("Readiness probe requested.")
    
   
    report = health_service.check_readiness()
    
   
    if not report.ready:
    
        response.status_code = 503
        logger.warning(f"Readiness check failed. Application is degraded. Latency: {report.latency_ms}ms")
    
   
    return report