# tests/integration/test_health_api.py

def test_liveness_api(api_client):
    """
    Proves that the FastAPI application booted successfully, registered its 
    routers, and can return a basic HTTP response without crashing.
    """
    # Act
    response = api_client.get("/health/liveness")

    # Assert
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"


def test_readiness_api(api_client):
    """
    Proves that the application can successfully connect to the PostgreSQL 
    database (Neon) and that the ReadinessReport schema is correctly formatted.
    """
    # Act
    response = api_client.get("/health/readiness")

    # Assert
    # If this fails with a 503, it means your Neon Database is currently unreachable
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ready"
    assert data["ready"] is True
    
    # Verify the granular dependency checks
    assert "database" in data["checks"]
    assert data["checks"]["database"] is True
    
    # Verify that telemetry is being tracked (type and bounds)
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (float, int))
    assert data["latency_ms"] >= 0