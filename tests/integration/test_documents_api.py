# tests/integration/test_documents_api.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT, NOVATECH_SECURITY_MARKDOWN
from tests.fixtures.api import build_url_ingestion_payload


def test_upload_txt_document(api_client):
    """
    Proves that a standard plain text file can be successfully uploaded, 
    parsed, chunked, embedded, and persisted to the vector database.
    """
    # 1. Arrange: Create the multipart form data
    files = {
        "file": (
            "novatech_policy.txt",
            NOVATECH_POLICY_TEXT,
            "text/plain"
        )
    }

    # 2. Act: Send through the FastAPI HTTP boundary
    response = api_client.post("/documents/upload", files=files)

    # 3. Assert: Verify HTTP Status and exact Contract Schema
    assert response.status_code == 200
    data = response.json()
    
    assert data["documents_processed"] > 0
    assert data["chunks_generated"] > 0
    assert data["chunks_persisted"] > 0
    assert isinstance(data["warnings"], list)
    assert data["elapsed_time_sec"] >= 0


def test_upload_markdown_document(api_client):
    """
    Proves multi-format ingestion by validating that Markdown files are 
    correctly accepted and routed by the LoaderFactory.
    """
    # 1. Arrange
    files = {
        "file": (
            "novatech_security.md",
            NOVATECH_SECURITY_MARKDOWN,
            "text/markdown"
        )
    }

    # 2. Act
    response = api_client.post("/documents/upload", files=files)

    # 3. Assert
    assert response.status_code == 200
    data = response.json()
    
    assert data["documents_processed"] > 0
    assert data["chunks_generated"] > 0
    assert data["chunks_persisted"] > 0


def test_reject_unsupported_file(api_client):
    """
    Proves that the API boundary actively protects the ingestion pipeline 
    from malicious or unsupported file extensions.
    """
    # 1. Arrange
    files = {
        "file": (
            "malware.exe",
            b"fake executable content",
            "application/octet-stream"
        )
    }

    # 2. Act
    response = api_client.post("/documents/upload", files=files)

    # 3. Assert
    assert response.status_code == 415
    data = response.json()
    
    # Verify the FastAPI HTTPException was raised with the correct detail message
    assert "detail" in data
    assert "Unsupported file extension" in data["detail"]


def test_ingest_document_from_url(api_client):
    """
    Proves that the URL loader successfully reaches out to the internet, 
    extracts the HTML, and completes the vector persistence cycle.
    """
    # 1. Arrange: Use a stable URL (the one defined as the example in your schema)
    payload = build_url_ingestion_payload(
        source="https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    )

    # 2. Act
    response = api_client.post("/documents/url", json=payload)

    # 3. Assert
    assert response.status_code == 200
    data = response.json()
    
    assert data["documents_processed"] > 0
    assert data["chunks_generated"] > 0
    assert data["chunks_persisted"] > 0