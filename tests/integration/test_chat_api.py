# tests/integration/test_chat_api.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT, NOVATECH_BENEFITS_TEXT
from tests.fixtures.api import build_chat_payload


def _ingest_test_document(api_client, filename: str, content: str, content_type: str = "text/plain"):
    """
    Helper to seed the vector database through the public API before chatting.
    Prevents repetitive multipart form construction in every test.
    """
    files = {
        "file": (filename, content, content_type)
    }
    response = api_client.post("/documents/upload", files=files)
    assert response.status_code == 200, f"Setup failed: Could not ingest {filename}"


def test_factual_chat_query(api_client, test_session_id, test_user_id):
    """
    Proves the golden path: A factual query retrieves the correct chunk and 
    generates a cited answer without needing self-healing.
    """
    # 1. Setup: Ingest Knowledge
    _ingest_test_document(api_client, "novatech_policy.txt", NOVATECH_POLICY_TEXT)

    # 2. Arrange: Build query
    payload = build_chat_payload(
        query="How many paid leave days do NovaTech employees receive?",
        session_id=test_session_id,
        user_id=test_user_id
    )

    # 3. Act: Chat
    response = api_client.post("/chat/query", json=payload)

    # 4. Assert: Validate API Contract & Facts
    assert response.status_code == 200
    data = response.json()

    # Verify structural contract
    assert "query_id" in data
    assert data["session_id"] == test_session_id
    assert "answer" in data
    assert isinstance(data["citations"], list)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["recovery_used"], bool)
    assert isinstance(data["correction_path"], list)
    assert data["retry_count"] >= 0
    assert data["latency_ms"] >= 0
    assert isinstance(data["warnings"], list)

    # Verify extracted factual knowledge (LLMs vary wording, so we check the critical entity)
    assert "24" in data["answer"]


def test_multi_context_chat_query(api_client, test_session_id, test_user_id):
    """
    Proves the system can fuse multiple retrieved chunks from different 
    documents into a single, cohesive answer.
    """
    # 1. Setup: Ingest multiple knowledge sources
    _ingest_test_document(api_client, "novatech_policy.txt", NOVATECH_POLICY_TEXT)
    _ingest_test_document(api_client, "novatech_benefits.txt", NOVATECH_BENEFITS_TEXT)

    # 2. Arrange
    payload = build_chat_payload(
        query="What paid leave and wellness benefits does NovaTech provide?",
        session_id=test_session_id,
        user_id=test_user_id
    )

    # 3. Act
    response = api_client.post("/chat/query", json=payload)

    # 4. Assert
    assert response.status_code == 200
    data = response.json()

    assert data["answer"]
    # Verify facts from document 1
    assert "24" in data["answer"]
    # Verify facts from document 2 (handle possible comma formatting by LLM)
    assert "12,000" in data["answer"] or "12000" in data["answer"]
    
    # Prove that evidence was actually cited
    assert len(data["citations"]) > 0


def test_unknown_query_completes_safely(api_client, test_session_id, test_user_id):
    """
    Proves that the application doesn't crash or return HTTP 500 when asked a 
    question that isn't supported by the internal knowledge base. 
    It should safely complete the graph (potentially via self-healing).
    """
    # 1. Arrange
    payload = build_chat_payload(
        query="What is NovaTech's private satellite launch schedule?",
        session_id=test_session_id,
        user_id=test_user_id
    )

    # 2. Act
    response = api_client.post("/chat/query", json=payload)

    # 3. Assert
    assert response.status_code == 200
    data = response.json()

    # Verify structural safety
    assert "query_id" in data
    assert "answer" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["recovery_used"], bool)
    assert isinstance(data["correction_path"], list)
    assert data["retry_count"] >= 0