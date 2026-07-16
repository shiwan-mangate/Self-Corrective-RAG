# tests/integration/test_memory_api.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT
from tests.fixtures.api import build_chat_payload


def _ingest_test_document(api_client, filename: str, content: str, content_type: str = "text/plain"):
    """Helper to seed the vector database."""
    files = {
        "file": (filename, content, content_type)
    }
    response = api_client.post("/documents/upload", files=files)
    assert response.status_code == 200, f"Setup failed: Could not ingest {filename}"


def test_get_persisted_conversation_history(api_client, test_session_id, test_user_id):
    """
    Proves that a completed Chat API query successfully persists its conversation 
    turns to the PostgreSQL database, and that the Memory API correctly retrieves, 
    filters, and maps the history back to the public schema.
    """
    # 1. Setup: Ingest Knowledge
    _ingest_test_document(api_client, "novatech_policy.txt", NOVATECH_POLICY_TEXT)

    # 2. Arrange: Perform a chat query to generate conversation turns
    query_text = "How many paid leave days do NovaTech employees receive?"
    payload = build_chat_payload(
        query=query_text,
        session_id=test_session_id,
        user_id=test_user_id
    )
    
    chat_response = api_client.post("/chat/query", json=payload)
    assert chat_response.status_code == 200, "Chat request failed, cannot test memory."

    # 3. Act: Fetch the persisted memory
    memory_response = api_client.get(f"/memory/{test_session_id}")

    # 4. Assert: Validate the Memory Contract
    assert memory_response.status_code == 200
    data = memory_response.json()

    # Verify top-level structure
    assert "session" in data
    assert "messages" in data

    # Verify Session Metadata
    session = data["session"]
    assert session["session_id"] == test_session_id
    assert "created_at" in session
    assert "last_activity" in session
    assert session["message_count"] >= 2  # At least 1 user turn + 1 assistant turn
    assert isinstance(session["active"], bool)

    # Verify Public Messages
    messages = data["messages"]
    assert len(messages) >= 2

    for message in messages:
        assert "message_id" in message
        assert "role" in message
        assert "content" in message
        assert "timestamp" in message

    # Security & Privacy verification: Prove internal system/tool prompts are stripped
    assert all(msg["role"] in {"user", "assistant"} for msg in messages)

    # Prove the specific user interaction was captured accurately
    assert any(
        "paid leave" in msg["content"].lower() 
        for msg in messages if msg["role"] == "user"
    )


def test_unknown_session_returns_404(api_client):
    """
    Proves that the Memory API safely handles requests for non-existent sessions,
    and correctly maps the domain exception to a standardized 404 response.
    """
    # 1. Arrange
    fake_session_id = "session_that_does_not_exist_999"

    # 2. Act
    response = api_client.get(f"/memory/{fake_session_id}")

    # 3. Assert
    assert response.status_code == 404
    
    data = response.json()
    assert data["error_code"] == "SESSION_NOT_FOUND"
    assert "message" in data
    assert "request_id" in data
    
    # Detail should generally be None for standard expected errors in this architecture
    assert data.get("details") is None