# tests/integration/test_evaluation_api.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT
from tests.fixtures.api import build_chat_payload


def _ingest_test_document(api_client, filename: str, content: str, content_type: str = "text/plain"):
    """Helper to seed the vector database."""
    files = {
        "file": (filename, content, content_type)
    }
    response = api_client.post("/documents/upload", files=files)
    assert response.status_code == 200, f"Setup failed: Could not ingest {filename}"


def test_get_evaluation_for_completed_query(api_client, test_session_id, test_user_id):
    """
    Proves that a completed Chat API query successfully persists its EvaluationReport 
    to the database, and that the Evaluation API correctly maps the DB row back 
    to the public EvaluationResponse schema.
    """
    # 1. Setup: Ingest Knowledge
    _ingest_test_document(api_client, "novatech_policy.txt", NOVATECH_POLICY_TEXT)

    # 2. Arrange: Perform a chat query to generate an evaluation run
    payload = build_chat_payload(
        query="How many paid leave days do NovaTech employees receive?",
        session_id=test_session_id,
        user_id=test_user_id
    )
    chat_response = api_client.post("/chat/query", json=payload)
    assert chat_response.status_code == 200, "Chat request failed, cannot test evaluation."
    
    query_id = chat_response.json()["query_id"]

    # 3. Act: Fetch the evaluation
    eval_response = api_client.get(f"/evaluation/{query_id}")

    # 4. Assert: Validate the Evaluation Contract
    assert eval_response.status_code == 200
    data = eval_response.json()

    # Validate Top-Level Keys
    assert "grounding" in data
    assert "hallucination" in data
    assert "confidence" in data
    assert "ragas" in data
    assert "retry_recommended" in data

    # Validate Grounding
    assert isinstance(data["grounding"]["is_grounded"], bool)
    assert 0.0 <= data["grounding"]["confidence"] <= 1.0

    # Validate Hallucination
    assert isinstance(data["hallucination"]["detected"], bool)
    assert isinstance(data["hallucination"]["risk"], str)

    # Validate Confidence
    assert 0.0 <= data["confidence"]["score"] <= 1.0
    assert 0.0 <= data["confidence"]["retrieval_confidence"] <= 1.0
    assert 0.0 <= data["confidence"]["grounding_confidence"] <= 1.0

    # Validate Ragas (Live vs Benchmark fields are Optional)
    ragas = data["ragas"]
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        val = ragas[metric]
        assert val is None or (0.0 <= val <= 1.0), f"Ragas metric {metric} out of bounds: {val}"

    # Validate Policy Recommendation
    assert isinstance(data["retry_recommended"], bool)


def test_unknown_evaluation_returns_404(api_client):
    """
    Proves that the API safely handles requests for non-existent evaluations.
    If this fails with a 500, it exposes a bug in `EXCEPTION_MAPPINGS` in 
    `api/middleware/exception_handler.py` where `EvaluationNotFoundError` 
    was missed and defaulted to APPLICATION_ERROR.
    """
    # 1. Arrange
    fake_query_id = "req_this_does_not_exist_999"

    # 2. Act
    response = api_client.get(f"/evaluation/{fake_query_id}")

    # 3. Assert
    assert response.status_code == 404
    
    data = response.json()
    assert data["error_code"] == "EVALUATION_NOT_FOUND" or "NOT_FOUND" in data["error_code"]