# tests/e2e/test_full_rag_flow.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT
from tests.fixtures.api import build_chat_payload


def test_full_rag_flow(api_client, test_session_id, test_user_id):
    """
    The Golden Path E2E Test.
    Proves a real user can upload knowledge, ask a question, receive a grounded answer, 
    and then find both the evaluation and conversation persisted securely.
    """
    
    # ==========================================
    # Step 1: Upload the NovaTech Policy
    # ==========================================
    files = {
        "file": (
            "novatech_policy.txt",
            NOVATECH_POLICY_TEXT,
            "text/plain"
        )
    }
    upload_response = api_client.post("/documents/upload", files=files)
    
    assert upload_response.status_code == 200, "Ingestion failed, cannot continue E2E test."
    upload_data = upload_response.json()
    assert upload_data["documents_processed"] > 0
    assert upload_data["chunks_generated"] > 0
    assert upload_data["chunks_persisted"] > 0


    # ==========================================
    # Step 2: Ask a Factual Question
    # ==========================================
    payload = build_chat_payload(
        query="How many paid leave days do NovaTech employees receive?",
        session_id=test_session_id,
        user_id=test_user_id
    )
    chat_response = api_client.post("/chat/query", json=payload)
    
    
    # ==========================================
    # Step 3: Verify the Final Answer
    # ==========================================
    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    
    assert chat_data["query_id"]
    assert chat_data["session_id"] == test_session_id
    assert chat_data["answer"]
    
    # Verify the most important business fact was successfully retrieved and generated
    assert "24" in chat_data["answer"]
    
    # Verify the telemetry and contract
    assert len(chat_data["citations"]) > 0
    assert 0.0 <= chat_data["confidence"] <= 1.0
    assert chat_data["latency_ms"] >= 0
    assert chat_data["status"] == "success"
    
    
    # ==========================================
    # Step 4: Save the Query ID
    # ==========================================
    query_id = chat_data["query_id"]
    
    
    # ==========================================
    # Step 5 & 6: Retrieve & Verify Evaluation
    # ==========================================
    evaluation_response = api_client.get(f"/evaluation/{query_id}")
    assert evaluation_response.status_code == 200
    evaluation_data = evaluation_response.json()
    
    assert "grounding" in evaluation_data
    assert "hallucination" in evaluation_data
    assert "confidence" in evaluation_data
    assert "ragas" in evaluation_data
    
    # Because this is a controlled golden path test, we apply STRICT assertions.
    # We explicitly expect a perfectly grounded, hallucination-free report.
    assert evaluation_data["grounding"]["is_grounded"] is True
    assert evaluation_data["hallucination"]["detected"] is False
    assert 0.0 <= evaluation_data["confidence"]["score"] <= 1.0


    # ==========================================
    # Step 7 & 8: Retrieve & Verify Memory
    # ==========================================
    memory_response = api_client.get(f"/memory/{test_session_id}")
    assert memory_response.status_code == 200
    memory_data = memory_response.json()
    
    assert memory_data["session"]["session_id"] == test_session_id
    assert memory_data["session"]["message_count"] >= 2
    assert len(memory_data["messages"]) >= 2
    
    # Prove the specific user question was persisted
    assert any(
        "paid leave" in message["content"].lower()
        for message in memory_data["messages"]
        if message["role"] == "user"
    )
    
    # Prove the assistant's answer was persisted
    assert any(
        message["role"] == "assistant"
        for message in memory_data["messages"]
    )