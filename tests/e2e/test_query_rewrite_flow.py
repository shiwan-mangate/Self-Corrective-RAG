# tests/e2e/test_query_rewrite_flow.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT
from tests.fixtures.api import build_chat_payload


def test_query_rewrite_recovery_flow(api_client, test_session_id, test_user_id,force_query_rewrite,):
    """
    Proves the Self-Healing RAG system can detect a weak query, trigger a rewrite,
    loop back through the retrieval subsystem, and successfully recover the answer.
    """
    
    # ==========================================
    # Step 1: Upload the NovaTech Policy
    # ==========================================
    # We must ensure the knowledge actually exists in the DB so that the 
    # failure is caused by the query quality, not missing knowledge.
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
    assert upload_data["chunks_persisted"] > 0


    # ==========================================
    # Step 2: Send a Deliberately Weak Query
    # ==========================================
    # This query is designed to retrieve poorly or yield low confidence,
    # forcing the PolicyValidator to trigger a REWRITE_QUERY recovery.
    payload = build_chat_payload(
    query="How many paid leave days do NovaTech employees receive per year?",
    session_id=test_session_id,
    user_id=test_user_id,
)
    
    response = api_client.post("/chat/query", json=payload)
    
    
    # ==========================================
    # Step 3: Verify the API Completed Safely
    # ==========================================
    assert response.status_code == 200
    data = response.json()
    print("\n========== QUERY REWRITE DEBUG ==========")
    print("Status:", data["status"])
    print("Answer:", data["answer"])
    print("Confidence:", data["confidence"])
    print("Recovery Used:", data["recovery_used"])
    print("Correction Path:", data["correction_path"])
    print("Retry Count:", data["retry_count"])
    print("Citations:", data["citations"])
    print("=========================================\n")
    
    assert data["query_id"]
    assert data["answer"]


    # ==========================================
    # Step 4: Verify REWRITE_QUERY Happened
    # ==========================================
    # This proves the self-healing subsystem actively intervened.
    assert data["recovery_used"] is True
    assert isinstance(data["correction_path"], list)
    
    # Check that query rewriting was explicitly in the correction path.
    # We check case-insensitively to protect against Enum serialization changes.
    assert any("rewrite_query" in action.lower() for action in data["correction_path"]), \
        f"Expected REWRITE_QUERY in correction_path, got {data['correction_path']}"


    # ==========================================
    # Step 5: Verify LangGraph Retry Happened
    # ==========================================
    # Proves the graph actually cycled back to the retrieval phase.
    assert data["retry_count"] > 0


    # ==========================================
    # Step 6: Verify the Final Answer Recovered
    # ==========================================
    # Proves that the rewrite actually worked and extracted the factual data.
    assert "24" in data["answer"]
    
    # Proves the recovered answer is still properly grounded with evidence.
    assert len(data["citations"]) > 0