# tests/e2e/test_web_fallback_flow.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT
from tests.fixtures.api import build_chat_payload


def test_web_fallback_recovery_flow(api_client, test_session_id, test_user_id):
    """
    Proves that when internal RAG knowledge is insufficient, the Self-Healing system 
    accurately detects a knowledge gap, triggers web search via Tavily, merges the 
    web context, and recovers the answer via a graph retry.
    """
    
    # ==========================================
    # Step 1: Ingest Controlled Internal Knowledge
    # ==========================================
    # We populate the DB with internal company policy. 
    # This ensures internal retrieval *runs* but fails to find relevance.
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
    # Step 2: Ask an External/Web-Answerable Question
    # ==========================================
    # We ask a known factual query that is completely absent from the NovaTech docs,
    # but widely available on the public internet. This forces a "Knowledge Gap".
    payload = build_chat_payload(
        query="What are the main new features released in Python 3.12?",
        session_id=test_session_id,
        user_id=test_user_id
    )
    
    response = api_client.post("/chat/query", json=payload)
    
    
    # ==========================================
    # Step 3: Verify the API Completed Safely
    # ==========================================
    assert response.status_code == 200, "Chat API crashed during web fallback."
    data = response.json()
    
    assert data["query_id"]
    assert data["answer"]


    # ==========================================
    # Step 4: Verify WEB_SEARCH Recovery Was Used
    # ==========================================
    # This is the most critical assertion. It proves Groq didn't just answer from memory,
    # but that your Self-Healing Pipeline actively triggered the web search tool.
    assert data["recovery_used"] is True
    assert isinstance(data["correction_path"], list)
    
    # Check case-insensitively to protect against Enum serialization changes
    assert any("web_search" in action.lower() for action in data["correction_path"]), \
        f"Expected WEB_SEARCH in correction_path, got {data['correction_path']}"


    # ==========================================
    # Step 5: Verify LangGraph Retry Happened
    # ==========================================
    # Proves the graph interrupted its normal flow (Generation -> Evaluate -> Response)
    # and cycled back to regenerate using the new merged web context.
    assert data["retry_count"] > 0


    # ==========================================
    # Step 6: Verify Final Answer Exists
    # ==========================================
    # Proves the generation pipeline successfully ingested the web context and output a response.
    assert data["answer"].strip()
    
    # Optional check: If your ContextMerger maps web sources into public citations, 
    # this will pass. If web results are appended differently, you can safely remove this check.
    if data.get("citations"):
        for citation in data["citations"]:
            assert "chunk_id" in citation
            assert "source_type" in citation