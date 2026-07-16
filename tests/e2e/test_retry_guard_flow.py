# tests/e2e/test_retry_guard_flow.py

from tests.fixtures.api import build_chat_payload
from self_healing.constants import MAX_RECOVERY_RETRIES


def test_retry_guard_stops_failed_recovery_flow(api_client, test_session_id, test_user_id):
    """
    Proves that a query that consistently fails evaluation (even after recovery attempts)
    will safely exhaust the retry limit and terminate via the RetryGuard, returning 
    a valid, degraded public response instead of looping infinitely.
    """
    
    # ==========================================
    # Step 1: Send an Unrecoverable Query
    # ==========================================
    # We ask for a completely fictional, highly specific fact that does not exist 
    # in the internal database OR the public web. This guarantees that:
    # 1. Internal retrieval fails.
    # 2. Web search fails.
    # 3. Evaluation consistently fails on poor grounding/confidence.
    payload = build_chat_payload(
        query="What is the confidential launch authorization code for NovaTech's Project Orion?",
        session_id=test_session_id,
        user_id=test_user_id
    )
    
    # ==========================================
    # Step 2 & 3: Act & Prove HTTP Completed
    # ==========================================
    # If this request hangs or times out, the Retry Guard is broken and the graph is in an infinite loop.
    response = api_client.post("/chat/query", json=payload)
    
    assert response.status_code == 200, "Retry exhaustion caused an HTTP 500 error instead of a graceful degradation."
    data = response.json()
    
    
    # ==========================================
    # Step 4: Verify Recovery Was Attempted
    # ==========================================
    assert data["recovery_used"] is True
    assert isinstance(data["correction_path"], list)
    assert len(data["correction_path"]) > 0, "System gave up immediately without trying to recover."


    # ==========================================
    # Step 5 & 6: Verify Retry Limit Was Reached
    # ==========================================
    # This is the most critical assertion: it proves the graph cycled exactly 
    # up to the configured limit and was forcefully terminated.
    assert data["retry_count"] > 0
    assert data["retry_count"] == MAX_RECOVERY_RETRIES, \
        f"Expected graph to terminate at exactly {MAX_RECOVERY_RETRIES} retries, but got {data['retry_count']}."


    # ==========================================
    # Step 7: Verify Degraded/Insufficient Status
    # ==========================================
    # The system must not pretend it succeeded. 
    assert data["status"] != "SUCCESS", "Response status claimed SUCCESS despite exhausting all retries."
    
    # The LLM's final generated answer should gracefully admit lack of information
    answer = data["answer"].lower()
    assert any(word in answer for word in ["insufficient", "cannot", "don't have", "not", "unable"]), \
        f"Model hallucinated an answer instead of admitting failure: {data['answer']}"


    # ==========================================
    # Step 8: Verify Public API Schema Remains Valid
    # ==========================================
    # Proves the Graph safely mapped its terminated state back to the ChatResponse contract
    assert data["query_id"]
    assert data["session_id"] == test_session_id
    assert isinstance(data["citations"], list)
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] >= 0
    assert isinstance(data["warnings"], list)