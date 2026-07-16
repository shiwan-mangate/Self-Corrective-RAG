# tests/e2e/test_memory_flow.py

from tests.fixtures.documents import NOVATECH_POLICY_TEXT
from tests.fixtures.api import build_chat_payload


def test_contextual_memory_flow(api_client, test_session_id, test_user_id):
    """
    Proves the RAG system leverages persistent conversational memory 
    to understand and correctly answer contextual follow-up queries.
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
    assert upload_data["chunks_persisted"] > 0


    # ==========================================
    # Step 2: Send the First Chat Request (Establish Context)
    # ==========================================
    first_payload = build_chat_payload(
        query="What is NovaTech's remote work policy?",
        session_id=test_session_id,
        user_id=test_user_id
    )
    
    first_response = api_client.post("/chat/query", json=first_payload)
    
    assert first_response.status_code == 200
    first_data = first_response.json()
    assert first_data["answer"]
    
    # Verify the initial correct fact was retrieved
    first_answer = first_data["answer"].lower()
    assert "3" in first_answer or "three" in first_answer


    # ==========================================
    # Step 3: Send a Contextual Follow-Up
    # ==========================================
    # The word "that" relies entirely on the memory subsystem correctly 
    # injecting the previous turn into the current graph state.
    second_payload = build_chat_payload(
        query="How many days per week is that?",
        session_id=test_session_id,
        user_id=test_user_id
    )
    
    second_response = api_client.post("/chat/query", json=second_payload)
    
    assert second_response.status_code == 200
    second_data = second_response.json()
    
    
    # ==========================================
    # Step 4 & 5: Verify the Follow-Up Answer & Session Linking
    # ==========================================
    # Prove both queries belong to the same session
    assert first_data["session_id"] == test_session_id
    assert second_data["session_id"] == test_session_id
    
    # Prove the system resolved "that" to "remote work policy"
    assert second_data["answer"]
    second_answer = second_data["answer"].lower()
    assert "3" in second_answer or "three" in second_answer


    # ==========================================
    # Step 6: Verify Conversation Persistence
    # ==========================================
    memory_response = api_client.get(f"/memory/{test_session_id}")
    
    assert memory_response.status_code == 200
    memory_data = memory_response.json()
    
    assert memory_data["session"]["session_id"] == test_session_id
    
    # We had 2 user queries and 2 assistant responses, so expect >= 4
    assert memory_data["session"]["message_count"] >= 4
    assert len(memory_data["messages"]) >= 4
    
    user_messages = [
        message for message in memory_data["messages"] 
        if message["role"] == "user"
    ]
    
    # Prove both exact user queries were permanently recorded
    assert any("remote work policy" in msg["content"].lower() for msg in user_messages)
    assert any("how many days per week is that" in msg["content"].lower() for msg in user_messages)