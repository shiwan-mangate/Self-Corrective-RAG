# tests/fixtures/api.py

"""
Reusable API payload builders for Integration and E2E tests.
Contains plain Python functions returning dictionaries. No pytest fixtures, 
HTTP clients, or database logic belong here.
"""

from typing import Dict, Any, Optional


def build_chat_payload(
    query: str,
    session_id: str,
    user_id: Optional[str] = None,
    query_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a valid JSON payload for POST /chat/query.
    Omits optional fields if they are not provided to accurately 
    simulate standard client requests.
    """
    payload: Dict[str, Any] = {
        "query": query,
        "session_id": session_id
    }
    
    if user_id is not None:
        payload["user_id"] = user_id
        
    if query_id is not None:
        payload["query_id"] = query_id
        
    return payload


def build_url_ingestion_payload(source: str) -> Dict[str, str]:
    """
    Constructs a valid JSON payload for POST /documents/url.
    """
    return {
        "source": source
    }