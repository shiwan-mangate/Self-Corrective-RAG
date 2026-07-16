import hashlib
import logging
from typing import List, Optional

from self_healing.models import RetryState, RecoveryActionType
from self_healing.constants import MAX_RECOVERY_RETRIES

logger = logging.getLogger(__name__)


class RetryManager:
    """
    The exclusive owner and mutator of the RetryState.
    Prevents infinite loops, duplicate generation attempts, and redundant API calls.
    """

    def __init__(self, state: Optional[RetryState] = None):
        self.state = state or RetryState(max_retries=MAX_RECOVERY_RETRIES)

    def get_state(self) -> RetryState:
        """Returns the current snapshot of the state."""
        return self.state

    def reset(self) -> RetryState:
        """Clears the state for a brand new user request."""
        self.state = RetryState(max_retries=MAX_RECOVERY_RETRIES)
        return self.state
    
    def load_state(self, state: RetryState) -> None:
        """
        Loads the current retry state from the LangGraph execution context.
        This ensures the manager knows how many times the orchestrator has already looped.
        """
        self.state = state



    def can_retry(self) -> bool:
        """Checks if the system has hit the global retry ceiling."""
        allowed = self.state.retry_count < self.state.max_retries
        if not allowed:
            logger.warning(
                f"Retry limit reached | Retries={self.state.retry_count}/{self.state.max_retries}"
            )
        return allowed

    def try_retry(self, failure_reason: Optional[str] = None) -> bool:
        """
        Compound check-and-execute method. 
        If a retry is allowed, increments the counter and returns True.
        """
        if self.can_retry():
            self.increment_retry(failure_reason)
            return True
        return False

    def has_visited_query(self, query: str) -> bool:
        """Checks if we have already evaluated this exact phrasing."""
        return query.lower().strip() in self.state.visited_queries

    def has_visited_context(self, context_text: str) -> bool:
        """
        Calculates the SHA-256 of the context to ensure we don't 
        send the exact same context to the Generation Subsystem twice.
        """
        checksum = self._hash_text(context_text)
        return checksum in self.state.visited_context_hashes

    def has_visited_action_sequence(self, actions: List[RecoveryActionType]) -> bool:
        """
        Prevents executing the exact same recovery loop twice.
        """
        signature = self._build_action_signature(actions)
        return signature in self.state.visited_action_sequences

    def can_use_web_search(self) -> bool:
        """Prevents spamming the Tavily API repeatedly in one request."""
        return not self.state.web_search_used



    def increment_retry(self, failure_reason: Optional[str] = None) -> None:
        """Advances the loop counter."""
        self.state.retry_count += 1
        if failure_reason:
            self.state.last_failure_reason = failure_reason
            
        logger.info(f"Retry incremented: {self.state.retry_count}/{self.state.max_retries} | Reason={failure_reason}")
        
        if self.state.retry_count >= self.state.max_retries:
            logger.warning("Maximum recovery retry threshold reached.")

    def record_query(self, query: str) -> None:
        """Saves a query phrasing to prevent duplicate rewriting."""
        normalized_query = query.lower().strip()
        if normalized_query not in self.state.visited_queries:
            self.state.visited_queries.append(normalized_query)

    def record_context(self, context_text: str) -> None:
        """Hashes and saves a context to prevent duplicate generation/evaluation."""
        checksum = self._hash_text(context_text)
        if checksum not in self.state.visited_context_hashes:
            self.state.visited_context_hashes.append(checksum)

    def record_action_sequence(self, actions: List[RecoveryActionType]) -> None:
        """Records a specific execution plan signature."""
        signature = self._build_action_signature(actions)
        if signature not in self.state.visited_action_sequences:
            self.state.visited_action_sequences.append(signature)

    def mark_web_search_used(self) -> None:
        """Flags that external data has been injected into the loop."""
        self.state.web_search_used = True



    def _hash_text(self, text: str) -> str:
        """Creates a fast, deterministic hash for large strings."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _build_action_signature(self, actions: List[RecoveryActionType]) -> str:
        """Creates a readable signature for a sequence of actions (e.g., 'rewrite_query->retry_retrieval')."""
        action_names = [a.value for a in actions]
        return "->".join(action_names)