# memory/summarization/policy.py

import logging
from memory.models import MemoryRequest, ConversationHistory, SummaryDecision, SummaryTrigger
from memory.constants import (
    SUMMARY_TRIGGER_MESSAGES, 
    SUMMARY_TRIGGER_TOKENS, 
    MIN_MESSAGES_FOR_SUMMARY
)

logger = logging.getLogger(__name__)


class SummaryPolicy:
    """
    The Rules Engine for Memory Compression.
    
    Architecture Note:
    - NO EXECUTION: Does not call LLMs, databases, or orchestrators.
    - PURE LOGIC: Deterministically evaluates thresholds and explicit system requests.
    - STRICT TYPING: Returns a Pydantic SummaryDecision using the SummaryTrigger Enum.
    """

    def should_summarize(self, request: MemoryRequest, history: ConversationHistory) -> SummaryDecision:
        """
        Public API: Evaluates the current state and decides if compression is required.
        """
        
     
        if request.force_refresh_summary:
            return SummaryDecision(
                should_summarize=True,
                reason="Summarization explicitly requested via MemoryRequest override.",
                triggered_by=SummaryTrigger.FORCE_REFRESH
            )

  
        if self._is_history_too_small(history):
            return SummaryDecision(
                should_summarize=False,
                reason=f"History has fewer than {MIN_MESSAGES_FOR_SUMMARY} messages.",
                triggered_by=SummaryTrigger.INSUFFICIENT_MESSAGES
            )

        
        if self._exceeded_token_limit(history):
            return SummaryDecision(
                should_summarize=True,
                reason=f"History tokens ({history.total_tokens}) breached the {SUMMARY_TRIGGER_TOKENS} threshold.",
                triggered_by=SummaryTrigger.TOKEN_THRESHOLD
            )

        
        if self._exceeded_message_limit(history):
            return SummaryDecision(
                should_summarize=True,
                reason=f"History length ({history.total_messages}) breached the {SUMMARY_TRIGGER_MESSAGES} threshold.",
                triggered_by=SummaryTrigger.MESSAGE_THRESHOLD
            )

        
        return SummaryDecision(
            should_summarize=False,
            reason="All thresholds are within safe operational limits.",
            triggered_by=SummaryTrigger.WITHIN_LIMITS
        )



    def _is_history_too_small(self, history: ConversationHistory) -> bool:
        """Prevents the LLM from summarizing single-turn or empty conversations."""
        return history.total_messages < MIN_MESSAGES_FOR_SUMMARY

    def _exceeded_token_limit(self, history: ConversationHistory) -> bool:
        """Evaluates if the raw text is getting too expensive/large for the context window."""
        return history.total_tokens >= SUMMARY_TRIGGER_TOKENS

    def _exceeded_message_limit(self, history: ConversationHistory) -> bool:
        """Evaluates if the sheer volume of message objects exceeds UI or payload limits."""
        return history.total_messages >= SUMMARY_TRIGGER_MESSAGES