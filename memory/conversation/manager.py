# memory/conversation/manager.py

import logging
from typing import Tuple, Optional

from memory.models import (
    ConversationHistory, 
    ConversationSummary, 
    ConversationMessage,
    PruningPlan,           # Contains: message_ids_to_remove (Set[str]), remaining_history (ConversationHistory)
    ConversationSaveResult # Contains: messages_saved (int), timestamps, message_ids, etc.
)
from memory.conversation.history import ConversationHistoryService
from memory.conversation.storage import BaseConversationStorage
from memory.constants import MAX_RECENT_MESSAGES

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    The Operations Manager for the Conversation lifecycle.
    
    Architecture Note:
    - PURE COORDINATOR: Contains no database logic, no LLM calls, and no algorithmic list-slicing.
    - Bridges the gap between pure persistence (Storage) and pure manipulation (HistoryService).
    """

    def __init__(
        self, 
        storage: BaseConversationStorage, 
        history_service: ConversationHistoryService
    ):
        self.storage = storage
        self.history_service = history_service

    def load_history(self, session_id: str, query_id: Optional[str] = None) -> ConversationHistory:
        """
        Loads the raw message list from the database and runs it through 
        the History Service to guarantee mathematically accurate token/message counts.
        """
        raw_history = self.storage.load_history(session_id, query_id)
        
        # Ensure the domain object is perfectly calculated before returning it to the pipeline
        return self.history_service.recalculate_totals(raw_history)

    def load_summary(self, session_id: str, query_id: Optional[str] = None) -> Optional[ConversationSummary]:
        """
        Retrieves the latest active summary for the session, if one exists.
        """
        return self.storage.load_summary(session_id, query_id)

    def load_context(
        self, session_id: str, query_id: Optional[str] = None
    ) -> Tuple[ConversationHistory, Optional[ConversationSummary]]:
        """
        Convenience API for the Pipeline to fetch the complete Memory state in one call.
        """
        history = self.load_history(session_id, query_id)
        summary = self.load_summary(session_id, query_id)
        return history, summary

    def commit_summary(
        self, 
        session_id: str, 
        new_summary: ConversationSummary, 
        active_history: ConversationHistory,
        query_id: Optional[str] = None,
        keep_recent_n: int = MAX_RECENT_MESSAGES
    ) -> ConversationHistory:
        """
        The critical state-transition method. 
        Safely commits the new summary, removes the summarized messages from the DB,
        and returns the freshly pruned, re-calculated history.
        """
        logger.info(f"Committing new summary for Session {session_id} | QueryID={query_id}")

        # 1. Ask the History Service for a complete pruning blueprint
        pruning_plan: PruningPlan = self.history_service.build_pruning_plan(
            history=active_history, 
            keep_recent_n=keep_recent_n
        )

        # 2. Transactionally persist the new state via Storage
        self.storage.commit_summary(
            session_id=session_id,
            summary=new_summary,
            message_ids_to_delete=pruning_plan.message_ids_to_remove,
            query_id=query_id
        )
        
        logger.debug(
            f"Pruned {len(pruning_plan.message_ids_to_remove)} messages. "
            f"Active history now contains {pruning_plan.remaining_history.total_messages} messages "
            f"| QueryID={query_id}"
        )

        # 3. Return the pre-calculated, immutable active history directly from the plan
        return pruning_plan.remaining_history

    def save_turn(
        self, 
        session_id: str, 
        user_message: ConversationMessage, 
        assistant_message: ConversationMessage,
        query_id: Optional[str] = None
    ) -> ConversationSaveResult:
        """
        Called by the Pipeline at the end of a successful generation cycle.
        Appends the structured messages to the database and returns a domain result.
        """
        logger.debug(f"Saving conversation turn for Session {session_id} | QueryID={query_id}")
        
        # The storage layer handles the SQL BEGIN/COMMIT transaction
        messages_saved = self.storage.save_turn(
            session_id=session_id, 
            user_message=user_message, 
            assistant_message=assistant_message,
            query_id=query_id
        )
        
        return ConversationSaveResult(
            session_id=session_id,
            query_id=query_id,
            messages_saved=messages_saved,
            user_message_id=user_message.message_id,
            assistant_message_id=assistant_message.message_id
        )

    def clear_history(self, session_id: str, query_id: Optional[str] = None) -> None:
        """
        Deletes all messages and the summary for a specific session.
        Useful for account resets, GDPR compliance, or testing workflows.
        """
        logger.info(f"Clearing entire conversation state for Session {session_id} | QueryID={query_id}")
        self.storage.clear_history(session_id=session_id, query_id=query_id)