import logging
from typing import Optional

from self_healing.recovery.retry_manager import RetryManager
from retrieval.models import SearchQuery, AnalyzedQuery
from retrieval.rewriter.base import BaseQueryRewriter

logger = logging.getLogger(__name__)


class RecoveryQueryRewriter:
    """
    Adapter between the Self-Healing Subsystem and the Retrieval Subsystem.
    """
    def __init__(
        self, 
        retrieval_rewriter: BaseQueryRewriter, 
        retry_manager: RetryManager
    ):
        self.retrieval_rewriter = retrieval_rewriter
        self.retry_manager = retry_manager

    def execute(
        self, 
        search_query: SearchQuery,
        previous_analysis: AnalyzedQuery
    ) -> Optional[SearchQuery]:
        original_text = search_query.query
        
        logger.info(f"RecoveryQueryRewriter invoked for: '{original_text}'")

        # FIX 1: Prevent double-incrementing. Just check if we can retry.
        if not self.retry_manager.can_retry():
            logger.warning("Rewrite aborted: Retry limit reached.")
            return None

        self.retry_manager.record_query(original_text)
        forced_analysis = previous_analysis.model_copy(update={"needs_rewrite": True})

        try:
            # FIX 2: Pass the REAL query_id to the Retrieval layer
            rewritten_result = self.retrieval_rewriter.rewrite(
                analyzed_query=forced_analysis,
                chat_history=search_query.chat_history,
                query_id=search_query.query_id  # <--- Threaded here!
            )
        except Exception as e:
            logger.error(f"Retrieval Query Rewriter failed during recovery: {e}")
            return None

        new_text = rewritten_result.rewritten_query or rewritten_result.normalized_query

        if new_text.lower().strip() == original_text.lower().strip():
            logger.info("Rewrite aborted: LLM returned the exact same query.")
            return None

        if self.retry_manager.has_visited_query(new_text):
            logger.info(f"Rewrite aborted: The new query '{new_text}' was already tried previously.")
            return None

        self.retry_manager.record_query(new_text)
        
        logger.info(
            f"Recovery Rewrite Successful [Attempt {self.retry_manager.get_state().retry_count}]: "
            f"'{original_text}' -> '{new_text}'"
        )

        return SearchQuery(
            query_id=search_query.query_id,
            query=new_text,
            chat_history=search_query.chat_history,
            top_k=search_query.top_k,
            filters=search_query.filters
        )