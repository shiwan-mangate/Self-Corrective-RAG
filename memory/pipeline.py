## memory/pipeline.py
import time
import logging

from memory.models import (
    MemoryRequest,
    SaveConversationRequest,
    MemoryResponse,
    MemoryMetadata,
    SummaryGenerationRequest 
)
from memory.constants import MEMORY_CONTEXT_TOKEN_BUDGET
from memory.constants import MEMORY_CONTEXT_TOKEN_BUDGET, MAX_RECENT_MESSAGES
from memory.session.manager import SessionManager
from memory.conversation.manager import ConversationManager
from memory.summarization.policy import SummaryPolicy
from memory.summarization.summarizer import ConversationSummarizer 
from memory.builder.context_builder import MemoryContextBuilder

logger = logging.getLogger(__name__)


class MemoryPipeline:
    """
    The Orchestrator for Layer 4 (Memory).
    Strictly coordinates dependencies. Contains NO business logic, 
    token math, text formatting, or database transactions.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        conversation_manager: ConversationManager,
        summary_policy: SummaryPolicy,
        summarizer: ConversationSummarizer, 
        context_builder: MemoryContextBuilder
    ):
        self.session_manager = session_manager
        self.conversation_manager = conversation_manager
        self.summary_policy = summary_policy
        self.summarizer = summarizer
        self.context_builder = context_builder

    def build_context(self, request: MemoryRequest) -> MemoryResponse:
        start_time = time.perf_counter()
        
        logger.info(
            f"Building Memory Context | QueryID={request.query_id} | "
            f"SessionID={request.session_id} | ConversationID={request.conversation_id}"
        )

       
        session = self.session_manager.load_or_create(
            session_id=request.session_id,
            user_id=request.user_id,
            query_id=request.query_id
        )

        
        active_history, current_summary = self.conversation_manager.load_context(session.session_id)

        summary_generated_this_turn = False
        initial_message_count = active_history.total_messages

    
        policy_decision = self.summary_policy.should_summarize(request, active_history)
        
       
        if policy_decision.should_summarize:
            logger.info(
                f"Memory threshold breached | QueryID={request.query_id} | "
                f"SessionID={session.session_id}. Triggering summarization."
            )
            messages_to_compress = self.conversation_manager.history_service.get_messages_to_summarize(
                history=active_history,
                keep_recent_n=MAX_RECENT_MESSAGES
            )
           
            summarize_req = SummaryGenerationRequest(
                query_id=request.query_id,
                session_id=session.session_id,
                messages_to_summarize=messages_to_compress,
                previous_summary=current_summary,
                force_refresh=request.force_refresh_summary
            )
            summary_result = self.summarizer.summarize(summarize_req)
            
            
            if summary_result.success:
                active_history = self.conversation_manager.commit_summary(
                    session_id=session.session_id,
                    new_summary=summary_result.summary,
                    active_history=active_history,
                    query_id=request.query_id
                )
                current_summary = summary_result.summary
                summary_generated_this_turn = True

      
        context = self.context_builder.build(
            session=session,
            history=active_history,
            summary=current_summary,
            preferences=None 
        )

       
        metadata = MemoryMetadata(
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            messages_loaded=initial_message_count,
            history_messages_used=active_history.total_messages,
            summary_used=current_summary is not None,
            summary_generated=summary_generated_this_turn,
            token_budget=MEMORY_CONTEXT_TOKEN_BUDGET
        )

        logger.info(
            f"Memory Context Built | QueryID={request.query_id} | "
            f"Latency={metadata.latency_ms}ms"
        )
        
        return MemoryResponse(context=context, metadata=metadata)

    def save_turn(self, request: SaveConversationRequest) -> None:
        """
        Called by LangGraph at the very end of the execution loop to persist the interaction.
        """
        logger.info(
            f"Saving conversation turn | QueryID={request.query_id} | "
            f"SessionID={request.session_id}"
        )
        
        
        save_result = self.conversation_manager.save_turn(
            session_id=request.session_id,
            user_message=request.user_message,
            assistant_message=request.assistant_message,
            query_id=request.query_id
        )
        
        
        self.session_manager.update_activity(
            session_id=request.session_id,
            message_delta=save_result.messages_saved,  
        )