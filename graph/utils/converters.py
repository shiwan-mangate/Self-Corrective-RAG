# graph/utils/converters.py

from typing import List, Dict

from graph.state import GraphState
from memory.models import MemoryRequest,SaveConversationRequest
from retrieval.models import SearchQuery, QueryIntent
from generation.models import GenerationRequest
from evaluation.models import EvaluationRequest
from self_healing.models import SelfHealingRequest, RetryState

# Avoid magic numbers by importing subsystem defaults
from config.constants import DEFAULT_TOP_K 

# Future-proofing: Replace ValueError with custom graph exceptions later
# from shared.exceptions import MissingGraphStateError


class GraphStateConverter:
    """
    The universal translator for the Graph.
    Maps GraphState into the strict, isolated Request models expected by each pipeline.
    Contains NO business logic, NO LLM calls, and NO orchestration logic.
    """

    @staticmethod
    def build_memory_request(state: GraphState) -> MemoryRequest:
        """Translates GraphState into a MemoryRequest."""
        return MemoryRequest(
            query_id=state.execution.query_id,
            session_id=state.request.session_id,
            current_query=state.request.query,
            user_id=state.request.user_id,
            conversation_id=None,
            force_refresh_summary=False
        )

    @staticmethod
    def build_search_request(state: GraphState) -> SearchQuery:
        """
        Translates GraphState into a SearchQuery.
        Intelligently handles Self-Healing loops by checking for a rewritten query.
        """
        
        target_query = state.request.query
        if state.recovery and state.recovery.recovery_context.rewritten_query:
            target_query = state.recovery.recovery_context.rewritten_query

        
        chat_history: List[Dict[str, str]] = []
        if state.memory and state.memory.context and state.memory.context.active_history:
            for msg in state.memory.context.active_history.messages:
                chat_history.append({"role": msg.role.value, "content": msg.content})

        return SearchQuery(
            query_id=state.execution.query_id,
            query=target_query,
            chat_history=chat_history,
            top_k=DEFAULT_TOP_K
        )

    @staticmethod
    def build_generation_request(state: GraphState) -> GenerationRequest:
        """
        Translates GraphState into a GenerationRequest.
        Fuses Memory context, Retrieval context, and potential Web Search contexts.
        """
        if not state.retrieval:
            raise ValueError("Cannot convert to GenerationRequest: RetrievalContext is missing.")

       
        target_context = state.retrieval.context
        
        
        if state.recovery and state.recovery.recovery_context.merged_context:
            target_context = state.recovery.recovery_context.merged_context

        
        memory_string = ""
        chat_history: List[Dict[str, str]] = []
        
        if state.memory and state.memory.context:
            memory_string = state.memory.context.formatted_context_string
            for msg in state.memory.context.active_history.messages:
                chat_history.append({"role": msg.role.value, "content": msg.content})

        combined_context = f"{memory_string}\n\n{target_context}".strip()

      
        optimized_query = state.retrieval.rewritten_question or state.retrieval.question
        
        
        intent = getattr(state.retrieval, "intent", QueryIntent.UNKNOWN)

        return GenerationRequest(
            query_id=state.execution.query_id,
            original_query=state.retrieval.question,
            optimized_query=optimized_query,
            context=combined_context,
            intent=intent, 
            available_citations=state.retrieval.citations,
            retrieval_metadata=state.retrieval.metadata,
            chat_history=chat_history
        )

    @staticmethod
    def build_evaluation_request(state: GraphState) -> EvaluationRequest:
        """Translates GraphState into an EvaluationRequest."""
        if not state.retrieval or not state.generation:
            raise ValueError("Cannot convert to EvaluationRequest: Missing Retrieval or Generation state.")

        raw_chunks = [chunk.text for chunk in state.retrieval.chunks]
        optimized_query = state.retrieval.rewritten_question or state.retrieval.question

        return EvaluationRequest(
            query_id=state.execution.query_id,
            original_query=state.retrieval.question,
            optimized_query=optimized_query,
            context=state.retrieval.context,
            answer=state.generation.answer,
            citations=state.generation.citations,
            retrieval_metadata=state.retrieval.metadata,
            generation_metadata=state.generation.generation_metadata,
            retrieved_chunks=raw_chunks
        )

    @staticmethod
    def build_self_healing_request(state: GraphState) -> SelfHealingRequest:
        """Translates GraphState into a SelfHealingRequest."""
        if not state.evaluation:
            raise ValueError("Cannot convert to SelfHealingRequest: Missing EvaluationReport.")
        
        if not state.retrieval:
            raise ValueError("Cannot convert to SelfHealingRequest: Missing RetrievalContext.")

        if state.recovery and state.recovery.retry_state:
            retry_state = state.recovery.retry_state
        else:
            retry_state = RetryState()

        return SelfHealingRequest(
            query_id=state.execution.query_id,
            evaluation_report=state.evaluation,
            retry_state=retry_state,
            retrieval_metadata=state.retrieval.metadata  # <--- THE MISSING PIECE!
        )
    
    # Inside class GraphStateConverter in graph/utils/converters.py:


    @staticmethod
    def to_memory_save_request(state: 'GraphState') -> 'SaveConversationRequest':
        """Translates the finalized GraphState into the strict Memory persistence payload."""
        from memory.models import SaveConversationRequest, ConversationMessage, MessageRole
        
        # 1. Formulate the User Message
        user_msg = ConversationMessage(
            role=MessageRole.USER,
            content=state.request.query,
            metadata={"query_id": state.execution.query_id}
        )

        # 2. Formulate the Assistant Message (and pack telemetry into metadata!)
        is_clarification = (state.response.status.value == "partial") if state.response else False
        answer_text = state.response.answer if state.response else "An orchestration error occurred."

        assistant_msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=answer_text,
            metadata={
                "query_id": state.execution.query_id,
                "is_clarification": is_clarification,
                "latency_ms": state.execution.total_latency_ms,
                "recovery_used": state.response.recovery_used if state.response else False,
                "termination_reason": state.execution.termination_reason
            }
        )

        # 3. Return the original, DB-safe contract
        return SaveConversationRequest(
            query_id=state.execution.query_id,
            session_id=state.request.session_id,
            user_message=user_msg,
            assistant_message=assistant_msg
        )