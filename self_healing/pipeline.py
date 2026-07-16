## self_healing/pipeline.py
import time
import logging
from typing import List, Optional

from self_healing.models import (
    SelfHealingRequest, 
    RecoveryPlan, 
    RecoveryDecision,
    RecoveryActionType,
    RecoveryAction,
    RecoveryContext,
    RecoveryStatus,
    RecoveryMetadata,
    KnowledgeGap
)

# Cross-Subsystem Dependencies
from retrieval.models import AnalyzedQuery, RankedChunk, SearchQuery  

# The Specialists (Dependencies)
from self_healing.validator.base import BaseRecoveryValidator
from self_healing.recovery.retry_manager import RetryManager
from self_healing.recovery.query_rewriter import RecoveryQueryRewriter
from self_healing.recovery.web_search import WebSearchService
from self_healing.recovery.context_merge import ContextMerger
from self_healing.knowledge.gap_detector import GapDetector
from self_healing.knowledge.knowledge_manager import KnowledgeManager
from self_healing.knowledge.ingestion_trigger import IngestionTrigger

logger = logging.getLogger(__name__)


class SelfHealingPipeline:
    """
    The Orchestrator of the Self-Healing Subsystem.
    Coordinates Diagnosis, Execution of specific fallback tools, and Learning.
    Returns a strict RecoveryPlan contract for the Master Orchestrator (LangGraph) 
    to decide the next global routing step.
    """

    def __init__(
        self,
        validator: BaseRecoveryValidator,
        retry_manager: RetryManager,
        query_rewriter: RecoveryQueryRewriter,
        web_search: WebSearchService,
        context_merger: ContextMerger,
        gap_detector: GapDetector,
        knowledge_manager: KnowledgeManager,
        ingestion_trigger: IngestionTrigger
    ):
        self.validator = validator
        self.retry_manager = retry_manager
        self.query_rewriter = query_rewriter
        self.web_search = web_search
        self.context_merger = context_merger
        self.gap_detector = gap_detector
        self.knowledge_manager = knowledge_manager
        self.ingestion_trigger = ingestion_trigger

    def heal(
        self, 
        request: SelfHealingRequest, 
        original_analysis: AnalyzedQuery,
        internal_chunks: List[RankedChunk]
    ) -> RecoveryPlan:
        """
        The single public entry point.
        """
        start_time = time.time()
        logger.info(f"SelfHealingPipeline starting | QueryID={request.query_id}")

        # 1. Isolate State Safely
        self.retry_manager.load_state(request.retry_state)
        
        recovery_context = RecoveryContext(
            original_query=original_analysis.original_query,
            internal_chunks=internal_chunks
        )

        # ==========================================
        # Stage 1: Diagnosis
        # ==========================================
        decision = self.validator.validate(request)
        
        if not decision.requires_recovery:
            logger.info("Diagnosis complete: No recovery required (PASS).")
            return self._build_plan(
                request, decision, recovery_context, 
                status=RecoveryStatus.SKIPPED, continue_pipeline=False,
                actions=[], latency_ms=(time.time() - start_time) * 1000
            )

        # ==========================================
        # Stage 2: Loop Prevention
        # ==========================================
        if not self.retry_manager.can_retry():
            decision.requires_recovery = False
            decision.suggested_actions = [RecoveryActionType.STOP]
            decision.reason += " (Forced stop: Max retries exceeded)"
            
        elif self.retry_manager.has_visited_action_sequence(decision.suggested_actions):
            decision.requires_recovery = False
            decision.suggested_actions = [RecoveryActionType.STOP]
            decision.reason += " (Forced stop: Infinite loop prevented)"

        # ==========================================
        # Stage 3: Execute Specific Fallbacks
        # ==========================================
        executed_actions: List[RecoveryAction] = []
        execution_order = 1
        
        # This tells LangGraph what to do next
        continue_pipeline = False 
        final_status = RecoveryStatus.SUCCESS 

        if decision.requires_recovery:
            for action_type in decision.suggested_actions:
                action_record = RecoveryAction(
                    action_type=action_type,
                    execution_order=execution_order,
                    description=f"Executing fallback action: {action_type.value}", 
                    completed=False,
                    success=False
                )
                
                try:
                    if action_type == RecoveryActionType.REWRITE_QUERY:
                        sq = SearchQuery(
                            query_id=request.query_id,
                            query=original_analysis.original_query,
                            chat_history=[],
                            top_k=5
                        )
                        new_query = self.query_rewriter.execute(
                            search_query=sq,
                            previous_analysis=original_analysis
                        )
                        if new_query:
                            recovery_context.rewritten_query = new_query.query
                            action_record.success = True
                            continue_pipeline = True # Route back to Retrieval

                    elif action_type == RecoveryActionType.WEB_SEARCH:
                        web_chunks = self.web_search.search(query=original_analysis.original_query)
                        if web_chunks:
                            recovery_context.web_chunks = web_chunks
                            action_record.success = True
                        else:
                            action_record.error_message = "Web search returned 0 results."
                        
                        # ---> THE FIX: ALWAYS CONTINUE AFTER WEB SEARCH <---
                        # If web chunks is empty, generation will just apologize, evaluation
                        # will fail, and the next self-healing loop will hit the RetryGuard limit.
                        continue_pipeline = True 

                    elif action_type == RecoveryActionType.MERGE_CONTEXT:
                        merged_text = self.context_merger.merge(recovery_context)
                        # We accept empty strings as a valid merged context
                        if merged_text is not None: 
                            recovery_context.merged_context = merged_text
                            action_record.success = True
                        continue_pipeline = True

                    elif action_type == RecoveryActionType.RETRY_RETRIEVAL:
                        # Self-Healing does not retrieve. It just flags the routing requirement.
                        action_record.success = True
                        continue_pipeline = True
                        
                    elif action_type == RecoveryActionType.ASK_CLARIFICATION:
                        action_record.success = True
                        continue_pipeline = False
                        final_status = RecoveryStatus.COMPLETED

                    elif action_type == RecoveryActionType.STOP:
                        action_record.success = True
                        continue_pipeline = False
                        final_status = RecoveryStatus.FAILED

                    action_record.completed = True

                except Exception as e:
                    logger.exception(f"Recovery Action {action_type.value} failed.")
                    action_record.error_message = str(e)
                    final_status = RecoveryStatus.PARTIAL_SUCCESS 

                executed_actions.append(action_record)
                execution_order += 1

        # ==========================================
        # Stage 4: Long-Term Learning
        # ==========================================
        detected_gap = None
        
        if any(a.action_type == RecoveryActionType.WEB_SEARCH and a.success for a in executed_actions):
            detected_gap = self.gap_detector.detect(
                recovery_context=recovery_context,
                retry_state=self.retry_manager.get_state(),
                original_analysis=original_analysis,
                query_id=request.query_id
            )
            
            if detected_gap:
                gap_action = RecoveryAction(
                    action_type=RecoveryActionType.LOG_KNOWLEDGE_GAP,
                    execution_order=execution_order,
                    description="Logging detected knowledge gap for future ingestion.", 
                    completed=True,
                    success=True
                )
                
                updated_gap = self.knowledge_manager.process_detected_gap(detected_gap)
                trigger_success = self.ingestion_trigger.evaluate_and_trigger(updated_gap)
                
                if getattr(trigger_success, 'success', trigger_success): 
                    self.knowledge_manager.mark_resolved(updated_gap.missing_topic)

                executed_actions.append(gap_action)

        # ==========================================
        # Stage 5: Finalize State & Build Plan
        # ==========================================
        if decision.requires_recovery:
            self.retry_manager.record_action_sequence(decision.suggested_actions)
            self.retry_manager.increment_retry(failure_reason=decision.reason)

        latency = (time.time() - start_time) * 1000

        return self._build_plan(
            request=request, 
            decision=decision, 
            context=recovery_context,
            status=final_status,
            continue_pipeline=continue_pipeline,
            actions=executed_actions,
            latency_ms=latency,
            detected_gap=detected_gap
        )

    def _build_plan(
        self, 
        request: SelfHealingRequest, 
        decision: RecoveryDecision, 
        context: RecoveryContext,
        status: RecoveryStatus,
        continue_pipeline: bool,
        actions: List[RecoveryAction],
        latency_ms: float,
        detected_gap: Optional[KnowledgeGap] = None
    ) -> RecoveryPlan:
        """Helper to construct the strict Pydantic contract."""
        
        metadata = RecoveryMetadata(
            latency_ms=latency_ms,
            actions_executed=len(actions),
            total_recovery_time_ms=latency_ms 
        )
        
        return RecoveryPlan(
            query_id=request.query_id,
            status=status,
            continue_pipeline=continue_pipeline,
            decision=decision,
            actions=actions,
            recovery_context=context,
            retry_state=self.retry_manager.get_state(),
            knowledge_gap=detected_gap,
            metadata=metadata
        )