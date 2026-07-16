import logging
from typing import Optional, List

from .base import BaseRecoveryValidator
from self_healing.models import (
    SelfHealingRequest, 
    RecoveryDecision, 
    RecoveryActionType,
    RetryState
)
from self_healing.constants import (
    MAX_RECOVERY_RETRIES,
    STOP_ON_MAX_RETRIES,
    HALLUCINATION_FORCE_STRICT_GROUNDING,
    HALLUCINATION_MAX_RETRIES,
    ALLOW_WEB_SEARCH_AFTER_RETRY,
    ALLOW_QUERY_REWRITE,
    MIN_INTERNAL_SIMILARITY_FOR_WEB_SEARCH
)

logger = logging.getLogger(__name__)


class PolicyValidator(BaseRecoveryValidator):
    """
    The Autonomous Brain of the Self-Healing Subsystem.
    Reads raw Evaluation facts (not opinions) and maps them to a RecoveryDecision.
    """

    def validate(self, request: SelfHealingRequest) -> RecoveryDecision:
        report = request.evaluation_report
        state = request.retry_state

        logger.info(
            f"Recovery Validator Analyzing | QueryID={request.query_id} | "
            f"RetryCount={state.retry_count}/{MAX_RECOVERY_RETRIES}"
        )

        if report.decision.value == "pass":
            return self._create_decision(
                requires_recovery=False,
                reason="Evaluation passed. No recovery needed.",
                actions=[]
            )

        # Step 2: Have we exceeded our safety limits?
        limit_decision = self._check_retry_limits(state)
        if limit_decision:
            self._log_decision(request.query_id, limit_decision)
            return limit_decision

        # Step 3: Route based purely on Evaluation Facts
        
        # 3a. Hallucination takes ultimate priority (Fabrication is dangerous)
        if report.hallucination.has_hallucination:
            decision = self._handle_hallucination(state)
            self._log_decision(request.query_id, decision)
            return decision

        # 3b. Ungrounded (Factually okay, but not from our DB)
        if not report.grounding.is_grounded:
            
            # Extract raw signal from Evaluation metadata
            retrieval_sim = request.retrieval_metadata.statistics.average_similarity

            # ESCALATION CHECK 1: Did we already try rewriting the query?
            already_rewrote = any("rewrite_query" in seq for seq in state.visited_action_sequences)

            # ESCALATION CHECK 2: Did we already try web search?
            already_web_searched = state.web_search_used

            if already_web_searched:
                # We tried everything and it still failed (or Web Search found nothing).
                # The model is deeply confused or the query is impossible.
                decision = self._handle_clarification()
            elif retrieval_sim >= MIN_INTERNAL_SIMILARITY_FOR_WEB_SEARCH or already_rewrote:
                decision = self._handle_missing_knowledge()
            else:
                decision = self._handle_poor_retrieval()
                
            self._log_decision(request.query_id, decision)
            return decision

    # ==========================================
    # Factory Methods & Telemetry
    # ==========================================

    def _create_decision(
        self, 
        requires_recovery: bool, 
        reason: str, 
        actions: List[RecoveryActionType]
    ) -> RecoveryDecision:
        """Centralized factory for creating decisions. Makes future schema changes easy."""
        return RecoveryDecision(
            requires_recovery=requires_recovery,
            reason=reason,
            suggested_actions=actions
        )

    def _log_decision(self, query_id: str, decision: RecoveryDecision) -> None:
        """Centralized logging for traceability."""
        action_names = [a.value for a in decision.suggested_actions]
        logger.info(
            f"Recovery Decision Reached | QueryID={query_id} | "
            f"Recovery={decision.requires_recovery} | "
            f"Actions={action_names} | "
            f"Reason='{decision.reason}'"
        )

    # ==========================================
    # Discrete Policy Engines
    # ==========================================

    def _check_retry_limits(self, state: RetryState) -> Optional[RecoveryDecision]:
        if STOP_ON_MAX_RETRIES and state.retry_count >= MAX_RECOVERY_RETRIES:
            return self._create_decision(
                requires_recovery=True,
                reason=f"Max retries ({state.retry_count}/{MAX_RECOVERY_RETRIES}) exceeded. Terminating loop.",
                actions=[
                    RecoveryActionType.LOG_KNOWLEDGE_GAP, 
                    RecoveryActionType.ASK_CLARIFICATION, 
                    RecoveryActionType.STOP
                ]
            )
        return None

    def _handle_hallucination(self, state: RetryState) -> RecoveryDecision:
        if state.retry_count >= HALLUCINATION_MAX_RETRIES:
            return self._create_decision(
                requires_recovery=True,
                reason="Model repeatedly hallucinating. Terminating to save tokens.",
                actions=[RecoveryActionType.ASK_CLARIFICATION, RecoveryActionType.STOP]
            )
            
        actions = []
        if HALLUCINATION_FORCE_STRICT_GROUNDING:
            actions.append(RecoveryActionType.STRICT_GROUNDING)
        actions.append(RecoveryActionType.RETRY_RETRIEVAL)
        
        return self._create_decision(
            requires_recovery=True,
            reason="Active hallucination detected. Enforcing strict context boundaries.",
            actions=actions
        )

    def _handle_poor_retrieval(self) -> RecoveryDecision:
        if not ALLOW_QUERY_REWRITE:
            return self._abort_unsupported("Query rewriting is disabled by policy.")
            
        return self._create_decision(
            requires_recovery=True,
            reason="Poor internal retrieval match. Expanding query.",
            actions=[RecoveryActionType.REWRITE_QUERY, RecoveryActionType.RETRY_RETRIEVAL]
        )

    def _handle_missing_knowledge(self) -> RecoveryDecision:
        if not ALLOW_WEB_SEARCH_AFTER_RETRY:
            return self._abort_unsupported("Web search is disabled by policy.")
            
        return self._create_decision(
            requires_recovery=True,
            reason="Internal context lacks answers despite good vector match. Fetching external data.",
            actions=[
                RecoveryActionType.WEB_SEARCH, 
                RecoveryActionType.MERGE_CONTEXT,
                RecoveryActionType.LOG_KNOWLEDGE_GAP # Crucial addition for self-learning
            ]
        )

    def _handle_clarification(self) -> RecoveryDecision:
        return self._create_decision(
            requires_recovery=True,
            reason="System is completely uncertain. Prompting user for more details.",
            actions=[RecoveryActionType.ASK_CLARIFICATION, RecoveryActionType.STOP]
        )

    def _abort_unsupported(self, reason: str) -> RecoveryDecision:
        return self._create_decision(
            requires_recovery=True,
            reason=f"Recovery aborted: {reason}",
            actions=[RecoveryActionType.ASK_CLARIFICATION, RecoveryActionType.STOP]
        )