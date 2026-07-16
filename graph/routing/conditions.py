# graph/routing/conditions.py

import logging
from graph.state import GraphState
from graph.models import ExecutionStrategy

logger = logging.getLogger(__name__)

class RoutingConditions:
    """
    Read-only inspectors for LangGraph conditional edges.
    Acts as the single source of truth for the Router.
    """

    @staticmethod
    def get_next_strategy(state: GraphState) -> ExecutionStrategy:
        """
        Determines the next ExecutionStrategy by reading ONLY the finalized ExecutionState.
        Logic: 
        1. If retry is disallowed, always TERMINATE.
        2. Otherwise, follow the strategy pre-calculated by the RetryGuard.
        """
        execution = state.execution

        # 1. Circuit Breaker Override
        if not execution.retry_allowed:
            logger.debug(f"Condition | QueryID={execution.query_id} | Circuit Breaker Active -> TERMINATE")
            return ExecutionStrategy.TERMINATE

        # 2. Return the pre-synchronized strategy
        logger.debug(
            f"Condition | QueryID={execution.query_id} | "
            f"Strategy -> {execution.execution_strategy.value.upper()}"
        )
        return execution.execution_strategy