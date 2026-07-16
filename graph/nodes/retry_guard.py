import logging
from graph.state import GraphState
from graph.models import NodeType, ExecutionStrategy
from graph.utils.tracing import GraphTracer

logger = logging.getLogger(__name__)

class RetryGuardNode:
    """
    Pure orchestration node. Contains NO AI pipelines and NO converters.
    Acts as the circuit breaker for the LangGraph workflow to prevent infinite recovery loops.
    Reads the immutable RecoveryPlan and updates the mutable ExecutionState.
    """
    
    def __call__(self, state: GraphState) -> dict:
        """
        Executes the retry policy logic.
        Returns a dictionary update for LangGraph to merge strictly into the execution folder.
        """
      
        with GraphTracer.trace_node(state.execution, NodeType.RETRY_GUARD):
            
            
            execution = state.execution
            recovery = state.recovery
            
          
            if not recovery or not recovery.retry_state:
                execution.retry_allowed = False
                execution.termination_reason = "Missing RecoveryPlan or RetryState."
                execution.execution_strategy = ExecutionStrategy.TERMINATE
                return {"execution": execution}

           
            retry_state = recovery.retry_state

           
            execution.retry_count = retry_state.retry_count
            execution.max_retries = retry_state.max_retries  

          
            retry_allowed = retry_state.retry_count < retry_state.max_retries
            execution.retry_allowed = retry_allowed
            
            if not retry_allowed:
                execution.termination_reason = "Maximum retries exceeded."
                execution.execution_strategy = ExecutionStrategy.TERMINATE
                logger.warning(
                    f"RetryGuard Triggered | QueryID={execution.query_id} | "
                    f"Max Retries ({execution.max_retries}) exhausted. Forcing termination."
                )
            else:
                
                if recovery.continue_pipeline:
                   
                    action_types = [a.action_type.value for a in recovery.actions]
                    
                    if "rewrite_query" in action_types or "retry_retrieval" in action_types:
                        execution.execution_strategy = ExecutionStrategy.RESTART_RETRIEVAL
                    elif "web_search" in action_types or "merge_context" in action_types:
                        execution.execution_strategy = ExecutionStrategy.RESTART_GENERATION
                    else:
                       
                        execution.execution_strategy = ExecutionStrategy.RESTART_RETRIEVAL
                else:
                    execution.execution_strategy = ExecutionStrategy.RETURN_RESPONSE

                logger.debug(
                    f"RetryGuard Passed | QueryID={execution.query_id} | "
                    f"Attempt {execution.retry_count}/{execution.max_retries}. "
                    f"Routing Strategy: {execution.execution_strategy.name}"
                )
                

        return {"execution": execution}