# graph/nodes/response.py

import logging
from graph.state import GraphState
from graph.models import NodeType, ResponseStatus 
from graph.utils.tracing import GraphTracer

logger = logging.getLogger(__name__)

class ResponseNode:
    """
    Pure orchestration node. Assembles the final response folder for the internal GraphState.
    Leaves public API mapping strictly to the FastAPI routers.
    """

    def __call__(self, state: GraphState) -> dict:
        """
        Executes the response assembly logic.
        Returns a dictionary update for LangGraph to merge strictly into the response folder.
        """
        # 1. Start the Black Box Recorder
        with GraphTracer.trace_node(state.execution, NodeType.RESPONSE):
            
            # Extract only the relevant folders
            execution = state.execution
            generation = state.generation
            evaluation = state.evaluation
            recovery = state.recovery


            status = ResponseStatus.FAILED
            answer = execution.termination_reason or "An unexpected orchestration error occurred."
            citations = []
            confidence = 0.0
            recovery_used = execution.retry_count > 0


            if generation and generation.answer:
                status = ResponseStatus.SUCCESS
                answer = generation.answer
                citations = getattr(generation, "citations", [])
                
                if evaluation and getattr(evaluation, "confidence", None):
                    # Safely handle potential attribute names
                    conf_obj = evaluation.confidence
                    confidence = getattr(conf_obj, "overall_confidence", getattr(conf_obj, "overall_score", 0.0))


            elif recovery and getattr(recovery, "user_message", None):
                status = ResponseStatus.PARTIAL
                answer = recovery.user_message


            response_data = {
                "status": status,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "recovery_used": recovery_used
            }

            logger.info(
                f"Internal Response Assembled | QueryID={execution.query_id} | "
                f"Status={status.name} | Latency={execution.total_latency_ms}ms"
            )

        # Return ONLY the dictionary for LangGraph to merge into the state
        return {"response": response_data}