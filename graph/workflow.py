## graph/workflow.py
import uuid
import logging
import traceback
from langgraph.graph.state import CompiledStateGraph

from graph.state import GraphState, ChatRequest, ExecutionState

logger = logging.getLogger(__name__)

class GraphWorkflow:
    """
    The singular public entry point for the AI orchestration layer.
    Responsibilities:
    1. Receive API request.
    2. Initialize the blank GraphState whiteboard.
    3. Invoke the Compiled LangGraph.
    4. Return the internal GraphState to the router.
    5. Act as the ultimate exception boundary to protect the API layer.
    """

    def __init__(self, compiled_graph: CompiledStateGraph):
        self.compiled_graph = compiled_graph

    def run(self, request: ChatRequest) -> GraphState:
        """
        Executes a complete AI lifecycle for a given user query.
        """
        logger.info(f"Workflow Started | Session={request.session_id} | Query='{request.query[:50]}...'")
        
        initial_state = GraphState(
            request=request,
            execution=ExecutionState()
        )

        try:
           
            final_state_dict = self.compiled_graph.invoke(initial_state.model_dump())
            
        
            final_state = GraphState(**final_state_dict)

            logger.info(
                f"Workflow Finished | Session={request.session_id} | "
                f"QueryID={final_state.execution.query_id} | Latency={final_state.execution.total_latency_ms}ms"
            )
            
          
            return final_state

        except Exception as e:
            
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            fallback_query_id = request.query_id or f"crash-{uuid.uuid4().hex[:8]}"
            
            logger.critical(
                f"Workflow Crashed | Session={request.session_id} | "
                f"QueryID={fallback_query_id} | Error={error_msg}\n{stack_trace}"
            )

            
            raise RuntimeError(f"LangGraph execution crashed: {error_msg}") from e