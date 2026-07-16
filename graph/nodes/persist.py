# graph/nodes/persist.py

import logging
from graph.state import GraphState
from graph.models import NodeType
from graph.utils.tracing import GraphTracer
from graph.utils.converters import GraphStateConverter
from memory.pipeline import MemoryPipeline

logger = logging.getLogger(__name__)

class PersistNode:
    """
    The orchestrator wrapper for persistence.
    Saves the final conversation turn back into the Memory Subsystem.
    Provides perfect symmetry: Memory loads at START, Memory saves at END.
    Does NOT modify the GraphState.
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline):
        # Dependency Injection: Provided at graph-build time
        self.memory_pipeline = memory_pipeline

    def __call__(self, state: GraphState) -> dict:
        """
        Executes the persistence logic.
        Returns an EMPTY dictionary because the state is finalized and immutable at this point.
        """
        # 1. Start the Black Box Recorder
        with GraphTracer.trace_node(state.execution, NodeType.PERSIST):
            
            # Defensive check: Ensure we actually have a request and response to save
            if not state.request or not state.response:
                logger.warning(
                    f"PersistNode skipped: Missing request or response for QueryID={state.execution.query_id}"
                )
                return {}

            # 2. Convert GraphState into the strict SaveConversationRequest
            save_request = GraphStateConverter.to_memory_save_request(state)
            
            # 3. Call the Memory Subsystem
            try:
                self.memory_pipeline.save_turn(save_request)
                logger.info(
                    f"Conversation persisted | Session={save_request.session_id} | "
                    f"QueryID={save_request.query_id}"
                )
            except Exception as e:
                # We catch exceptions here because a database timeout or failure 
                # shouldn't crash the API response that has already been successfully 
                # formulated and is ready to be returned to the user.
                logger.error(f"Failed to persist conversation memory: {str(e)}")

        # 4. Return an empty update.
        # LangGraph will merge nothing, leaving the state exactly as it was.
        return {}