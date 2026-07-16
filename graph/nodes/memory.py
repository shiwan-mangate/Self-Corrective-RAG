# graph/nodes/memory.py

from graph.state import GraphState
from graph.models import NodeType
from graph.utils.tracing import GraphTracer
from graph.utils.converters import GraphStateConverter
from memory.pipeline import MemoryPipeline

class MemoryNode:
    """
    The orchestrator wrapper for the Memory Subsystem.
    Follows the strict lifecycle: Trace -> Convert -> Execute -> Update State.
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline):
        
        self.memory_pipeline = memory_pipeline

    def __call__(self, state: GraphState) -> dict:
        """
        Executes the node logic. 
        Returns a dictionary of updates for LangGraph to merge into the master state.
        """
        with GraphTracer.trace_node(state.execution, NodeType.MEMORY):

            memory_request = GraphStateConverter.build_memory_request(state)

            memory_response = self.memory_pipeline.build_context(memory_request)

        return {"memory": memory_response}