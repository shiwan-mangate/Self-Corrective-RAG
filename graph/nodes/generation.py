# graph/nodes/generation.py

from graph.state import GraphState
from graph.models import NodeType
from graph.utils.tracing import GraphTracer
from graph.utils.converters import GraphStateConverter
from generation.pipeline import GenerationPipeline

class GenerationNode:
    """
    The orchestrator wrapper for the Generation Subsystem.
    Follows the strict lifecycle: Trace -> Convert -> Execute -> Update State.
    Unaware of prompt templates, citation extraction, or LLM mechanics.
    """
    
    def __init__(self, generation_pipeline: GenerationPipeline):
        # Dependency Injection: Pipeline is provided at graph-build time
        self.generation_pipeline = generation_pipeline

    def __call__(self, state: GraphState) -> dict:
        """
        Executes the generation logic. 
        Returns a dictionary update for LangGraph to merge into the master state.
        """
        # 1. Start the Black Box Recorder
        with GraphTracer.trace_node(state.execution, NodeType.GENERATION):
            
            # 2. Convert GraphState into the strict Request
            # The converter handles fusing Memory, Retrieval, and Web Search fallbacks
            request = GraphStateConverter.build_generation_request(state)
            
            # 3. Call the Subsystem Pipeline (Business Logic)
            response = self.generation_pipeline.generate(request)
            
        # 4. Return the specific folder update to LangGraph
        return {"generation": response}