# graph/nodes/retrieval.py

from graph.state import GraphState
from graph.models import NodeType
from graph.utils.tracing import GraphTracer
from graph.utils.converters import GraphStateConverter
from retrieval.pipeline import RetrievalPipeline

class RetrievalNode:
    """
    The orchestrator wrapper for the Retrieval Subsystem.
    Follows the strict lifecycle: Trace -> Convert -> Execute -> Update State.
    Unaware of business logic, database implementations, or query analysis strategies.
    """
    
    def __init__(self, retrieval_pipeline: RetrievalPipeline):
       
        self.retrieval_pipeline = retrieval_pipeline

    def __call__(self, state: GraphState) -> dict:
        """
        Executes the retrieval logic. 
        Returns a dictionary update for LangGraph to merge into the master state.
        """
       
        with GraphTracer.trace_node(state.execution, NodeType.RETRIEVAL):
            

            search_request = GraphStateConverter.build_search_request(state)
            
           
            retrieval_response = self.retrieval_pipeline.retrieve(search_request)
            
        return {"retrieval": retrieval_response}