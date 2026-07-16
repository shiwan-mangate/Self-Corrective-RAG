from graph.state import GraphState
from graph.models import NodeType
from graph.utils.tracing import GraphTracer
from graph.utils.converters import GraphStateConverter
from self_healing.pipeline import SelfHealingPipeline

# ---> ADDED IMPORTS for the missing data models <---
from retrieval.models import AnalyzedQuery, QueryIntent, SearchType

class SelfHealingNode:
    """
    The orchestrator wrapper for the Self-Healing Subsystem.
    Acts as the 'Doctor' of the AI system, prescribing a RecoveryPlan based on the EvaluationReport.
    Follows the strict lifecycle: Trace -> Convert -> Execute -> Update State.
    Makes NO routing decisions; merely prepares the plan for the router.
    """
    
    def __init__(self, self_healing_pipeline: SelfHealingPipeline):
        # Dependency Injection: Pipeline is provided at graph-build time
        self.self_healing_pipeline = self_healing_pipeline

    def __call__(self, state: GraphState) -> dict:
        """
        Executes the self-healing logic. 
        Returns a dictionary update for LangGraph to merge into the master state.
        """
        # 1. Start the Black Box Recorder
        with GraphTracer.trace_node(state.execution, NodeType.SELF_HEALING):
            
            # 2. Convert GraphState into the strict Request
            request = GraphStateConverter.build_self_healing_request(state)
            
            # ---> FIX: Extract the missing positional arguments from GraphState <---
            retrieval_state = state.retrieval
            internal_chunks = retrieval_state.chunks if retrieval_state else []
            
            # Reconstruct the original analysis context from the retrieval state
            original_query = retrieval_state.question if retrieval_state else state.request.query
            intent = getattr(retrieval_state, "intent", QueryIntent.UNKNOWN) if retrieval_state else QueryIntent.UNKNOWN
            
            original_analysis = AnalyzedQuery(
                original_query=original_query,
                normalized_query=original_query.lower(),
                intent=intent,
                needs_history=False,
                needs_rewrite=False,
                search_type=SearchType.SIMILARITY,
                top_k=5
            )
            
            # 3. Call the Subsystem Pipeline with ALL required arguments
            response = self.self_healing_pipeline.heal(
                request=request,
                original_analysis=original_analysis,
                internal_chunks=internal_chunks
            )
            
        # 4. Return the specific folder update to LangGraph
        return {"recovery": response}