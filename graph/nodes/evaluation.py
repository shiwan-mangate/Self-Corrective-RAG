# graph/nodes/evaluation.py

from graph.state import GraphState
from graph.models import NodeType
from graph.utils.tracing import GraphTracer
from graph.utils.converters import GraphStateConverter
from evaluation.pipeline import EvaluationPipeline

class EvaluationNode:
    """
    The orchestrator wrapper for the Evaluation Subsystem.
    Acts as the quality gate for the AI system.
    Produces the EvaluationReport, but makes NO routing decisions.
    Follows the strict lifecycle: Trace -> Convert -> Execute -> Update State.
    """
    
    def __init__(self, evaluation_pipeline: EvaluationPipeline):
        # Dependency Injection: Pipeline is provided at graph-build time
        self.evaluation_pipeline = evaluation_pipeline

    def __call__(self, state: GraphState) -> dict:
        """
        Executes the evaluation logic. 
        Returns a dictionary update for LangGraph to merge into the master state.
        """
        # 1. Start the Black Box Recorder
        with GraphTracer.trace_node(state.execution, NodeType.EVALUATION):
            
            # 2. Convert GraphState into the strict Request
            # The converter safely packs the Generated Answer against the Retrieved Context
            request = GraphStateConverter.build_evaluation_request(state)
            
            # 3. Call the Evaluation Subsystem
            # The pipeline performs the complete evaluation workflow:
            # Grounding → Hallucination → RAGAS → Confidence → 
            # Decision → Retry Recommendation
            response = self.evaluation_pipeline.evaluate(request)
            
        # 4. Return the specific folder update to LangGraph
        return {"evaluation": response}