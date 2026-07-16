# graph/builders/graph_builder.py

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# State and Models
from graph.state import GraphState
from graph.models import NodeType

# Routing
from graph.routing.conditions import RoutingConditions
from graph.routing.router import GraphRouter

# Nodes
from graph.nodes.input_preparation import input_preparation_node
from graph.nodes.memory import MemoryNode
from graph.nodes.retrieval import RetrievalNode
from graph.nodes.generation import GenerationNode
from graph.nodes.evaluation import EvaluationNode
from graph.nodes.self_healing import SelfHealingNode
from graph.nodes.retry_guard import RetryGuardNode
from graph.nodes.response import ResponseNode
from graph.nodes.persist import PersistNode

# Subsystem Pipelines (Type Hints for DI)
from memory.pipeline import MemoryPipeline
from retrieval.pipeline import RetrievalPipeline
from generation.pipeline import GenerationPipeline
from evaluation.pipeline import EvaluationPipeline
from self_healing.pipeline import SelfHealingPipeline

logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    The master architect of the LangGraph workflow.
    Responsible ONLY for structural topology and dependency injection.
    Contains ZERO business logic, state mutation, or execution behavior.
    """

    def __init__(
        self,
        memory_pipeline: MemoryPipeline,
        retrieval_pipeline: RetrievalPipeline,
        generation_pipeline: GenerationPipeline,
        evaluation_pipeline: EvaluationPipeline,
        self_healing_pipeline: SelfHealingPipeline
    ):
        # Dependency Injection: Receive all initialized subsystems
        self.memory_pipeline = memory_pipeline
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_pipeline = generation_pipeline
        self.evaluation_pipeline = evaluation_pipeline
        self.self_healing_pipeline = self_healing_pipeline

    def build(self) -> CompiledStateGraph:
        """
        Assembles and compiles the isolated nodes into an executable AI workflow.
        """
        logger.info("Initializing LangGraph StateMachine blueprint...")
        builder = StateGraph(GraphState)

        # 1. Instantiate the nodes independently (highly testable)
        nodes = self._create_nodes()
        
        # 2. Register the topology
        self._register_nodes(builder, nodes)
        self._register_edges(builder)
        self._register_conditional_edges(builder)

        # 3. Compile the Workflow
        compiled_graph = builder.compile()
        logger.info("LangGraph workflow successfully compiled.")
        return compiled_graph

    def _create_nodes(self) -> dict:
        """
        Instantiates all execution nodes and injects their dependencies.
        Returns a dictionary mapping NodeType values to their callable nodes.
        """
        return {
            NodeType.INPUT_PREPARATION.value: input_preparation_node,
            NodeType.MEMORY.value: MemoryNode(self.memory_pipeline),
            NodeType.RETRIEVAL.value: RetrievalNode(self.retrieval_pipeline),
            NodeType.GENERATION.value: GenerationNode(self.generation_pipeline),
            NodeType.EVALUATION.value: EvaluationNode(self.evaluation_pipeline),
            NodeType.SELF_HEALING.value: SelfHealingNode(self.self_healing_pipeline),
            NodeType.RETRY_GUARD.value: RetryGuardNode(),
            NodeType.RESPONSE.value: ResponseNode(),
            NodeType.PERSIST.value: PersistNode(self.memory_pipeline)
        }

    def _register_nodes(self, builder: StateGraph, nodes: dict) -> None:
        """Registers the pre-instantiated nodes with the graph builder."""
        for node_name, node_instance in nodes.items():
            builder.add_node(node_name, node_instance)

    def _register_edges(self, builder: StateGraph) -> None:
        """Wires all predictable, linear paths (The Straight Line)."""
        
        # The On-Ramp
        builder.add_edge(START, NodeType.INPUT_PREPARATION.value)
        builder.add_edge(NodeType.INPUT_PREPARATION.value, NodeType.MEMORY.value)
        
        # The Core Linear AI Pipeline
        builder.add_edge(NodeType.MEMORY.value, NodeType.RETRIEVAL.value)
        builder.add_edge(NodeType.RETRIEVAL.value, NodeType.GENERATION.value)
        builder.add_edge(NodeType.GENERATION.value, NodeType.EVALUATION.value)
        builder.add_edge(NodeType.EVALUATION.value, NodeType.SELF_HEALING.value)
        builder.add_edge(NodeType.SELF_HEALING.value, NodeType.RETRY_GUARD.value)
        
        # The Final Exit Sequence (Symmetric DB Saving)
        builder.add_edge(NodeType.RESPONSE.value, NodeType.PERSIST.value)
        builder.add_edge(NodeType.PERSIST.value, END)

    def _register_conditional_edges(self, builder: StateGraph) -> None:
        """Wires the single decision fork (The Brain)."""
        
        builder.add_conditional_edges(
            source=NodeType.RETRY_GUARD.value,
            path=RoutingConditions.get_next_strategy,   # The "What happened?"
            path_map=GraphRouter.get_route_table()      # The decoupled "Where to go next?"
        )