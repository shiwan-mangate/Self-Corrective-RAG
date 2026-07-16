# graph/routing/router.py

import logging
from graph.models import ExecutionStrategy, NodeType

logger = logging.getLogger(__name__)

class GraphRouter:
    """
    Pure data-driven routing table.
    Translates the 'What' (ExecutionStrategy) into the 'Where' (NodeType).
    Contains ZERO state inspection, ZERO business logic, and ZERO framework dependencies.
    """

    # The Single Source of Truth for Workflow Paths
    ROUTE_TABLE = {
        ExecutionStrategy.RESTART_RETRIEVAL: NodeType.RETRIEVAL.value,
        ExecutionStrategy.RESTART_GENERATION: NodeType.GENERATION.value,
        ExecutionStrategy.RETURN_RESPONSE: NodeType.RESPONSE.value,
        
        # TERMINATE flows to RESPONSE to guarantee a user-facing apology
        # and ensure the failure trace is persisted to the database.
        ExecutionStrategy.TERMINATE: NodeType.RESPONSE.value,
    }

    @staticmethod
    def route(strategy: ExecutionStrategy) -> str:
        """
        Looks up the destination node for a given strategy.
        Safe fallback provided to prevent catastrophic graph deadlocks.
        """
        destination = GraphRouter.ROUTE_TABLE.get(strategy)

        if not destination:
            logger.error(f"Router | Unknown Strategy: '{strategy}'. Forcing fallback to RESPONSE.")
            return NodeType.RESPONSE.value

        logger.debug(f"Router | Strategy={strategy.name} -> Destination={destination}")
        return destination
    
    @classmethod
    def get_route_table(cls) -> dict:
        """Exposes the routing table securely. Allows future dynamic route generation."""
        return cls.ROUTE_TABLE