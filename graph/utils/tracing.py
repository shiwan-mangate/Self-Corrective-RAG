# graph/utils/tracing.py

import time
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from graph.state import ExecutionState
from graph.models import NodeExecution, NodeType, NodeStatus

logger = logging.getLogger(__name__)

class GraphTracer:
    """
    The 'Black Box Recorder' of the LangGraph workflow.
    Owns all execution bookkeeping, telemetry, and latency tracking.
    Never modifies business data; only updates ExecutionState.
    """

    @staticmethod
    @contextmanager
    def trace_node(state: ExecutionState, node_type: NodeType) -> Generator[NodeExecution, None, None]:
        """
        A context manager that automatically handles the entire lifecycle of a node execution.
        Records start times, calculates attempt numbers, catches exceptions, 
        and guarantees that the node is safely finalized in the trace.
        """
        start_perf = time.perf_counter()
        execution = GraphTracer._start_node(state, node_type)
        
        try:
            # Yield control to the node's business logic (Pipeline execution)
            yield execution
            
            # Finalize on Success
            GraphTracer._complete_node(state, execution, start_perf)
            
        except Exception as e:
            # Finalize on Failure
            GraphTracer._fail_node(state, execution, start_perf, e)
            
            # Re-raise the exception so the Graph Orchestrator (or Error Node) can handle it
            raise

    @staticmethod
    def _start_node(state: ExecutionState, node_type: NodeType) -> NodeExecution:
        """Internal helper to initialize a node execution."""
        
        # Calculate attempt number by inspecting the historical trace
        previous_attempts = sum(
            1 for exec_record in state.execution_trace.executions 
            if exec_record.node_type == node_type
        )
        current_attempt = previous_attempts + 1

        # Differentiate between first run and recovery runs
        status = NodeStatus.RUNNING if current_attempt == 1 else NodeStatus.RETRYING

        execution = NodeExecution(
            node_type=node_type,
            status=status,
            attempt=current_attempt,
            start_time=datetime.now(timezone.utc)
        )

        # Update global execution state using the Enum directly
        state.current_node = node_type
        state.execution_trace.executions.append(execution)
        state.visited_nodes.append(node_type.value)

        logger.debug(
            f"Trace | Node={node_type.value} | Status={status.value.upper()} | "
            f"Attempt={current_attempt} | QueryID={state.query_id}"
        )
        return execution

    @staticmethod
    def _complete_node(state: ExecutionState, execution: NodeExecution, start_perf: float) -> None:
        """Internal helper to mark a node as successfully finished."""
        execution.status = NodeStatus.COMPLETED
        execution.end_time = datetime.now(timezone.utc)
        execution.latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
        
        # O(1) accumulation
        state.total_latency_ms = round(state.total_latency_ms + execution.latency_ms, 2)
        
        logger.debug(
            f"Trace | Node={execution.node_type.value} | Status=COMPLETED | "
            f"Latency={execution.latency_ms}ms | QueryID={state.query_id}"
        )

    @staticmethod
    def _fail_node(state: ExecutionState, execution: NodeExecution, start_perf: float, error: Exception) -> None:
        """Internal helper to mark a node as failed."""
        execution.status = NodeStatus.FAILED
        execution.error_message = f"{type(error).__name__}: {str(error)}"
        execution.end_time = datetime.now(timezone.utc)
        execution.latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
        
        # O(1) accumulation
        state.total_latency_ms = round(state.total_latency_ms + execution.latency_ms, 2)
        
        logger.error(
            f"Trace | Node={execution.node_type.value} | Status=FAILED | "
            f"Error='{execution.error_message}' | QueryID={state.query_id}"
        )

    @staticmethod
    def mark_skipped(state: ExecutionState, node_type: NodeType, reason: str) -> None:
        """Records that a node was intentionally bypassed by the router."""
        execution = NodeExecution(
            node_type=node_type,
            status=NodeStatus.SKIPPED,
            attempt=0,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            latency_ms=0.0,
            metadata={"skip_reason": reason}
        )
        state.execution_trace.executions.append(execution)
        logger.debug(f"Trace | Node={node_type.value} | Status=SKIPPED | Reason='{reason}'")

    @staticmethod
    def finish_execution(state: ExecutionState) -> None:
        """Marks the global boundary for the entire request lifecycle."""
        # Note: We assume state.start_time is populated by Input Preparation
        end_time = datetime.now(timezone.utc)
        # Global latency could also be computed here if desired, 
        # but the incremental O(1) accumulation covers total_latency_ms.
        logger.info(
            f"Workflow Execution Complete | QueryID={state.query_id} | "
            f"Total Nodes={len(state.visited_nodes)} | Total Latency={state.total_latency_ms}ms"
        )