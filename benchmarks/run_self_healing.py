import os
import json
import time
import uuid
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
# Core and Database Infrastructure
from core.container import ApplicationContainer
from database.connection import db_manager

# Domain Models & Enums
from graph.state import GraphState, ChatRequest
from evaluation.models import EvaluationDecision
from evaluation.ragas.metrics import RagasEvaluationMode
from graph.utils.converters import GraphStateConverter

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("Benchmark_SelfHealing")

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

QUESTIONS_FILE = DATASET_DIR / "questions.json"
GROUND_TRUTH_FILE = DATASET_DIR / "ground_truth.json"
OUTPUT_FILE = RESULTS_DIR / "self_healing.json"

def load_json(filepath: Path) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_benchmark():
    # 1. Ensure Results Directory Exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Datasets
    logger.info("Loading benchmark datasets...")
    questions = load_json(QUESTIONS_FILE)
    ground_truths = {item["id"]: item["ground_truth"] for item in load_json(GROUND_TRUTH_FILE)}

    # 3. Boot the Core Container & DB
    logger.info("Initializing Application Container & Database...")
    db_manager.initialize()
    container = ApplicationContainer()
    evaluation_pipeline = container.evaluation_pipeline
    
    benchmark_results = []

    logger.info(f"Starting Self-Healing Benchmark over {len(questions)} questions...")

    # 4. Execute the Benchmark Loop
    for q in questions:
        q_id = q["id"]
        question_text = q["question"]
        expected_answer = ground_truths.get(q_id, "")
        
        # Map conversation memory (Null means new session, else use the specified thread)
        session_id = q["conversation_id"] if q.get("conversation_id") else f"sh_bench_{uuid.uuid4().hex[:8]}"
        trace_id = f"trace_sh_{q_id}"
        
        logger.info(f"--- Processing Q{q_id}: {question_text} ---")

        # We request a fresh DB transaction scope per question
        with next(db_manager.get_session()) as db_session:
            # Create the full Production LangGraph Workflow
            workflow = container.create_workflow(db_session)
            
            # Formulate the Input exactly as the API would
            chat_req = ChatRequest(
                query=question_text,
                session_id=session_id,
                user_id="benchmark_runner",
                query_id=trace_id
            )
            
            initial_state = GraphState(request=chat_req)

            start_time = time.perf_counter()

            try:
                # --- A. EXECUTE PRODUCTION GRAPH ---
                # We invoke the graph directly rather than workflow.run() 
                # so we can intercept the COMPLETE final GraphState payload.
                final_state_dict = workflow.compiled_graph.invoke(initial_state.model_dump())
                final_state = GraphState(**final_state_dict)
                
                latency = round((time.perf_counter() - start_time), 2)

                # --- B. EXTRACT TELEMETRY & RECOVERY DATA ---
                execution = final_state.execution
                response = final_state.response
                retrieval = final_state.retrieval
                recovery = final_state.recovery
                
                retry_count = execution.retry_count
                recovery_used = retry_count > 0
                
                # Reconstruct the correction path from the RetryState history
                correction_path = []
                if recovery and recovery.retry_state and recovery.retry_state.visited_action_sequences:
                    for seq in recovery.retry_state.visited_action_sequences:
                        correction_path.extend(seq.split("->"))

                # --- C. OFFLINE BENCHMARK EVALUATION (RAGAS) ---
                # The graph evaluated itself in "LIVE" mode to self-heal. 
                # Now we evaluate its *final* output against the ground truth for "BENCHMARK" metrics.
                eval_report = None
                if retrieval and final_state.generation:
                    eval_req = GraphStateConverter.build_evaluation_request(final_state)
                    eval_report = evaluation_pipeline.evaluate(
                        request=eval_req,
                        mode=RagasEvaluationMode.BENCHMARK,
                        ground_truth=expected_answer
                    )

                # --- D. RECOVERY SUCCESS CALCULATION ---
                # Recovery Success = Self-Healing triggered AND the final evaluation passed
                recovery_success = False
                if recovery_used and eval_report:
                    recovery_success = (eval_report.decision == EvaluationDecision.PASS)

                # --- E. RECORD RESULT ---
                metrics = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "hallucination_detected": True
                }
                
                if eval_report:
                    metrics = {
                        "faithfulness": eval_report.ragas.faithfulness_score,
                        "answer_relevancy": eval_report.ragas.answer_relevancy_score,
                        "context_precision": eval_report.ragas.context_precision_score,
                        "context_recall": eval_report.ragas.context_recall_score,
                        "hallucination_detected": eval_report.hallucination.has_hallucination
                    }

                benchmark_results.append({
                    "id": q_id,
                    "category": q["category"],
                    "question": question_text,
                    "generated_answer": response.answer if response else "NO_RESPONSE",
                    "retrieved_context": retrieval.context if retrieval else "NO_CONTEXT",
                    "latency_sec": latency,
                    "recovery_used": recovery_used,
                    "correction_path": correction_path,
                    "retry_count": retry_count,
                    "recovery_success": recovery_success,
                    "metrics": metrics
                })
                
                logger.info(
                    f"Q{q_id} Complete | Latency: {latency}s | "
                    f"Recovery Used: {recovery_used} (Retries: {retry_count}) | "
                    f"Faithfulness: {metrics['faithfulness']}"
                )

            except Exception as e:
                logger.error(f"Failed on Q{q_id}: {str(e)}")
                benchmark_results.append({
                    "id": q_id,
                    "category": q["category"],
                    "question": question_text,
                    "error": str(e)
                })

    # 5. Dump to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)
        
    logger.info(f"Self-Healing Benchmark Complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()