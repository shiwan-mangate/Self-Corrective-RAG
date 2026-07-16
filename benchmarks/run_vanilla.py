import os
import json
import time
import uuid
import logging
from pathlib import Path
import sys
# Core and Database Infrastructure (Required)
sys.path.append(str(Path(__file__).parent.parent))
# Core and Database Infrastructure
from core.container import ApplicationContainer
from database.connection import db_manager
from retrieval.models import SearchQuery, QueryIntent

# Domain Models
from memory.models import MemoryRequest, SaveConversationRequest, ConversationMessage, MessageRole
from retrieval.models import SearchQuery
from generation.models import GenerationRequest
from evaluation.models import EvaluationRequest
from evaluation.ragas.metrics import RagasEvaluationMode

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("Benchmark_Vanilla")

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

QUESTIONS_FILE = DATASET_DIR / "questions.json"
GROUND_TRUTH_FILE = DATASET_DIR / "ground_truth.json"
OUTPUT_FILE = RESULTS_DIR / "vanilla.json"

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
    
    # Grab stateless pipelines from the container
    generation_pipeline = container.generation_pipeline
    evaluation_pipeline = container.evaluation_pipeline
    
    # We will store the final results here
    benchmark_results = []

    # Get a fresh DB Session for the benchmark run
    with next(db_manager.get_session()) as db_session:
        
        # Instantiate request-scoped pipelines
        memory_pipeline = container.create_memory_pipeline(db_session)
        
        # To make Retrieval "Vanilla", we instantiate the pipeline with a Dummy Rewriter 
        # so it NEVER rewrites queries, proving the baseline limitations.
        from retrieval.rewriter.base import BaseQueryRewriter
        class DummyRewriter(BaseQueryRewriter):
            def rewrite(self, analyzed_query, chat_history, query_id=""):
                return analyzed_query
        
        from retrieval.search.retriever import Retriever
        from retrieval.pipeline import RetrievalPipeline
        from database.repositories.retrieval_repository import RetrievalRepository
        from database.retrieval_service import RetrievalService
        
        retriever = Retriever(container.embedding_service, RetrievalService(RetrievalRepository(db_session)))
        vanilla_retrieval_pipeline = RetrievalPipeline(
            analyzer=container.query_analyzer,
            rewriter=DummyRewriter(), # <--- VANILLA LOBOTOMY
            retriever=retriever,
            filter_engine=container.retrieval_filter,
            reranker=container.reranker,
            context_builder=container.retrieval_context_builder
        )

        logger.info(f"Starting Vanilla Benchmark over {len(questions)} questions...")

        # 4. Execute the Benchmark Loop
        for q in questions:
            q_id = q["id"]
            question_text = q["question"]
            expected_answer = ground_truths.get(q_id, "")
            
            # Map conversation memory (Null means new session, else use the specified thread)
            session_id = q["conversation_id"] if q.get("conversation_id") else f"vanilla_bench_{uuid.uuid4().hex[:8]}"
            trace_id = f"trace_v_{q_id}"
            
            logger.info(f"--- Processing Q{q_id}: {question_text} ---")
            start_time = time.perf_counter()

            try:
                # --- A. MEMORY PHASE ---
                mem_req = MemoryRequest(
                    query_id=trace_id,
                    session_id=session_id,
                    current_query=question_text
                )
                mem_res = memory_pipeline.build_context(mem_req)
                chat_history = [{"role": m.role.value, "content": m.content} for m in mem_res.context.active_history.messages]

                # --- B. VANILLA RETRIEVAL PHASE ---
                search_req = SearchQuery(
                    query_id=trace_id, 
                    query=question_text, 
                    chat_history=chat_history, 
                    top_k=5
                )
                retrieval_context = vanilla_retrieval_pipeline.retrieve(search_req)

                # --- C. GENERATION PHASE ---
                gen_req = GenerationRequest(
                    query_id=trace_id,
                    original_query=question_text,
                    optimized_query=question_text, # No rewrite
                    context=retrieval_context.context,
                    intent=QueryIntent.UNKNOWN, # <--- THE FIX
                    available_citations=retrieval_context.citations,
                    retrieval_metadata=retrieval_context.metadata,
                    chat_history=chat_history
                )
                gen_res = generation_pipeline.generate(gen_req)

                # --- D. METRICS EVALUATION PHASE (RAGAS) ---
                eval_req = EvaluationRequest(
                    query_id=trace_id,
                    original_query=question_text,
                    optimized_query=question_text,
                    context=retrieval_context.context,
                    answer=gen_res.answer,
                    citations=gen_res.citations,
                    retrieval_metadata=retrieval_context.metadata,
                    generation_metadata=gen_res.generation_metadata,
                    retrieved_chunks=[c.text for c in retrieval_context.chunks]
                )
                
                # Execute RAGAS in BENCHMARK mode to capture Context Precision/Recall
                eval_report = evaluation_pipeline.evaluate(
                    request=eval_req,
                    mode=RagasEvaluationMode.BENCHMARK,
                    ground_truth=expected_answer
                )

                # --- E. SAVE MEMORY FOR NEXT TURN ---
                user_msg = ConversationMessage(role=MessageRole.USER, content=question_text)
                asst_msg = ConversationMessage(role=MessageRole.ASSISTANT, content=gen_res.answer)
                memory_pipeline.save_turn(SaveConversationRequest(
                    query_id=trace_id, session_id=session_id, 
                    user_message=user_msg, assistant_message=asst_msg
                ))

                latency = round((time.perf_counter() - start_time), 2)

                # --- F. RECORD RESULT ---
                benchmark_results.append({
                    "id": q_id,
                    "category": q["category"],
                    "question": question_text,
                    "generated_answer": gen_res.answer,
                    "retrieved_context": retrieval_context.context,
                    "latency_sec": latency,
                    "metrics": {
                        "faithfulness": eval_report.ragas.faithfulness_score,
                        "answer_relevancy": eval_report.ragas.answer_relevancy_score,
                        "context_precision": eval_report.ragas.context_precision_score,
                        "context_recall": eval_report.ragas.context_recall_score,
                        "hallucination_detected": eval_report.hallucination.has_hallucination
                    }
                })
                logger.info(f"Q{q_id} Complete | Latency: {latency}s | Faithfulness: {eval_report.ragas.faithfulness_score}")

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
        
    logger.info(f"Vanilla Benchmark Complete! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark()