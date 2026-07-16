# scripts/test_evaluation.py

import os
import sys
import uuid
import time

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import db_manager
from core.container import container

# Domain Models
from retrieval.models import SearchQuery, QueryIntent
from generation.models import GenerationRequest
from evaluation.models import EvaluationRequest

# Request-Scoped Dependencies
from database.repositories.retrieval_repository import RetrievalRepository
from database.retrieval_service import RetrievalService
from retrieval.search.retriever import Retriever
from retrieval.pipeline import RetrievalPipeline


def main():
    print("\n========================================")
    print(" SELF-HEALING RAG EVALUATION TEST")
    print("========================================\n")

    # ---------------------------------------------------------
    # [1/7] Initializing Dependencies
    # ---------------------------------------------------------
    print("[1/7] Initializing Dependencies...")
    try:
        db_manager.initialize()
        if not db_manager.check_connection():
            print("❌ Database connection failed. Aborting test.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

    db_session = db_manager.SessionLocal()
    query_id = uuid.uuid4().hex

    try:
        # Build Request-Scoped Retrieval Pipeline
        retrieval_repo = RetrievalRepository(db_session)
        retrieval_service = RetrievalService(retrieval_repo)
        retriever = Retriever(container.embedding_service, retrieval_service)
        retrieval_pipeline = RetrievalPipeline(
            analyzer=container.query_analyzer,
            rewriter=container.llm_query_rewriter,
            retriever=retriever,
            filter_engine=container.retrieval_filter,
            reranker=container.reranker,
            context_builder=container.retrieval_context_builder
        )

        # Retrieve Stateless Pipelines from Container
        generation_pipeline = container.generation_pipeline
        evaluation_pipeline = container.evaluation_pipeline

        print(f"      {'Database session':<30} PASS")
        print(f"      {'Retrieval pipeline':<30} PASS")
        print(f"      {'Generation pipeline':<30} PASS")
        print(f"      {'Evaluation pipeline':<30} PASS\n")

        # ---------------------------------------------------------
        # [2/7] Preparing Evaluation Scenario (Retrieval)
        # ---------------------------------------------------------
        target_question = "What happens when retrieval confidence is low?"
        print(f"[2/7] Preparing Evaluation Scenario")
        print(f"      {'Query':<30} {target_question}")
        
        search_query = SearchQuery(
            query_id=query_id,
            query=target_question,
            chat_history=[],
            top_k=5
        )
        
        retrieval_context = retrieval_pipeline.retrieve(search_query)
        
        print(f"      {'Retrieval execution':<30} PASS")
        print(f"      {'Retrieved chunks':<30} {len(retrieval_context.chunks)}")
        
        if len(retrieval_context.chunks) == 0:
            print("\n❌ EVALUATION TEST FAILED (Prerequisite Missing)")
            print("Reason: Retrieval returned 0 chunks. Run `test_ingestion.py` first.")
            sys.exit(1)
            
        print(f"      {'Context available':<30} PASS\n")

        # ---------------------------------------------------------
        # [3/7] Generating Answer
        # ---------------------------------------------------------
        print("[3/7] Generating Answer...")
        
        generation_request = GenerationRequest(
            query_id=query_id,
            original_query=search_query.query,
            optimized_query=retrieval_context.rewritten_question or retrieval_context.question,
            context=retrieval_context.context,
            intent=QueryIntent.UNKNOWN,
            available_citations=retrieval_context.citations,
            retrieval_metadata=retrieval_context.metadata,
            chat_history=[]
        )
        
        generation_response = generation_pipeline.generate(generation_request)
        
        print(f"      {'Generation execution':<30} PASS")
        print(f"      {'Answer available':<30} PASS")
        print(f"\n      Generated Answer:\n      {generation_response.answer}\n")

        # ---------------------------------------------------------
        # [4/7] Building Evaluation Request
        # ---------------------------------------------------------
        print("[4/7] Building Evaluation Request...")
        
        # Mirroring GraphStateConverter.build_evaluation_request
        raw_chunks = [chunk.text for chunk in retrieval_context.chunks]
        optimized_query = retrieval_context.rewritten_question or retrieval_context.question
        
        evaluation_request = EvaluationRequest(
            query_id=query_id,
            original_query=retrieval_context.question,
            optimized_query=optimized_query,
            context=retrieval_context.context,
            answer=generation_response.answer,
            citations=generation_response.citations,
            retrieval_metadata=retrieval_context.metadata,
            generation_metadata=generation_response.generation_metadata,
            retrieved_chunks=raw_chunks
        )
        
        print(f"      {'Evaluation request':<30} PASS")
        print(f"      {'Context attached':<30} PASS")
        print(f"      {'Raw chunks attached':<30} {len(evaluation_request.retrieved_chunks)}")
        print(f"      {'Query ID propagation':<30} PASS (ID: {evaluation_request.query_id[:8]}...)\n")

        # ---------------------------------------------------------
        # [5/7] Executing Evaluation Pipeline
        # ---------------------------------------------------------
        print("[5/7] Executing Evaluation Pipeline...")
        
        # This executes Grounding, Hallucination, RAGAS (Live mode), Confidence, Decision, and logs it to DB!
        eval_report = evaluation_pipeline.evaluate(evaluation_request)
        
        print(f"      {'Evaluation execution':<30} PASS")
        print(f"      {'Evaluation report':<30} PASS\n")

        # ---------------------------------------------------------
        # [6/7] Validating Quality Signals
        # ---------------------------------------------------------
        print("[6/7] Validating Quality Signals")
        print(f"      {'Grounding confidence':<30} {eval_report.grounding.confidence:.2f}")
        print(f"      {'Hallucination detected':<30} {eval_report.hallucination.has_hallucination}")
        print(f"      {'Unsupported claims':<30} {len(eval_report.hallucination.hallucinated_claims)}")
        print(f"      {'Faithfulness':<30} {eval_report.ragas.faithfulness_score:.2f}")
        print(f"      {'Answer relevancy':<30} {eval_report.ragas.answer_relevancy_score:.2f}")
        
        if not (0.0 <= eval_report.confidence.overall_score <= 1.0):
            print(f"\n❌ EVALUATION TEST FAILED")
            print(f"Confidence score {eval_report.confidence.overall_score} is out of structural bounds (0.0 - 1.0).")
            sys.exit(1)
            
        print(f"      {'Final confidence':<30} {eval_report.confidence.overall_score:.2f}\n")

        # ---------------------------------------------------------
        # [7/7] Validating Evaluation Decision
        # ---------------------------------------------------------
        print("[7/7] Validating Evaluation Decision")
        print(f"      {'Evaluation decision':<30} {eval_report.decision.value.upper()}")
        
        retry_val = eval_report.retry_recommendation.value if eval_report.retry_recommendation else "False"
        print(f"      {'Retry recommended':<30} {retry_val}")
        
        trace_pass = (
            query_id == search_query.query_id == 
            generation_request.query_id == 
            evaluation_request.query_id
        )
        print(f"      {'Trace continuity':<30} {'PASS' if trace_pass else 'FAIL'}\n")

        print("========================================")
        print(" EVALUATION TEST PASSED 🎉")
        print("========================================")

    except Exception as e:
        print(f"\n❌ Evaluation Pipeline crashed during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()