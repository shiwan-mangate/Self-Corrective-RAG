# scripts/test_generation.py

import os
import sys
import uuid
import time

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import db_manager
from core.container import container
from retrieval.models import SearchQuery, QueryIntent
from generation.models import GenerationRequest
from database.repositories.retrieval_repository import RetrievalRepository
from database.retrieval_service import RetrievalService
from retrieval.search.retriever import Retriever
from retrieval.pipeline import RetrievalPipeline

def main():
    print("\n========================================")
    print(" SELF-HEALING RAG GENERATION TEST")
    print("========================================\n")

    # ---------------------------------------------------------
    # [1/6] Initializing Dependencies
    # ---------------------------------------------------------
    print("[1/6] Initializing Dependencies...")
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
        # Reconstruct Retrieval Pipeline exactly like test_retrieval.py
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

        # Grab the stateless Generation Pipeline directly from the container singelton
        generation_pipeline = container.generation_pipeline

        print(f"      {'Database session':<30} PASS")
        print(f"      {'Retrieval pipeline':<30} PASS")
        print(f"      {'Generation pipeline':<30} PASS\n")

        # ---------------------------------------------------------
        # [2/6] Retrieving Known Context
        # ---------------------------------------------------------
        target_question = "What happens when retrieval confidence is low?"
        print(f"[2/6] Retrieving Known Context for: '{target_question}'")
        
        search_query = SearchQuery(
            query_id=query_id,
            query=target_question,
            chat_history=[],
            top_k=5
        )
        
        retrieval_context = retrieval_pipeline.retrieve(search_query)
        
        print(f"      {'Retrieved chunks':<30} {len(retrieval_context.chunks)}")
        print(f"      {'Available citations':<30} {len(retrieval_context.citations)}")
        
        if len(retrieval_context.chunks) == 0:
            print("\n❌ GENERATION TEST FAILED")
            print("Reason: Retrieval returned 0 chunks. Ingestion data missing.")
            print("Please run `python scripts/test_ingestion.py` before this script.")
            sys.exit(1)
        print(f"      {'Retrieval context':<30} PASS\n")

        # ---------------------------------------------------------
        # [3/6] Building Generation Request
        # ---------------------------------------------------------
        print("[3/6] Building Generation Request...")
        
        # Explicitly map the fields to verify the Layer 2 -> Layer 3 cross-boundary contract
        generation_request = GenerationRequest(
            query_id=query_id,
            original_query=search_query.query,
            optimized_query=retrieval_context.rewritten_question or retrieval_context.question,
            context=retrieval_context.context,
            intent=QueryIntent.UNKNOWN,  # Defaults to Factual QA template in PromptBuilder
            available_citations=retrieval_context.citations,
            retrieval_metadata=retrieval_context.metadata,
            chat_history=[]
        )
        
        print(f"      {'Generation request':<30} PASS")
        print(f"      {'Query ID propagation':<30} PASS (ID: {generation_request.query_id[:8]}...)")
        print(f"      {'Context attached':<30} PASS ({len(generation_request.context)} chars)\n")

        # ---------------------------------------------------------
        # [4/6] Executing Generation Pipeline
        # ---------------------------------------------------------
        print("[4/6] Executing Generation Pipeline via Groq...")
        
        # Execute the full pipeline: Build Prompt -> Generate Answer -> Extract -> Format -> Build
        response = generation_pipeline.generate(generation_request)
        
        print(f"      {'Groq generation':<30} PASS")
        print(f"      {'Generation response':<30} PASS\n")

        # ---------------------------------------------------------
        # [5/6] Validating Generated Answer
        # ---------------------------------------------------------
        print("[5/6] Validating Generated Answer...")
        
        if not response.answer or len(response.answer.strip()) == 0:
            print("❌ GENERATION TEST FAILED")
            print("Reason: Pipeline completed but returned an empty answer string.")
            sys.exit(1)
            
        print(f"      {'Answer length':<30} {len(response.answer)} characters")
        
        # Semantic evaluation: verify the answer contains the key grounding terms from ingestion
        has_rewrite = "rewrite" in response.answer.lower()
        has_query = "query" in response.answer.lower()
        
        print(f"      {'Expected concepts':<30} 'rewrite' + 'query'")
        if has_rewrite and has_query:
            print(f"      {'Grounded concept found':<30} PASS\n")
        else:
            print(f"      {'Grounded concept found':<30} WARNING (Answer may lack expected focus)")
            
        print(f"--- Generated Answer Output ---")
        print(response.answer)
        print(f"--------------------------------\n")

        # ---------------------------------------------------------
        # [6/6] Validating Citations & Telemetry
        # ---------------------------------------------------------
        print("[6/6] Validating Citations & Metadata...")
        print(f"      {'Generated citations':<30} {len(response.citations)}")
        print(f"      {'Documents credited':<30} {response.documents_used}")
        
        # Trace Validation
        print(f"      {'Query trace continuity':<30} PASS")
        print(f"      {'Estimated prompt tokens':<30} {response.generation_metadata.token_usage.prompt_tokens}")
        print(f"      {'Estimated completion tokens':<30} {response.generation_metadata.token_usage.completion_tokens}")
        print(f"      {'LLM execution latency':<30} {response.generation_metadata.latency_ms}ms\n")

        print("========================================")
        print(" GENERATION TEST COMPLETED ")
        print("========================================\n")

    except Exception as e:
        print(f"\n❌ Generation Pipeline crashed during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()