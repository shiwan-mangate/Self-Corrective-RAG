# scripts/test_retrieval.py

import os
import sys
import uuid
import time

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import db_manager
from core.container import container
from retrieval.models import SearchQuery
from database.repositories.retrieval_repository import RetrievalRepository
from database.retrieval_service import RetrievalService
from retrieval.search.retriever import Retriever
from retrieval.pipeline import RetrievalPipeline

def main():
    print("\n========================================")
    print(" SELF-HEALING RAG RETRIEVAL TEST")
    print("========================================\n")

    # ---------------------------------------------------------
    # [1/5] Initialization
    # ---------------------------------------------------------
    print("[1/5] Initializing Database & Pipeline...")
    try:
        db_manager.initialize()
        if not db_manager.check_connection():
            print("❌ Database connection failed. Aborting test.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

    db_session = db_manager.SessionLocal()

    try:
        # Wire the request-scoped RetrievalPipeline (mirroring container.create_workflow)
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
        print(f"      {'Database session':<30} PASS")
        print(f"      {'Retrieval pipeline':<30} PASS\n")

        # ---------------------------------------------------------
        # [2/5] Building Search Query
        # ---------------------------------------------------------
        print("[2/5] Building Search Query...")
        
        # We deliberately ask about the exact edge-case ingested in test_ingestion.py
        target_question = "What happens when retrieval confidence is low?"
        query_id = uuid.uuid4().hex
        
        search_query = SearchQuery(
            query_id=query_id,
            query=target_question,
            chat_history=[],
            top_k=5
        )
        
        print(f"      {'Query':<30} {target_question}")
        print(f"      {'Query ID':<30} {query_id}")
        print(f"      {'Search request':<30} PASS\n")

        # ---------------------------------------------------------
        # [3/5] Executing Retrieval Pipeline
        # ---------------------------------------------------------
        print("[3/5] Executing Retrieval Pipeline...")
        start_time = time.time()
        
        # THE CORE EXECUTION BOUNDARY
        context = retrieval_pipeline.retrieve(search_query)
        
        elapsed = time.time() - start_time
        
        if not context:
            print("❌ Pipeline returned None instead of a RetrievalContext.")
            sys.exit(1)
            
        print(f"      {'Pipeline execution':<30} PASS ({elapsed:.2f}s)")
        print(f"      {'Retrieved chunks':<30} {len(context.chunks)}")
        
        if len(context.chunks) == 0:
            print("\n❌ RETRIEVAL TEST FAILED")
            print("Reason: No chunks were retrieved for the known integration-test query.")
            print("Did you run `test_ingestion.py` first?")
            sys.exit(1)
            
        if not context.context or len(context.context.strip()) == 0:
            print("\n❌ RETRIEVAL TEST FAILED")
            print("Reason: Chunks were found, but the assembled context string is empty.")
            sys.exit(1)
            
        print(f"      {'Context assembled':<30} PASS ({len(context.context)} chars)\n")

        # ---------------------------------------------------------
        # [4/5] Validating Retrieval Quality
        # ---------------------------------------------------------
        print("[4/5] Validating Retrieval Quality...")
        expected_phrase = "rewrite the query"
        
        # Check semantic success: Did it actually find the right information?
        found_expected_knowledge = any(
            expected_phrase.lower() in chunk.text.lower() 
            for chunk in context.chunks
        )
        
        print(f"      {'Expected topic':<30} '{expected_phrase}'")
        
        if found_expected_knowledge:
            print(f"      {'Relevant content found':<30} PASS")
        else:
            print(f"      {'Relevant content found':<30} FAIL")
            print("\n❌ RETRIEVAL TEST FAILED")
            print("Reason: Vectors were returned, but they did not contain the expected knowledge.")
            print("Top fetched text:")
            print(f"'{context.chunks[0].text}'")
            sys.exit(1)
            
        # Telemetry observation (Query Rewriting shouldn't strictly happen here, but we log it)
        print(f"      {'Query rewritten':<30} {'YES' if context.rewritten_question else 'NO'}")
        
        # Metadata validation
        if not context.metadata or context.metadata.statistics.total_chunks_retrieved == 0:
            print(f"      {'Retrieval metadata':<30} FAIL (Missing or 0 total chunks)")
            sys.exit(1)
            
        print(f"      {'Retrieval metadata':<30} PASS\n")

        # ---------------------------------------------------------
        # [5/5] Validating Sources
        # ---------------------------------------------------------
        print("[5/5] Validating Sources...")
        
        print(f"      {'Citations':<30} {len(context.citations)}")
        if len(context.citations) == 0:
            print("❌ RETRIEVAL TEST FAILED")
            print("Reason: Retrieval succeeded, but citation generation failed (0 citations).")
            sys.exit(1)
            
        print(f"      {'Citation generation':<30} PASS\n")
        
        print("========================================")
        print(" RETRIEVAL TEST PASSED 🎉")
        print("========================================")
        print(f"Top Source: {context.citations[0].source}")
        print(f"Top Score : {context.citations[0].score}")
        print("========================================")

    except Exception as e:
        print(f"\n❌ Retrieval Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always clean up the request-scoped database transaction
        db_session.close()

if __name__ == "__main__":
    main()