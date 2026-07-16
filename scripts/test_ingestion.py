# scripts/test_ingestion.py

import os
import sys
from sqlalchemy import text

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import db_manager
from core.container import container

# Define a deterministic test document that downstream scripts (like Retrieval) will rely on
TEST_FILE_NAME = "self_healing_rag_integration_test.txt"
TEST_DOCUMENT_CONTENT = """
The Self-Healing RAG system evaluates retrieval quality.
If retrieval confidence is low, the system can rewrite the query.
Web search via Tavily is used as a fallback for missing internal knowledge.
The Memory Subsystem automatically compresses conversations exceeding 4000 tokens.
NovaTech employees receive 24 paid leave days per year and may work remotely three days per week.
"""

def create_test_file() -> str:
    """Creates a temporary deterministic file for the loader to consume."""
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), TEST_FILE_NAME))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(TEST_DOCUMENT_CONTENT.strip())
    return file_path

def main():
    print("\n========================================")
    print(" SELF-HEALING RAG INGESTION TEST")
    print("========================================\n")

    # 1. Setup Data
    file_path = create_test_file()
    print(f"[1/4] Created test document: {TEST_FILE_NAME}")

    # 2. Database Initialization
    print("[2/4] Initializing Database & Container...")
    try:
        db_manager.initialize()
        if not db_manager.check_connection():
            print("❌ Database connection failed. Aborting test.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

    db_session = db_manager.SessionLocal()

    # 3. Pre-Flight Cleanup (Idempotency)
    # Delete chunks matching our unique test text so we start with a clean slate
    try:
        db_session.execute(
            text("""
                DELETE FROM document_chunks 
                WHERE text LIKE '%NovaTech employees receive 24 paid leave days%'
            """)
        )
        db_session.commit()
    except Exception as e:
        print(f"⚠️ Pre-flight cleanup failed or skipped: {e}")
        db_session.rollback()

    # 4. Pipeline Execution
    print("[3/4] Executing Ingestion Pipeline...")
    try:
        # Create the pipeline exactly as the API/Background worker would
        ingestion_pipeline = container.create_ingestion_pipeline(db_session)
        
        # Execute the full pipeline (Load -> Parse -> Clean -> Meta -> Chunk -> Embed -> Save)
        result = ingestion_pipeline.ingest(source=file_path)
        
        print(f"\n      {'Pipeline Status':<25} {'SUCCESS' if result.documents_processed > 0 else 'FAILED'}")
        print(f"      {'Documents Processed':<25} {result.documents_processed}")
        print(f"      {'Chunks Generated':<25} {result.chunks_generated}")
        print(f"      {'Chunks Persisted':<25} {result.chunks_persisted}")
        print(f"      {'Latency':<25} {result.elapsed_time_sec}s\n")
        
        if result.chunks_persisted == 0:
            print("❌ Ingestion yielded 0 chunks. Aborting database verification.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Ingestion Pipeline crashed: {e}")
        db_session.close()
        sys.exit(1)

    # 5. Database Verification
    print("[4/4] Verifying Vector Persistence in Neon...")
    try:
        with db_manager.engine.connect() as conn:
            # BULLETPROOF QUERY: Search the actual textual content, not the metadata
            query = text("""
                SELECT 
                    COUNT(*) AS chunk_count,
                    COUNT(embedding) AS embedding_count,
                    COUNT(DISTINCT document_id) AS document_count
                FROM document_chunks
                WHERE text LIKE '%NovaTech employees receive 24 paid leave days%'
            """)

            db_result = conn.execute(query).fetchone()
            
            db_chunk_count = db_result[0]
            db_embedded_count = db_result[1]
            db_document_count = db_result[2]
            
            print(f"      {'Expected Chunks':<25} {result.chunks_persisted}")
            print(f"      {'Actual DB Chunks':<25} {db_chunk_count}")
            print(f"      {'Actual Embeddings':<25} {db_embedded_count}")
            print(f"      {'Actual DB Documents':<25} {db_document_count}")
            
            # Strict Assertions
            if db_chunk_count != result.chunks_persisted:
                print(f"\n      {'DB Verification':<25} FAIL")
                print("❌ The database chunk count does not match the pipeline telemetry.")
                sys.exit(1)
                
            if db_document_count != result.documents_processed:
                print(f"\n      {'DB Verification':<25} FAIL")
                print("❌ The distinct document_id footprint does not match processed documents.")
                sys.exit(1)
                
            print(f"\n      {'DB Verification':<25} PASS")
            
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        sys.exit(1)
    finally:
        db_session.close()

    print("\n========================================")
    print(" INGESTION TEST PASSED 🎉")
    print("========================================\n")


if __name__ == "__main__":
    main()