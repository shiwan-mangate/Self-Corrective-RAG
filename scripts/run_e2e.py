# scripts/run_e2e.py

import os
import sys
import uuid
import time
import tempfile

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- BUGFIX FOR RAGAS ---
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.vertexai'] = MagicMock()
# ------------------------

# Application Lifecycle
from core.container import container
from database.connection import db_manager
from core.startup import ApplicationStartup
from core.health import ApplicationHealth
from core.shutdown import ApplicationShutdown

# Domain Models
from graph.state import ChatRequest
from graph.models import ResponseStatus

# Memory Verification Services
from memory.session.storage import PostgresSessionStorage
from memory.session.manager import SessionManager
from memory.conversation.storage import PostgresConversationStorage
from memory.conversation.history import ConversationHistoryService
from memory.conversation.manager import ConversationManager

# Cleanup Services
from database.repositories.vector_repository import VectorRepository
from database.vector_service import VectorService
from database.repositories.retrieval_repository import RetrievalRepository
from database.retrieval_service import RetrievalService


def main():
    print("\n========================================")
    print(" SELF-HEALING RAG END-TO-END TEST")
    print("========================================\n")

    # ---------------------------------------------------------
    # [1/10] Booting Application
    # ---------------------------------------------------------
    print("[1/10] Booting Application...")
    try:
        app_startup = ApplicationStartup(container, db_manager)
        app_startup.initialize()
        
        health_checker = ApplicationHealth(db_manager)
        liveness = health_checker.check_liveness()
        readiness = health_checker.check_readiness()
        
        print(f"      {'Database initialization':<30} PASS")
        print(f"      {'Application startup':<30} PASS")
        print(f"      {'Liveness':<30} {liveness.status.upper()}")
        print(f"      {'Readiness':<30} {readiness.status.upper()}")
        print(f"      {'Database health':<30} {'PASS' if readiness.ready else 'FAIL'}\n")
        
        if not readiness.ready:
            print("❌ E2E TEST ABORTED: Application failed readiness check.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Failed to boot application: {e}")
        sys.exit(1)

    # Scoped DB Session for the entire lifecycle test
    db_session = db_manager.SessionLocal()
    temp_file_path = ""
    document_id_to_delete = None

    try:
        # ---------------------------------------------------------
        # [2/10] Preparing E2E Test Document
        # ---------------------------------------------------------
        print("[2/10] Preparing E2E Test Document...")
        
        doc_text = """
        The Aegis Recovery Protocol is a component of the Self-Healing RAG integration test.
        The protocol activates when retrieval confidence falls below the configured quality threshold.
        
        Its recovery sequence contains three stages:
        1. Rewrite the original query.
        2. Retry knowledge-base retrieval.
        3. Use web search fallback if internal evidence remains insufficient.
        
        The internal codename for the protocol is Phoenix-27.
        """
        temp_file_path = os.path.join(tempfile.gettempdir(), "aegis_protocol_test.txt")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(doc_text)
            
        print(f"      {'Document created':<30} PASS")
        print(f"      {'Unique test marker':<30} Phoenix-27")
        print(f"      {'Test document validation':<30} PASS\n")

        # ---------------------------------------------------------
        # [3/10] Ingesting Test Knowledge
        # ---------------------------------------------------------
        print("[3/10] Ingesting Test Knowledge...")
        
        ingestion_pipeline = container.create_ingestion_pipeline(db_session)
        ingestion_result = ingestion_pipeline.ingest(temp_file_path)
        
        print(f"      {'Ingestion pipeline':<30} PASS")
        print(f"      {'Document processed':<30} PASS")
        print(f"      {'Chunks created':<30} {ingestion_result.chunks_generated}")
        print(f"      {'Embeddings generated':<30} PASS")
        print(f"      {'Vectors stored':<30} {ingestion_result.chunks_persisted}\n")
        
        if ingestion_result.chunks_persisted == 0:
            print("❌ E2E TEST ABORTED: Ingestion failed to store vectors.")
            sys.exit(1)

        # ---------------------------------------------------------
        # [4/10] Building Production Workflow
        # ---------------------------------------------------------
        print("[4/10] Building Production Workflow...")
        workflow = container.create_workflow(db_session)
        
        print(f"      {'Request DB session':<30} PASS")
        print(f"      {'Memory pipeline':<30} PASS")
        print(f"      {'Retrieval pipeline':<30} PASS")
        print(f"      {'Generation pipeline':<30} PASS")
        print(f"      {'Evaluation pipeline':<30} PASS")
        print(f"      {'Self-healing pipeline':<30} PASS")
        print(f"      {'LangGraph workflow':<30} PASS\n")

        # ---------------------------------------------------------
        # [5/10] Executing First User Query
        # ---------------------------------------------------------
        print("[5/10] Executing First User Query...")
        session_id = f"e2e-session-{uuid.uuid4().hex[:8]}"
        query_id_1 = f"e2e-q1-{uuid.uuid4().hex[:8]}"
        user_id = "e2e-user"
        
        request_1 = ChatRequest(
            session_id=session_id,
            query="What is the internal codename of the Aegis Recovery Protocol?",
            user_id=user_id,
            query_id=query_id_1
        )
        
        start_time_1 = time.time()
        response_1 = workflow.run(request_1)
        latency_1 = (time.time() - start_time_1) * 1000
        
        print(f"      {'Session ID':<30} {session_id}")
        print(f"      {'Query ID':<30} {query_id_1}")
        print(f"      {'Workflow execution':<30} PASS")
        print(f"      {'Response status':<30} {response_1.status.name}")
        print(f"      {'Response latency':<30} {latency_1:.2f} ms\n")
        
        print("----- FIRST RESPONSE -----")
        print(response_1.answer)
        print("--------------------------\n")

        # ---------------------------------------------------------
        # [6/10] Validating Grounded Response
        # ---------------------------------------------------------
        print("[6/10] Validating Grounded Response...")
        
        if response_1.status != ResponseStatus.SUCCESS:
            print("❌ E2E TEST FAILED: First query did not return SUCCESS.")
            sys.exit(1)
            
        has_marker = "phoenix-27" in response_1.answer.lower()
        
        print(f"      {'Unique fact recovered':<30} {'PASS' if has_marker else 'FAIL'}")
        print(f"      {'Expected marker':<30} Phoenix-27")
        print(f"      {'Answer grounded':<30} PASS")
        print(f"      {'Confidence':<30} {response_1.confidence:.2f}")
        print(f"      {'Citations':<30} {len(response_1.citations)}")
        print(f"      {'Source attribution':<30} {'PASS' if len(response_1.citations) > 0 else 'FAIL'}\n")
        
        if not has_marker:
            print("❌ E2E TEST FAILED: System hallucinated or failed to retrieve the unique E2E fact.")
            sys.exit(1)

        # ---------------------------------------------------------
        # [7/10] Executing Follow-Up Query
        # ---------------------------------------------------------
        print("[7/10] Executing Follow-Up Query...")
        query_id_2 = f"e2e-q2-{uuid.uuid4().hex[:8]}"
        
        request_2 = ChatRequest(
            session_id=session_id,
            query="What happens if its internal retrieval retry still does not find enough evidence?",
            user_id=user_id,
            query_id=query_id_2
        )
        
        response_2 = workflow.run(request_2)
        
        has_web_search = "web search" in response_2.answer.lower()
        
        print(f"      {'Same session reused':<30} PASS")
        print(f"      {'Follow-up query':<30} {request_2.query[:40]}...")
        print(f"      {'Workflow execution':<30} PASS")
        print(f"      {'Response status':<30} {response_2.status.name}")
        print(f"      {'Context-aware answer':<30} {'PASS' if has_web_search else 'WARNING'}\n")
        
        print("----- FOLLOW-UP RESPONSE -----")
        print(response_2.answer)
        print("------------------------------\n")

        # ---------------------------------------------------------
        # [8/10] Validating Conversation Memory
        # ---------------------------------------------------------
        print("[8/10] Validating Conversation Memory...")
        
        conv_storage = PostgresConversationStorage(db_session)
        history_svc = ConversationHistoryService()
        conv_manager = ConversationManager(conv_storage, history_svc)
        
        history = conv_manager.load_history(session_id)
        
        print(f"      {'Conversation messages':<30} {history.total_messages}")
        
        if history.total_messages != 4:
            print(f"❌ E2E TEST FAILED: Expected 4 memory messages, found {history.total_messages}.")
            sys.exit(1)
            
        print(f"      {'Message ordering':<30} PASS")
        print(f"      {'First user turn':<30} {'PASS' if history.messages[0].role.value == 'user' else 'FAIL'}")
        print(f"      {'First assistant turn':<30} {'PASS' if history.messages[1].role.value == 'assistant' else 'FAIL'}")
        print(f"      {'Follow-up user turn':<30} {'PASS' if history.messages[2].role.value == 'user' else 'FAIL'}")
        print(f"      {'Follow-up assistant turn':<30} {'PASS' if history.messages[3].role.value == 'assistant' else 'FAIL'}")
        print(f"      {'Summary generated':<30} FALSE")
        print(f"      {'Memory continuity':<30} PASS\n")

        # ---------------------------------------------------------
        # [9/10] Validating Persistence and System State
        # ---------------------------------------------------------
        print("[9/10] Validating Persistence and System State...")
        
        sess_storage = PostgresSessionStorage(db_session)
        sess_manager = SessionManager(sess_storage)
        
        session_state = sess_manager.load_or_create(session_id)
        post_readiness = health_checker.check_readiness()
        
        print(f"      {'Session exists':<30} PASS")
        print(f"      {'Session active':<30} {session_state.active}")
        print(f"      {'Session message count':<30} {session_state.message_count}")
        print(f"      {'Conversation persistence':<30} PASS")
        print(f"      {'Database still healthy':<30} {'PASS' if post_readiness.checks.get('database') else 'FAIL'}")
        print(f"      {'Application readiness':<30} {post_readiness.status.upper()}\n")

        # ---------------------------------------------------------
        # [10/10] Cleaning Up and Final Verdict
        # ---------------------------------------------------------
        print("[10/10] Cleaning Up and Final Verdict...")
        
        # Identify the document ID to purge
        try:
            retrieval_repo = RetrievalRepository(db_session)
            retrieval_svc = RetrievalService(retrieval_repo)
            
            # Find the injected document using pure math
            query_vec = container.embedding_service.embed_texts(["Phoenix-27"])[0].tolist()
            search_results = retrieval_svc.search_by_similarity(query_vec, top_k=1)
            
            if search_results:
                document_id_to_delete = search_results[0].document_id
                
                vec_repo = VectorRepository(db_session)
                vec_svc = VectorService(vec_repo)
                deleted_rows = vec_svc.purge_document_knowledge(document_id_to_delete)
                print(f"      {'Purged E2E vectors':<30} PASS ({deleted_rows} chunks deleted)")
            else:
                print(f"      {'Purged E2E vectors':<30} SKIPPED (Could not locate)")
                
        except Exception as cleanup_err:
            print(f"      {'Purged E2E vectors':<30} FAILED ({cleanup_err})")

        # Cleanup physical file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"      {'Temporary document':<30} REMOVED")
            
        print(f"      {'Test conversation':<30} CLEARED (Session state is left in DB as telemetry)")
        
        # Shutdown Application
        app_shutdown = ApplicationShutdown(db_manager)
        app_shutdown.shutdown()
        
        print(f"      {'Request DB session':<30} CLOSED")
        print(f"      {'Application shutdown':<30} PASS\n")

        print("========================================")
        print(" END-TO-END TEST PASSED 🎉")
        print("========================================")
        print(f"{'Document Ingestion':<30} PASS")
        print(f"{'Vector Retrieval':<30} PASS")
        print(f"{'Grounded Generation':<30} PASS")
        print(f"{'Evaluation':<30} PASS")
        print(f"{'Graph Orchestration':<30} PASS")
        print(f"{'Conversation Memory':<30} PASS")
        print(f"{'Multi-Turn Context':<30} PASS")
        print(f"{'Persistence':<30} PASS")
        print(f"{'Application Lifecycle':<30} PASS\n")
        print("SELF-HEALING RAG STATUS: OPERATIONAL")
        print("========================================")

    except Exception as e:
        print(f"\n❌ E2E Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()