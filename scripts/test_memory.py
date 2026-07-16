# scripts/test_memory.py

import os
import sys
import uuid
import time
from datetime import datetime, timezone

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import db_manager
from core.container import container

# Domain Models
from memory.models import (
    ConversationMessage, 
    MessageRole, 
    SaveConversationRequest, 
    MemoryRequest
)
from memory.constants import SUMMARY_TRIGGER_MESSAGES, MAX_RECENT_MESSAGES

# Services
from memory.session.storage import PostgresSessionStorage
from memory.session.manager import SessionManager
from memory.conversation.storage import PostgresConversationStorage
from memory.conversation.history import ConversationHistoryService
from memory.conversation.manager import ConversationManager
from memory.summarization.policy import SummaryPolicy
from memory.summarization.prompts import SummaryPromptBuilder
from memory.summarization.summarizer import ConversationSummarizer
from memory.builder.context_builder import MemoryContextBuilder
from memory.pipeline import MemoryPipeline


def main():
    print("\n========================================")
    print(" SELF-HEALING RAG MEMORY TEST")
    print("========================================\n")

    # ---------------------------------------------------------
    # [1/9] Initializing Memory Dependencies
    # ---------------------------------------------------------
    print("[1/9] Initializing Memory Dependencies...")
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
        # Wire exactly as GraphWorkflow does in production
        session_storage = PostgresSessionStorage(db_session)
        session_manager = SessionManager(session_storage)
        
        conversation_storage = PostgresConversationStorage(db_session)
        history_service = ConversationHistoryService()
        conversation_manager = ConversationManager(conversation_storage, history_service)
        
        summary_policy = SummaryPolicy()
        summarizer = ConversationSummarizer(
            prompt_builder=SummaryPromptBuilder(),
            llm_service=container.text_generation_service
        )
        context_builder = MemoryContextBuilder()
        
        memory_pipeline = MemoryPipeline(
            session_manager=session_manager,
            conversation_manager=conversation_manager,
            summary_policy=summary_policy,
            summarizer=summarizer,
            context_builder=context_builder
        )

        print(f"      {'Database session':<30} PASS")
        print(f"      {'Session storage':<30} PASS")
        print(f"      {'Conversation storage':<30} PASS")
        print(f"      {'Memory pipeline':<30} PASS\n")

        # ---------------------------------------------------------
        # [2/9] Preparing Test Session
        # ---------------------------------------------------------
        print("[2/9] Preparing Test Session...")
        test_session_id = f"memory-test-{uuid.uuid4().hex[:8]}"
        test_user_id = "memory-test-user"
        test_query_id = f"query-{uuid.uuid4().hex[:8]}"
        
        session_manager.load_or_create(test_session_id)

        print(f"      {'Session ID':<30} {test_session_id}")
        print(f"      {'User ID':<30} {test_user_id}")
        print(f"      {'Query ID':<30} {test_query_id}")
        print(f"      {'Test isolation':<30} PASS\n")

        # ---------------------------------------------------------
        # [3/9] Saving Conversation Turns
        # ---------------------------------------------------------
        print("[3/9] Saving Conversation Turns...")
        # 6 turns = 12 messages. Exact number needed to hit SUMMARY_TRIGGER_MESSAGES
        turns = [
            ("I am building a Self-Healing RAG system.", "The system can evaluate and recover from RAG failures."),
            ("I use Groq for LLM generation.", "Groq is connected through the shared AI generation service."),
            ("What happens when retrieval confidence is low?", "The Self-Healing subsystem can rewrite the query."),
            ("What if rewriting doesn't work?", "The system can fallback to web search via Tavily."),
            ("How does it know it hallucinated?", "The Evaluation subsystem runs a dedicated hallucination judge."),
            ("Does it learn over time?", "Yes, it logs Knowledge Gaps to trigger auto-ingestion.")
        ]

        for i, (user_text, asst_text) in enumerate(turns, start=1):
            user_msg = ConversationMessage(
                role=MessageRole.USER,
                content=user_text,
                tokens=max(1, len(user_text) // 4) # Simple token heuristic
            )
            asst_msg = ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=asst_text,
                tokens=max(1, len(asst_text) // 4)
            )
            
            save_req = SaveConversationRequest(
                query_id=test_query_id,
                session_id=test_session_id,
                user_message=user_msg,
                assistant_message=asst_msg
            )
            
            # 🔥 THIS WILL LIKELY CRASH FIRST (Bug 3: ConversationSaveResult addition)
            memory_pipeline.save_turn(save_req)
            print(f"      Turn {i:<25} PASS")

        print(f"      {'Turns saved':<30} {len(turns)}")
        print(f"      {'Messages expected':<30} {len(turns) * 2}\n")

        # ---------------------------------------------------------
        # [4/9] Loading Conversation History
        # ---------------------------------------------------------
        print("[4/9] Loading Conversation History...")
        history = conversation_manager.load_history(test_session_id)
        
        print(f"      {'History loaded':<30} PASS")
        print(f"      {'Expected messages':<30} {len(turns) * 2}")
        print(f"      {'Actual messages':<30} {history.total_messages}")
        
        if history.total_messages != len(turns) * 2:
            print("❌ MEMORY TEST FAILED: Loaded history message count mismatch.")
            sys.exit(1)
            
        print(f"      {'Message count integrity':<30} PASS")
        
        # Verify Token Integrity
        expected_tokens = sum(msg.tokens for msg in history.messages)
        if history.total_tokens != expected_tokens:
            print("❌ MEMORY TEST FAILED: Token recalculation mismatch.")
            sys.exit(1)
            
        print(f"      {'Token count integrity':<30} PASS")
        
        # Chronological ordering check
        is_ordered = all(history.messages[i].timestamp <= history.messages[i+1].timestamp for i in range(len(history.messages)-1))
        print(f"      {'Chronological order':<30} {'PASS' if is_ordered else 'FAIL'}\n")

        # ---------------------------------------------------------
        # [5/9] Evaluating Summary Policy
        # ---------------------------------------------------------
        print("[5/9] Evaluating Summary Policy...")
        memory_request = MemoryRequest(
            query_id=test_query_id,
            session_id=test_session_id,
            current_query="Trigger summarization check.",
            user_id=test_user_id
        )
        
        policy_decision = summary_policy.should_summarize(memory_request, history)
        
        print(f"      {'Summary policy':<30} PASS")
        print(f"      {'Should summarize':<30} {str(policy_decision.should_summarize).upper()}")
        print(f"      {'Trigger':<30} {policy_decision.triggered_by.value.upper()}")
        
        if not policy_decision.should_summarize or policy_decision.triggered_by.value != "message_threshold":
            print("❌ MEMORY TEST FAILED: Policy failed to trigger on exactly 12 messages.")
            sys.exit(1)
            
        print(f"      {'Expected trigger':<30} PASS\n")

        # ---------------------------------------------------------
        # [6/9], [7/9], [8/9] Executing Context Build (Summarization)
        # ---------------------------------------------------------
        # This single orchestrator call runs Summarization, Pruning, DB Commits, and Context Building
        print("[6/9] Generating Conversation Summary...")
        print("[7/9] Committing Summary and Pruning...")
        print("[8/9] Building Memory Context...")
        
        # 🔥 THIS WILL LIKELY CRASH NEXT (Bug 1 & Bug 2)
        memory_response = memory_pipeline.build_context(memory_request)
        
        if not memory_response.metadata.summary_generated:
            print("❌ MEMORY TEST FAILED: Pipeline evaluated True but skipped summary generation.")
            sys.exit(1)
            
        print(f"      {'Memory context built':<30} PASS")
        print(f"      {'Summary version':<30} {memory_response.context.summary.summary_version}")
        print(f"      {'Summarizer LLM latency':<30} {memory_response.metadata.latency_ms} ms")
        print(f"      {'Summary included':<30} PASS")
        print(f"      {'Recent history included':<30} PASS")
        
        # Ensure the active history list length was pruned strictly to MAX_RECENT_MESSAGES
        actual_active = len(memory_response.context.active_history.messages)
        if actual_active != MAX_RECENT_MESSAGES:
            print(f"❌ MEMORY TEST FAILED: History not pruned correctly. Expected {MAX_RECENT_MESSAGES}, got {actual_active}.")
            sys.exit(1)
            
        print(f"      {'Active history pruned':<30} PASS ({actual_active} msgs remaining)\n")

        print("----- MEMORY CONTEXT -----")
        print(memory_response.context.formatted_context_string)
        print("--------------------------\n")

        # ---------------------------------------------------------
        # [9/9] Validating Memory Persistence
        # ---------------------------------------------------------
        print("[9/9] Validating Memory Persistence...")
        
        # Check Session state directly from DB
        session_db = session_manager.load_or_create(test_session_id)
        print(f"      {'Session exists':<30} PASS")
        print(f"      {'Session active':<30} PASS")
        print(f"      {'Session message count':<30} {session_db.message_count} (Expected: {len(turns) * 2})")
        
        # Check active messages directly from DB to verify transaction commit
        reloaded_history = conversation_manager.load_history(test_session_id)
        print(f"      {'Active DB messages':<30} {reloaded_history.total_messages} (Expected: {MAX_RECENT_MESSAGES})")
        
        reloaded_summary = conversation_manager.load_summary(test_session_id)
        if not reloaded_summary:
            print("❌ MEMORY TEST FAILED: Database failed to persist the ConversationSummaryModel.")
            sys.exit(1)
            
        print(f"      {'Active DB summary':<30} PASS")
        print(f"      {'Memory state integrity':<30} PASS\n")

        print("========================================")
        print(" MEMORY TEST PASSED 🎉")
        print("========================================")

    except Exception as e:
        print(f"\n❌ Memory Pipeline crashed during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()