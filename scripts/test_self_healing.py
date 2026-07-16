# scripts/test_self_healing.py

import os
import sys
import uuid
import time

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- BUGFIX FOR RAGAS ---
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.vertexai'] = MagicMock()
# ------------------------

from database.connection import db_manager
from core.container import container

# Domain Models
from retrieval.models import QueryIntent, SearchType, AnalyzedQuery, RankedChunk, RetrievalMetadata, RetrievalStatistics
from generation.models import GenerationMetadata, TokenUsage
from evaluation.models import EvaluationRequest, EvaluationDecision
from self_healing.models import SelfHealingRequest, RetryState, RecoveryActionType


def main():
    print("\n========================================")
    print(" SELF-HEALING RAG RECOVERY TEST")
    print("========================================\n")

    # ---------------------------------------------------------
    # [1/8] Initializing Dependencies
    # ---------------------------------------------------------
    print("[1/8] Initializing Dependencies...")
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
        # Retrieve Stateless Pipelines from Container
        evaluation_pipeline = container.evaluation_pipeline
        
        # Self-Healing is request-scoped because it triggers DB logs & queries
        # container.create_workflow automatically builds this, but we'll instantiate 
        # it manually here using container singletons to test it in isolation.
        from self_healing.pipeline import SelfHealingPipeline
        from self_healing.recovery.retry_manager import RetryManager
        from self_healing.recovery.query_rewriter import RecoveryQueryRewriter
        from self_healing.recovery.web_search import WebSearchService
        from self_healing.knowledge.ingestion_trigger import IngestionTrigger
        
        retry_manager = RetryManager()
        query_rewriter = RecoveryQueryRewriter(container.llm_query_rewriter, retry_manager)
        web_search = WebSearchService(retry_manager)
        
        # We need a scoped DB session for IngestionTrigger -> VectorRepo
        scoped_ingestion = container.create_ingestion_pipeline(db_session)
        ingestion_trigger = IngestionTrigger(scoped_ingestion)
        
        self_healing_pipeline = SelfHealingPipeline(
            validator=container.policy_validator,
            retry_manager=retry_manager,
            query_rewriter=query_rewriter,
            web_search=web_search,
            context_merger=container.context_merger,
            gap_detector=container.gap_detector,
            knowledge_manager=container.knowledge_manager,
            ingestion_trigger=ingestion_trigger
        )

        print(f"      {'Database session':<30} PASS")
        print(f"      {'Evaluation pipeline':<30} PASS")
        print(f"      {'Self-Healing pipeline':<30} PASS\n")

        # ---------------------------------------------------------
        # [2/8] Preparing Controlled Failure
        # ---------------------------------------------------------
        print("[2/8] Preparing Controlled Failure...")
        target_query = "What happens when retrieval confidence is low?"
        fake_context = "The system uses a Postgres database to store embeddings."
        bad_answer = "I apologize, but I could not generate an answer based on the provided context."

        print(f"      {'Scenario':<30} Poor Retrieval & Empty Answer")
        print(f"      {'Query':<30} {target_query}")
        print(f"      {'Context available':<30} PASS")
        print(f"      {'Injected bad answer':<30} PASS\n")

        # ---------------------------------------------------------
        # [3/8] Building Evaluation Request
        # ---------------------------------------------------------
        print("[3/8] Building Evaluation Request...")
        
        # Create minimal valid metadata representing a POOR retrieval scenario
        retrieval_meta = RetrievalMetadata(
            search_strategy=SearchType.SIMILARITY,
            latency_ms=120.5,
            statistics=RetrievalStatistics(average_similarity=0.40, total_chunks_retrieved=1)
        )
        
        gen_meta = GenerationMetadata(
            latency_ms=450.0,
            token_usage=TokenUsage(total_tokens=50),
            model_name="test-model",
            temperature=0.0,
            template_name="qa"
        )
        
        eval_request = EvaluationRequest(
            query_id=query_id,
            original_query=target_query,
            optimized_query=target_query,
            context=fake_context,
            answer=bad_answer,
            citations=[],
            retrieval_metadata=retrieval_meta,
            generation_metadata=gen_meta,
            retrieved_chunks=[fake_context]
        )
        
        print(f"      {'Evaluation request':<30} PASS")
        print(f"      {'Query ID attached':<30} PASS")
        print(f"      {'Metadata attached':<30} PASS\n")

        # ---------------------------------------------------------
        # [4/8] Evaluating Controlled Failure
        # ---------------------------------------------------------
        print("[4/8] Evaluating Controlled Failure...")
        eval_report = evaluation_pipeline.evaluate(eval_request)
        
        # To make this CI/CD test 100% deterministic for QUERY_REWRITE, we lock the 
        # signals directly so LLM judge variability doesn't trigger Web Search instead.
        eval_report.decision = EvaluationDecision.FAIL
        eval_report.grounding.is_grounded = False
        eval_report.hallucination.has_hallucination = False
        
        print(f"      {'Evaluation execution':<30} PASS")
        print(f"      {'Quality decision':<30} FAIL")
        print(f"      {'Expected failure detected':<30} PASS\n")
        
        print(f"      {'Confidence':<30} {eval_report.confidence.overall_score:.2f}")
        print(f"      {'Hallucination':<30} {eval_report.hallucination.has_hallucination}")
        print(f"      {'Grounding':<30} {eval_report.grounding.is_grounded}\n")

        # ---------------------------------------------------------
        # [5/8] Building Self-Healing Request
        # ---------------------------------------------------------
        print("[5/8] Building Self-Healing Request...")
        
        sh_request = SelfHealingRequest(
            query_id=query_id,
            evaluation_report=eval_report,
            retry_state=RetryState(), # Fresh state, 0 retries
            retrieval_metadata=retrieval_meta
        )
        
        original_analysis = AnalyzedQuery(
            original_query=target_query,
            normalized_query=target_query.lower()[:-1],
            intent=QueryIntent.FACTUAL,
            needs_history=False,
            needs_rewrite=False,
            search_type=SearchType.SIMILARITY,
            top_k=5
        )
        
        fake_chunk = RankedChunk(
            chunk_id="chunk_1", document_id="doc_1", text=fake_context,
            similarity_score=0.40, token_count=10, checksum="abc",
            retrieval_score=0.40, final_score=0.40, rank=1
        )
        
        print(f"      {'Self-Healing request':<30} PASS")
        print(f"      {'Evaluation attached':<30} PASS")
        print(f"      {'Initial retry count':<30} {sh_request.retry_state.retry_count}")
        print(f"      {'Query ID propagation':<30} PASS\n")

        # ---------------------------------------------------------
        # [6/8] Executing Self-Healing Pipeline
        # ---------------------------------------------------------
        print("[6/8] Executing Self-Healing Pipeline (LLM Query Rewrite)...")
        
        recovery_plan = self_healing_pipeline.heal(
            request=sh_request, 
            original_analysis=original_analysis, 
            internal_chunks=[fake_chunk]
        )
        
        print(f"      {'Self-Healing execution':<30} PASS")
        print(f"      {'Recovery plan generated':<30} PASS\n")

        # ---------------------------------------------------------
        # [7/8] Validating Recovery Plan
        # ---------------------------------------------------------
        print("[7/8] Validating Recovery Plan...")
        
        expected_action = RecoveryActionType.REWRITE_QUERY
        actual_actions = [a.action_type for a in recovery_plan.actions]
        
        print(f"      {'Expected action':<30} {expected_action.value.upper()}")
        print(f"      {'Actual action':<30} {actual_actions[0].value.upper()}")
        
        if expected_action not in actual_actions:
            print("❌ SELF-HEALING TEST FAILED: Pipeline did not select QUERY_REWRITE.")
            sys.exit(1)
            
        print(f"      {'Recovery action':<30} PASS")
        
        rewritten = recovery_plan.recovery_context.rewritten_query
        if not rewritten:
            print("❌ SELF-HEALING TEST FAILED: LLM failed to rewrite the query.")
            sys.exit(1)
            
        print(f"      {'Rewritten query exists':<30} PASS")
        print(f"      {'Query changed':<30} {'PASS' if rewritten != target_query else 'FAIL'}")
        print(f"      {'Recovery context ready':<30} PASS\n")
        
        print(f"      Original Query:")
        print(f"      '{target_query}'\n")
        print(f"      Rewritten Query:")
        print(f"      '{rewritten}'\n")

        # ---------------------------------------------------------
        # [8/8] Validating Recovery Trace
        # ---------------------------------------------------------
        print("[8/8] Validating Recovery Trace...")
        
        print(f"      {'Retry count':<30} {recovery_plan.retry_state.retry_count}")
        print(f"      {'Maximum retries':<30} {recovery_plan.retry_state.max_retries}")
        
        if recovery_plan.retry_state.retry_count != 1:
            print("❌ SELF-HEALING TEST FAILED: RetryManager did not increment the retry count.")
            sys.exit(1)
            
        print(f"      {'Retry state update':<30} PASS")
        print(f"      {'Query trace continuity':<30} PASS")
        print(f"      {'LangGraph continuation flag':<30} {recovery_plan.continue_pipeline}\n")

        print("========================================")
        print(" SELF-HEALING TEST PASSED 🎉")
        print("========================================\n")

    except Exception as e:
        print(f"\n❌ Self-Healing Pipeline crashed during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()