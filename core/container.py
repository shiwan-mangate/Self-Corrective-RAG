import logging
from functools import cached_property
from sqlalchemy.orm import Session

# ==========================================
# Infrastructure & AI Integrations
# ==========================================
from config.settings import settings
from database.connection import db_manager
from ai.text_generation_service import AITextGenerationService
from ai.groq_client import GroqClient
from ai.embedding_service import AIEmbeddingService
from ai.groq_judge_service import GroqJudgeService
from ai.ragas_adapters import GroqRagasLLM, NativeRagasEmbeddings

# ==========================================
# Ingestion Subsystem
# ==========================================
from ingestion.loaders.factory import LoaderFactory
from ingestion.parser import DocumentParser
from ingestion.cleaner import DocumentCleaner
from ingestion.metadata import MetadataPipeline
from ingestion.chunker.pipeline import ChunkingPipeline
from ingestion.embedding import EmbeddingPipeline
from ingestion.pipeline import IngestionPipeline

# ==========================================
# Database Repositories & Services
# ==========================================
from database.repositories.vector_repository import VectorRepository
from database.repositories.retrieval_repository import RetrievalRepository
from database.repositories.knowledge_repository import PostgresKnowledgeStorage
from database.vector_service import VectorService
from database.retrieval_service import RetrievalService

# ==========================================
# Memory Subsystem
# ==========================================
from memory.session.storage import PostgresSessionStorage
from memory.conversation.storage import PostgresConversationStorage
from memory.session.manager import SessionManager
from memory.conversation.manager import ConversationManager
from memory.conversation.history import ConversationHistoryService
from memory.summarization.policy import SummaryPolicy
from memory.summarization.prompts import SummaryPromptBuilder
from memory.summarization.summarizer import ConversationSummarizer
from memory.builder.formatters import MarkdownMemoryFormatter
from memory.builder.context_builder import MemoryContextBuilder
from memory.pipeline import MemoryPipeline

# ==========================================
# Retrieval Subsystem
# ==========================================
from retrieval.analyzers.intent import IntentDetector
from retrieval.analyzers.query_analyzer import QueryAnalyzer
from retrieval.rewriter.prompt_builder import RewritePromptBuilder
from retrieval.rewriter.query_rewrite import LLMQueryRewriter
from retrieval.search.retriever import Retriever
from retrieval.search.reranker import Reranker
from retrieval.search.filters import RetrievalFilter
from retrieval.search.context_builder import ContextBuilder as RetrievalContextBuilder
from retrieval.pipeline import RetrievalPipeline

# ==========================================
# Generation Subsystem
# ==========================================
from generation.prompts.prompt_builder import PromptBuilder
from generation.generator.answer_generator import LLMAnswerGenerator
from generation.citations.extractor import CitationExtractor
from generation.citations.formatter import CitationFormatter
from generation.response.formatter import ResponseFormatter
from generation.pipeline import GenerationPipeline

# ==========================================
# Evaluation Subsystem
# ==========================================
from evaluation.grounding.verifier import LLMGroundingVerifier
from evaluation.hallucination.detector import LLMHallucinationDetector
from evaluation.confidence.scorer import DeterministicConfidenceScorer
from evaluation.ragas.dataset_builder import RagasDatasetBuilder
from evaluation.ragas.evaluator import RagasEvaluator
from evaluation.logger import EvaluationLogger
from evaluation.pipeline import EvaluationPipeline

# ==========================================
# Self-Healing Subsystem
# ==========================================
from self_healing.validator.validator import PolicyValidator
from self_healing.recovery.retry_manager import RetryManager
from self_healing.recovery.query_rewriter import RecoveryQueryRewriter
from self_healing.recovery.web_search import WebSearchService
from self_healing.recovery.context_merge import ContextMerger
from self_healing.knowledge.gap_detector import GapDetector
from self_healing.knowledge.knowledge_manager import KnowledgeManager
from self_healing.knowledge.ingestion_trigger import IngestionTrigger
from self_healing.pipeline import SelfHealingPipeline

# ==========================================
# LangGraph Orchestration
# ==========================================
from graph.builders.graph_builder import GraphBuilder
from graph.workflow import GraphWorkflow

logger = logging.getLogger(__name__)


class ApplicationContainer:
    """
    The Composition Root for the Self-Healing RAG.
    
    Architecture Rules:
    1. ZERO Business Logic: This file only instantiates and wires dependencies.
    2. @cached_property: Heavy AI clients and stateless pipelines load exactly once.
    3. Factory Methods: Database-dependent pipelines are instantiated dynamically 
       per-request to prevent SQLAlchemy session leaks.
    """

    # ---------------------------------------------------------
    # GLOBAL SINGLETONS (Stateless & Heavy Resources)
    # ---------------------------------------------------------
    
    @cached_property
    def groq_client(self) -> GroqClient:
        return GroqClient()
        
    @cached_property
    def text_generation_service(self) -> AITextGenerationService:
        # Explicit interface casting for downstream consumers
        return self.groq_client

    @cached_property
    def embedding_service(self) -> AIEmbeddingService:
        return AIEmbeddingService()

    @cached_property
    def judge_service(self) -> GroqJudgeService:
        return GroqJudgeService()

    @cached_property
    def ragas_llm(self) -> GroqRagasLLM:
        return GroqRagasLLM()

    @cached_property
    def ragas_embeddings(self) -> NativeRagasEmbeddings:
        return NativeRagasEmbeddings(self.embedding_service)

    # --- Memory Pure Logic ---
    @cached_property
    def conversation_history_service(self) -> ConversationHistoryService:
        return ConversationHistoryService()

    @cached_property
    def summary_policy(self) -> SummaryPolicy:
        return SummaryPolicy()

    @cached_property
    def memory_context_builder(self) -> MemoryContextBuilder:
        return MemoryContextBuilder(formatter=MarkdownMemoryFormatter())

    @cached_property
    def conversation_summarizer(self) -> ConversationSummarizer:
        return ConversationSummarizer(
            prompt_builder=SummaryPromptBuilder(),
            llm_service=self.text_generation_service
        )

    # --- Retrieval Pure Logic ---
    @cached_property
    def query_analyzer(self) -> QueryAnalyzer:
        return QueryAnalyzer(IntentDetector())

    @cached_property
    def llm_query_rewriter(self) -> LLMQueryRewriter:
        return LLMQueryRewriter(
            llm_service=self.text_generation_service,
            prompt_builder=RewritePromptBuilder()
        )

    @cached_property
    def retrieval_filter(self) -> RetrievalFilter:
        return RetrievalFilter()

    @cached_property
    def reranker(self) -> Reranker:
        return Reranker()

    @cached_property
    def retrieval_context_builder(self) -> RetrievalContextBuilder:
        return RetrievalContextBuilder()

    # --- Generation Pipeline (Stateless) ---
    @cached_property
    def generation_pipeline(self) -> GenerationPipeline:
        return GenerationPipeline(
            prompt_builder=PromptBuilder(),
            answer_generator=LLMAnswerGenerator(self.text_generation_service),
            citation_extractor=CitationExtractor(),
            citation_formatter=CitationFormatter(),
            response_formatter=ResponseFormatter()
        )

    # --- Evaluation Pipeline (Stateless) ---
    @cached_property
    def evaluation_logger(self) -> EvaluationLogger:
        # Passes the session factory, safe to cache as a singleton
        return EvaluationLogger(db_manager.SessionLocal)

    @cached_property
    def evaluation_pipeline(self) -> EvaluationPipeline:
        return EvaluationPipeline(
            grounding_verifier=LLMGroundingVerifier(self.judge_service),
            hallucination_detector=LLMHallucinationDetector(self.judge_service),
            confidence_scorer=DeterministicConfidenceScorer(),
            ragas_builder=RagasDatasetBuilder(),
            ragas_evaluator=RagasEvaluator(self.ragas_llm, self.ragas_embeddings),
            evaluation_logger=self.evaluation_logger
        )

    # --- Self-Healing Pure Logic ---
    @cached_property
    def policy_validator(self) -> PolicyValidator:
        return PolicyValidator()

    @cached_property
    def context_merger(self) -> ContextMerger:
        return ContextMerger()

    @cached_property
    def gap_detector(self) -> GapDetector:
        return GapDetector()

    @cached_property
    def knowledge_manager(self) -> KnowledgeManager:
        # Safe to cache because PostgresKnowledgeStorage takes the SessionLocal factory
        return KnowledgeManager(storage=PostgresKnowledgeStorage(db_manager.SessionLocal))

    # --- Ingestion Pure Logic ---
    @cached_property
    def document_parser(self) -> DocumentParser:
        return DocumentParser()

    @cached_property
    def document_cleaner(self) -> DocumentCleaner:
        return DocumentCleaner()

    @cached_property
    def metadata_pipeline(self) -> MetadataPipeline:
        return MetadataPipeline()

    @cached_property
    def chunking_pipeline(self) -> ChunkingPipeline:
        return ChunkingPipeline()

    @cached_property
    def embedding_pipeline(self) -> EmbeddingPipeline:
        return EmbeddingPipeline(self.embedding_service)


    # ---------------------------------------------------------
    # REQUEST-SCOPED FACTORIES (Database Session Bound)
    # ---------------------------------------------------------

    def create_ingestion_pipeline(self, db_session: Session) -> IngestionPipeline:
        """
        Creates an Ingestion Pipeline bound to the current transaction scope.
        """
        vector_repo = VectorRepository(db_session)
        vector_service = VectorService(vector_repo)
        
        return IngestionPipeline(
            loader_factory=LoaderFactory,  # Passed strictly as type
            parser=self.document_parser,
            cleaner=self.document_cleaner,
            metadata_extractor=self.metadata_pipeline,
            chunker=self.chunking_pipeline,
            embedder=self.embedding_pipeline,
            vector_service=vector_service
        )

    def create_memory_pipeline(self, db_session: Session) -> MemoryPipeline:
        """
        Creates a Memory Pipeline bound to the current transaction scope.
        Extracted so it can be used independently by benchmark runners.
        """
        session_storage = PostgresSessionStorage(db_session)
        conversation_storage = PostgresConversationStorage(db_session)
        
        session_manager = SessionManager(session_storage)
        conversation_manager = ConversationManager(
            conversation_storage, 
            self.conversation_history_service
        )
        
        return MemoryPipeline(
            session_manager=session_manager,
            conversation_manager=conversation_manager,
            summary_policy=self.summary_policy,
            summarizer=self.conversation_summarizer,
            context_builder=self.memory_context_builder
        )

    def create_workflow(self, db_session: Session) -> GraphWorkflow:
        """
        The Master Assembly Line.
        Creates a structurally complete, mathematically perfect LangGraph Workflow.
        Ties database repositories to the EXACT session yielded by FastAPI.
        """
        
        # 1. Database Repositories
        vector_repo = VectorRepository(db_session)
        retrieval_repo = RetrievalRepository(db_session)

        # 2. Database Services & Managers
        retrieval_service = RetrievalService(retrieval_repo)

        # 3. Request-Scoped Memory Pipeline (Using the newly extracted method)
        memory_pipeline = self.create_memory_pipeline(db_session)

        # 4. Request-Scoped Retrieval Pipeline
        retriever = Retriever(self.embedding_service, retrieval_service)
        retrieval_pipeline = RetrievalPipeline(
            analyzer=self.query_analyzer,
            rewriter=self.llm_query_rewriter,
            retriever=retriever,
            filter_engine=self.retrieval_filter,
            reranker=self.reranker,
            context_builder=self.retrieval_context_builder
        )

        # 5. Request-Scoped Self-Healing Pipeline
        retry_manager = RetryManager()  # State mutable per request
        query_rewriter = RecoveryQueryRewriter(self.llm_query_rewriter, retry_manager)
        web_search = WebSearchService(retry_manager)
        
        # Ingestion triggered dynamically requires the scoped DB session
        scoped_ingestion = self.create_ingestion_pipeline(db_session)
        ingestion_trigger = IngestionTrigger(scoped_ingestion)

        self_healing_pipeline = SelfHealingPipeline(
            validator=self.policy_validator,
            retry_manager=retry_manager,
            query_rewriter=query_rewriter,
            web_search=web_search,
            context_merger=self.context_merger,
            gap_detector=self.gap_detector,
            knowledge_manager=self.knowledge_manager,
            ingestion_trigger=ingestion_trigger
        )

        # 6. Graph Compilation
        graph_builder = GraphBuilder(
            memory_pipeline=memory_pipeline,
            retrieval_pipeline=retrieval_pipeline,
            generation_pipeline=self.generation_pipeline,  # Thread-safe Singleton
            evaluation_pipeline=self.evaluation_pipeline,  # Thread-safe Singleton
            self_healing_pipeline=self_healing_pipeline
        )
        
        compiled_graph = graph_builder.build()
        
        # 7. Expose Public Orchestrator
        return GraphWorkflow(compiled_graph)

# Global Container Instance to be imported by core/startup.py and api/dependencies.py
container = ApplicationContainer()