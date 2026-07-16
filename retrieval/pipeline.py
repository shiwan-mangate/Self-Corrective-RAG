import logging
from retrieval.models import SearchQuery, RetrievalContext, RetrievalStatistics
from retrieval.analyzers.base import BaseQueryAnalyzer
from retrieval.rewriter.base import BaseQueryRewriter
from retrieval.search.retriever import Retriever
from retrieval.search.filters import RetrievalFilter
from retrieval.search.reranker import Reranker
from retrieval.search.context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        analyzer: BaseQueryAnalyzer,
        rewriter: BaseQueryRewriter,
        retriever: Retriever,
        filter_engine: RetrievalFilter,
        reranker: Reranker,
        context_builder: ContextBuilder,
    ):
        self.analyzer = analyzer
        self.rewriter = rewriter
        self.retriever = retriever
        self.filter_engine = filter_engine
        self.reranker = reranker
        self.context_builder = context_builder

    def retrieve(self, search_query: SearchQuery) -> RetrievalContext:
        logger.info(f"Initiating Retrieval Pipeline for: '{search_query.query}'")
        
       
        analyzed_query = self.analyzer.analyze(search_query)
        
       
        analyzed_query = self.rewriter.rewrite(analyzed_query, search_query.chat_history)
        
      
        retrieval_result = self.retriever.retrieve(analyzed_query)
        initial_count = len(retrieval_result.chunks)
        
       
        filtered_chunks = self.filter_engine.filter(
            chunks=retrieval_result.chunks,
            analyzed_query=analyzed_query
        )
        filtered_count = len(filtered_chunks)
        
       
        ranked_chunks = self.reranker.rerank(
            chunks=filtered_chunks,
            analyzed_query=analyzed_query
        )
        
        
        pipeline_stats = RetrievalStatistics(
            total_chunks_retrieved=initial_count,
            filtered_chunks=initial_count - filtered_count,
            reranked_chunks=len(ranked_chunks)
        )
        
        context = self.context_builder.build(
            ranked_chunks=ranked_chunks,
            analyzed_query=analyzed_query,
            original_query=search_query,
            pipeline_statistics=pipeline_stats
        )
        
        logger.info(f"Retrieval Pipeline Complete. Strategy: {analyzed_query.search_type.value}")
        return context