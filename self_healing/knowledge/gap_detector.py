import logging
from typing import Optional

from self_healing.models import RecoveryContext, RetryState, KnowledgeGap
from self_healing.constants import MIN_QUERY_LENGTH_FOR_GAP

from evaluation.constants import MIN_GOOD_RETRIEVAL_SIMILARITY
from retrieval.models import AnalyzedQuery

logger = logging.getLogger(__name__)


class GapDetector:
    """
    Observes the outcome of a Recovery loop and determines if the system 
    succeeded by relying on external data because its internal DB was deficient.
    """

    def detect(
        self, 
        recovery_context: RecoveryContext,
        retry_state: RetryState,
        original_analysis: AnalyzedQuery,
        query_id: str
    ) -> Optional[KnowledgeGap]:
        """
        Analyzes the execution state. Returns a KnowledgeGap if internal data was 
        missing but the external web search found the answer.
        Returns None if no gap was detected or the query was junk.
        """
        logger.info(f"GapDetector analyzing Recovery State | QueryID={query_id}")

        original_query = recovery_context.original_query

        
        if len(original_query) < MIN_QUERY_LENGTH_FOR_GAP:
            logger.debug(f"Query too short ({len(original_query)} chars) to form a meaningful gap.")
            return None

       
        if not retry_state.web_search_used:
            logger.debug("No external sources were used. No gap detected.")
            return None

        
        if not recovery_context.web_chunks:
            logger.debug("Web search was triggered but found no chunks. The topic might just be unanswerable.")
            return None

        
        best_internal_score = max([c.similarity_score for c in recovery_context.internal_chunks], default=0.0)
        
        if best_internal_score >= MIN_GOOD_RETRIEVAL_SIMILARITY:
            logger.debug(f"Internal retrieval was actually strong ({best_internal_score}). This is likely a hallucination issue, not a knowledge gap.")
            return None

       
        missing_topic = self._extract_topic(original_query, original_analysis)

        logger.info(f"KNOWLEDGE GAP DETECTED | Topic: '{missing_topic}' | Query: '{original_query}'")

       
        return KnowledgeGap(
            missing_topic=missing_topic,
            failed_queries=[original_query],
            frequency=1,
            resolved=False,
            last_query_id=query_id
        )

    def _extract_topic(self, original_query: str, analysis: AnalyzedQuery) -> str:
        """
        Derives the core missing subject using safe, cheap fallbacks 
        rather than expensive LLM calls.
        """
        if analysis.entities:
            return " + ".join(analysis.entities).title()
        if analysis.keywords:
            return " ".join(analysis.keywords[:3]).title()
        cleaned_query = original_query.replace("What is", "").replace("Explain", "").strip(" ?.")
        return cleaned_query.title()