import logging
from typing import Optional, Callable
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from self_healing.models import KnowledgeGap
from self_healing.constants import AUTO_TRIGGER_INGESTION, KNOWLEDGE_GAP_TRIGGER_COUNT, KNOWLEDGE_GAP_EXPIRY_DAYS

from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)



class TriggerResult(BaseModel):
    """Rich diagnostic payload explaining the trigger's decision."""
    triggered: bool = Field(default=False)
    success: bool = Field(default=False)
    reason: str = Field(...)
    chunks_persisted: int = Field(default=0)


class IngestionTrigger:
    """
    Adapter between the Self-Healing Subsystem and the Ingestion Subsystem.
    
    Responsibility:
    Evaluates whether a Knowledge Gap has accumulated enough evidence (frequency)
    to justify permanently altering the database. If so, delegates to Layer 1.
    """

    def __init__(
        self, 
        ingestion_pipeline: IngestionPipeline,
        source_resolver: Optional[Callable[[str], str]] = None
    ):
        self.ingestion_pipeline = ingestion_pipeline
        self.source_resolver = source_resolver

    def evaluate_and_trigger(self, gap: KnowledgeGap) -> TriggerResult:
        """
        Evaluates a single KnowledgeGap against system policies.
        Returns a detailed TriggerResult to inform the KnowledgeManager what happened.
        """
        logger.info(f"IngestionTrigger evaluating gap: '{gap.missing_topic}'")

        
        if not AUTO_TRIGGER_INGESTION:
            return TriggerResult(reason="Auto-trigger disabled globally.")

       
        if gap.resolved:
            return TriggerResult(reason="Gap already resolved.")

       
        days_since_last_seen = (datetime.now(timezone.utc) - gap.last_detected).days
        if days_since_last_seen > KNOWLEDGE_GAP_EXPIRY_DAYS:
            return TriggerResult(reason=f"Gap expired ({days_since_last_seen} days old).")

     
        if gap.frequency < KNOWLEDGE_GAP_TRIGGER_COUNT:
            return TriggerResult(
                reason=f"Insufficient evidence. Frequency={gap.frequency}/{KNOWLEDGE_GAP_TRIGGER_COUNT}."
            )

      
        logger.info(f"THRESHOLD MET: Gap '{gap.missing_topic}' reached {gap.frequency} hits. Preparing ingestion.")
        
        target_source = self._resolve_source(gap.missing_topic)
        if not target_source:
            return TriggerResult(
                triggered=True, 
                success=False, 
                reason=f"Failed to resolve ingestion source for topic: '{gap.missing_topic}'."
            )

     
        try:
            result = self.ingestion_pipeline.ingest(source=target_source, source_type="auto")
            
            if result.chunks_persisted > 0:
                logger.info(
                    f"Auto-Ingestion Successful for '{gap.missing_topic}'. "
                    f"Persisted {result.chunks_persisted} new vectors in {result.elapsed_time_sec}s."
                )
                return TriggerResult(
                    triggered=True,
                    success=True,
                    reason="Ingestion completed successfully.",
                    chunks_persisted=result.chunks_persisted
                )
            else:
                logger.warning(f"Auto-Ingestion ran for '{gap.missing_topic}' but yielded 0 chunks.")
                return TriggerResult(
                    triggered=True,
                    success=False,
                    reason="Ingestion ran but yielded 0 chunks."
                )
                
        except Exception as e:
            logger.error(f"Catastrophic failure during auto-ingestion for '{gap.missing_topic}': {str(e)}")
            return TriggerResult(
                triggered=True,
                success=False,
                reason=f"Ingestion pipeline crashed: {str(e)}"
            )

    def _resolve_source(self, missing_topic: str) -> Optional[str]:
        """
        Determines WHERE to get the documents for this missing topic.
        """
        if self.source_resolver:
            return self.source_resolver(missing_topic)
        return f"tavily:{missing_topic}"