import logging
from typing import List, Optional, Protocol
from datetime import datetime, timezone

from self_healing.models import KnowledgeGap

logger = logging.getLogger(__name__)

# ==========================================
# Storage Contract (Repository Pattern)
# ==========================================
class KnowledgeStorage(Protocol):
    """
    Abstract contract for persisting Knowledge Gaps.
    Ensures the Manager remains completely decoupled from PostgreSQL/SQLite.
    """
    def get_by_topic(self, missing_topic: str) -> Optional[KnowledgeGap]: ...
    def save(self, gap: KnowledgeGap) -> None: ...
    def get_all(self, include_resolved: bool = False) -> List[KnowledgeGap]: ...
    def get_top_missing_topics(self, limit: int) -> List[KnowledgeGap]: ...
    def get_resolved_topics(self) -> List[KnowledgeGap]: ...
    def delete_expired(self, expiry_days: int) -> int: ...


class InMemoryKnowledgeStorage:
    """A simple in-memory implementation for testing/local dev."""
    def __init__(self):
        self._db = {}
        
    def get_by_topic(self, missing_topic: str) -> Optional[KnowledgeGap]:
        return self._db.get(missing_topic)
        
    def save(self, gap: KnowledgeGap) -> None:
        self._db[gap.missing_topic] = gap
        
    def get_all(self, include_resolved: bool = False) -> List[KnowledgeGap]:
        if include_resolved:
            return list(self._db.values())
        return [g for g in self._db.values() if not g.resolved]

    def get_top_missing_topics(self, limit: int) -> List[KnowledgeGap]:
        gaps = [g for g in self._db.values() if not g.resolved]
        # Basic Python sorting fallback for in-memory
        gaps.sort(key=lambda x: (x.frequency, x.last_detected), reverse=True)
        return gaps[:limit]

    def get_resolved_topics(self) -> List[KnowledgeGap]:
        return [g for g in self._db.values() if g.resolved]

    def delete_expired(self, expiry_days: int) -> int:
        now = datetime.now(timezone.utc)
        to_delete = [k for k, g in self._db.items() if not g.resolved and (now - g.last_detected).days > expiry_days]
        for k in to_delete: del self._db[k]
        return len(to_delete)


# ==========================================
# The Core Manager
# ==========================================
class KnowledgeManager:
    """
    The System's Long-Term Memory.
    Maintains the 'Librarian's Notebook' of missing topics, tracking 
    frequencies, user query variations, and resolution states over time.
    """

    def __init__(self, storage: KnowledgeStorage):
        # Strict Dependency Injection - forces the caller to provide the DB implementation
        self.storage = storage

    def process_detected_gap(self, new_gap: KnowledgeGap) -> KnowledgeGap:
        """
        Ingests a fresh gap from the GapDetector.
        Applies business rules: normalization, query deduplication, and regression detection.
        Returns the updated, authoritative Gap model.
        """
        now = datetime.now(timezone.utc)
        
        # 1. Normalize at the boundary and create a pure copy to avoid mutating inputs
        normalized_topic = new_gap.missing_topic.lower().strip()
        normalized_gap = new_gap.model_copy(update={"missing_topic": normalized_topic})
        
        logger.info(f"KnowledgeManager processing gap: '{normalized_topic}'")

        existing_gap = self.storage.get_by_topic(normalized_topic)

        if existing_gap:
            # 2. Update Statistics
            existing_gap.frequency += 1
            existing_gap.last_detected = now
            
            # 3. Update Traceability
            existing_gap.last_query_id = normalized_gap.last_query_id 
            
            # 4. Safely merge the new user phrasing into the historical list
            for query in normalized_gap.failed_queries:
                if query not in existing_gap.failed_queries:
                    existing_gap.failed_queries.append(query)

           
            if existing_gap.resolved:
                logger.warning(
                    f"Regression! Topic '{existing_gap.missing_topic}' was marked "
                    f"resolved, but failed again. Re-opening gap."
                )
                existing_gap.resolved = False
                existing_gap.resolved_at = None

            updated_gap = existing_gap
            logger.info(f"Updated existing gap '{updated_gap.missing_topic}'. Frequency now {updated_gap.frequency}.")

        else:
           
            updated_gap = normalized_gap
            updated_gap.last_detected = now
            logger.info(f"Recorded new knowledge gap: '{updated_gap.missing_topic}'.")

       
        self.storage.save(updated_gap)
        logger.debug(f"Knowledge Gap persisted | Topic='{normalized_topic}'")
        
        return updated_gap

    def mark_resolved(self, missing_topic: str) -> None:
        """
        Called by the Orchestrator after the IngestionTrigger successfully 
        updates the database. Prevents future repeated ingestions.
        """
        now = datetime.now(timezone.utc)
        normalized_topic = missing_topic.lower().strip()
        gap = self.storage.get_by_topic(normalized_topic)
        
        if gap:
            gap.resolved = True
            gap.resolved_at = now
            gap.last_detected = now 
            self.storage.save(gap)
            logger.info(f"Successfully marked knowledge gap '{normalized_topic}' as RESOLVED.")
        else:
            logger.warning(f"Attempted to resolve unknown topic: '{normalized_topic}'")


    
    def get_topic_status(self, missing_topic: str) -> Optional[KnowledgeGap]:
        """Fetches the exact state of a single topic (e.g., 'GPT-6')."""
        normalized_topic = missing_topic.lower().strip()
        return self.storage.get_by_topic(normalized_topic)

    def get_top_missing_topics(self, limit: int = 10) -> List[KnowledgeGap]:
        """
        Returns the most requested missing topics. 
        Delegates to the storage layer to utilize efficient SQL ORDER BY operations.
        """
        return self.storage.get_top_missing_topics(limit=limit)
        
    def get_resolved_topics(self) -> List[KnowledgeGap]:
        """Returns all topics that the system has successfully self-healed."""
        return self.storage.get_resolved_topics()