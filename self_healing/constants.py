"""
Global Configuration and Policies for the Self-Healing Subsystem.
Centralizes execution constraints, feature flags, and intelligence thresholds.
"""

from evaluation.constants import MIN_GOOD_RETRIEVAL_SIMILARITY # Import to guarantee consistency

# ==========================================
# 1. Retry Policy (Loop Prevention)
# ==========================================
MAX_RECOVERY_RETRIES = 3
STOP_ON_MAX_RETRIES = True
ALLOW_WEB_SEARCH_AFTER_RETRY = True
ALLOW_QUERY_REWRITE = True
ALLOW_CONTEXT_MERGE = True

# Safety backoff for external APIs (e.g., Tavily rate limits)
RETRY_BACKOFF_SECONDS = 2 

# ==========================================
# 2. Priority Routing Policy
# ==========================================
# If multiple errors exist in the Evaluation Report, which takes precedence?
# Lower number = Higher Priority
PRIORITY_HALLUCINATION = 1
PRIORITY_GROUNDING = 2
PRIORITY_RETRIEVAL = 3
PRIORITY_CONFIDENCE = 4

# ==========================================
# 3. Hallucination Policy
# ==========================================
HALLUCINATION_FORCE_STRICT_GROUNDING = True
HALLUCINATION_MAX_RETRIES = 1  # Stop burning tokens if the model is repeatedly fabricating.

# ==========================================
# 4. Grounding & Web Search Policy
# ==========================================
# We share the evaluation threshold so self-healing routing perfectly matches the evaluation diagnosis.
MIN_INTERNAL_SIMILARITY_FOR_WEB_SEARCH = MIN_GOOD_RETRIEVAL_SIMILARITY 

ENABLE_WEB_SEARCH = True
MAX_WEB_RESULTS = 5
WEB_SEARCH_TIMEOUT_SEC = 10

# ==========================================
# 5. Context Merge Policy
# ==========================================
MAX_INTERNAL_DOCUMENTS = 5
MAX_WEB_DOCUMENTS = 3
MAX_MERGED_CONTEXT_TOKENS = 6000 

# Deduplication policy during merge
REMOVE_DUPLICATES = True
MIN_CONTEXT_SIMILARITY_FOR_DEDUP = 0.95

# ==========================================
# 6. Knowledge Gap Policy (Long-Term Learning)
# ==========================================
AUTO_TRIGGER_INGESTION = True

KNOWLEDGE_GAP_TRIGGER_COUNT = 5  
KNOWLEDGE_GAP_EXPIRY_DAYS = 90   

# Filter out nonsense queries ("hi", "test") so they don't trigger automated ingestion.
MIN_QUERY_LENGTH_FOR_GAP = 10    

# ==========================================
# 7. Feedback Policy
# ==========================================
NEGATIVE_FEEDBACK_THRESHOLD = 5
POSITIVE_FEEDBACK_THRESHOLD = 20

# ==========================================
# 8. Recovery Timing & Latency
# ==========================================
MAX_RECOVERY_TIME_MS = 15000  

# ==========================================
# 9. Observability & Logging
# ==========================================
LOG_RECOVERY_PREVIEW_CHARS = 250
RECOVERY_PIPELINE_VERSION = "1.0"