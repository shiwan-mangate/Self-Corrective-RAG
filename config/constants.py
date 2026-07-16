from retrieval.models import SearchType

# ==========================================
# Search Defaults
# ==========================================
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MIN_TOP_K = 1

DEFAULT_SEARCH_TYPE = SearchType.SIMILARITY

# ==========================================
# Query Length Thresholds
# ==========================================
# Used by the analyzer to determine if a query is too vague to search directly
SHORT_QUERY_WORDS = 3
LONG_QUERY_WORDS = 15

# ==========================================
# Similarity Thresholds
# ==========================================
# Used by filters.py to drop irrelevant chunks before generation
MIN_SIMILARITY_SCORE = 0.60
HIGH_SIMILARITY_SCORE = 0.85

# ==========================================
# Analyzer Intent Keywords
# ==========================================
# Triggers the needs_history flag
FOLLOW_UP_KEYWORDS = {
    "it", "they", "them", "that", "those", 
    "this", "its", "their", "he", "she", 
    "his", "her", "these"
}

# Triggers QueryIntent.COMPARISON and potentially larger top_k
COMPARISON_KEYWORDS = {
    "compare", "difference", "versus", "vs", 
    "better", "similar", "unlike", "distinguish", 
    "pros and cons"
}

# Triggers QueryIntent.SUMMARY and skips history injection
SUMMARY_KEYWORDS = {
    "summarize", "summary", "brief", "overview", 
    "tldr", "tl;dr", "outline", "gist"
}

# Triggers QueryIntent.EXPLANATION
EXPLANATION_KEYWORDS = {
    "what", "why", "how", "explain", 
    "describe", "clarify", "define", "meaning"
}

# ==========================================
# Context & Metadata Defaults
# ==========================================
DEFAULT_LANGUAGE = "en"

# Used by context_builder.py to ensure the LLM's context window isn't blown out
DEFAULT_TOKEN_BUDGET = 4000

MAX_HISTORY_MESSAGES = 4