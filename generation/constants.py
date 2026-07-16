"""
Global Configuration and Magic Numbers for the Generation Subsystem.
Centralizes all LLM hyperparameters, prompt constraints, and fallback behaviors.
"""

# ==========================================
# LLM Generation Defaults
# ==========================================
# These control the "creativity" and determinism of the response.
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.1  # Low temperature for factual, grounded RAG responses
DEFAULT_TOP_P = 0.95
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0


# ==========================================
# Prompt Defaults & Behaviors
# ==========================================
# Controls how the Prompt Builder constructs instructions for the LLM
DEFAULT_PROMPT_TEMPLATE = "qa"

# True = Refuse to answer if context is missing. False = Use parametric memory.
REQUIRE_CONTEXT_GROUNDING = True

# True = Allow LLM to add supplementary knowledge. False = Strict extraction only.
ALLOW_EXTERNAL_KNOWLEDGE = False


# ==========================================
# Token Budgets & Limits
# ==========================================
# Hard limits to prevent API cost blowouts and context window crashes.
MAX_OUTPUT_TOKENS = 1024
MIN_OUTPUT_TOKENS = 50

# How much of the LLM context window we dedicate specifically to retrieved chunks.
MAX_CONTEXT_BUDGET = 4000 
# Tokens reserved for the System Prompt instructions and Chat History.
SYSTEM_PROMPT_RESERVED_TOKENS = 500 


# ==========================================
# Citation Formatting Defaults
# ==========================================
# e.g., 'brackets' -> [1], 'superscript' -> ¹, 'xml' -> <ref>1</ref>
DEFAULT_CITATION_STYLE = "brackets"

# Regex pattern used by the Citation Extractor to parse the LLM's output.
# Matches formats like [1], [2], [1, 2], [doc_1]
CITATION_REGEX_PATTERN = r"\[([a-zA-Z0-9_,\s]+)\]"


# ==========================================
# Fallback Behaviors
# ==========================================
# Standardized strings returned when the pipeline cannot proceed.
DEFAULT_EMPTY_RESPONSE = "I apologize, but I could not generate an answer based on the provided context."

# Used when the Retrieval Subsystem returns 0 chunks, but the user still asked a question.
DEFAULT_NO_CONTEXT_RESPONSE = (
    "I apologize, but I couldn't find any relevant information in the knowledge base "
    "to answer your question. Could you try rephrasing or asking about a different topic?"
)

# Used if the ContextBuilder upstream failed to budget correctly and the LLM rejects the prompt.
DEFAULT_CONTEXT_TOO_LARGE_RESPONSE = (
    "The retrieved evidence was too large to process. Please try asking a more specific question."
)

# Used if the LLM API times out, rate-limits, or fails entirely.
DEFAULT_GENERATION_ERROR_RESPONSE = (
    "I encountered an error while trying to synthesize the answer. "
    "Please try again in a moment."
)


# ==========================================
# Observability Defaults
# ==========================================
# How many characters of the generated prompt to log to prevent console flooding.
LOG_PROMPT_PREVIEW_CHARS = 300


from enum import Enum

class CitationStyle(str, Enum):
    BRACKETS = "brackets"
    SUPERSCRIPT = "superscript"

DEFAULT_CITATION_STYLE = CitationStyle.BRACKETS