# memory/constants.py

"""
Configuration policies and thresholds for the Memory Subsystem.
No business logic, prompt templates, or domain models should live here.
"""

# ==========================================
# Conversation Window
# ==========================================
# How many literal recent messages to keep before relying strictly on the summary
MAX_RECENT_MESSAGES = 8

# The absolute maximum token footprint the raw history is allowed to consume
MAX_HISTORY_TOKENS = 3000

# ==========================================
# Summarization Policy
# ==========================================
# Trigger summarization if the unsummarized message count exceeds this
SUMMARY_TRIGGER_MESSAGES = 12

# Trigger summarization if the unsummarized token count exceeds this (overrides message count)
SUMMARY_TRIGGER_TOKENS = 4000

MIN_MESSAGES_FOR_SUMMARY = 4
# Cap on the LLM's output length when generating a summary
SUMMARY_MAX_TOKENS = 500
SUMMARY_TEMPERATURE = 0.0
# We use a smaller, faster model for background memory compression (e.g., Llama 3 8B)
SUMMARY_MODEL = "llama3-8b-8192"
MIN_SUMMARY_LENGTH = 50
# ==========================================
# Session Management
# ==========================================
# How long before a session is considered stale/expired (72 hours)
SESSION_TIMEOUT_HOURS = 72

# Whether to automatically generate a session if a user provides an unknown session_id
AUTO_CREATE_SESSION = True

# ==========================================
# Context Budget
# ==========================================
# The absolute max tokens the entire MemoryContext (Summary + History) 
# can consume in the final Generation prompt.
MEMORY_CONTEXT_TOKEN_BUDGET = 2000

# ==========================================
# Context Formatting
# ==========================================
# Standardized headers for the formatted_context_string output
SUMMARY_HEADER = "[SYSTEM NOTE: SUMMARY OF OLDER CONVERSATION]"
HISTORY_HEADER = "[RECENT CONVERSATION EXACT TRANSCRIPT]"
CURRENT_QUERY_HEADER = "[CURRENT USER QUESTION]"

# ==========================================
# Reliability
# ==========================================
# How many times to retry the Summarizer LLM call if it times out
MAX_SUMMARIZATION_RETRIES = 2

# ==========================================
# Metadata & Observability
# ==========================================
MEMORY_PIPELINE_VERSION = "1.0.0"

# Feature flags for operational logging
ENABLE_MEMORY_LOGGING = True
LOG_CONTEXT_BUILD_TIME = True
LOG_SUMMARIZATION_EVENTS = True