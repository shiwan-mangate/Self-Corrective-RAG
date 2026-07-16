# memory/builder/context_builder.py

from typing import Optional, Dict, Any
from memory.models import (
    Session, 
    ConversationHistory, 
    ConversationSummary, 
    MemoryContext,
    UserPreferences
)
from memory.builder.formatters import BaseMemoryFormatter, MarkdownMemoryFormatter


class MemoryContextBuilder:
    """
    The Packaging Layer for the Memory Subsystem.
    
    Architecture Note:
    - PURE ORCHESTRATOR: Delegates all string manipulation to the injected Formatter.
    - OPAQUE ABSTRACTION: Transforms structured domain objects into a single 
      prompt-ready string so downstream pipelines (Generation) do not need to 
      understand Memory's internal data structures.
    """

    def __init__(self, formatter: Optional[BaseMemoryFormatter] = None):
        # Default to Markdown if no specific formatter is injected
        self.formatter = formatter or MarkdownMemoryFormatter()

    def build(
        self, 
        session: Session, 
        history: ConversationHistory, 
        summary: Optional[ConversationSummary] = None,
        preferences: Optional[UserPreferences] = None
    ) -> MemoryContext:
        """
        Assembles the final MemoryContext object containing both the structured state
        and the fully formatted, prompt-ready string for the LLM.
        """
        
        # 1. Delegate text conversion to the Formatter Strategy
        formatted_string = self.formatter.format(
            history=history,
            summary=summary,
            preferences=preferences
        )
        
        # 2. Return the immutable package
        return MemoryContext(
            session=session,
            active_history=history,
            summary=summary,
            user_preferences=preferences,
            formatted_context_string=formatted_string
        )