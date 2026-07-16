# memory/builder/formatters.py

from abc import ABC, abstractmethod
from typing import Optional

from memory.models import ConversationHistory, ConversationSummary, UserPreferences

class BaseMemoryFormatter(ABC):
    """
    Abstract contract for translating Domain Models into prompt-ready strings.
    Allows swapping between Markdown, XML, or JSON formatting without touching the Builder.
    """
    @abstractmethod
    def format(
        self,
        history: ConversationHistory,
        summary: Optional[ConversationSummary],
        preferences: Optional[UserPreferences]
    ) -> str:
        pass


class MarkdownMemoryFormatter(BaseMemoryFormatter):
    """
    Standard markdown formatter. 
    Outputs raw conversation data cleanly without overarching 'Prompt' wrappers.
    """
    
    SUMMARY_TEMPLATE = "Conversation Summary:\n{summary}"
    HISTORY_TEMPLATE = "Recent Conversation:\n{history}"
    PREFERENCES_TEMPLATE = "User Preferences:\n{preferences}"

    def format(
        self,
        history: ConversationHistory,
        summary: Optional[ConversationSummary],
        preferences: Optional[UserPreferences]
    ) -> str:
        parts = []

        
        if summary and summary.summary:
            parts.append(self.SUMMARY_TEMPLATE.format(summary=summary.summary))

        
        if history and history.messages:
            lines = []
            for msg in history.messages:
               
                role_label = msg.role.value.capitalize()
                lines.append(f"{role_label}: {msg.content}")
            
            history_str = "\n".join(lines)
            parts.append(self.HISTORY_TEMPLATE.format(history=history_str))

      
        if preferences:
            
            pref_data = preferences.model_dump(exclude={"user_id"}, exclude_none=True)
            
            lines = []
            for key, value in pref_data.items():
               
                if isinstance(value, list) and not value:
                    continue 
                
                
                val_str = ", ".join(value) if isinstance(value, list) else str(value)
                
                
                formatted_key = key.replace('_', ' ').title()
                lines.append(f"- {formatted_key}: {val_str}")
            
            if lines:
                pref_str = "\n".join(lines)
                parts.append(self.PREFERENCES_TEMPLATE.format(preferences=pref_str))

        if not parts:
            return ""

        return "\n\n".join(parts)