from typing import List, Dict, Tuple
from retrieval.models import AnalyzedQuery
from retrieval.rewriter.prompts import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE
from config.constants import MAX_HISTORY_MESSAGES

class RewritePromptBuilder:
    """
    Isolates the string manipulation required to assemble the LLM prompt.
    """

    def build(self, analyzed_query: AnalyzedQuery, chat_history: List[Dict[str, str]]) -> Tuple[str, str]:
        history_text = self._format_history(chat_history) if analyzed_query.needs_history else "No prior history."
        
        user_prompt = QUERY_REWRITE_USER_TEMPLATE.format(
            history=history_text,
            query=analyzed_query.original_query
        )
        return QUERY_REWRITE_SYSTEM_PROMPT, user_prompt

    def _format_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Extracts recent conversation turns and structures them clearly for the LLM.
        """
        if not chat_history:
            return "No prior history."
            
        formatted = []
        for msg in chat_history[-MAX_HISTORY_MESSAGES:]:  
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}:\n{content}\n")
            
        return "\n".join(formatted).strip()