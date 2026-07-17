# retrieval/rewriter/query_rewrite.py
import logging
import time
from typing import List, Dict
from retrieval.models import AnalyzedQuery
from retrieval.rewriter.base import BaseQueryRewriter
from retrieval.rewriter.prompt_builder import RewritePromptBuilder
from ai.text_generation_service import AITextGenerationService

logger = logging.getLogger(__name__)


class LLMQueryRewriter(BaseQueryRewriter):
    """
    Evaluates analyzed intent flags. If a rewrite is mandated, invokes 
    the generic Text Generation service to expand pronouns and vague context.
    """

    def __init__(
        self, 
        llm_service: AITextGenerationService, 
        prompt_builder: RewritePromptBuilder = None
    ):
        self.llm = llm_service
        self.prompt_builder = prompt_builder or RewritePromptBuilder()

    def rewrite(
        self, 
        analyzed_query: AnalyzedQuery, 
        chat_history: List[Dict[str, str]],
        query_id: str = "unknown_query"
    ) -> AnalyzedQuery:
        
        if not analyzed_query.needs_rewrite:
            return analyzed_query

        start_time = time.time()
        system_prompt, user_prompt = self.prompt_builder.build(analyzed_query, chat_history)
        
        try:
          
            raw_rewritten = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                query_id=query_id,
                temperature=0.0
            )
            
     
            clean_rewritten = self._sanitize_output(raw_rewritten, fallback_query=analyzed_query.normalized_query)
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Rewrite Complete | Intent: {analyzed_query.intent.value.upper()} | "
                f"History: {analyzed_query.needs_history} | Original: '{analyzed_query.original_query}' | "
                f"Rewritten: '{clean_rewritten}' | Time: {elapsed_time:.4f}s"
            )
            
          
            return analyzed_query.model_copy(update={
                "rewritten_query": clean_rewritten,
                "rewrite_performed": True
            })
            
        except Exception as e:
            
            logger.warning(f"Query Rewrite failed. Falling back to normalized query. Error: {str(e)}")
            return analyzed_query.model_copy(update={
                "rewritten_query": None,
                "rewrite_performed": False
            })

    def _sanitize_output(self, raw_text: str, fallback_query: str) -> str:
        """
        Strips chatty preambles and quotes from the LLM. 
        Falls back to the normalized query if the LLM output evaluates to empty.
        """
        text = raw_text.strip()
        
        
        prefixes_to_strip = [
            "Sure", "Here is the rewritten query:", "Here is the rewritten search query:",
            "Rewritten Standalone Query:", "Rewrite:", "Question:"
        ]
        
        for prefix in prefixes_to_strip:
            if text.lower().startswith(prefix.lower()):
               
                text = text[len(prefix):].lstrip(" :\n").strip()
                break  
                
     
        text = text.strip('"\'')
        
        if len(text) < 3:
            return fallback_query
            
        return text