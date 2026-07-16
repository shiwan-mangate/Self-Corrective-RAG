import logging
from typing import Dict, List

from retrieval.models import Citation
from generation.models import GenerationRequest, PromptPackage
from generation.prompts.base import BasePromptBuilder
from generation.prompts import templates

logger = logging.getLogger(__name__)


class PromptBuilder(BasePromptBuilder):
    """
    Transforms a GenerationRequest and Pipeline Policies into a final PromptPackage.
    """

    PROMPT_MAP = {
        "summary": templates.SUMMARY_SYSTEM_PROMPT,
        "comparison": templates.COMPARISON_SYSTEM_PROMPT,
        "explanation": templates.EXPLANATION_SYSTEM_PROMPT,
        "factual": templates.QA_SYSTEM_PROMPT,
        "unknown": templates.QA_SYSTEM_PROMPT,
        "follow_up": templates.QA_SYSTEM_PROMPT
    }

    def build(self, request: GenerationRequest, strict_grounding: bool) -> PromptPackage:
        
        # 1. Select the Persona / System Prompt based on intent
        intent_str = request.intent.value.lower() if request.intent else "unknown"
        system_prompt = self.PROMPT_MAP.get(intent_str, templates.QA_SYSTEM_PROMPT)
        
        # 2. Build the Instruction Block
        grounding_instruction = (
            templates.STRICT_GROUNDING_INSTRUCTION 
            if strict_grounding 
            else templates.HYBRID_GROUNDING_INSTRUCTION
        )
        instructions_block = f"{grounding_instruction}\n\n{templates.CITATION_INSTRUCTION}"

        # 3. Format the Conversation History
        history_text = self._format_history(request.chat_history)

        # 4. Determine the Question to Ask
        question_to_ask = request.optimized_query if request.optimized_query else request.original_query
        
        # 5. Inject everything into the User Prompt Template
        user_prompt = templates.USER_PROMPT_TEMPLATE.format(
            history=history_text,             # <-- INJECTED HERE
            context=request.context,
            question=question_to_ask,
            instructions=instructions_block
        )

        # 6. Map the available citations strictly by index (1-based)
        citation_map: Dict[int, Citation] = {}
        for index, citation in enumerate(request.available_citations, start=1):
            citation_map[index] = citation

        logger.info(
            f"Prompt Built | Intent: {intent_str.upper()} | "
            f"Template Used: {intent_str} | "
            f"Mapped Citations: {len(citation_map)}"
        )

        return PromptPackage(
            query_id=request.query_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            template_name=intent_str,
            intent=request.intent,
            citation_map=citation_map
        )

    def _format_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Extracts recent conversation turns and structures them clearly for the LLM.
        Limits to the last 5 messages to preserve the token budget.
        """
        if not chat_history:
            return "No prior history."
            
        formatted = []
        # Slice to the last 5 messages to prevent context window overflow
        for msg in chat_history[-5:]:  
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}:\n{content}\n")
            
        return "\n".join(formatted).strip()