import logging
import time

from generation.models import PromptPackage, GeneratedAnswer, TokenUsage
from generation.constants import DEFAULT_TEMPERATURE, MAX_OUTPUT_TOKENS
from shared.exceptions import GenerationExecutionError


from generation.generator.base import BaseAnswerGenerator
from ai.text_generation_service import AITextGenerationService

logger = logging.getLogger(__name__)


class LLMAnswerGenerator(BaseAnswerGenerator):
    """
    The central execution boundary for text synthesis.
    
    Responsibility:
    Takes a fully baked PromptPackage, calls the AI service, and captures 
    execution telemetry to satisfy the GeneratedAnswer contract.
    """

    def __init__(self, llm_service: AITextGenerationService):
        self.llm_service = llm_service

    def generate(self, prompt_package: PromptPackage) -> GeneratedAnswer:
        start_time = time.time()

        try:
            raw_text = self.llm_service.generate(
                system_prompt=prompt_package.system_prompt,
                user_prompt=prompt_package.user_prompt,
                query_id=prompt_package.query_id,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS
            )

           
            generation_time_ms = round((time.time() - start_time) * 1000, 2)
            token_usage = self._estimate_tokens(
                system_prompt=prompt_package.system_prompt,
                user_prompt=prompt_package.user_prompt,
                raw_text=raw_text
            )

            # 3. Rich Observability Logging
            logger.info(
                f"Generation Executed | Intent: {prompt_package.intent.value.upper()} | "
                f"Template: {prompt_package.template_name} | "
                f"Time: {generation_time_ms}ms | Est. Tokens: {token_usage.total_tokens}"
            )

            # 4. Return the strict domain contract
            return GeneratedAnswer(
                answer=raw_text,
                generation_time_ms=generation_time_ms,
                token_usage=token_usage,
                finish_reason=None  # We do not guess. Wait for the API to explicitly provide this.
            )

        except Exception as e:
            logger.error(f"LLM Generation failed during execution: {str(e)}")
            raise GenerationExecutionError(f"LLM API Error: {str(e)}") from e

    def _estimate_tokens(self, system_prompt: str, user_prompt: str, raw_text: str) -> TokenUsage:
        """
        Private helper to calculate/estimate token consumption.
        Future Upgrade: Replace deterministic math with `tiktoken` or true API metadata.
        """
        est_prompt_tokens = int((len(system_prompt) + len(user_prompt)) / 4)
        est_comp_tokens = int(len(raw_text) / 4)
        
        return TokenUsage(
            prompt_tokens=est_prompt_tokens,
            completion_tokens=est_comp_tokens,
            total_tokens=est_prompt_tokens + est_comp_tokens
        )