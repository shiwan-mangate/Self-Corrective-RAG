# memory/summarization/summarizer.py

import time
import logging

from memory.models import SummaryGenerationRequest, SummaryGenerationResult, ConversationSummary
from memory.summarization.prompts import SummaryPromptBuilder
from memory.constants import (
    SUMMARY_MODEL, 
    SUMMARY_TEMPERATURE, 
    MAX_SUMMARIZATION_RETRIES, 
    MIN_SUMMARY_LENGTH
)

from ai.text_generation_service import AITextGenerationService

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    The pure AI engine for Memory Compression.
    Coordinates the PromptBuilder and your shared AITextGenerationService.
    
    Architecture Note:
    - 100% REALIGNED: Integrates with your native Layer 0 'ai/' services.
    - Captures latency and token diagnostics locally.
    - Builds the strict ConversationSummary domain object.
    """

    def __init__(
        self, 
        prompt_builder: SummaryPromptBuilder, 
        llm_service: AITextGenerationService
    ):
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service

    def summarize(self, request: SummaryGenerationRequest) -> SummaryGenerationResult:
        """
        Executes the LLM summarization pipeline using the centralized GroqClient.
        Returns a structured Result object detailing success/failure and the generated summary.
        """
        start_time = time.perf_counter()
        
        if not request.messages_to_summarize:
            logger.debug(
                f"No messages provided to compress for Session {request.session_id} "
                f"| QueryID={request.query_id}"
            )
            return SummaryGenerationResult(
                success=False,
                summary=None,
                latency_ms=0.0,
                total_tokens=0,
                model_name=SUMMARY_MODEL,
                error_message="No messages provided to summarize."
            )

        logger.info(
            f"Triggering LLM compression for {len(request.messages_to_summarize)} messages "
            f"| Session={request.session_id} | QueryID={request.query_id}"
        )

        # 1. Build prompts via our isolated factory
        system_prompt = self.prompt_builder.get_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(
            messages_to_summarize=request.messages_to_summarize,
            previous_summary=request.previous_summary
        )

        generated_text = None
        error_msg = None

        # 2. Execute LLM Call against your production service with retries
        for attempt in range(MAX_SUMMARIZATION_RETRIES):
            try:
                # Calls your native GroqClient under the hood
                raw_response = self.llm_service.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    query_id=request.query_id,
                    temperature=SUMMARY_TEMPERATURE,
                    max_tokens=512  # Or pulled from memory/constants.py
                )
                
                text = raw_response.strip()

                # 3. Strict Validation of AI text response
                self._validate_summary(text)

                generated_text = text
                error_msg = None
                break

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Summarization attempt {attempt + 1} failed for Session {request.session_id} "
                    f"| QueryID={request.query_id} | Error: {error_msg}"
                )
                generated_text = None

        total_latency = round((time.perf_counter() - start_time) * 1000, 2)

        # 4. Handle Failure
        if not generated_text:
            logger.error(
                f"Summarization completely failed for Session {request.session_id} "
                f"| QueryID={request.query_id}"
            )
            return SummaryGenerationResult(
                success=False,
                summary=None,
                latency_ms=total_latency,
                total_tokens=0,
                model_name=SUMMARY_MODEL,
                error_message=f"Max retries exceeded. Last error: {error_msg}"
            )

        # Token heuristic tracking calculated on the text boundary
        estimated_tokens = len(generated_text) // 4
        
        # 5. Build the strict Domain Object
        new_version = (request.previous_summary.summary_version + 1) if request.previous_summary else 1
        
        new_summary = ConversationSummary(
            summary=generated_text,
            covered_messages=len(request.messages_to_summarize),
            summary_version=new_version,
            model_name=SUMMARY_MODEL
        )

        logger.debug(
            f"Compression successful | Version={new_version} | Latency={total_latency}ms | "
            f"Est. Tokens={estimated_tokens} | QueryID={request.query_id}"
        )

        return SummaryGenerationResult(
            success=True,
            summary=new_summary,
            latency_ms=total_latency,
            total_tokens=estimated_tokens,
            model_name=SUMMARY_MODEL,
            error_message=None
        )

    def _validate_summary(self, text: str) -> None:
        """
        Inspects the raw text for clear indicator formatting or refusal strings.
        """
        if not text:
            raise ValueError("LLM returned an empty summary.")
            
        if len(text) < MIN_SUMMARY_LENGTH:
            raise ValueError(f"Summary text is too short ({len(text)} < {MIN_SUMMARY_LENGTH}).")
            
        lower_text = text.lower()
        refusals = [
            "i cannot summarize", 
            "i don't know", 
            "i am unable", 
            "i do not have",
            "as an ai"
        ]
        
        for refusal in refusals:
            if refusal in lower_text:
                raise ValueError(f"LLM explicitly refused: matched refusal string '{refusal}'.")