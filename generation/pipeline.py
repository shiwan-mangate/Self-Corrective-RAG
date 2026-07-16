# generation/pipeline.py
import logging

from generation.models import GenerationRequest, GenerationResponse
from generation.constants import REQUIRE_CONTEXT_GROUNDING, DEFAULT_CITATION_STYLE

from generation.prompts.base import BasePromptBuilder
from generation.generator.base import BaseAnswerGenerator
from generation.citations.extractor import CitationExtractor
from generation.citations.formatter import CitationFormatter
from generation.response.formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """
    The master orchestrator for Layer 3 (Generation).
    
    Responsibility:
    Provides the exclusive public API for turning a structured RetrievalContext 
    (via GenerationRequest) into a final, cited GenerationResponse.
    Contains NO business logic, text formatting, or LLM execution.
    """

    def __init__(
        self,
        prompt_builder: BasePromptBuilder,
        answer_generator: BaseAnswerGenerator,
        citation_extractor: CitationExtractor,
        citation_formatter: CitationFormatter,
        response_formatter: ResponseFormatter
    ):
        self.prompt_builder = prompt_builder
        self.answer_generator = answer_generator
        self.citation_extractor = citation_extractor
        self.citation_formatter = citation_formatter
        self.response_formatter = response_formatter

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Executes the linear Generation Subsystem flow:
        Build Prompt -> Generate Answer -> Extract Citations -> Format -> Assemble.
        """
        logger.info(f"Initiating Generation Pipeline for query: '{request.optimized_query}'")

       
        strict_grounding = REQUIRE_CONTEXT_GROUNDING

      
        prompt_package = self.prompt_builder.build(
            request=request, 
            strict_grounding=strict_grounding
        )

        
        generated_answer = self.answer_generator.generate(
            prompt_package=prompt_package
        )

       
        resolved_citations = self.citation_extractor.extract(
            answer=generated_answer,
            citation_map=prompt_package.citation_map
        )

       
        formatted_text = self.citation_formatter.format_text(
            text=generated_answer.answer,
            citation_map=prompt_package.citation_map,
            style=DEFAULT_CITATION_STYLE
        )

        response = self.response_formatter.build(
            request=request,
            prompt_package=prompt_package,
            generated_answer=generated_answer,
            resolved_citations=resolved_citations,
            formatted_text=formatted_text
        )

        logger.info("Generation Pipeline Complete. Returning structured payload.")
        
        return response